import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# REAMOSTRAGEM DO DECODER: TRILINEAR, O MESMO QUE AS OUTRAS REDES 3D DAQUI USAM PARA SUBIR
def resample(x, size):
    if tuple(x.shape[2:]) == tuple(size):
        return x

    return F.interpolate(x, size=tuple(size), mode='trilinear', align_corners=False)


# NORMALIZACAO: o artigo usa BatchNorm, mas ele treinou fatias 2D e aqui o lote e de 2 volumes. Com lote 2 as
# estatisticas moveis saem ruins e a rede fica pior em eval() do que em train() por motivo estatistico - o
# mesmo efeito ja medido na MACNN deste repositorio. GroupNorm nao depende do lote e e o que as demais redes
# daqui usam. norm='batch' volta a condicao do artigo
def getNorm(channels, norm):
    if norm == 'batch':
        return nn.BatchNorm3d(channels)

    groups = 8 if channels >= 8 and channels % 8 == 0 else 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


# CONV + NORMA + ReLU: A UNIDADE BASICA DA FAULT-SEG-NET ("apos a convolucao de cada camada sao adicionadas
# as camadas de batch normalization e ReLU", SECAO 3.2). O kernel aceita tupla para o eixo das fatias
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, dilation=1, norm='group', act=True):
        super().__init__()
        sizes   = kernel if isinstance(kernel, tuple) else (kernel,) * 3
        padding = tuple(dilation * (size - 1) // 2 for size in sizes)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=sizes, padding=padding, dilation=dilation, bias=False)
        self.norm = getNorm(out_channels, norm)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# MODULO MULTIESCALA RESIDUAL (FIGURA 2): TRES RAMOS DILATADOS EM PARALELO, SOMADOS ENTRE SI E A ENTRADA.
# O artigo fixa o campo receptivo de cada ramo (3, 7 e 9) e quantas convolucoes ele tem (uma 3x3; uma 3x3
# mais uma 1x1; duas 3x3 mais uma 1x1). As dilatacoes 1 e 3 sao as unicas que fecham as tres contas ao mesmo
# tempo: 1+2*1 = 3, 1+2*3 = 7 e 3+2*3 = 9. A 1x1 nao altera campo receptivo, so mistura canais.
# A ULTIMA CONV DE CADA RAMO NAO TEM ReLU: o texto diz "apos a convolucao de cada camada" (secao 3.2), mas
# a figura 2 poe o ReLU depois da soma, e se os ramos ja saissem retificados essa soma seria de tres termos
# nao-negativos com F_s nao-negativo - o ReLU da figura nunca faria nada e o modulo so saberia somar, nunca
# suprimir uma resposta espuria. A figura e o residual do He et al. que o artigo cita decidem: ativa depois
# da soma - e o A/B confirma: 0.166 de val_iou contra 0.153 do jeito errado, nas mesmas 15 epocas
class MultiScaleResidual(nn.Module):
    BRANCHES = (((3, 1),),
                ((3, 3), (1, 1)),
                ((3, 1), (3, 3), (1, 1)))

    def __init__(self, channels, norm='group'):
        super().__init__()
        self.branches = nn.ModuleList([self.build(channels, recipe, norm) for recipe in self.BRANCHES])
        self.act      = nn.ReLU(inplace=True)

    def build(self, channels, recipe, norm):
        last   = len(recipe) - 1
        blocks = [ConvBlock(channels, channels, kernel, dilation, norm, step < last) for step, (kernel, dilation) in enumerate(recipe)]
        return nn.Sequential(*blocks)

    # EQUACOES 2 E 3: F_barra = ReLU(O1 + O2 + O3 + F)
    def forward(self, x):
        total = x

        for branch in self.branches:
            total = total + branch(x)

        return self.act(total)


# ATENCAO FAULT-SEG (FIGURA 3): SUBSTITUI A CONCATENACAO DA SKIP POR UMA ATENCAO ESPACIAL POR CANAL.
# F_q (encoder) e F_k (decoder reamostrado) sao concatenados e projetados de volta a C canais (eq. 5); duas
# 1-convs desse fundido dao a matriz S por produto e softmax (eq. 6); S pondera F_v e o resultado volta por
# soma (eq. 7). O eixo dos tokens e o tempo (z): na figura 3 a matriz S e C x H x H, e H e o eixo vertical da
# secao sismica - o mesmo em que a falha se prolonga e em que a segmentacao perde continuidade.
# S E POR SECAO INLINE, COMO NO ARTIGO: ele e 2D e cada S compara as linhas de UMA secao ao longo dos seus W
# tracos (d = W = 128). A versao anterior deste port achatava o plano x*y inteiro na dimensao da feature
# (d = 16384 no estagio 1). Agora cada fatia x tem a sua S de z por z sobre os tracos y dela, a escala 1/sqrt(d)
# volta a ser a do 2D, e o produto passa pelo scaled_dot_product_attention, que nao materializa as (lote*canal*x)
# matrizes de S. A norma sobre F_l continua (sem ela S nasce quase one-hot, peso maximo medio 0.71 por linha).
# O QUE A REDE APRENDE AQUI, NAS DUAS VERSOES: S uniforme. Peso na diagonal 0.04-0.06 no model_6 antigo e 0.010
# nesta (o uniforme e 1/128 = 0.008), distancia media |z - z'| de 42-45 contra 42.7 do uniforme - a atencao soma,
# em cada coluna (x, y), a media vertical das features de todas as profundidades (66% da norma do fundido), um vies
# "falha atravessa o tile na vertical" que o sintetico premia. A rede depende dele: zerar o termo na inferencia leva
# a mascara no Marlim de 3.5% para 9.1% da secao. E o proprio mecanismo do artigo com este dado, nao erro do port
class FaultSegAttention(nn.Module):
    def __init__(self, channels, norm='group'):
        super().__init__()
        self.fuse  = nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False)
        self.norm  = getNorm(channels, norm)
        self.query = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.key   = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    # (lote, canal, x, z, y) -> (lote*canal*x, 1, z, y): cada canal de cada secao inline vira uma matriz propria
    def tokens(self, x):
        batch, channels, sx, sz, sy = x.shape
        return x.reshape(batch * channels * sx, 1, sz, sy)

    def forward(self, down, up, neighbors):
        fused = self.norm(self.fuse(torch.cat([down, up], dim=1)))
        query = self.tokens(self.query(fused))
        key   = self.tokens(self.key(fused))
        mixed = F.scaled_dot_product_attention(query, key, self.tokens(neighbors.contiguous()))
        return mixed.reshape(fused.shape) + fused


# PERDA DO MSR (EQUACAO 4): DISTANCIA ENTRE A FEATURE LOCAL E A REFINADA, AMBAS NORMALIZADAS EM L2 - ela
# compara direcoes, nao valores, e por isso vale no maximo 2 por estagio. A identidade sqrt(2 - 2*cos) da o
# mesmo numero e o mesmo gradiente e parece mais barata, mas nao e: medida no estagio 1 (lote 2, 16 canais,
# 128^3) ela retem 0.500 GiB ate a backward contra 0.250 GiB desta, porque o cosseno guarda as duas features
# normalizadas e a subtracao guarda so a diferenca. Feature identica ou zerada dos dois lados da gradiente
# nulo, nao NaN
def msrLoss(local, refined):
    total = 0.0

    for before, after in zip(local, refined):
        difference = F.normalize(before.flatten(1), dim=1) - F.normalize(after.flatten(1), dim=1)
        total      = total + difference.norm(dim=1).mean()

    return total


# FAULT-SEG-NET (LI ET AL., 2023 - COMPUTERS AND GEOTECHNICS 158, 105412) EM 3D: U-NET DE CINCO ESTAGIOS EM
# QUE CADA ESTAGIO DO ENCODER E REFINADO PELO MODULO MULTIESCALA RESIDUAL E CADA SKIP VIRA UMA ATENCAO
# FAULT-SEG. O artigo e 2D e injeta as fatias vizinhas por F_v; em 3D o volume ja traz as vizinhas, e o
# equivalente direto e uma convolucao (3,1,1) que mistura so o eixo das fatias (o inline, x de (x,z,y)).
# Saida em logits: o sigmoid vive na loss e na inferencia, como o resto do projeto espera
class FaultSegNet(nn.Module):
    NEIGHBOR = (3, 1, 1)
    POOL     = 2
    STAGES   = 5

    # PESO DA PERDA DE SEGMENTACAO CONTRA A DO MSR (EQUACAO 12). O artigo usa 0.7; o padrao aqui e 1.0, que
    # desliga o termo do MSR, porque ele foi medido como nocivo neste pipeline: em 15 epocas sobre 40 tiles
    # em 64^3, eta=0.7 da val_iou 0.166 com smooth_dice e 0.002 com a composta do artigo, contra 0.277 e
    # 0.269 com eta=1.0 - desligar o termo rende mais que dobrar num_filters (0.228 com f=32 e eta=0.7).
    # Nao e colapso dos ramos do MSR: a norma dos pesos deles fica em 100.4% da inicial. O termo e de fato
    # minimizado, e cedo - L_MSR cai de 2.02 para 0.24 em 8 epocas enquanto o train_iou so vai de 0.062 a
    # 0.103; com eta=1.0 o L_MSR fica em 3.78 e o train_iou chega a 0.218. Ele tem gradiente sobre F_s
    # tambem, entao prende o encoder a um criterio que nao e o da segmentacao. ETA = 0.7 volta ao artigo
    ETA = 1.0

    # O MSR E O DECODER DO ESTAGIO 1 (RESOLUCAO PLENA) SAO RECALCULADOS NA BACKWARD EM VEZ DE GUARDADOS: COM f=32 EM 128^3 E LOTE 2 A
    # REDE PASSAVA DE 19 GiB E NAO DIVIDIA A PLACA COM MAIS NADA. Sem dropout nem estado dentro deles, a saida e a mesma
    CHECKPOINT = True

    def __init__(self, in_channels=1, num_classes=1, base_filters=16, dropout_rate=0.0,
                 input_shape=(128, 128, 128), norm='group'):
        super().__init__()
        self.input_shape = tuple(int(size) for size in input_shape)
        self.penalty     = 0.0

        f      = int(base_filters)
        widths = [f * 2 ** stage for stage in range(self.STAGES)]
        skips  = widths[:-1]
        inputs = [in_channels] + skips
        drop   = float(dropout_rate)

        self.encoders = nn.ModuleList([ConvBlock(source, width, 3, 1, norm) for source, width in zip(inputs, widths)])

        # O MSR SO ENTRA NOS QUATRO PRIMEIROS ESTAGIOS: A SAIDA DO QUINTO NAO TEM DETALHE LOCAL (EQUACAO 2)
        self.refiners  = nn.ModuleList([MultiScaleResidual(width, norm) for width in skips])
        self.neighbors = nn.ModuleList([ConvBlock(width, width, self.NEIGHBOR, 1, norm) for width in skips])

        # A SAIDA DO QUINTO ESTAGIO PASSA POR UMA CONV 3x3 + ReLU ANTES DE SUBIR (A SETA VERTICAL DO FUNDO DA FIGURA 1)
        self.bottom = ConvBlock(widths[-1], widths[-1], 3, 1, norm)

        self.ups        = nn.ModuleList([nn.Conv3d(deep, width, kernel_size=1) for width, deep in zip(skips, widths[1:])])
        self.attentions = nn.ModuleList([FaultSegAttention(width, norm) for width in skips])
        self.decoders   = nn.ModuleList([ConvBlock(width, width, 3, 1, norm) for width in skips])

        self.pool = nn.MaxPool3d(kernel_size=self.POOL, stride=self.POOL)
        self.drop = nn.Dropout3d(drop) if drop > 0 else nn.Identity()
        self.out  = nn.Conv3d(widths[0], num_classes, kernel_size=1)

    # PERDA TOTAL DO ARTIGO (EQUACAO 12): L = eta*L_composta + (1-eta)*L_MSR. O termo do MSR so existe
    # dentro da rede, entao e ela que fecha a soma; o Trainer chama isto quando o modelo oferece o metodo
    def compose(self, loss):
        return self.ETA * loss + (1.0 - self.ETA) * self.penalty

    def full(self, stage, unit, *args):
        if self.CHECKPOINT and self.training and stage == 0:
            return checkpoint(unit, *args, use_reentrant=False)

        return unit(*args)

    # UM ESTAGIO DO DECODER: SOBE O NIVEL DE BAIXO, ATENCAO FAULT-SEG COM A SKIP E A CONV DO ESTAGIO
    def decode(self, stage, x, skip):
        up = self.ups[stage](resample(x, skip.shape[2:]))
        return self.decoders[stage](self.attentions[stage](skip, up, self.neighbors[stage](skip)))

    def forward(self, x):
        local, refined = [], []

        for stage, encoder in enumerate(self.encoders):
            x = encoder(x if stage == 0 else self.pool(x))

            if stage < len(self.refiners):
                local.append(x)
                x = self.full(stage, self.refiners[stage], x)
                refined.append(x)

            x = self.drop(x)

        self.penalty = msrLoss(local, refined) if self.ETA < 1.0 else 0.0
        x = self.bottom(x)

        for stage in reversed(range(len(self.decoders))):
            x = self.full(stage, self.decode, stage, x, refined[stage])

        return self.out(x)
