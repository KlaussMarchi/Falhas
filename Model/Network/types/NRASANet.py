import torch
import torch.nn as nn
import torch.nn.functional as F


# REAMOSTRAGEM: TRILINEAR COM align_corners=False, A MESMA DAS OUTRAS REDES DAQUI PARA SUBIR
def resample(x, size):
    if tuple(x.shape[2:]) == tuple(size):
        return x

    return F.interpolate(x, size=tuple(size), mode='trilinear', align_corners=False)


# NORMALIZACAO: GroupNorm, como as outras redes daqui usam no lugar do BN dos artigos (lote 2)
def getNorm(channels):
    groups = 8 if channels >= 8 and channels % 8 == 0 else 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


# CONV 3 + NORMA + ReLU: A UNIDADE DA REDE INTEIRA (TRIANGULOS AMARELOS DAS FIGURAS 1-3, "Conv+ReLU"). O artigo nao
# desenha normalizacao nas convs, e sem ela a rede nao treina neste pipeline: medido com lote 2 e lr 1e-4 em recortes de
# 96^3 do dataset_74, IoU de treino e validacao 0.0 nas 6 primeiras epocas com a perda NR e nas 3 primeiras sem ela; na
# inicializacao o desvio cai de ~1.2 na saida do encoder para 0.07 nas unidades do decoder e 0.008 no logit, que sai
# quase constante. A norma vem antes do ReLU, como em todas as convs normalizadas daqui
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.norm = getNorm(out_channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ATENCAO DE RECONSTRUCAO DE CANAL (CRA, FIGURA 4, EQS. 10-15): pesos por canal pelos pools globais medio e maximo,
# separacao dos canais pelo limiar 0.5 e reconstrucao cruzada das metades - a SRU do SCConv (Li et al. 2023) com os
# pesos do CBAM no lugar do gamma do GroupNorm. Tres leituras, cada uma decidida pelo proprio artigo:
# 1. a "conv 3x3x3" da eq. 11 age sobre um mapa 1x1x1 (o pool e global): so o tap central enxerga dado, entao ela e
#    exatamente uma conv 1x1x1 C->C - a mesma para os dois pools, como no CBAM;
# 2. sem o ReLU da eq. 11 na saida da conv: com ele F_A + F_M >= 0 e o sigmoide da eq. 12 nunca cai abaixo de 0.5, o
#    limiar das eqs. 13-14 nao separa canal nenhum, e a figura 4 diz que os pesos vao a ~0 fora das falhas;
# 3. W1 e W2 como as eqs. 13 e 14 estao impressas: W1 mantem os canais de falha (W > 0.5) e atenua os de ruido, W2 o
#    contrario; F1 "contem principalmente falha" e F2 "principalmente ruido", como o texto descreve
class CRA(nn.Module):
    THRESHOLD = 0.5

    def __init__(self, channels):
        super().__init__()
        groups    = 8 if channels >= 8 and channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        w = torch.sigmoid(self.conv(x.mean(dim=(2, 3, 4), keepdim=True)) + self.conv(x.amax(dim=(2, 3, 4), keepdim=True)))

        fault = torch.where(w > self.THRESHOLD, torch.ones_like(w), w) * x
        noise = torch.where(w < self.THRESHOLD, torch.ones_like(w), w) * x

        f11, f12 = torch.chunk(fault, 2, dim=1)
        f21, f22 = torch.chunk(noise, 2, dim=1)
        return torch.cat([f11 + f22, f12 + f21], dim=1)


# UNIDADE DO ENCODER ANINHADO (FIGURA 2, EQS. 1-4): conv de entrada, CRA, e uma U-Net de uma conv por nivel que desce
# sempre ate o fundo da rede (o ultimo nivel de toda unidade tem a largura maxima, 256 no artigo), com a saida do CRA
# somada na saida (eq. 3). O numero de niveis cai de 5 para 1 ao longo do encoder; a En_5 e so conv + CRA
class NestedUnit(nn.Module):
    def __init__(self, in_channels, widths):
        super().__init__()
        self.entry = ConvBlock(in_channels, widths[0])
        self.cra   = CRA(widths[0])
        self.downs = nn.ModuleList([ConvBlock(source, width) for source, width in zip(widths[:-1], widths[1:])])
        self.ups   = nn.ModuleList([ConvBlock(deep + width, width) for width, deep in zip(widths[:-1], widths[1:])])

    def forward(self, x):
        first = self.cra(self.entry(x))
        skips = [first]

        for down in self.downs:
            skips.append(down(F.max_pool3d(skips[-1], kernel_size=2)))

        y = skips[-1]

        for up, skip in zip(reversed(self.ups), reversed(skips[:-1])):
            y = up(torch.cat([resample(y, skip.shape[2:]), skip], dim=1))

        return first + y if len(skips) > 1 else first


# ATENCAO DE DUAS CABECAS DO DECODER (FIGURA 3, EQS. 6-9): por cabeca, Q e K por conv 1x1 + ReLU, V por conv 3x3 + ReLU,
# M = softmax(Q K^T / sqrt(d_c)) sobre os voxels do nivel, e as duas saidas M V concatenadas e fundidas por conv 1x1.
# CADA CABECA SOMA A PROPRIA ENTRADA (M V + F'), o atalho de todo bloco de self-attention desde o Vaswani et al. que o
# artigo cita: a figura 3 nao desenha, mas sem ele o De_1 entrega ao logit so medias ponderadas de V numa janela - com a
# atencao ainda uniforme do inicio do treino o logit sai constante por janela e a rede nao localiza a falha (IoU 0.0 nas
# primeiras epocas, com ou sem a perda NR). A atencao e global e exata enquanto o nivel tem ate GLOBAL tokens (16^3: o De_4 de
# um tile 128^3, onde cada token ja cobre 8^3 voxels e a atencao enxerga o tile inteiro); acima disso ela roda em janelas
# de WINDOW^3 sem sobreposicao (De_3, De_2 e De_1). A versao global nesses niveis nao cabe em hardware nenhum que treine
# com lote 2: medido na P6000 (forward + backward do kernel eficiente, lote 2, duas cabecas), global no De_3 (32^3 tokens)
# custa 24.7 s por passo e janelas de 16^3 no De_1 17 s, contra 0.6 s do De_4 global e 2.7 s das janelas de 8^3 nos tres
# niveis juntos. O artigo nao diz como fez - o De_1 global teria 2,1 milhoes de tokens, 4,4e12 entradas de M por cabeca.
# A atencao entra pelo scaled_dot_product_attention, que nesta placa usa o kernel eficiente e nunca materializa M
class TwoHeadAttention(nn.Module):
    HEADS  = 2
    GLOBAL = 16 ** 3
    WINDOW = 8

    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.query    = nn.Conv3d(channels, self.HEADS * channels, kernel_size=1)
        self.key      = nn.Conv3d(channels, self.HEADS * channels, kernel_size=1)
        self.value    = nn.Conv3d(channels, self.HEADS * channels, kernel_size=3, padding=1)
        self.out      = nn.Conv3d(self.HEADS * channels, out_channels, kernel_size=1)

    # (lote, cabecas*C, x, z, y) -> (lote*janelas, cabecas, tokens, C): cada janela vira uma sequencia propria
    def tokens(self, x, window):
        batch, _, sx, sz, sy = x.shape
        wx, wz, wy = window
        x = x.reshape(batch, self.HEADS, self.channels, sx // wx, wx, sz // wz, wz, sy // wy, wy)
        x = x.permute(0, 3, 5, 7, 1, 4, 6, 8, 2)
        return x.reshape(-1, self.HEADS, wx * wz * wy, self.channels)

    def volume(self, x, shape, window):
        batch, _, sx, sz, sy = shape
        wx, wz, wy = window
        x = x.reshape(batch, sx // wx, sz // wz, sy // wy, self.HEADS, wx, wz, wy, self.channels)
        x = x.permute(0, 4, 8, 1, 5, 2, 6, 3, 7)
        return x.reshape(batch, self.HEADS * self.channels, sx, sz, sy)

    def window(self, size):
        if size[0] * size[1] * size[2] <= self.GLOBAL:
            return tuple(size)

        if any(s % self.WINDOW for s in size):
            raise ValueError(f'nivel {tuple(size)} acima de {self.GLOBAL} tokens e nao divisivel pela janela {self.WINDOW}')

        return (self.WINDOW,) * 3

    def forward(self, x):
        size   = x.shape[2:]
        window = self.window(size)
        q = self.tokens(F.relu(self.query(x)), window)
        k = self.tokens(F.relu(self.key(x)), window)
        v = self.tokens(F.relu(self.value(x)), window)

        mixed = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        mixed = self.volume(mixed, (x.shape[0], 0, *size), window) + x.repeat(1, self.HEADS, 1, 1, 1)
        return self.out(mixed.contiguous())


# UNIDADE DO DECODER (FIGURA 3): SOBE O NIVEL DE BAIXO, CONCATENA A SAIDA DA UNIDADE DO ENCODER, CONV + NORMA + ReLU E ATENCAO
class DecoderUnit(nn.Module):
    def __init__(self, deep, skip, width, out_channels):
        super().__init__()
        self.fuse      = ConvBlock(deep + skip, width)
        self.attention = TwoHeadAttention(width, out_channels)

    def forward(self, x, skip):
        return self.attention(self.fuse(torch.cat([resample(x, skip.shape[2:]), skip], dim=1)))


# PERDA NR (EQ. 5): O QUE A DESCIDA PERDE EM CADA UNIDADE DO ENCODER, ||F_i - resize(MaxPool(F_i))||^2 PARA i = 1..4.
# Como erro quadratico MEDIO: a soma sobre os 16x128^3 voxels do F_1 valeria milhoes contra um Dice em [0, 1], e o peso
# 0.3 da eq. 17 so faz sentido com os dois termos na mesma ordem de grandeza
def nrLoss(features):
    total = 0.0

    for feature in features:
        total = total + F.mse_loss(feature, resample(F.max_pool3d(feature, kernel_size=2), feature.shape[2:]))

    return total


# NRA-SANET (SUN, JIANG, DU & GONG, 2025 - GEOPHYSICAL PROSPECTING 73, 575-594): ENCODER DE CINCO U-NETS RESIDUAIS
# ANINHADAS (5, 4, 3, 2 E 1 NIVEIS, TODAS TERMINANDO NA LARGURA MAXIMA), CADA UMA COM UM CRA NA ENTRADA, E DECODER DE
# QUATRO UNIDADES DE ATENCAO DE DUAS CABECAS. num_filters = C1 (16 no artigo); as larguras dobram ate 16*C1.
# Custo medido na P6000 com f=16, lote 2, 128^3: 15,2 M parametros, 13,7 GiB de pico no treino sem checkpoint.
# Saida em logits: o sigmoide da eq. 9 vive na loss e na inferencia, como o resto do projeto espera
class NRASANet(nn.Module):
    UNITS = 5

    # PESO DA PERDA DE SEGMENTACAO CONTRA A NR (EQ. 17: 0.7 x Dice + 0.3 x NR). O padrao e 1.0, que desliga o termo NR,
    # porque ele foi medido como nocivo aqui - o mesmo que o MSR da Fault-Seg-Net: recortes de 64^3 do dataset_74, lr 1e-4,
    # mesma semente, val_iou na epoca 4 de 0.213 com eta=0.7 contra 0.300 com eta=1.0 (e, antes da norma e do atalho da
    # atencao, IoU 0.0 por 6 epocas com a perda total caindo so pelo termo NR). Ele puxa a saida de cada unidade do encoder
    # para ser constante em blocos de 2x2x2, o oposto do que uma falha de 1-2 voxels precisa. O Trainer chama compose()
    # quando o modelo tem o metodo; com ETA = 1.0 o termo nem e calculado. ETA = 0.7 volta ao artigo
    ETA = 1.0

    def __init__(self, in_channels=1, num_classes=1, base_filters=16, dropout_rate=0.0, input_shape=(128, 128, 128)):
        super().__init__()
        self.input_shape = tuple(int(size) for size in input_shape)
        self.penalty     = 0.0

        f      = int(base_filters)
        widths = [f * 2 ** level for level in range(self.UNITS)]
        inputs = [in_channels] + widths[:-1]
        drop   = float(dropout_rate)

        self.encoders = nn.ModuleList([NestedUnit(source, widths[unit:]) for unit, source in enumerate(inputs)])
        self.decoders = nn.ModuleList([DecoderUnit(widths[level + 1], widths[level], widths[level], widths[level]) for level in range(self.UNITS - 2, 0, -1)])
        self.top      = DecoderUnit(widths[1], widths[0], widths[0], num_classes)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop = nn.Dropout3d(drop) if drop > 0 else nn.Identity()

    # PERDA TOTAL DO ARTIGO (EQ. 17): L = eta * L_segmentacao + (1 - eta) * L_NR
    def compose(self, loss):
        return self.ETA * loss + (1.0 - self.ETA) * self.penalty

    def forward(self, x):
        features, raw = [], []

        for index, encoder in enumerate(self.encoders):
            source = x if index == 0 else self.pool(features[-1])
            raw.append(encoder(source))
            features.append(self.drop(raw[-1]))

        self.penalty = nrLoss(raw[:-1]) if self.ETA < 1.0 else 0.0

        y = features[-1]

        for decoder, skip in zip(self.decoders, reversed(features[1:-1])):
            y = decoder(y, skip)

        return self.top(y, features[0])
