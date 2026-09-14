import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# REAMOSTRAGEM UNICA DO ARTIGO: TRILINEAR PARA SUBIR E PARA DESCER (A U-NET DO WU USA VIZINHO MAIS PROXIMO)
def resample(x, size):
    if tuple(x.shape[2:]) == tuple(size):
        return x

    return F.interpolate(x, size=tuple(size), mode='trilinear', align_corners=False)


# NORMALIZACAO: o artigo usa BatchNorm, mas ele treinou com lote efetivo 16 e aqui o lote e 2. Com lote 2 as
# estatisticas moveis saem ruins e a rede fica pior em eval() do que em train() por motivo estatistico, nao
# de generalizacao: medido ao longo de 10 epocas, o gap treino-validacao do MACNN com BN ficou em +0.031
# (desvio 0.024) enquanto os quatro baselines do repo ficam entre -0.013 e -0.053 (desvio 0.007 a 0.013).
# GroupNorm nao depende do lote e e o que as outras redes daqui ja usam. norm='batch' volta ao artigo
def getNorm(channels, norm):
    if norm == 'batch':
        return nn.BatchNorm3d(channels)

    groups = 8 if channels >= 8 and channels % 8 == 0 else 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


# CONV 3 + NORMA + ReLU: A UNIDADE BASICA DE TODO O MACNN (VERDE E AZUL DA FIGURA 3)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, norm='group'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm = getNorm(out_channels, norm)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# BLOCO DE TRIPLA CONVOLUCAO (FIGURA 3), QUE SUBSTITUI A DUPLA CONVOLUCAO DA U-NET: a do meio e dilatada
# (rate 2) para ampliar o campo receptivo, a entrada do bloco volta por concatenacao na terceira conv
# (a "skip connection") e a primeira conv volta por soma na saida (a "skip addition entre a primeira e a
# terceira conv"). A tupla (L1, L2, L3) da figura 1 e (entrada, largura do meio, saida): esta leitura
# reproduz os totais publicados com quatro algarismos - 11,6990 M em 3D e 3,9011 M em 2D contra 11,7 e 3,9
class TripleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout=0.0, norm='group'):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, mid_channels, dilation=1, norm=norm)
        self.conv2 = ConvBlock(mid_channels, mid_channels, dilation=2, norm=norm)
        self.conv3 = ConvBlock(mid_channels + in_channels, out_channels, dilation=1, norm=norm)

        # A SOMA DA PRIMEIRA CONV NA SAIDA SO PRECISA DE PROJECAO SE A LARGURA MUDAR (NUNCA MUDA NO ARTIGO)
        self.shortcut = nn.Identity() if mid_channels == out_channels else nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.drop     = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        first  = self.conv1(x)
        second = self.drop(self.conv2(first))
        third  = self.conv3(torch.cat([second, x], dim=1))
        return third + self.shortcut(first)


# ATENCAO ESPACIAL (FIGURA 2a): ONDE PROCURAR FALHAS. Cada encoder e reamostrado para o nivel alvo e
# comprimido a 1 canal por tres caminhos (media, maximo e 1-conv); o kernel 7 da o campo receptivo grande
class SpatialAttention(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv3d(channels, 1, kernel_size=1) for channels in encoder_channels])
        self.mergers     = nn.ModuleList([nn.Conv3d(3, 1, kernel_size=7, padding=3) for _ in encoder_channels])

    # SUBIR O ENCODER 3 INTEIRO PARA 128^3 CUSTA 2 GiB QUE SO SERVEM PARA SEREM COMPRIMIDOS A 3 CANAIS;
    # o checkpoint recalcula esse trecho no backward em vez de guarda-lo, e o resultado e bit a bit o mesmo
    # (nao ha BN nem dropout aqui). So os 3 canais ja comprimidos ficam na memoria
    def compress(self, encoder, projection, size):
        scaled = resample(encoder, size)
        return torch.cat([scaled.mean(dim=1, keepdim=True),
                          scaled.amax(dim=1, keepdim=True),
                          projection(scaled)], dim=1)

    def forward(self, encoders, size):
        total = 0

        for encoder, projection, merger in zip(encoders, self.projections, self.mergers):
            if self.training:
                pooled = checkpoint(self.compress, encoder, projection, size, use_reentrant=False)
            else:
                pooled = self.compress(encoder, projection, size)

            total = total + merger(pooled)

        return torch.sigmoid(total)


# POOLS GLOBAIS DE CADA ENCODER (MEDIA E MAXIMO POR CANAL): NAO DEPENDEM DO NIVEL ALVO, ENTAO OS TRES
# BLOCOS DE ATENCAO REUSAM O MESMO CALCULO. amax da o mesmo tensor que adaptive_max_pool3d, 265x mais rapido
def globalPools(encoders):
    return [(x.mean(dim=(2, 3, 4), keepdim=True), x.amax(dim=(2, 3, 4), keepdim=True)) for x in encoders]


# ATENCAO DE CANAL (FIGURA 2b): O QUE PROCURAR. A mesma 1-conv atende os pools medio e maximo do encoder
# (o mesmo simbolo K nas equacoes 3 e 4). A reamostragem das equacoes e omitida aqui porque ela antecede um
# pooling global: nao muda a media e move o maximo apenas na borda, e custaria subir o encoder 3 inteiro
class ChannelAttention(nn.Module):
    def __init__(self, encoder_channels, out_channels, hidden):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv3d(channels, hidden, kernel_size=1) for channels in encoder_channels])
        self.merger      = nn.Conv3d(hidden, out_channels, kernel_size=1)

    def forward(self, pools):
        total = 0

        for (average, maximum), projection in zip(pools, self.projections):
            total = total + projection(average) + projection(maximum)

        return torch.sigmoid(self.merger(total))


# BLOCO DE ATENCAO ESPACO-CANAL MULTIESCALA (FIGURA 2c/2d): refina UM encoder usando os TRES, por produto
# elemento a elemento com os dois mapas, antes de concatenar no decoder do mesmo nivel
class MultiscaleAttention(nn.Module):
    def __init__(self, encoder_channels, out_channels, hidden):
        super().__init__()
        self.spatial = SpatialAttention(encoder_channels)
        self.channel = ChannelAttention(encoder_channels, out_channels, hidden)

    def forward(self, encoders, index, pools):
        target = encoders[index]
        return target * self.spatial(encoders, target.shape[2:]) * self.channel(pools)


# MACNN (GAO ET AL., 2022 - GEOPHYSICS 87/1, N13-N29): U-NET DE TRES NIVEIS EM QUE CADA ENCODER E REFINADO
# POR ATENCAO ESPACO-CANAL CONSTRUIDA A PARTIR DOS TRES ENCODERS, NAO SO DO SEU PROPRIO NIVEL.
# Saida em logits: o sigmoid da figura 1 vive na loss e na inferencia, como o resto do projeto espera
class MACNN(nn.Module):
    POOL = 2

    def __init__(self, in_channels=1, num_classes=1, base_filters=32, dropout_rate=0.0,
                 input_shape=(128, 128, 128), norm='group'):
        super().__init__()
        self.input_shape = tuple(int(size) for size in input_shape)

        f       = int(base_filters)
        widths  = [f, f * 2, f * 4]
        bottom  = f * 8
        drop    = float(dropout_rate)

        self.size_multiple = tuple(self.POOL ** len(widths) for _ in range(3))

        self.encoder1 = TripleConv(in_channels, widths[0], widths[0], dropout=drop, norm=norm)
        self.encoder2 = TripleConv(widths[0], widths[1], widths[1], dropout=drop, norm=norm)
        self.encoder3 = TripleConv(widths[1], widths[2], widths[2], dropout=drop, norm=norm)
        self.link     = TripleConv(widths[2], bottom, bottom, dropout=drop, norm=norm)

        self.pool = nn.MaxPool3d(kernel_size=self.POOL, stride=self.POOL)

        # UM BLOCO DE ATENCAO POR NIVEL, TODOS ALIMENTADOS PELOS TRES ENCODERS
        self.attentions = nn.ModuleList([MultiscaleAttention(widths, channels, channels) for channels in widths])

        self.decoder3 = TripleConv(bottom + widths[2], widths[2], widths[2], dropout=drop, norm=norm)
        self.decoder2 = TripleConv(widths[2] + widths[1], widths[1], widths[1], dropout=drop, norm=norm)
        self.decoder1 = TripleConv(widths[1] + widths[0], widths[0], widths[0], dropout=drop, norm=norm)

        self.head = ConvBlock(widths[0], widths[0], dilation=1, norm=norm)
        self.out  = nn.Conv3d(widths[0], num_classes, kernel_size=1)

    # DECODER: SOBE POR INTERPOLACAO TRILINEAR E CONCATENA O ENCODER JA REFINADO PELA ATENCAO
    def up(self, x, skip, block):
        merged = torch.cat([resample(x, skip.shape[2:]), skip], dim=1)
        return block(merged)

    def forward(self, x):
        first  = self.encoder1(x)
        second = self.encoder2(self.pool(first))
        third  = self.encoder3(self.pool(second))
        bottom = self.link(self.pool(third))

        encoders = [first, second, third]
        pools    = globalPools(encoders)
        refined  = [attention(encoders, index, pools) for index, attention in enumerate(self.attentions)]

        x = self.up(bottom, refined[2], self.decoder3)
        x = self.up(x, refined[1], self.decoder2)
        x = self.up(x, refined[0], self.decoder1)
        return self.out(self.head(x))
