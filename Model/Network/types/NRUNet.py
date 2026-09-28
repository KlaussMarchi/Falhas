import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# REAMOSTRAGEM: TRILINEAR COM align_corners=False, A MESMA DO CODIGO DO AUTOR (lanl/scf) E DAS OUTRAS REDES DAQUI
def resample(x, size):
    if tuple(x.shape[2:]) == tuple(size):
        return x

    return F.interpolate(x, size=tuple(size), mode='trilinear', align_corners=False)


# NORMALIZACAO: o artigo usa BatchNorm com lote 16; aqui o lote e 2, e com lote 2 as estatisticas moveis saem
# ruins - o mesmo efeito ja medido na MACNN, do mesmo autor. O proprio autor trocou o BN por InstanceNorm no
# codigo de 2025 (lanl/scf), que tambem nao depende do lote. GroupNorm e o que as redes daqui usam; todas as
# larguras da NRU sao multiplas de 8. norm='batch' volta ao artigo
def getNorm(channels, norm):
    if norm == 'batch':
        return nn.BatchNorm3d(channels)

    groups = 8 if channels >= 8 and channels % 8 == 0 else 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


# CONV 3 + NORMA + ReLU: TODA CAMADA DAS SUB-REDES (FIGURA 3, "Conv + BN + ReLU"), COM DILATACAO OPCIONAL
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, norm='group'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm = getNorm(out_channels, norm)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# U-NET RESIDUAL INTERNA (FIGURA 3, EQUACOES 10-19): uma conv de entrada I, um U com as larguras e dilatacoes da
# receita, e a soma de I na saida (E = I + U(I)). O pooling de cada nivel vem ANTES da conv do nivel, e a camada
# dilatada do centro tambem desce um nivel: na figura a barra cinza e mais baixa que a azul anterior e a seta verde
# que sai dela e um upsampling x2. O codigo do autor (resu1/resu2/resu3 em lanl/scf, 2025) confirma, com um
# terceiro pooling antes da conv dilatada. Aquele codigo tem uma conv a mais (up0) antes da soma, que nao esta na
# figura nem na contagem de 8 camadas do texto - fica de fora
class ResidualUNet(nn.Module):
    def __init__(self, in_channels, base, widths, dilations, pools, norm='group'):
        super().__init__()
        sizes      = [base * width for width in widths]
        self.pools = pools

        self.first    = ConvBlock(in_channels, sizes[0], 1, norm)
        self.encoders = nn.ModuleList([ConvBlock(source, size, dilation, norm) for source, size, dilation in zip([sizes[0]] + sizes[:-1], sizes, dilations)])

        # O DECODER DO NIVEL i RECEBE O NIVEL i+1 (JA DECODIFICADO, OU O CENTRO) CONCATENADO COM O ENCODER i
        self.decoders = nn.ModuleList([ConvBlock(sizes[level + 1] + sizes[level], sizes[level], dilations[level], norm) for level in reversed(range(len(sizes) - 1))])

    def forward(self, x):
        first = self.first(x)
        skips = []
        y     = first

        for encoder, pool in zip(self.encoders, self.pools):
            y = encoder(F.max_pool3d(y, kernel_size=2) if pool else y)
            skips.append(y)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            y = decoder(torch.cat([resample(y, skip.shape[2:]), skip], dim=1))

        return first + y


# NRU (GAO, HUANG & ZHENG, 2022 - IEEE TGRS 60, 4502215): TRES NIVEIS (ENCODERS 1-3, DECODERS 2-1) EM QUE CADA
# ENCODER E DECODER E UMA U-NET RESIDUAL PROPRIA, E TRES MAPAS DE FALHA (UM POR NIVEL) FUNDIDOS NO FINAL.
# Saida em logits: o sigmoid da equacao 9 vive na loss e na inferencia, como o resto do projeto espera
class NRUNet(nn.Module):
    POOL = 2

    # RECEITAS DA FIGURA 3, COM A LARGURA RELATIVA A BASE DE CADA UNIDADE: (a) 32-64-128-512, (b) 64-128-512 e
    # (c) 16-32-32-64 - esta sem nenhum pooling e com as dilatacoes 1, 2, 4 e 8 (EQUACAO 15)
    UNITS = {
        'high': {'widths': (1, 2, 4, 16), 'dilations': (1, 1, 1, 2), 'pools': (False, True, True, True)},
        'medium': {'widths': (1, 2, 8), 'dilations': (1, 1, 2), 'pools': (False, True, True)},
        'low': {'widths': (1, 2, 2, 4), 'dilations': (1, 2, 4, 8), 'pools': (False, False, False, False)}
    }

    CHECKPOINT = True

    def __init__(self, in_channels=1, num_classes=1, base_filters=32, dropout_rate=0.0,
                 input_shape=(128, 128, 128), norm='group'):
        super().__init__()
        self.input_shape = tuple(int(size) for size in input_shape)

        f    = int(base_filters)
        C1   = f
        C2   = 2 * f
        C3   = f // 2
        drop = float(dropout_rate)

        # EQUACOES 10-18: C1 = 32, C2 = 64 E C3 = 16 NO ARTIGO, OU SEJA f, 2f E f/2
        self.encoder1 = ResidualUNet(in_channels, C1, norm=norm, **self.UNITS['high'])
        self.encoder2 = ResidualUNet(C1, C2, norm=norm, **self.UNITS['medium'])
        self.encoder3 = ResidualUNet(C2, C3, norm=norm, **self.UNITS['low'])
        self.decoder2 = ResidualUNet(C2 + C3, C2, norm=norm, **self.UNITS['medium'])
        self.decoder1 = ResidualUNet(C1 + C2, C1, norm=norm, **self.UNITS['high'])

        # EQUACOES 6-9: UM MAPA POR NIVEL (CONV 3, SEM NORMA NEM ATIVACAO), FUNDIDOS POR UMA CONV 1
        self.heads = nn.ModuleList([nn.Conv3d(channels, num_classes, kernel_size=3, padding=1) for channels in (C1, C2, C3)])
        self.fuse  = nn.Conv3d(3 * num_classes, num_classes, kernel_size=1)

        self.pool = nn.MaxPool3d(kernel_size=self.POOL, stride=self.POOL)
        self.drop = nn.Dropout3d(drop) if drop > 0 else nn.Identity()

    # AS DUAS U-NETS EM RESOLUCAO PLENA SAO RECALCULADAS NA BACKWARD EM VEZ DE GUARDADAS: SEM ISSO A REDE PASSA DE 19 GiB
    # EM 128^3 COM LOTE 2; COM ISSO FICA EM 13.6 GiB E 4.6 s/passo NA P6000. A saida e bit a bit a mesma (sem dropout
    # dentro delas, e o GroupNorm nao tem estado)
    def full(self, unit, x):
        if self.CHECKPOINT and self.training:
            return checkpoint(unit, x, use_reentrant=False)

        return unit(x)

    def forward(self, x):
        first  = self.drop(self.full(self.encoder1, x))
        second = self.drop(self.encoder2(self.pool(first)))
        third  = self.drop(self.encoder3(self.pool(second)))

        middle = self.drop(self.decoder2(torch.cat([second, resample(third, second.shape[2:])], dim=1)))
        top    = self.full(self.decoder1, torch.cat([first, resample(middle, first.shape[2:])], dim=1))

        maps = [head(level) for head, level in zip(self.heads, (top, middle, third))]
        maps = [resample(level, x.shape[2:]) for level in maps]
        return self.fuse(torch.cat(maps, dim=1))
