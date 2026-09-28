import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# REAMOSTRAGEM DA FUSAO: TRILINEAR COM align_corners=True, COMO NO CODIGO DA FAULT-NET (douyimin/FaultNet), DE ONDE O
# ARTIGO PARTE. Aqui toda descida e uma conv de stride 2, que amostra os voxels pares; subir com align_corners=False
# deslocaria a grade meio voxel em relacao a essa amostragem
def resample(x, size):
    if tuple(x.shape[2:]) == tuple(size):
        return x

    return F.interpolate(x, size=tuple(size), mode='trilinear', align_corners=True)


# NORMALIZACAO: a Fault-Net usa BatchNorm, e aqui o lote e 2 - o mesmo problema ja medido na MACNN. As larguras desta
# rede sao multiplos de 9 (9, 18 e 36 no artigo), entao o GroupNorm usa 9 grupos. norm='batch' volta ao artigo
def getNorm(channels, norm):
    if norm == 'batch':
        return nn.BatchNorm3d(channels)

    return nn.GroupNorm(num_groups=len(FaultEdgeFormer.DIRECTIONS), num_channels=channels)


# CONV + NORMA + ReLU6, O "Conv3D" DO ALGORITMO 1. O ReLU6 e o da Fault-Net, que o artigo estende
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, norm='group', act=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.norm = getNorm(out_channels, norm)
        self.act  = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# SOBEL 3D MULTIDIRECIONAL (EQUACOES 1-9): cada kernel e a derivada ao longo de uma direcao, com a suavizacao [1, 2, 1]
# em todo eixo que a direcao nao toca. As nove direcoes dao exatamente as nove matrizes do artigo (3 eixos, e as duas
# diagonais de cada plano). Cada kernel e escalado por um gamma treinavel que nasce em 1 (secao 2.1); a saida fica
# linear e com sinal, como na figura 1c - a conv seguinte e que ativa
class SobelConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        repeats = out_channels // len(FaultEdgeFormer.DIRECTIONS)
        kernels = torch.stack([self.build(direction) for direction in FaultEdgeFormer.DIRECTIONS])

        self.register_buffer('kernels', kernels[:, None].repeat(repeats, in_channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.ones(out_channels))

    def build(self, direction):
        offsets = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), dtype=torch.float32)
        weights = torch.tensor(direction, dtype=torch.float32)
        smooth  = torch.where(weights == 0, 2 - offsets.abs(), torch.ones_like(offsets)).prod(dim=1)
        return (offsets @ weights * smooth).reshape(3, 3, 3)

    def forward(self, x):
        return F.conv3d(x, self.kernels * self.gamma[:, None, None, None, None], padding=1)


# BOTTLENECK DA RESNET (O DO ESTAGIO 2 DO ALGORITMO 1): 1-3-1 COM EXPANSAO 4 E ATALHO PROJETADO
class Bottleneck(nn.Module):
    EXPANSION = 4

    def __init__(self, channels, norm='group'):
        super().__init__()
        width = channels * self.EXPANSION

        self.reduce   = ConvBlock(channels, channels, 1, norm=norm)
        self.middle   = ConvBlock(channels, channels, 3, norm=norm)
        self.expand   = ConvBlock(channels, width, 1, norm=norm, act=False)
        self.shortcut = ConvBlock(channels, width, 1, norm=norm, act=False)
        self.act      = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.expand(self.middle(self.reduce(x))) + self.shortcut(x))


# JANELA E DESLOCAMENTO EFETIVOS NUM EIXO: SE O VOLUME CABE NUMA JANELA, A JANELA E O VOLUME E NAO HA DESLOCAMENTO
def windowPlan(sizes, window, shifted):
    windows = tuple(min(window, size) for size in sizes)
    shifts  = tuple(w // 2 if shifted and size > window else 0 for w, size in zip(windows, sizes))
    return windows, shifts


# (B, D, H, W, C) -> (B*nW, C, wd, wh, ww): CADA JANELA VIRA UM VOLUME, PRONTO PARA A CONV DE PROJECAO
def partition(x, windows):
    B, D, H, W, C = x.shape
    wd, wh, ww    = windows
    x = x.reshape(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(-1, C, wd, wh, ww)


# INVERSO DO partition: (B*nW, N, C) -> (B, D, H, W, C)
def merge(tokens, windows, shape):
    B, D, H, W = shape
    wd, wh, ww = windows
    x = tokens.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, D, H, W, -1)


# ATENCAO POR JANELA COM PROJECAO CONVOLUCIONAL (FIGURA 2b, EQUACOES 13-16): Q, K e V saem de uma conv 3x3x3 aplicada
# DENTRO de cada janela, com zero-padding na borda dela - o artigo atribui a isso os artefatos em grade da figura 16.
# O vies de posicao relativa e o da Swin; na janela deslocada a mascara separa as regioes que o roll juntou.
# A atencao usa o kernel eficiente em memoria do PyTorch: medido numa janela 7^3 do fluxo de 64^3, ele retem 26 MB
# por bloco contra 1.2 GB da matriz explicita, desde que o vies entre em broadcast (1, heads, N, N) e nao expandido
class WindowAttention(nn.Module):
    BIAS_STD = 0.02
    MASKED   = -100.0

    # INDICES DO VIES E MASCARAS DAS JANELAS DESLOCADAS, QUE SO DEPENDEM DA GEOMETRIA: UMA COPIA PARA TODOS OS BLOCOS.
    # Guardadas por bloco e em float, as mascaras do fluxo de 64^3 somavam 1.9 GB (470 MB por bloco deslocado)
    indexes = {}
    masks   = {}

    def __init__(self, channels, heads, window, dropout=0.0):
        super().__init__()
        self.heads  = heads
        self.window = window

        self.qkv   = nn.Conv3d(channels, 3 * channels, kernel_size=3, padding=1)
        self.proj  = nn.Linear(channels, channels)
        self.drop  = nn.Dropout(dropout)
        self.table = nn.Parameter(torch.zeros((2 * window - 1) ** 3, heads))
        nn.init.trunc_normal_(self.table, std=self.BIAS_STD)

    # VIES DE CADA PAR (i, j) DE VOXELS DA JANELA, LIDO DA TABELA; O INDICE VALE PARA QUALQUER JANELA <= A DO ARTIGO
    def getBias(self, windows):
        key = (windows, self.window, self.table.device)

        if key not in self.indexes:
            coords   = torch.stack(torch.meshgrid(*[torch.arange(w) for w in windows], indexing='ij')).flatten(1)
            relative = coords[:, :, None] - coords[:, None, :] + self.window - 1
            span     = 2 * self.window - 1
            self.indexes[key] = (relative[0] * span * span + relative[1] * span + relative[2]).to(self.table.device)

        return self.table[self.indexes[key]].permute(2, 0, 1)[None]

    # PARES DE VOXELS DA JANELA DESLOCADA QUE O ROLL TROUXE DE REGIOES DIFERENTES DO VOLUME, QUE NAO PODEM SE ENXERGAR
    def getMask(self, sizes, windows, shifts, device):
        key = (sizes, windows, shifts, device)

        if key not in self.masks:
            regions = torch.zeros(1, *sizes, 1, device=device)
            slices  = [(slice(0, -w), slice(-w, -s), slice(-s, None)) if s else (slice(None),) for w, s in zip(windows, shifts)]

            for label, (d, h, w) in enumerate(itertools.product(*slices)):
                regions[:, d, h, w, :] = label

            labels = partition(regions, windows).flatten(1)
            self.masks[key] = labels[:, None, :] != labels[:, :, None]

        return self.masks[key]

    def forward(self, x, windows, shifts):
        B, D, H, W, C = x.shape
        heads         = self.heads
        N             = windows[0] * windows[1] * windows[2]

        qkv     = self.qkv(partition(x, windows)).flatten(2)
        q, k, v = qkv.view(-1, 3, heads, C // heads, N).permute(1, 0, 2, 4, 3)
        bias    = self.getBias(windows)

        if any(shifts):
            count = q.shape[0] // B
            mask  = (bias + self.getMask((D, H, W), windows, shifts, x.device)[:, None] * self.MASKED).reshape(1, count * heads, N, N)
            out   = F.scaled_dot_product_attention(*[t.reshape(B, count * heads, N, -1) for t in (q, k, v)], attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

        out = out.reshape(-1, heads, N, C // heads).transpose(1, 2).reshape(-1, N, C)
        return merge(self.drop(self.proj(out)), windows, (B, D, H, W))


# MLP COM CONV DEPTHWISE ENTRE AS DUAS LINEARES (FIGURA 2c, EQUACAO 17): Linear -> GELU -> DWConv 3 -> GELU -> Linear.
# A conv roda no volume inteiro, fora das janelas. Toda entrada de conv sai de um permute como copia contigua: com os
# strides de canais-por-ultimo o cuDNN desta P6000 toma outro caminho, 7x mais lento (0.189 s contra 0.028 s na
# depthwise de 36 canais em 64^3, lote 2)
class ConvMLP(nn.Module):
    RATIO = 4

    def __init__(self, channels, dropout=0.0):
        super().__init__()
        hidden = channels * self.RATIO

        self.fc1  = nn.Linear(channels, hidden)
        self.dw   = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2  = nn.Linear(hidden, channels)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.act(self.fc1(x)).permute(0, 4, 1, 2, 3).contiguous()
        hidden = self.act(self.dw(hidden)).permute(0, 2, 3, 4, 1)
        return self.drop(self.fc2(hidden))


# BLOCO SWIN MELHORADO (FIGURA 2a, EQUACAO 10): LN -> (S)W-MSA -> SOMA, LN -> MLP -> SOMA. O volume e completado com
# zeros ate um multiplo da janela e deslocado pela metade dela nos blocos impares, como na Swin
class SwinBlock(nn.Module):
    def __init__(self, channels, heads, window, shifted, dropout=0.0):
        super().__init__()
        self.window  = window
        self.shifted = shifted

        self.norm1 = nn.LayerNorm(channels)
        self.attn  = WindowAttention(channels, heads, window, dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp   = ConvMLP(channels, dropout)

    def forward(self, x):
        x               = x.permute(0, 2, 3, 4, 1)
        B, D, H, W, C   = x.shape
        windows, shifts = windowPlan((D, H, W), self.window, self.shifted)
        pads            = [(-size) % w for size, w in zip((D, H, W), windows)]

        y = F.pad(self.norm1(x), (0, 0, 0, pads[2], 0, pads[1], 0, pads[0]))
        y = torch.roll(y, shifts=[-s for s in shifts], dims=(1, 2, 3)) if any(shifts) else y
        y = self.attn(y, windows, shifts)
        y = torch.roll(y, shifts=shifts, dims=(1, 2, 3)) if any(shifts) else y

        x = x + y[:, :D, :H, :W]
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 4, 1, 2, 3).contiguous()


# TROCA ENTRE RESOLUCOES (EQUACOES 19-20), NA FORMA DA HRNET DA FAULT-NET: subir e conv 1x1 + norma e depois trilinear;
# descer e uma conv 3x3x3 de stride 2 por nivel, com o alinhamento de canais na ultima; a saida e a soma, ativada
class Exchange(nn.Module):
    def __init__(self, source, target, widths, norm='group'):
        super().__init__()
        steps = target - source

        if steps < 0:
            self.path = ConvBlock(widths[source], widths[target], 1, norm=norm, act=False)

        if steps == 0:
            self.path = nn.Identity()

        if steps > 0:
            hops      = [ConvBlock(widths[source], widths[source], 3, 2, norm=norm) for _ in range(steps - 1)]
            self.path = nn.Sequential(*hops, ConvBlock(widths[source], widths[target], 3, 2, norm=norm, act=False))

    def forward(self, x, size):
        return resample(self.path(x), size)


# MODULO MULTIESCALA (SECAO 2.3): CADA FLUXO PASSA PELA SUA UNIDADE TRANSFORMER (DOIS BLOCOS SWIN, O SEGUNDO
# DESLOCADO) E DEPOIS TODOS TROCAM INFORMACAO. outputs=1 devolve so a resolucao mais alta, a unica que o ultimo modulo usa
class MultiScaleModule(nn.Module):
    def __init__(self, widths, heads, outputs, dropout=0.0, norm='group'):
        super().__init__()
        window = FaultEdgeFormer.WINDOW

        self.units     = nn.ModuleList([nn.ModuleList([SwinBlock(width, head, window, shifted, dropout) for shifted in (False, True)]) for width, head in zip(widths, heads)])
        self.exchanges = nn.ModuleList([nn.ModuleList([Exchange(source, target, widths, norm) for source in range(len(widths))]) for target in range(outputs)])
        self.act       = nn.ReLU6(inplace=True)

    # UNIDADE TRANSFORMER DE UM FLUXO. No treino cada bloco e recalculado na backward em vez de guardado: sem isso a rede
    # passa de 19 GiB em 128^3 com lote 2 (o artigo treinou com lote 1 em recortes de 96^3, 4.7x menos voxels por passo).
    # O recalculo e exato - o checkpoint repete o mesmo sorteio do dropout
    def transform(self, unit, x):
        for block in unit:
            x = checkpoint(block, x, use_reentrant=False) if self.training and FaultEdgeFormer.CHECKPOINT else block(x)

        return x

    def forward(self, streams):
        streams = [self.transform(unit, x) for unit, x in zip(self.units, streams)]
        return [self.act(sum(exchange(x, streams[target].shape[2:]) for exchange, x in zip(row, streams))) for target, row in enumerate(self.exchanges)]


# FAULTEDGEFORMER (DI, LIU, CHANG, TIAN, MA & DONG, 2026 - JOURNAL OF GEOPHYSICS AND ENGINEERING 23(4), 1285-1311):
# SOBEL TREINAVEL NA PRIMEIRA CAMADA E UMA HRNET 3D DE TRES FLUXOS (H/2, H/4, H/8) EM QUE CADA UNIDADE BASICA E UM PAR
# DE BLOCOS SWIN COM CONV. O algoritmo 1 e a figura 3 desenham um modulo por estagio, mas a rede da tabela 1 tem dois,
# como a Fault-Net em que ela se baseia: com dois modulos, MLP de razao 4 e vies de posicao relativa, as variantes sem
# a projecao convolucional batem os totais publicados - 0.3226 M (v3, publicado 0.32) e 0.3629 M (v2, publicado 0.36).
# Medido aqui (10 epocas no dataset_74, smooth_dice, lr 1e-4): aprende devagar neste protocolo, val_iou 0.376 contra
# 0.593 da Unet3D_V2 no mesmo orcamento, e F1 de sticks no patch 1200 0.239 contra 0.319.
# Saida em logits: o sigmoid do estagio 5 vive na loss e na inferencia, como o resto do projeto espera
class FaultEdgeFormer(nn.Module):
    DIRECTIONS = ((0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1))    # Sx, Sy, Sz, Sxy45, Sxy135, Sxz45, Sxz135, Syz45, Syz135
    WINDOW     = 7
    HEADS      = (1, 2, 4)
    MODULES    = 2
    CHECKPOINT = True

    def __init__(self, in_channels=1, num_classes=1, base_filters=9, dropout_rate=0.0,
                 input_shape=(128, 128, 128), norm='group'):
        super().__init__()
        self.input_shape = tuple(int(size) for size in input_shape)

        f = int(base_filters)

        if f % len(self.DIRECTIONS):
            raise ValueError(f'num_filters={f}: a largura base do FaultEdgeFormer e um multiplo dos 9 kernels de Sobel (9 no artigo)')

        widths = (f, 2 * f, 4 * f)
        drop   = float(dropout_rate)

        self.stem       = nn.Sequential(SobelConv(in_channels, f), ConvBlock(f, f, 3, norm=norm), ConvBlock(f, f, 3, 2, norm=norm))
        self.bottleneck = Bottleneck(f, norm=norm)
        self.branch1    = ConvBlock(f * Bottleneck.EXPANSION, widths[0], 3, norm=norm)
        self.branch2    = ConvBlock(f * Bottleneck.EXPANSION, widths[1], 3, 2, norm=norm)
        self.branch3    = ConvBlock(widths[1], widths[2], 3, 2, norm=norm)

        self.stage3 = nn.ModuleList([MultiScaleModule(widths[:2], self.HEADS[:2], 2, drop, norm) for _ in range(self.MODULES)])
        self.stage4 = nn.ModuleList([MultiScaleModule(widths, self.HEADS, 3 if index < self.MODULES - 1 else 1, drop, norm) for index in range(self.MODULES)])
        self.out    = nn.Conv3d(widths[0], num_classes, kernel_size=1)
        self.apply(self.initialize)

    # INICIALIZACAO DAS LINEARES DA SWIN (NORMAL TRUNCADA 0.02, VIES ZERO): OS RAMOS RESIDUAIS NASCEM PEQUENOS
    def initialize(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=WindowAttention.BIAS_STD)
            nn.init.zeros_(module.bias)

    # ESTAGIO 5: A CONV 1x1 VEM ANTES DA INTERPOLACAO, AO CONTRARIO DO ALGORITMO 1 - AS DUAS SAO LINEARES E COMUTAM
    # (OS PESOS DA TRILINEAR SOMAM 1), E ASSIM SO 1 CANAL, NAO 9, SOBE PARA A RESOLUCAO PLENA
    def forward(self, x):
        size    = x.shape[2:]
        wide    = self.bottleneck(self.stem(x))
        streams = [self.branch1(wide), self.branch2(wide)]

        for module in self.stage3:
            streams = module(streams)

        streams = streams + [self.branch3(streams[1])]

        for module in self.stage4:
            streams = module(streams)

        return resample(self.out(streams[0]), size)
