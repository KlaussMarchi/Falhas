import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resaceunet import AttentionGate3D, DilatedResBlock, ResACEBlock, get_norm_3d


# KERNEL DO PLANO DA FALHA: LARGO EM x E y (ONDE A LAMINA E FINA) E ESTREITO EM z (ONDE ELA E CONTINUA);
# eixo curto cai para o maior impar que cabe, nunca para 1, senao o gate perde o contexto lateral
def planeKernel(shape, wide=7, thin=3):
    return tuple(max(min(k, s - (s + 1) % 2), 1) for s, k in zip(shape, (wide, thin, wide)))


# PLANO DE POOLING: GASTA A REDUCAO NO EIXO CONTINUO (z) PRIMEIRO E POUPA OS EIXOS FINOS (x, y)
def poolPlan(input_shape, stages=4, floor=4, targets=(8, 32, 8), deep_axis=1):
    plan = [[1, 1, 1] for _ in range(stages)]

    for axis, (size, target) in enumerate(zip(input_shape, targets)):
        budget, reduction = 0, 1

        while reduction * 2 <= target and size // (reduction * 2) >= floor:
            reduction *= 2
            budget   += 1

        offset = max(budget - stages, 0)

        for step in range(budget):
            stage = max(step - offset, 0) if axis == deep_axis else max(stages - budget + step, 0)
            plan[stage][axis] *= 2

    return tuple(tuple(p) for p in plan)


# SHAPE NOMINAL DE CADA NIVEL, DO 0 (RESOLUCAO PLENA) AO MAIS PROFUNDO (DIMENSIONA KERNELS E DILATACOES)
def levelShapes(input_shape, pools):
    shapes = [tuple(int(s) for s in input_shape)]

    for pool in pools:
        shapes.append(tuple(max(s // p, 1) for s, p in zip(shapes[-1], pool)))

    return shapes


# NIVEL 0: PILHA RESIDUAL ESTREITA QUE NUNCA E REDUZIDA, UNICO CAMINHO COM PRECISAO DE 1 VOXEL
class FineTrunk(nn.Module):
    def __init__(self, in_channels, channels, depth, dropout, spatial_kernel):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            get_norm_3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.blocks = nn.Sequential(*[ResACEBlock(channels, channels, dropout=dropout, spatial_kernel=spatial_kernel)
                                      for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.stem(x))


# ESTAGIO DO ENCODER: REDUZ PRIMEIRO E SO ENTAO PROCESSA, PARA A PROFUNDIDADE CUSTAR O PRECO DO NIVEL
class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, pool, dropout, drop_path, spatial_kernel):
        super().__init__()
        self.pool   = nn.MaxPool3d(kernel_size=pool, stride=pool) if max(pool) > 1 else nn.Identity()
        self.drop   = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.blocks = nn.Sequential(*[ResACEBlock(in_channels if i == 0 else out_channels, out_channels,
                                                  dropout=dropout, drop_path=drop_path, spatial_kernel=spatial_kernel)
                                      for i in range(depth)])

    def forward(self, x):
        return self.blocks(self.drop(self.pool(x)))


# ESTAGIO DO DECODER: UPSAMPLE TRILINEAR ATE O SKIP (SUB-VOXEL), ATTENTION GATE, CONCAT E BLOCO RESIDUAL
class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout, drop_path, spatial_kernel):
        super().__init__()
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.up_norm = get_norm_3d(out_channels)
        self.act     = nn.LeakyReLU(0.1, inplace=True)
        self.gate    = AttentionGate3D(skip_channels, out_channels)
        self.drop    = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.block   = ResACEBlock(out_channels + skip_channels, out_channels, dropout=dropout,
                                   drop_path=drop_path, spatial_kernel=spatial_kernel)

    def forward(self, x, skip):
        x    = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x    = self.act(self.up_norm(self.up_conv(x)))
        skip = self.gate(skip, x)
        return self.block(self.drop(torch.cat([x, skip], dim=1)))


# GARGALO: CONVS DILATADAS NO PLANO DA FALHA; CONTEXTO BARATO, SEM GASTAR CAPACIDADE ONDE O TETO E BAIXO
class ContextBottleneck(nn.Module):
    def __init__(self, in_channels, channels, dropout, drop_path, shape, spatial_kernel):
        super().__init__()
        kernel = tuple(3 if s >= 3 else 1 for s in shape)
        small  = tuple(2 if k == 3 and s >= 5 else 1 for k, s in zip(kernel, shape))
        large  = tuple(4 if k == 3 and s >= 9 else 1 for k, s in zip(kernel, shape))

        self.entry   = ResACEBlock(in_channels, channels, kernel_size=kernel, dropout=dropout,
                                   drop_path=drop_path, spatial_kernel=spatial_kernel)
        self.dilated = nn.Sequential(
            DilatedResBlock(channels, dilation=small, kernel_size=kernel),
            DilatedResBlock(channels, dilation=large, kernel_size=kernel),
        )

    def forward(self, x):
        return self.dilated(self.entry(x))


# CABECA DE FUSAO: TRAZ OS ESTAGIOS FINOS DO DECODER PARA A RESOLUCAO PLENA E DECIDE LA, NAO NO GARGALO
class FusionHead(nn.Module):
    def __init__(self, stage_channels, fine_channels, fuse_channels, classes, dropout, spatial_kernel):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv3d(ch, fuse_channels, kernel_size=1, bias=False)
                                          for ch in stage_channels])

        total = fine_channels + fuse_channels * len(stage_channels)
        self.refine = ResACEBlock(total, fine_channels, dropout=dropout, spatial_kernel=spatial_kernel)
        self.head   = nn.Conv3d(fine_channels, classes, kernel_size=1)

    def forward(self, fine, stages):
        size  = fine.shape[2:]
        feats = [fine]

        for projection, stage in zip(self.projections, stages):
            mapped = projection(stage)
            feats.append(mapped if mapped.shape[2:] == size else
                         F.interpolate(mapped, size=size, mode='trilinear', align_corners=False))

        return self.head(self.refine(torch.cat(feats, dim=1)))


# U-NET RESIDUAL COM ATENCAO REBALANCEADA PARA A BORDA: NIVEL 0 PRESERVADO, POOLING z-FIRST E FUSAO MULTI-ESCALA
class ResACEUnet_GRVA(nn.Module):
    # profundidade nos niveis 1 e 2, que ja tiveram z reduzido mas ainda resolvem a lamina de 2 voxels
    DEPTHS      = (3, 3, 2, 2)
    # a largura para de dobrar no fim: o nivel mais profundo tem teto de IoU 0.14 e faz 4% do trabalho,
    # entao dobrar ali so move capacidade para onde ela nao pode ajudar
    WIDTHS      = (1, 2, 4, 4)
    # profundidade estocastica quase desligada: o gap treino-val medido e de apenas +0.02 (o modelo
    # sub-ajusta, nao sobre-ajusta) e com batch_size=2 o DropPath zera o bloco inteiro com frequencia
    DROP_PATH_MAX = 0.05
    FUSE_STAGES = 3                 # os tres estagios mais finos do decoder; o mais profundo ja chega pela cadeia
    FAULT_PRIOR = 0.074             # fracao de voxels de falha medida no dataset_wu (p10 0.058, p90 0.097)

    def __init__(self, in_channels=1, num_classes=1, base_filters=32, dropout_rate=0.1,
                 input_shape=(128, 128, 128), fine_filters=None, fine_depth=2, fuse_filters=None):
        super().__init__()
        self.input_shape = tuple(int(s) for s in input_shape)

        stages = len(self.DEPTHS)
        pools  = poolPlan(self.input_shape, stages=stages)
        shapes = levelShapes(self.input_shape, pools)
        self.pools = pools

        multiple = [1, 1, 1]
        for pool in pools:
            multiple = [m * p for m, p in zip(multiple, pool)]
        self.size_multiple = tuple(multiple)

        f    = int(base_filters)
        fine = int(fine_filters) if fine_filters else max(f // 2, 8)
        fuse = int(fuse_filters) if fuse_filters else max(f // 4, 4)
        enc  = [f * w for w in self.WIDTHS]

        dr      = float(dropout_rate)
        kernels = [planeKernel(shape) for shape in shapes]

        # ambas as escalas sao indexadas por NIVEL, nao por estagio: pouco dropout e pouca profundidade
        # estocastica onde a precisao de 1 voxel e o produto, o cheio onde ha capacidade sobrando
        half  = stages // 2
        drops = [dr * min(0.25 * 2 ** level, 1.0) for level in range(stages + 1)]
        paths = [self.DROP_PATH_MAX * max(level - half, 0) / (stages - half) for level in range(stages + 1)]

        self.trunk = FineTrunk(in_channels, fine, fine_depth, drops[0], kernels[0])

        self.encoders = nn.ModuleList()
        channels = fine
        for stage in range(stages):
            level = stage + 1
            self.encoders.append(EncoderStage(channels, enc[stage], self.DEPTHS[stage], pools[stage],
                                              drops[level], paths[level], kernels[level]))
            channels = enc[stage]

        self.bottleneck = ContextBottleneck(enc[-1], enc[-1], dr * 1.5, paths[-1], shapes[-1], kernels[-1])

        # o decoder desce de volta ate o nivel 0; o skip do nivel 0 e o tronco, entao o ultimo estagio sai estreito
        skip_channels = [fine] + enc[:-1]
        self.decoders = nn.ModuleList()
        channels = enc[-1]
        for stage in range(stages):
            level = stages - 1 - stage
            self.decoders.append(DecoderStage(channels, skip_channels[level], skip_channels[level],
                                              drops[level], paths[level], kernels[level]))
            channels = skip_channels[level]

        fused = list(reversed(skip_channels))[-self.FUSE_STAGES:]
        self.fusion = FusionHead(fused, fine, fuse, num_classes, drops[0], kernels[0])

        self.apply(self.initWeights)

        # depois do apply, senao o kaiming da cabeca (fan_out=1, std~1.4) estoura o logit e a saida nasce em ~0.21;
        # o bias no logit da prevalencia medida faz a rede comecar calibrada em vez de aprender "quase tudo e fundo"
        nn.init.normal_(self.fusion.head.weight, std=0.01)
        nn.init.constant_(self.fusion.head.bias, math.log(self.FAULT_PRIOR / (1.0 - self.FAULT_PRIOR)))

    # INICIALIZACAO: KAIMING NAS CONVS, TRUNC-NORMAL NOS LINEARS, NORMAS EM IDENTIDADE
    @staticmethod
    def initWeights(module):
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # FORWARD: PAD MINIMO, TRONCO EM RESOLUCAO PLENA, ENCODER/GARGALO/DECODER, FUSAO E CROP AO TAMANHO ORIGINAL
    def forward(self, x):
        d, h, w = x.shape[2:]
        pad = [(-s) % m for s, m in zip((d, h, w), self.size_multiple)]

        if any(pad):
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))

        fine  = self.trunk(x)
        feats = [fine]

        out = fine
        for encoder in self.encoders:
            out = encoder(out)
            feats.append(out)

        out = self.bottleneck(out)

        stages = []
        for decoder, skip in zip(self.decoders, reversed(feats[:-1])):
            out = decoder(out, skip)
            stages.append(out)

        logits = self.fusion(fine, stages[-self.FUSE_STAGES:])

        if any(pad):
            logits = logits[:, :, :d, :h, :w]

        return logits
