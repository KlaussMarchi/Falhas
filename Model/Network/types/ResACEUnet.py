import torch
import torch.nn as nn
import torch.nn.functional as F


# CRIA GROUPNORM COM AJUSTE AUTOMATICO DO NUMERO DE GRUPOS (ESTAVEL EM BATCHES PEQUENOS)
def get_norm_3d(num_channels, num_groups=8):
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


# MONTA KERNEL POR EIXO: USA K ONDE O EIXO COMPORTA E 1 ONDE O EIXO E PEQUENO DEMAIS
def axis_kernel(shape, k):
    return tuple(k if s >= k else 1 for s in shape)


# STOCHASTIC DEPTH: ZERA ALEATORIAMENTE O RAMO RESIDUAL DE ALGUMAS AMOSTRAS NO TREINO
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)
        return x * mask / keep_prob


# ATENCAO DE CANAL (SQUEEZE-AND-EXCITATION): PONDERA CADA CANAL POR MEDIA E MAXIMO GLOBAIS
class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # open-gate init: sigmoid(+2) ~ 0.88, block starts close to a plain residual unit
        self.gate_bias = nn.Parameter(torch.full((channels,), 2.0))

    def forward(self, x):
        b, c = x.shape[:2]
        avg = x.mean(dim=(2, 3, 4))
        mx = x.amax(dim=(2, 3, 4))
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx) + self.gate_bias).view(b, c, 1, 1, 1)
        return x * gate


# ATENCAO ESPACIAL: GERA MAPA DE SALIENCIA VOXEL A VOXEL A PARTIR DAS ESTATISTICAS DOS CANAIS
class SpatialGate3D(nn.Module):
    def __init__(self, kernel_size=(1, 7, 7)):
        super().__init__()
        pad = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(2.0))   # open-gate init (~0.88)

    def forward(self, x):
        stats = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(stats) + self.gate_bias)


# UNIDADE ACE: ATENCAO DE CANAL SEGUIDA DA ATENCAO ESPACIAL (ORDEM CBAM)
class ACE3D(nn.Module):
    def __init__(self, channels, reduction=4, spatial_kernel=(1, 7, 7)):
        super().__init__()
        self.channel = ChannelAttention3D(channels, reduction)
        self.spatial = SpatialGate3D(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


# BLOCO RESIDUAL BASE (ENCODER E DECODER): CONV-GN-ATIVACAO-CONV-GN-ACE SOMADO AO ATALHO
class ResACEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), dropout=0.0, drop_path=0.0, reduction=4, spatial_kernel=(1, 7, 7)):
        super().__init__()
        pad = tuple(k // 2 for k in kernel_size)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.norm1 = get_norm_3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.norm2 = get_norm_3d(out_channels)

        self.ace = ACE3D(out_channels, reduction=reduction, spatial_kernel=spatial_kernel)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm_3d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.dropout(self.act(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        out = self.ace(out)
        return self.act(residual + self.drop_path(out))


# ESTAGIO DO ENCODER: BLOCOS RESIDUAIS + MAXPOOL; RETORNA (SKIP PRE-POOL, TENSOR POOLADO)
class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, pool_size, dropout, drop_path,
                 spatial_kernel=(1, 7, 7)):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(ResACEBlock(in_channels if i == 0 else out_channels, out_channels,
                                      dropout=dropout, drop_path=drop_path,
                                      spatial_kernel=spatial_kernel))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)
        self.pool_drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        feats = self.blocks(x)
        return feats, self.pool_drop(self.pool(feats))


# CONV RESIDUAL DILATADA: AMPLIA O CAMPO RECEPTIVO DO BOTTLENECK SEM REDUZIR RESOLUCAO
class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation=(1, 2, 2), kernel_size=(1, 3, 3)):
        super().__init__()
        pad = tuple(d * (k // 2) for k, d in zip(kernel_size, dilation))
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            get_norm_3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return x + self.body(x)


# BLOCO TRANSFORMER PRE-NORM: SELF-ATTENTION GLOBAL + MLP COM LAYERSCALE E DROPPATH
class TransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.0, drop_path=0.0,
                 layer_scale_init=1e-2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, tokens):
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + self.drop_path(self.gamma1 * attn_out)
        tokens = tokens + self.drop_path(self.gamma2 * self.mlp(self.norm2(tokens)))
        return tokens


# BOTTLENECK: CONVS DILATADAS + SELF-ATTENTION GLOBAL COM EMBEDDING POSICIONAL INTERPOLAVEL
class GlobalContextBottleneck(nn.Module):
    def __init__(self, channels, grid=(1, 16, 16), depth=2, num_heads=8, mlp_ratio=2.0,
                 dropout=0.0, drop_path=0.1, dilation_small=(1, 2, 2), dilation_large=(1, 4, 4),
                 conv_kernel=(1, 3, 3)):
        super().__init__()
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.dilated = nn.Sequential(
            DilatedResBlock(channels, dilation=dilation_small, kernel_size=conv_kernel),
            DilatedResBlock(channels, dilation=dilation_large, kernel_size=conv_kernel),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, *grid))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock3D(channels, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               dropout=dropout, drop_path=drop_path)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(channels)

    def forward(self, x):
        x = self.dilated(x)

        b, c, d, h, w = x.shape
        pos = self.pos_embed
        if pos.shape[2:] != (d, h, w):
            pos = F.interpolate(pos, size=(d, h, w), mode='trilinear', align_corners=False)

        tokens = (x + pos).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm_out(tokens)
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


# ATTENTION GATE: O DECODER DECIDE VOXEL A VOXEL QUANTO DO SKIP DO ENCODER PASSA
class AttentionGate3D(nn.Module):
    def __init__(self, skip_channels, gate_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(skip_channels // 2, 8)
        self.theta = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.gate_bias = nn.Parameter(torch.tensor(2.0))   # open-gate init (~0.88)

    def forward(self, x, g):
        attn = torch.sigmoid(self.psi(self.act(self.theta(x) + self.phi(g))) + self.gate_bias)
        return x * attn


# ESTAGIO DO DECODER: UPSAMPLE + CONV + SKIP COM ATTENTION GATE + CONCAT + BLOCO RESIDUAL
class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, up_size, dropout, drop_path, spatial_kernel=(1, 7, 7)):
        super().__init__()
        self.up_size = tuple(up_size)
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.up_norm = get_norm_3d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.gate = AttentionGate3D(skip_channels, out_channels)
        self.cat_drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.block = ResACEBlock(out_channels + skip_channels, out_channels, dropout=dropout, drop_path=drop_path, spatial_kernel=spatial_kernel)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.up_size, mode='nearest')
        x = self.act(self.up_norm(self.up_conv(x)))
        skip = self.gate(skip, x)
        x = self.cat_drop(torch.cat([x, skip], dim=1))
        return self.block(x)


# U-NET RESIDUAL COM ATENCAO E BOTTLENECK TRANSFORMER; POOLING E KERNELS DERIVADOS DO INPUT_SHAPE
class ResACEUnet(nn.Module):
    DEPTHS = (1, 1, 2, 2)   # extra blocks where features are semantic & cheap

    def __init__(self, in_channels=1, num_classes=6, base_filters=32, dropout_rate=0.1, deep_supervision=False, transformer_depth=2, num_heads=8, mlp_ratio=2.0, input_shape=(4, 256, 256)):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.input_shape = tuple(input_shape)

        pools = self._compute_pools(self.input_shape, num_stages=len(self.DEPTHS))
        self.pools = pools
        stage_shapes = self._stage_shapes(self.input_shape, pools)
        bot_shape = self._bottleneck_shape(self.input_shape, pools)

        stage_spatial_kernels = [axis_kernel(shape, 7) for shape in stage_shapes]
        bot_spatial_kernel = axis_kernel(bot_shape, 7)
        bot_kernel = axis_kernel(bot_shape, 3)
        bot_dil_small = tuple(2 if k == 3 else 1 for k in bot_kernel)
        bot_dil_large = tuple(4 if k == 3 else 1 for k in bot_kernel)

        f = base_filters
        enc_ch = [f, f * 2, f * 4, f * 8]
        bot_ch = f * 16

        # minimal divisibility required by the cumulative pooling strides
        mult = [1, 1, 1]
        for pool in pools:
            mult = [m * p for m, p in zip(mult, pool)]
        self.size_multiple = tuple(mult)

        dr = dropout_rate
        enc_drops = [dr * 0.25, dr * 0.5, dr * 1.0, dr * 1.0]
        dec_drops = [dr * 1.0, dr * 1.0, dr * 0.5, dr * 0.25]   # deep -> shallow
        drop_paths = [0.0, 0.02, 0.05, 0.08]

        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for ch_out, depth, pool, drop, dpath, sk in zip(
                enc_ch, self.DEPTHS, pools, enc_drops, drop_paths, stage_spatial_kernels):
            self.encoders.append(EncoderStage(ch_in, ch_out, depth, pool, drop, dpath, spatial_kernel=sk))
            ch_in = ch_out

        self.bottleneck_in = ResACEBlock(enc_ch[-1], bot_ch, kernel_size=bot_kernel, dropout=dr * 1.5, drop_path=0.1, spatial_kernel=bot_spatial_kernel)
        self.bottleneck = GlobalContextBottleneck(
            bot_ch, grid=bot_shape, depth=transformer_depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dr * 0.5, drop_path=0.1,
            dilation_small=bot_dil_small, dilation_large=bot_dil_large,
            conv_kernel=bot_kernel,
        )

        self.decoders = nn.ModuleList()
        ch_in = bot_ch
        for ch_out, pool, drop, dpath, sk in zip(
                reversed(enc_ch), reversed(pools), dec_drops, reversed(drop_paths),
                reversed(stage_spatial_kernels)):
            self.decoders.append(DecoderStage(ch_in, ch_out, ch_out, pool, drop, dpath, spatial_kernel=sk))
            ch_in = ch_out

        self.head = nn.Conv3d(enc_ch[0], num_classes, kernel_size=1)
        if deep_supervision:
            self.ds_head3 = nn.Conv3d(enc_ch[2], num_classes, kernel_size=1)   # dec3 (4f)
            self.ds_head2 = nn.Conv3d(enc_ch[1], num_classes, kernel_size=1)   # dec2 (2f)

        self.apply(self._init_weights)

    # DERIVA OS POOLS POR ESTAGIO: CADA EIXO RECEBE SUAS REDUCOES NOS ESTAGIOS MAIS PROFUNDOS
    @staticmethod
    def _compute_pools(input_shape, num_stages=4):
        pools = [[1, 1, 1] for _ in range(num_stages)]
        for axis, size in enumerate(input_shape):
            n_halvings, s = 0, size
            while s >= 2 and n_halvings < num_stages:
                s //= 2
                n_halvings += 1
            for stage in range(num_stages - n_halvings, num_stages):
                pools[stage][axis] = 2
        return tuple(tuple(p) for p in pools)

    # SHAPE NOMINAL DA ENTRADA DE CADA ESTAGIO DO ENCODER (DIMENSIONA OS KERNELS ESPACIAIS)
    @staticmethod
    def _stage_shapes(input_shape, pools):
        shapes = [tuple(input_shape)]
        for pool in pools[:-1]:
            shapes.append(tuple(s // p for s, p in zip(shapes[-1], pool)))
        return shapes

    # SHAPE NOMINAL DO FEATURE MAP NO BOTTLENECK (GRID DO POS_EMBED E KERNELS)
    @staticmethod
    def _bottleneck_shape(input_shape, pools):
        d, h, w = input_shape
        for pd, ph, pw in pools:
            d, h, w = d // pd, h // ph, w // pw
        return (d, h, w)

    # INICIALIZACAO: KAIMING NAS CONVS, TRUNC-NORMAL NOS LINEARS, NORMAS EM IDENTIDADE
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # FORWARD: PAD MINIMO, ENCODER COM SKIPS, BOTTLENECK, DECODER COM GATES, CROP AO TAMANHO ORIGINAL
    def forward(self, x):
        d, h, w = x.shape[2:]

        pad_d = (-d) % self.size_multiple[0]
        pad_h = (-h) % self.size_multiple[1]
        pad_w = (-w) % self.size_multiple[2]
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        skips = []
        for encoder in self.encoders:
            feats, x = encoder(x)
            skips.append(feats)

        x = self.bottleneck_in(x)
        x = self.bottleneck(x)

        decoded = []
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
            decoded.append(x)

        logits = self.head(x)
        if pad_d or pad_h or pad_w:
            logits = logits[:, :, :d, :h, :w]

        if self.deep_supervision and self.training:
            ds3 = F.interpolate(self.ds_head3(decoded[1]), size=(d, h, w), mode='trilinear', align_corners=False)
            ds2 = F.interpolate(self.ds_head2(decoded[2]), size=(d, h, w), mode='trilinear', align_corners=False)
            return [logits, ds3, ds2]

        return logits