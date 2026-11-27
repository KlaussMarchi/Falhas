import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import trunc_normal_
from .ResACEUnet_Zu import ACE


# PAR DE CONVOLUÇÕES 3x3x3 COM ReLU DA UNET DO WU, SEM NORMALIZAÇÃO, COM ATALHO RESIDUAL (1x1 QUANDO MUDA O NÚMERO DE CANAIS)
class ResPair(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1    = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2    = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))) + self.shortcut(x))


# BLOCO DE TOKENS PRÉ-NORMA COM O ACE DO ARTIGO DO ZU E UM MLP; CADA RAMO PASSA POR UM GANHO QUE NASCE EM 1e-6 (LAYER SCALE), ENTÃO O BLOCO COMEÇA COMO IDENTIDADE
class TokenBlock(nn.Module):
    def __init__(self, channels, tokens, proj, heads, dropout):
        super().__init__()
        self.norm1  = nn.LayerNorm(channels)
        self.norm2  = nn.LayerNorm(channels)
        self.ace    = ACE(input_size=tokens, hidden_size=channels, proj_size=proj, channel_attn_drop=dropout, spatial_attn_drop=dropout, num_heads=heads)
        self.mlp    = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * channels, channels))
        self.gamma1 = nn.Parameter(1e-6 * torch.ones(channels))
        self.gamma2 = nn.Parameter(1e-6 * torch.ones(channels))

    def forward(self, x):
        shape  = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.gamma1 * self.ace(self.norm1(tokens))
        tokens = tokens + self.gamma2 * self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(shape)


# HÍBRIDA DA UNET DO WU (ESQUELETO, SUBIDA POR VIZINHO CONCATENADA AO SALTO, ENTRADA EM z, SEM NORMALIZAÇÃO) COM PARES RESIDUAIS E OS BLOCOS ACE DO ZU NOS NÍVEIS 1/4 E 1/8
class ResACEUnetGrva2(nn.Module):
    WIDTHS = (2, 4, 8, 16)    # CANAIS POR NÍVEL EM MÚLTIPLOS DE base_filters: O DOBRO DA UNET DO WU
    LEVELS = (2, 3)           # NÍVEIS COM BLOCOS DE TOKENS: 1/4 E 1/8 DA ENTRADA, COMO OS ESTÁGIOS ACE DO ZU
    PROJ   = (64, 32)         # TOKENS PROJETADOS NA ATENÇÃO ESPACIAL DE CADA NÍVEL (OS DO ZU)

    def __init__(self, in_channels=1, num_classes=1, base_filters=16, img_size=(128, 128, 128), depths=(3, 3), heads=4, dropout=0.0, center=0.0, scale=1.0):
        super().__init__()
        self.center, self.scale = center, scale
        f = [base_filters * m for m in self.WIDTHS]
        n = lambda k: int(np.prod([s // 2 ** k for s in img_size]))    # O ACE PROJETA OS TOKENS DE UM TAMANHO FIXO: A REDE FICA PRESA À JANELA DO TREINO

        self.encoder = nn.ModuleList([ResPair(i, o) for i, o in zip([in_channels] + f[:-1], f)])
        self.tokens  = nn.ModuleList([nn.Sequential(*[TokenBlock(f[k], n(k), p, heads, dropout) for _ in range(d)]) for k, d, p in zip(self.LEVELS, depths, self.PROJ)])
        self.decoder = nn.ModuleList([ResPair(f[k + 1] + f[k], f[k]) for k in (2, 1, 0)])
        self.drop    = nn.Dropout3d(dropout)
        self.out     = nn.Conv3d(f[0], num_classes, 1)

        # CONVOLUÇÕES COM A INICIALIZAÇÃO DO KERAS DA UNET DO WU (SEM NORMALIZAÇÃO, É ELA QUE FIXA A ESCALA DO SINAL); LINEARES COM A DO ZU
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x     = (x - self.center) / self.scale
        skips = []

        for k, block in enumerate(self.encoder):
            x = block(x if k == 0 else F.max_pool3d(x, 2))
            x = self.tokens[self.LEVELS.index(k)](x) if k in self.LEVELS else x
            skips.append(x)

        x = self.drop(skips.pop())

        for block in self.decoder:
            x = block(torch.cat([F.interpolate(x, scale_factor=2, mode='nearest'), skips.pop()], 1))

        return self.out(x)
