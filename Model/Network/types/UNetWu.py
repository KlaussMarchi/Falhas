import torch
import torch.nn as nn
import torch.nn.functional as F


# UNET DO FAULTSEG3D (WU ET AL. 2019, unet3.py EM github.com/xinwucwp/faultSeg): 15 CONVOLUÇÕES 3x3x3 COM ReLU, SEM NORMALIZAÇÃO E
# SEM DROPOUT, MAX POOLING 2, SUBIDA POR VIZINHO MAIS PRÓXIMO CONCATENADA AO SALTO ([SUBIDA, SALTO]) E CONV 1x1 NO FIM. COM
# base_filters = 16 SÃO OS CANAIS 16-32-64-128 E OS 1,46 MILHÃO DE PARÂMETROS DO fseg-70. DEVOLVE LOGITS (A SIGMOIDE FICA NO Trainer).
# (x - center) / scale LEVA A ENTRADA À ESCALA EM QUE A PRIMEIRA CONVOLUÇÃO APRENDE (O PADRÃO 0 E 1 NÃO MEXE NELA)
class UNetWu(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_filters=16, center=0.0, scale=1.0):
        super().__init__()
        self.center, self.scale = center, scale
        f     = [base_filters * m for m in (1, 2, 4, 8)]
        sizes = [(in_channels, f[0]), (f[0], f[0]), (f[0], f[1]), (f[1], f[1]), (f[1], f[2]), (f[2], f[2]), (f[2], f[3]), (f[3], f[3]), (f[3] + f[2], f[2]), (f[2], f[2]), (f[2] + f[1], f[1]), (f[1], f[1]), (f[1] + f[0], f[0]), (f[0], f[0])]
        self.convs = nn.ModuleList([nn.Conv3d(i, o, 3, padding=1) for i, o in sizes])
        self.out   = nn.Conv3d(f[0], num_classes, 1)

        # INICIALIZAÇÃO DO Conv3D DO KERAS (glorot_uniform NO NÚCLEO, BIAS ZERO), A DO TREINO DO WU. SEM NORMALIZAÇÃO NA REDE ELA DEFINE
        # O PONTO DE PARTIDA: COM A PADRÃO DO PYTORCH O SINAL CHEGA À SAÍDA ~13x MENOR (MEDIDO NUM TILE DO dataset_wu)
        for conv in [*self.convs, self.out]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    def pair(self, x, k):
        return F.relu(self.convs[k + 1](F.relu(self.convs[k](x))))

    def forward(self, x):
        c1 = self.pair((x - self.center) / self.scale, 0)
        c2 = self.pair(F.max_pool3d(c1, 2), 2)
        c3 = self.pair(F.max_pool3d(c2, 2), 4)
        c4 = self.pair(F.max_pool3d(c3, 2), 6)
        c5 = self.pair(torch.cat([F.interpolate(c4, scale_factor=2, mode='nearest'), c3], 1), 8)
        c6 = self.pair(torch.cat([F.interpolate(c5, scale_factor=2, mode='nearest'), c2], 1), 10)
        c7 = self.pair(torch.cat([F.interpolate(c6, scale_factor=2, mode='nearest'), c1], 1), 12)
        return self.out(c7)
