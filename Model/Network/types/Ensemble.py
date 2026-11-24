import torch
import torch.nn as nn
import torch.nn.functional as F


# MÉDIA DAS PROBABILIDADES DE VÁRIAS REDES TREINADAS COM SEMENTES DIFERENTES: SAI O LOGIT DA MÉDIA, ENTÃO QUEM APLICA A SIGMOIDE RECEBE A MÉDIA.
# power ≠ 1 DEVOLVE A MÉDIA ELEVADA A ELE (O ENSEMBLE CALIBRADO: power < 1 DEVOLVE O PONTO DE OPERAÇÃO DOS MEMBROS, QUE A MÉDIA AMACIA).
# flips É A LISTA DE EIXOS ESPELHADOS NA ENTRADA (N, C, inline, z, xline), EX. [[], [2], [4], [2, 4]]: CADA MEMBRO VÊ CADA ESPELHO, A SAÍDA É
# DESESPELHADA E ENTRA NA MESMA MÉDIA (TTA LATERAL); SEM flips, NENHUM ESPELHO. scales É A LISTA DE ESCALAS EM QUE CADA JANELA TAMBÉM É VISTA
# (EX. [1, 0.7]: AS FALHAS FICAM 0,7 DO TAMANHO); SEM scales, SÓ A NATIVA
class Ensemble(nn.Module):
    def __init__(self, members, power=1.0, flips=None, scales=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.power   = power
        self.flips   = [tuple(dims) for dims in flips] if flips else [()]
        self.scales  = scales or [1.0]

    # UMA PASSADA NUMA ESCALA: A JANELA É REAMOSTRADA (TRILINEAR), COMPLETADA ATÉ MÚLTIPLO DE 16 (A dbrnet DIVIDE x E y POR 16), PREDITA, CORTADA E DEVOLVIDA AO TAMANHO ORIGINAL
    def scaled(self, member, x, scale):
        if scale == 1:
            return torch.sigmoid(member(x))
        size = [max(16, round(n * scale)) for n in x.shape[2:]]
        pad  = [p for n in size[::-1] for p in (0, -n % 16)]
        prob = torch.sigmoid(member(F.pad(F.interpolate(x, size=size, mode='trilinear', align_corners=False), pad, mode='replicate')))
        return F.interpolate(prob[..., :size[0], :size[1], :size[2]], size=x.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, x):
        flip = lambda t, dims: torch.flip(t, dims) if dims else t
        prob = torch.stack([flip(self.scaled(member, flip(x, dims), scale), dims) for scale in self.scales for dims in self.flips for member in self.members]).mean(dim=0) ** self.power
        return torch.logit(prob, eps=1e-6)
