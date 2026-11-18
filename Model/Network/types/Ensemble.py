import torch
import torch.nn as nn


# MÉDIA DAS PROBABILIDADES DE VÁRIAS REDES TREINADAS COM SEMENTES DIFERENTES: SAI O LOGIT DA MÉDIA, ENTÃO QUEM APLICA A SIGMOIDE RECEBE A MÉDIA.
# power ≠ 1 DEVOLVE A MÉDIA ELEVADA A ELE (O ENSEMBLE CALIBRADO: power < 1 DEVOLVE O PONTO DE OPERAÇÃO DOS MEMBROS, QUE A MÉDIA AMACIA).
# flips É A LISTA DE EIXOS ESPELHADOS NA ENTRADA (N, C, inline, z, xline), EX. [[], [2], [4], [2, 4]]: CADA MEMBRO VÊ CADA ESPELHO, A SAÍDA É
# DESESPELHADA E ENTRA NA MESMA MÉDIA (TTA LATERAL); SEM flips, NENHUM ESPELHO
class Ensemble(nn.Module):
    def __init__(self, members, power=1.0, flips=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.power   = power
        self.flips   = [tuple(dims) for dims in flips] if flips else [()]

    def forward(self, x):
        flip = lambda t, dims: torch.flip(t, dims) if dims else t
        prob = torch.stack([flip(torch.sigmoid(member(flip(x, dims))), dims) for dims in self.flips for member in self.members]).mean(dim=0) ** self.power
        return torch.logit(prob, eps=1e-6)
