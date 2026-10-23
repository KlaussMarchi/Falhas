import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# CURVA ROC POR VOXEL SEM GUARDAR AS PROBABILIDADES: CADA LOTE CONTA, NO PRÓPRIO DEVICE, QUANTOS VOXELS DE FALHA E DE FUNDO CAEM
# ENTRE CADA PAR DE LIMIARES, E O TPR E O FPR DE UM LIMIAR SÃO A SOMA DAS FAIXAS ACIMA DELE. OS LIMIARES SÃO LOG-ODDS
# (log(p / (1 - p))) IGUALMENTE ESPAÇADOS EM [-limit, limit]: A GRADE NÃO PERDE RESOLUÇÃO PERTO DE p = 0 E p = 1, ONDE FICA QUASE
# TODO VOXEL, E O LOG-ODDS 0 É EXATAMENTE O LIMIAR 0,5 DA MATRIZ DE CONFUSÃO
class ROC:
    COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')    # SEMPRE NESTA ORDEM
    INK    = '#0b0b0b'
    MUTED  = '#52514e'
    GRID   = '#e4e4e1'
    MARKS  = {
        'threshold': {'marker': 'o', 'markersize': 8, 'markeredgecolor': 'white', 'markeredgewidth': 1.5, 'linestyle': 'none', 'zorder': 4},
        'youden':    {'marker': '*', 'markersize': 15, 'markerfacecolor': '#eb6834', 'markeredgecolor': 'white', 'markeredgewidth': 1.2, 'linestyle': 'none', 'zorder': 5},
    }

    def __init__(self, bins=4000, limit=20.0):
        half = max(bins // 2, 1)
        self.edges     = (torch.arange(-half, half + 1, dtype=torch.float64) * (limit / half)).float()
        self.positives = torch.zeros(len(self.edges) + 1, dtype=torch.int64)
        self.negatives = torch.zeros(len(self.edges) + 1, dtype=torch.int64)

    # --- A CONTA ---

    # logits É A SAÍDA DA REDE (B, C, ...) E masks O RÓTULO. NA SIGMOIDE (C = 1) O LOG-ODDS É O PRÓPRIO LOGIT; NA SOFTMAX É O DE
    # FALHA (QUALQUER CLASSE ≠ 0) CONTRA O FUNDO. COM 2 CLASSES ELE É logit[1] - logit[0] E O LIMIAR 0,5 É O argmax DA MATRIZ;
    # COM MAIS, O argmax NÃO É LIMIAR DE NENHUM ESCORE E O 0,5 DA CURVA É p(falha) > 0,5
    def update(self, logits, masks):
        logits = logits.detach().float()
        scores = logits[:, 0] if logits.shape[1] == 1 else torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
        target = masks.reshape(scores.shape).bool()
        index  = torch.bucketize(scores, self.edges.to(scores.device))    # QUANTOS LIMIARES FICAM ESTRITAMENTE ABAIXO DO VOXEL

        self.positives += torch.bincount(index[target], minlength=len(self.positives)).cpu()
        self.negatives += torch.bincount(index[~target], minlength=len(self.negatives)).cpu()
        return self

    # SEM VOXEL DE FALHA OU SEM VOXEL DE FUNDO NÃO EXISTE CURVA: A AUC E O YOUDEN SAEM nan E A FIGURA DIZ ISSO
    @property
    def defined(self):
        return bool(self.positives.sum() and self.negatives.sum())

    # UM PONTO POR LIMIAR, DO MAIS BAIXO (-inf, TUDO É FALHA, (1, 1)) AO MAIS ALTO (+inf, NADA É FALHA, (0, 0)); O VOXEL CONTA
    # COMO FALHA QUANDO O LOG-ODDS DELE É MAIOR QUE O LIMIAR, O MESMO > 0,5 DA MATRIZ; threshold É O LIMIAR EM PROBABILIDADE
    def curve(self):
        tp    = np.append(self.positives.flip(0).cumsum(0).flip(0).numpy(), 0)
        fp    = np.append(self.negatives.flip(0).cumsum(0).flip(0).numpy(), 0)
        logit = np.concatenate([[-np.inf], self.edges.double().numpy(), [np.inf]])
        return {'threshold': 1 / (1 + np.exp(-logit)), 'tpr': tp / max(tp[0], 1), 'fpr': fp / max(fp[0], 1), 'tp': tp, 'fp': fp}

    # ÁREA PELO TRAPÉZIO ENTRE LIMIARES VIZINHOS: VOXELS DA MESMA FAIXA CONTAM COMO EMPATE, A CONVENÇÃO USUAL DA ROC
    @property
    def auc(self):
        if not self.defined:
            return float('nan')

        curve    = self.curve()
        fpr, tpr = curve['fpr'][::-1], curve['tpr'][::-1]
        return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

    # PONTO DA CURVA NO LIMIAR DE PROBABILIDADE MAIS PRÓXIMO DE p
    def at(self, p=0.5):
        curve = self.curve()
        index = int(np.argmin(np.abs(curve['threshold'] - p)))
        return {key: float(curve[key][index]) for key in ('threshold', 'tpr', 'fpr')}

    # LIMIAR QUE MAXIMIZA O ÍNDICE DE YOUDEN (J = TPR - FPR)
    def youden(self):
        if not self.defined:
            return dict.fromkeys(('threshold', 'tpr', 'fpr', 'j'), float('nan'))

        curve = self.curve()
        index = int(np.argmax(curve['tpr'] - curve['fpr']))
        return {'threshold': float(curve['threshold'][index]), 'tpr': float(curve['tpr'][index]), 'fpr': float(curve['fpr'][index]), 'j': float(curve['tpr'][index] - curve['fpr'][index])}

    def summary(self):
        point, best = self.at(0.5), self.youden()
        return {'auc': self.auc, 'tpr_0.5': point['tpr'], 'fpr_0.5': point['fpr'], 'youden_threshold': best['threshold'], 'youden_tpr': best['tpr'], 'youden_fpr': best['fpr'], 'youden_j': best['j']}

    # --- A FIGURA ---

    # A CURVA, O ● DO LIMIAR 0,5 E, SE youden, A ★ DO MELHOR J; SEM RÓTULOS: A LEGENDA É MONTADA POR QUEM CHAMA
    def draw(self, ax, color, youden=True, fill=False, linestyle='-'):
        curve = self.curve()
        if fill:
            ax.fill_between(curve['fpr'], curve['tpr'], color=color, alpha=0.08, linewidth=0)

        ax.plot(curve['fpr'], curve['tpr'], color=color, linewidth=2, linestyle=linestyle, solid_capstyle='round', solid_joinstyle='round')

        point = self.at(0.5)
        ax.plot([point['fpr']], [point['tpr']], markerfacecolor=color, **self.MARKS['threshold'])

        if youden and self.defined:
            best = self.youden()
            ax.plot([best['fpr']], [best['tpr']], **self.MARKS['youden'])

    # EIXOS: A DIAGONAL DO ACASO SEM LEGENDA, GRADE FINA ATRÁS DE TUDO, TEXTO EM CINZA E SÓ AS BORDAS DE BAIXO E DA ESQUERDA
    @classmethod
    def style(cls, ax):
        ax.plot([0, 1], [0, 1], color=cls.MUTED, linewidth=0.8, linestyle=(0, (4, 4)), alpha=0.6, zorder=0)
        ax.set_xlim(-0.01, 1), ax.set_ylim(0, 1.01)
        ax.set_aspect('equal')
        ax.grid(color=cls.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=cls.MUTED, labelsize=10, length=3)
        ax.set_xlabel('False Positive Rate', color=cls.INK, fontsize=11)
        ax.set_ylabel('True Positive Rate', color=cls.INK, fontsize=11)

        for side, spine in ax.spines.items():
            spine.set_color('#a3a29c')
            spine.set_visible(side in ('left', 'bottom'))

    # FIGURA DE UM MODELO: A CURVA COM A ÁREA PINTADA E A LEGENDA CURTA (AUC, ●, ★)
    def figure(self, title=None):
        fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor='white')
        color   = self.COLORS[0]

        self.draw(ax, color, fill=True)
        self.style(ax)

        handles = [Line2D([], [], color=color, linewidth=2, label=f'AUC = {self.auc:.4f}' if self.defined else 'AUC undefined (one class only)'), Line2D([], [], markerfacecolor=color, label='threshold 0.5', **self.MARKS['threshold'])]
        if self.defined:
            handles.append(Line2D([], [], label=f'threshold {self.youden()["threshold"]:.2g} (Youden)', **self.MARKS['youden']))

        ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=10, labelcolor=self.INK)
        ax.set_title(title, color=self.INK, fontsize=12, pad=12)
        fig.tight_layout()
        return fig

    def save(self, path, title=None):
        fig = self.figure(title)
        fig.savefig(path, bbox_inches='tight', dpi=200, facecolor='white')
        plt.close(fig)
        return path

    # CURVAS DE VÁRIOS MODELOS NOS MESMOS EIXOS, COM O ● DO LIMIAR 0,5 DE CADA UM; curves = {rótulo: ROC}. AS 8 CORES SEGUEM A
    # ORDEM DA PALETA E DO 9º MODELO EM DIANTE VOLTAM COM OUTRO TRAÇO; A LEGENDA FICA FORA DOS EIXOS PARA CABER QUALQUER NÚMERO
    @classmethod
    def compare(cls, curves, title=None):
        fig, ax    = plt.subplots(figsize=(11, 6.4), facecolor='white')
        colors     = [cls.COLORS[i % len(cls.COLORS)] for i in range(len(curves))]
        linestyles = [('-', '--', ':', '-.')[i // len(cls.COLORS) % 4] for i in range(len(curves))]

        for curve, color, linestyle in zip(curves.values(), colors, linestyles):
            curve.draw(ax, color, youden=False, linestyle=linestyle)

        cls.style(ax)

        handles = [Line2D([], [], color=color, linewidth=2, linestyle=linestyle, label=f'{label}  ·  AUC {curve.auc:.4f}') for (label, curve), color, linestyle in zip(curves.items(), colors, linestyles)]
        handles.append(Line2D([], [], markerfacecolor=cls.MUTED, label='threshold 0.5', **cls.MARKS['threshold']))
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=9, labelcolor=cls.INK)
        ax.set_title(title, color=cls.INK, fontsize=12, pad=12)
        fig.tight_layout()
        return fig
