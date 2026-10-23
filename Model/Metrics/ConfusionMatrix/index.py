import numpy as np
import matplotlib.pyplot as plt


# MATRIZ DE CONFUSÃO BINÁRIA POR VOXEL: SOMA tp, fp, fn E tn DE TODOS OS LOTES ANTES DE DIVIDIR, A MESMA CONTA DO test_iou DO TREINO
class ConfusionMatrix:
    def __init__(self, counts=(0, 0, 0, 0)):
        self.tp, self.fp, self.fn, self.tn = (int(value) for value in counts)

    # pred E target SÃO TENSORES BOOLEANOS DO MESMO SHAPE: A PREDIÇÃO JÁ NO LIMIAR E O RÓTULO; SHAPES DIFERENTES SE PROPAGARIAM
    # (BROADCAST) E CONTARIAM VOXELS DE MAIS SEM ERRO, POR ISSO PARAM AQUI
    def update(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f'predição {tuple(pred.shape)} e rótulo {tuple(target.shape)} com shapes diferentes')

        self.tp += int((pred & target).sum())
        self.fp += int((pred & ~target).sum())
        self.fn += int((~pred & target).sum())
        self.tn += int((~pred & ~target).sum())
        return self

    @property
    def counts(self):
        return (self.tp, self.fp, self.fn, self.tn)

    def metrics(self):
        tp, fp, fn, tn = self.counts
        return {
            'accuracy':  (tp + tn) / max(tp + fp + fn + tn, 1),
            'precision': tp / max(tp + fp, 1),
            'recall':    tp / max(tp + fn, 1),
            'iou':       tp / max(tp + fp + fn, 1),
            'f1':        2 * tp / max(2 * tp + fp + fn, 1),
        }

    # LINHA = RÓTULO, COLUNA = PREDIÇÃO; A COR É A FRAÇÃO DA LINHA (O ACERTO DE CADA CLASSE) E O TEXTO TRAZ OS VOXELS
    def plot(self, ax, title=None):
        matrix = np.array([[self.tn, self.fp], [self.fn, self.tp]])
        rates  = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image  = ax.imshow(rates, cmap='Blues', vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{rates[i, j]:.1%}\n{matrix[i, j]:,} voxels', ha='center', va='center', fontsize=11, color='white' if rates[i, j] > 0.5 else 'black')

        ax.set_xticks([0, 1]), ax.set_yticks([0, 1])
        ax.set_xlabel('Prediction'), ax.set_ylabel('Ground Truth')
        ax.set_title(title)
        return image

    # UMA FIGURA POR MATRIZ, ATÉ 3 POR LINHA, COM UMA BARRA DE CORES SÓ; matrices = {título: ConfusionMatrix}
    @staticmethod
    def grid(matrices):
        cols      = min(len(matrices), 3)
        rows      = -(-len(matrices) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

        for ax in axes.flat[len(matrices):]:
            ax.axis('off')

        for ax, (title, matrix) in zip(axes.flat, matrices.items()):
            image = matrix.plot(ax, title)

        fig.colorbar(image, ax=axes.ravel().tolist())
        return fig

    # FIGURA DE UMA MATRIZ SÓ, GRAVADA EM path
    def save(self, path, title=None):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        fig.colorbar(self.plot(ax, title), ax=ax)
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        return path
