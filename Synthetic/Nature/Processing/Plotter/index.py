import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from ..Gaussian.index import Gaussian


# COMPARA OS ALGORITMOS DE UM MESMO PROBLEMA A PARTIR DO DataFrame QUE A CÉLULA MONTA: UM FIGURE COMPARATIVO (BARRAS + RELEVO DO VENCEDOR)
class Plotter:
    FLOOR  = 1e-8   # limiar de zero do CEC'14 (seção 2.1): abaixo disso conta como ter achado o ótimo
    SPAN   = 100    # razão entre a maior e a menor barra a partir da qual o eixo vira log
    PANELS = {'erro': 'melhor corrida', 'mediana': 'mediana das corridas'}

    def __init__(self, board, value='erro', label='algorithm', title='Comparativo', maximize=False, target=None):
        self.board  = board
        self.target = target
        # sem as colunas do compare() sobra a aptidão crua, e a tolerância da competição não se aplica
        self.value  = value if value in board else 'f'
        self.label  = label
        self.title  = title
        self.error  = self.value != 'f'
        self.up     = maximize and not self.error

    # O QUADRO DA CAMPANHA: AS CORRIDAS DE CADA ALGORITMO VIRAM ERRO AO ÓTIMO, ESTATÍSTICAS
    @staticmethod
    def compare(runs, target, ref=None):
        erros = pd.DataFrame({run.name: np.abs(np.array(run.scores) - target) for run in runs})
        erros[erros < Plotter.FLOOR] = 0.0   # abaixo da tolerância o erro conta como zero (CEC'14, seção 2.1)
        board = pd.DataFrame({'erro': erros.min(), 'mediana': erros.median(), 'media': erros.mean(), 'desvio': erros.std(), 'pior': erros.max(), 'acertos': (erros == 0).sum()})
        if ref is not None:
            board['p'] = [1.0 if name == ref else mannwhitneyu(erros[name], erros[ref]).pvalue for name in erros]
        return board.rename_axis('algorithm').reset_index()

    # AS COLUNAS DE BARRA DO QUADRO: A MELHOR DE n CORRIDAS E A MEDIANA DAS n DIZEM COISAS DIFERENTES E
    # CHEGAM A DISCORDAR DE VENCEDOR — SEM O QUADRO CRU DO info(), ONDE SÓ EXISTE A APTIDÃO
    def columns(self):
        return [v for v in self.PANELS if v in self.board] or [self.value]

    # O QUADRO (BARRAS + RELEVO), A CONVERGÊNCIA E, POR FIM, A AMOSTRA DE CORRIDAS DO VENCEDOR
    def plot(self, best=None, runs=None, save=None):
        portrait = best.portrait() if best is not None else None
        columns  = self.columns()
        fig      = plt.figure(figsize=(5.8 * (len(columns) + (portrait is not None)), 4.8))
        grid     = fig.add_gridspec(1, len(columns) + (portrait is not None))

        for k, value in enumerate(columns):
            self.bars(fig.add_subplot(grid[0, k]), value)
        if portrait:
            portrait.shape(fig.add_subplot(grid[0, len(columns)], projection=portrait.projection()))
        fig.suptitle(self.title, fontweight='bold')
        self.close(self.path(save, 'board'))

        self.evolution(portrait, runs, save)
        if portrait is not None and len(best.scores) > 1:
            self.spread(best, save)

    # OS ALGORITMOS NO MESMO EIXO E, LOGO ABAIXO, A MELHOR CORRIDA DO VENCEDOR — SÃO DUAS LEITURAS DA MESMA
    # BUSCA (ERRO CONTRA AVALIAÇÕES, FAIXA DA POPULAÇÃO CONTRA GERAÇÃO) E FICAM NA MESMA FIGURA, EMPILHADAS
    def evolution(self, portrait, runs, save):
        rows = bool(runs) + (portrait is not None)
        if not rows:
            return
        fig  = plt.figure(figsize=(11.6, 4.2 * rows))
        grid = fig.add_gridspec(rows, 1)

        if runs:
            self.curves(fig.add_subplot(grid[0, 0]), runs)
        if portrait is not None:
            portrait.metrics(fig.add_subplot(grid[rows - 1, 0]))
        self.close(self.path(save, 'curves'))

    # CONVERGÊNCIA DOS ALGORITMOS NO MESMO EIXO: A CORRIDA VENCEDORA DE CADA UM, ERRO CONTRA AVALIAÇÕES
    def curves(self, ax, runs):
        for run in runs:
            rec  = run.optimizer.recorder.records
            walk = np.asarray(rec['max' if self.up else 'min'], float).ravel()
            walk = (np.maximum if self.up else np.minimum).accumulate(walk)
            ax.plot(np.cumsum(rec['nevals']), np.maximum(np.abs(walk - self.target), self.FLOOR)
                    if self.target is not None else walk, linewidth=1.4, label=str(run.name))
        if self.target is not None:
            ax.set_yscale('log')
            ax.axhline(self.FLOOR, color='crimson', linestyle='--', linewidth=1.2)
        ax.set(xlabel='Avaliações', ylabel='Erro ao ótimo' if self.target is not None else 'Aptidão',
               title=f'{self.title} — melhor corrida de cada algoritmo')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)

    # A AMOSTRA DAS n CORRIDAS DO VENCEDOR, SOZINHA NA FIGURA: SINO E Q-Q LADO A LADO
    def spread(self, best, save):
        gauss = Gaussian(best.scores, self.target, self.up, best.name)
        fig   = plt.figure(figsize=(13.5, 4.6))
        grid  = fig.add_gridspec(1, 2)

        gauss.bell(fig.add_subplot(grid[0, 0]))
        gauss.qq(fig.add_subplot(grid[0, 1]))
        self.close(self.path(save, 'runs'))

    # CADA PAINEL ORDENA PELA PRÓPRIA COLUNA, MAS A COR SEGUE O ALGORITMO: SEM ISSO O MESMO NOME TROCA DE
    # COR ENTRE O PAINEL DO MELHOR E O DA MEDIANA, QUE É JUSTAMENTE ONDE OS DOIS DISCORDAM DE VENCEDOR
    def bars(self, ax, value=None):
        value  = value or self.value
        board  = self.board.sort_values(value, ascending=not self.up)  # do melhor para o pior
        tone   = {str(n): k for k, n in enumerate(self.board[self.label])}
        names  = [str(n) for n in board[self.label]]
        raw    = board[value].to_numpy(float)
        height = np.maximum(raw, self.FLOOR) if self.error else raw
        hit    = int((raw <= self.FLOOR).sum())
        bars   = ax.bar(names, height, color=[plt.cm.tab10(tone[n] % 10) for n in names])

        if height.min() > 0 and height.max() / height.min() > self.SPAN:
            ax.set_yscale('log')   # o erro varia ordens de grandeza entre algoritmos
        if self.error:
            ax.axhline(self.FLOOR, color='crimson', linestyle='--', linewidth=1.2, label=f'tolerância {self.FLOOR:g}')
            ax.legend(fontsize=8)
        for bar, v in zip(bars, raw):
            text = '≈0' if self.error and v <= self.FLOOR else f'{v:.3g}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha='center', va='bottom', fontsize=9, fontweight='bold')

        head = self.PANELS.get(value, 'Aptidão por algoritmo')
        ax.set(title=f'{head} — {hit} de {len(raw)} no ótimo' if self.error else head,
               ylabel='Erro ao ótimo' if self.error else 'Aptidão')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelrotation=30)

    def close(self, save):
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()

    # UM save, DOIS ARQUIVOS: O SUFIXO ENTRA ANTES DA EXTENSÃO
    def path(self, save, tag):
        if not save:
            return None
        root, dot, ext = save.rpartition('.')
        return f'{root}-{tag}.{ext}' if dot else f'{save}-{tag}.png'
