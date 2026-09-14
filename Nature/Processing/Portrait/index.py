import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# DESENHA UM RUN: CONVERGÊNCIA, RELEVO E DISPERSÃO POR VARIÁVEL. CRIADA PELOS plot* DO MODELO, NUNCA PELO NOTEBOOK.
class Portrait:
    SPAN = 100    # razão entre o maior e o menor valor a partir da qual a métrica vai para escala log
    GRID = 120    # resolução do relevo do plotGraph (GRID² avaliações da objective)
    COLS = 3      # painéis por linha nas grades de variáveis
    BINS = 30     # bins de geração da linha de tendência temporal

    def __init__(self, problem, recorder, best=None, bestScore=None, xLabel='Geração', name=''):
        self.problem   = problem
        self.recorder  = recorder
        self.best      = best
        self.bestScore = bestScore
        self.xLabel    = xLabel
        self.name      = name
        self.rng       = np.random.default_rng(0)  # subamostragem do scatter, isolada da busca

    # QUEM CRIA O RETRATO É O MODELO, QUE NÃO SABE O PRÓPRIO NOME: SEM ELE O TÍTULO SAI SEM PARÊNTESES
    def tag(self, title):
        return f'{title} ({self.name})' if self.name else title

    # SEM OS PENALIZADOS: UM |f| >= WORST SOZINHO ACHATA TODOS OS PAINÉIS
    def samples(self):
        X, y, gen = self.recorder.cloud()
        if y is None:
            return None, None, None
        keep = np.abs(y) < self.problem.WORST
        return (X[keep], y[keep], gen[keep]) if keep.any() else (None, None, None)

    # AS DUAS VARIÁVEIS DE MAIOR |ρ| DE SPEARMAN CONTRA A MÉTRICA
    def impact(self, X, y):
        if X is None or self.problem.nVars <= 2:
            return list(range(self.problem.nVars))[:2]
        rank = [abs(self.spearman(X[:, i], y)) if X[:, i].max() > X[:, i].min() else 0.0
                for i in range(self.problem.nVars)]
        return list(np.argsort(rank)[::-1][:2])

    def spearman(self, x, y):
        return float(stats.spearmanr(x, y).statistic)

    def close(self, save):
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()

    def plotMetrics(self, save=None):
        plt.figure(figsize=(12, 4.5))
        self.metrics(plt.gca())
        self.close(save)

    # A FAIXA DA POPULAÇÃO E O MELHOR ACUMULADO, NO EIXO QUE O CHAMADOR ABRIU
    def metrics(self, ax):
        records  = self.recorder.records
        gen      = records['gen']
        maximize = self.problem.weight > 0
        lo, hi   = np.asarray(records['min'])[:, 0], np.asarray(records['max'])[:, 0]
        best     = np.maximum.accumulate(hi) if maximize else np.minimum.accumulate(lo)

        ax.fill_between(gen, lo, hi, color='steelblue', alpha=0.25, linewidth=0, label='Population')
        ax.plot(gen, best, color='crimson', linewidth=2.0, label='Current Best')
        span = np.concatenate([best, lo, hi])
        if span.min() > 0 and span.max() / span.min() > self.SPAN:
            ax.set_yscale('log')
        ax.set(title=f'{self.tag("Convergência")} — melhor {best[-1]:.6g} em {gen[-1]} {self.xLabel.lower()[:-2]}ões',
               xlabel=self.xLabel, ylabel='Métrica')
        ax.legend(); ax.grid(True, alpha=0.3, which='both')

    # O RELEVO EM LOG, CALCULADO UMA VEZ: O plotGraph E O Plotter DESENHAM O MESMO RESULTADO
    def landscape(self):
        X, y, _ = self.samples()
        ix, iy  = self.impact(X, y)
        gx, gy  = self.problem.genes[ix], self.problem.genes[iy]
        XX, YY  = np.meshgrid(np.linspace(gx['low'], gx['up'], self.GRID), np.linspace(gy['low'], gy['up'], self.GRID))
        # fatia pelo ótimo: as outras variáveis ficam na melhor solução, senão o corte é arbitrário
        genomes = np.tile(self.problem.encode(self.best), (XX.size, 1))
        genomes[:, ix] = XX.ravel()
        genomes[:, iy] = YY.ravel()
        raw   = self.problem.score(genomes, map)[:, 0]
        floor = min(float(raw.min()), float(self.bestScore))
        return XX, YY, self.shade(raw, floor).reshape(XX.shape), gx, gy, floor

    # SÓ EM 2D O RELEVO É UMA SUPERFÍCIE, E AÍ O EIXO PRECISA NASCER 3D
    def projection(self):
        return '3d' if self.problem.nVars == 2 else None

    # O PAINEL DE RELEVO MAIS INFORMATIVO PARA A DIMENSÃO DO PROBLEMA
    def shape(self, ax):
        if self.problem.nVars == 1:
            return self.curve(ax)
        XX, YY, Z, gx, gy, floor = self.landscape()
        if self.problem.nVars == 2:
            return self.surface(ax, XX, YY, Z, gx, gy, floor)
        self.heatmap(ax, XX, YY, Z, gx, gy)

    def plotGraph(self, save=None):
        if self.problem.nVars == 1:
            plt.figure(figsize=(9, 5))
            self.curve(plt.subplot(1, 1, 1))
            return self.close(save)

        XX, YY, Z, gx, gy, floor = self.landscape()
        wide = self.problem.nVars == 2
        fig  = plt.figure(figsize=(14, 5.5) if wide else (8, 6))
        grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.25)) if wide else fig.add_gridspec(1, 1)
        self.heatmap(fig.add_subplot(grid[0, 0]), XX, YY, Z, gx, gy)
        if wide:
            self.surface(fig.add_subplot(grid[0, 1], projection='3d'), XX, YY, Z, gx, gy, floor)
        self.close(save)

    def plotVariables(self, mode='range', save=None):
        if mode not in ('range', 'temporal'):
            raise ValueError(f"mode inválido '{mode}'; use 'range' ou 'temporal'.")
        X, y, gen = self.samples()
        rows      = max(1, -(-self.problem.nVars // self.COLS))
        draw      = self.range if mode == 'range' else self.temporal

        plt.figure(figsize=(14, 4 * rows))
        for i, g in enumerate(self.problem.genes):
            draw(plt.subplot(rows, self.COLS, i + 1), i, g, X, y, gen)
        self.close(save)

    # O RÓTULO E A COLUNA DA VARIÁVEL JÁ CORTADA NOS BOUNDS; SEM NUVEM A COLUNA VOLTA None
    def column(self, ax, i, g, X):
        ax.grid(True, alpha=0.3)
        bv    = self.best[g['name']] if self.best else '?'
        title = f"{g['name']}={bv:.4g}" if isinstance(bv, float) else f"{g['name']}={bv}"
        if X is None:
            return bv, title, None
        v = np.clip(X[:, i], g['low'], g['up'])
        return bv, title, v if g['type'] == 'real' else np.round(v)

    # ONDE A MÉTRICA MELHORA DENTRO DOS BOUNDS, COM A TENDÊNCIA E O ÓTIMO
    def range(self, ax, i, g, X, y, gen):
        bv, title, x = self.column(ax, i, g, X)
        if x is None:
            return ax.set(title=title, xlabel=g['name'], ylabel='Métrica')
        x, v = self.spread(x, y)   # cobertura ~uniforme: a massa convergida não domina a reta
        ax.scatter(x, v, c=v, cmap='viridis', s=22, alpha=0.55, edgecolors='none')
        if x.max() > x.min():
            coef = np.polyfit(x, v, 1)
            line = np.linspace(x.min(), x.max(), 100)
            good = (coef[0] > 0) == (self.problem.weight > 0)
            ax.plot(line, np.polyval(coef, line), '--', color='#2ca02c' if good else '#d62728', linewidth=2)
            title = f"{title}   ρ={self.spearman(x, v):+.2f}"
        self.frame(ax, v)
        self.mark(ax, g, bv, self.bestScore)
        self.ticks(ax, g)
        ax.set(title=title, xlabel=g['name'], ylabel='Métrica')

    # A MESMA VARIÁVEL AO LONGO DA BUSCA: SE CONVERGE, E EM QUE GERAÇÃO
    def temporal(self, ax, i, g, X, y, gen):
        bv, title, v = self.column(ax, i, g, X)
        if v is None:
            return ax.set(title=title, xlabel=self.xLabel, ylabel=g['name'])
        ax.scatter(gen, v, c=y, cmap='viridis', s=22, alpha=0.55, edgecolors='none')
        edges = np.linspace(gen.min(), gen.max() + 1e-9, self.BINS + 1)
        slot  = np.clip(np.digitize(gen, edges) - 1, 0, self.BINS - 1)
        mid   = [(np.median(v[slot == b]), (edges[b] + edges[b + 1]) / 2) for b in range(self.BINS) if (slot == b).any()]
        # mediana por bloco, não média: um outlier de exploração desloca a média
        ax.plot([t for _, t in mid], [m for m, _ in mid], color='crimson', linewidth=1.8, label='Mediana')
        if self.best:
            ax.axhline(self.spot(g, bv), color='red', linestyle='--', linewidth=1.2, label='Melhor')
        ax.set_ylim(g['low'], g['up'])
        self.ticks(ax, g, axis='y')
        ax.legend(fontsize=7)
        ax.set(title=title, xlabel=self.xLabel, ylabel=g['name'])

    # VAI PELO score E NÃO PELA objective CRUA: PASSA PELO decode E DESENHA O QUE O OTIMIZADOR ENXERGOU
    def curve(self, ax):
        g  = self.problem.genes[0]
        xs = np.linspace(g['low'], g['up'], 1000)
        ys = self.problem.score(xs.reshape(-1, 1), map)[:, 0]
        ax.plot(xs, ys, color='steelblue', label='f')
        if self.best:
            ax.scatter([self.spot(g, self.best[g['name']])], [self.bestScore], color='red', s=90,
                       edgecolors='black', linewidths=.8, zorder=5, label=f'Melhor: f={self.bestScore:.4g}')
            ax.legend(fontsize=8)
        self.ticks(ax, g)
        ax.set(title=self.tag('Função objetivo 1D'), xlabel=g['name'], ylabel='Métrica')
        ax.grid(True, alpha=0.3)

    def heatmap(self, ax, XX, YY, Z, gx, gy):
        ax.contourf(XX, YY, Z, levels=40, cmap='viridis')
        ax.contour(XX, YY, Z, levels=12, colors='white', linewidths=0.4, alpha=0.5)
        if self.best:
            ax.scatter([self.spot(gx, self.best[gx['name']])], [self.spot(gy, self.best[gy['name']])],
                       s=140, marker='*', color='red', edgecolors='black', linewidths=.8, zorder=5, label='Melhor')
            ax.legend(fontsize=8, loc='upper right')
        extra = '' if self.problem.nVars == 2 else ' — demais variáveis fixas no ótimo'
        ax.set(title=f'{self.tag("Relevo em log")}{extra}', xlabel=gx['name'], ylabel=gy['name'])
        ax.grid(True, alpha=0.3)

    def surface(self, ax, XX, YY, Z, gx, gy, floor):
        ax.plot_surface(XX, YY, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.85)
        if self.best:
            x, y = self.spot(gx, self.best[gx['name']]), self.spot(gy, self.best[gy['name']])
            z    = float(self.shade(self.bestScore, floor))
            ax.plot([x, x], [y, y], [z, Z.max()], color='red', linewidth=1.2, linestyle='--', zorder=10)
            ax.scatter([x], [y], [z], color='red', s=160, marker='*', depthshade=False, zorder=11, label='Melhor')
            ax.legend(fontsize=8, loc='upper left')
        ax.set(title=self.tag(f'f = {self.bestScore:.4g}'), xlabel=gx['name'], ylabel=gy['name'])

    # LOG DESLOCADO PELO PISO: SEM ISSO O RELEVO VIRA UM PICO NUM PLANO
    def shade(self, z, floor):
        return np.log10(np.asarray(z, float) - floor + 1.0)

    def spot(self, g, value):
        return g['values'].index(value) if g['type'] == 'cat' else float(value)

    # CORTA A CAUDA ESPARSA DO LADO DO ÓTIMO E MANTÉM INTEIRA A NUVEM DE EXPLORAÇÃO DO OUTRO
    def frame(self, ax, v):
        lo = np.percentile(v, 5) if self.problem.weight < 0 else v.min()
        hi = v.max() if self.problem.weight < 0 else np.percentile(v, 95)
        if self.bestScore is not None:
            lo, hi = min(lo, self.bestScore), max(hi, self.bestScore)  # o corte por densidade cairia sobre o ótimo
        if hi > lo:
            pad = 0.05 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

    # SEM LEGENDA: O MARCADOR DELA CAI DENTRO DO PAINEL E SE CONFUNDE COM UM SEGUNDO ÓTIMO
    def mark(self, ax, g, bv, score):
        if self.best is not None and score is not None:
            ax.scatter([self.spot(g, bv)], [score], s=75, color='red', edgecolors='black', linewidths=.8, zorder=6)

    def ticks(self, ax, g, axis='x'):
        setTicks  = ax.set_xticks if axis == 'x' else ax.set_yticks
        setLabels = ax.set_xticklabels if axis == 'x' else ax.set_yticklabels
        if g['type'] == 'cat':
            setTicks(range(len(g['values']))); setLabels([str(v) for v in g['values']], fontsize=7)
        if g['type'] == 'bool':
            setTicks([0, 1]); setLabels(['False', 'True'])

    # A MAIORIA DAS AVALIAÇÕES CAI PERTO DO ÓTIMO: SEM SUBAMOSTRAR, A NUVEM VIRA UMA BOLA SOBRE ELE
    def spread(self, x, y, bins=40, per=20):
        uniq  = np.unique(np.round(np.column_stack([x, y]), 6), axis=0)   # colapsa a pilha convergida
        x, y  = uniq[:, 0], uniq[:, 1]
        if x.max() <= x.min():
            return x, y
        edges = np.linspace(x.min(), x.max(), bins + 1)
        slot  = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
        keep  = []
        for b in range(bins):
            w = np.where(slot == b)[0]
            keep.append(w if len(w) <= per else self.rng.choice(w, per, replace=False))
        k = np.concatenate(keep)
        return x[k], y[k]
