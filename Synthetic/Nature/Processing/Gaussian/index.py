import numpy as np
from scipy import stats


# A AMOSTRA DAS n CORRIDAS DE UM ALGORITMO: HISTOGRAMA COM NORMAL AJUSTADA E IC DE 95% POR t DE STUDENT,
# MAIS O Q-Q PLOT COM SHAPIRO-WILK. COM target A MÉTRICA VIRA ERRO E A ANÁLISE INTEIRA ACONTECE EM log10.
class Gaussian:
    ALPHA  = 0.05      # nível dos dois testes: t bilateral de 95% e normalidade rejeitada com p <= 0.05
    FLOOR  = 1e-8      # limiar de zero do CEC'14 (seção 2.1): piso do log, senão a corrida que achou o ótimo vira -infinito
    SIGMAS = 2.2       # desvios que a janela do painel cobre: o sino inteiro sem sobra de painel vazio
    DECADE = 2.0       # meia-janela quando a amostra não tem dispersão nenhuma (décadas no erro)
    BINS   = (5, 12)   # piso e teto de bins: abaixo de 5 não sobra forma para ler, acima de 12 cada bin vira ruído
    TINY   = 1e-12     # dispersão relativa abaixo disso é ruído de float: as corridas caíram todas no mesmo ponto

    def __init__(self, scores, target=None, up=False, name=''):
        self.name = name
        self.up   = up
        data      = np.asarray(scores, float)
        self.data = np.abs(data - target) if target is not None else data
        self.log  = target is not None
        self.u    = np.log10(np.maximum(self.data, self.FLOOR)) if self.log else self.data

    # t CRÍTICO BILATERAL DE 95% PARA n-1 GRAUS DE LIBERDADE
    def get(self, n):
        return float(stats.t.ppf(1.0 - self.ALPHA / 2.0, n - 1))

    # SEM DISPERSÃO NÃO HÁ JANELA, BINS, NORMAL NEM TESTE DE NORMALIDADE PARA TIRAR DAÍ
    def flat(self):
        lo, hi = float(self.u.min()), float(self.u.max())
        return hi - lo <= self.TINY * max(abs(lo), abs(hi), 1.0)

    # mean/median/std/best/worst NA ESCALA ORIGINAL; mu/sigma/lo/hi NA ESCALA DA ANÁLISE, COM STUDENT
    def info(self):
        n     = len(self.u)
        mu    = float(self.u.mean())
        sigma = float(self.u.std(ddof=1)) if n > 1 and not self.flat() else 0.0
        t     = self.get(n)
        half  = float(t * sigma / np.sqrt(n))
        # com n < 3 o W é 1 por construção e sem dispersão não há desvio da reta: o Shapiro nem é chamado
        w, p  = (1.0, 1.0) if n < 3 or self.flat() else stats.shapiro(self.u)

        best, worst = (self.data.max(), self.data.min()) if self.up else (self.data.min(), self.data.max())
        return {
            'mean': float(self.data.mean()), 'median': float(np.median(self.data)),
            'std': float(self.data.std(ddof=1)) if n > 1 else 0.0,
            'best': float(best), 'worst': float(worst),
            'mu': mu, 'sigma': sigma, 't': t, 'student': half, 'lo': mu - half, 'hi': mu + half,
            'w': float(w), 'p': float(p), 'normal': p > self.ALPHA
        }

    # OS BINS, CADA CORRIDA COMO UM PONTO DENTRO DA SUA BARRA, A NORMAL AJUSTADA E O IC DE 95%
    def bell(self, ax):
        res         = self.info()
        n           = len(self.u)
        mu, sigma   = res['mu'], res['sigma']
        left, right = self.window(mu, sigma)

        # sem dispersão sai uma barra só, de meia unidade para cada lado
        bins         = 1 if self.flat() else int(min(np.clip(np.ceil(2 * n ** (1 / 3)), *self.BINS), n))  # regra de Rice
        span         = (mu - 0.5, mu + 0.5) if self.flat() else (float(self.u.min()), float(self.u.max()))
        count, edges = np.histogram(self.u, bins=bins, range=span)
        dens         = count / (n * (edges[1] - edges[0]))

        x   = np.linspace(left, right, 400)
        y   = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)) if sigma > 0 else None
        top = max(float(dens.max()), 0.0 if y is None else float(y.max()))

        ax.bar(self.spot(edges[:-1]), dens, width=self.spot(edges[1:]) - self.spot(edges[:-1]), align='edge',
               color='steelblue', alpha=0.40, edgecolor='#1f3b4d', linewidth=0.9, label=f'{bins} bins')

        # cada corrida empilhada dentro da própria barra: a barra é a pilha de pontos
        slot = np.clip(np.digitize(self.u, edges) - 1, 0, bins - 1)
        rank = np.zeros(n)
        for b in np.unique(slot):
            rank[slot == b] = np.arange(count[b])
        ax.scatter(self.spot(self.u), dens[slot] * (rank + 0.5) / count[slot], s=18, color='#0d2b3a', zorder=5, label=f'{n} corridas')

        if y is not None:
            ax.plot(self.spot(x), y, color='#1f3b4d', linewidth=1.8, label='Normal ajustada')

        # em log a média volta como geométrica, que não é o µ da caixa: o rótulo diz qual é a linha
        ax.axvline(self.spot(mu), color='crimson', linestyle='--', linewidth=1.5, label='Média geométrica' if self.log else 'Média')
        ax.axvspan(self.spot(res['lo']), self.spot(res['hi']), color='crimson', alpha=0.12)
        for edge in (res['lo'], res['hi']):
            ax.axvline(self.spot(edge), color='crimson', linestyle=':', linewidth=1.4)
        ax.plot([], [], color='crimson', linestyle=':', linewidth=1.4, label='$\\pm t_{crit}$ (95%)')

        if self.log:
            ax.set_xscale('log')
        ax.set_xlim(self.spot(left), self.spot(right))
        ax.set_ylim(0, top * 1.45)

        ax.text(0.03, 0.97, self.summary(res), transform=ax.transAxes, verticalalignment='top', fontsize=8.5,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.legend(fontsize=7, loc='upper right')
        ax.set(title=self.tag('Distribuição das corridas'), xlabel='Erro ao ótimo' if self.log else 'Aptidão', ylabel='Densidade')
        ax.grid(True, alpha=0.3)

    # OS PONTOS SOBRE A RETA DA NORMAL AJUSTADA E O VEREDITO DE SHAPIRO
    def qq(self, ax):
        res  = self.info()
        q, v = stats.probplot(self.u, dist='norm', fit=False)
        # a reta sai dos pontos ordenados, não de µ e σ: uma corrida fora da curva entortaria a referência
        coef = [0.0, res['mu']] if self.flat() else np.polyfit(q, v, 1)

        ax.plot(q, np.polyval(coef, q), color='crimson', linewidth=1.5, label='Normal ajustada')
        ax.scatter(q, v, s=30, color='steelblue', edgecolors='#1f3b4d', linewidths=0.6, zorder=4, label='Corridas')
        ax.legend(fontsize=7, loc='upper left')
        ax.set(title=self.verdict(res), xlabel='Quantis teóricos',
               ylabel=f"Quantis da amostra{' em log' if self.log else ''}")
        ax.grid(True, alpha=0.3)

    # O SINO INTEIRO SEM CORTAR CORRIDA, SEM DESCER ABAIXO DO PISO DO LOG NEM ABRIR DÉCADA VAZIA
    def window(self, mu, sigma):
        lo, hi = float(self.u.min()), float(self.u.max())
        half   = max(self.SIGMAS * sigma, hi - mu, mu - lo) * 1.15
        if self.flat():
            half = self.DECADE if self.log else max(abs(mu) * 0.5, 1.0)
        left, right = mu - half, mu + half
        if self.log:
            left, right = max(left, np.log10(self.FLOOR) - 1.5), min(right, hi + 1.0)
        return left, right

    def summary(self, res):
        lo, hi = float(self.spot(res['lo'])), float(self.spot(res['hi']))
        return (f"µ = {res['mean']:.4g}\nmediana = {res['median']:.4g}\nσ = {res['std']:.3g}\n"
                f"best = {res['best']:.4g}\nworst = {res['worst']:.4g}\n"
                f"IC 95% = [{lo:.3g}, {hi:.3g}]")

    # AMOSTRA SEM DISPERSÃO NÃO É NORMAL, É DEGENERADA: O W = 1 VEM DO flat(), NÃO DO SHAPIRO. E O log ACHATA
    # DOIS CASOS OPOSTOS NO MESMO -8 — TODAS NO ALVO E TODAS TRAVADAS —, QUE O VEREDITO PRECISA SEPARAR.
    def verdict(self, res):
        if self.flat() and self.log and float(self.data.max()) <= self.FLOOR:
            return f'as {len(self.u)} corridas atingiram o alvo (erro < {self.FLOOR:g})'
        if self.flat():
            return f'σ = 0 — as {len(self.u)} corridas pararam no mesmo ponto'
        status = 'NORMAL' if res['normal'] else 'NÃO NORMAL'
        return f"W = {res['w']:.4f} | p = {res['p']:.3f} → {status}"

    # A AMOSTRA É DAS CORRIDAS DE UM ALGORITMO SÓ, E O TÍTULO É QUEM DIZ DE QUAL
    def tag(self, title):
        return f'{title} ({self.name})' if self.name else title

    # A ESCALA DA ANÁLISE DE VOLTA PARA A ORIGINAL, QUE É ONDE OS EIXOS SÃO ROTULADOS
    def spot(self, v):
        return 10.0 ** v if self.log else v
