import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, metrics, limits=(0, 1.10), title='Metric Analysis'):
        keys = list(metrics.keys())
        values = list(metrics.values())
        n      = len(metrics)
        colors = plt.cm.tab10(np.arange(n)) if n <= 10 else plt.cm.viridis(np.linspace(0, 1, n))
        bars = plt.bar(keys, values, color=colors)
        
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.title(title)
        
        if limits:
            plt.ylim(limits)

        for bar in bars:
            height = bar.get_height()
            offset = (limits[1] * 0.01) if limits else (height * 0.01)

            plt.text(
                bar.get_x() + bar.get_width() / 2., # Posição X (centro da barra)
                height + offset,                    # Posição Y (topo da barra + respiro)
                f'{height:.3f}',                    # O texto (valor formatado)
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black'
            )

        if len(keys) >= 4:
            plt.xticks(rotation=45, ha='right')