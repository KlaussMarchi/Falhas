# RÉGUA DE PERFIL (EXPLORAÇÃO DA FASE B): DESCRITORES DE UM TILE 128³, O MESMO CÓDIGO NO REAL E NO SINTÉTICO, NA ESCALA
# QUE A REDE VÊ ([0, 1]: O p01/p99 DO BLOCO NO REAL, O MIN-MAX DO VOLUME NO SINTÉTICO 'normalize')
import os, re, json, cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)


def gauss(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


# DESCRITORES DE UM TILE PELAS SEÇÕES (z, lateral): COMPOSIÇÃO, CONTRASTE, PERÍODO, MERGULHO, CONTINUIDADE E RUÍDO FINO
class TileRuler:
    ALIVE     = 0.0552                  # ENVELOPE RMS DO REFLETOR VIVO NA ESCALA [0, 1] (OTSU DO MARLIM)
    WATER_TOL = 0.002                   # |x - ÁGUA| QUE AINDA É ÁGUA
    DEAD_TOL  = 0.02                    # TRAÇO QUE NUNCA SAI DISSO É TRAÇO MORTO
    SLICES    = (32, 64, 96)            # SEÇÕES INLINE (EIXO 0) E CROSSLINE (EIXO 2) MEDIDAS
    STEEP     = (40, 50, 60, 70, 80)    # MERGULHOS DE LINEAMENTO TIPO FALHA, EM GRAUS
    FLAT      = (0, 10, 20)             # MERGULHOS DE LINEAMENTO AO LONGO DA CAMADA
    LINE      = 31                      # COMPRIMENTO DO NÚCLEO DE LINEAMENTO EM PX

    def __init__(self, water=None):
        self.water   = water
        self.kernels = {angle: self.getKernel(angle) for angle in self.STEEP + self.FLAT}

    def get(self, tile):
        tile  = np.asarray(tile, np.float32)
        rows  = [self.getSection(tile[i]) for i in self.SLICES] + [self.getSection(tile[:, :, j].T) for j in self.SLICES]
        table = pd.DataFrame(rows)
        out   = table.mean(numeric_only=True).to_dict()
        out.update({f'{k}_il': v for k, v in table.iloc[:3].mean(numeric_only=True).items() if k in ('steep90', 'dipP50', 'dipP90', 'disc5')})
        out.update({f'{k}_xl': v for k, v in table.iloc[3:].mean(numeric_only=True).items() if k in ('steep90', 'dipP50', 'dipP90', 'disc5')})
        out.update(self.getVolume(tile))
        return out

    # O QUE SÓ FAZ SENTIDO NO VOLUME: CONTRASTE, SATURAÇÃO, TRAÇOS MORTOS E O RUÍDO FINO DA SUB-BANDA HHH DE HAAR
    def getVolume(self, tile):
        center = self.water if self.water is not None else float(np.median(tile))
        a      = tile - center
        dead   = (np.abs(a) < self.DEAD_TOL).all(axis=1)
        t      = tile[:tile.shape[0] // 2 * 2, :tile.shape[1] // 2 * 2, :tile.shape[2] // 2 * 2]
        signs  = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        hhh    = sum((-1) ** (i + j + k) * t[i::2, j::2, k::2] for i, j, k in signs) / np.sqrt(8)
        rock   = np.abs(a[::2, ::2, ::2]) > self.WATER_TOL
        sd     = float(a[np.abs(a) > self.WATER_TOL].std()) if (np.abs(a) > self.WATER_TOL).any() else 1e-6
        fine   = float(np.median(np.abs(hhh[rock])) / 0.6745 / (sd + 1e-9)) if rock.any() else np.nan
        return {
            'sigma': float(tile.std()),
            'sat': float(((tile <= 0.002) | (tile >= 0.998)).mean()),
            'deadTraces': float(dead.mean()),
            'fine': fine
        }

    def getSection(self, section):
        x      = section.astype(np.float32)
        center = self.water if self.water is not None else float(np.median(x))
        a      = x - center
        water  = np.abs(a) < self.WATER_TOL
        env    = np.sqrt(gauss(a * a, 12))
        live   = env > self.ALIVE
        out    = {'water': float(water.mean()), 'live': float(live.mean())}

        if live.mean() < 0.05:
            return out

        zs      = a / (float(a[live].std()) + 1e-8)
        columns = live.mean(0) > 0.5
        out.update(self.getSpectrum(zs, columns))
        out.update(self.getStructure(zs, live))
        out.update(self.getContinuity(zs, live, columns))
        out.update(self.getContrast(a, live))
        return out

    # PERÍODO VERTICAL PELO CENTRÓIDE DO ESPECTRO EM z NAS COLUNAS VIVAS, E A LARGURA RELATIVA DA BANDA
    def getSpectrum(self, zs, columns):
        if columns.sum() < 8:
            return {}

        vert  = zs[:, columns] - zs[:, columns].mean(0)
        power = (np.abs(np.fft.rfft(vert, axis=0)) ** 2).mean(1)[1:]
        freqs = np.fft.rfftfreq(vert.shape[0])[1:]
        mean  = float((freqs * power).sum() / (power.sum() + 1e-12))
        width = float(np.sqrt(((freqs - mean) ** 2 * power).sum() / (power.sum() + 1e-12)))
        return {'period': 1.0 / (mean + 1e-12), 'bandwidth': width / (mean + 1e-12)}

    # TENSOR DE ESTRUTURA: COERÊNCIA E MERGULHO DOS REFLETORES, PONDERADOS PELA ENERGIA DO GRADIENTE
    def getStructure(self, zs, live):
        smooth = gauss(zs, 1.0)
        gz     = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        gx     = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        Jzz, Jxx, Jzx = gauss(gz * gz, 4.0), gauss(gx * gx, 4.0), gauss(gz * gx, 4.0)
        trace  = (Jzz + Jxx) * live
        coh    = np.sqrt(np.maximum((Jzz - Jxx) ** 2 + 4 * Jzx ** 2, 0)) / (Jzz + Jxx + 1e-12)
        dip    = np.abs(np.degrees(0.5 * np.arctan2(2 * Jzx, Jzz - Jxx)))
        order  = np.argsort(dip.ravel())
        cum    = np.cumsum(trace.ravel()[order]) / (trace.sum() + 1e-12)
        pick   = lambda q: float(dip.ravel()[order][min(np.searchsorted(cum, q), cum.size - 1)])
        return {'coherence': float((coh * trace).sum() / (trace.sum() + 1e-12)), 'dipP50': pick(0.5), 'dipP90': pick(0.9)}

    # SEMBLANCE SEGUINDO O MERGULHO EM 5 E 17 TRAÇOS, CORRELAÇÃO LATERAL E LINEAMENTOS DE DESCONTINUIDADE ÍNGREMES × DEITADOS
    def getContinuity(self, zs, live, columns):
        s5, s17 = self.semblance(zs, 2), self.semblance(zs, 8)
        disc    = np.clip(1 - s5, 0, 1).astype(np.float32)
        steep   = np.max([cv2.filter2D(disc, -1, self.kernels[a], borderType=cv2.BORDER_REFLECT) for a in self.STEEP], axis=0)
        flat    = np.max([cv2.filter2D(disc, -1, self.kernels[a], borderType=cv2.BORDER_REFLECT) for a in self.FLAT], axis=0)
        col     = zs[:, columns] / (zs[:, columns].std(0, keepdims=True) + 1e-8) if columns.sum() > 20 else None
        return {
            'disc5': float((s5 < 0.8)[live].mean()),
            'disc17': float((s17 < 0.8)[live].mean()),
            'steep90': float(np.percentile(steep[live], 90)),
            'flat90': float(np.percentile(flat[live], 90)),
            'lag4': float((col[:, :-4] * col[:, 4:]).mean()) if col is not None else np.nan,
            'lag16': float((col[:, :-16] * col[:, 16:]).mean()) if col is not None else np.nan
        }

    # FORMA DA DISTRIBUIÇÃO E VARIAÇÃO DO CONTRASTE DENTRO DO TILE (TOPO × BASE)
    def getContrast(self, a, live):
        vals  = a[live]
        zs    = (vals - vals.mean()) / (vals.std() + 1e-8)
        third = a.shape[0] // 3
        top   = a[:third][live[:third]]
        base  = a[-third:][live[-third:]]
        trend = float(np.log((top.std() + 1e-6) / (base.std() + 1e-6))) if top.size > 100 and base.size > 100 else np.nan
        return {'kurt': float((zs ** 4).mean()), 'trend': trend, 'sigmaLive': float(vals.std())}

    # NÚCLEO DE LINHA DE LINE px NO MERGULHO DADO (0 = HORIZONTAL), NORMALIZADO PARA MÉDIA
    def getKernel(self, angle):
        size   = self.LINE + (self.LINE + 1) % 2
        kernel = np.zeros((size, size), np.float32)
        c, r   = size // 2, self.LINE / 2
        dz, dx = np.sin(np.radians(angle)), np.cos(np.radians(angle))
        cv2.line(kernel, (int(round(c - r * dx)), int(round(c - r * dz))), (int(round(c + r * dx)), int(round(c + r * dz))), 1.0, 1)
        return kernel / kernel.sum()

    # SEMBLANCE DE 2·half+1 TRAÇOS SEGUINDO O MERGULHO LOCAL (A MESMA DO Regions/Marlim, JANELA DE 9 AMOSTRAS)
    @staticmethod
    def semblance(sec, half=2, win=4):
        smooth = gauss(sec, 1.0)
        gz     = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        gx     = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        Jzz, Jxx, Jzx = gauss(gz * gz, 4.0), gauss(gx * gx, 4.0), gauss(gz * gx, 4.0)
        dip    = np.clip(-np.tan(0.5 * np.arctan2(2 * Jzx, Jzz - Jxx)), -3, 3)
        zz, xx = np.mgrid[:sec.shape[0], :sec.shape[1]].astype(np.float32)
        traces = [cv2.remap(sec, xx + k, zz + dip * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for k in range(-half, half + 1)]
        window = np.ones((2 * win + 1, 1), np.float32)
        num    = cv2.filter2D(sum(traces) ** 2, -1, window, borderType=cv2.BORDER_REFLECT)
        den    = cv2.filter2D(sum(t * t for t in traces) * len(traces), -1, window, borderType=cv2.BORDER_REFLECT)
        return num / (den + 1e-12)


# AMOSTRAS SEM RÓTULO DE UM BLOCO: OS TILES DO patch_metadata.json, COM A POSIÇÃO DO NÚCLEO DE CADA UM
class BlockSampler:
    def __init__(self, folder, meta=None):
        self.folder = folder
        info        = json.load(open(meta or f'{folder}/patch_metadata.json'))
        self.size   = info['patch_size']
        self.ov     = info['overlap_voxels']
        self.shape  = tuple(info.get('full_shape', info['original_shape']))
        self.counts = tuple(info['grid_shape'])
        self.stride = self.size - 2 * self.ov

    def names(self):
        return [f'patch_d{d:03d}_h{h:03d}_w{w:03d}.dat' for d, h, w in np.ndindex(*self.counts)]

    # ORIGEM DO NÚCLEO DO TILE NO VOLUME (O ÚLTIMO DE CADA EIXO ENCOSTA NO FIM), COMO NO TileGrid
    def core(self, name):
        index = list(map(int, re.match(r'patch_d(\d+)_h(\d+)_w(\d+)\.dat', name).groups()))
        last  = [n // self.stride for n in self.shape]
        return [max(0, n - self.stride) if i == l else i * self.stride for i, l, n in zip(index, last, self.shape)]

    def get(self, name):
        return np.fromfile(os.path.join(self.folder, name), np.float32).reshape((self.size,) * 3)
