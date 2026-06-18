import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndimage
import os, json


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.default_rng(seed=seed)
    print('random seed done')


class SyntheticGenerator:
    def __init__(self, shape=(128, 128, 128), seed=None):
        # ── Image Format ─────────────────────────────────────────────
        self.margin = 64                  # Buffer para absorver dobras extremas nas bordas com segurança
        self.finalShape = shape           # (nx, ny, nz) final output volume size

        # ── Refletividade (Estratigrafia) ────────────────────────────
        self.layerRange = (100, 230)      # Qtd de camadas. ↑ Imagem cheia de linhas finas. ↓ Blocos grossos e lisos.
        self.layerThickness = (1, 2)      # Espessura. ↑ Camadas mais grossas. ↓ Camadas bem fininhas.

        # ── Dobramentos (Folding) ────────────────────────────────────
        self.foldCount = (15, 30)         # Qtd de dobras. ↑ Imagem muito ondulada. ↓ Terreno plano.
        self.foldSigma = (8, 44)          # Largura da dobra. ↑ Dobras largas e suaves. ↓ Dobras curtas e apertadas.
        self.foldAmplitude = (-17, 17)    # Altura da dobra. ↑ Picos e vales extremos. ↓ Dobras rasas.
        self.foldDamping   = 1.5          # Perda de força. ↑ A dobra some rápido no fundo. ↓ A dobra desce até a base.
        self.foldBaseShift = (-1.6, 1.6)  # Posição Z. ↑/↓ Sobe ou desce o desenho inteiro na imagem.

        # ── Cisalhamento / Inclinação (Shearing) ─────────────────────
        self.shearOffset   = (-2.8, 2.8)  # Deslocamento lateral. ↑/↓ Empurra todo o bloco para o lado.
        self.shearGradient = (-0.1, 0.1)  # Inclinação (Mergulho). ↑ Camadas ficam na diagonal. ↓ Ficam na horizontal.

        # ── Falhas (Faulting) ────────────────────────────────────────
        self.faultCount = (4, 7)          # Qtd de falhas. ↑ Imagem toda fraturada. ↓ Imagem mais inteira.
        self.faultThrow = (0, 22)         # Tamanho do degrau. ↑ Desencontro gigante nas linhas. ↓ Quebra quase invisível.
        self.faultDipAngle = (20, 75)     # Ângulo. ↑ Falha quase em pé (vertical). ↓ Falha deitada.
        
        self.faultRoughness  = 3.3        # Textura do corte. ↑ Corte tremido/áspero. ↓ Corte liso como navalha.
        self.faultRoughSigma = 4.5        # Tamanho da tremedeira. ↑ Ondas grandes na falha. ↓ Ondinhas curtas.
        self.faultDecaySigma = (33, 83)   # Arrasto. ↑ A linha entorta muito antes de quebrar. ↓ Quebra seca.

        self.faultZoneWidth  = 1.2        # Espessura do rótulo. ↑ A máscara da falha fica grossa. ↓ Fica fina.
        self.faultThreshold  = 0.8        # Filtro de rótulo. ↑ Marca só falha grande. ↓ Marca qualquer rachadurazinha.

        self.faultCurveProb  = 0.30       # Chance de curvar. ↑ Falha faz formato de colher (lístrica). ↓ Falha reta.
        self.faultCurveMax   = 6.7        # Força da curva. ↑ Curva muito fechada. ↓ Curva leve.

        # ── Assinatura Sísmica (Wavelet) ─────────────────────────────
        self.waveletFreq = (81, 117)      # Resolução. ↑ Imagem super nítida. ↓ Imagem borrada e grossa.
        self.waveletDuration = 0.08       # "Eco" do sinal. ↑ O traço borra verticalmente. ↓ Sinal limpo e curto.
        self.waveletDt = 0.002            # Amostragem. ↑ Imagem pode ficar pixelada/serrilhada. ↓ Imagem contínua.

        # ── Ruído Final (Noise) ──────────────────────────────────────
        self.noiseLevel = (0.00, 0.10)    # Chuvisco. ↑ Imagem cheia de ruído (ruim). ↓ Imagem limpa (perfeita).

        self.nx = self.finalShape[0] + 2 * self.margin
        self.ny = self.finalShape[1] + 2 * self.margin
        self.nz = self.finalShape[2] + 2 * self.margin
        self.shape = (self.nx, self.ny, self.nz)
        self._ix = None
        self._iy = None
        self._iz = None

        if seed:
            seed_everything(seed)

    def _get_indices(self):
        if self._ix is None:
            self._ix, self._iy, self._iz = np.indices(self.shape, dtype=np.int32)
        return self._ix, self._iy, self._iz

    def _clear_indices(self):
        self._ix = None
        self._iy = None
        self._iz = None

    def get(self):
        self._clear_indices()

        r1d = self.genReflectivity()                # 1D — no 3D tile
        folded = self.applyFolding(r1d)             # samples 1D r1d directly
        del r1d
        sheared = self.applyShearing(folded)
        del folded
        faulted, mask = self.applyFaulting(sheared)
        del sheared
        image = self.applyWavelet(faulted)
        del faulted
        image = self.applyNoise(image)

        self._clear_indices()

        image = self.crop(image)
        mask  = self.crop(mask)
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        return image.astype(np.float32), mask.astype(np.uint8)

    def set(self, options):
        for k, v in options.items():
            setattr(self, k, v)

    def _generate_single(self, args):
        import numpy as np
        import os

        i, imgDir, mskDir, seed = args
        np.random.seed(seed)
        image, mask = self.get()
        image, mask = np.transpose(image, (0, 2, 1)), np.transpose(mask, (0, 2, 1))

        np.save(os.path.join(imgDir, f"img_{i:04d}.npy"), image)
        np.save(os.path.join(mskDir, f"img_{i:04d}.npy"), mask)

    def dataset(self, n=200, outputDir="output", n_jobs=None):
        from Utils.index import setFolder
        import concurrent.futures
        import multiprocessing
        import os
        from tqdm import tqdm

        imgDir = os.path.join(outputDir, "images")
        mskDir = os.path.join(outputDir, "masks")
        setFolder(imgDir)
        setFolder(mskDir)

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()

        base_seed = np.random.randint(0, 1000000)
        tasks = [(i, imgDir, mskDir, base_seed + i) for i in range(n)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            list(tqdm(executor.map(self._generate_single, tasks), total=n, desc="Generating dataset"))

    def genReflectivity(self):
        r1d = np.zeros(self.nz, dtype=np.float64)
        nLayers = np.random.randint(*self.layerRange)

        for _ in range(nLayers):
            pos = np.random.randint(0, self.nz)
            thickness = np.random.randint(*self.layerThickness)
            r1d[pos : pos + thickness] = np.random.uniform(-1, 1)

        return r1d

    def applyFolding(self, r1d):
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        a0 = np.random.uniform(*self.foldBaseShift)
        nGaussians = np.random.randint(*self.foldCount)
        shift2d = np.zeros((self.nx, self.ny), dtype=np.float64)

        for _ in range(nGaussians):
            x0 = np.random.uniform(-self.nx * 0.3, self.nx * 1.3)
            y0 = np.random.uniform(-self.ny * 0.3, self.ny * 1.3)
            sigmaX = np.random.uniform(*self.foldSigma)
            sigmaY = np.random.uniform(*self.foldSigma)
            theta  = np.random.uniform(0, np.pi)
            amp = np.random.uniform(*self.foldAmplitude)

            dx = xx - x0
            dy = yy - y0
            cosT, sinT = np.cos(theta), np.sin(theta)
            u = cosT * dx + sinT * dy
            v = -sinT * dx + cosT * dy
            shift2d += amp * np.exp(-(u**2 / (2 * sigmaX**2) + v**2 / (2 * sigmaY**2)))

        zGrid = np.arange(self.nz, dtype=np.float64)
        damping = self.foldDamping * zGrid / (self.nz - 1)

        # s1[x,y,z] = a0 + shift2d[x,y] * damping[z]   (same order as original)
        s1 = a0 + shift2d[:, :, None] * damping[None, None, :]
        z_target = zGrid[None, None, :] + s1
        del s1

        return ndimage.map_coordinates(r1d, [z_target], order=3, mode="nearest")

    def applyShearing(self, reflectivity):
        e0 = np.random.uniform(*self.shearOffset)
        f  = np.random.uniform(*self.shearGradient)
        g  = np.random.uniform(*self.shearGradient)

        ix, iy, iz = self._get_indices()
        x_lin = np.arange(self.nx, dtype=np.float64)
        y_lin = np.arange(self.ny, dtype=np.float64)
        s2_2d = e0 + f * x_lin[:, None] + g * y_lin[None, :]

        iz_shifted = iz.astype(np.float64)
        iz_shifted += s2_2d[:, :, None]

        return ndimage.map_coordinates(reflectivity, [ix, iy, iz_shifted], order=3, mode="nearest")

    def applyFaulting(self, reflectivity):
        masks = np.zeros(self.shape, dtype=np.uint8)
        model = reflectivity                     # no copy — first map_coordinates returns new array

        numFaults = np.random.randint(*self.faultCount)
        ix, iy, iz = self._get_indices()

        # Pre-convert to float32 once (reused via .copy() per fault)
        ix_f = ix.astype(np.float32)
        iy_f = iy.astype(np.float32)
        iz_f = iz.astype(np.float32)

        for i in range(numFaults):
            p0 = np.random.uniform(0.15, 0.85, 3) * np.array(self.shape)

            dip_angle  = np.random.uniform(*self.faultDipAngle)
            dip_rad    = np.deg2rad(dip_angle)
            strike_rad = np.random.uniform(0, 2 * np.pi)
            nx_n = np.sin(dip_rad) * np.cos(strike_rad)
            ny_n = np.sin(dip_rad) * np.sin(strike_rad)
            nz_n = np.cos(dip_rad) * np.random.choice([-1.0, 1.0])
            normal = np.array([nx_n, ny_n, nz_n])

            strike = np.array([-normal[1], normal[0], 0.0])
            strikeNorm = np.linalg.norm(strike)
            strike = np.array([1.0, 0.0, 0.0]) if strikeNorm < 1e-6 else strike / strikeNorm
            dip = np.cross(normal, strike)
            dip /= np.linalg.norm(dip)

            # Coordinate distances from fault centre
            dx = ix - p0[0]
            dy = iy - p0[1]
            dz = iz - p0[2]

            distStrike     = strike[0] * dx + strike[1] * dy + strike[2] * dz
            distDip        = dip[0]    * dx + dip[1]    * dy + dip[2]    * dz
            distPlane_base = normal[0] * dx + normal[1] * dy + normal[2] * dz
            del dx, dy, dz                              # ← free ~3 full volumes early

            # Listric (curved) fault bend
            bend = 0.0
            if np.random.random() < self.faultCurveProb:
                max_dist = max(self.shape) / 1.5
                intensidade_base = np.random.uniform(self.faultCurveMax * 0.5, self.faultCurveMax)
                direcao = np.random.choice([-1.0, 1.0])
                curve_intensity = intensidade_base * direcao
                bend = curve_intensity * ((distDip / max_dist) ** 2)

            # Rough fault surface (filtered noise)
            _noise = np.random.normal(0, 1, self.shape)
            noisePlane = ndimage.gaussian_filter(_noise, sigma=self.faultRoughSigma)
            del _noise
            noisePlane *= self.faultRoughness                    # in-place

            # Signed distance to fault surface
            distPlane = distPlane_base + noisePlane - bend
            del distPlane_base, noisePlane

            # Throw (displacement) map
            maxDisp  = np.random.uniform(*self.faultThrow)
            throwMap = self.computeThrowMap(distStrike, distDip, maxDisp)
            del distStrike, distDip

            # Apply throw to hanging-wall side
            hw = distPlane > 0
            throw_hw = throwMap[hw]

            # Copy pre-converted float32 coords (memcpy ≈ 2× faster than astype)
            ixShifted = ix_f.copy()
            iyShifted = iy_f.copy()
            izShifted = iz_f.copy()

            ixShifted[hw] += throw_hw * dip[0]
            iyShifted[hw] += throw_hw * dip[1]
            izShifted[hw] += throw_hw * dip[2]

            model = ndimage.map_coordinates(
                model, [ixShifted, iyShifted, izShifted], order=1, mode="nearest")
            masks = ndimage.map_coordinates(
                masks, [ixShifted, iyShifted, izShifted], order=0, mode="constant", cval=0)

            faultZone = (np.abs(distPlane) <= self.faultZoneWidth) & (np.abs(throwMap) > self.faultThreshold)
            masks[faultZone] = 1

            # Free per-fault temporaries
            del distPlane, throwMap, hw, throw_hw
            del ixShifted, iyShifted, izShifted, bend, faultZone

        del ix_f, iy_f, iz_f
        return model, masks

    def computeThrowMap(self, distStrike, distDip, maxDisp):
        """Compute displacement map for a single fault (gaussian or linear decay)"""
        if np.random.random() < 0.5:
            sigmaPlane = np.random.uniform(*self.faultDecaySigma)
            # result = maxDisp * exp(-(dS² + dD²) / (2σ²))
            result = distStrike ** 2                    # new array
            result += distDip ** 2                      # in-place  (dD² freed after)
            result /= -(2.0 * sigmaPlane ** 2)          # in-place  (== -(sum) / (2σ²))
            np.exp(result, out=result)                  # in-place
            result *= maxDisp                           # in-place
            return result

        planeExtent = np.sqrt(self.nx**2 + self.ny**2 + self.nz**2)
        direction   = np.random.choice([-1, 1])
        # result = maxDisp * clip(0.5 + direction * dD / extent, 0, 1)
        result = distDip / planeExtent                  # new array
        result *= direction                             # in-place  (== dir * dD / ext)
        result += 0.5                                   # in-place
        np.clip(result, 0, 1, out=result)               # in-place
        result *= maxDisp                               # in-place
        return result

    def applyWavelet(self, model):
        """Convolve with a Ricker wavelet along the Z axis."""
        f = np.random.uniform(*self.waveletFreq)
        t = np.arange(-self.waveletDuration, self.waveletDuration, self.waveletDt)
        wavelet = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-((np.pi * f * t) ** 2))
        return ndimage.convolve1d(model, wavelet, axis=2)

    def applyNoise(self, image):
        """Add band-limited Gaussian noise scaled to signal amplitude."""
        scale = np.random.uniform(*self.noiseLevel) * np.std(image)
        noise = np.random.normal(0.0, 1.0, image.shape)
        noise = ndimage.gaussian_filter(noise, sigma=(1.0, 1.0, 0.5))
        noise *= (scale / (np.std(noise) + 1e-8))

        image += noise                                  # in-place (saves one full allocation)
        image = ndimage.gaussian_filter(image, sigma=(0.5, 0.5, 0))
        return image

    def crop(self, volume):
        """Removes the safety margin to extract the final shape volume."""
        x0, x1 = self.margin, self.nx - self.margin
        y0, y1 = self.margin, self.ny - self.margin
        z0, z1 = self.margin, self.nz - self.margin
        return volume[x0:x1, y0:y1, z0:z1]

    def getMetrics(self):
        return {
            "shape": self.shape,
            "margin": self.margin,
            "layerRange": self.layerRange,
            "layerThickness": self.layerThickness,
            "foldCount": self.foldCount,
            "foldSigma": self.foldSigma,
            "foldAmplitude": self.foldAmplitude,
            "foldDamping": self.foldDamping,
            "foldBaseShift": self.foldBaseShift,
            "shearOffset": self.shearOffset,
            "shearGradient": self.shearGradient,
            "faultCount": self.faultCount,
            "faultThrow": self.faultThrow,
            "faultDipAngle": self.faultDipAngle,
            "faultRoughness": self.faultRoughness,
            "faultRoughSigma": self.faultRoughSigma,
            "faultDecaySigma": self.faultDecaySigma,
            "faultZoneWidth": self.faultZoneWidth,
            "faultThreshold": self.faultThreshold,
            "faultCurveProb": self.faultCurveProb,
            "faultCurveMax": self.faultCurveMax,
            "waveletFreq": self.waveletFreq,
            "waveletDuration": self.waveletDuration,
            "waveletDt": self.waveletDt,
            "noiseLevel": self.noiseLevel
        }

    def print(self):
        print(json.dumps(self.getMetrics(), indent=4))