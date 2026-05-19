import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndimage
from utils import setFolder, formatAxis
import os, json


class SyntheticGenerator:
    def __init__(self, shape=(128, 128, 128)):
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
        self.faultThreshold  = 1.3        # Filtro de rótulo. ↑ Marca só falha grande. ↓ Marca qualquer rachadurazinha.

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

    def get(self):
        data = self.genReflectivity()
        data = self.applyFolding(data)
        data = self.applyShearing(data)
        data, mask = self.applyFaulting(data)
        image = self.applyWavelet(data)
        image = self.applyNoise(image)

        image = self.crop(image)
        mask  = self.crop(mask)
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        return image.astype(np.float32), mask.astype(np.uint8)

    def dataset(self, n=200, outputDir="output"):
        imgDir = os.path.join(outputDir, "images")
        mskDir = os.path.join(outputDir, "masks")
        setFolder(imgDir)
        setFolder(mskDir)

        for i in tqdm(range(n), desc="Generating dataset"):
            image, mask = self.get()
            image, mask = formatAxis(image), formatAxis(mask)
            
            np.save(os.path.join(imgDir, f"img_{i:04d}.npy"), image)
            np.save(os.path.join(mskDir, f"img_{i:04d}.npy"), mask)

    def genReflectivity(self):
        """Create 1D layered reflectivity tiled across the volume."""
        r1d = np.zeros(self.nz)
        nLayers = np.random.randint(*self.layerRange)

        for _ in range(nLayers):
            pos = np.random.randint(0, self.nz)
            thickness = np.random.randint(*self.layerThickness)
            r1d[pos : pos + thickness] = np.random.uniform(-1, 1)

        return np.tile(r1d, (self.nx, self.ny, 1))
        
    def applyFolding(self, reflectivity):
        """Deform layers with rotated anisotropic Gaussian folds."""
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        a0 = np.random.uniform(*self.foldBaseShift)
        nGaussians = np.random.randint(*self.foldCount)
        shift2d    = np.zeros((self.nx, self.ny))

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

        zGrid = np.arange(self.nz)
        damping = self.foldDamping * zGrid / (self.nz - 1)
        s1 = a0 + shift2d[:, :, np.newaxis] * damping

        ix, iy, iz = np.indices(self.shape)
        return ndimage.map_coordinates(reflectivity, [ix, iy, iz + s1], order=3, mode="nearest")

    def applyShearing(self, reflectivity):
        """Apply linear shear (dip/tilt) along X and Y axes."""
        e0 = np.random.uniform(*self.shearOffset)
        f  = np.random.uniform(*self.shearGradient)
        g  = np.random.uniform(*self.shearGradient)

        ix, iy, iz = np.indices(self.shape)
        s2 = e0 + f * ix + g * iy
        return ndimage.map_coordinates(reflectivity, [ix, iy, iz + s2], order=3, mode="nearest")

    def applyFaulting(self, reflectivity):
        """Inject faults with displacement and produce binary mask."""
        masks = np.zeros(self.shape, dtype=np.uint8)
        model = np.copy(reflectivity)

        numFaults  = np.random.randint(*self.faultCount)
        ix, iy, iz = np.indices(self.shape)

        for i in range(numFaults):
            p0 = np.random.uniform(0.15, 0.85, 3) * np.array(self.shape)

            dip_angle  = np.random.uniform(*self.faultDipAngle)
            dip_rad    = np.deg2rad(dip_angle)
            strike_rad = np.random.uniform(0, 2 * np.pi)
            nx = np.sin(dip_rad) * np.cos(strike_rad)
            ny = np.sin(dip_rad) * np.sin(strike_rad)
            nz = np.cos(dip_rad) * np.random.choice([-1.0, 1.0])
            normal = np.array([nx, ny, nz])

            strike = np.array([-normal[1], normal[0], 0.0])
            strikeNorm = np.linalg.norm(strike)
            strike = np.array([1.0, 0.0, 0.0]) if strikeNorm < 1e-6 else strike / strikeNorm
            dip = np.cross(normal, strike)
            dip /= np.linalg.norm(dip)

            dx = ix - p0[0]
            dy = iy - p0[1]
            dz = iz - p0[2]

            distStrike = strike[0] * dx + strike[1] * dy + strike[2] * dz
            distDip = dip[0] * dx + dip[1] * dy + dip[2] * dz
            bend    = 0.0
            
            if np.random.random() < self.faultCurveProb:
                max_dist = max(self.shape) / 1.5 
                intensidade_base = np.random.uniform(self.faultCurveMax * 0.5, self.faultCurveMax)
                direcao = np.random.choice([-1.0, 1.0])
                curve_intensity = intensidade_base * direcao
                bend = curve_intensity * ((distDip / max_dist) ** 2)

            noisePlane = ndimage.gaussian_filter(np.random.normal(0, 1, self.shape), sigma=self.faultRoughSigma) * self.faultRoughness
            distPlane  = normal[0] * dx + normal[1] * dy + normal[2] * dz + noisePlane - bend
            maxDisp  = np.random.uniform(*self.faultThrow)
            throwMap = self.computeThrowMap(distStrike, distDip, maxDisp)

            shift_x = np.zeros_like(ix, dtype=float)
            shift_y = np.zeros_like(iy, dtype=float)
            shift_z = np.zeros_like(iz, dtype=float)

            hw = distPlane > 0
            shift_x[hw] = throwMap[hw] * dip[0]
            shift_y[hw] = throwMap[hw] * dip[1]
            shift_z[hw] = throwMap[hw] * dip[2]

            ixShifted = ix.astype(float) + shift_x
            iyShifted = iy.astype(float) + shift_y
            izShifted = iz.astype(float) + shift_z

            model = ndimage.map_coordinates(model, [ixShifted, iyShifted, izShifted], order=1, mode="nearest")
            masks = ndimage.map_coordinates(masks, [ixShifted, iyShifted, izShifted], order=0, mode="constant", cval=0)
            faultZone = (np.abs(distPlane) <= self.faultZoneWidth) & (np.abs(throwMap) > self.faultThreshold)
            masks[faultZone] = 1

        return model, masks

    def computeThrowMap(self, distStrike, distDip, maxDisp):
        """Compute displacement map for a single fault (gaussian or linear decay)."""
        if np.random.random() < 0.5:
            sigmaPlane = np.random.uniform(*self.faultDecaySigma)
            return maxDisp * np.exp(-(distStrike**2 + distDip**2) / (2 * sigmaPlane**2))

        planeExtent = np.sqrt(self.nx**2 + self.ny**2 + self.nz**2)
        direction   = np.random.choice([-1, 1])
        return maxDisp * np.clip(0.5 + direction * distDip / planeExtent, 0, 1)

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
        
        image = (image + noise)
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