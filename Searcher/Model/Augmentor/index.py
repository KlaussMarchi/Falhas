import random
import numpy as np
from scipy import ndimage

"""
Usage example in notebook:

    from Augmentor.index import Compose

    AUGMENTATIONS = [
        {"name": "RandomFlip",        "args": {"axes": [0, 1, 2], "p": 0.5}},
        {"name": "Rot90",             "args": {"axes": [1, 2],    "p": 0.5}},
        {"name": "GaussianNoise",     "args": {"std": [0.01, 0.05], "p": 0.5}},
        {"name": "GaussianBlur",      "args": {"sigma": [0.5, 1.5], "p": 0.3}},
        {"name": "IntensityScale",    "args": {"low": 0.9, "high": 1.1, "p": 0.3}},
        {"name": "IntensityShift",    "args": {"low": -0.1, "high": 0.1, "p": 0.3}},
        {"name": "GammaCorrection",   "args": {"low": 0.7, "high": 1.5, "p": 0.3}},
        {"name": "ElasticDeformation","args": {"alpha": 4, "sigma": 1, "p": 0.2}},
        {"name": "CoarseDropout",     "args": {"n": 3, "size": [8, 16], "fill": 0, "p": 0.3}},
        {"name": "Transpose",         "args": {"axes": [[0,1],[0,2],[1,2]], "p": 0.3}},
        {"name": "Normalize",         "args": {"mean": 0.0, "std": 1.0}},
        {"name": "Clip",              "args": {"low": -3.0, "high": 3.0}},
    ]

    class CustomDataset(Dataset):
        def __init__(self, df, multiclass=False, augmentations=None):
            self.df = df.reset_index(drop=True)
            self.multiclass = multiclass
            self.augmentor  = Compose(augmentations) if augmentations else None

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row  = self.df.iloc[idx]
            img  = np.load(row.img_path).astype(np.float32)
            mask = np.load(row.mask_path).astype(np.float32)

            if self.augmentor:
                img, mask = self.augmentor(img, mask)

            img_tensor  = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
            return (img_tensor, mask_tensor)
"""


class Transform3D:
    """Base class for 3D augmentation transforms."""
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return self.apply(img, mask)
        return img, mask

    def apply(self, img, mask):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Spatial transforms (applied to BOTH image and mask)
# ─────────────────────────────────────────────────────────────────────────────

class RandomFlip(Transform3D):
    """Random flip along each of the specified axes independently."""
    def __init__(self, axes=None, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes or [0, 1, 2]

    def apply(self, img, mask):
        for axis in self.axes:
            if random.random() < 0.5:
                img = np.flip(img, axis=axis).copy()
                mask = np.flip(mask, axis=axis).copy()
        return img, mask


class Flip(Transform3D):
    """Flip along a single fixed axis."""
    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def apply(self, img, mask):
        return np.flip(img, axis=self.axis).copy(), np.flip(mask, axis=self.axis).copy()


class Rot90(Transform3D):
    """Random 90/180/270 degree rotation in the specified plane."""
    def __init__(self, axes=None, k=None, **kwargs):
        super().__init__(**kwargs)
        self.axes = tuple(axes or [1, 2])
        self.k = k  # None = random choice from 1,2,3

    def apply(self, img, mask):
        k = self.k if self.k is not None else random.randint(1, 3)
        return np.rot90(img, k=k, axes=self.axes).copy(), np.rot90(mask, k=k, axes=self.axes).copy()


class Transpose(Transform3D):
    """Random axis swap from a list of axis pairs."""
    def __init__(self, axes=None, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes or [[0, 1], [0, 2], [1, 2]]

    def apply(self, img, mask):
        pair = random.choice(self.axes)
        return np.swapaxes(img, pair[0], pair[1]).copy(), np.swapaxes(mask, pair[0], pair[1]).copy()


class ElasticDeformation(Transform3D):
    """
    Elastic deformation of the 3D volume.
    alpha: deformation intensity
    sigma: smoothing of the displacement field
    """
    def __init__(self, alpha=4.0, sigma=1.0, order=3, mask_order=0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.mask_order = mask_order

    def apply(self, img, mask):
        shape = img.shape
        dx = ndimage.gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dy = ndimage.gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dz = ndimage.gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha

        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )

        coords = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1),
        ]

        img = ndimage.map_coordinates(img, coords, order=self.order, mode='reflect').astype(np.float32)
        mask = ndimage.map_coordinates(mask, coords, order=self.mask_order, mode='reflect').astype(mask.dtype)
        return img, mask


class RandomShift(Transform3D):
    """Random translation along each axis (in voxels)."""
    def __init__(self, max_shift=None, fill=0, mask_fill=0, **kwargs):
        super().__init__(**kwargs)
        self.max_shift = max_shift or [4, 4, 4]
        self.fill = fill
        self.mask_fill = mask_fill

    def apply(self, img, mask):
        shifts = [random.randint(-s, s) for s in self.max_shift]
        img = ndimage.shift(img, shifts, order=0, mode='constant', cval=self.fill).astype(np.float32)
        mask = ndimage.shift(mask, shifts, order=0, mode='constant', cval=self.mask_fill).astype(mask.dtype)
        return img, mask


class RandomZoom(Transform3D):
    """Random zoom/scale of the volume."""
    def __init__(self, low=0.9, high=1.1, order=3, mask_order=0, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high
        self.order = order
        self.mask_order = mask_order

    def apply(self, img, mask):
        factor = random.uniform(self.low, self.high)
        shape = img.shape
        img = ndimage.zoom(img, factor, order=self.order).astype(np.float32)
        mask = ndimage.zoom(mask, factor, order=self.mask_order).astype(mask.dtype)
        img = self._crop_or_pad(img, shape)
        mask = self._crop_or_pad(mask, shape)
        return img, mask

    def _crop_or_pad(self, vol, target_shape):
        result = np.zeros(target_shape, dtype=vol.dtype)
        slices_src = []
        slices_dst = []
        for s, t in zip(vol.shape, target_shape):
            if s >= t:
                start = (s - t) // 2
                slices_src.append(slice(start, start + t))
                slices_dst.append(slice(0, t))
            else:
                start = (t - s) // 2
                slices_src.append(slice(0, s))
                slices_dst.append(slice(start, start + s))
        result[tuple(slices_dst)] = vol[tuple(slices_src)]
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Intensity transforms (applied to image ONLY)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNoise(Transform3D):
    """Additive Gaussian noise. std can be float or [min, max] for random range."""
    def __init__(self, std=0.02, mean=0.0, **kwargs):
        super().__init__(**kwargs)
        self.std = std
        self.mean = mean

    def apply(self, img, mask):
        if isinstance(self.std, (list, tuple)):
            std = random.uniform(self.std[0], self.std[1])
        else:
            std = self.std
        noise = np.random.normal(self.mean, std, img.shape).astype(np.float32)
        return img + noise, mask


class GaussianBlur(Transform3D):
    """3D Gaussian blur. sigma can be float or [min, max] for random range."""
    def __init__(self, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def apply(self, img, mask):
        if isinstance(self.sigma, (list, tuple)):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma
        return ndimage.gaussian_filter(img, sigma=sigma).astype(np.float32), mask


class IntensityScale(Transform3D):
    """Multiply intensity by a random factor in [low, high]."""
    def __init__(self, low=0.9, high=1.1, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def apply(self, img, mask):
        factor = random.uniform(self.low, self.high)
        return (img * factor).astype(np.float32), mask


class IntensityShift(Transform3D):
    """Add a random offset to intensity in [low, high]."""
    def __init__(self, low=-0.1, high=0.1, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def apply(self, img, mask):
        offset = random.uniform(self.low, self.high)
        return (img + offset).astype(np.float32), mask


class GammaCorrection(Transform3D):
    """Random gamma correction. Assumes input roughly in [0, 1] range."""
    def __init__(self, low=0.7, high=1.5, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def apply(self, img, mask):
        gamma = random.uniform(self.low, self.high)
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8:
            return img, mask
        normalized = (img - mn) / (mx - mn)
        corrected = np.power(normalized, gamma)
        return (corrected * (mx - mn) + mn).astype(np.float32), mask


class CoarseDropout(Transform3D):
    """Randomly erase cubic patches from the volume (image only)."""
    def __init__(self, n=3, size=None, fill=0, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.size = size or [8, 16]  # [min, max] side length
        self.fill = fill

    def apply(self, img, mask):
        for _ in range(self.n):
            if isinstance(self.size, (list, tuple)):
                s = random.randint(self.size[0], self.size[1])
            else:
                s = self.size
            d = random.randint(0, max(0, img.shape[0] - s))
            h = random.randint(0, max(0, img.shape[1] - s))
            w = random.randint(0, max(0, img.shape[2] - s))
            img[d:d+s, h:h+s, w:w+s] = self.fill
        return img, mask


class Contrast(Transform3D):
    """Random contrast adjustment around the mean."""
    def __init__(self, low=0.75, high=1.25, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def apply(self, img, mask):
        factor = random.uniform(self.low, self.high)
        mean = img.mean()
        return ((img - mean) * factor + mean).astype(np.float32), mask


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic transforms (always applied, p=1.0 default)
# ─────────────────────────────────────────────────────────────────────────────

class Normalize(Transform3D):
    """Z-score normalization."""
    def __init__(self, mean=None, std=None, **kwargs):
        kwargs.setdefault('p', 1.0)
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def apply(self, img, mask):
        mean = self.mean if self.mean is not None else img.mean()
        std = self.std if self.std is not None else img.std()
        if std < 1e-8:
            return img, mask
        return ((img - mean) / std).astype(np.float32), mask


class Clip(Transform3D):
    """Clip intensity values."""
    def __init__(self, low=-3.0, high=3.0, **kwargs):
        kwargs.setdefault('p', 1.0)
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def apply(self, img, mask):
        return np.clip(img, self.low, self.high).astype(np.float32), mask


# ─────────────────────────────────────────────────────────────────────────────
# Registry + Compose
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY = {
    # Spatial
    "Flip":               Flip,
    "RandomFlip":         RandomFlip,
    "Rot90":              Rot90,
    "Transpose":          Transpose,
    "ElasticDeformation": ElasticDeformation,
    "RandomShift":        RandomShift,
    "RandomZoom":         RandomZoom,
    # Intensity
    "GaussianNoise":      GaussianNoise,
    "GaussianBlur":       GaussianBlur,
    "IntensityScale":     IntensityScale,
    "IntensityShift":     IntensityShift,
    "GammaCorrection":    GammaCorrection,
    "CoarseDropout":      CoarseDropout,
    "Contrast":           Contrast,
    # Deterministic
    "Normalize":          Normalize,
    "Clip":               Clip,
}


class Compose:
    """
    Builds a 3D augmentation pipeline from a JSON-compatible list.

    Args:
        config: list of dicts, each with "name" and optional "args".

    Example:
        [
            {"name": "RandomFlip",      "args": {"axes": [0, 1, 2], "p": 0.5}},
            {"name": "Rot90",           "args": {"axes": [1, 2], "p": 0.5}},
            {"name": "GaussianNoise",   "args": {"std": [0.01, 0.05], "p": 0.5}},
            {"name": "GaussianBlur",    "args": {"sigma": [0.5, 1.5], "p": 0.3}},
            {"name": "IntensityScale",  "args": {"low": 0.9, "high": 1.1, "p": 0.3}},
            {"name": "IntensityShift",  "args": {"low": -0.1, "high": 0.1, "p": 0.3}},
            {"name": "GammaCorrection", "args": {"low": 0.7, "high": 1.5, "p": 0.2}},
            {"name": "CoarseDropout",   "args": {"n": 3, "size": [8, 16], "p": 0.3}},
            {"name": "Contrast",        "args": {"low": 0.8, "high": 1.2, "p": 0.3}},
            {"name": "ElasticDeformation","args": {"alpha": 4, "sigma": 1, "p": 0.15}},
            {"name": "RandomShift",     "args": {"max_shift": [4, 4, 4], "p": 0.2}},
            {"name": "RandomZoom",      "args": {"low": 0.9, "high": 1.1, "p": 0.2}},
            {"name": "Normalize"},
            {"name": "Clip",            "args": {"low": -3, "high": 3}}
        ]
    """

    def __init__(self, config):
        self.transforms = []
        for item in (config or []):
            name = item["name"]
            args = item.get("args", {})
            cls = REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown transform: '{name}'. Available: {list(REGISTRY.keys())}")
            self.transforms.append(cls(**args))

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

    def __repr__(self):
        lines = [f"Compose(["]
        for t in self.transforms:
            lines.append(f"  {t.__class__.__name__}(p={t.p}),")
        lines.append("])")
        return "\n".join(lines)
