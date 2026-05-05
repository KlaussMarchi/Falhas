"""
report_generator.py
Generates a multi-page PDF comparing seismic volumes from two datasets (Wu and GRvA),
with fault mask overlays and a visual separator between dataset groups.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FAULT_CMAP  = ListedColormap(["black", "white", "white", "white"])
OVERLAY_CMAP = ListedColormap(["red"])
SEISMIC_CMAP = "gray"
SEPARATOR_COLOR = "#444444"
SEPARATOR_LW    = 2.0
SEPARATOR_HEIGHT_RATIO = 0.12   # fraction relative to a data row (height_ratios)


# ── Data containers ───────────────────────────────────────────────────────────
@dataclass
class RowSpec:
    """Describes one row of slices to be rendered."""
    volume:       np.ndarray
    label:        str
    is_mask:      bool
    overlay_mask: Optional[np.ndarray] = None


# ── VolumeLoader ──────────────────────────────────────────────────────────────
class VolumeLoader:
    """Loads and normalises a 3-D numpy volume from disk."""

    @staticmethod
    def load(path: str) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            log.warning("File not found: %s", path)
            return None
        vol = np.load(path)
        if vol.ndim > 3:
            vol = np.squeeze(vol)
        if vol.ndim != 3:
            raise ValueError(f"Expected a 3-D array, got shape {vol.shape} for {path}")
        return vol


# ── Slicer ────────────────────────────────────────────────────────────────────
class Slicer:
    """Extracts and orientates the three orthogonal mid-plane slices."""

    @staticmethod
    def get_mid_slices(vol: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int, int]]:
        mid_x = vol.shape[0] // 2
        mid_y = vol.shape[1] // 2
        mid_z = vol.shape[2] // 2

        yz = np.array(vol[mid_x, :, :])          # inline
        xz = np.array(vol[:, mid_y, :])           # crossline
        xy = np.array(vol[:, :, mid_z])            # depth

        # Reorient so that depth and crossline planes look natural
        slices = [yz, np.rot90(xy, -1), xz]
        return slices, (mid_x, mid_y, mid_z)


# ── PDFReportGenerator ────────────────────────────────────────────────────────
class PDFReportGenerator:
    """
    Orchestrates loading, plotting, and saving a multi-page PDF that compares
    seismic volumes from two datasets side-by-side.

    Dataset 0 (Wu)   rows are rendered first.
    A visible horizontal separator divides them from
    Dataset 1 (GRvA) rows below.
    """

    def __init__(
        self,
        ds0_img_dir:  str,
        ds0_mask_dir: str,
        ds1_img_dir:  str,
        ds1_mask_dir: str,
        output_pdf:   str,
    ) -> None:
        self.ds0_img_dir  = ds0_img_dir
        self.ds0_mask_dir = ds0_mask_dir
        self.ds1_img_dir  = ds1_img_dir
        self.ds1_mask_dir = ds1_mask_dir
        self.output_pdf   = output_pdf

    # ── File discovery ────────────────────────────────────────────────────────

    def _resolve_path(self, directory: str, idx: int, *prefixes: str) -> str:
        """
        Try multiple filename patterns (e.g. ``img_0003.npy``, ``3.npy``)
        and return the first one that exists.
        """
        candidates = [
            os.path.join(directory, f"{prefix}{idx:04d}.npy") for prefix in prefixes
        ] + [os.path.join(directory, f"{idx}.npy")]

        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[-1]   # return last candidate so the caller can report it

    def _get_file_paths(self, idx: int) -> tuple[str, str, str, str]:
        ds0_img  = os.path.join(self.ds0_img_dir,  f"{idx}.npy")
        ds0_mask = os.path.join(self.ds0_mask_dir, f"{idx}.npy")
        ds1_img  = self._resolve_path(self.ds1_img_dir,  idx, "img_")
        ds1_mask = self._resolve_path(self.ds1_mask_dir, idx, "img_")
        return ds0_img, ds0_mask, ds1_img, ds1_mask

    # ── Plotting helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _imshow_slice(
        ax: plt.Axes,
        data: np.ndarray,
        is_mask: bool,
    ) -> None:
        """Render a single 2-D slice on *ax*."""
        if is_mask:
            ax.imshow(data, cmap=FAULT_CMAP, vmin=0, vmax=3)
        else:
            ax.imshow(data, cmap=SEISMIC_CMAP)

    @staticmethod
    def _overlay_fault(ax: plt.Axes, mask_slice: np.ndarray) -> None:
        """Overlay fault pixels in semi-transparent red on *ax*."""
        binary = (mask_slice > 0).astype(float)
        masked = np.ma.masked_where(binary == 0, binary)
        ax.imshow(masked, cmap=OVERLAY_CMAP, alpha=0.8, interpolation="none")

    def _plot_row(
        self,
        axes_row: list[plt.Axes],
        row_spec: RowSpec,
    ) -> None:
        """Fill one row of three axes with the inline / crossline / depth slices."""
        slices, (mid_x, mid_y, mid_z) = Slicer.get_mid_slices(row_spec.volume)

        overlay_slices: Optional[list[np.ndarray]] = None
        if row_spec.overlay_mask is not None:
            overlay_slices, _ = Slicer.get_mid_slices(row_spec.overlay_mask)

        col_labels = [
            f"{row_spec.label}\nSlice X={mid_x} (inline)",
            f"{row_spec.label}\nSlice Y={mid_y} (crossline)",
            f"{row_spec.label}\nSlice Z={mid_z} (depth)",
        ]

        for col_idx, ax in enumerate(axes_row):
            self._imshow_slice(ax, slices[col_idx], row_spec.is_mask)
            if overlay_slices is not None:
                self._overlay_fault(ax, overlay_slices[col_idx])
            ax.set_title(col_labels[col_idx], fontsize=11)
            ax.axis("off")

    # ── Separator ─────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_separator(ax_sep: plt.Axes, label: str = "── GRvA Dataset ──") -> None:
        """
        Draw a full-width horizontal rule plus a centred label inside *ax_sep*,
        which should be a very thin 'spacer' axes spanning all columns.
        """
        ax_sep.set_xlim(0, 1)
        ax_sep.set_ylim(0, 1)
        ax_sep.axis("off")

        # Horizontal line across the full axes width
        line = Line2D(
            [0.0, 1.0], [0.5, 0.5],
            transform=ax_sep.transAxes,
            color=SEPARATOR_COLOR,
            linewidth=SEPARATOR_LW,
            linestyle="--",
        )
        ax_sep.add_line(line)

        # Centred label on top of the line
        ax_sep.text(
            0.5, 0.5, f"  {label}  ",
            transform=ax_sep.transAxes,
            ha="center", va="center",
            fontsize=12, fontstyle="italic",
            color=SEPARATOR_COLOR,
            bbox=dict(facecolor="white", edgecolor="none", pad=3),
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(self, num_samples: int = 25) -> None:
        """
        Iterate over sample indices, build one PDF page per sample, and save.
        """
        with PdfPages(self.output_pdf) as pdf:
            processed = 0
            for i in range(num_samples):
                paths = self._get_file_paths(i)
                missing = [p for p in paths if not os.path.exists(p)]
                if missing:
                    log.warning("Sample %d — skipping, missing files: %s", i, missing)
                    continue

                vols = [VolumeLoader.load(p) for p in paths]
                if any(v is None for v in vols):
                    log.warning("Sample %d — skipping due to load error.", i)
                    continue

                ds0_img, ds0_mask, ds1_img, ds1_mask = vols

                # Row specs: 2 for Wu, 2 for GRvA
                wu_rows: list[RowSpec] = [
                    RowSpec(ds0_img, "Wu Synthetic — Seismic",      is_mask=False),
                    RowSpec(ds0_img, "Wu Synthetic — Fault Overlay", is_mask=False, overlay_mask=ds0_mask),
                ]
                grva_rows: list[RowSpec] = [
                    RowSpec(ds1_img, "GRvA Synthetic — Seismic",      is_mask=False),
                    RowSpec(ds1_img, "GRvA Synthetic — Fault Overlay", is_mask=False, overlay_mask=ds1_mask),
                ]

                n_data_rows = len(wu_rows) + len(grva_rows)

                # ── Build GridSpec with an extra thin separator row ────────────
                # height_ratios: [wu_row, wu_row, sep, grva_row, grva_row]
                sep_h = SEPARATOR_HEIGHT_RATIO
                hr = [1.0] * len(wu_rows) + [sep_h] + [1.0] * len(grva_rows)

                # Extra height accounts for the title area (1.2 in) + breathing room
                fig_height = 5 * n_data_rows + 1.2
                fig = plt.figure(figsize=(15, fig_height), constrained_layout=False)

                # Reserve the top 4 % for the suptitle, content fills the rest
                title_top = 0.97   # where the title sits (figure fraction)
                content_top = 0.93 # topmost edge of the subplot grid

                gs = GridSpec(
                    nrows=len(hr), ncols=3,
                    figure=fig,
                    height_ratios=hr,
                    hspace=0.40,
                    wspace=0.05,
                    top=content_top,
                    bottom=0.02,
                    left=0.02,
                    right=0.98,
                )

                fig.suptitle(
                    f"Dataset Comparison — Sample {i}",
                    fontsize=18, fontweight="bold", y=title_top,
                )

                # ── Wu rows ───────────────────────────────────────────────────
                for r, row_spec in enumerate(wu_rows):
                    axes_row = [fig.add_subplot(gs[r, c]) for c in range(3)]
                    self._plot_row(axes_row, row_spec)

                # ── Separator row ─────────────────────────────────────────────
                sep_row_idx = len(wu_rows)
                ax_sep = fig.add_subplot(gs[sep_row_idx, :])   # span all columns
                self._draw_separator(ax_sep, label="GRvA Dataset  ▼")

                # ── GRvA rows ─────────────────────────────────────────────────
                for r, row_spec in enumerate(grva_rows):
                    row_idx = sep_row_idx + 1 + r
                    axes_row = [fig.add_subplot(gs[row_idx, c]) for c in range(3)]
                    self._plot_row(axes_row, row_spec)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                processed += 1
                log.info("Sample %d processed  (%d/%d)", i, processed, num_samples)

        log.info("PDF saved → %s  (%d pages)", self.output_pdf, processed)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generator = PDFReportGenerator(
        ds0_img_dir  = "dataset0/images",
        ds0_mask_dir = "dataset0/masks",
        ds1_img_dir  = "dataset1/images",
        ds1_mask_dir = "dataset1/masks",
        output_pdf   = "output.pdf",
    )
    generator.generate(num_samples=15)