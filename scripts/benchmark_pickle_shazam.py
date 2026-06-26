"""Benchmark stereo models + RealSense hardware depth + Shazam estimator against Pickle CAD GT.

Extends ``benchmark_pickle_trained_isaac.py`` by adding the CPU-only Shazam
Gabor-bank stereo depth estimator (``scripts/shazam_depth_estimator.py``) as an
additional evaluation method, alongside the original / fine-tuned / ISAAC-tuned
Fast-FoundationStereo models, the RealSense hardware depth, and the Pickle CAD
ground truth (projected via ``DataSource.get_item_projected``).

Metric descriptions match ``benchmark_pickle_trained_isaac.py``. All depth
values are in millimetres (mm) throughout this script.

Usage:
  cd /path/to/Fast-FoundationStereo
  python scripts/benchmark_pickle_shazam.py [--out_dir reports/benchmark_pickle_shazam]
                                            [--shazam_scale 0.5]
                                            [--shazam_max_disp 64]
"""

import argparse
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

# Driver 580 / CUDA 13.0 does not enumerate devices without CUDA_VISIBLE_DEVICES set.
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            text=True
        )
        indices = ','.join(line.strip() for line in out.splitlines() if line.strip())
        if indices:
            os.environ['CUDA_VISIBLE_DEVICES'] = indices
    except Exception:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch

from core.utils.utils import InputPadder
import Utils as U
from scripts.data_manager_pickle import DataSource
from metrics import (
    BenchmarkResults,
    FrameMetrics,
    compute_metrics,
    aggregate,
)
from report import ReportGenerator


# ── shazam_depth_estimator import shim ────────────────────────────────────────
# shazam_depth_estimator.py imports a few helper modules from an external
# ``Utils\src`` checkout that is not present in this workspace
# (``common.RectSelector``, ``logger.log``, ``depth_data_source.DataSource``,
# ``opencv_realsense_camera.RealSense``/``draw_str``). Stub them out in
# ``sys.modules`` so the module is importable without those packages.
def _install_shazam_stubs() -> None:
    def _stub(name: str, **attrs):
        if name in sys.modules:
            return
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    class _NullRectSelector:
        def __init__(self, *a, **kw): pass
        def draw(self, *a, **kw): pass

    class _NullRealSense:
        def __init__(self, *a, **kw): pass

    def _noop(*a, **kw): pass

    _stub("logger", log=logging.getLogger("shazam"))
    _stub("common", RectSelector=_NullRectSelector)
    _stub("depth_data_source", DataSource=type("DataSource", (), {}))
    _stub(
        "opencv_realsense_camera",
        RealSense=_NullRealSense,
        draw_str=_noop,
    )


_install_shazam_stubs()
from shazam_depth_estimator import ShazamDepthEstimator  # noqa: E402


# ── constants ─────────────────────────────────────────────────────────────────

# Default Pickle manifest (Excel). The path is automatically translated from
# Windows UNC to the local POSIX mount by ``data_manager_pickle.translate_path``.
PICKLE_EXCEL = (
    r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0\2026_06"
    r"\yg_pickle\\2026-06-22--14-48-11\Pickle_Scene_Capture_336222073841"
    r"\data path.xlsx"
)

ORIGINAL_PATH   = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
FINETUNED_PATH  = f'{code_dir}/../weights/23-36-37/model_finetuned_pickle_epoch_020.pth'
ISAACTUNED_PATH = f'{code_dir}/../weights/23-36-37/model_finetuned_isaac_epoch_037.pth'
DEFAULT_OUT     = f'{code_dir}/../reports/benchmark_pickle_shazam'

# Projection method used to render CAD-based ground-truth depth.
PROJECTION_METHOD = "splat"

# Shazam runs on CPU and scales O(H·W·D). ``gabor_image_disparity_down_up_full_volume``
# internally builds a 3-level pyramid with max_disparity=128 at the finest level,
# so the full-resolution cost volume is ~ (H · W · 3 · 128 · 4 bytes). To keep memory
# and latency in check on CPU we optionally pre-downsample the IR pair and rescale
# the returned disparity back to native resolution.
SHAZAM_SCALE = 0.5   # pre-downscale factor applied to IR pair before Gabor matching

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

ITERS = 8           # GRU update iterations for FFS models
N_VIZ = 12          # frames saved for visual comparison in the report

# Depth threshold for the "close-range" coverage metric — in mm
CLOSE_RANGE_THRESHOLD_MM = 20.0

# Distance bins used for the per-bin MAE curve — all in mm
DIST_BINS_MM: List[Tuple[float, float]] = [
    (0.0,    100.0),
    (100.0,  200.0),
    (200.0,  450.0),
    (450.0,  1000.0),
    (1000.0, 1500.0),
]
BIN_LABELS_MM  = ["0–100 mm", "100–200 mm", "200–450 mm", "450–1000 mm", "1000–1500 mm"]
BIN_CENTERS_MM = [50.0, 150.0, 325.0, 725.0, 1250.0]

METHODS: Dict[str, Dict[str, str]] = {
    "original":   {"label": "FFS Original",                  "color": "#2980b9"},
    "finetuned":  {"label": "FFS Fine-tuned (Pickle)",       "color": "#e74c3c"},
    "isaactuned": {"label": "FFS Fine-tuned (ISAAC)",        "color": "#8e44ad"},
    "shazam":     {"label": "Shazam (Gabor, CPU)",           "color": "#16a085"},
    "depth_rs":   {"label": "RealSense Hardware Depth",      "color": "#f39c12"},
    "pickle_gt":  {"label": "Pickle CAD GT (projected)",     "color": "#27ae60"},
}
GT_NAME     = "pickle_gt"
RS_NAME     = "depth_rs"
SHAZAM_NAME = "shazam"


# ── shazam wrapper ────────────────────────────────────────────────────────────

class ShazamRunner:
    """Thin wrapper around ``ShazamDepthEstimator`` exposing a clean
    ``infer_depth_mm(left, right, bf)`` API.

    Delegates disparity estimation to ``ShazamDepthEstimator.gabor_image_disparity_down_up_full_volume``
    (3-level Gabor pyramid cost volume + softmax + argmax). The IR pair is
    optionally pre-downsampled to bound the CPU cost / memory of the internal
    full-resolution cost volume; the resulting disparity map is then rescaled
    back to the native image resolution.

    The underlying function creates a number of matplotlib figures as a side
    effect. With the ``Agg`` backend ``plt.show()`` is a no-op, but figures
    accumulate in memory; ``plt.close('all')`` is called after every invocation
    to release them.
    """

    def __init__(self, scale: float = SHAZAM_SCALE) -> None:
        self.scale = float(scale)
        self.estimator = ShazamDepthEstimator()

    @staticmethod
    def _to_gray2d(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = img[..., 0]
        return img

    @staticmethod
    def _safe_debug_row(h: int) -> int:
        # ``gabor_image_disparity_down_up_full_volume`` unconditionally indexes
        # ``prob_total[debug_row ± 10, ...]``, so the row must satisfy
        # 10 ≤ debug_row ≤ H - 11 even when no debug visualisation is wanted.
        return int(min(max(10, h // 2), max(11, h - 11)))

    def infer_depth_mm(self, left: np.ndarray, right: np.ndarray, bf: float) -> np.ndarray:
        left  = self._to_gray2d(left).astype(np.float32)
        right = self._to_gray2d(right).astype(np.float32)
        H, W  = left.shape[:2]

        if self.scale != 1.0:
            new_w = max(32, int(round(W * self.scale)))
            new_h = max(32, int(round(H * self.scale)))
            left_s  = cv.resize(left,  (new_w, new_h), interpolation=cv.INTER_AREA)
            right_s = cv.resize(right, (new_w, new_h), interpolation=cv.INTER_AREA)
        else:
            left_s, right_s = left, right

        Hs = left_s.shape[0]
        try:
            disp_s = self.estimator.gabor_image_disparity_down_up_full_volume(
                left_s, right_s, debug_row=self._safe_debug_row(Hs)
            ).astype(np.float32)
        finally:
            # Release figures created by the estimator before returning.
            plt.close('all')

        # Rescale disparity back to native resolution. Disparity is measured in
        # pixels, so it must also be multiplied by the inverse spatial scale.
        if self.scale != 1.0:
            disp = cv.resize(disp_s, (W, H), interpolation=cv.INTER_LINEAR)
            disp = disp / self.scale
        else:
            disp = disp_s

        depth_mm = np.zeros_like(disp, dtype=np.float32)
        valid = disp > 0.5  # subpixel floor avoids divide-by-zero/overflow
        depth_mm[valid] = float(bf) / disp[valid]
        return depth_mm


# ── mm-based metric helpers ───────────────────────────────────────────────────

def compute_bin_mae_mm(pred_mm: np.ndarray, gt_mm: np.ndarray) -> List[float]:
    """MAE (mm) per distance bin; returns NaN for bins with no valid GT pixels."""
    result = []
    for lo, hi in DIST_BINS_MM:
        mask = (gt_mm >= lo) & (gt_mm < hi) & (gt_mm > 0) & (pred_mm > 0)
        if mask.sum() == 0:
            result.append(float("nan"))
        else:
            mask = mask & (np.abs(pred_mm - gt_mm) < 20.0)  # ignore extreme outliers
            result.append(float(np.abs(pred_mm[mask] - gt_mm[mask]).mean()))
    return result


# ── inference helpers (FFS models) ────────────────────────────────────────────

def _preprocess_ir(left: np.ndarray, right: np.ndarray):
    """Convert uint8/uint16 IR images to CUDA float tensors (3-channel pseudo-RGB)."""
    left  = np.clip(left.astype(np.float32),  0, 255)
    right = np.clip(right.astype(np.float32), 0, 255)
    left  = np.stack([left,  left,  left],  axis=-1)
    right = np.stack([right, right, right], axis=-1)
    left_t  = torch.as_tensor(left).float()[None].permute(0, 3, 1, 2).to(DEVICE)
    right_t = torch.as_tensor(right).float()[None].permute(0, 3, 1, 2).to(DEVICE)
    return left_t, right_t


@torch.no_grad()
def infer_depth_mm(model, left: np.ndarray, right: np.ndarray, bf: float) -> np.ndarray:
    """Run stereo inference on an IR pair; return depth map in mm (H×W float32)."""
    left_t, right_t = _preprocess_ir(left, right)
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)

    with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
        disp = model.forward(left_t, right_t, iters=ITERS, test_mode=True)

    disp = padder.unpad(disp.float())
    disp_np = disp.cpu().numpy().reshape(left.shape[:2]).clip(0, None)

    depth_mm = np.zeros_like(disp_np)
    valid = disp_np > 0
    depth_mm[valid] = bf / disp_np[valid]
    return depth_mm


def load_model(path: str):
    logging.info(f"Loading model from {path}")
    model = torch.load(path, map_location='cpu', weights_only=False)
    model.cuda().eval()
    return model


# ── mm-aware report generator ─────────────────────────────────────────────────

class ReportGeneratorMM(ReportGenerator):
    """ReportGenerator subclass with all axis labels and colorbars in mm."""

    _bin_labels  = BIN_LABELS_MM
    _bin_centers = BIN_CENTERS_MM

    def __init__(self, results, stats, output_dir) -> None:
        super().__init__(results, stats, output_dir)
        self._selected_viz_indices: List[int] = []

    def _get_selected_viz_indices(self, n_pick: int = 4) -> List[int]:
        if self._selected_viz_indices:
            return self._selected_viz_indices
        n_total = len(self._r.viz_frames)
        if n_total == 0:
            self._selected_viz_indices = []
            return self._selected_viz_indices
        n = min(n_pick, n_total)
        rng = np.random.default_rng(42)
        self._selected_viz_indices = sorted(rng.choice(n_total, size=n, replace=False).tolist())
        return self._selected_viz_indices

    def _fig_depth_comparison(self) -> str:
        if not self._r.viz_frames:
            return self._empty_fig("depth_comparison.png", "No viz frames")
        sel = self._get_selected_viz_indices(n_pick=12)
        if not sel:
            return self._empty_fig("depth_comparison.png", "No viz frames")
        vf0 = self._r.viz_frames[sel[0]]
        method_names = [n for n in self._r.method_names if n in vf0]
        nrows = len(sel)
        ncols = len(method_names)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        axes = np.atleast_2d(axes)
        cmap = self._depth_cmap()
        for r, frame_idx in enumerate(sel):
            vf = self._r.viz_frames[frame_idx]
            for c, name in enumerate(method_names):
                ax = axes[r, c]
                if name not in vf:
                    ax.axis("off")
                    continue
                im = ax.imshow(vf[name], cmap=cmap, vmin=1.0, vmax=1500.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm")
                title = self._r.method_labels.get(name, name)
                if c == 0:
                    title = f"Frame {frame_idx + 1} • {title}"
                ax.set_title(title, fontsize=9, wrap=True)
                ax.axis("off")
        fig.suptitle("Depth Map Comparison (12 random frames) — values in mm",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, "depth_comparison.png")

    def _fig_error_maps(self) -> str:
        if not self._r.viz_frames or not self._non_gt:
            return self._empty_fig("error_maps.png", "No comparison methods")
        sel = self._get_selected_viz_indices(n_pick=12)
        if not sel:
            return self._empty_fig("error_maps.png", "No viz frames")
        vf0 = self._r.viz_frames[sel[0]]
        names = ([self._gt] if self._gt in vf0 else []) + [n for n in self._non_gt if n in vf0]
        if not names:
            return self._empty_fig("error_maps.png", "Ground truth not available in viz frame")
        nrows = len(sel)
        ncols = len(names)
        cmap = plt.get_cmap("hot").copy()
        cmap.set_under("#222222")
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        axes = np.atleast_2d(axes)
        for r, frame_idx in enumerate(sel):
            vf = self._r.viz_frames[frame_idx]
            gt = vf.get(self._gt)
            if gt is None:
                for c in range(ncols):
                    axes[r, c].axis("off")
                continue
            for c, name in enumerate(names):
                ax = axes[r, c]
                if name not in vf:
                    ax.axis("off")
                    continue
                pred = vf[name]
                valid = (gt > 0) & (pred > 0)
                err = np.where(valid, np.abs(pred - gt), 0.0).astype(np.float32)
                im = ax.imshow(err, cmap=cmap, vmin=1.0, vmax=10.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|error| (mm)")
                mean_err = float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else 0.0
                label = self._r.method_labels.get(name, name)
                if c == 0:
                    ax.set_title(f"Frame {frame_idx + 1} • {label}\nMAE={mean_err:.0f} mm", fontsize=9)
                else:
                    ax.set_title(f"{label}\nMAE={mean_err:.0f} mm", fontsize=9)
                ax.axis("off")
        gt_label = self._r.method_labels.get(self._gt, self._gt)
        fig.suptitle(f"Absolute Error vs {gt_label} (12 random frames, mm)", fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, "error_maps.png")

    def _fig_distance_error_curve(self) -> str:
        if not self._non_gt:
            return self._empty_fig("distance_error_curve.png", "No comparison methods")
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in self._non_gt:
            bin_data = self._r.dist_bin_mae.get(name, [])
            if not bin_data:
                continue
            arr = np.array(bin_data)
            mean_per_bin = np.array([
                np.nanmean(arr[:, i]) if np.any(~np.isnan(arr[:, i])) else 0.0
                for i in range(arr.shape[1])
            ])
            color = self._r.method_colors.get(name, "#888")
            label = self._r.method_labels.get(name, name)
            ax.plot(self._bin_centers, mean_per_bin, marker="o", color=color,
                    label=label, linewidth=2, markersize=7)
        ax.set_xticks(self._bin_centers)
        ax.set_xticklabels(self._bin_labels, fontsize=9)
        ax.set_xlabel("Distance range", fontsize=10)
        ax.set_ylabel("Mean Absolute Error (mm)", fontsize=10)
        ax.set_title("Depth Error vs Distance", fontsize=12)
        ax.set_ylim(0, 10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "distance_error_curve.png")

    def _fig_error_histograms(self) -> str:
        if not self._non_gt or not self._r.viz_frames:
            return self._empty_fig("error_histograms.png", "No comparison data")
        names = [n for n in self._non_gt
                 if any(n in vf and self._gt in vf for vf in self._r.viz_frames)]
        if not names:
            return self._empty_fig("error_histograms.png", "No viz data for comparison")
        n = len(names)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.asarray(axes).flatten().tolist()
        for ax, name in zip(axes, names):
            errors = []
            for vf in self._r.viz_frames:
                if name not in vf or self._gt not in vf:
                    continue
                pred, gt = vf[name], vf[self._gt]
                valid = (gt > 0) & (pred > 0)
                if valid.any():
                    errors.extend(np.abs(pred[valid] - gt[valid]).tolist())
            if not errors:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                continue
            color = self._r.method_colors.get(name, "#888")
            label = self._r.method_labels.get(name, name)
            ax.hist(errors, bins=50, range=(0.0, 500.0), color=color,  alpha=0.8, edgecolor="none")
            mean_e = float(np.mean(errors))
            ax.axvline(mean_e, color="red", linestyle="--", linewidth=1.5,  label=f"mean={mean_e:.0f} mm")
            ax.set_xlabel("Absolute error (mm)", fontsize=9)
            ax.set_ylabel("Pixel count", fontsize=9)
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=8)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Per-Pixel Error Distribution (vs GT, viz frames)", fontsize=11)
        fig.tight_layout()
        return self._save(fig, "error_histograms.png")

    def _fig_summary_table(self) -> str:
        if not self._stats:
            return self._empty_fig("summary_table.png", "No stats")
        cols = ["Method", "MRE* (%)", "MRE (%)", "MAE (mm)", "δ1 (%)",
                "Coverage (%)", "FPS", "GPU %", "GT?"]
        gt_rows, other_rows = [], []
        for name, s in self._stats.items():
            is_gt = (name == self._gt)
            row = [
                s.label,
                "—" if is_gt else f"{s.mre_pen_mean * 100:.1f}",
                "—" if is_gt else f"{s.mre_mean * 100:.1f}",
                "—" if is_gt else f"{s.mae_mean:.1f}",
                "—" if is_gt else f"{s.delta1_mean:.1f}",
                f"{s.coverage_mean:.1f}",
                f"{s.fps_mean:.1f}" if s.fps_mean < 999 else "≈30",
                f"{s.gpu_load_mean:.0f}" if s.gpu_load_mean > 0 else "—",
                "★ GT" if is_gt else "",
            ]
            (gt_rows if is_gt else other_rows).append((name, row))
        ordered = gt_rows + other_rows
        cell_text = [r for _, r in ordered]
        n = len(ordered)
        fig, ax = plt.subplots(figsize=(14, 1.0 + 0.55 * n))
        ax.axis("off")
        table = ax.table(cellText=cell_text, colLabels=cols,
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f7f7f7")
            cell.set_edgecolor("#cccccc")
            if row > 0 and cell_text[row - 1][-1] == "★ GT":
                cell.set_facecolor("#d5f5d5")
        ax.set_title("Depth Quality Summary (errors in mm)", fontsize=12,
                     pad=10, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "summary_table.png")

    def _fig_close_range_analysis(self) -> str:
        names = list(self._r.method_names)
        if not names:
            return self._empty_fig("close_range_analysis.png", "No methods")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        labels    = [self._r.method_labels.get(n, n) for n in names]
        coverages = [self._stats[n].close_range_coverage if n in self._stats else 0.0
                     for n in names]
        colors    = [self._r.method_colors.get(n, "#888") for n in names]
        bars = ax1.bar(labels, coverages, color=colors, alpha=0.85, edgecolor="white")
        ax1.bar_label(bars, labels=[f"{v:.1f}%" for v in coverages], padding=3, fontsize=7)
        ax1.set_ylabel(f"Coverage at < {CLOSE_RANGE_THRESHOLD_MM:.0f} mm (%)", fontsize=10)
        ax1.set_title(f"Close-Range Coverage (< {CLOSE_RANGE_THRESHOLD_MM:.0f} mm)", fontsize=11)
        ax1.tick_params(axis="x", rotation=45, labelsize=7)
        ax1.set_xticklabels(labels, ha="right")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", alpha=0.3)
        for name in names:
            vals = self._r.close_range_valid.get(name, [])
            if not vals:
                continue
            color = self._r.method_colors.get(name, "#888")
            label = self._r.method_labels.get(name, name)
            ax2.plot(range(1, len(vals) + 1), vals, color=color,
                     label=label, alpha=0.8, linewidth=1.5)
        ax2.set_xlabel("Frame", fontsize=10)
        ax2.set_ylabel(f"Coverage at < {CLOSE_RANGE_THRESHOLD_MM:.0f} mm (%)", fontsize=10)
        ax2.set_title("Close-Range Coverage per Frame", fontsize=11)
        ax2.legend(fontsize=6, loc="best")
        ax2.grid(alpha=0.3)
        ax2.set_ylim(-5, 115)
        fig.suptitle("Close-Range Depth Analysis", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "close_range_analysis.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out_dir',       default=DEFAULT_OUT,
                        help='Output directory for the report')
    parser.add_argument('--pickle_excel',  default=PICKLE_EXCEL,
                        help='Path to the Pickle manifest Excel (data path.xlsx). '
                             'Windows UNC paths are auto-translated to the local mount.')
    parser.add_argument('--original',      default=ORIGINAL_PATH,
                        help='Path to original model weights')
    parser.add_argument('--finetuned',     default=FINETUNED_PATH,
                        help='Path to Pickle fine-tuned model weights')
    parser.add_argument('--isaactuned',    default=ISAACTUNED_PATH,
                        help='Path to ISAAC fine-tuned model weights')
    parser.add_argument('--shazam_scale',  type=float, default=SHAZAM_SCALE,
                        help='Pre-downscale factor applied to the IR pair before Shazam matching '
                             '(smaller = faster + less memory, less accurate).')
    parser.add_argument('--n_viz',         type=int, default=N_VIZ,
                        help='Frames saved for visual comparison')
    parser.add_argument('--projection',    default=PROJECTION_METHOD,
                        choices=('splat', 'raycast', 'open3d'),
                        help='CAD projection method used to render GT depth.')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load FFS models ───────────────────────────────────────────────────────
    models = {}
    if Path(args.finetuned).exists():
        models["finetuned"] = load_model(args.finetuned)
    else:
        raise FileNotFoundError(
            f"Fine-tuned model not found at {args.finetuned}. "
            f"Pass --finetuned <path> or set FINETUNED_PATH to an existing checkpoint."
        )

    if Path(args.isaactuned).exists():
        models["isaactuned"] = load_model(args.isaactuned)
    else:
        raise FileNotFoundError(
            f"ISAAC fine-tuned model not found at {args.isaactuned}. "
            f"Pass --isaactuned <path> or set ISAACTUNED_PATH to an existing checkpoint."
        )

    models["original"] = load_model(args.original)

    # ── init Shazam estimator (CPU, no weights) ───────────────────────────────
    logging.info(f"Initialising Shazam estimator (scale={args.shazam_scale})")
    shazam = ShazamRunner(scale=args.shazam_scale)

    # active_methods includes GT, RS hardware, Shazam, and all NN models
    active_methods = [GT_NAME, RS_NAME, SHAZAM_NAME] + list(models.keys())

    # ── dataset ───────────────────────────────────────────────────────────────
    source = DataSource(train_mode=False)
    n = source.init_directory(excel_path=args.pickle_excel)
    logging.info(f"Found {n} samples in {args.pickle_excel}")
    if n == 0:
        logging.error("No samples found — check --pickle_excel path")
        return

    # ── accumulators ──────────────────────────────────────────────────────────
    all_metrics       = []
    viz_frames        = []
    valid_acc         = {}
    dist_bin_mae      = {m: [] for m in active_methods}
    close_range_valid = {m: [] for m in active_methods}
    # NN models AND shazam track per-frame latency
    timing_ms_raw     = {m: [] for m in list(models.keys()) + [SHAZAM_NAME]}
    H = W = None

    for idx in range(n):
        data  = source.get_item_and_scene_projected(idx)
        left  = data['ir_left_img']
        right = data['ir_right_img']
        gt_mm = data['depth_cad_projected'].astype(np.float32)
        rs_mm = data['depth_img'].astype(np.float32)
        bf    = data['bf']

        if gt_mm.shape != rs_mm.shape:
            logging.warning(
                f"Item {idx}: depth_cad_projected {gt_mm.shape} != depth_img {rs_mm.shape}; skipping"
            )
            continue

        if H is None:
            H, W = gt_mm.shape[:2]
            for m in active_methods:
                valid_acc[m] = np.zeros((H, W), np.float32)

        frame_depths = {GT_NAME: gt_mm, RS_NAME: rs_mm}

        # FFS models (GPU)
        for mname, model in models.items():
            t0 = time.monotonic()
            frame_depths[mname] = infer_depth_mm(model, left, right, bf)
            timing_ms_raw[mname].append((time.monotonic() - t0) * 1000.0)

        # Shazam (CPU)
        t0 = time.monotonic()
        frame_depths[SHAZAM_NAME] = shazam.infer_depth_mm(left, right, bf)
        timing_ms_raw[SHAZAM_NAME].append((time.monotonic() - t0) * 1000.0)

        # ── per-frame metrics (all values in mm) ──────────────────────────────
        gt_close_mask = (gt_mm > 0) & (gt_mm < CLOSE_RANGE_THRESHOLD_MM)
        n_close = int(gt_close_mask.sum())

        for mname in active_methods:
            pred = frame_depths[mname]
            valid_acc[mname] += (pred > 0).astype(np.float32)

            if mname == GT_NAME:
                fm = FrameMetrics(
                    GT_NAME, 0.0, 0.0, 0.0, 100.0,
                    float((pred > 0).mean()) * 100.0, 0.0,
                    mae_pen=0.0, mre_pen=0.0,
                )
            elif mname == RS_NAME:
                fm = compute_metrics(pred, gt_mm, elapsed_ms=0.0, method_name=RS_NAME)
            else:
                fm = compute_metrics(pred, gt_mm, timing_ms_raw[mname][-1], mname)

            all_metrics.append(fm)
            dist_bin_mae[mname].append(compute_bin_mae_mm(pred, gt_mm))

            close_cov = (
                float((pred[gt_close_mask] > 0).mean()) * 100.0
                if n_close > 0 else 0.0
            )
            close_range_valid[mname].append(close_cov)

        if idx < args.n_viz:
            viz_frames.append({k: v.copy() for k, v in frame_depths.items()})

        if (idx + 1) % 50 == 0 or (idx + 1) == n:
            logging.info(f"  {idx + 1}/{n} frames processed")

    for m in active_methods:
        valid_acc[m] /= max(n, 1)

    # ── aggregate timing ──────────────────────────────────────────────────────
    mean_timing: Dict[str, float] = {
        m: float(np.mean(ts)) if ts else 0.0
        for m, ts in timing_ms_raw.items()
    }
    mean_timing[GT_NAME] = 0.0
    mean_timing[RS_NAME] = 1000.0 / 30.0

    # ── build BenchmarkResults ────────────────────────────────────────────────
    method_configs = {
        "original":  {"model_path": args.original},
        "finetuned": {"model_path": args.finetuned},
        "isaactuned": {"model_path": args.isaactuned},
        SHAZAM_NAME: {
            "estimator":  "ShazamDepthEstimator.gabor_image_disparity_down_up_full_volume (CPU)",
            "scale":      str(args.shazam_scale),
            "levels":     "3 (pyramid)",
            "max_disp":   "128 (finest level)",
        },
        RS_NAME: {"source": "RealSense hardware depth (depth_img, ~30 FPS)"},
        GT_NAME: {
            "source": (
                f"Pickle CAD-rendered ground-truth depth via "
                f"DataSource.get_item_projected(method={args.projection!r})"
            )
        },
    }

    results = BenchmarkResults(
        method_names=active_methods,
        method_labels={m: METHODS[m]["label"] for m in active_methods},
        method_colors={m: METHODS[m]["color"] for m in active_methods},
        ground_truth_name=GT_NAME,
        n_frames=n,
        width=W,
        height=H,
        all_metrics=all_metrics,
        viz_frames=viz_frames,
        coverage_maps=valid_acc,
        dist_bin_mae=dist_bin_mae,
        close_range_valid=close_range_valid,
        source=(
            f"Pickle scene-capture  •  {args.pickle_excel}  "
            f"•  projection={args.projection}  •  shazam_scale={args.shazam_scale}"
        ),
        method_configs=method_configs,
    )

    stats = aggregate(results, mean_timing)
    if RS_NAME in stats:
        stats[RS_NAME].fps_mean = 30.0

    # ── generate report ───────────────────────────────────────────────────────
    reporter = ReportGeneratorMM(results, stats, out_dir)
    reporter.generate()


if __name__ == '__main__':
    main()
