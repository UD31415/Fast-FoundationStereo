"""Benchmark FastFoundationStereo + RealSense depth fusion vs baselines on the Inbolt dataset.

Extends benchmark_inbolt_fs.py by adding depth-fusion models as additional methods.

Methods compared
----------------
  original          : FFS pretrained, no fine-tuning, stereo only
  finetuned         : FFS fine-tuned on Inbolt (stereo only)
  finetuned_isaac   : FFS fine-tuned on ISAAC (stereo only)
  depth_rs          : RealSense hardware depth (no stereo)
  zivid_gt          : Projected Zivid depth (ground truth)

Usage:
  cd /home/adiroha/repos/Fast-FoundationStereo
  python scripts/benchmark_inbolt_fs_depthrs.py [--out_dir reports/inbolt_ffs_depthrs]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Utils as U
from core.utils.utils import InputPadder

# Must be imported before torch.load so the depthrs models can be unpickled

from benchmark_inbolt import (
    DepthBinAccumulator,
    _preprocess_ir,
    infer_depth_m,
    load_model,
    plot_depth_vs_distance,
    ITERS,
)
#from benchmark_inbolt import ReportGeneratorInbolt, resolve_finetuned_model_path
from scripts.data_manager_inbolt import DataSource, CAMERA_MATRIX_RS, DIST_COEFFS_RS
from metrics import (
    BenchmarkResults,
    FrameMetrics,
    compute_bin_mae,
    compute_metrics,
    aggregate,
    CLOSE_RANGE_THRESHOLD_M,
)
from report import ReportGenerator

# ── custom report generator ──────────────────────────────────────────────────

class ReportGeneratorInbolt(ReportGenerator):
    """Custom report generator that shows 4 frames in depth comparison and error maps."""

    def __init__(self, results, stats, output_dir) -> None:
        super().__init__(results, stats, output_dir)
        self._selected_viz_indices = []

    def _get_selected_viz_indices(self, n_pick: int = 4):
        """Return cached random frame indices used consistently across report sections."""
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

        sel = self._get_selected_viz_indices(n_pick=4)
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
                im = ax.imshow(vf[name], cmap=cmap, vmin=0.1, vmax=2.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")
                title = self._r.method_labels.get(name, name)
                if c == 0:
                    title = f"Frame {frame_idx + 1} • {title}"
                ax.set_title(title, fontsize=9, wrap=True)
                ax.axis("off")

        fig.suptitle("Depth Map Comparison (4 random frames) — values in meters",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, "depth_comparison.png")

    def _fig_error_maps(self) -> str:
        if not self._r.viz_frames or not self._non_gt:
            return self._empty_fig("error_maps.png", "No comparison methods")

        sel = self._get_selected_viz_indices(n_pick=4)
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
                im = ax.imshow(err, cmap=cmap, vmin=0.001, vmax=0.1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|error| (m)")
                mean_err = float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else 0.0
                label = self._r.method_labels.get(name, name)
                if c == 0:
                    ax.set_title(f"Frame {frame_idx + 1} • {label}\nMAE={mean_err:.4f} m", fontsize=9)
                else:
                    ax.set_title(f"{label}\nMAE={mean_err:.4f} m", fontsize=9)
                ax.axis("off")

        gt_label = self._r.method_labels.get(self._gt, self._gt)
        fig.suptitle(f"Absolute Error vs {gt_label} (4 random frames, m)", fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, "error_maps.png")




# ── constants ────────────────────────────────────────────────────────────────

#DATA_DIR       = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260415T084601Z-3-001/Data Collection'
#BF              = 50.102706998586 * 385.509887695312  #49470.45   # focal_px * baseline_mm  (calibrated from camera)
DATA_DIR       = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260518-03' 
BF              = 50.102706998586 * 642.4910888671875 # new data 3 2026-05-18

ORIGINAL_PATH  = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
FINETUNED_PATH = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt-20260415_epoch_111.pth'
ISAACTUNED_PATH= f'{code_dir}/../weights/23-36-37/model_finetuned_isaac_epoch_037.pth'
DEFAULT_OUT    = f'{code_dir}/../reports/benchmark_inbolt_trained_isaac_set2'
N_VIZ          = 6

METHODS: Dict[str, Dict[str, str]] = {
    'original':          {'label': 'FFS Original',                         'color': '#2980b9'},
    'finetuned':         {'label': 'FFS Fine-tuned (Inbolt)',              'color': '#e74c3c'},
    'isaactuned':        {'label': 'FFS fine-tuned (ISAAC)',               'color': '#8e44ad'},
    'depth_rs':          {'label': 'RealSense Hardware Depth',             'color': '#f39c12'},
    'zivid_gt':          {'label': 'Zivid GT (projected to RS)',           'color': '#27ae60'},
}
GT_NAME = 'zivid_gt'
RS_NAME = 'depth_rs'
RS_FPS  = 30.0


# ── depth-fusion inference ────────────────────────────────────────────────────

def load_depthrs_model(path: str):
    """Load a FastFoundationStereoDepthRS checkpoint."""
    logging.info(f"Loading depth-fusion model from {path}")
    model = torch.load(path, map_location='cpu', weights_only=False)
    model.cuda().eval()
    return model


def resolve_depthrs_model_path(preferred_path: str) -> Optional[str]:
    """Return an existing depthrs v1 checkpoint path, or None if not found."""
    preferred = Path(preferred_path)
    if preferred.exists():
        return str(preferred)

    weights_dir = Path(code_dir) / '..' / 'weights'
    candidates = sorted(weights_dir.glob('**/model_finetuned_inbolt_depthrs_epoch_*.pth'))
    if candidates:
        chosen = candidates[-1]
        logging.warning(f'Preferred depthrs model not found at {preferred}. Using {chosen}')
        return str(chosen)

    return None

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out_dir',   default=DEFAULT_OUT,    help='Output directory')
    parser.add_argument('--data_dir',  default=DATA_DIR,       help='Dataset root')
    parser.add_argument('--original',  default=ORIGINAL_PATH,  help='Original FFS weights')
    parser.add_argument('--finetuned', default=FINETUNED_PATH, help='Stereo-only fine-tuned weights')
    parser.add_argument('--isaactuned',default=ISAACTUNED_PATH,    help='Stereo-only fine-tuned weights on ISAAC')
    parser.add_argument('--n_viz', type=int, default=N_VIZ,     help='Frames saved for visual comparison')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load models ───────────────────────────────────────────────────────────
    models          = {}   # name → model  (stereo-only interface)

    finetuned_path = args.finetuned
    if Path(finetuned_path).exists():
        models['finetuned'] = load_model(finetuned_path)
    else:
        logging.warning(f'Stereo fine-tuned model not found at {args.finetuned} — skipping')

    isaactuned_path = args.isaactuned
    if Path(isaactuned_path).exists():
        models['isaactuned'] = load_model(isaactuned_path)
    else:
        logging.warning(f'Depth-fusion v1 model not found at {args.isaactuned} — skipping')

    models['original'] = load_model(args.original)        

    # depthrs_v2_path = resolve_depthrs_v2_model_path(args.depthrs_v2)
    # if depthrs_v2_path is not None:
    #     depthrs_models['depthrs_v2'] = load_depthrs_model(depthrs_v2_path)
    # else:
    #     logging.warning(f'Depth-fusion v2 model not found at {args.depthrs_v2} — skipping')

    all_model_names = list(models.keys()) # + list(depthrs_models.keys())
    active_methods  = [GT_NAME, RS_NAME] + all_model_names

    # ── dataset ───────────────────────────────────────────────────────────────
    source = DataSource(train_mode=False)
    n = source.init_directory(input_rectified=args.data_dir)
    logging.info(f'Found {n} samples in {args.data_dir}')
    if n == 0:
        logging.error('No samples found — check DATA_DIR path')
        return

    # ── accumulators ──────────────────────────────────────────────────────────
    all_metrics        = []
    viz_frames         = []
    valid_acc          = {}
    dist_bin_mae       = {m: [] for m in active_methods}
    close_range_valid  = {m: [] for m in active_methods}
    timing_ms_raw      = {m: [] for m in all_model_names}
    H = W = None

    depth_acc_keys = [GT_NAME, RS_NAME] + all_model_names
    depth_accs     = {k: DepthBinAccumulator() for k in depth_acc_keys}

    for idx in range(n):
        data  = source.get_item_projected(idx)
        left  = data['left']
        right = data['right']
        gt_mm = data['depth_zivid'].astype(np.float32)
        rs_mm = data['depth_rs'].astype(np.float32)

        h, w = gt_mm.shape[:2]
        if rs_mm.shape != (h, w):
            import cv2
            rs_mm = cv2.resize(rs_mm, (w, h), interpolation=cv2.INTER_NEAREST)

        if H is None:
            H, W = h, w
            for m in active_methods:
                valid_acc[m] = np.zeros((H, W), np.float32)

        gt_m = gt_mm / 1000.0
        rs_m = rs_mm / 1000.0

        frame_depths = {GT_NAME: gt_m, RS_NAME: rs_m}

        # stereo-only models
        for mname, model in models.items():
            t0 = time.monotonic()
            frame_depths[mname] = infer_depth_m(model, left, right, bf = BF)
            timing_ms_raw[mname].append((time.monotonic() - t0) * 1000.0)


        gt_close_mask = (gt_m > 0) & (gt_m < CLOSE_RANGE_THRESHOLD_M)
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
                fm = compute_metrics(pred, gt_m, elapsed_ms=0.0, method_name=RS_NAME)
            else:
                fm = compute_metrics(pred, gt_m, timing_ms_raw[mname][-1], mname)

            all_metrics.append(fm)
            dist_bin_mae[mname].append(compute_bin_mae(pred, gt_m))

            close_cov = (
                float((pred[gt_close_mask] > 0).mean()) * 100.0
                if n_close > 0 else 0.0
            )
            close_range_valid[mname].append(close_cov)

        depth_accs[GT_NAME].update(gt_m, gt_m)
        depth_accs[RS_NAME].update(rs_m, gt_m)
        for mname in all_model_names:
            depth_accs[mname].update(frame_depths[mname], gt_m)

        if idx < args.n_viz:
            viz_frames.append({k: v.copy() for k, v in frame_depths.items()})

        if (idx + 1) % 20 == 0 or (idx + 1) == n:
            logging.info(f'  {idx + 1}/{n} frames processed')

    for m in active_methods:
        valid_acc[m] /= max(n, 1)

    mean_timing = {
        m: float(np.mean(ts)) if ts else 0.0
        for m, ts in timing_ms_raw.items()
    }
    mean_timing[GT_NAME] = 0.0
    mean_timing[RS_NAME] = 1000.0 / RS_FPS

    method_configs = {
        'original':  {'model_path': args.original},
        RS_NAME:     {'source': f'RealSense hardware depth (~{RS_FPS:.0f} FPS)'},
        GT_NAME:     {'source': 'Projected Zivid depth map (ground truth)'},
    }
    if 'finetuned' in models and finetuned_path:
        method_configs['finetuned'] = {'model_path': finetuned_path}
    if 'isaactuned' in models and isaactuned_path:
        method_configs['isaactuned'] = {'model_path': isaactuned_path}
    # if 'depthrs_v2' in models and depthrs_v2_path:
    #     method_configs['depthrs_v2'] = {'model_path': depthrs_v2_path}

    results = BenchmarkResults(
        method_names=active_methods,
        method_labels={m: METHODS[m]['label'] for m in active_methods},
        method_colors={m: METHODS[m]['color'] for m in active_methods},
        ground_truth_name=GT_NAME,
        n_frames=n,
        width=W,
        height=H,
        all_metrics=all_metrics,
        viz_frames=viz_frames,
        coverage_maps=valid_acc,
        dist_bin_mae=dist_bin_mae,
        close_range_valid=close_range_valid,
        source=f'INBOLT dataset ({args.data_dir})',
        method_configs=method_configs,
    )

    stats = aggregate(results, mean_timing)
    if RS_NAME in stats:
        stats[RS_NAME].fps_mean = RS_FPS

    reporter = ReportGeneratorInbolt(results, stats, out_dir)
    reporter.generate()

    # depth-vs-distance accuracy plot
    plot_labels = {k: METHODS[k]['label'] for k in METHODS}
    plot_colors = {k: METHODS[k]['color'] for k in METHODS}

    labeled_accs = {
        plot_labels.get(k, k): v
        for k, v in depth_accs.items()
        if depth_accs[k].count.sum() > 0
    }
    labeled_colors = {
        plot_labels.get(k, k): plot_colors.get(k)
        for k in depth_accs
        if depth_accs[k].count.sum() > 0
    }

    plot_depth_vs_distance(
        accumulators=labeled_accs,
        colors=labeled_colors,
        out_path=out_dir / 'depth_vs_distance.png',
    )
    logging.info(f'All outputs written to {out_dir}')


if __name__ == '__main__':
    main()
