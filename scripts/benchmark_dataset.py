"""Benchmark FastFoundationStereo (original + fine-tuned) on the RealSense D405
captures stored in ``data/d405``.

The dataset is laid out as three PNG files per frame::

    data/d405/imageL_d16_<idx>.png   # left  IR image  (uint8, HxW)
    data/d405/imageR_d16_<idx>.png   # right IR image  (uint8, HxW)
    data/d405/imageD_d16_<idx>.png   # hardware depth  (uint16, mm)

For each frame we run both stereo models on the left/right IR pair and produce
depth maps in metres. The hardware RealSense depth (``imageD_*``) is loaded for
reference. Depth maps for both models are stored on disk as 16-bit PNGs in mm.
Finally, a self-contained HTML report is written that visually compares, frame
by frame, the RealSense depth against the two FFS predictions.

Usage::

    cd /home/adiroha/repos/Fast-FoundationStereo
    python scripts/benchmark_dataset.py \\
        [--data_dir data/d405] \\
        [--out_dir reports/d405_benchmark]
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import Utils as U
from benchmark_inbolt import infer_depth_m, load_model
import benchmark_inbolt as bi


# ── constants ────────────────────────────────────────────────────────────────

DATA_DIR        = f'{code_dir}/../data/d405'
ORIGINAL_PATH   = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
FINETUNED_PATH  = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt_0518_epoch_067.pth'
DEFAULT_OUT     = f'{code_dir}/../reports/benchmark_d405'

# RealSense D405 IR intrinsics / baseline (used to convert disparity → depth).
# fx (px) for native 1280x720 IR ≈ 638.77, stereo baseline ≈ 18.0 mm.
D405_FX_PX       = 638.77
D405_BASELINE_MM = 18.0
BF               = D405_FX_PX * D405_BASELINE_MM   # focal_px * baseline_mm

METHODS: Dict[str, Dict[str, str]] = {
    'depth_rs':  {'label': 'RealSense Hardware Depth', 'color': '#f39c12'},
    'original':  {'label': 'FFS Original',             'color': '#2980b9'},
    'finetuned': {'label': 'FFS Fine-tuned',           'color': '#e74c3c'},
}


# ── data loading ─────────────────────────────────────────────────────────────

_INDEX_RE = re.compile(r'imageL_d16_(\d+)\.png$')


def discover_frames(data_dir: Path) -> List[str]:
    """Return sorted list of frame indices that have L/R/D triplets available."""
    indices: List[str] = []
    for p in sorted(data_dir.glob('imageL_d16_*.png')):
        m = _INDEX_RE.search(p.name)
        if not m:
            continue
        idx = m.group(1)
        right = data_dir / f'imageR_d16_{int(idx):03d}.png'
        depth = data_dir / f'imageD_d16_{int(idx):03d}.png'
        if right.exists() and depth.exists():
            indices.append(idx)
        else:
            logging.warning(f'Skipping frame {idx}: missing right/depth pair')
    return indices


def load_frame(data_dir: Path, idx: str):
    left  = cv2.imread(str(data_dir / f'imageL_d16_{int(idx):03d}.png'), cv2.IMREAD_UNCHANGED)
    right = cv2.imread(str(data_dir / f'imageR_d16_{int(idx):03d}.png'), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(data_dir / f'imageD_d16_{int(idx):03d}.png'), cv2.IMREAD_UNCHANGED)
    # crop image center of size 480x640 images to match FFS input resolution
    # the input image is 720x1280, 

    if left is not None and right is not None:
        h, w = left.shape
        top, bot = (h - 480) // 2, (w - 640) // 2
        left = left[top:top + 480, bot:bot + 640]
        right = right[top:top + 480, bot:bot + 640]
        depth = depth[top:top + 480, bot:bot + 640]
    return left, right, depth


# ── report ───────────────────────────────────────────────────────────────────

def _save_depth_png_mm(depth_m: np.ndarray, path: Path) -> None:
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), depth_mm)


def _load_depth_png_m(path: Path) -> np.ndarray:
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        return np.zeros((0, 0), np.float32)
    return depth_mm.astype(np.float32) / 1000.0


def _render_comparison(left: np.ndarray, depths: Dict[str, np.ndarray],
                       frame_idx: str, out_path: Path,
                       vmin: float = 0.1, vmax: float = 1.0) -> None:
    ncols = 1 + len(depths)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.6))
    axes[0].imshow(left, cmap='gray')
    axes[0].set_title(f'Frame {frame_idx} • Left IR')
    axes[0].axis('off')

    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('#222222')
    for ax, (name, d) in zip(axes[1:], depths.items()):
        masked = np.where(d > 0, d, np.nan)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m')
        ax.set_title(METHODS[name]['label'])
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _write_html_report(out_dir: Path, frame_indices: List[str],
                       method_names: List[str], timings_ms: Dict[str, List[float]],
                       comparison_pngs: List[Path]) -> Path:
    html_path = out_dir / 'report.html'

    rows = []
    for name in method_names:
        if name in timings_ms and timings_ms[name]:
            arr = np.asarray(timings_ms[name], dtype=np.float64)
            mean_ms = float(arr.mean())
            fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            rows.append(
                f'<tr><td>{METHODS[name]["label"]}</td>'
                f'<td>{mean_ms:.1f}</td><td>{fps:.1f}</td></tr>'
            )
        else:
            rows.append(
                f'<tr><td>{METHODS[name]["label"]}</td><td>n/a</td><td>n/a</td></tr>'
            )
    timing_table = (
        '<table border="1" cellpadding="6" cellspacing="0">'
        '<tr><th>Method</th><th>Mean inference time (ms)</th><th>FPS</th></tr>'
        + ''.join(rows) + '</table>'
    )

    figures_html = []
    for idx, png in zip(frame_indices, comparison_pngs):
        rel = png.relative_to(out_dir).as_posix()
        figures_html.append(
            f'<h3>Frame {idx}</h3>'
            f'<img src="{rel}" style="max-width:100%; height:auto;"/>'
        )

    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>D405 FFS Benchmark</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 24px auto; padding: 0 16px; }}
h1 {{ color: #2c3e50; }}
h3 {{ margin-top: 28px; color: #34495e; }}
table {{ border-collapse: collapse; }}
th {{ background: #ecf0f1; }}
</style></head><body>
<h1>RealSense D405 — FastFoundationStereo Benchmark</h1>
<p><b>Dataset:</b> {len(frame_indices)} frames loaded from <code>data/d405</code>.</p>
<p>Each row compares the left IR image, the RealSense hardware depth
(<code>imageD_*</code>), and depth maps produced by the original and
fine-tuned FastFoundationStereo models on the same stereo pair. Depth
maps are rendered with a common colour scale (m).</p>

<h2>Inference timing</h2>
{timing_table}

<h2>Per-frame depth comparison</h2>
{''.join(figures_html)}

</body></html>
"""
    html_path.write_text(body)
    return html_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data_dir',  default=DATA_DIR)
    parser.add_argument('--out_dir',   default=DEFAULT_OUT)
    parser.add_argument('--original',  default=ORIGINAL_PATH)
    parser.add_argument('--finetuned', default=FINETUNED_PATH)
    parser.add_argument('--bf', type=float, default=BF,
                        help='focal_px * baseline_mm used to convert disparity to depth')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='limit number of frames processed (0 = all)')
    args = parser.parse_args()

    U.set_logging_format()

    data_dir = Path(args.data_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    (out_dir / 'depth_original').mkdir(parents=True, exist_ok=True)
    (out_dir / 'depth_finetuned').mkdir(parents=True, exist_ok=True)
    (out_dir / 'comparisons').mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        logging.error(f'Dataset directory not found: {data_dir}')
        return

    # Patch BF inside benchmark_inbolt so its infer_depth_m uses the D405 value.
    bi.BF = args.bf
    logging.info(f'Using BF = {args.bf:.3f} (fx_px * baseline_mm)')

    frame_indices = discover_frames(data_dir)
    if args.max_frames > 0:
        frame_indices = frame_indices[:args.max_frames]
    if not frame_indices:
        logging.error(f'No frames found in {data_dir}')
        return
    logging.info(f'Found {len(frame_indices)} frames in {data_dir}')

    # ── enumerate models that exist on disk ──────────────────────────────────
    model_specs = []
    if Path(args.original).exists():
        model_specs.append(('original', args.original))
    else:
        logging.warning(f'Original model not found: {args.original}')
    if Path(args.finetuned).exists():
        model_specs.append(('finetuned', args.finetuned))
    else:
        logging.warning(f'Fine-tuned model not found: {args.finetuned}')
    if not model_specs:
        logging.error('No models available — aborting')
        return

    method_names = ['depth_rs'] + [m for m, _ in model_specs]
    timings_ms: Dict[str, List[float]] = {m: [] for m, _ in model_specs}

    # ── 1) load each model in turn and dump per-frame depth to PNGs ──────────
    sub_dirs = {'original': 'depth_original', 'finetuned': 'depth_finetuned'}
    for mname, mpath in model_specs:
        sub_dir = out_dir / sub_dirs[mname]
        model = load_model(mpath)
        try:
            for i, idx in enumerate(frame_indices):
                left, right, _ = load_frame(data_dir, idx)
                if left is None or right is None:
                    logging.warning(f'Skipping frame {idx}: failed to load L/R')
                    continue
                t0 = time.monotonic()
                depth_m = infer_depth_m(model, left, right)
                timings_ms[mname].append((time.monotonic() - t0) * 1000.0)
                _save_depth_png_mm(depth_m, sub_dir / f'depth_{idx}.png')
                if (i + 1) % 5 == 0 or (i + 1) == len(frame_indices):
                    logging.info(f'[{mname}] {i + 1}/{len(frame_indices)} frames processed')
        finally:
            del model
            torch.cuda.empty_cache()

    # ── 2) build per-frame comparison figures from saved depth maps ──────────
    comparison_pngs: List[Path] = []
    for idx in frame_indices:
        left, _, depth_mm = load_frame(data_dir, idx)
        if left is None or depth_mm is None:
            continue
        frame_depths: Dict[str, np.ndarray] = {
            'depth_rs': depth_mm.astype(np.float32) / 1000.0,
        }
        for mname, _ in model_specs:
            d = _load_depth_png_m(out_dir / sub_dirs[mname] / f'depth_{idx}.png')
            if d.size > 0:
                frame_depths[mname] = d
        cmp_png = out_dir / 'comparisons' / f'compare_{idx}.png'
        _render_comparison(left, frame_depths, idx, cmp_png)
        comparison_pngs.append(cmp_png)

    html_path = _write_html_report(
        out_dir, frame_indices, method_names, timings_ms, comparison_pngs,
    )
    logging.info(f'Report written to {html_path}')
    logging.info(f'All outputs under {out_dir}')


if __name__ == '__main__':
    main()
