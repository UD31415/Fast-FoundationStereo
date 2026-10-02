"""Benchmark the Shazam MobileNet student against its Fast-FoundationStereo teacher on Pickle data.

Methods compared against the Pickle CAD GT (``DataSource.get_item_and_scene_projected``):
  - FFS teacher (GPU)
  - Shazam MobileNet student, network on GPU (``ShazamDepthEstimator.multiscale_disparity_mobile_net``)
  - Shazam MobileNet student, network on CPU (optional, ``--cpu_student``)
  - Shazam sequential fusion baseline (the merge of ``multiscale_disparity_features``)
  - RealSense hardware depth (``depth_img`` from the Pickle data)

The standard mm report of ``benchmark_pickle_shazam.py`` is generated, plus:
  - timing_breakdown.png       : student volume construction (CPU) vs network time vs teacher
  - teacher_agreement.png      : EPE / bad-1 / bad-3 / edge bad-3 of each method vs the teacher and vs GT (disparity px)
  - confidence_analysis.png    : sparsification curve and coverage / MAE vs confidence threshold
  - mobile_net_summary.csv     : the numbers behind these figures

Usage:
  cd /path/to/Fast-FoundationStereo
  python scripts/benchmark_mobile_net.py [--student weights/mobile_net/shazam_mobile_net.pth] [--max_frames 20]
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

from shazam_mobile_net import install_shazam_stubs, count_parameters
install_shazam_stubs()

# importing benchmark_pickle_shazam also sets CUDA_VISIBLE_DEVICES and the Agg backend
from benchmark_pickle_shazam import (
    ReportGeneratorMM, infer_depth_mm, load_model, compute_bin_mae_mm,
    PICKLE_EXCEL, FINETUNED_PATH, CLOSE_RANGE_THRESHOLD_MM, DEVICE,
)
import matplotlib.pyplot as plt
import numpy as np

import Utils as U
from scripts.data_manager_pickle import DataSource
from metrics import BenchmarkResults, FrameMetrics, compute_metrics, aggregate
from shazam_depth_estimator import ShazamDepthEstimator


# ── constants ─────────────────────────────────────────────────────────────────

STUDENT_PATH    = f'{code_dir}/../weights/mobile_net/shazam_mobile_net.pth'
DEFAULT_OUT     = f'{code_dir}/../reports/benchmark_mobile_net'
N_VIZ           = 12
CONF_THR        = 0.5
CONF_SAMPLES    = 20000     # pixels per frame kept for the confidence analysis

METHODS: Dict[str, Dict[str, str]] = {
    "ffs_teacher":    {"label": "FFS Teacher (Pickle fine-tuned)",  "color": "#e74c3c"},
    "mobile_net":     {"label": "Shazam MobileNet (net on GPU)",    "color": "#2980b9"},
    "mobile_net_cpu": {"label": "Shazam MobileNet (net on CPU)",    "color": "#5dade2"},
    "shazam":         {"label": "Shazam Sequential Fusion (CPU)",   "color": "#16a085"},
    "depth_rs":       {"label": "RealSense Hardware Depth",         "color": "#f39c12"},
    "pickle_gt":      {"label": "Pickle CAD GT (projected)",        "color": "#27ae60"},
}
GT_NAME, RS_NAME, TEACHER_NAME = "pickle_gt", "depth_rs", "ffs_teacher"
STUDENT_NAME, STUDENT_CPU_NAME, SHAZAM_NAME = "mobile_net", "mobile_net_cpu", "shazam"
AGREE_KEYS      = ['fill', 'epe', 'bad1', 'bad3', 'edge_bad3']


# ── helpers ───────────────────────────────────────────────────────────────────

def depth_to_disp(depth_mm, bf):
    disp            = np.zeros_like(depth_mm, dtype=np.float32)
    valid           = depth_mm > 0
    disp[valid]     = bf / depth_mm[valid]
    return disp


def disp_to_depth(disp, bf):
    depth           = np.zeros_like(disp, dtype=np.float32)
    valid           = disp > 0.5     # subpixel floor avoids divide-by-zero/overflow
    depth[valid]    = bf / disp[valid]
    return depth


def mean_dict(rows: List[dict]) -> dict:
    return {k: float(np.nanmean([r[k] for r in rows])) if rows else float('nan') for k in AGREE_KEYS}


# ── report ────────────────────────────────────────────────────────────────────

class ReportGeneratorMobileNet(ReportGeneratorMM):
    """ReportGeneratorMM plus the student / teacher performance analysis."""

    def __init__(self, results, stats, output_dir, analysis: dict) -> None:
        super().__init__(results, stats, output_dir)
        self._a = analysis

    def generate(self) -> None:
        fig_paths = [
            self._fig_depth_comparison(),
            self._fig_error_maps(),
            self._fig_coverage_heatmaps(),
            self._fig_distance_error_curve(),
            self._fig_error_histograms(),
            self._fig_summary_table(),
            self._fig_close_range_analysis(),
            self._fig_timing_bars(),
            self._fig_timing_breakdown(),
            self._fig_teacher_agreement(),
            self._fig_confidence_analysis(),
        ]
        self._write_json()
        self._write_html([p for p in fig_paths if p])
        self._write_csv()
        print(f"\nReport written to: {self._out / 'index.html'}")

    def _fig_timing_breakdown(self) -> str:
        t           = {k: float(np.mean(v)) for k, v in self._a['timing'].items() if v}
        bars        = [("FFS teacher (GPU)",        [t.get('ffs_teacher', 0.0)],              ["#e74c3c"]),
                       ("MobileNet (GPU net)",      [t.get('volumes', 0.0), t.get('net_gpu', 0.0)], ["#95a5a6", "#2980b9"])]
        if 'net_cpu' in t:
            bars.append(("MobileNet (CPU net)",     [t.get('volumes', 0.0), t['net_cpu']],    ["#95a5a6", "#5dade2"]))
        if 'shazam' in t:
            bars.append(("Shazam sequential (CPU)", [t['shazam']],                            ["#16a085"]))

        fig, ax     = plt.subplots(figsize=(9, 4.5))
        for i, (label, parts, colors) in enumerate(bars):
            left    = 0.0
            for val, col in zip(parts, colors):
                ax.barh(i, val, left=left, color=col, edgecolor="white")
                left += val
            ax.text(left, i, f"  {left:.0f} ms", va="center", fontsize=9)
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([b[0] for b in bars], fontsize=9)
        ax.set_xlabel("Mean time per frame (ms)")
        ax.barh([], [], color="#95a5a6", label="feature volumes (CPU)")
        ax.legend(fontsize=8, loc="lower right")
        p           = self._a['params']
        ax.set_title(f"Timing breakdown  •  params: teacher {p.get('ffs_teacher', 0):,}  student {p.get('mobile_net', 0):,}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "timing_breakdown.png")

    def _fig_teacher_agreement(self) -> str:
        agree       = self._a['agree']
        names       = [n for n in agree if agree[n]['teacher'] or agree[n]['gt']]
        if not names:
            return self._empty_fig("teacher_agreement.png", "No agreement data")
        keys        = ['epe', 'bad1', 'bad3', 'edge_bad3']
        fig, axes   = plt.subplots(2, len(keys), figsize=(4 * len(keys), 7))
        for r, ref in enumerate(['teacher', 'gt']):
            for c, k in enumerate(keys):
                ax      = axes[r, c]
                vals    = [mean_dict(agree[n][ref])[k] for n in names]
                colors  = [self._r.method_colors.get(n, "#888") for n in names]
                bars    = ax.bar(range(len(names)), vals, color=colors)
                ax.bar_label(bars, labels=[f"{v:.2f}" if k == 'epe' else f"{v:.1f}" for v in vals], fontsize=7, padding=2)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels([self._r.method_labels.get(n, n) for n in names], rotation=30, ha="right", fontsize=7)
                ax.set_title(f"{k} vs {'FFS teacher' if ref == 'teacher' else 'CAD GT'}" + (" (px)" if k == 'epe' else " (%)"), fontsize=9)
                ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Disparity agreement (bad-N = % pixels with error > N px)", fontsize=11)
        fig.tight_layout()
        return self._save(fig, "teacher_agreement.png")

    def _fig_confidence_analysis(self) -> str:
        cs          = self._a['conf_curves']
        if cs is None:
            return self._empty_fig("confidence_analysis.png", "No confidence data")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(cs['fractions'] * 100, cs['mae_by_conf'], marker="o", color="#2980b9", label="sorted by confidence")
        ax1.plot(cs['fractions'] * 100, cs['mae_oracle'], linestyle="--", color="#7f8c8d", label="oracle (sorted by error)")
        ax1.set_xlabel("Pixels kept (%)")
        ax1.set_ylabel("MAE vs GT (mm)")
        ax1.set_title("Sparsification curve (MobileNet student)")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        ax2.plot(cs['thresholds'], cs['coverage'], marker="o", color="#27ae60", label="coverage (%)")
        ax2.set_xlabel("Confidence threshold")
        ax2.set_ylabel("Coverage of GT pixels (%)", color="#27ae60")
        ax2.axvline(self._a['conf_thr'], color="red", linestyle=":", label=f"conf_thr={self._a['conf_thr']}")
        ax3         = ax2.twinx()
        ax3.plot(cs['thresholds'], cs['mae_at_thr'], marker="s", color="#c0392b", label="MAE (mm)")
        ax3.set_ylabel("MAE vs GT (mm)", color="#c0392b")
        ax2.set_title("Coverage / accuracy vs confidence threshold")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "confidence_analysis.png")

    def _write_csv(self) -> None:
        path        = self._out / "mobile_net_summary.csv"
        timing      = {k: float(np.mean(v)) if v else float('nan') for k, v in self._a['timing'].items()}
        total_ms    = {TEACHER_NAME: timing.get('ffs_teacher'), SHAZAM_NAME: timing.get('shazam'),
                       STUDENT_NAME: timing.get('volumes', 0) + timing.get('net_gpu', 0),
                       STUDENT_CPU_NAME: timing.get('volumes', 0) + timing.get('net_cpu', float('nan'))}
        with open(path, "w", newline="") as f:
            w       = csv.writer(f)
            w.writerow(["method", "label", "time_ms", "params"] +
                       [f"{k}_vs_teacher" for k in AGREE_KEYS] + [f"{k}_vs_gt" for k in AGREE_KEYS] +
                       ["mae_mm", "delta1", "coverage"])
            for name, a in self._a['agree'].items():
                at, ag  = mean_dict(a['teacher']), mean_dict(a['gt'])
                s       = self._stats.get(name)
                w.writerow([name, self._r.method_labels.get(name, name), f"{total_ms.get(name, float('nan')):.1f}",
                            self._a['params'].get(name.replace('_cpu', ''), '')] +
                           [f"{at[k]:.3f}" for k in AGREE_KEYS] + [f"{ag[k]:.3f}" for k in AGREE_KEYS] +
                           ([f"{s.mae_mean:.2f}", f"{s.delta1_mean:.2f}", f"{s.coverage_mean:.2f}"] if s else ["", "", ""]))
            w.writerow([])
            w.writerow(["timing_component", "mean_ms"])
            for k, v in timing.items():
                w.writerow([k, f"{v:.1f}"])
        logging.info(f"Summary CSV written to {path}")


def confidence_curves(conf, err):
    "sparsification (MAE of the most confident fraction) and coverage / MAE vs threshold"
    if conf.size == 0:
        return None
    fractions       = np.linspace(0.05, 1.0, 20)
    by_conf         = err[np.argsort(-conf)]
    by_err          = np.sort(err)
    n               = len(err)
    keep            = [max(1, int(round(f * n))) for f in fractions]
    thresholds      = np.linspace(0.0, 0.95, 20)
    return {
        'fractions':    fractions,
        'mae_by_conf':  np.array([by_conf[:k].mean() for k in keep]),
        'mae_oracle':   np.array([by_err[:k].mean() for k in keep]),
        'thresholds':   thresholds,
        'coverage':     np.array([100.0 * np.mean(conf >= t) for t in thresholds]),
        'mae_at_thr':   np.array([err[conf >= t].mean() if np.any(conf >= t) else np.nan for t in thresholds]),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir',      default=DEFAULT_OUT,  help='Output directory for the report')
    parser.add_argument('--pickle_excel', default=PICKLE_EXCEL, help='Pickle manifest Excel')
    parser.add_argument('--teacher',      default=FINETUNED_PATH, help='FFS teacher weights (full model pickle)')
    parser.add_argument('--student',      default=STUDENT_PATH, help='MobileNet student checkpoint (finetune_mobile_net.py)')
    parser.add_argument('--n_viz',        type=int,   default=N_VIZ, help='Frames saved for visual comparison')
    parser.add_argument('--conf_thr',     type=float, default=CONF_THR, help='Student confidence threshold')
    parser.add_argument('--cpu_student',  action='store_true', help='Also time the student network on CPU')
    parser.add_argument('--skip_shazam',  action='store_true', help='Skip the sequential Shazam baseline')
    parser.add_argument('--max_frames',   type=int,   default=None, help='Limit the number of frames')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── models ────────────────────────────────────────────────────────────────
    if not Path(args.student).exists():
        raise FileNotFoundError(f"Student checkpoint not found at {args.student}. Run scripts/finetune_mobile_net.py first.")
    teacher             = load_model(args.teacher)
    student             = ShazamDepthEstimator()
    student.load_mobile_net(args.student, device=str(DEVICE))
    student_cpu         = None
    if args.cpu_student:
        student_cpu     = ShazamDepthEstimator()
        student_cpu.load_mobile_net(args.student, device='cpu')
    shazam              = None if args.skip_shazam else ShazamDepthEstimator()
    params              = {TEACHER_NAME: count_parameters(teacher), STUDENT_NAME: count_parameters(student.mobile_net)}
    max_disparity       = student.mobile_net.cfg['D']
    logging.info(f"Parameters : teacher {params[TEACHER_NAME]:,}  student {params[STUDENT_NAME]:,}")

    disp_methods        = [TEACHER_NAME, STUDENT_NAME] + ([STUDENT_CPU_NAME] if student_cpu else []) + ([SHAZAM_NAME] if shazam else [])
    active_methods      = [GT_NAME, RS_NAME] + disp_methods

    # ── dataset ───────────────────────────────────────────────────────────────
    source              = DataSource(train_mode=False)
    n                   = source.init_directory(excel_path=args.pickle_excel)
    n                   = n if args.max_frames is None else min(n, args.max_frames)
    logging.info(f"Using {n} samples from {args.pickle_excel}")
    if n == 0:
        logging.error("No samples found — check --pickle_excel path")
        return

    # ── accumulators ──────────────────────────────────────────────────────────
    all_metrics, viz_frames, valid_acc = [], [], {}
    dist_bin_mae        = {m: [] for m in active_methods}
    close_range_valid   = {m: [] for m in active_methods}
    method_ms           = {m: [] for m in disp_methods}
    timing              = {'ffs_teacher': [], 'volumes': [], 'net_gpu': [], 'net_cpu': [], 'shazam': []}
    agree               = {m: {'teacher': [], 'gt': []} for m in disp_methods + [RS_NAME]}
    conf_pool, err_pool = [], []
    rng                 = np.random.default_rng(0)
    H = W = None
    n_done              = 0

    for idx in range(n):
        data            = source.get_item_and_scene_projected(idx)
        left, right     = data['ir_left_img'], data['ir_right_img']
        gt_mm           = data['depth_cad_projected'].astype(np.float32)
        rs_mm           = data['depth_img'].astype(np.float32)
        bf              = float(data['bf'])
        if gt_mm.shape != rs_mm.shape:
            logging.warning(f"Item {idx}: depth_cad_projected {gt_mm.shape} != depth_img {rs_mm.shape}; skipping")
            continue
        if H is None:
            H, W        = gt_mm.shape[:2]
            valid_acc   = {m: np.zeros((H, W), np.float32) for m in active_methods}

        left_g          = (left[..., 0] if left.ndim == 3 else left).astype(np.float32)
        right_g         = (right[..., 0] if right.ndim == 3 else right).astype(np.float32)
        frame_depths    = {GT_NAME: gt_mm, RS_NAME: rs_mm}
        frame_disps     = {RS_NAME: depth_to_disp(rs_mm, bf)}

        # teacher (GPU)
        t0              = time.monotonic()
        frame_depths[TEACHER_NAME] = infer_depth_mm(teacher, left, right, bf)
        timing['ffs_teacher'].append((time.monotonic() - t0) * 1000.0)
        method_ms[TEACHER_NAME].append(timing['ffs_teacher'][-1])
        frame_disps[TEACHER_NAME] = depth_to_disp(frame_depths[TEACHER_NAME], bf)

        # student : feature volumes once (CPU), network on GPU and optionally CPU
        t0              = time.monotonic()
        volumes         = student.multiscale_feature_volumes(left_g, right_g, max_disparity=max_disparity)
        timing['volumes'].append((time.monotonic() - t0) * 1000.0)

        disp_s, conf_s  = student.multiscale_disparity_mobile_net(left_g, right_g, conf_thr=args.conf_thr, device=str(DEVICE), volumes=volumes)
        timing['net_gpu'].append(student.timing['net_ms'])
        method_ms[STUDENT_NAME].append(timing['volumes'][-1] + timing['net_gpu'][-1])
        frame_disps[STUDENT_NAME] = disp_s
        disp_raw        = student.disp_raw

        if student_cpu:
            disp_c, _   = student_cpu.multiscale_disparity_mobile_net(left_g, right_g, conf_thr=args.conf_thr, device='cpu', volumes=volumes)
            timing['net_cpu'].append(student_cpu.timing['net_ms'])
            method_ms[STUDENT_CPU_NAME].append(timing['volumes'][-1] + timing['net_cpu'][-1])
            frame_disps[STUDENT_CPU_NAME] = disp_c
        del volumes

        # sequential Shazam baseline (same result as multiscale_disparity_features, streamed)
        if shazam:
            t0          = time.monotonic()
            frame_disps[SHAZAM_NAME] = shazam.multiscale_disparity_fusion(left_g, right_g, fusion='sequential', max_disparity=max_disparity).astype(np.float32)
            timing['shazam'].append((time.monotonic() - t0) * 1000.0)
            method_ms[SHAZAM_NAME].append(timing['shazam'][-1])
            plt.close('all')

        for m in disp_methods:
            if m != TEACHER_NAME:
                frame_depths[m] = disp_to_depth(frame_disps[m], bf)

        # ── disparity agreement vs teacher and vs GT ──────────────────────────
        gt_disp         = depth_to_disp(gt_mm, bf)
        for m in agree:
            if m != TEACHER_NAME:
                agree[m]['teacher'].append(student.evaluate_disparity(frame_disps[m], frame_disps[TEACHER_NAME], max_disparity))
            agree[m]['gt'].append(student.evaluate_disparity(frame_disps[m], gt_disp, max_disparity))

        # ── confidence analysis (unthresholded student disparity) ─────────────
        valid           = (gt_mm > 0) & (disp_raw > 0.5)
        valid[:, :max_disparity] = False
        ys, xs          = np.nonzero(valid)
        if ys.size:
            sel         = rng.choice(ys.size, size=min(CONF_SAMPLES, ys.size), replace=False)
            ys, xs      = ys[sel], xs[sel]
            conf_pool.append(conf_s[ys, xs])
            err_pool.append(np.abs(bf / disp_raw[ys, xs] - gt_mm[ys, xs]))

        # ── per-frame depth metrics (mm) ──────────────────────────────────────
        gt_close_mask   = (gt_mm > 0) & (gt_mm < CLOSE_RANGE_THRESHOLD_MM)
        n_close         = int(gt_close_mask.sum())
        for m in active_methods:
            pred        = frame_depths[m]
            valid_acc[m] += (pred > 0).astype(np.float32)
            if m == GT_NAME:
                fm      = FrameMetrics(GT_NAME, 0.0, 0.0, 0.0, 100.0, float((pred > 0).mean()) * 100.0, 0.0, mae_pen=0.0, mre_pen=0.0)
            elif m == RS_NAME:
                fm      = compute_metrics(pred, gt_mm, elapsed_ms=0.0, method_name=RS_NAME)
            else:
                fm      = compute_metrics(pred, gt_mm, method_ms[m][-1], m)
            all_metrics.append(fm)
            dist_bin_mae[m].append(compute_bin_mae_mm(pred, gt_mm))
            close_range_valid[m].append(float((pred[gt_close_mask] > 0).mean()) * 100.0 if n_close > 0 else 0.0)

        if idx < args.n_viz:
            viz_frames.append({k: v.copy() for k, v in frame_depths.items()})
        n_done         += 1
        if (idx + 1) % 10 == 0 or (idx + 1) == n:
            logging.info(f"  {idx + 1}/{n} frames  teacher {timing['ffs_teacher'][-1]:.0f} ms  "
                         f"student vol {timing['volumes'][-1]:.0f} + net {timing['net_gpu'][-1]:.0f} ms")

    for m in active_methods:
        valid_acc[m] /= max(n_done, 1)

    # ── aggregate ─────────────────────────────────────────────────────────────
    mean_timing         = {m: float(np.mean(ts)) if ts else 0.0 for m, ts in method_ms.items()}
    mean_timing[GT_NAME] = 0.0
    mean_timing[RS_NAME] = 1000.0 / 30.0

    method_configs = {
        TEACHER_NAME:   {"model_path": args.teacher, "valid_iters": "8"},
        STUDENT_NAME:   {"model_path": args.student, "max_disp": str(max_disparity), "engine_resolution": f"{W}x{H}"},
        SHAZAM_NAME:    {"estimator": "ShazamDepthEstimator.multiscale_disparity_fusion(fusion='sequential')", "max_disp": str(max_disparity)},
        RS_NAME:        {"source": "RealSense hardware depth (depth_img, ~30 FPS)"},
        GT_NAME:        {"source": "Pickle CAD-rendered ground-truth depth via DataSource.get_item_and_scene_projected"},
    }
    if student_cpu:
        method_configs[STUDENT_CPU_NAME] = dict(method_configs[STUDENT_NAME], device="cpu")

    results = BenchmarkResults(
        method_names=active_methods,
        method_labels={m: METHODS[m]["label"] for m in active_methods},
        method_colors={m: METHODS[m]["color"] for m in active_methods},
        ground_truth_name=GT_NAME,
        n_frames=n_done,
        width=W,
        height=H,
        all_metrics=all_metrics,
        viz_frames=viz_frames,
        coverage_maps=valid_acc,
        dist_bin_mae=dist_bin_mae,
        close_range_valid=close_range_valid,
        source=f"Pickle scene-capture  •  {args.pickle_excel}  •  student={Path(args.student).name}  •  conf_thr={args.conf_thr}",
        method_configs={m: c for m, c in method_configs.items() if m in active_methods},
    )
    stats = aggregate(results, mean_timing)
    if RS_NAME in stats:
        stats[RS_NAME].fps_mean = 30.0

    conf_all            = np.concatenate(conf_pool) if conf_pool else np.zeros(0)
    err_all             = np.concatenate(err_pool) if err_pool else np.zeros(0)
    analysis = {
        'timing':       {k: v for k, v in timing.items() if v},
        'params':       params,
        'agree':        agree,
        'conf_thr':     args.conf_thr,
        'conf_curves':  confidence_curves(conf_all, err_all),
    }

    for m, a in agree.items():
        at, ag = mean_dict(a['teacher']), mean_dict(a['gt'])
        logging.info(f"{METHODS[m]['label']:34s} vs teacher epe={at['epe']:.2f} bad3={at['bad3']:.1f}%  "
                     f"vs GT epe={ag['epe']:.2f} bad3={ag['bad3']:.1f}% fill={ag['fill']:.1f}%")

    ReportGeneratorMobileNet(results, stats, out_dir, analysis).generate()


if __name__ == '__main__':
    main()
