"""Benchmark stereo depths on Pickle data and compare them with an ICP-based pose metric.

This script combines the depth-evaluation flow from
``scripts/benchmark_pickle_trained_pickle.py`` with the pose/ICP workflow in
``scripts/data_manager_pickle.py``.

For each evaluated depth map it:
  1. compares the predicted depth against the Pickle CAD-rendered ground truth,
  2. back-projects the depth into a 3D point cloud using the camera intrinsics,
  3. filters the point cloud to the CAD bounding box,
  4. runs ICP against the CAD point cloud, and
  5. reports the ICP fitness / RMSE as an additional metric for comparing methods.

The resulting summary is written to CSV and a small scatter plot is saved under
``--out_dir`` for quick comparison of depth error vs ICP alignment quality.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")
sys.path.append(code_dir)

# Driver 580 / CUDA 13.0 does not enumerate devices without CUDA_VISIBLE_DEVICES set.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        indices = ",".join(line.strip() for line in out.splitlines() if line.strip())
        if indices:
            os.environ["CUDA_VISIBLE_DEVICES"] = indices
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d

from core.utils.utils import InputPadder
import Utils as U
from scripts.data_manager_pickle import (
    DataSource,
    cad_bbox_from_pcd,
    filter_by_cad_bbox,
)
from metrics import (
    BenchmarkResults,
    FrameMetrics,
    aggregate,
    compute_metrics,
)
from scripts.benchmark_pickle_trained_pickle import ReportGeneratorMM


PICKLE_EXCEL = (
    r"\\svm.realsenseai.com\RealSense_Validation\VIDB\Public\Stavush\Pickle\Data\data for model training 25_6_26"
    r"\data_25_06.xlsx"
)
ORIGINAL_PATH = f"{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth"
FINETUNED_PATH = f"{code_dir}/../weights/23-36-37/model_finetuned_pickle_260625_epoch_010.pth"
DEFAULT_OUT = f"{code_dir}/../reports/benchmark_pickle_pose"
PROJECTION_METHOD = "raycast"
ITERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS: Dict[str, Dict[str, str]] = {
    "original": {"label": "FFS Original", "color": "#2980b9"},
    "finetuned": {"label": "FFS Fine-tuned (Pickle)", "color": "#e74c3c"},
    "depth_rs": {"label": "RealSense Hardware Depth", "color": "#f39c12"},
    "pickle_gt": {"label": "Pickle CAD GT (projected)", "color": "#27ae60"},
}
GT_NAME = "pickle_gt"
RS_NAME = "depth_rs"
CLOSE_RANGE_THRESHOLD_MM = 550.0
DIST_BINS_MM: List[Tuple[float, float]] = [
    (0.0, 200.0),
    (200.0, 300.0),
    (300.0, 400.0),
    (400.0, 500.0),
    (500.0, 600.0),
    (600.0, 700.0),
    (700.0, 800.0),
    (800.0, 900.0),
    (900.0, 1000.0),
]
BIN_LABELS_MM = ["0–200 mm", "200–300 mm", "300–400 mm", "400–500 mm", "500–600 mm", "600–700 mm", "700–800 mm", "800–900 mm", "900–1000 mm"]
BIN_CENTERS_MM = [100.0, 250.0, 350.0, 450.0, 550.0, 650.0, 750.0, 850.0, 950.0]


def _preprocess_ir(left: np.ndarray, right: np.ndarray):
    """Convert uint8/uint16 IR images to CUDA float tensors (3-channel pseudo-RGB)."""
    left = np.clip(left.astype(np.float32), 0, 255)
    right = np.clip(right.astype(np.float32), 0, 255)
    left = np.stack([left, left, left], axis=-1)
    right = np.stack([right, right, right], axis=-1)
    left_t = torch.as_tensor(left).float()[None].permute(0, 3, 1, 2).to(DEVICE)
    right_t = torch.as_tensor(right).float()[None].permute(0, 3, 1, 2).to(DEVICE)
    return left_t, right_t


@torch.no_grad()
def infer_depth_mm(model, left: np.ndarray, right: np.ndarray, bf: float) -> np.ndarray:
    """Run stereo inference on an IR pair and return a depth map in millimetres."""
    left_t, right_t = _preprocess_ir(left, right)
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)

    with torch.amp.autocast("cuda", enabled=True, dtype=U.AMP_DTYPE):
        disp = model.forward(left_t, right_t, iters=ITERS, test_mode=True)

    disp = padder.unpad(disp.float())
    disp_np = disp.cpu().numpy().reshape(left.shape[:2]).clip(0, None)

    depth_mm = np.zeros_like(disp_np, dtype=np.float32)
    valid = disp_np > 0
    depth_mm[valid] = bf / disp_np[valid]
    return depth_mm


def load_model(path: str):
    logging.info(f"Loading model from {path}")
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.cuda().eval()
    return model


def compute_bin_mae_mm(pred_mm: np.ndarray, gt_mm: np.ndarray, edge_mask: np.ndarray) -> List[float]:
    """MAE (mm) per distance bin; returns NaN for bins with no valid GT pixels."""
    result = []
    for lo, hi in DIST_BINS_MM:
        mask = (gt_mm >= lo) & (gt_mm < hi) & (gt_mm > 0) & (pred_mm > 0) & edge_mask
        if mask.sum() == 0:
            result.append(float("nan"))
        else:
            result.append(float(np.abs(pred_mm[mask] - gt_mm[mask]).mean()))
    return result


def depth_to_point_cloud(depth_mm: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a depth image to a camera-frame point cloud in metres."""
    depth_mm = np.asarray(depth_mm, dtype=np.float32)
    if depth_mm.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    h, w = depth_mm.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    valid = depth_mm > 0
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    z = depth_mm[valid] / 1000.0
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts


def compute_icp_metric(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud) -> Dict[str, float]:
    """Run ICP against the CAD point cloud and return a compact pose quality summary."""
    if source_pcd is None or target_pcd is None:
        return {"fitness": float("nan"), "inlier_rmse": float("nan"), "translation_m": float("nan"), "rotation_deg": float("nan")}

    src_points = np.asarray(source_pcd.points, dtype=np.float64)
    tgt_points = np.asarray(target_pcd.points, dtype=np.float64)
    if src_points.size == 0 or tgt_points.size == 0:
        return {"fitness": 0.0, "inlier_rmse": float("inf"), "translation_m": float("inf"), "rotation_deg": float("inf")}

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(tgt_points)

    init = np.eye(4)
    threshold = 0.01
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    rotation = result.transformation[:3, :3]
    trace = np.trace(rotation)
    rotation_deg = float(np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))
    translation_m = float(np.linalg.norm(result.transformation[:3, 3]))
    return {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "translation_m": translation_m,
        "rotation_deg": rotation_deg,
    }


def build_icp_summary_for_depth(depth_mm: np.ndarray, item: Dict[str, Any], source: DataSource) -> Dict[str, float]:
    """Back-project a depth map to a point cloud and compare it to the CAD cloud with ICP."""
    if item.get("cad_pcd_aligned") is None:
        return {"fitness": float("nan"), "inlier_rmse": float("nan"), "translation_m": float("nan"), "rotation_deg": float("nan")}

    intrinsics, _ = source.get_intrinsics_matrix(item)
    pts = depth_to_point_cloud(depth_mm, intrinsics)
    if pts.shape[0] == 0:
        return {"fitness": 0.0, "inlier_rmse": float("inf"), "translation_m": float("inf"), "rotation_deg": float("inf")}

    pcd_depth = o3d.geometry.PointCloud()
    pcd_depth.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    cad_pcd = item["cad_pcd_aligned"]
    try:
        bbox = cad_bbox_from_pcd(cad_pcd)
        pcd_filtered = filter_by_cad_bbox(pcd_depth, bbox)
    except Exception:
        pcd_filtered = pcd_depth

    if len(pcd_filtered.points) < 10:
        pcd_filtered = pcd_depth

    return compute_icp_metric(pcd_filtered, cad_pcd)


def save_summary_csv(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    import csv

    out_path = out_dir / "summary_metrics.csv"
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return out_path

    fieldnames = [
        "method",
        "mae_mm",
        "rmse_mm",
        "mre_pct",
        "delta1_pct",
        "coverage_pct",
        "fps",
        "fitness",
        "inlier_rmse_m",
        "translation_m",
        "rotation_deg",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return out_path


def save_scatter_plot(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        color = METHODS.get(row["method"], {}).get("color", "#888")
        ax.scatter(
            row["mae_mm"],
            row["inlier_rmse_m"],
            s=120,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=row["method"],
        )
        ax.annotate(row["method"], (row["mae_mm"], row["inlier_rmse_m"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("MAE (mm)")
    ax.set_ylabel("ICP RMSE (m)")
    ax.set_title("Depth Error vs ICP Alignment")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "depth_vs_icp_scatter.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=DEFAULT_OUT, help="Output directory for the benchmark report")
    parser.add_argument("--pickle_excel", default=PICKLE_EXCEL, help="Path to the Pickle manifest Excel")
    parser.add_argument("--original", default=ORIGINAL_PATH, help="Path to the original model weights")
    parser.add_argument("--finetuned", default=FINETUNED_PATH, help="Path to the fine-tuned model weights")
    parser.add_argument("--n_frames", type=int, default=20, help="Number of frames to evaluate")
    parser.add_argument("--n_viz", type=int, default=12, help="Number of visualisation frames saved for the report")
    parser.add_argument("--projection", default=PROJECTION_METHOD, choices=("splat", "raycast", "open3d"), help="Projection method for the GT depth")
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, Any] = {}
    for name, path in (("finetuned", args.finetuned), ("original", args.original)):
        if not Path(path).exists():
            logging.warning("Skipping %s model: file not found at %s", name, path)
            continue
        models[name] = load_model(path)

    if not models:
        logging.warning("No model weights were loaded; evaluating only GT and hardware depth baselines.")

    active_methods = [GT_NAME, RS_NAME] + list(models.keys())

    source = DataSource(train_mode=False)
    n = source.init_directory(excel_path=args.pickle_excel)
    logging.info(f"Indexed {n} captures from {args.pickle_excel}")
    if n <= 0:
        raise RuntimeError("No samples found; check --pickle_excel")

    frame_count = min(n, max(1, args.n_frames))
    indices = np.random.choice(n, size=frame_count, replace=False)
    logging.info(f"Evaluating {len(indices)} frames")

    rows: List[Dict[str, Any]] = []
    per_method_frames: Dict[str, List[Dict[str, Any]]] = {name: [] for name in active_methods}
    all_metrics: List[FrameMetrics] = []
    viz_frames: List[Dict[str, np.ndarray]] = []
    valid_acc: Dict[str, np.ndarray] = {}
    dist_bin_mae: Dict[str, List[List[float]]] = {name: [] for name in active_methods}
    edge_dist_bin_mae: Dict[str, List[List[float]]] = {name: [] for name in active_methods}
    close_range_valid: Dict[str, List[float]] = {name: [] for name in active_methods}
    timing_ms_raw: Dict[str, List[float]] = {name: [] for name in models}
    H = W = None

    for frame_idx, idx in enumerate(indices):
        item = source.get_item_and_scene_projected(int(idx))
        gt_mm = item["depth_scene_projected"].astype(np.float32)
        rs_mm = item["depth_img"].astype(np.float32)
        bf = float(item["bf"])
        edge_mask = item["edge_mask"].astype(bool)

        if H is None:
            H, W = gt_mm.shape[:2]
            for method_name in active_methods:
                valid_acc[method_name] = np.zeros((H, W), np.float32)

        frame_depths = {GT_NAME: gt_mm, RS_NAME: rs_mm}
        timing_ms: Dict[str, float] = {}

        for mname, model in models.items():
            t0 = time.monotonic()
            frame_depths[mname] = infer_depth_mm(model, item["ir_left_img"], item["ir_right_img"], bf)
            timing_ms[mname] = (time.monotonic() - t0) * 1000.0
            timing_ms_raw[mname].append(timing_ms[mname])

        gt_close_mask = (gt_mm > 0) & (gt_mm < CLOSE_RANGE_THRESHOLD_MM)
        n_close = int(gt_close_mask.sum())

        for method_name in active_methods:
            pred_depth = frame_depths[method_name]
            valid_acc[method_name] += (pred_depth > 0).astype(np.float32)

            metrics = compute_metrics(pred_depth, gt_mm, timing_ms.get(method_name, 0.0), method_name)
            all_metrics.append(metrics)
            dist_bin_mae[method_name].append(compute_bin_mae_mm(pred_depth, gt_mm, np.ones_like(gt_mm, dtype=bool)))
            edge_dist_bin_mae[method_name].append(compute_bin_mae_mm(pred_depth, gt_mm, edge_mask))

            if n_close > 0:
                close_cov = float((pred_depth[gt_close_mask] > 0).mean()) * 100.0
            else:
                close_cov = 0.0
            close_range_valid[method_name].append(close_cov)

            icp_metrics = build_icp_summary_for_depth(pred_depth, item, source)
            frame_row = {
                "method": method_name,
                "mae_mm": metrics.mae,
                "rmse_mm": metrics.rmse,
                "mre_pct": metrics.mre * 100.0,
                "delta1_pct": metrics.delta1,
                "coverage_pct": metrics.coverage,
                "fps": float("inf") if metrics.time_ms <= 0 else 1000.0 / metrics.time_ms,
                "fitness": icp_metrics.get("fitness", float("nan")),
                "inlier_rmse_m": icp_metrics.get("inlier_rmse", float("nan")),
                "translation_m": icp_metrics.get("translation_m", float("nan")),
                "rotation_deg": icp_metrics.get("rotation_deg", float("nan")),
            }
            rows.append(frame_row)
            per_method_frames[method_name].append(frame_row)

        if frame_idx < args.n_viz:
            viz_frames.append({name: frame_depths[name].copy() for name in active_methods})

    if not rows:
        raise RuntimeError("No rows were produced; the benchmark did not complete")

    for method_name in active_methods:
        valid_acc[method_name] /= max(len(indices), 1)

    summary_rows: List[Dict[str, Any]] = []
    for method_name in active_methods:
        frames = per_method_frames.get(method_name, [])
        if not frames:
            continue
        summary_rows.append(
            {
                "method": method_name,
                "mae_mm": float(np.mean([r["mae_mm"] for r in frames])),
                "rmse_mm": float(np.mean([r["rmse_mm"] for r in frames])),
                "mre_pct": float(np.mean([r["mre_pct"] for r in frames])),
                "delta1_pct": float(np.mean([r["delta1_pct"] for r in frames])),
                "coverage_pct": float(np.mean([r["coverage_pct"] for r in frames])),
                "fps": float(np.mean([r["fps"] for r in frames])) if np.isfinite([r["fps"] for r in frames]).any() else float("inf"),
                "fitness": float(np.mean([r["fitness"] for r in frames])),
                "inlier_rmse_m": float(np.mean([r["inlier_rmse_m"] for r in frames])),
                "translation_m": float(np.mean([r["translation_m"] for r in frames])),
                "rotation_deg": float(np.mean([r["rotation_deg"] for r in frames])),
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: (row["inlier_rmse_m"], row["mae_mm"]))
    summary_csv = save_summary_csv(summary_rows, out_dir)
    scatter_path = save_scatter_plot(summary_rows, out_dir)

    mean_timing: Dict[str, float] = {
        method_name: float(np.mean(timing_ms_raw.get(method_name, []))) if timing_ms_raw.get(method_name) else 0.0
        for method_name in models.keys()
    }
    mean_timing[GT_NAME] = 0.0
    mean_timing[RS_NAME] = 1000.0 / 30.0

    method_configs = {
        "original": {"model_path": args.original},
    }
    if "finetuned" in models:
        method_configs["finetuned"] = {"model_path": args.finetuned}
    method_configs[RS_NAME] = {"source": "RealSense hardware depth (depth_img, ~30 FPS)"}
    method_configs[GT_NAME] = {"source": f"Pickle CAD-rendered ground-truth depth via get_item_and_scene_projected(method={args.projection!r})"}

    results = BenchmarkResults(
        method_names=active_methods,
        method_labels={method_name: METHODS[method_name]["label"] for method_name in active_methods},
        method_colors={method_name: METHODS[method_name]["color"] for method_name in active_methods},
        ground_truth_name=GT_NAME,
        n_frames=len(indices),
        width=W,
        height=H,
        all_metrics=all_metrics,
        viz_frames=viz_frames,
        coverage_maps=valid_acc,
        dist_bin_mae=dist_bin_mae,
        close_range_valid=close_range_valid,
        source=f"Pickle scene-capture • {args.pickle_excel} • projection={args.projection}",
        method_configs=method_configs,
    )

    stats = aggregate(results, mean_timing)
    if RS_NAME in stats:
        stats[RS_NAME].fps_mean = 30.0

    edge_mae_mean: Dict[str, float] = {}
    for method_name, vals in edge_dist_bin_mae.items():
        if vals:
            arr = np.array(vals, dtype=float)
            edge_mae_mean[method_name] = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")
        else:
            edge_mae_mean[method_name] = float("nan")

    reporter = ReportGeneratorMM(
        results,
        stats,
        out_dir,
        edge_mae_per_method=edge_mae_mean,
        edge_dist_bin_mae=edge_dist_bin_mae,
    )
    reporter.generate()

    logging.info("Saved summary CSV to %s", summary_csv)
    logging.info("Saved scatter plot to %s", scatter_path)

    print("\nDepth + ICP benchmark summary")
    print("-" * 96)
    print(f"{'method':<14} {'MAE (mm)':>10} {'ICP RMSE (m)':>14} {'fitness':>10} {'delta1 (%)':>12}")
    print("-" * 96)
    for row in summary_rows:
        print(
            f"{row['method']:<14} {row['mae_mm']:>10.1f} {row['inlier_rmse_m']:>14.4f} {row['fitness']:>10.3f} {row['delta1_pct']:>12.1f}"
        )
    print("-" * 96)


if __name__ == "__main__":
    main()
