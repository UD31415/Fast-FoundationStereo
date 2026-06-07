"""
Dataset management for the ISAC reflective_test capture dataset.

Folder layout expected under the dataset root (default
``C:\\Work\\Data\\reflective_test``)::

    reflective_test/
        episode_XX/
            <view>/                     # view in {front, overhead, side, wrist}
                ir_left/frame_NNNN.png       # uint8  (H, W)
                ir_right/frame_NNNN.png      # uint8  (H, W)
                depth_left/frame_NNNN.png    # uint16 (H, W) depth in mm, left camera frame
                depth_right/frame_NNNN.png   # uint16 (H, W) depth in mm, right camera frame
                depth_tyzx/frame_NNNN.png    # uint16 (H, W) depth in mm, primary depth
                depth_tyzx_viz/frame_NNNN.png   # uint8 RGB visualization
                depth_viz_left/frame_NNNN.png   # uint8 RGB visualization
                depth_viz_right/frame_NNNN.png  # uint8 RGB visualization
                rgb_left/frame_NNNN.png      # uint8 RGB
                rgb_right/frame_NNNN.png     # uint8 RGB

This module mirrors the structure of ``data_manager_pickle.py``:

* ``DataSource.init_directory(root)`` enumerates all samples on disk and
  caches per-sample file paths.
* ``DataSource.get_item(index)`` returns one fully loaded sample
  (IR pair, depth, and a back-projected Open3D point cloud).
* ``TestDataSource`` provides ``unittest`` tests that visualize the data
  (Matplotlib for the images, Open3D for the point cloud).
"""

from __future__ import annotations

import logging as log
import os
import re
import unittest
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


log.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log.INFO)


# Default dataset root on disk.
DEFAULT_ROOT = r"C:\Work\Data\Depth\reflective_test"
DEFAULT_ROOT = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/isaac_datasets/reflective_test'


# Known view names. Sub-folders not present on disk are silently skipped.
KNOWN_VIEWS = ("wrist",)# ("front", "overhead", "side", "wrist")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_RE = re.compile(r"frame_(\d+)\.png$", re.IGNORECASE)


def _frame_index(name: str) -> int:
    """Return the integer NNNN from a ``frame_NNNN.png`` filename, or -1."""
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else -1


def depth_to_point_cloud(
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color_img: Optional[np.ndarray] = None,
    depth_scale_m: float = 1.0e-3,
    z_min_m: float = 0.05,
    z_max_m: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Back-project a uint16 depth (mm) image to an Open3D point cloud.

    Parameters
    ----------
    depth_mm : ``(H, W)`` array of depth in millimetres (uint16/float).
    fx, fy, cx, cy : pinhole intrinsics in pixels.
    color_img : optional ``(H, W, 3)`` BGR image used to colour the cloud.
    depth_scale_m : multiplier from raw depth units to metres (mm -> m).
    z_min_m, z_max_m : valid depth range in metres.
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"depth_mm must be 2D, got shape {depth_mm.shape}")

    h, w = depth_mm.shape
    z = depth_mm.astype(np.float32) * float(depth_scale_m)

    valid = (z > z_min_m) & (z < z_max_m) & np.isfinite(z)
    if not np.any(valid):
        return o3d.geometry.PointCloud()

    vs, us = np.nonzero(valid)
    zs = z[vs, us]
    xs = (us.astype(np.float32) - cx) * zs / fx
    ys = (vs.astype(np.float32) - cy) * zs / fy

    xyz = np.stack([xs, ys, zs], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if color_img is not None and color_img.ndim == 3 and color_img.shape[:2] == (h, w):
        # OpenCV is BGR; Open3D expects RGB in [0, 1].
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        colors = rgb[vs, us].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

# ------------------------------------------------------------------
# Masking
# ------------------------------------------------------------------

def estimate_normals_from_depth_map(depth_map):
    """
    Estimates the surface normal vector for each pixel in a depth map
    using the image gradient (Sobel operator).

    Args:
        depth_map (np.ndarray): A single-channel depth image (e.g., CV_32F or CV_64F).
                                Depth values must be in a consistent metric (e.g., meters).

    Returns:
        np.ndarray: A 3-channel image (H, W, 3) where each pixel contains the
                    (nx, ny, nz) unit normal vector, as CV_32F.
    """
    # 1. Convert to CV_32F for accurate gradient calculation
    if depth_map.dtype != np.float32:
        depth_map = depth_map.astype(np.float32)

    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)   

    # 2. Calculate Derivatives using Sobel Operator (Gradient)
    # The kernel size 'ksize=1' is often preferred for depth maps as it corresponds 
    # to a 3x1 or 1x3 kernel, providing a close approximation of the derivative.
    ksize = 1 
    
    # Calculate dz/du (gradient in X/horizontal direction)
    # dx=1, dy=0
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=ksize, borderType=cv2.BORDER_DEFAULT)
    
    # Calculate dz/dv (gradient in Y/vertical direction)
    # dx=0, dy=1
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=ksize, borderType=cv2.BORDER_DEFAULT)

    # 3. Construct the Normal Vector Components
    # The normal vector is proportional to n = (-dz/du, -dz/dv, 1)
    
    # Reshape the gradients to (H, W, 1) for stacking
    grad_x = grad_x[:, :, np.newaxis]
    grad_y = grad_y[:, :, np.newaxis]
    
    # Create the 'z' component of the direction vector, which is always 1
    # np.ones_like creates an array with the same shape and type as the gradient arrays
    z_component = np.ones_like(grad_x)

    # Stack the components to create the direction vector (H, W, 3)
    # The X and Y gradients are negated: -dz/du and -dz/dv
    direction_vectors = np.concatenate((-grad_x, -grad_y, z_component), axis=2)

    # 4. Normalize the Direction Vectors
    # Calculate the magnitude (Euclidean norm) of each (nx, ny, nz) vector
    # axis=2 computes the norm across the 3 channels
    magnitude = np.linalg.norm(direction_vectors, axis=2, keepdims=True)
    
    # Use np.divide and np.where to prevent division by zero for magnitude=0
    # Set normals to (0, 0, 0) or another placeholder where magnitude is zero (flat or invalid depth)
    normals = np.divide(direction_vectors, magnitude, out=np.zeros_like(direction_vectors), where=magnitude != 0)

    return normals  


def create_object_mask(
    depth_gt: np.ndarray,
    gradient_threshold: float = 10.0,
    min_depth: float = 1.0,
    max_depth: Optional[float] = None,
    min_object_size: int = 0,
    connectivity: int = 4,
) -> np.ndarray:
    """Segment a depth image into objects using local depth gradients.

    Two neighbouring pixels are considered to belong to the same object when
    both are valid (within ``[min_depth, max_depth]``) and the absolute depth
    difference between them is at most ``gradient_threshold`` (same units as
    ``depth_gt``, typically millimetres). The function walks every pixel,
    builds an adjacency graph over those neighbour links, and assigns one
    label per connected component.

    Parameters
    ----------
    depth_gt : ``(H, W)`` ground-truth depth image (uint16/float).
    gradient_threshold : maximum allowed |Δdepth| between neighbours to keep
        them in the same component.
    min_depth, max_depth : valid depth range. Pixels outside this range are
        treated as background and assigned label 0.
    min_object_size : drop components with fewer pixels than this (re-labelled
        as background 0). Set to 0 to keep all components.
    connectivity : ``4`` (default) or ``8`` neighbourhood.

    Returns
    -------
    labels : ``(H, W)`` ``int32`` mask. ``0`` is background; valid object
        pixels carry the same positive integer if and only if they belong to
        the same connected component.
    """
    if depth_gt.ndim != 2:
        raise ValueError(f"depth_gt must be 2D, got shape {depth_gt.shape}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    h, w = depth_gt.shape
    depth = depth_gt.astype(np.float32)

    valid = depth >= float(min_depth)
    if max_depth is not None:
        valid &= depth <= float(max_depth)
    
    normals = estimate_normals_from_depth_map(depth)

    n = h * w

    def _edges_gradient(off_y: int, off_x: int) -> tuple[np.ndarray, np.ndarray]:
        a = depth[:h - off_y if off_y else None, :w - off_x if off_x else None]
        b = depth[off_y:, off_x:]
        va = valid[:h - off_y if off_y else None, :w - off_x if off_x else None]
        vb = valid[off_y:, off_x:]
        keep = va & vb & (np.abs(a - b) <= float(gradient_threshold))
        ys, xs = np.nonzero(keep)
        rows = ys * w + xs
        cols = (ys + off_y) * w + (xs + off_x)
        return rows, cols
    
    def _edges(off_y: int, off_x: int) -> tuple[np.ndarray, np.ndarray]:
        a = normals[:h - off_y if off_y else None, :w - off_x if off_x else None,:]
        b = normals[off_y:, off_x:,:]
        va = valid[:h - off_y if off_y else None, :w - off_x if off_x else None]
        vb = valid[off_y:, off_x:]
        aling = np.einsum('ijk,ijk->ij', a, b) # cosine similarity of normals
        keep = va & vb & (aling >= np.cos(np.deg2rad(gradient_threshold)))
        ys, xs = np.nonzero(keep)
        rows = ys * w + xs
        cols = (ys + off_y) * w + (xs + off_x)
        return rows, cols

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []

    # Right neighbour.
    r, c = _edges(0, 2); rows_list.append(r); cols_list.append(c)
    # Down neighbour.
    r, c = _edges(2, 0); rows_list.append(r); cols_list.append(c)

    if connectivity == 8:
        # Down-right diagonal.
        r, c = _edges(1, 1); rows_list.append(r); cols_list.append(c)
        # Down-left diagonal.
        a = depth[:-1, 1:]
        b = depth[1:, :-1]
        va = valid[:-1, 1:]
        vb = valid[1:, :-1]
        keep = va & vb & (np.abs(a - b) <= float(gradient_threshold))
        ys, xs = np.nonzero(keep)  # xs is offset into the cropped view starting at col=1
        rows_list.append(ys * w + (xs + 1))
        cols_list.append((ys + 1) * w + xs)

    rows = np.concatenate(rows_list).astype(np.int32, copy=False)
    cols = np.concatenate(cols_list).astype(np.int32, copy=False)
    data = np.ones(rows.size, dtype=np.uint8)

    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, raw_labels = connected_components(graph, directed=False)
    raw_labels = raw_labels.reshape(h, w)

    # Re-label: background (invalid) -> 0; valid components -> 1, 2, 3, ...
    out = np.zeros((h, w), dtype=np.int32)
    valid_labels = raw_labels[valid]
    if valid_labels.size > 0:
        _, inverse = np.unique(valid_labels, return_inverse=True)
        out[valid] = inverse.astype(np.int32) + 1

    # Optionally drop components smaller than ``min_object_size``.
    if min_object_size > 0 and out.max() > 0:
        counts = np.bincount(out.ravel())
        keep_mask = counts >= int(min_object_size)
        keep_mask[0] = True  # background always kept
        remap = np.zeros(counts.size, dtype=np.int32)
        new_id = 1
        for old_id in range(1, counts.size):
            if keep_mask[old_id]:
                remap[old_id] = new_id
                new_id += 1
        out = remap[out]

    return out


def colorize_label_mask(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Map an integer label image to a random RGB colour image for display.

    Label 0 (background) is rendered black.
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}")
    max_label = int(labels.max())
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    return palette[labels]


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------

class DataSource:
    """Class-based loader for the ISAC reflective_test dataset."""

    def __init__(self) -> None:
        self.root: Optional[Path] = None
        #self.depth_kind: str = DEFAULT_DEPTH_KIND
        # Each entry describes one (episode, view, frame) sample.
        self.items: list[dict[str, Any]] = []
        log.info("ISAC DataSource is defined")

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    # Discovery / indexing
    # ------------------------------------------------------------------

    def init_directory(
        self,
        root: str | os.PathLike[str] = DEFAULT_ROOT,
        views: Optional[tuple[str, ...]] = None,
        episodes: Optional[tuple[str, ...]] = None,
    ) -> int:
        """Scan ``root`` and build the per-sample index.

        Returns the number of indexed samples.
        """
        self.root = Path(root)
        self.items.clear()

        if not self.root.exists():
            log.error(f"ISAC dataset root does not exist: {self.root}")
            return 0

        episode_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and (episodes is None or p.name in episodes)
        )
        view_names = views if views is not None else KNOWN_VIEWS

        for ep_dir in episode_dirs:
            for view in view_names:
                view_dir = ep_dir / view
                if not view_dir.is_dir():
                    continue

                ir_left_dir   = view_dir / "ir_left"
                ir_right_dir  = view_dir / "ir_right"
                depth_gt_dir   = view_dir / "depth_left"
                depth_rs_dir  = view_dir / "depth_tyzx"
                rgb_left_dir  = view_dir / "rgb_left"
                rgb_right_dir = view_dir / "rgb_right"

                if not (ir_left_dir.is_dir() and ir_right_dir.is_dir() and depth_rs_dir.is_dir() and depth_gt_dir.is_dir()):
                    log.warning(
                        f"Skipping {view_dir}: missing one of ir_left/ir_right/depth_left/depth_tyzx"
                    )
                    continue

                frame_indices = sorted(
                    _frame_index(p.name)
                    for p in ir_left_dir.glob("frame_*.png")
                    if _frame_index(p.name) >= 0
                )

                for fi in frame_indices:
                    name = f"frame_{fi:04d}.png"
                    ir_l = ir_left_dir / name
                    ir_r = ir_right_dir / name
                    depth_rs  = depth_rs_dir / name
                    depth_gt   = depth_gt_dir / name
                    if not (ir_l.exists() and ir_r.exists() and depth_rs.exists() and depth_gt.exists()):
                        continue

                    self.items.append({
                        "episode": ep_dir.name,
                        "view": view,
                        "frame_index": fi,
                        "ir_left_path": str(ir_l),
                        "ir_right_path": str(ir_r),
                        "depth_rs_path": str(depth_rs),
                        "depth_gt_path": str(depth_gt),
                        "rgb_left_path": str((rgb_left_dir / name)) if (rgb_left_dir / name).exists() else "",
                        "rgb_right_path": str((rgb_right_dir / name)) if (rgb_right_dir / name).exists() else "",
                    })

        log.info(
            f"ISAC DataSource: indexed {len(self.items)} samples from {self.root}"
        )
        return len(self.items)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_png(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
        if not path or not os.path.exists(path):
            return None
        img = cv2.imread(path, flags)
        if img is None:
            log.warning(f"cv2.imread returned None for: {path}")
        return img

    def estimate_intrinsics(self, image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
        """Return a reasonable pinhole guess (fx, fy, cx, cy) for a given image.

        No calibration files ship with the dataset, so we assume a ~60 deg
        horizontal FOV pinhole centred on the image. Override via
        :meth:`set_intrinsics` if true values become available.
        """
        h, w = image_shape
        if hasattr(self, "_intrinsics_override") and self._intrinsics_override is not None:
            return self._intrinsics_override
        # ~60 deg horizontal FOV -> fx = (w/2) / tan(30 deg)
        fx = (w * 0.5) / np.tan(np.deg2rad(30.0))
        fy = fx
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        return float(fx), float(fy), float(cx), float(cy)

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Override the heuristic pinhole intrinsics."""
        self._intrinsics_override = (float(fx), float(fy), float(cx), float(cy))

    def get_item(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Load one sample (IR pair, depth, and point cloud) by index."""
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Sample index out of range: {index}")

        meta = self.items[index]

        ir_left   = self.load_png(meta["ir_left_path"])
        ir_right  = self.load_png(meta["ir_right_path"])
        depth_rs_img = self.load_png(meta["depth_rs_path"])
        depth_gt_img = self.load_png(meta["depth_gt_path"])
        rgb_left  = self.load_png(meta["rgb_left_path"]) if meta["rgb_left_path"] else None

        if ir_left is None or ir_right is None or depth_rs_img is None or depth_gt_img is None:
            raise RuntimeError(f"Failed to load sample {index}: {meta}")

        h, w = depth_rs_img.shape[:2]
        fx, fy, cx, cy = self.estimate_intrinsics((h, w))

        pcd = depth_to_point_cloud(
            depth_mm=depth_rs_img,
            fx=fx, fy=fy, cx=cx, cy=cy,
            color_img=rgb_left if (rgb_left is not None and rgb_left.shape[:2] == (h, w)) else None,
        )

        item: dict[str, Any] = {
            "index": index,
            "episode": meta["episode"],
            "view": meta["view"],
            "frame_index": meta["frame_index"],
            "ir_left_path": meta["ir_left_path"],
            "ir_right_path": meta["ir_right_path"],
            "depth_rs_path": meta["depth_rs_path"],
            "depth_gt_path": meta["depth_gt_path"],
            "rgb_left_path": meta["rgb_left_path"],
            "rgb_right_path": meta["rgb_right_path"],
            "ir_left_img": ir_left,
            "ir_right_img": ir_right,
            "depth_rs_img": depth_rs_img,
            "depth_gt_img": depth_gt_img,
            "rgb_left_img": rgb_left,
            "depth_pcd": pcd,
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": w, "height": h},
        }

        if debug:
            self.show_item(item)
            self.draw_point_cloud(item)

        return item
    
  

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def show_subset(
        self,
        img_list: list[Optional[np.ndarray]],
        ttl_list: list[str],
        suptitle: str = "",
    ) -> None:
        """Display a list of images in a grid."""
        img_num = len(img_list)
        col_num = min(img_num, 2)
        row_num = int(np.ceil(img_num / col_num))
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
        axes = np.array(axes).reshape(row_num, col_num)
        for k in range(img_num):
            ri, ci = k // col_num, k % col_num
            img = img_list[k]
            if img is None:
                axes[ri, ci].axis('off')
                axes[ri, ci].set_title(ttl_list[k] + " (missing)")
                continue
            # If BGR, convert to RGB for display.
            if img.ndim == 3 and img.shape[2] == 3:
                disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[ri, ci].imshow(disp)
            else:
                axes[ri, ci].imshow(img, cmap='gray' if img.ndim == 2 and img.dtype == np.uint8 else None)
            axes[ri, ci].set_title(ttl_list[k])
        for k in range(img_num, row_num * col_num):
            axes[k // col_num, k % col_num].axis('off')
        if suptitle:
            fig.suptitle(suptitle)
        plt.show(block=False)

    def show_item(self, item: dict[str, Any]) -> None:
        """Show IR-left, IR-right and depth for a loaded item."""
        suptitle = (
            f"{item['episode']} / {item['view']} / frame {item['frame_index']:04d}"
        )
        self.show_subset(
            [item["ir_left_img"], item["ir_right_img"], item["depth_rs_img"], item["depth_gt_img"]],
            ["IR left", "IR right", f"Depth RS  [mm]", f"Depth GT [mm]"],
            suptitle=suptitle,
        )

    def draw_point_cloud(self, item: dict[str, Any]) -> None:
        """Open an Open3D viewer with the back-projected cloud and a camera axis."""
        pcd = item["depth_pcd"]
        if len(pcd.points) == 0:
            log.warning("Point cloud is empty; nothing to draw.")
            return
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw([pcd, axis])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataSource(unittest.TestCase):
    """Basic tests for the ISAC reflective_test data source."""

    def test_init_directory(self):
        ds = DataSource()
        count = ds.init_directory()
        log.info(f"Indexed {count} samples")
        self.assertGreater(count, 0)
        # Sanity-check the first entry.
        first = ds.items[0]
        self.assertIn("ir_left_path", first)
        self.assertIn("ir_right_path", first)
        self.assertIn("depth_rs_path", first)
        self.assertIn("depth_gt_path", first)
        self.assertTrue(os.path.exists(first["ir_left_path"]))
        self.assertTrue(os.path.exists(first["ir_right_path"]))
        self.assertTrue(os.path.exists(first["depth_rs_path"]))
        self.assertTrue(os.path.exists(first["depth_gt_path"]))

    def test_get_item(self):
        ds = DataSource()
        count = ds.init_directory()
        self.assertGreater(count, 0)

        out = ds.get_item(4, debug=False)
        self.assertEqual(out["ir_left_img"].ndim, 2)
        self.assertEqual(out["ir_right_img"].ndim, 2)
        self.assertEqual(out["depth_rs_img"].dtype, np.uint16)
        self.assertEqual(out["ir_left_img"].shape, out["ir_right_img"].shape)
        self.assertEqual(out["depth_rs_img"].shape[:2], out["ir_left_img"].shape[:2])
        self.assertEqual(out["depth_gt_img"].shape[:2], out["ir_left_img"].shape[:2])
        self.assertGreater(len(out["depth_pcd"].points), 0)

    def test_show_images(self):
        ds = DataSource()
        count = ds.init_directory()
        if count == 0:
            log.warning("No samples found, skipping show test.")
            return

        rng = np.random.default_rng(0)
        sample_indices = rng.integers(0, count, size=min(4, count))
        for k in sample_indices:
            out = ds.get_item(int(k), debug=False)
            ds.show_item(out)
        plt.show()

    def test_draw_point_cloud(self):
        ds = DataSource()
        count = ds.init_directory()
        if count == 0:
            log.warning("No samples found, skipping point cloud test.")
            return
        rng = np.random.default_rng(0)
        sample_indices = rng.integers(0, count, size=min(4, count))
        for k in sample_indices:
            out = ds.get_item(int(k), debug=False)
            ds.draw_point_cloud(out)

    def test_create_object_mask(self):
        ds = DataSource()
        count = ds.init_directory()
        if count == 0:
            log.warning("No samples found, skipping object-mask test.")
            return

        k = 3
        out = ds.get_item(k, debug=False)
        #ds.show_item(out)

        depth_gt = out["depth_gt_img"]
        depth_rs = out["depth_rs_img"]

        labels = create_object_mask(
            depth_gt,
            gradient_threshold=15, # degrees between normals
            min_depth=10.0,
            min_object_size=100,
        )

        # Sanity: shape, dtype, label range.
        self.assertEqual(labels.shape, depth_gt.shape)
        self.assertEqual(labels.dtype, np.int32)
        self.assertGreaterEqual(int(labels.min()), 0)
        # Background (depth == 0) must be labelled 0.
        invalid = depth_gt < 1
        if invalid.any():
            self.assertTrue(np.all(labels[invalid] == 0))
        # Expect at least one real object.
        self.assertGreaterEqual(int(labels.max()), 1)
        log.info(
            f"create_object_mask: found {int(labels.max())} components "
            f"(threshold=10mm)"
        )

        # Visualize.
        ds.show_subset(
            [depth_rs, depth_gt, colorize_label_mask(labels), out["ir_left_img"]],
            ["depth_rs [mm]", "depth_gt [mm]", f"object mask (n={int(labels.max())})", "IR left"],
            suptitle=f"{out['episode']} / {out['view']} / frame {out['frame_index']:04d}",
        )
        plt.show()


def RunTest() -> None:
    tst = TestDataSource()
    # tst.test_init_directory()
    # tst.test_get_item()
    #tst.test_show_images()
    # tst.test_draw_point_cloud()
    tst.test_create_object_mask()


if __name__ == '__main__':
    RunTest()
