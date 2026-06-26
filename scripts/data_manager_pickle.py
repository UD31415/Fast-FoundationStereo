"""Dataset management for the ISAC (Pickle) scene-capture dataset.

This module is compatible with the data layout consumed by
``scripts/Uri_26_06_17.py``:

* A top-level Excel manifest (``data path.xlsx``) lists every artefact
  produced by a capture session. Relevant columns:

      - ``Label``: ``"raw data (json file)"``, ``"ICP"``, ...
      - ``STL name``: e.g. ``"cube_100x100x100"`` (used to filter captures).
      - ``Data Path``: path to a per-session JSON or ICP CSV.

* Each JSON file describes one capture session and contains:

      - ``cad_path``           — STL of the scanned object (mm).
      - ``t_camera_tooltip``   — hand-eye calibration (camera ← tooltip).
      - ``intrinsics``         — ``{fx, fy, ppx, ppy, [coeffs]}``.
      - ``captures[]``         — list of frames, each with:
            ``images``         — list of ``{image_type, path}`` entries
                                 (``Vertices``, ``Depth``, ``IR``, ``RightIR``,
                                 ``RGB8``).
            ``robot_position`` — ``{x, y, z, rx, ry, rz}`` (mm / degrees).

  ``t_camera_cad`` is reconstructed from the robot pose using two
  dataset-level calibration constants (``T_CAD_TO_USER``, ``T_USER_TO_BASE``)
  via the chain ``T_tooltip_camera @ T_base_to_tool @ T_user_to_base @
  T_cad_to_user`` — see :func:`compose_t_camera_cad`.

* Each ICP CSV row corresponds to a capture index and provides
  ``t_camera_cad`` and ``icp_matrix`` so a refined ``t_camera_cad`` can be
  built as ``icp_matrix @ t_camera_cad``.

The :class:`DataSource` mirrors the API used by the other ``data_manager_*``
scripts in this repo:

* :meth:`DataSource.init_directory` indexes the manifest and caches per-
  session metadata.
* :meth:`DataSource.get_item` returns one fully loaded sample (IR pair,
  depth, back-projected point cloud, CAD aligned to the camera frame).
* :meth:`DataSource.get_item_with_icp` returns the same item with the
  ICP-refined CAD alignment populated when an ICP CSV is available.
* :class:`TestDataSource` provides ``unittest`` smoke tests.
"""

from __future__ import annotations

import ast
import copy
import json
import logging as log
import os
import re
import unittest
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

#from scripts.Uri_26_06_17 import T_camera_cad


log.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log.INFO)


# ---------------------------------------------------------------------------
# Path translation: Windows UNC -> local POSIX mount
# ---------------------------------------------------------------------------
# Maps the UNC roots stored inside the Excel manifest and scene JSONs to the
# mount points used on the current host. Add new entries here if more shares
# get mounted under different local roots.
UNC_TO_POSIX_MAP: dict[str, str] = {
    r"\\svm.realsenseai.com\RealSense_Validation": "/mnt/validation",
}


def translate_path(p: str | os.PathLike[str] | None) -> str:
    """Translate a Windows UNC path to its local POSIX equivalent.

    Non-string / empty / already-POSIX paths are returned unchanged (as ``str``).
    """
    if p is None:
        return ""
    s = str(p)
    if not s:
        return s
    if os.sep == "/" and ("\\" in s or s.startswith("\\\\")):
        for unc_root, posix_root in UNC_TO_POSIX_MAP.items():
            if s.lower().startswith(unc_root.lower()):
                s = posix_root + s[len(unc_root):]
                break
        s = s.replace("\\", "/")
    return s


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# set 1 - with aligned data
DEFAULT_EXCEL = (
    r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0\2026_06"
    r"\yg_pickle\2026-06-16--15-12-54\Pickle_Scene_Capture_336222073841"
    r"\data path.xlsx"
)
# here you have additional data with stronger angles and translations. Tell me if the alignment holds even here
DEFAULT_EXCEL = (
    r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0\2026_06"
    r"\yg_pickle\2026-06-18--10-01-03\Pickle_Scene_Capture_336222073841"
    r"\data path.xlsx"
)

# working
DEFAULT_EXCEL = (
    r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0\2026_06"
    r"\yg_pickle\\2026-06-22--14-48-11\Pickle_Scene_Capture_336222073841"
    r"\data path.xlsx"
)

DEFAULT_EXCEL = (
    r"\\svm.realsenseai.com\RealSense_Validation\VIDB\Public\Stavush\Pickle\Data\data for model training 25_6_26"
    r"\data_25_06.xlsx"
)

#DEFAULT_STL_NAME = "cube_100x100x100"

LABEL_RAW = "raw data (json file)"
LABEL_ICP = "ICP"

# Image type strings used inside the per-session JSON.
IMG_TYPE_VERTICES = "Vertices"
IMG_TYPE_DEPTH = "Depth"
IMG_TYPE_IR_LEFT = "IR"
IMG_TYPE_IR_RIGHT = "RightIR"
IMG_TYPE_RGB = "RGB8"
MAX_Z_DISTANCE_MM = 1200  # max z to show


# ---------------------------------------------------------------------------
# File-format helpers
# ---------------------------------------------------------------------------

_RES_RE = re.compile(r"(\d+)x(\d+)")


def parse_resolution(filename: str) -> tuple[int, int]:
    """Return ``(width, height)`` parsed from a ``..._1280x720_...`` filename."""
    match = _RES_RE.search(filename)
    if not match:
        raise ValueError(
            f"Cannot parse resolution from filename: {filename!r}. "
            "Expected a pattern like '1280x720'."
        )
    return int(match.group(1)), int(match.group(2))


def load_vertices_to_pcd(path: Path) -> o3d.geometry.PointCloud:
    """Load a ``Vertices`` ``.bin`` file and return an Open3D point cloud (m)."""
    width, height = parse_resolution(path.name)
    pixel_data_bytes = width * height * 3 * 4  # float32 xyz

    raw = path.read_bytes()
    if len(raw) < pixel_data_bytes:
        raise ValueError(
            f"File too small: expected at least {pixel_data_bytes} bytes "
            f"for {width}x{height} vertices, got {len(raw)}."
        )

    xyz_flat = np.frombuffer(raw[:pixel_data_bytes], dtype=np.float32).copy()
    xyz = xyz_flat.reshape(height * width, 3)

    valid = (xyz[:, 2] != 0.0) & (xyz[:, 2] < MAX_Z_DISTANCE_MM)
    xyz = xyz[valid] * 0.001  # mm -> m

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return pcd


def read_bin_image_with_metadata(
    path: str, fsize: tuple[int, int], fbpp: int
) -> tuple[Optional[np.ndarray], bytes]:
    """Read a raw ``.bin`` image with optional trailing metadata bytes."""
    pixel_count = fsize[0] * fsize[1]
    pixel_data_bytes = pixel_count * (fbpp // 8)

    raw = Path(path).read_bytes()
    if len(raw) < pixel_data_bytes:
        log.warning(
            f"BIN file too small: expected {pixel_data_bytes} bytes for "
            f"{fsize[0]}x{fsize[1]} image with {fbpp} bpp, got {len(raw)}."
        )
        return None, b""

    if fbpp == 16:
        dtype = np.uint16
    elif fbpp == 24:
        dtype = np.uint8
    else:
        dtype = np.uint8

    img_flat = np.frombuffer(raw[:pixel_data_bytes], dtype=dtype).copy()
    if fbpp == 24:
        img = img_flat.reshape(fsize[1], fsize[0], 3)
    else:
        img = img_flat.reshape(fsize[1], fsize[0])
    return img, raw[pixel_data_bytes:]


def _decode_bin_layout(path: str) -> tuple[tuple[int, int], int]:
    """Heuristically decode ``(width, height)`` and ``bpp`` from a bin path."""
    name = os.path.basename(path)
    try:
        fsize = parse_resolution(name)
    except ValueError:
        fsize = (640, 480)

    upper = name
    if "Depth" in upper:
        fbpp = 16
    elif "RightIR" in upper or "IR" in upper:
        fbpp = 8
    elif "RGB" in upper:
        fbpp = 24
    else:
        fbpp = 8
        log.warning(f"Cannot determine image bpp from filename: {path}")
    return fsize, fbpp


def load_image_any(path: str) -> Optional[np.ndarray]:
    """Read an image given a ``.png``/``.jpg``/``.bin`` path."""
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            log.warning(f"cv2.imread returned None for: {path}")
        return img
    if ext == ".bin":
        fsize, fbpp = _decode_bin_layout(path)
        img, _ = read_bin_image_with_metadata(path, fsize, fbpp)
        return img
    log.warning(f"Unsupported image format: {path}")
    return None


# ---------------------------------------------------------------------------
# Pose helpers (mirroring the convention used by ``Uri_26_06_17.py``)
# ---------------------------------------------------------------------------

# Dataset-level calibration constants from ``Uri_26_06_17.py``. They describe
# the CAD-frame placement on the calibration target (``T_CAD_TO_USER``) and
# the rigid transform from the user (calibration) frame to the robot base
# (``T_USER_TO_BASE``). They are constant across every capture session in the
# 2026-06-16 dataset.
T_CAD_TO_USER = np.array(
    [
        [9.99999959e-01,  9.09367280e-05, -2.72211424e-04,  4.26479000e-01],
        [9.07571176e-05, -9.99999778e-01, -6.59759117e-04, -5.75152000e-01],
        [-2.72271360e-04, 6.59734385e-04, -9.99999745e-01,  4.60300000e-03],
        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ],
    dtype=np.float64,
)

T_USER_TO_BASE = np.array(
    [
        [9.99943033e-01,  5.68845315e-03,  9.03171840e-03,  4.52634000e-01],
        [5.68252948e-03, -9.99983622e-01,  6.81399922e-04, -5.87378000e-01],
        [9.03544659e-03, -6.30038099e-04, -9.99958981e-01, -4.18047000e-01],
        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ],
    dtype=np.float64,
)


def _invert_rigid(t: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 rigid transform."""
    r = t[:3, :3]
    p = t[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def build_t_tool_to_base(robot_position: dict[str, Any]) -> np.ndarray:
    """Build ``T_tool_to_base`` from a ``robot_position`` dict.

    Reproduces the construction from ``Uri_26_06_17.py``:

    * ``x, y, z`` are in millimetres and are converted to metres.
    * ``rx, ry, rz`` are extrinsic Euler angles in degrees and the rotation
      is composed as ``R = Rz @ Ry @ Rx``.
    """
    t_be = np.array(
        [
            float(robot_position.get("x", 0.0)),
            float(robot_position.get("y", 0.0)),
            float(robot_position.get("z", 0.0)),
        ],
        dtype=np.float64,
    ) / 1000.0

    rx = np.deg2rad(float(robot_position.get("rx", 0.0)))
    ry = np.deg2rad(float(robot_position.get("ry", 0.0)))
    rz = np.deg2rad(float(robot_position.get("rz", 0.0)))

    r_x = np.array(
        [
            [1.0, 0.0,        0.0       ],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx),  np.cos(rx)],
        ],
        dtype=np.float64,
    )
    r_y = np.array(
        [
            [ np.cos(ry), 0.0, np.sin(ry)],
            [ 0.0,        1.0, 0.0       ],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ],
        dtype=np.float64,
    )
    r_z = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz),  np.cos(rz), 0.0],
            [0.0,         0.0,        1.0],
        ],
        dtype=np.float64,
    )
    r_be = r_z @ r_y @ r_x

    t_tool_to_base = np.eye(4, dtype=np.float64)
    t_tool_to_base[:3, :3] = r_be
    t_tool_to_base[:3, 3] = t_be
    return t_tool_to_base


def compose_t_camera_cad(
    t_camera_tooltip: np.ndarray,
    robot_position: dict[str, Any],
    t_cad_to_user: np.ndarray = T_CAD_TO_USER,
    t_user_to_base: np.ndarray = T_USER_TO_BASE,
) -> np.ndarray:
    """Build ``T_camera_cad`` exactly like ``Uri_26_06_17.py``.

    The chain is:

        T_camera_cad = T_tooltip_camera
                       @ T_base_to_tool
                       @ T_user_to_base
                       @ T_cad_to_user

    where ``T_tooltip_camera`` is the inverse of the hand-eye calibration
    (``t_camera_tooltip``) and ``T_base_to_tool`` is the inverse of
    :func:`build_t_tool_to_base` applied to the robot pose.
    """
    t_tooltip_camera = _invert_rigid(
        np.asarray(t_camera_tooltip, dtype=np.float64)
    )
    t_tool_to_base = build_t_tool_to_base(robot_position)
    t_base_to_tool = _invert_rigid(t_tool_to_base)
    return (
        t_tooltip_camera
        @ t_base_to_tool
        @ np.asarray(t_user_to_base, dtype=np.float64)
        @ np.asarray(t_cad_to_user, dtype=np.float64)
    )

def interpolate_points_to_grid(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    image_size: tuple[int, int],
    method: str = "linear",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Interpolate scattered points ``(u, v, z)`` onto a regular ``(h, w)`` grid.

    Builds a regular grid via ``np.meshgrid`` over the image and uses
    ``scipy.interpolate.griddata`` to interpolate the ``z`` values at every
    pixel center.

    Args:
        u: 1D array of horizontal pixel coordinates of the input samples.
        v: 1D array of vertical pixel coordinates of the input samples.
        z: 1D array of values to interpolate at the ``(u, v)`` locations.
        image_size: ``(h, w)`` size of the output grid.
        method: One of ``"linear"``, ``"nearest"``, ``"cubic"`` (passed to
            ``griddata``).
        fill_value: Value used for grid points outside the convex hull of the
            input samples (ignored when ``method="nearest"``).

    Returns:
        ``(h, w)`` float32 array of interpolated values.
    """
    from scipy.interpolate import griddata

    u = np.asarray(u, dtype=np.float32).ravel()
    v = np.asarray(v, dtype=np.float32).ravel()
    z = np.asarray(z, dtype=np.float32).ravel()

    if not (u.shape == v.shape == z.shape):
        raise ValueError("u, v, z must have the same number of elements")

    h, w = image_size

    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    u, v, z = u[valid], v[valid], z[valid]

    if u.size == 0:
        return np.full((h, w), fill_value, dtype=np.float32)

    ui, vi = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    grid_z = griddata(
        points=np.column_stack((u, v)),
        values=z,
        xi=(ui, vi),
        method=method,
        fill_value=fill_value,
    )
    return grid_z.astype(np.float32)


# ---------------------------------------------------------------------------
# Generic depth / mask helpers (kept from the previous version of this file)
# ---------------------------------------------------------------------------

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
    """Back-project a uint16/float depth image (mm) to an Open3D point cloud."""
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

    if (
        color_img is not None
        and color_img.ndim == 3
        and color_img.shape[:2] == (h, w)
    ):
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        colors = rgb[vs, us].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def estimate_normals_from_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """Per-pixel surface normals from a depth map via the Sobel gradient."""
    if depth_map.dtype != np.float32:
        depth_map = depth_map.astype(np.float32)

    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

    ksize = 1
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=ksize)

    grad_x = grad_x[:, :, np.newaxis]
    grad_y = grad_y[:, :, np.newaxis]
    z_component = np.ones_like(grad_x)

    direction = np.concatenate((-grad_x, -grad_y, z_component), axis=2)
    magnitude = np.linalg.norm(direction, axis=2, keepdims=True)
    return np.divide(
        direction,
        magnitude,
        out=np.zeros_like(direction),
        where=magnitude != 0,
    )

def create_object_mask(
    depth_gt: np.ndarray,
    gradient_threshold: float = 10.0,
    min_depth: float = 1.0,
    max_depth: Optional[float] = None,
    min_object_size: int = 0,
    connectivity: int = 4,
) -> np.ndarray:
    """Segment a depth image into objects using normal-based gradients."""
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

    def _edges(off_y: int, off_x: int) -> tuple[np.ndarray, np.ndarray]:
        a = normals[: h - off_y if off_y else None, : w - off_x if off_x else None]
        b = normals[off_y:, off_x:]
        va = valid[: h - off_y if off_y else None, : w - off_x if off_x else None]
        vb = valid[off_y:, off_x:]
        align = np.einsum("ijk,ijk->ij", a, b)
        keep = va & vb & (align >= np.cos(np.deg2rad(gradient_threshold)))
        ys, xs = np.nonzero(keep)
        rows = ys * w + xs
        cols = (ys + off_y) * w + (xs + off_x)
        return rows, cols

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []

    r, c = _edges(0, 1); rows_list.append(r); cols_list.append(c)
    r, c = _edges(1, 0); rows_list.append(r); cols_list.append(c)

    if connectivity == 8:
        r, c = _edges(1, 1); rows_list.append(r); cols_list.append(c)
        a = normals[:-1, 1:]
        b = normals[1:, :-1]
        va = valid[:-1, 1:]
        vb = valid[1:, :-1]
        align = np.einsum("ijk,ijk->ij", a, b)
        keep = va & vb & (align >= np.cos(np.deg2rad(gradient_threshold)))
        ys, xs = np.nonzero(keep)
        rows_list.append(ys * w + (xs + 1))
        cols_list.append((ys + 1) * w + xs)

    rows = np.concatenate(rows_list).astype(np.int32, copy=False)
    cols = np.concatenate(cols_list).astype(np.int32, copy=False)
    data = np.ones(rows.size, dtype=np.uint8)

    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, raw_labels = connected_components(graph, directed=False)
    raw_labels = raw_labels.reshape(h, w)

    out = np.zeros((h, w), dtype=np.int32)
    valid_labels = raw_labels[valid]
    if valid_labels.size > 0:
        _, inverse = np.unique(valid_labels, return_inverse=True)
        out[valid] = inverse.astype(np.int32) + 1

    if min_object_size > 0 and out.max() > 0:
        counts = np.bincount(out.ravel())
        keep_mask = counts >= int(min_object_size)
        keep_mask[0] = True
        remap = np.zeros(counts.size, dtype=np.int32)
        new_id = 1
        for old_id in range(1, counts.size):
            if keep_mask[old_id]:
                remap[old_id] = new_id
                new_id += 1
        out = remap[out]

    return out

def colorize_label_mask(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Map an integer label image to a deterministic RGB colour image."""
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
    """Loader for the ISAC / Pickle Excel-driven capture dataset."""

    def __init__(self, train_mode = False) -> None:
        self.excel_path: Optional[Path] = None
        self.stl_name: Optional[str] = None
        self.manifest_df: Optional[pd.DataFrame] = None

        # Per-session metadata, keyed by JSON path.
        self.sessions: dict[str, dict[str, Any]] = {}
        # Cached CAD point clouds keyed by cad_path.
        self._cad_cache: dict[str, o3d.geometry.PointCloud] = {}
        # Cached CAD triangle meshes (in metres) keyed by cad_path.
        self._cad_mesh_cache: dict[str, o3d.geometry.TriangleMesh] = {}

        self._background_mesh_cache: dict[str, o3d.geometry.TriangleMesh] = {}  # background mesh - plate

        # Flat per-capture index. Each entry has at least:
        #   ``json_path``, ``capture_idx``, plus eager copies of useful fields.
        self.items: list[dict[str, Any]] = []

        # Lazily loaded ICP refinements: ``capture_uid -> 4x4 matrix``.
        self.icp_table: Optional[pd.DataFrame] = None
        self.icp_path: Optional[Path] = None
        self.train_mode = train_mode

        log.info("Pickle DataSource is defined")

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    # Discovery / indexing
    # ------------------------------------------------------------------

    def init_directory(
        self,
        excel_path: str | os.PathLike[str] = DEFAULT_EXCEL,
        json_paths: Optional[Sequence[str]] = None,
    ) -> int:
        """Index every capture across all matching session JSONs.

        Parameters
        ----------
        excel_path : path to the dataset manifest (``data_14_6_26.xlsx``).
        stl_name : optional filter on the ``STL name`` column.
        json_paths : explicit list of JSONs to use instead of the manifest.

        Returns
        -------
        int : the number of indexed capture items.
        """
        self.items.clear()
        self.sessions.clear()
        self._cad_cache.clear()
        self._cad_mesh_cache.clear()
        self._background_mesh_cache.clear()

        if json_paths is not None:
            session_jsons: list[str] = [translate_path(p) for p in json_paths]
            self.excel_path = None
            self.manifest_df = None
        else:
            self.excel_path = Path(translate_path(excel_path))
            if not self.excel_path.exists():
                log.error(f"Manifest Excel does not exist: {self.excel_path}")
                return 0

            df = pd.read_excel(self.excel_path)
            self.manifest_df = df
            mask = df["Label"].astype(str) == LABEL_RAW
            # if stl_name is not None:
            #     mask &= df["STL name"].astype(str) == str(stl_name)
            session_jsons = [translate_path(p) for p in df.loc[mask, "Data Path"].tolist()]

        #self.stl_name = stl_name

        for json_path in session_jsons:
            self.index_session(json_path)

        log.info(
            f"DataSource: indexed {len(self.items)} captures across "
            f"{len(self.sessions)} sessions"
        )
        return len(self.items)

    # Default root for ``init_multi_directory``. Each direct subfolder named with
    # the timestamp pattern ``YYYY-MM-DD--HH-MM-SS`` is expected to contain a
    # ``Pickle_Scene_Capture*/data path.xlsx`` manifest.
    DEFAULT_MULTI_ROOT = (
        r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0"
        r"\2026_06\yg_pickle"
    )
    _SESSION_FOLDER_RE = re.compile(r"^\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}$")

    def init_multi_directory(
        self,
        root: str | os.PathLike[str] = DEFAULT_MULTI_ROOT,
        excel_glob: str = "**/data path.xlsx",
        session_pattern: Optional[re.Pattern[str] | str] = None,
        keywords: Optional[Sequence[str]] = None,
    ) -> int:
        """Index captures from every ``data path.xlsx`` under ``root``.

        Walks each timestamped subfolder (e.g. ``2026-06-24--16-56-29``) under
        ``root``, locates its ``Pickle_Scene_Capture*/data path.xlsx`` manifest,
        and accumulates all captures into ``self.items``. Useful when many
        sessions need to be benchmarked together rather than one Excel at a
        time as in :meth:`init_directory`.

        Parameters
        ----------
        root : root directory containing dated session folders. May be a UNC
            path; it is translated via :func:`translate_path`.
        excel_glob : glob (relative to each session folder) used to locate the
            manifest Excel file. Defaults to ``"**/data path.xlsx"`` so the
            ``Pickle_Scene_Capture*`` subfolder is matched automatically.
        session_pattern : optional regex (string or compiled) used to filter
            session-folder names. Defaults to the ``YYYY-MM-DD--HH-MM-SS``
            timestamp pattern.
        keywords : optional list of case-insensitive substrings; if given,
            only session folders whose name contains at least one keyword are
            included.

        Returns
        -------
        int : the total number of indexed capture items.
        """
        self.items.clear()
        self.sessions.clear()
        self._cad_cache.clear()
        self._cad_mesh_cache.clear()
        self._background_mesh_cache.clear()

        root_path = Path(translate_path(root))
        if not root_path.is_dir():
            log.error(f"Multi-directory root not found or unreadable: {root_path}")
            self.excel_path = None
            self.manifest_df = None
            return 0

        if isinstance(session_pattern, str):
            session_re = re.compile(session_pattern)
        elif session_pattern is None:
            session_re = self._SESSION_FOLDER_RE
        else:
            session_re = session_pattern

        keywords_upper = [k.upper() for k in keywords] if keywords else None

        session_folders = sorted(
            p for p in root_path.iterdir()
            if p.is_dir() and session_re.match(p.name)
            and (keywords_upper is None
                 or any(kw in p.name.upper() for kw in keywords_upper))
        )
        if not session_folders:
            log.warning(
                f"No timestamped session folders matched under {root_path} "
                f"(pattern={session_re.pattern!r}, keywords={keywords})"
            )
            self.excel_path = None
            self.manifest_df = None
            return 0

        excel_paths: list[Path] = []
        for folder in session_folders:
            matches = sorted(folder.glob(excel_glob))
            if not matches:
                log.warning(f"No manifest found in {folder} (glob={excel_glob!r})")
                continue
            if len(matches) > 1:
                log.info(
                    f"{len(matches)} manifests in {folder}; using first: "
                    f"{matches[0].name}"
                )
            excel_paths.append(matches[0])

        if not excel_paths:
            log.warning(f"No manifest Excel files discovered under {root_path}")
            self.excel_path = None
            self.manifest_df = None
            return 0

        # Collect JSONs from every manifest, then run the regular indexing.
        all_session_jsons: list[str] = []
        manifest_frames: list[pd.DataFrame] = []
        for xlsx in excel_paths:
            try:
                df = pd.read_excel(xlsx)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Failed to read manifest {xlsx}: {exc}")
                continue
            df = df.copy()
            df["__manifest_path"] = str(xlsx)
            manifest_frames.append(df)

            mask = df["Label"].astype(str) == LABEL_RAW
            all_session_jsons.extend(
                translate_path(p) for p in df.loc[mask, "Data Path"].tolist()
            )

        # Record bookkeeping that mirrors ``init_directory``.
        self.excel_path = excel_paths[0]
        self.manifest_df = (
            pd.concat(manifest_frames, ignore_index=True) if manifest_frames else None
        )

        for json_path in all_session_jsons:
            self.index_session(json_path)

        log.info(
            f"DataSource: indexed {len(self.items)} captures across "
            f"{len(self.sessions)} sessions from {len(excel_paths)} manifests "
            f"under {root_path}"
        )
        return len(self.items)

    def index_session(self, json_path: str) -> None:
        """Load one session JSON and append its captures to ``self.items``."""
        path = Path(json_path)
        if not path.exists():
            log.warning(f"Session JSON missing, skipping: {path}")
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                scene = json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Failed to read JSON {path}: {exc}")
            return

        captures = list(scene.get("captures", []))
        if not captures:
            log.warning(f"Session has no captures: {path}")
            return

        cad_name            = scene.get("cad_name", "")
        cad_path            = translate_path(scene.get("cad_path", ""))
        t_camera_tooltip    = np.array(scene.get("t_camera_tooltip", np.eye(4)), dtype=np.float64)
        intrinsics          = scene.get("intrinsics", {}) or {}
        baseline_mm         = scene.get("baseline", 0.05) * 1000.0
        bf                  = baseline_mm * intrinsics.get("fx", 660.0)

        background_cad_name = 'thorlab matrix'
        background_cad_path = r"..\assets\thorlab matrix.STL"
        #mesh_backg      = self.load_background_mesh(background_cad_path)                

        #baseline_mm = scene.get("baseline", 0.05) * 1000.0
        self.sessions[str(path)] = {
            "scene"                 : scene,
            "cad_path"              : cad_path,
            "cad_name"              : cad_name,
            "background_cad_path"   : background_cad_path,
            "background_cad_name"   : background_cad_name,
            "t_camera_tooltip"      : t_camera_tooltip,
            "intrinsics"            : intrinsics,
            "bf"                    : bf,
            "baseline_mm"           : baseline_mm,
        }

        for ci, capture in enumerate(captures):
            paths = self._extract_capture_paths(capture)
            self.items.append(
                {
                    "json_path": str(path),  # session address
                    "capture_idx": ci,
                    **paths,
                    "t_robot_position": capture.get("robot_position", np.eye(4)),
                }
            )

    @staticmethod
    def _extract_capture_paths(capture: dict[str, Any]) -> dict[str, str]:
        """Pull the well-known image paths out of one capture entry."""
        out = {
            "ir_left_path": "",
            "ir_right_path": "",
            "depth_path": "",
            "rgb_path": "",
            "vertices_path": "",
        }
        for img in capture.get("images", []) or []:
            kind = img.get("image_type", "")
            p = translate_path(img.get("path", "") or "")
            if kind == IMG_TYPE_IR_LEFT:
                out["ir_left_path"] = p
            elif kind == IMG_TYPE_IR_RIGHT:
                out["ir_right_path"] = p
            elif kind == IMG_TYPE_DEPTH:
                out["depth_path"] = p
            elif kind == IMG_TYPE_RGB:
                out["rgb_path"] = p
            elif kind == IMG_TYPE_VERTICES:
                out["vertices_path"] = p
        return out

    # ------------------------------------------------------------------
    # CAD / intrinsics helpers
    # ------------------------------------------------------------------
    def render_points_from_mesh(self, mesh_t: o3d.t.geometry.TriangleMesh) -> o3d.geometry.PointCloud:

        # 1. Load legacy mesh and convert to Tensor API
        # mesh_legacy = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        # mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

        # 2. Build the Raycasting Scene
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)

        # 3. Create a 3D coordinate grid matrix
        # Adjust bounds and grid spacing based on your mesh dimensions
        grid_spacing = 0.1
        bbox = mesh_t.get_axis_aligned_bounding_box()
        x = np.arange(bbox.min_bound[0], bbox.max_bound[0], grid_spacing)
        y = np.arange(bbox.min_bound[1], bbox.max_bound[1], grid_spacing)
        z = np.arange(bbox.min_bound[2], bbox.max_bound[2], grid_spacing)

        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).astype(np.float32)

        # 4. Filter grid coordinates near the shell boundary
        # Only project grid coordinates within 1 voxel size of the mesh surface
        distances = scene.compute_distance(grid_points).numpy()
        surface_mask = distances <= grid_spacing
        candidate_grid_points = grid_points[surface_mask]

        # 5. Snap the remaining grid coordinates directly onto the outer surface
        ans = scene.compute_closest_point(candidate_grid_points)
        surface_snapped_points = ans['points'].numpy()

        # 6. Save out to a legacy PointCloud container
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(surface_snapped_points)

        # Optional visualization
        o3d.visualization.draw_geometries([sampled_pcd])

        return sampled_pcd

    def load_cad_pcd(self, cad_path: str) -> Optional[o3d.geometry.PointCloud]:
        """Load and cache the sampled CAD point cloud (in metres)."""
        if not cad_path:
            return None
        if cad_path in self._cad_cache:
            return self._cad_cache[cad_path]

        if not os.path.exists(cad_path):
            log.warning(f"CAD path missing: {cad_path}")
            return None

        mesh = o3d.io.read_triangle_mesh(cad_path)
        if mesh.is_empty():
            log.warning(f"Failed to load CAD mesh: {cad_path}")
            return None

        # STL units are millimetres; convert to metres.
        mesh.scale(0.001, center=(0.0, 0.0, 0.0))
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        cad_pcd = mesh.sample_points_poisson_disk(
            number_of_points=150000,
            use_triangle_normal=True,
        )

        # # Define your grid reference step size (voxel size)
        # voxel_size = 0.0001 

        # # Simplify the mesh using vertex clustering based on the grid size
        # # Contraction option determines how points pool together (e.g., Average)
        # simplified_mesh = mesh.simplify_vertex_clustering(
        #     voxel_size=voxel_size,
        #     contraction=o3d.geometry.SimplificationContraction.Average
        # )

        #simplified_mesh = mesh.subdivide_loop(number_of_iterations=10)

        # # Extract the remaining grid-aligned points
        # cad_pcd = o3d.geometry.PointCloud()
        # cad_pcd.points = simplified_mesh.vertices

        #cad_pcd = self.render_points_from_mesh(mesh)

        self._cad_cache[cad_path]      = cad_pcd
        # Also cache the metres-scaled mesh for raycast rendering.
        self._cad_mesh_cache[cad_path] = mesh
        return cad_pcd

    def load_cad_mesh(self, cad_path: str) -> Optional[o3d.geometry.TriangleMesh]:
        """Load and cache the CAD mesh, scaled to metres."""
        if not cad_path:
            return None
        if cad_path in self._cad_mesh_cache:
            return self._cad_mesh_cache[cad_path]

        if not os.path.exists(cad_path):
            log.warning(f"CAD path missing: {cad_path}")
            return None

        mesh = o3d.io.read_triangle_mesh(cad_path)
        if mesh.is_empty():
            log.warning(f"Failed to load CAD mesh: {cad_path}")
            return None

        mesh.scale(0.001, center=(0.0, 0.0, 0.0))  # mm -> m
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        self._cad_mesh_cache[cad_path] = mesh
        return mesh
    
    def load_background_mesh_flat(self,
        mesh_cam: o3d.geometry.TriangleMesh,
        background_offset_m: float = 0.0,
        background_size_m: float = 2.0,
        background_z_m: Optional[float] = 0.0,
    ) -> o3d.geometry.TriangleMesh:
        """Render ``mesh_cam`` plus a flat rectangular background to a depth image.

        Builds a scene composed of the input mesh (already in the camera
        frame) and a large flat rectangle parallel to the image plane,
        placed behind the object so that every camera ray hits something.
        The combined scene is then rasterized with the same pinhole
        ray-casting pipeline used by :meth:`render_mesh_depth`.

        Parameters
        ----------
        mesh_cam : ``TriangleMesh`` whose vertices are in the *camera*
            frame, in metres.
        cam_matrix : 3x3 pinhole intrinsic ``K`` (pixels).
        frame_size : ``(H, W)``.
        background_offset_m : distance (m) placed behind the mesh's
            furthest +Z vertex when ``background_z_m`` is not specified.
        background_size_m : edge length (m) of the square background plane.
        background_z_m : explicit camera-Z (m) for the background plane.
            Overrides ``background_offset_m`` when provided.
        output_units : ``"mm"`` (default) or ``"m"``.

        Returns
        -------
        ``(H, W)`` ``float32`` depth image. Pixels are guaranteed to be
        populated as long as the background plane covers the field of view.
        """
        #if self._background_mesh_cache is not None:
        if len(self._background_mesh_cache) > 0:
            return self._background_mesh_cache

        # Determine where to place the background plane.
        if background_z_m is None:
            if mesh_cam is None or mesh_cam.is_empty():
                bg_z = max(1.0, float(background_offset_m))
            else:
                verts = np.asarray(mesh_cam.vertices, dtype=np.float64)
                z_max = float(verts[:, 2].max()) if verts.size else 0.0
                bg_z = z_max + float(background_offset_m)
        else:
            bg_z = float(background_z_m)

        # if bg_z <= 0.0:
        #     raise ValueError(
        #         f"Background plane must lie in front of the camera (z>0), got z={bg_z}"
        #     )

        # Build a flat rectangular background facing the camera.
        half = 0.5 * float(background_size_m)
        bg_vertices = np.array(
            [
                [-half, -half, bg_z],
                [ half, -half, bg_z],
                [ half,  half, bg_z],
                [-half,  half, bg_z],
            ],
            dtype=np.float64,
        )
        # Two triangles, double-sided so orientation does not matter.
        bg_triangles = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [0, 2, 1], [0, 3, 2],
            ],
            dtype=np.int32,
        )
        bg_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(bg_vertices),
            triangles=o3d.utility.Vector3iVector(bg_triangles),
        )
        self._background_mesh_cache = bg_mesh
        return bg_mesh
    
    def load_background_mesh(self, background_cad_path: str) -> Optional[o3d.geometry.TriangleMesh]:
        """Load and cache the background CAD mesh, scaled to metres."""
        if len(background_cad_path) < 2:
            background_cad_path = r".\assets\thorlab matrix.STL"  # Default path to the background CAD mesh

        if self._background_mesh_cache:
            return self._background_mesh_cache
        
        # resolve for linus machine
        if not os.path.exists(background_cad_path):
            background_cad_path = r"./assets/thorlab matrix.STL"

        if not os.path.exists(background_cad_path):
            log.warning(f"Background CAD path missing: {background_cad_path}")
            return None

        mesh = o3d.io.read_triangle_mesh(background_cad_path)
        if mesh.is_empty():
            log.warning(f"Failed to load background CAD mesh: {background_cad_path}")
            return None

        mesh.scale(0.001, center=(0.0, 0.0, 0.0))  # mm -> m
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        

        # pcd = mesh.sample_points_poisson_disk(
        #     number_of_points=10000,
        #     use_triangle_normal=True
        # )
        
        # transform the mesh to align with the camera frame
        R = mesh.get_rotation_matrix_from_xyz((0, 0,np.pi/2))
        # # Apply rotation
        mesh.rotate(R, center=(0, 0, 0))
        #pcd.transform(T_camera_cad)

        self._background_mesh_cache = mesh
        return mesh

    #def get_intrinsics_matrix(self, json_path: str) -> tuple[np.ndarray, np.ndarray]:
    def get_intrinsics_matrix(self, item) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(K, dist_coeffs)`` for one session."""
        #info = self.sessions.get(json_path, {})
        intr = item.get("intrinsics", {})
        fx = float(intr.get("fx", 1.0))
        fy = float(intr.get("fy", 1.0))
        cx = float(intr.get("ppx", 0.0))
        cy = float(intr.get("ppy", 0.0))
        k = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        coeffs = np.asarray(intr.get("coeffs", [0.0] * 5), dtype=np.float32).reshape(-1)
        if coeffs.size == 0:
            coeffs = np.zeros((5,), dtype=np.float32)
        return k, coeffs

    # ------------------------------------------------------------------
    # ICP refinement (optional)
    # ------------------------------------------------------------------

    def load_icp_csv(
        self,
        icp_csv_path: Optional[str | os.PathLike[str]] = None,
    ) -> bool:
        """Load an ICP CSV (with ``icp_matrix`` and ``t_camera_cad`` columns).

        If no path is supplied, the function tries to read it from the Excel
        manifest by selecting rows where ``Label == 'ICP'`` (and matching
        ``stl_name`` when available).
        """
        if icp_csv_path is None:
            if self.manifest_df is None:
                log.warning(
                    "Cannot infer ICP CSV path: no manifest was loaded. "
                    "Pass icp_csv_path explicitly."
                )
                return False
            df = self.manifest_df
            mask = df["Label"].astype(str) == LABEL_ICP
            if self.stl_name is not None:
                mask &= df["STL name"].astype(str) == str(self.stl_name)
            paths = df.loc[mask, "Data Path"].tolist()
            if not paths:
                log.warning("No ICP rows found in manifest.")
                return False
            icp_csv_path = paths[0]

        path = Path(str(icp_csv_path))
        if not path.exists():
            log.warning(f"ICP CSV not found: {path}")
            return False

        self.icp_table = pd.read_csv(path)
        self.icp_path = path
        log.info(f"Loaded ICP table with {len(self.icp_table)} rows from {path}")
        return True

    def load_icp_t_camera_cad(self, capture_idx: int) -> Optional[np.ndarray]:
        """Return the ICP-refined ``t_camera_cad`` for a capture, if any."""
        if self.icp_table is None:
            return None
        if capture_idx < 0 or capture_idx >= len(self.icp_table):
            return None
        row = self.icp_table.iloc[capture_idx]
        try:
            t_icp = np.array(ast.literal_eval(row["icp_matrix"]), dtype=np.float64)
            t_base = np.array(ast.literal_eval(row["t_camera_cad"]), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Failed to parse ICP row {capture_idx}: {exc}")
            return None
        return t_icp @ t_base

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def get_item(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Load one fully populated capture sample by global index."""
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Sample index out of range: {index}")

        #log.info(f"Loading item {index}/{len(self.items)}...")
        meta            = self.items[index]
        json_path       = meta["json_path"]
        session         = self.sessions[json_path]
        scene           = session["scene"]
        #capture         = scene["captures"][meta["capture_idx"]]
        bf              = session["bf"]

        ir_left         = load_image_any(meta["ir_left_path"])
        ir_right        = load_image_any(meta["ir_right_path"])
        depth_img       = load_image_any(meta["depth_path"])
        rgb             = load_image_any(meta["rgb_path"]) if meta["rgb_path"] else None

        
        
        vertices_path   = meta["vertices_path"]
        depth_pcd_raw: Optional[o3d.geometry.PointCloud] = None
        if vertices_path and os.path.exists(vertices_path):
            try:
                depth_pcd_raw = load_vertices_to_pcd(Path(vertices_path))
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Failed to load vertices {vertices_path}: {exc}")

        # Pose composition: camera <- CAD using Uri_26_06_17.py's convention.
        t_camera_tooltip    = session["t_camera_tooltip"]
        t_robot_position    = meta["t_robot_position"]
        t_tool_to_base      = build_t_tool_to_base(t_robot_position)
        t_camera_cad        = compose_t_camera_cad(t_camera_tooltip, t_robot_position)

        cad_pcd             = self.load_cad_pcd(session["cad_path"])
        cad_pcd_aligned: Optional[o3d.geometry.PointCloud] = None
        if cad_pcd is not None:
            cad_pcd_aligned = copy.deepcopy(cad_pcd)
            cad_pcd_aligned.transform(t_camera_cad)

        # ICP-refined alignment, when available.
        t_camera_cad_icp    = self.load_icp_t_camera_cad(meta["capture_idx"])
        cad_pcd_aligned_icp: Optional[o3d.geometry.PointCloud] = None
        if cad_pcd is not None and t_camera_cad_icp is not None:
            cad_pcd_aligned_icp = copy.deepcopy(cad_pcd)
            cad_pcd_aligned_icp.transform(t_camera_cad_icp)

        intrinsics           = session["intrinsics"]
        item: dict[str, Any] = {
            "index"                 : index,
            "json_path"             : json_path,
            "capture_idx"           : meta["capture_idx"],
            # paths
            "ir_left_path"          : meta["ir_left_path"],
            "ir_right_path"         : meta["ir_right_path"],
            "depth_path"            : meta["depth_path"],
            "rgb_path"              : meta["rgb_path"],
            "vertices_path"         : vertices_path,
            "cad_path"              : session["cad_path"],
            # images / clouds
            "ir_left_img"           : ir_left,
            "ir_right_img"          : ir_right,
            "depth_img"             : depth_img,
            "rgb_img"               : rgb,
            "depth_pcd_raw"         : depth_pcd_raw,
            "cad_pcd"               : cad_pcd,
            "cad_pcd_aligned"       : cad_pcd_aligned,
            "cad_pcd_aligned_icp"   : cad_pcd_aligned_icp,
            # transforms / metadata
            "t_camera_tooltip"      : t_camera_tooltip,
            "t_tool_to_base"        : t_tool_to_base,
            "t_user_to_base"        : T_USER_TO_BASE,
            "t_cad_to_user"         : T_CAD_TO_USER,
            "t_camera_cad"          : t_camera_cad,
            "t_camera_cad_icp"      : t_camera_cad_icp,
            "t_robot_position"      : t_robot_position,
            'bf'                    : bf,
            "intrinsics"            : intrinsics,
        }

        if debug:
            self.draw_point_clouds(item)

        return item

    def get_item_with_icp(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Like :meth:`get_item` but ensures the ICP table has been loaded."""
        if self.icp_table is None:
            self.load_icp_csv()
        return self.get_item(index, debug=debug)

    # ------------------------------------------------------------------
    # CAD -> camera-image projection
    # ------------------------------------------------------------------

    def project_mesh_on_camera(self, mesh, cam_matrix, cam_dist_coeffs, cam_extrinsics, frame_size=(720, 1280)):
        # NOTE: Open3D's ``OffscreenRenderer`` (Filament backend) is not
        # supported on Windows wheels (it requires EGL headless). On Windows
        # we use a hidden ``Visualizer`` window + ``capture_screen_float_buffer``
        # to render a mesh to an RGB image with explicit pinhole intrinsics.

        image_height, image_width = frame_size

        # Build pinhole intrinsics from a vertical FOV of 60 deg.
        # fov_deg = 60.0
        # fy = image_height / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
        # fx = fy
        # cx = image_width / 2.0
        # cy = image_height / 2.0
        fx, fy, cx, cy = cam_matrix[0, 0], cam_matrix[1, 1], cam_matrix[0, 2], cam_matrix[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, fx, fy, cx, cy)

        # Mesh to render.
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        mesh.compute_vertex_normals()

        # # Look-at extrinsic: camera at (2, 2, 2), looking at origin, +Z up.
        # eye = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        # center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        # up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # f = (center - eye)
        # f /= np.linalg.norm(f)
        # s = np.cross(f, up)
        # s /= np.linalg.norm(s)
        # u = np.cross(s, f)
        # # Open3D's pinhole convention: +Z forward, +Y down.
        # # Build R such that R @ world = camera; rows are camera basis vectors.
        # rotation = np.stack([s, -u, f], axis=0)
        # translation = -rotation @ eye
        # extrinsic = np.eye(4, dtype=np.float64)
        # extrinsic[:3, :3] = rotation
        # extrinsic[:3, 3] = translation

        extrinsic = cam_extrinsics

        vis = o3d.visualization.Visualizer()
        created = vis.create_window(
            visible=False, width=image_width, height=image_height
        )
        if not created:
            self.skipTest("Could not create Open3D window for offscreen render")

        try:
            vis.add_geometry(mesh)

            ctr             = vis.get_view_control()
            params          = ctr.convert_to_pinhole_camera_parameters()
            params.intrinsic = intrinsic
            params.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters( params, allow_arbitrary=True)

            vis.poll_events()
            vis.update_renderer()
            float_img       = vis.capture_screen_float_buffer(do_render=True)
        finally:
            vis.destroy_window()

        image_ndarray = np.asarray(float_img)
        #self.assertEqual(image_ndarray.shape, (image_height, image_width, 3))

        # Save out the rendered RGB frame for inspection.
        rgb_uint8 = np.ascontiguousarray(
            (np.clip(image_ndarray, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        return rgb_uint8

    @staticmethod
    def project_3d_to_camera_depth(
        points_3d_m: np.ndarray,
        cam_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        frame_size: tuple[int, int],
    ) -> np.ndarray:
        """Project 3D camera-frame points (metres) to a depth image (mm).

        Mirrors ``DataSource.project_3d_to_camera_depth`` in
        ``data_manager_pickle.py``: uses ``cv2.projectPoints`` and a z-buffer
        (``np.minimum.at``) so the nearest depth wins per pixel.

        Returns an ``(H, W)`` ``float32`` image; invalid pixels are ``0``.
        """
        if points_3d_m.ndim != 2 or points_3d_m.shape[1] != 3:
            raise ValueError("points_3d_m must have shape (N, 3)")

        h, w = frame_size
        valid = np.isfinite(points_3d_m).all(axis=1) & (points_3d_m[:, 2] > 1e-6)
        if not np.any(valid):
            log.warning("No valid 3D points to project.")
            return np.zeros((h, w), dtype=np.float32)

        # metres -> millimetres (depth image convention)
        points_3d_mm = (points_3d_m * 1000.0).astype(np.float32)
        pts = points_3d_mm[valid]

        projected_pts, _ = cv2.projectPoints(
            pts.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            cam_matrix.astype(np.float32),
            dist_coeffs.astype(np.float32),
        )

        uv = projected_pts.reshape(-1, 2)
        u_idx = np.rint(uv[:, 0]).astype(np.int32)
        v_idx = np.rint(uv[:, 1]).astype(np.int32)

        in_bounds = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
        if not np.any(in_bounds):
            return np.zeros((h, w), dtype=np.float32)

        u_idx = u_idx[in_bounds]
        v_idx = v_idx[in_bounds]
        z_vals_mm = pts[in_bounds, 2].astype(np.float32)

        lin = v_idx * w + u_idx
        depth_buffer = np.full(h * w, np.inf, dtype=np.float32)
        np.minimum.at(depth_buffer, lin, z_vals_mm)
        depth_projected = depth_buffer.reshape(h, w)
        depth_projected[~np.isfinite(depth_projected)] = 0.0

        #depth_projected_2 = interpolate_points_to_grid(u_idx, v_idx, z_vals_mm, image_size=(h, w))

        return depth_projected

    @staticmethod
    def render_mesh_depth(
        mesh: o3d.geometry.TriangleMesh,
        cam_matrix: np.ndarray,
        frame_size: tuple[int, int],
        extrinsic: Optional[np.ndarray] = None,
        output_units: str = "mm",
    ) -> np.ndarray:
        """Rasterize a ``TriangleMesh`` into a dense depth image via ray casting.

        Uses :class:`open3d.t.geometry.RaycastingScene` to cast one ray per
        pixel through a pinhole camera. Unlike
        :meth:`project_3d_to_camera_depth`, this produces a fully dense
        silhouette (no holes between samples).

        Parameters
        ----------
        mesh : ``TriangleMesh`` whose vertices are in the *camera* frame
            (or supply ``extrinsic`` to transform from world to camera).
        cam_matrix : 3x3 pinhole intrinsic ``K`` (pixels). The mesh must be
            in metres for ``K`` to be interpreted correctly.
        frame_size : ``(H, W)``.
        extrinsic : optional 4x4 world->camera transform; defaults to
            identity (i.e. mesh already in the camera frame).
        output_units : ``"mm"`` (default) or ``"m"``.

        Returns
        -------
        ``(H, W)`` ``float32`` depth image. Pixels where the ray missed the
        mesh are ``0``.

        Notes
        -----
        Pinhole only — undistort the input image first if your camera has
        non-trivial distortion coefficients.
        """
        h, w = frame_size
        if mesh is None or mesh.is_empty():
            return np.zeros((h, w), dtype=np.float32)

        if extrinsic is None:
            extrinsic = np.eye(4, dtype=np.float64)

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)

        K_t = o3d.core.Tensor(np.asarray(cam_matrix, dtype=np.float64))
        E_t = o3d.core.Tensor(np.asarray(extrinsic, dtype=np.float64))

        rays = scene.create_rays_pinhole(
            intrinsic_matrix=K_t,
            extrinsic_matrix=E_t,
            width_px=w,
            height_px=h,
        )
        ans = scene.cast_rays(rays)

        t_hit = ans["t_hit"].numpy()        # (H, W) ray length to first hit
        rays_np = rays.numpy()              # (H, W, 6): [ox,oy,oz, dx,dy,dz]
        dz = rays_np[..., 5]
        depth = t_hit * dz                  # camera-Z depth, in mesh units (m)
        depth[~np.isfinite(depth)] = 0.0

        if output_units == "mm":
            depth = depth * 1000.0
        elif output_units != "m":
            raise ValueError("output_units must be 'mm' or 'm'")
        return depth.astype(np.float32)

    # @staticmethod
    # def render_scene_depth(
    #     mesh_cam: o3d.geometry.TriangleMesh,
    #     cam_matrix: np.ndarray,
    #     frame_size: tuple[int, int],
    #     background_offset_m: float = 0.2,
    #     background_size_m: float = 5.0,
    #     background_z_m: Optional[float] = None,
    #     output_units: str = "mm",
    # ) -> np.ndarray:
    #     """Render ``mesh_cam`` plus a flat rectangular background to a depth image.

    #     Builds a scene composed of the input mesh (already in the camera
    #     frame) and a large flat rectangle parallel to the image plane,
    #     placed behind the object so that every camera ray hits something.
    #     The combined scene is then rasterized with the same pinhole
    #     ray-casting pipeline used by :meth:`render_mesh_depth`.

    #     Parameters
    #     ----------
    #     mesh_cam : ``TriangleMesh`` whose vertices are in the *camera*
    #         frame, in metres.
    #     cam_matrix : 3x3 pinhole intrinsic ``K`` (pixels).
    #     frame_size : ``(H, W)``.
    #     background_offset_m : distance (m) placed behind the mesh's
    #         furthest +Z vertex when ``background_z_m`` is not specified.
    #     background_size_m : edge length (m) of the square background plane.
    #     background_z_m : explicit camera-Z (m) for the background plane.
    #         Overrides ``background_offset_m`` when provided.
    #     output_units : ``"mm"`` (default) or ``"m"``.

    #     Returns
    #     -------
    #     ``(H, W)`` ``float32`` depth image. Pixels are guaranteed to be
    #     populated as long as the background plane covers the field of view.
    #     """
    #     h, w = frame_size

    #     # Determine where to place the background plane.
    #     if background_z_m is None:
    #         if mesh_cam is None or mesh_cam.is_empty():
    #             bg_z = max(1.0, float(background_offset_m))
    #         else:
    #             verts = np.asarray(mesh_cam.vertices, dtype=np.float64)
    #             z_max = float(verts[:, 2].max()) if verts.size else 0.0
    #             bg_z = z_max + float(background_offset_m)
    #     else:
    #         bg_z = float(background_z_m)

    #     if bg_z <= 0.0:
    #         raise ValueError(
    #             f"Background plane must lie in front of the camera (z>0), got z={bg_z}"
    #         )

    #     # Build a flat rectangular background facing the camera.
    #     half = 0.5 * float(background_size_m)
    #     bg_vertices = np.array(
    #         [
    #             [-half, -half, bg_z],
    #             [ half, -half, bg_z],
    #             [ half,  half, bg_z],
    #             [-half,  half, bg_z],
    #         ],
    #         dtype=np.float64,
    #     )
    #     # Two triangles, double-sided so orientation does not matter.
    #     bg_triangles = np.array(
    #         [
    #             [0, 1, 2], [0, 2, 3],
    #             [0, 2, 1], [0, 3, 2],
    #         ],
    #         dtype=np.int32,
    #     )
    #     bg_mesh = o3d.geometry.TriangleMesh(
    #         vertices=o3d.utility.Vector3dVector(bg_vertices),
    #         triangles=o3d.utility.Vector3iVector(bg_triangles),
    #     )

    #     # Compose the scene: input mesh + background plane.
    #     if mesh_cam is not None and not mesh_cam.is_empty():
    #         combined = o3d.geometry.TriangleMesh(mesh_cam) + bg_mesh
    #     else:
    #         combined = bg_mesh

    #     return DataSource.render_mesh_depth(
    #         combined,
    #         cam_matrix,
    #         frame_size,
    #         extrinsic=None,
    #         output_units=output_units,
    #     )

    def get_item_projected(
        self,
        index: int,
        debug: bool = False,
        method: str = "splat",
    ) -> dict[str, Any]:
        """Return one sample with the CAD geometry projected to a depth image.

        Adds two fields to the item returned by :meth:`get_item`:

        * ``"depth_cad_projected"`` — ``(H, W)`` ``float32`` depth in mm.
        * ``"cad_img_projected"``   — alias kept for compatibility with the
          legacy ``data_manager_pickle.py``.

        Parameters
        ----------
        method : ``"splat"`` (default) projects the sampled CAD point cloud
            with ``cv2.projectPoints`` and a z-buffer (sparse, distortion-
            aware). ``"raycast"`` rasterizes the CAD *mesh* with
            :meth:`render_mesh_depth` (dense, pinhole-only).
        use_icp : when ``True`` the ICP-refined CAD alignment is used
            (calls :meth:`load_icp_csv` if necessary).
        """
        if method not in ("splat", "raycast","open3d"):
            raise ValueError(f"Unknown projection method: {method!r}")


        item        = self.get_item(index, debug=False)

        depth_img   = item.get("depth_img")
        if depth_img is None:
            raise RuntimeError(
                f"Item {index}: depth_img is missing; cannot derive frame size."
            )
        h, w        = depth_img.shape[:2]
        cam_matrix, cam_dist_coeffs = self.get_intrinsics_matrix(item["json_path"])

        if method == "splat":

            cad_cloud = item.get("cad_pcd_aligned")
            if cad_cloud is None or len(cad_cloud.points) == 0:
                log.warning(
                    f"Item {index}: no aligned CAD cloud available; "
                    "depth_cad_projected will be zeros."
                )
                depth_cad_projected = np.zeros((h, w), dtype=np.float32)
            else:
                cad_points_cam       = np.asarray(cad_cloud.points, dtype=np.float32)
                depth_projected_path = item["depth_path"].replace('.bin', '_projected.png')  # for debug visualization of projected depth maps
                if os.path.exists(depth_projected_path):
                    log.info(f"Item {index}: loading pre-computed projected depth from {depth_projected_path}")
                    depth_cad_projected = cv2.imread(depth_projected_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                else:
                    depth_cad_projected = self.project_3d_to_camera_depth(
                        cad_points_cam, cam_matrix, cam_dist_coeffs, frame_size=(h, w)
                    )
                    #cv2.imwrite(depth_projected_path, depth_cad_projected.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])  # save projected depth for visualization  
        
        
        elif method == "raycast":

            t_camera_cad = item.get("t_camera_cad")
            mesh         = self.load_cad_mesh(item["cad_path"])
            if mesh is None or t_camera_cad is None:
                log.warning(
                    f"Item {index}: no mesh or pose for raycast; "
                    "depth_cad_projected will be zeros."
                )
                depth_cad_projected = np.zeros((h, w), dtype=np.float32)
            else:
                if np.linalg.norm(np.asarray(cam_dist_coeffs).ravel()) > 1e-3:
                    log.warning(
                        "render_mesh_depth assumes a pinhole camera; "
                        "non-zero distortion coefficients will be ignored."
                    )
                depth_projected_path = item["depth_path"].replace('.bin', '_projected.png')  # for debug visualization of projected depth maps
                if os.path.exists(depth_projected_path):
                    log.info(f"Item {index}: loading pre-computed projected depth from {depth_projected_path}")
                    depth_cad_projected = cv2.imread(depth_projected_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                else:                    
                    mesh_cam            = copy.deepcopy(mesh)
                    mesh_cam.transform(np.asarray(t_camera_cad, dtype=np.float64))
                    depth_cad_projected = self.render_mesh_depth( mesh_cam, cam_matrix, frame_size=(h, w), output_units="mm")
                    #cv2.imwrite(depth_projected_path, depth_cad_projected.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])  # save projected depth for visualization  

        else: # method == "open3d"

            t_camera_cad = item.get("t_camera_cad")
            mesh = self.load_cad_mesh(item["cad_path"])
            if mesh is None or t_camera_cad is None:
                log.warning(
                    f"Item {index}: no mesh or pose for raycast; "
                    "depth_cad_projected will be zeros."
                )
                depth_cad_projected = np.zeros((h, w), dtype=np.float32)
            else:
                if np.linalg.norm(np.asarray(cam_dist_coeffs).ravel()) > 1e-3:
                    log.warning(
                        "render_mesh_depth assumes a pinhole camera; "
                        "non-zero distortion coefficients will be ignored."
                    )
                depth_cad_projected = self.project_mesh_on_camera(
                    mesh, cam_matrix, cam_dist_coeffs, t_camera_cad, frame_size=(h, w)
                )
                # Convert the rendered RGB back to a depth image by taking the red channel.
                depth_cad_projected = depth_cad_projected[..., 0].astype(np.float32)

        item["depth_cad_projected"] = depth_cad_projected
        item["cad_img_projected"] = depth_cad_projected  # legacy alias
        item["projection_method"] = method

        if debug:
            depth_rs            = depth_img.astype(np.float32)
            err                 = np.zeros_like(depth_rs, dtype=np.float32)
            valid               = (depth_rs > 0) & (depth_cad_projected > 0)
            err[valid]          = depth_rs[valid] - depth_cad_projected[valid]
            self.show_subset(
                [
                    item.get("ir_left_img"),
                    item.get("ir_right_img"),
                    depth_rs,
                    depth_cad_projected,
                    err,
                ],
                [
                    "left (RS)",
                    "right (RS)",
                    "depth RS (mm)",
                    f"depth CAD ({method}) (mm)",
                    "error RS-CAD (mm)",
                ],
            )
            
            self.draw_point_clouds(item)
            plt.show()

        return item
    

    def get_item_and_scene_projected(
        self,
        index: int,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Return one sample with the CAD geometry + Background projected to a depth image.

        Adds two fields to the item returned by :meth:`get_item`:

        * ``"depth_cad_projected"`` — ``(H, W)`` ``float32`` depth in mm.
        * ``"cad_img_projected"``   — alias kept for compatibility with the
          legacy ``data_manager_pickle.py``.

        Parameters
        ----------
        method : ``"splat"`` (default) projects the sampled CAD point cloud
            with ``cv2.projectPoints`` and a z-buffer (sparse, distortion-
            aware). ``"raycast"`` rasterizes the CAD *mesh* with
            :meth:`render_mesh_depth` (dense, pinhole-only).
        use_icp : when ``True`` the ICP-refined CAD alignment is used
            (calls :meth:`load_icp_csv` if necessary).
        """
        item            = self.get_item(index, debug=False)

        depth_img       = item["depth_img"]
        h, w            = depth_img.shape[:2]

        #cam_matrix, _   = self.get_intrinsics_matrix(item["json_path"])
        cam_matrix, _   = self.get_intrinsics_matrix(item)

        # load object CAD
        mesh            = self.load_cad_mesh(item["cad_path"])
        t_camera_cad    = item["t_camera_cad"]
        mesh_cad_cam     = copy.deepcopy(mesh)
        mesh_cad_cam.transform(np.asarray(t_camera_cad, dtype=np.float64))

        # load background mesh
        background_cad_path = r"..\assets\thorlab matrix.STL"
        #mesh_backg      = self.load_background_mesh(mesh)   
        mesh_backg      = self.load_background_mesh(background_cad_path)                
        mesh_backg_cam  = copy.deepcopy(mesh_backg)
        mesh_backg_cam.transform(np.asarray(t_camera_cad, dtype=np.float64))  

        # Compose the scene: input mesh + background plane.
        #mesh_combined        = o3d.geometry.TriangleMesh(mesh_cad_cam) + mesh_backg_cam
        mesh_combined     = mesh_cad_cam + mesh_backg_cam

        # Render the depth of the combined scene.
        depth_scene       = self.render_mesh_depth(mesh_combined, cam_matrix, frame_size=(h, w), output_units="mm")

        # Object-only render (no background) for comparison / debugging.
        depth_obj       = self.render_mesh_depth(mesh_cad_cam, cam_matrix, frame_size=(h, w), output_units="mm")


        item["depth_scene_projected"] = depth_scene
        item["depth_cad_projected"] = depth_obj
        item["projection_method"] = 'raycast'
        # show meshes and depth maps for visual inspection
        #source.draw_point_clouds(item)

        if debug:


            # background -only render ( background) for comparison / debugging.
            depth_backg     = self.render_mesh_depth(mesh_backg_cam, cam_matrix, frame_size=(h, w), output_units="mm")
        

            depth_rs            = depth_img.astype(np.float32)
            err                 = np.zeros_like(depth_rs, dtype=np.float32)
            valid               = (depth_rs > 0) & (depth_scene > 0)
            err[valid]          = depth_rs[valid] - depth_scene[valid]
            self.show_subset(
                [
                    item.get("ir_left_img"),
                    item.get("ir_right_img"),
                    depth_obj,
                    depth_rs,
                    depth_scene,
                    err,
                ],
                [
                    "left (RS)",
                    "right (RS)",
                    "depth CAD (mm)",
                    "depth RS (mm)",
                    f"depth CAD + background (mm)",
                    "error RS-CAD (mm)",
                ],
            )
            
            self.draw_point_clouds(item)
            plt.show()  

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
        """Display a list of images in a 2-column grid."""
        img_num = len(img_list)
        col_num = min(img_num>>1, 3)
        row_num = int(np.ceil(img_num / col_num))
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
        axes = np.array(axes).reshape(row_num, col_num)
        for k in range(img_num):
            ri, ci = k // col_num, k % col_num
            img = img_list[k]
            if img is None:
                axes[ri, ci].axis("off")
                axes[ri, ci].set_title(ttl_list[k] + " (missing)")
                continue
            if img.ndim == 3 and img.shape[2] == 3:
                disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[ri, ci].imshow(disp)
            else:
                if 'ir' in ttl_list[k].lower():
                    vmax = 200
                elif 'depth' in ttl_list[k].lower():
                    vmax = 1000
                else:
                    vmax = np.percentile(img, 95) #if img.dtype == np.uint8 else 1000
                #vmax = 600 if k == 3 else vmax
                axes[ri, ci].imshow(img, cmap="gray" if img.ndim == 2 and img.dtype == np.uint8 else None, vmax = vmax )
            axes[ri, ci].set_title(ttl_list[k])
        # for k in range(img_num, row_num * col_num):
        #     axes[k // col_num, k % col_num].axis("off")
        if suptitle:
            fig.suptitle(suptitle)
        plt.show(block=False)

    def show_item(self, item: dict[str, Any]) -> None:
        """Show IR-left, IR-right, depth and RGB for a loaded item."""
        suptitle = (
            f"{Path(item['json_path']).parent.name} / capture {item['capture_idx']:04d}"
        )
        self.show_subset( [ item.get("ir_left_img"),  item.get("ir_right_img"),    item.get("depth_img"),  item.get("rgb_img"),  ],
            ["IR left", "IR right", "Depth [mm]", "RGB"],
            suptitle=suptitle,
        )

    def draw_point_clouds(self, item: dict[str, Any]) -> None:
        """Open an Open3D viewer with the depth + CAD clouds."""
        clouds                  = []
        depth_pcd               = item.get("depth_pcd_raw")
        cad_pcd                 = item.get("cad_pcd")
        cad_pcd_aligned         = item.get("cad_pcd_aligned")
        cad_pcd_aligned_icp     = item.get("cad_pcd_aligned_icp")

        if depth_pcd is not None and len(depth_pcd.points) > 0:
            depth_pcd = copy.deepcopy(depth_pcd)
            depth_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red
            clouds.append(depth_pcd)

        if cad_pcd is not None:
            cad_pcd = copy.deepcopy(cad_pcd)
            cad_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray (CAD origin)
            clouds.append(cad_pcd)

        if cad_pcd_aligned is not None:
            cad_pcd_aligned = copy.deepcopy(cad_pcd_aligned)
            cad_pcd_aligned.paint_uniform_color([0.0, 0.0, 1.0])  # blue
            clouds.append(cad_pcd_aligned)

        if cad_pcd_aligned_icp is not None:
            cad_pcd_aligned_icp = copy.deepcopy(cad_pcd_aligned_icp)
            cad_pcd_aligned_icp.paint_uniform_color([0.0, 0.0, 0.0])  # black
            clouds.append(cad_pcd_aligned_icp)

        if not clouds:
            log.warning("No point clouds to display.")
            return

        # Camera frustum (in the camera frame, so identity extrinsic).
        json_path = item.get("json_path")
        depth_img = item.get("depth_img")
        if json_path and depth_img is not None:
            try:
                K, _ = self.get_intrinsics_matrix(item)
                h, w = depth_img.shape[:2]
                frustum = self.make_camera_frustum(
                    K, frame_size=(h, w), scale=0.1, color=(1.0, 0.5, 0.0)
                )
                clouds.append(frustum)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Could not build camera frustum: {exc}")

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        clouds.append(axis)
        o3d.visualization.draw(clouds)

    @staticmethod
    def make_camera_frustum(
        cam_matrix: np.ndarray,
        frame_size: tuple[int, int],
        scale: float = 0.1,
        extrinsic: Optional[np.ndarray] = None,
        color: tuple[float, float, float] = (1.0, 0.5, 0.0),
    ) -> o3d.geometry.LineSet:
        """Build an Open3D ``LineSet`` representing a pinhole camera frustum.

        The frustum has its apex at the camera centre and a rectangular base
        sitting at ``z = scale`` (in the camera's coordinate system). Image
        corners are back-projected through ``cam_matrix`` so the frustum's
        aspect ratio and field-of-view match the actual camera intrinsics.

        Parameters
        ----------
        cam_matrix : 3x3 pinhole intrinsic ``K`` (pixels).
        frame_size : ``(H, W)`` of the image.
        scale : depth (in mesh units, typically metres) at which the base
            plane is drawn. Larger values yield a larger frustum.
        extrinsic : optional 4x4 ``world_T_camera`` transform. Defaults to
            identity, i.e. the frustum is drawn in the camera frame.
        color : RGB tuple in ``[0, 1]`` applied to every line.

        Returns
        -------
        ``open3d.geometry.LineSet`` — pass it to ``draw_geometries`` /
        ``draw`` along with your point clouds.
        """
        h, w = frame_size
        K = np.asarray(cam_matrix, dtype=np.float64).reshape(3, 3)
        K_inv = np.linalg.inv(K)

        # Image corners (u, v, 1): TL, TR, BR, BL.
        corners_uv = np.array(
            [
                [0.0, 0.0, 1.0],
                [w - 1, 0.0, 1.0],
                [w - 1, h - 1, 1.0],
                [0.0, h - 1, 1.0],
            ],
            dtype=np.float64,
        )
        # Back-project to rays and place each corner at depth = scale (z = scale).
        rays = (K_inv @ corners_uv.T).T          # (4, 3)
        rays /= rays[:, 2:3]                     # normalise so z == 1
        far_corners = rays * scale               # (4, 3) at z = scale

        # Vertices: 0 = camera centre, 1..4 = far-plane corners.
        verts = np.zeros((5, 3), dtype=np.float64)
        verts[1:] = far_corners

        if extrinsic is not None:
            E = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
            verts_h = np.hstack([verts, np.ones((5, 1))])
            verts = (E @ verts_h.T).T[:, :3]

        # Edges: apex -> 4 corners + the rectangular base + a small "up" tick.
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],   # apex to corners
            [1, 2], [2, 3], [3, 4], [4, 1],   # base rectangle
        ]

        # "Up" indicator: tick from the midpoint of the top edge upward.
        top_mid = 0.5 * (verts[1] + verts[2])
        # Direction perpendicular to top edge, pointing away from base centre.
        base_centre = verts[1:].mean(axis=0)
        up_dir = top_mid - base_centre
        up_norm = np.linalg.norm(up_dir)
        if up_norm > 1e-9:
            up_dir = up_dir / up_norm * (0.15 * scale)
            verts = np.vstack([verts, top_mid + up_dir])
            lines.append([1, len(verts) - 1])
            lines.append([2, len(verts) - 1])

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(verts)
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
        )
        return ls

    # ------------------------------------------------------------------
    # Grid-detection / pose recovery (ported from data_manager_pickle.py)
    # ------------------------------------------------------------------

    def get_grid_coordinates(
        self,
        img_left: np.ndarray,
        debug: bool = False,
    ) -> list[tuple[int, int]]:
        """Detect grid-of-dots peak coordinates in the centre of an IR image.

        Ported from ``DataSource.get_grid_coordinates`` in
        ``data_manager_pickle.py``: a Difference-of-Gaussians filter on a
        400x400 centre crop, then non-maximum suppression with a maximum
        filter and connected-component labelling.

        Returns a list of ``(x, y)`` peaks in the original image's pixel
        coordinates.
        """
        from scipy.ndimage import maximum_filter, label

        if img_left is None or img_left.size == 0:
            return []
        if img_left.ndim == 3:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

        h, w = img_left.shape[:2]
        center_h, center_w = h // 2, w // 2
        crop_size = 200
        img_gray = img_left[
            center_h - crop_size:center_h + crop_size,
            center_w - crop_size:center_w + crop_size,
        ].astype(np.float32)

        sigma1 = 2.0
        sigma2 = sigma1 * 1.6
        blur1 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma1)
        blur2 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma2)
        img_dog = -(blur1 - blur2)

        neighborhood_size = 37
        local_max = maximum_filter(img_dog, size=neighborhood_size) == img_dog

        threshold = 0.1
        background = img_dog < threshold
        erased = local_max.copy()
        erased[background] = False

        labeled, num_features = label(erased)

        peaks: list[tuple[int, int]] = []
        for i in range(1, num_features + 1):
            ys, xs = np.where(labeled == i)
            if xs.size == 0:
                continue
            peaks.append((int(np.mean(xs)), int(np.mean(ys))))

        log.info(f"get_grid_coordinates: {len(peaks)} dots found")

        coordinates: list[tuple[int, int]] = [
            (x + center_w - crop_size, y + center_h - crop_size) for (x, y) in peaks
        ]

        if debug:
            keypoints = [
                cv2.KeyPoint(x=float(x), y=float(y), size=1) for (x, y) in peaks
            ]
            img_with_kp = cv2.drawKeypoints(
                img_gray.astype(np.uint8),
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            plt.figure(figsize=(8, 6))
            plt.imshow(img_with_kp)
            plt.title(f"DoG peaks ({len(peaks)})")
            plt.show(block=False)

        return coordinates

    def match_grid_to_cad(
        self,
        img_left: np.ndarray,
        json_path: str,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate camera->CAD pose by matching detected grid dots to a CAD grid.

        Ported from ``DataSource.match_grid_to_cad`` in
        ``data_manager_pickle.py``. Uses iterative Hungarian matching +
        ``cv2.solvePnPRansac`` against a synthetic 30x30 CAD grid (25 mm
        spacing). Returns ``(rvec, tvec)`` in mm.

        Parameters
        ----------
        img_left : grayscale IR image used for dot detection.
        json_path : the session JSON path; used to fetch the camera
            intrinsics (``K``, distortion).
        """
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        # Synthetic CAD grid in mm.
        grid_spacing_mm = 25.0
        grid_size = (30, 30)
        grid_points_3d = np.array(
            [
                (x * grid_spacing_mm, y * grid_spacing_mm, 0.0)
                for x in range(-grid_size[0] // 2, grid_size[0] // 2 + 1)
                for y in range(-grid_size[1] // 2, grid_size[1] // 2 + 1)
            ],
            dtype=np.float32,
        )
        cad_points = grid_points_3d[:, :2]

        # Detect dots in the image.
        img_points_list = self.get_grid_coordinates(img_left, debug=debug)
        if len(img_points_list) < 4:
            log.warning(
                "match_grid_to_cad: too few image points "
                f"({len(img_points_list)}) for a robust pose estimate"
            )
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        img_points = np.asarray(img_points_list, dtype=np.float32)

        # Centre of detected points -> origin assumption.
        dist_to_mean        = np.sum(np.abs(img_points - np.mean(img_points, axis=0)), axis=1)
        img_index_center    = int(np.argmin(dist_to_mean))
        img_center_pixel    = img_points[img_index_center].copy()
        img_points_centered = img_points - img_center_pixel

        cam_matrix, dist_coeffs = self.get_intrinsics_matrix(json_path)

        img_radii           = np.linalg.norm(img_points_centered, axis=1)
        cad_radii           = np.linalg.norm(cad_points, axis=1)
        img_index_sorted    = np.argsort(img_radii)
        cad_index_sorted    = np.argsort(cad_radii)

        n_match_points = 7
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        tvec[2] = 500.0  # initial depth guess in mm

        success_any = False
        while True:
            if (
                n_match_points >= len(img_points)
                or n_match_points >= len(cad_points)
            ):
                log.warning("match_grid_to_cad: exhausted available points")
                break

            img_pts_cur = img_points[img_index_sorted[:n_match_points]]
            cad_pts_cur = grid_points_3d[cad_index_sorted[:n_match_points], :]

            cad_pts_proj = cv2.projectPoints(
                cad_pts_cur.reshape(-1, 1, 3),
                rvec, tvec, cam_matrix, dist_coeffs,
            )[0].reshape(-1, 2)

            distance_matrix = cdist(img_pts_cur, cad_pts_proj, metric="euclidean")
            gate_pix = max(grid_spacing_mm * 0.5, 50.0 / max(1.0, np.sqrt(n_match_points)))
            cost_matrix = distance_matrix.copy()
            cost_matrix[cost_matrix > gate_pix] = 1e6

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            valid_pair_mask = distance_matrix[row_ind, col_ind] <= gate_pix
            if int(valid_pair_mask.sum()) < 4:
                log.warning(
                    f"match_grid_to_cad: only {int(valid_pair_mask.sum())} valid pairs "
                    f"at radius={n_match_points}, gate={gate_pix:.1f}px"
                )
                n_match_points = n_match_points * 2 + 3
                continue

            row_ind = row_ind[valid_pair_mask]
            col_ind = col_ind[valid_pair_mask]

            img_pnp = img_pts_cur[row_ind].astype(np.float32)
            cad_pnp = cad_pts_cur[col_ind].astype(np.float32)

            success, rvec, tvec, inlier_mask = cv2.solvePnPRansac(
                cad_pnp,
                img_pnp,
                cam_matrix,
                dist_coeffs,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=10.0,
                iterationsCount=100,
                confidence=0.95,
            )
            if not success:
                log.warning(
                    f"match_grid_to_cad: solvePnP failed at radius={n_match_points}"
                )
                break

            success_any = True
            num_inliers = inlier_mask.size if inlier_mask is not None else 0
            log.info(
                f"match_grid_to_cad: radius={n_match_points}, inliers={num_inliers}"
            )

            if debug:
                plt.figure(figsize=(8, 6))
                plt.imshow(img_left, cmap="gray")
                plt.scatter(
                    img_pts_cur[:, 0], img_pts_cur[:, 1],
                    c="blue", label="Detected",
                )
                plt.scatter(
                    cad_pts_proj[:, 0], cad_pts_proj[:, 1],
                    c="red", label="Projected CAD",
                )
                plt.legend()
                plt.title(f"Radius={n_match_points}, Inliers={num_inliers}")
                plt.show(block=False)

            n_match_points = n_match_points * 2 + 3

        if not success_any:
            log.warning("match_grid_to_cad: no successful PnP iteration")
        return rvec.reshape(3), tvec.reshape(3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataSource(unittest.TestCase):
    """Tests mirroring ``data_manager_pickle.py`` but using the xls manifest.

    Mapping between the legacy class and this class:

      * ``init_directory(scene_id)``         -> ``init_directory()`` (xls)
      * ``out["left_img"]``                  -> ``out["ir_left_img"]``
      * ``out["right_img"]``                 -> ``out["ir_right_img"]``
      * ``out["depth_rs_img"]``              -> ``out["depth_img"]``
      * ``draw_scene(item, ...)``            -> ``draw_point_clouds(item)``
      * ``load_csv_with_icp_results(idx)``   -> ``load_icp_csv()`` + ``_icp_t_camera_cad(idx)``

    Tests for methods that have no equivalent in the xls-based DataSource
    (``get_item_projected``, ``get_item_icp_projected``, ``get_grid_coordinates``,
    ``match_grid_to_cad``) are kept under the same names but ``skipTest`` so
    the test list stays consistent across both modules.
    """

    def test_init_directory(self):
        source = DataSource()
        count = source.init_directory()
        log.info(f"Indexed {count} captures")
        self.assertTrue(count > 0)
        first = source.items[0]
        self.assertIn("json_path", first)
        self.assertIn("capture_idx", first)
        self.assertIn("vertices_path", first)

    def test_get_item(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)

        out = source.get_item(min(4, count - 1), debug=True)
        self.assertIn("depth_pcd_raw", out)
        self.assertIn("cad_pcd_aligned", out)
        self.assertEqual(out["t_camera_cad"].shape, (4, 4))
        self.assertTrue(out["depth_pcd_raw"] is not None and len(out["depth_pcd_raw"].points) > 0)

    def test_show_images(self):
        p = DataSource()
        img_num = p.init_directory()
        if img_num == 0:
            log.warning("No images found.")
            return
        for k in np.random.randint(0, img_num, size=min(2, img_num)):
            out = p.get_item(int(k), debug=False)
            self.assertTrue(out["ir_left_img"] is not None and out["ir_left_img"].size > 0)
            p.show_subset(
                [out["ir_left_img"], out["ir_right_img"], out["depth_img"]],
                ["left (RS)", "right (RS)", "depth RS (mm)"],
            )
        plt.show()

    def test_draw_scene(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        item = source.get_item(0, debug=False)
        # New API: draw_point_clouds is the equivalent of draw_scene.
        source.draw_point_clouds(item)

    def test_get_item_projected(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        item_id = np.random.randint(0, count)
        out = source.get_item_projected(item_id, debug=True)
        self.assertIn("depth_cad_projected", out)
        self.assertEqual(out["depth_cad_projected"].shape, out["depth_img"].shape)

    def test_get_item_projected_raycast(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        item_id = np.random.randint(0, count)
        out = source.get_item_projected(item_id, debug=True, method="raycast")
        self.assertIn("depth_cad_projected", out)
        self.assertEqual(out["projection_method"], "raycast")
        self.assertEqual(out["depth_cad_projected"].shape, out["depth_img"].shape)
        # Raycast produces a fully dense silhouette: at least one pixel hit.
        self.assertGreater(int(np.count_nonzero(out["depth_cad_projected"])), 0)

    def test_get_item_projected_open3d(self):
        source          = DataSource()
        count           = source.init_directory()
        self.assertTrue(count > 0)
        item_id         = np.random.randint(0, count)
        out = source.get_item_projected(item_id, debug=True, method="open3d")
        self.assertIn("depth_cad_projected", out)
        self.assertEqual(out["projection_method"], "open3d  ")
        self.assertEqual(out["depth_cad_projected"].shape, out["depth_img"].shape)
        # Open3D produces a fully dense silhouette: at least one pixel hit.
        self.assertGreater(int(np.count_nonzero(out["depth_cad_projected"])), 0)

    def test_get_item_and_scene(self):
        """Smoke-test :meth:`DataSource.render_scene_depth`.

        Loads one item, transforms the CAD mesh into the camera frame, and
        renders a depth image containing the object plus a flat background
        rectangle. Validates the output shape and that both the object and
        the background contribute pixels.
        """
        source          = DataSource()
        count           = source.init_directory()
        self.assertTrue(count > 0)

        item_id         = int(np.random.randint(0, count))
        item            = source.get_item(item_id, debug=False)

        depth_img       = item["depth_img"]
        self.assertIsNotNone(depth_img)
        h, w            = depth_img.shape[:2]

        cam_matrix, _   = source.get_intrinsics_matrix(item["json_path"])

        # load CAD
        mesh            = source.load_cad_mesh(item["cad_path"])
        t_camera_cad    = item["t_camera_cad"]
        self.assertIsNotNone(mesh)
        self.assertIsNotNone(t_camera_cad)
        mesh_cad_cam        = copy.deepcopy(mesh)
        mesh_cad_cam.transform(np.asarray(t_camera_cad, dtype=np.float64))

        # load background mesh
        mesh_backg      = source.load_background_mesh(mesh)
        self.assertIsNotNone(mesh_backg)
        self.assertIsNotNone(t_camera_cad)
        mesh_backg_cam        = copy.deepcopy(mesh_backg)
        mesh_backg_cam.transform(np.asarray(t_camera_cad, dtype=np.float64))  

        # Compose the scene: input mesh + background plane.
        #mesh_combined        = o3d.geometry.TriangleMesh(mesh_cad_cam) + mesh_backg_cam
        mesh_combined        = mesh_cad_cam + mesh_backg_cam

        # Render the depth of the combined scene.
        depth_scene         = source.render_mesh_depth(mesh_combined, cam_matrix, frame_size=(h, w), output_units="mm")

        self.assertEqual(depth_scene.shape, (h, w))
        self.assertEqual(depth_scene.dtype, np.float32)
        # Background should fill (almost) every pixel.
        self.assertGreater(int(np.count_nonzero(depth_scene)), int(0.9 * h * w))

        # Object-only render (no background) for comparison / debugging.
        depth_obj       = source.render_mesh_depth(mesh_cad_cam, cam_matrix, frame_size=(h, w), output_units="mm")
        obj_pixels      = int(np.count_nonzero(depth_obj))
        self.assertGreater(obj_pixels, 0)

        # background -only render ( background) for comparison / debugging.
        depth_backg     = source.render_mesh_depth(mesh_backg_cam, cam_matrix, frame_size=(h, w), output_units="mm")
        backg_pixels = int(np.count_nonzero(depth_backg))        

        # show meshes and depth maps for visual inspection
        #source.draw_point_clouds(item)

        source.show_subset(
            [depth_img.astype(np.float32), depth_obj, depth_scene, depth_backg],
            ["depth RS (mm)", "depth CAD only (mm)", "depth CAD+bg (mm)", "depth background only (mm)"],
            suptitle=f"render_scene_depth (item {item_id})",
        )
        plt.show()

    def test_get_item_and_scene_projected(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        item_id = np.random.randint(0, count)
        out = source.get_item_and_scene_projected(item_id, debug=True)
        self.assertIn("depth_cad_projected", out)
        self.assertEqual(out["projection_method"], "raycast")
        self.assertEqual(out["depth_cad_projected"].shape, out["depth_img"].shape)
        # Raycast produces a fully dense silhouette: at least one pixel hit.
        self.assertGreater(int(np.count_nonzero(out["depth_cad_projected"])), 0)

    def test_project_on_camera(self):
        # NOTE: Open3D's ``OffscreenRenderer`` (Filament backend) is not
        # supported on Windows wheels (it requires EGL headless). On Windows
        # we use a hidden ``Visualizer`` window + ``capture_screen_float_buffer``
        # to render a mesh to an RGB image with explicit pinhole intrinsics.

        image_width, image_height = 640, 480

        # Build pinhole intrinsics from a vertical FOV of 60 deg.
        fov_deg = 60.0
        fy = image_height / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
        fx = fy
        cx = image_width / 2.0
        cy = image_height / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            image_width, image_height, fx, fy, cx, cy
        )

        # Mesh to render.
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        mesh.compute_vertex_normals()

        # Look-at extrinsic: camera at (2, 2, 2), looking at origin, +Z up.
        eye = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        f = (center - eye)
        f /= np.linalg.norm(f)
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        # Open3D's pinhole convention: +Z forward, +Y down.
        # Build R such that R @ world = camera; rows are camera basis vectors.
        rotation = np.stack([s, -u, f], axis=0)
        translation = -rotation @ eye
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation

        vis = o3d.visualization.Visualizer()
        created = vis.create_window(
            visible=False, width=image_width, height=image_height
        )
        if not created:
            self.skipTest("Could not create Open3D window for offscreen render")

        try:
            vis.add_geometry(mesh)

            ctr = vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            params.intrinsic = intrinsic
            params.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters( params, allow_arbitrary=True)

            vis.poll_events()
            vis.update_renderer()
            float_img = vis.capture_screen_float_buffer(do_render=True)
        finally:
            vis.destroy_window()

        image_ndarray = np.asarray(float_img)
        self.assertEqual(image_ndarray.shape, (image_height, image_width, 3))

        # Save out the rendered RGB frame for inspection.
        rgb_uint8 = np.ascontiguousarray(
            (np.clip(image_ndarray, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        out_path = Path("projected_mesh_output.png").resolve()
        ok = o3d.io.write_image(str(out_path), o3d.geometry.Image(rgb_uint8))
        log.info(f"test_project_on_camera: wrote {out_path} (ok={ok})")
        self.assertTrue(out_path.exists())

    def test_show_icp_alignment(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        loaded = source.load_icp_csv()
        if not loaded:
            self.skipTest("No ICP CSV available in the manifest")
        item = source.get_item_with_icp(min(7, count - 1), debug=True)
        self.assertIsNotNone(item.get("t_camera_cad_icp"))

    def test_get_item_icp_projected(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        loaded = source.load_icp_csv()
        if not loaded:
            self.skipTest("No ICP CSV available in the manifest")
        item_id = min(4, count - 1)
        out = source.get_item_projected(item_id, debug=True, use_icp=True)
        self.assertIn("depth_cad_projected", out)
        self.assertEqual(out["depth_cad_projected"].shape, out["depth_img"].shape)

    def test_get_grid_coordinates(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        item_id = min(3, count - 1)
        item = source.get_item(item_id, debug=False)
        coordinates = source.get_grid_coordinates(item["ir_left_img"], debug=True)
        plt.show()
        self.assertTrue(len(coordinates) > 0)

    def test_match_grid_to_cad(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        item_id = min(2, count - 1)
        item = source.get_item(item_id, debug=False)

        rvec, tvec = source.match_grid_to_cad(
            item["ir_left_img"], item["json_path"], debug=True
        )
        # Build T_camera_cad from rvec/tvec (mm) for inspection.
        R, _ = cv2.Rodrigues(rvec)
        t_camera_cad = np.eye(4, dtype=np.float64)
        t_camera_cad[:3, :3] = R
        t_camera_cad[:3, 3] = tvec / 1000.0  # mm -> m

        # Optionally compare with the ICP-refined pose if available.
        if source.load_icp_csv():
            t_camera_cad_icp = source._icp_t_camera_cad(item["capture_idx"])
            log.info(f"Estimated T_camera_cad (grid):\n{t_camera_cad}")
            log.info(f"ICP T_camera_cad:\n{t_camera_cad_icp}")
        else:
            log.info(f"Estimated T_camera_cad (grid):\n{t_camera_cad}")

        plt.show()
        self.assertEqual(rvec.shape, (3,))
        self.assertEqual(tvec.shape, (3,))

    def test_load_and_show_png(self):
        "loads png data"
        fnames = ["original",'depth_rs','finetuned','isaactuned','pickle_gt','left']
        fdata  = []
        for f in fnames:
            png_path = f"{f}_3.png"
            if not Path(png_path).exists():
                self.skipTest(f"PNG file {png_path} does not exist")
            img_png = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            fdata.append(img_png)

        p = DataSource()
        p.show_subset(fdata,fnames)
        plt.show()            


def RunTest() -> None:
    tst = TestDataSource()
    # tst.test_init_directory()
    # tst.test_get_item() # ok
    # tst.test_show_images() # ok
    #tst.test_draw_scene() # ok
    #tst.test_get_item_projected() # ok
    #tst.test_get_item_projected_raycast() # ok
    #tst.test_get_item_projected_open3d()
    #tst.test_get_item_and_scene() # ok
    tst.test_get_item_and_scene_projected()

    #tst.test_project_on_camera()
    # tst.test_show_icp_alignment()
    #tst.test_get_item_icp_projected()
    # tst.test_get_grid_coordinates()
    # tst.test_match_grid_to_cad()
    #tst.test_load_and_show_png()
    

if __name__ == "__main__":
    RunTest()
