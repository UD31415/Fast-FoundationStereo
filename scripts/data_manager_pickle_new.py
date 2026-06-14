"""Dataset management for the ISAC (Pickle) scene-capture dataset.

This module is compatible with the data layout consumed by
``scripts/Uri_14_06_26 1.py``:

* A top-level Excel manifest (``data_14_6_26.xlsx``) lists every artefact
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
            ``robot_position`` — ``{x, y, z, rx, ry, rz}``.
            ``t_tooltip_cad``  — CAD pose in the tooltip frame.

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


log.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log.INFO)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EXCEL = (
    r"\\svm.realsenseai.com\RealSense_Validation\VIDB\Public\Stavush\Pickle\Data"
    r"\Data for model training\data_14_6_26.xlsx"
)
DEFAULT_STL_NAME = "cube_100x100x100"

LABEL_RAW = "raw data (json file)"
LABEL_ICP = "ICP"

# Image type strings used inside the per-session JSON.
IMG_TYPE_VERTICES = "Vertices"
IMG_TYPE_DEPTH = "Depth"
IMG_TYPE_IR_LEFT = "IR"
IMG_TYPE_IR_RIGHT = "RightIR"
IMG_TYPE_RGB = "RGB8"


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

    valid = xyz[:, 2] != 0.0
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
# Pose helpers (mirroring the convention used by ``Uri_14_06_26 1.py``)
# ---------------------------------------------------------------------------

# Rotation about X by pi to align the CAD coordinate convention with the camera.
_R_FIX = o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0.0, 0.0])
_T_FIX = np.eye(4, dtype=np.float64)
_T_FIX[:3, :3] = _R_FIX


def _invert_rigid(t: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 rigid transform."""
    r = t[:3, :3]
    p = t[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def compose_t_camera_cad(
    t_camera_tooltip: np.ndarray,
    t_tooltip_cad_raw: np.ndarray,
) -> np.ndarray:
    """Build ``t_camera_cad`` exactly like ``Uri_14_06_26 1.py``.

    Steps reproduced from the reference script:

    1. Apply the X-axis flip ``_T_FIX`` to ``t_tooltip_cad`` so the CAD frame
       matches the camera convention.
    2. Invert the hand-eye matrix to get ``t_tooltip_camera``.
    3. Recompose ``t_tooltip_cad`` as ``T_rot @ T_trans`` (translate-then-
       rotate) so the translation is interpreted in the original tooltip
       frame.
    4. Chain ``t_tooltip_camera @ t_tooltip_cad`` to obtain ``t_camera_cad``.
    """
    t_tooltip_cad = _T_FIX @ np.asarray(t_tooltip_cad_raw, dtype=np.float64)

    t_tooltip_camera = _invert_rigid(np.asarray(t_camera_tooltip, dtype=np.float64))

    t_rot = np.eye(4, dtype=np.float64)
    t_rot[:3, :3] = t_tooltip_cad[:3, :3]
    t_trans = np.eye(4, dtype=np.float64)
    t_trans[:3, 3] = t_tooltip_cad[:3, 3]

    t_tooltip_cad = t_rot @ t_trans
    return t_tooltip_camera @ t_tooltip_cad


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

    def __init__(self) -> None:
        self.excel_path: Optional[Path] = None
        self.stl_name: Optional[str] = None
        self.manifest_df: Optional[pd.DataFrame] = None

        # Per-session metadata, keyed by JSON path.
        self.sessions: dict[str, dict[str, Any]] = {}
        # Cached CAD point clouds keyed by cad_path.
        self._cad_cache: dict[str, o3d.geometry.PointCloud] = {}
        # Cached CAD triangle meshes (in metres) keyed by cad_path.
        self._cad_mesh_cache: dict[str, o3d.geometry.TriangleMesh] = {}

        # Flat per-capture index. Each entry has at least:
        #   ``json_path``, ``capture_idx``, plus eager copies of useful fields.
        self.items: list[dict[str, Any]] = []

        # Lazily loaded ICP refinements: ``capture_uid -> 4x4 matrix``.
        self.icp_table: Optional[pd.DataFrame] = None
        self.icp_path: Optional[Path] = None

        log.info("Pickle DataSource is defined")

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    # Discovery / indexing
    # ------------------------------------------------------------------

    def init_directory(
        self,
        excel_path: str | os.PathLike[str] = DEFAULT_EXCEL,
        stl_name: Optional[str] = DEFAULT_STL_NAME,
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

        if json_paths is not None:
            session_jsons: list[str] = [str(p) for p in json_paths]
            self.excel_path = None
            self.manifest_df = None
        else:
            self.excel_path = Path(excel_path)
            if not self.excel_path.exists():
                log.error(f"Manifest Excel does not exist: {self.excel_path}")
                return 0

            df = pd.read_excel(self.excel_path)
            self.manifest_df = df
            mask = df["Label"].astype(str) == LABEL_RAW
            if stl_name is not None:
                mask &= df["STL name"].astype(str) == str(stl_name)
            session_jsons = [str(p) for p in df.loc[mask, "Data Path"].tolist()]

        self.stl_name = stl_name

        for json_path in session_jsons:
            self._index_session(json_path)

        log.info(
            f"ISAC DataSource: indexed {len(self.items)} captures across "
            f"{len(self.sessions)} sessions"
        )
        return len(self.items)

    def _index_session(self, json_path: str) -> None:
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

        cad_path = scene.get("cad_path", "")
        t_camera_tooltip = np.array(
            scene.get("t_camera_tooltip", np.eye(4)), dtype=np.float64
        )
        intrinsics = scene.get("intrinsics", {}) or {}

        self.sessions[str(path)] = {
            "scene": scene,
            "cad_path": cad_path,
            "t_camera_tooltip": t_camera_tooltip,
            "intrinsics": intrinsics,
        }

        for ci, capture in enumerate(captures):
            paths = self._extract_capture_paths(capture)
            self.items.append(
                {
                    "json_path": str(path),
                    "capture_idx": ci,
                    **paths,
                    "robot_position": capture.get("robot_position", {}),
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
            p = img.get("path", "") or ""
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

    def _load_cad_pcd(self, cad_path: str) -> Optional[o3d.geometry.PointCloud]:
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
            number_of_points=90000,
            use_triangle_normal=True,
        )
        self._cad_cache[cad_path] = cad_pcd
        # Also cache the metres-scaled mesh for raycast rendering.
        self._cad_mesh_cache[cad_path] = mesh
        return cad_pcd

    def _load_cad_mesh(self, cad_path: str) -> Optional[o3d.geometry.TriangleMesh]:
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

    def get_intrinsics_matrix(self, json_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(K, dist_coeffs)`` for one session."""
        info = self.sessions.get(json_path, {})
        intr = info.get("intrinsics", {})
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

    def _icp_t_camera_cad(self, capture_idx: int) -> Optional[np.ndarray]:
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

        meta            = self.items[index]
        json_path       = meta["json_path"]
        session         = self.sessions[json_path]
        scene           = session["scene"]
        capture         = scene["captures"][meta["capture_idx"]]

        ir_left         = load_image_any(meta["ir_left_path"])
        ir_right        = load_image_any(meta["ir_right_path"])
        depth_img = load_image_any(meta["depth_path"])
        rgb = load_image_any(meta["rgb_path"]) if meta["rgb_path"] else None

        vertices_path = meta["vertices_path"]
        depth_pcd_raw: Optional[o3d.geometry.PointCloud] = None
        if vertices_path and os.path.exists(vertices_path):
            try:
                depth_pcd_raw = load_vertices_to_pcd(Path(vertices_path))
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Failed to load vertices {vertices_path}: {exc}")

        # Pose composition: camera <- CAD using Uri's convention.
        t_camera_tooltip = session["t_camera_tooltip"]
        t_tooltip_cad = np.array(capture.get("t_tooltip_cad", np.eye(4)), dtype=np.float64)
        t_camera_cad = compose_t_camera_cad(t_camera_tooltip, t_tooltip_cad)

        cad_pcd = self._load_cad_pcd(session["cad_path"])
        cad_pcd_aligned: Optional[o3d.geometry.PointCloud] = None
        if cad_pcd is not None:
            cad_pcd_aligned = copy.deepcopy(cad_pcd)
            cad_pcd_aligned.transform(t_camera_cad)

        # ICP-refined alignment, when available.
        t_camera_cad_icp = self._icp_t_camera_cad(meta["capture_idx"])
        cad_pcd_aligned_icp: Optional[o3d.geometry.PointCloud] = None
        if cad_pcd is not None and t_camera_cad_icp is not None:
            cad_pcd_aligned_icp = copy.deepcopy(cad_pcd)
            cad_pcd_aligned_icp.transform(t_camera_cad_icp)

        intrinsics = session["intrinsics"]
        item: dict[str, Any] = {
            "index": index,
            "json_path": json_path,
            "capture_idx": meta["capture_idx"],
            # paths
            "ir_left_path": meta["ir_left_path"],
            "ir_right_path": meta["ir_right_path"],
            "depth_path": meta["depth_path"],
            "rgb_path": meta["rgb_path"],
            "vertices_path": vertices_path,
            "cad_path": session["cad_path"],
            # images / clouds
            "ir_left_img": ir_left,
            "ir_right_img": ir_right,
            "depth_img": depth_img,
            "rgb_img": rgb,
            "depth_pcd_raw": depth_pcd_raw,
            "cad_pcd": cad_pcd,
            "cad_pcd_aligned": cad_pcd_aligned,
            "cad_pcd_aligned_icp": cad_pcd_aligned_icp,
            # transforms / metadata
            "t_camera_tooltip": t_camera_tooltip,
            "t_tooltip_cad": t_tooltip_cad,
            "t_camera_cad": t_camera_cad,
            "t_camera_cad_icp": t_camera_cad_icp,
            "robot_position": meta["robot_position"],
            "intrinsics": intrinsics,
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

    def get_item_projected(
        self,
        index: int,
        debug: bool = False,
        use_icp: bool = False,
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
        if method not in ("splat", "raycast"):
            raise ValueError(f"Unknown projection method: {method!r}")

        if use_icp and self.icp_table is None:
            self.load_icp_csv()

        item = self.get_item(index, debug=False)

        depth_img = item.get("depth_img")
        if depth_img is None:
            raise RuntimeError(
                f"Item {index}: depth_img is missing; cannot derive frame size."
            )
        h, w = depth_img.shape[:2]
        cam_matrix, dist_coeffs = self.get_intrinsics_matrix(item["json_path"])

        if method == "splat":
            cad_cloud = (
                item.get("cad_pcd_aligned_icp") if use_icp else item.get("cad_pcd_aligned")
            )
            if cad_cloud is None or len(cad_cloud.points) == 0:
                log.warning(
                    f"Item {index}: no aligned CAD cloud available; "
                    "depth_cad_projected will be zeros."
                )
                depth_cad_projected = np.zeros((h, w), dtype=np.float32)
            else:
                cad_points_cam = np.asarray(cad_cloud.points, dtype=np.float32)
                depth_cad_projected = self.project_3d_to_camera_depth(
                    cad_points_cam, cam_matrix, dist_coeffs, frame_size=(h, w)
                )
        else:  # raycast
            t_camera_cad = (
                item.get("t_camera_cad_icp") if use_icp else item.get("t_camera_cad")
            )
            mesh = self._load_cad_mesh(item["cad_path"])
            if mesh is None or t_camera_cad is None:
                log.warning(
                    f"Item {index}: no mesh or pose for raycast; "
                    "depth_cad_projected will be zeros."
                )
                depth_cad_projected = np.zeros((h, w), dtype=np.float32)
            else:
                if np.linalg.norm(np.asarray(dist_coeffs).ravel()) > 1e-3:
                    log.warning(
                        "render_mesh_depth assumes a pinhole camera; "
                        "non-zero distortion coefficients will be ignored."
                    )
                mesh_cam = copy.deepcopy(mesh)
                mesh_cam.transform(np.asarray(t_camera_cad, dtype=np.float64))
                depth_cad_projected = self.render_mesh_depth(
                    mesh_cam, cam_matrix, frame_size=(h, w), output_units="mm"
                )

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
            plt.show()
            self.draw_point_clouds(item)

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
        col_num = min(img_num, 3)
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
                vmax = np.percentile(img, 95) #if img.dtype == np.uint8 else 1000
                vmax = 600 if k == 3 else vmax
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
                K, _ = self.get_intrinsics_matrix(json_path)
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
        self.assertTrue(out["depth_pcd_raw"] is not None
                        and len(out["depth_pcd_raw"].points) > 0)

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
        self.skipTest(
            "get_grid_coordinates is not implemented in the xls-based DataSource"
        )

    def test_match_grid_to_cad(self):
        self.skipTest(
            "match_grid_to_cad is not implemented in the xls-based DataSource"
        )


def RunTest() -> None:
    tst = TestDataSource()
    # tst.test_init_directory()
    # tst.test_get_item()
    #tst.test_show_images()
    # tst.test_draw_scene()
    tst.test_get_item_projected()
    # tst.test_get_item_projected_raycast()
    # tst.test_show_icp_alignment()
    #tst.test_get_item_icp_projected()
    # tst.test_get_grid_coordinates()
    # tst.test_match_grid_to_cad()


if __name__ == "__main__":
    RunTest()
