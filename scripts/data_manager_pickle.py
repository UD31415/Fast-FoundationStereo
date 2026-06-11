"""
Dataset management for Pickle scene capture dataset.

Refactored to class style (similar to data_manager_inbolt.py):
- `DataSource.init_directory(...)` loads scene JSON and preloads all captures.
- `DataSource.get_item(index)` returns one sample.
- Includes unittest test functions and a small RunTest helper.
"""

from __future__ import annotations

import ast
import ast
import copy
import json
import logging as log
import matplotlib.pyplot as plt
import re
import unittest
from pathlib import Path
from typing import Any, Optional, Union
import os
import numpy as np
import open3d as o3d
import cv2
import pandas as pd
#from rosbags import image


log.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log.INFO)

#%% Helper functions for loading and processing data items.

def compose_t_camera_cad(
    t_camera_tooltip: np.ndarray,
    t_tooltip_cad_raw: np.ndarray,
) -> np.ndarray:
    """Build camera→CAD transform from camera→tooltip and tooltip→CAD.

    Recompose tooltip→CAD as translate-then-rotate to preserve the intended
    offset behavior before applying rotation.
    """
    t_rot = np.eye(4, dtype=np.float64)
    t_rot[:3, :3] = t_tooltip_cad_raw[:3, :3]
    t_trans = np.eye(4, dtype=np.float64)
    t_trans[:3, 3] = t_tooltip_cad_raw[:3, 3]

    t_tooltip_cad = t_rot @ t_trans
    #t_camera_cad = t_camera_tooltip @ t_tooltip_cad
    t_camera_cad = t_camera_tooltip @ t_tooltip_cad
    return t_camera_cad

def parse_resolution(filename: str) -> tuple[int, int]:
    """Return (width, height) extracted from a name like `..._1280x720_...`."""
    match = re.search(r"(\d+)x(\d+)", filename)
    if not match:
        raise ValueError(
            f"Cannot parse resolution from filename: {filename!r}. "
            "Expected a pattern like '1280x720'."
        )
    return int(match.group(1)), int(match.group(2))

def load_vertices_to_pcd(path: Path) -> o3d.geometry.PointCloud:
    """Load Vertices `.bin` and convert valid XYZ points into Open3D point cloud."""
    width, height = parse_resolution(path.name)
    pixel_data_bytes = width * height * 3 * 4  # float32 xyz

    raw = path.read_bytes()
    if len(raw) < pixel_data_bytes:
        raise ValueError(
            f"File too small: expected at least {pixel_data_bytes} bytes "
            f"for {width}x{height} vertices, got {len(raw)}."
        )

    xyz_flat    = np.frombuffer(raw[:pixel_data_bytes], dtype=np.float32).copy()
    xyz         = xyz_flat.reshape(height * width, 3)

    valid       = xyz[:, 2] != 0.0
    xyz         = xyz[valid] * 0.001  # mm -> m

    # Match previous convention used by this script.
    #xyz[:, 1] *= -1
    #xyz[:, 2] *= -1

    pcd             = o3d.geometry.PointCloud()
    pcd.points      = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return pcd

def read_bin_file_with_metadata_from_stav(path: str, fsize: tuple[int, int], fbpp: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Read binary image file with metadata appended after pixel data."""
    pixel_count = fsize[0] * fsize[1]
    pixel_data_bytes = pixel_count * (fbpp // 8)

    raw = Path(path).read_bytes()
    if len(raw) < pixel_data_bytes:
        raise ValueError(
            f"File too small: expected at least {pixel_data_bytes} bytes "
            f"for {fsize[0]}x{fsize[1]} image with {fbpp} bpp, got {len(raw)}."
        )

    img = np.frombuffer(raw[:pixel_data_bytes], dtype=np.uint16 if fbpp == 16 else np.uint8)
    img = img.reshape(fsize[1], fsize[0])  # height x width

    metadata_raw = raw[pixel_data_bytes:]
    metadata_str = metadata_raw.decode('utf-8', errors='ignore')
    metadata = json.loads(metadata_str) if metadata_str else {}

    return img, metadata

def read_bin_file_with_metadata(fname, Size=(640, 480), bpp=16):
    """Reads a binary file and returns it as a NumPy array.

    Args:
        fname (str): The name of the file to read.
        Size (tuple): The size of the image (width, height).
        bpp (int): The number of bits per pixel.

    Returns:
        np.ndarray: The image data as a NumPy array.
        np.ndarray: The meta data as a NumPy array uint8.
    """

    try:
        f = open(fname, 'rb')
    except IOError:
        print("Error: Could not open file", fname)
        return None, None

    if bpp > 32:
        dtype = np.uint64
    elif bpp > 16:
        dtype = np.uint32
    elif bpp > 8:
        dtype = np.uint16
    else:
        dtype = np.uint8

    try:
        A = np.fromfile(f, dtype=dtype, count=Size[0] * Size[1]).reshape(Size[::-1])
        H = np.fromfile(f, dtype=np.uint8, count=2500)
    except Exception as e:
        print(f"Error reading file {fname}: {e}")
        return None,None

    if bpp > 8:
        A = np.bitwise_and(A, 2**bpp - 1)

    f.close()
    string_meta = [chr(i) for i in H]
    meta_data   = ''.join(string_meta)
    #print(meta_data)

    return A,H

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

def interp_splat_close(u, v, z, image_size, kernel=3):
    h, w = image_size
    img = np.zeros((h, w), np.float32)
    lin = v.astype(np.int32) * w + u.astype(np.int32)
    buf = np.full(h * w, np.inf, np.float32)
    np.minimum.at(buf, lin, z.astype(np.float32))
    img = np.where(np.isfinite(buf), buf, 0).reshape(h, w)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)

def mesh_to_depth_o3d(mesh: o3d.geometry.TriangleMesh,
                      cam_matrix: np.ndarray,
                      image_size: tuple[int, int],
                      extrinsic: np.ndarray = np.eye(4)) -> np.ndarray:
    h, w = image_size
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(cam_matrix, o3d.core.Dtype.Float64),
        extrinsic_matrix=o3d.core.Tensor(extrinsic,  o3d.core.Dtype.Float64),
        width_px=w, height_px=h,
    )
    ans = scene.cast_rays(rays)
    depth = ans["t_hit"].numpy()                # distance along ray
    depth[~np.isfinite(depth)] = 0.0
    # convert ray distance to z-depth
    dirs = rays.numpy()[..., 3:6]
    depth = depth * dirs[..., 2] / np.linalg.norm(dirs, axis=-1)
    return depth.astype(np.float32)

#%% Main DataSource class definition.

class DataSource:
    """Class-based loader for scene captures and CAD alignment data."""

    def __init__(self):
        self.scene_json_path: Optional[Path] = None
        self.scene_data: dict[str, Any] = {}
        self.captures: list[dict[str, Any]] = []
        self.items: list[Optional[dict[str, Any]]] = []

        self.cad_path: Optional[str] = None
        self.t_camera_tooltip: Optional[np.ndarray] = None
        self.cad_pcd: Optional[o3d.geometry.PointCloud] = None
        self.df: Optional[pd.DataFrame] = None   # csv file with icp results, loaded on demand

        log.info("DataSource is defined")

    def __len__(self) -> int:
        return len(self.captures)
    
    def scene_json_load(self, scene_type = 1):
        if scene_type == 1:
            json_path = Path(__file__).with_name("scene.json")
        elif scene_type == 2:
            json_path = Path(r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0\2026_05\yg_pickle\2026-05-27--12-24-21\Pickle_Scene_Capture_336222073841\scene.json")
        elif scene_type == 3:
            json_path = Path(r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0\2026_05\yg_pickle\2026-05-27--13-50-40\Pickle_Scene_Capture_336222073841\scene.json")
        elif scene_type == 4:
            json_path = Path(r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0\2026_06\yg_pickle\2026-06-04--11-24-34\Pickle_Scene_Capture_336222073841\scene.json")
        else:
            raise ValueError(f"Unsupported scene type: {scene_type}")
        
        #json_path = Path(json_path)

        return json_path



    def init_directory(self, scene_type:int = 1) -> int:
        """Load scene metadata and optionally preload all captures into memory."""

        json_path           = self.scene_json_load(scene_type)

        # if len(scene_json_path) < 3:
        #     json_path = Path(__file__).with_name("scene.json")
        # else:
        #     json_path = Path(scene_json_path)

        if not json_path.exists():
            log.error(f"Scene JSON file not found: {json_path}")
            return 0

        self.scene_json_path = json_path
        with json_path.open("r", encoding="utf-8") as f:
            self.scene_data = json.load(f)

        self.cad_path = self.scene_data["cad_path"]
        self.cad_pcd  = self.load_cad_pcd(self.cad_path)        
        # self.t_camera_tooltip = np.array(self.scene_data["t_camera_tooltip"], dtype=np.float64)


        self.captures = list(self.scene_data.get("captures", []))
        self.items    = [None] * len(self.captures)
        for idx in range(len(self.captures)):
            self.items[idx] = self.load_capture_item(idx)

        log.info(f"DataSource: found {len(self.captures)} captures in {json_path}")
        return len(self.captures)

    def load_cad_pcd(self, cad_path: str) -> o3d.geometry.PointCloud:
        """Load CAD mesh and sample point cloud once (shared for all captures)."""
        cad_mesh = o3d.io.read_triangle_mesh(cad_path)
        if cad_mesh.is_empty():
            raise ValueError(f"Failed to load CAD mesh from: {cad_path}")

        # STL units are mm.
        cad_mesh.scale(0.001, center=(0, 0, 0))

        if not cad_mesh.has_vertex_normals():
            cad_mesh.compute_vertex_normals()

        num_samples     = 80000
        cad_pcd         = cad_mesh.sample_points_poisson_disk(number_of_points=num_samples,  use_triangle_normal=True)  

        # 2. Sample densely spaced points on the mesh surface
        # Increase 'number_of_points' to make the projection look completely solid
        #cad_pcd         = cad_mesh.sample_points_uniformly(number_of_points=num_samples)     
        return cad_pcd

    def get_vertices_path(self, capture: dict[str, Any]) -> str:
        """Extract vertices binary path for one capture."""
        for img in capture.get("images", []):
            if img.get("image_type") == "Vertices":
                path = img.get("path", "")
                if len(path) >= 3:
                    return path
                break
        raise ValueError("Capture does not contain a valid Vertices image path")

    def load_image_path(self, capture: dict[str, Any]) -> str:
        """Extract vertices binary path for one capture."""
        output_str = {"left_path": "", "right_path": "", "depth_path": "",  "rgb": "", "vertices_path": "",
                      "t_tooltip_cad": None, "device_configuration": None}
        
        t_tooltip_cad        = np.array(capture["t_tooltip_cad"], dtype=np.float64)
        device_configuration = capture.get("device_configuration", {})        

        for img in capture.get("images", []):

            if img.get("image_type") == "Depth":
                output_str["depth_path"] = img.get("path", "")
            elif img.get("image_type") == "IR":
                output_str["left_path"] = img.get("path", "")
            elif img.get("image_type") == "RightIR":
                output_str["right_path"] = img.get("path", "")
            elif img.get("image_type") == "RGB8":
                output_str["rgb_path"] = img.get("path", "")
            elif img.get("image_type") == "Vertices":                
                output_str["vertices_path"] = img.get("path", "")

        output_str["t_tooltip_cad"] = t_tooltip_cad
        output_str["device_configuration"] = device_configuration

        return output_str

    def load_capture_item(self, index: int) -> dict[str, Any]:
        """Create path to the data item for one capture index."""
        if len(self.captures) == 0:
            raise RuntimeError("DataSource was not initialized. Call init_directory first.")

        capture               = self.captures[index]
        output_str            = self.load_image_path(capture)
        return output_str
    
    def decode_fsize_bpp(self, path: str) -> tuple[int, int, int]:
        """Decode image size and bits-per-pixel from the filename."""

        if path.find("1280") != -1:
            fsize               = (1280,720)
        else:
            fsize               = (640,480)
        
        if path.find("Depth") != -1:
            fbpp                = 16
        elif path.find("IR") != -1 or path.find("RightIR") != -1:
            fbpp                = 8
        elif path.find("RGB") != -1:
            fsize               = (fsize[0],fsize[1],3)  # >>1 bug
            fsize               = (640,480,3)
            fbpp                = 8            
        else:
            fbpp               = 8
            log.error(f"Cannot determine image type from filename: {path}")

        return fsize, fbpp
    
    def load_image_data(self, path: str) -> Optional[np.ndarray]:
        """Load an image from the given path, returning None if it fails."""
        if not path or not os.path.exists(path):
            log.warning(f"Image path is invalid or does not exist: {path}")
            return None
        name, ext = os.path.splitext(path)
        if ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        elif ext.lower() in ['.bin']:
            "get an infrared depth frame from bin file"
            fsize, fbpp         = self.decode_fsize_bpp(path)
            img, img_metadata   = read_bin_file_with_metadata(path,fsize,fbpp)

        else:
            log.warning(f"Unsupported image format for path: {path}")
            return None            
        return img

    def load_item_data(self, index: int, use_aligned :bool = False) -> dict[str, Any]:
        """Create full data item for one capture index."""

        output_str          = self.items[index]
        left_img            = self.load_image_data(output_str['left_path'])
        right_img           = self.load_image_data(output_str['right_path'])
        depth_rs_img        = self.load_image_data(output_str['depth_path'])
        rgb_img             = self.load_image_data(output_str['rgb_path'])


        if left_img is None or right_img is None or depth_rs_img is None:
            log.warning(f"Failed to load sample {index}: {output_str}")
            return output_str        

        capture             = self.captures[index]
        vertices_path       = self.get_vertices_path(capture)
        t_camera_tooltip    = np.array(self.scene_data["t_camera_tooltip"], dtype=np.float64)
        t_tooltip_cad       = np.array(capture["t_tooltip_cad"], dtype=np.float64)
        #t_tooltip_cad       = np.linalg.inv(t_tooltip_cad)
        t_camera_cad        = compose_t_camera_cad(t_camera_tooltip, t_tooltip_cad)
        depth_pcd_raw       = load_vertices_to_pcd(Path(vertices_path))
        cad_pcd_aligned     = copy.deepcopy(self.cad_pcd)
        if use_aligned:
            cad_pcd_aligned.transform(t_camera_cad)

        return {
            "index": index,
            "capture": capture,
            "camera_vertices_path": vertices_path,
            "t_camera_tooltip": t_camera_tooltip,
            "t_tooltip_cad": t_tooltip_cad,
            "t_camera_cad": t_camera_cad,
            "cad_pcd": self.cad_pcd,
            "cad_pcd_aligned": cad_pcd_aligned,
            "depth_pcd_raw": depth_pcd_raw,
            "left_img": left_img,
            "right_img": right_img,
            "depth_rs_img": depth_rs_img,
            "rgb_img": rgb_img,
            "cad_img_projected": None, # cad projection data can be added later when needed
        }

    def get_item(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Return one loaded capture item by index."""
        if index < 0 or index >= len(self.captures):
            raise IndexError(f"Capture index out of range: {index}")

        item = self.load_item_data(index)
        if item is None:
            raise RuntimeError(f"Failed to build item at index {index}")

        if debug:
            cad_pcd             = copy.deepcopy(item["cad_pcd"])
            cad_pcd_aligned     = copy.deepcopy(item["cad_pcd_aligned"])
            depth_pcd_raw = copy.deepcopy(item["depth_pcd_raw"])

            cad_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            cad_pcd_aligned.paint_uniform_color([0.0, 0.0, 1.0])
            depth_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0])

            o3d.visualization.draw([cad_pcd, cad_pcd_aligned, depth_pcd_raw])

        return item
    
    def load_csv_with_icp_results(self, index: int) -> dict[int, dict[str, Any]]:
        """Load ICP results from a CSV file into a dictionary keyed by capture index."""

        csv_path = r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\Pickle\DeepCrunch_Results\a2991ebf-20ab-4887-a22d-cd84492517ef\icp_transformation_result.csv"
        if not os.path.exists(csv_path):
            log.warning(f"CSV file not found: {csv_path}")
            return {}
        
        # is it loaded already?
        if self.df is None:
            self.df      = pd.read_csv(csv_path)

        T_camera_cad_icp = np.array(ast.literal_eval(self.df.loc[index, 't_camera_cad_corrected']), dtype=np.float64)
        # icp_results = {}
        # for _, row in self.df.iterrows():
        #     try:
        #         index = int(row['index'])
        #         t_camera_cad_corrected = np.array(ast.literal_eval(row['t_camera_cad_corrected']), dtype=np.float64)
        #         icp_results[index] = {'t_camera_cad_corrected': t_camera_cad_corrected,       }
        #     except Exception as e:
        #         log.error(f"Error parsing row in CSV: {e}")
        return T_camera_cad_icp
    
    def load_csv_with_camera_tooltip_results(self, index: int) -> dict[int, dict[str, Any]]:
        """Load Camera Tooltip corrected results from a CSV file into a dictionary keyed by capture index."""

        csv_path = r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\Pickle\DeepCrunch_Results\a2991ebf-20ab-4887-a22d-cd84492517ef\icp_transformation_result.csv"
        if not os.path.exists(csv_path):
            log.warning(f"CSV file not found: {csv_path}")
            return {}
        
        # is it loaded already?
        if self.df is None:
            self.df      = pd.read_csv(csv_path)

        T_camera_tooltip = np.array(ast.literal_eval(self.df.loc[index, 't_camera_tooltip']), dtype=np.float64)

        return T_camera_tooltip    
    
    def get_item_with_icp(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Return one loaded capture item by index."""
        if index < 0 or index >= len(self.captures):
            raise IndexError(f"Capture index out of range: {index}")

        item                = self.load_item_data(index)
        if item is None:
            raise RuntimeError(f"Failed to build item at index {index}")
        
        # load help files that have icp results for visualization, but do not return them in the item dict since they are not needed for training or evaluation.
        if not debug:
            return item
        
        cad_pcd             = item["cad_pcd"]
        cad_pcd_aligned     = item["cad_pcd_aligned"]
        depth_pcd_raw       = item["depth_pcd_raw"]

    
        T_camera_cad_icp    = self.load_csv_with_icp_results(index)
        cad_pcd_aligned_icp = o3d.geometry.PointCloud(cad_pcd)
        cad_pcd_aligned_icp.transform(T_camera_cad_icp)

        camera_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(0.1))
        #camera_axis.transform(t_camera_tooltip)
        camera_axis_pcd.transform(T_camera_cad_icp)        


        cad_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # gray
        cad_pcd_aligned.paint_uniform_color([0, 0, 1]) # blue
        depth_pcd_raw.paint_uniform_color([1, 0, 0]) # red
        cad_pcd_aligned_icp.paint_uniform_color([0, 0, 0])#black            

        o3d.visualization.draw([cad_pcd, cad_pcd_aligned, depth_pcd_raw, cad_pcd_aligned_icp, camera_axis_pcd])

        return item    

    def get_camera_intrinsics_and_distortion(self) -> tuple[np.ndarray, np.ndarray]:
        """Read camera intrinsics/distortion from scene JSON."""
        intrinsics = self.scene_data.get("intrinsics", {})
        fx = float(intrinsics.get("fx", 1.0))
        fy = float(intrinsics.get("fy", 1.0))
        cx = float(intrinsics.get("ppx", 0.0))
        cy = float(intrinsics.get("ppy", 0.0))

        cam_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        coeffs = intrinsics.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
        dist_coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
        if dist_coeffs.size == 0:
            dist_coeffs = np.zeros((5,), dtype=np.float32)
        return cam_matrix, dist_coeffs

    def project_3d_to_camera_depth(self, points_3d_m: np.ndarray, cam_matrix: np.ndarray,  dist_coeffs: np.ndarray,
        frame_size: tuple[int, int],    ) -> np.ndarray:
        """Project 3D camera-frame points to a depth image (mm) using z-buffering."""
        if points_3d_m.ndim != 2 or points_3d_m.shape[1] != 3:
            raise ValueError("Input points_3d_m must have shape (N, 3)")

        h, w = frame_size
        valid = np.isfinite(points_3d_m).all(axis=1) & (points_3d_m[:, 2] > 1e-6)
        if not np.any(valid):
            log.error("No valid 3D points to project.")
            return np.zeros((h, w), dtype=np.float32)
        
        # transfer to mm
        points_3d_mm = points_3d_m * 1000.0

        pts = points_3d_mm[valid].astype(np.float32)
        projected_pts, _ = cv2.projectPoints(
            pts.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            cam_matrix.astype(np.float32),
            dist_coeffs.astype(np.float32),
        )

        uv    = projected_pts.reshape(-1, 2)
        u_idx = np.rint(uv[:, 0]).astype(np.int32)
        v_idx = np.rint(uv[:, 1]).astype(np.int32)

        in_bounds = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
        if not np.any(in_bounds):
            return np.zeros((h, w), dtype=np.float32)

        u_idx       = u_idx[in_bounds]
        v_idx       = v_idx[in_bounds]
        z_vals_mm   = (pts[in_bounds, 2]).astype(np.float32)

        # old way
        # interpolate depth values into the image using z-buffering
        # use regular grid indices for u and v to handle duplicates and out-of-order points correctly.        
        lin          = v_idx * w + u_idx
        depth_buffer = np.full(h * w, np.inf, dtype=np.float32)
        np.minimum.at(depth_buffer, lin, z_vals_mm)
        depth_projected = depth_buffer.reshape(h, w)
        depth_projected[~np.isfinite(depth_projected)] = 0.0

        # # do linear interpolation of depth values for points that project to the same pixel, taking the minimum (closest) depth.
        # depth_projected = interpolate_points_to_grid(u_idx, v_idx, z_vals_mm, image_size=(h, w))

        # depth_projected = interp_splat_close(u_idx, v_idx, z_vals_mm, image_size=(h, w), kernel=3)


        return depth_projected

    def get_item_projected(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Return one sample with CAD depth projected onto the camera image plane."""
        #item = self.get_item(index, debug=False)

        item                = self.load_item_data(index)
        if item is None:
            raise RuntimeError(f"Failed to build item at index {index}")     

        cad_pcd             = item["cad_pcd"]
        #cad_pcd_aligned     = item["cad_pcd_aligned"]
        depth_pcd_raw       = item["depth_pcd_raw"]
        t_tooltip_cad       = item["t_tooltip_cad"]
        depth_rs            = np.asarray(item["depth_rs_img"], dtype=np.float32)
        h, w                = depth_rs.shape[:2]  

        # use Camera Tooltip results
        #T_camera_tooltip        = self.load_csv_with_camera_tooltip_results(index)
        T_camera_tooltip2       = np.array([[0, -1, 0, 0], #degenerated tooltip to camera (assuming the target was formed relative to camera)
                                            [1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])


        #T_camera_cad = compose_t_camera_cad(t_camera_tooltip, t_tooltip_cad)        
        # create 180degree rotation around x axis to flip the CAD model since the Camera Tooltip results are based on flipped CAD model, and we want to visualize the effect of Camera Tooltip correction on the original CAD model.  
        T_flip_z_direction       = np.eye(4, dtype=np.float64)
        T_flip_z_direction[1, 1] = -1; T_flip_z_direction[2, 2] = -1

        T_camera_cad            = compose_t_camera_cad(T_camera_tooltip2, t_tooltip_cad)
        T_camera_cad            = T_flip_z_direction @ T_camera_cad
        T_camera_cad_icp        = T_camera_cad 
        cad_pcd_aligned         = o3d.geometry.PointCloud(cad_pcd)
        cad_pcd_aligned.transform(T_camera_cad_icp)        # T_cam_cad                 


        # CAD points already transformed into camera frame by t_camera_cad.
        cad_points_cam          = np.asarray(cad_pcd_aligned.points, dtype=np.float32)
        cam_matrix, dist_coeffs = self.get_camera_intrinsics_and_distortion()

        projected_path = item["camera_vertices_path"].replace(".bin", "_cad_projected.png")
        if os.path.exists(projected_path):
            depth_cad_projected = cv2.imread(projected_path, cv2.IMREAD_UNCHANGED)
            
        else:
            depth_cad_projected = self.project_3d_to_camera_depth( cad_points_cam,  cam_matrix,  dist_coeffs,  frame_size=(h, w))
            # cv2.imwrite(
            #     projected_path,
            #     depth_cad_projected.astype(np.uint16),
            #     [cv2.IMWRITE_PNG_COMPRESSION, 0],
            # )


        depth_cad_projected         = depth_cad_projected.astype(np.float32)
        item["cad_img_projected"]   = depth_cad_projected

        if debug:
            err = np.zeros_like(depth_rs, dtype=np.float32)
            valid = (depth_rs > 0) & (depth_cad_projected > 0)
            err[valid] = depth_rs[valid] - depth_cad_projected[valid]
            self.show_subset(
                [item["left_img"], item["right_img"], depth_rs, depth_cad_projected, err],
                ["left (RS)", "right (RS)", "depth RS (mm)", "depth CAD projected (mm)", "error RS-CAD (mm)"],vmax = 500,
            )
            plt.show()

        if debug:
            camera_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(0.1))
            #camera_axis_pcd.transform(T_flip_z_direction)        

            cad_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # gray
            depth_pcd_raw.paint_uniform_color([1, 0, 0]) # red
            cad_pcd_aligned.paint_uniform_color([0, 0, 0])#black            

            o3d.visualization.draw([cad_pcd, depth_pcd_raw, cad_pcd_aligned, camera_axis_pcd])

        return item

    def get_item_icp_projected(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Return one sample with CAD depth after ICP alignment projected onto the camera image plane."""

        item                = self.load_item_data(index, use_aligned=True)
        if item is None:
            raise RuntimeError(f"Failed to build item at index {index}")
    
        
        cad_pcd             = item["cad_pcd"]
        cad_pcd_aligned     = item["cad_pcd_aligned"]
        depth_pcd_raw       = item["depth_pcd_raw"]
        depth_rs            = np.asarray(item["depth_rs_img"], dtype=np.float32)
        h, w                = depth_rs.shape[:2]

        # use ICP results
        T_camera_cad_icp        = self.load_csv_with_icp_results(index)
        # create 180degree rotation around x axis to flip the CAD model since the ICP results are based on flipped CAD model, and we want to visualize the effect of ICP correction on the original CAD model.  
        T_flip_z_direction       = np.eye(4, dtype=np.float64)
        T_flip_z_direction[1, 1] = -1; T_flip_z_direction[2, 2] = -1
        T_camera_cad_icp        = T_camera_cad_icp #@ T_flip_z_direction
        #T_camera_cad_icp    = np.linalg.inv(T_camera_cad_icp) # apply ICP correction to the original camera-cad transform
        cad_pcd_aligned_icp     = o3d.geometry.PointCloud(cad_pcd)
        cad_pcd_aligned_icp.transform(T_camera_cad_icp)        # T_cam_cad

        # CAD points already transformed into camera frame by t_camera_cad.
        cad_points_cam          = np.asarray(cad_pcd_aligned_icp.points, dtype=np.float32)
        cam_matrix, dist_coeffs = self.get_camera_intrinsics_and_distortion()

        projected_path = item["camera_vertices_path"].replace(".bin", "_cad_projected.png")
        if os.path.exists(projected_path):
            depth_cad_projected = cv2.imread(projected_path, cv2.IMREAD_UNCHANGED)
            
        else:
            depth_cad_projected = self.project_3d_to_camera_depth(cad_points_cam, cam_matrix, dist_coeffs, frame_size=(h, w))
            
            #depth_cad_projected = mesh_to_depth_o3d(cad_pcd,cam_matrix, image_size=(h, w))

            # cv2.imwrite(
            #     projected_path,
            #     depth_cad_projected.astype(np.uint16),
            #     [cv2.IMWRITE_PNG_COMPRESSION, 0],
            # )

        depth_cad_projected         = depth_cad_projected.astype(np.float32)
        item["cad_img_projected"]   = depth_cad_projected

        if debug:
            err = np.zeros_like(depth_rs, dtype=np.float32)
            valid = (depth_rs > 0) & (depth_cad_projected > 0)
            err[valid] = depth_rs[valid] - depth_cad_projected[valid]
            self.show_subset(
                [item["left_img"], item["right_img"], depth_rs, depth_cad_projected, err],
                ["left (RS)", "right (RS)", "depth RS (mm)", "depth CAD projected (mm)", "error RS-CAD (mm)"],vmax = 500,
            )
            plt.show()

        if debug:
            camera_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(0.1))
            #camera_axis_pcd.transform(T_flip_z_direction)        

            cad_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # gray
            cad_pcd_aligned.paint_uniform_color([0, 0, 1]) # blue
            depth_pcd_raw.paint_uniform_color([1, 0, 0]) # red
            cad_pcd_aligned_icp.paint_uniform_color([0, 0, 0])#black            

            o3d.visualization.draw([cad_pcd, cad_pcd_aligned, depth_pcd_raw, cad_pcd_aligned_icp, camera_axis_pcd])


        return item

# ----------------------------------
# Recover camera pose.
# ----------------------------------

    def get_grid_coordinates(self, img_left, debug:bool=False) -> np.ndarray:
        """Estimate camera pose using ICP between CAD and depth point clouds."""
        # 1. Load the image
        from scipy.ndimage import maximum_filter, label

        # crop 400x400 center region for better blob detection
        h, w = img_left.shape[:2]
        center_h, center_w = h // 2, w // 2
        crop_size = 200
        img_gray = img_left[center_h - crop_size:center_h + crop_size, center_w - crop_size:center_w + crop_size].astype(np.float32) 


        sigma1 = 2.0
        sigma2 = sigma1 * 1.6
        blur1 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma1)
        blur2 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma2)
        img_dog = -(blur1 - blur2)

        # 2. find_local_maxima(filtered_image, neighborhood_size=5, threshold=0.1):
        """
        Finds local maxima in the filtered image within a specified neighborhood.
        """
        # 1. Apply a maximum filter in a local neighborhood
        neighborhood_size = 37
        local_max = maximum_filter(img_dog, size=neighborhood_size) == img_dog
        
        # 2. Filter out background noise by ensuring peaks are above a certain intensity
        threshold = 0.1
        background = (img_dog < threshold)
        erased_background = local_max.copy()
        erased_background[background] = False
        
        # 3. Label the unique peaks and get their coordinates
        labeled, num_features = label(erased_background)
    
        # Find the center of mass (coordinates) for each labeled maximum
        # scipy's label coordinates are returned as (y, x)
        peaks = []; 
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            y, x = np.where(mask)
            peaks.append((int(np.mean(x)), int(np.mean(y)))) # Store as (x, y)
            

        # 6. Extract and print the (x, y) coordinates
        print(f"Total dots found: {len(peaks)}")
        print("-" * 30)
        print("Index |   X-Coord  |   Y-Coord")
        print("-" * 30)

        coordinates = []
        for i, kp in enumerate(peaks):
            x, y = kp
            coordinates.append((x+center_w-crop_size, y+center_h-crop_size))
            print(f"{i+1:5d} | {x:10.2f} | {y:10.2f}")

        if not debug:
            return coordinates


        # 7. Optional: Visualize the results
        keypoints = []
        for i, kp in enumerate(peaks):
            x, y = kp
            keypoints.append(cv2.KeyPoint(x=float(x), y=float(y), size=1))

        # Draw detected blobs as red circles
        img_with_keypoints = cv2.drawKeypoints(
            img_gray.astype(np.uint8), 
            keypoints, 
            np.array([]), 
            (0, 0, 255), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Save and show the output image
        #cv2.imwrite('detected_dots.png', img_with_keypoints)
        plt.figure(figsize=(8, 6))
        plt.imshow(img_with_keypoints)
        plt.show(block=False)
        return coordinates

    def match_grid_to_cad(self, img_left:np.ndarray) -> np.ndarray:
        """Match detected grid points to CAD model points to estimate camera pose."""
        # This is a placeholder for the actual matching logic, which could involve:
        # 1. Generating a synthetic depth image from the CAD model using the current pose estimate.
        # 2. Detecting the same grid pattern in the synthetic depth image.
        # 3. Using a RANSAC-based approach to find the best transformation that aligns the detected grid points with the corresponding CAD points.
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        # For demonstration, we will return an identity matrix as a placeholder.
        # crearte 2D grid
        grid_spacing_mm     = 25.0
        grid_size           = (30,30) # 30x30 grid
        grid_points_3d      = []
        for x in range(-grid_size[0]//2, grid_size[0]//2 + 1):
            for y in range(-grid_size[1]//2, grid_size[1]//2 + 1):
                grid_points_3d.append((x * grid_spacing_mm, y * grid_spacing_mm, 0))
        
        grid_points_3d   = np.array(grid_points_3d, dtype=np.float32) # (N, 3)
        cad_points      = grid_points_3d[:,:2] # only x,y
        dist_points     = np.sum(np.abs(cad_points), axis=1) # center the cad points
        cad_index_center = np.argmin(dist_points) # 

        # image points
        img_points      = self.get_grid_coordinates(img_left, debug=True)
        img_points      = np.asarray(img_points, dtype=np.float32)

        # decide about the center of coordinates for image points. Most close point to mean
        dist_points     = np.sum(np.abs(img_points - np.mean(img_points, axis=0)), axis=1) # center the image points

        # the closest point to the center of the image points will be the origin of the grid, and we will assume it corresponds to the origin of the CAD model for simplicity. In practice, you would need a more robust method to determine the correct correspondences between image points and CAD points.
        img_index_center = np.argmin(dist_points) #
        img_center_pixel = img_points[img_index_center].copy()  # save original pixel offset for later PnP
        img_points_center= img_points - img_center_pixel  # center the image points

        # srarting from the img_point center define a radius and start matching points in the radius to the cad points around cad_center, 
        # and find the best matching transformation using RANSAC or a similar method. 
        # Using the best match estimate the camera pose relative to the CAD model.
        # Increase the radius iteratively until a sufficient number of matches are found or a maximum radius is reached.
        # Improve matches gradually by refining the pose estimate and re-projecting the CAD points to the image plane, then re-matching with the detected image points, and iterating until convergence.

        
        cam_matrix, dist_coeffs = self.get_camera_intrinsics_and_distortion()

        # --- 1. Initial scale (pixels per mm) from smallest non-zero radii ---
        img_radii       = np.linalg.norm(img_points_center, axis=1)
        cad_radii       = np.linalg.norm(cad_points, axis=1)
        img_index_sorted = np.argsort(img_radii)
        cad_index_sorted = np.argsort(cad_radii)

        # start from the closest points to the center 
        n_match_points   = 7
        rvec           = np.zeros(3, dtype=np.float64)
        tvec           = np.zeros(3, dtype=np.float64)
        tvec[2]        = 500.0 # initial depth guess in mm

        is_matched      = False
        while not is_matched:

            # protect
            if n_match_points >= len(img_points) or n_match_points >= len(cad_points):
                log.warning("match_grid_to_cad: not enough points to match")
                break

            # points in the current radius
            img_points_current = img_points[img_index_sorted[:n_match_points]] 
            cad_points_current = grid_points_3d[cad_index_sorted[:n_match_points],:]

            # project cad on image using current estimate of scale and rotation, and find the best matches between projected cad points and detected image points using nearest neighbor search within a certain radius.
            cad_points_projected = cv2.projectPoints(
                cad_points_current.reshape(-1, 1, 3), rvec, tvec, 
                cam_matrix, dist_coeffs )[0].reshape(-1, 2)
            
            # try to match using minimal distance and a reasonable threshold based on the current scale estimate (e.g., 5 pixels)
            # create a distance matrix between the points and find the best matches
            # Hungarian algorithm: globally optimal one-to-one assignment that minimizes total reprojection distance.
            distance_matrix = cdist(img_points_current, cad_points_projected, metric='euclidean')

            # Gate by a distance threshold so impossible pairs cannot be forced into the assignment.
            # Use a generous tolerance early (when the pose is rough) and tighten it as more points are matched.
            gate_pix = max(grid_spacing_mm * 0.5, 50.0 / max(1.0, np.sqrt(n_match_points)))
            cost_matrix = distance_matrix.copy()
            cost_matrix[cost_matrix > gate_pix] = 1e6  # large penalty for pairs beyond the gate

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Keep only assignments whose original distance is within the gate (drop the penalized ones).
            valid_pair_mask = distance_matrix[row_ind, col_ind] <= gate_pix
            if int(valid_pair_mask.sum()) < 4:
                log.warning(
                    f"match_grid_to_cad: hungarian found only {int(valid_pair_mask.sum())} valid pairs "
                    f"at radius={n_match_points}, gate={gate_pix:.1f}px"
                )
                n_match_points = n_match_points * 2 + 3
                continue

            row_ind             = row_ind[valid_pair_mask]
            col_ind             = col_ind[valid_pair_mask]

            img_points_pnp     = img_points_current[row_ind].astype(np.float32)
            cad_points_matched = cad_points_current[col_ind].astype(np.float32)
            # use the best matches to estimate a new transformation using RANSAC or a similar method
            # PnP estimation with RANSAC to find the best pose that aligns the matched points, and count the number of inliers based on a reprojection error threshold (e.g., 5 pixels).
            success, rvec, tvec, inlier_mask = cv2.solvePnPRansac(
                cad_points_matched,
                img_points_pnp,
                cam_matrix,
                dist_coeffs,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=10.0,
                iterationsCount=100,
                confidence=0.95
            )
            if not success:
                log.warning(f"match_grid_to_cad: solvePnP failed radius={n_match_points}")
                break

            # find the best match with the most inliers, and if the number of inliers is sufficient (e.g., at least 4), consider it a successful match and use the estimated transformation as the initial pose estimate for further refinement.
            num_inliers = inlier_mask.size if success else 0
            log.info(f"match_grid_to_cad: radius={n_match_points}, inliers={num_inliers}")

            mask_index  = inlier_mask.flatten()

            plt.figure(figsize=(8, 6))
            plt.imshow(img_left, cmap='gray')
            plt.scatter(img_points_current[:, 0], img_points_current[:, 1], c='blue', label='Detected Points')
            plt.scatter(cad_points_projected[:, 0], cad_points_projected[:, 1], c='red', label='Projected CAD Points')
            #plt.scatter(cad_points_matched[mask_index, 0], cad_points_matched[mask_index, 1], marker='o', c='green', label='Inlier Points')
            plt.legend()
            plt.title(f"Radius={n_match_points}, Inliers={num_inliers}")
            plt.show(block=False)

            n_match_points = n_match_points*2 + 3 # increase the radius to include more points in the next iteration

        return rvec, tvec

# ----------------------------------
# Visualization helpers.
# ----------------------------------

    def create_camera_frustum_lineset(
        self,
        image_size: tuple[int, int],
        frustum_depth_m: float = 0.15,
        camera_transform: Optional[np.ndarray] = None,
        color: tuple[float, float, float] = (1.0, 1.0, 0.0),
    ) -> o3d.geometry.LineSet:
        """Create a wireframe camera frustum (polygon + rays) as an Open3D LineSet."""
        intrinsics = self.scene_data.get("intrinsics", {})
        fx = float(intrinsics.get("fx", 1.0))
        fy = float(intrinsics.get("fy", 1.0))
        cx = float(intrinsics.get("ppx", image_size[0] * 0.5))
        cy = float(intrinsics.get("ppy", image_size[1] * 0.5))

        width, height = image_size
        if width <= 1 or height <= 1:
            width, height = 1280, 720

        # Frustum points in camera coordinates.
        z = float(frustum_depth_m)
        corners_px = np.array(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float64,
        )
        corners = np.zeros((4, 3), dtype=np.float64)
        corners[:, 0] = (corners_px[:, 0] - cx) / max(fx, 1e-9) * z
        corners[:, 1] = (corners_px[:, 1] - cy) / max(fy, 1e-9) * z
        corners[:, 2] = z

        points = np.vstack([np.zeros((1, 3), dtype=np.float64), corners])
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # rays
            [1, 2], [2, 3], [3, 4], [4, 1],  # image-plane polygon
        ]

        # Optional world transform.
        if camera_transform is not None:
            t = np.asarray(camera_transform, dtype=np.float64).reshape(4, 4)
            p_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
            points = (p_h @ t.T)[:, :3]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        return line_set

    def draw_scene(
        self,
        item_or_index: Union[int, dict[str, Any]],
        show_depth_cloud: bool = False,
        axis_size_m: float = 0.08,
        frustum_depth_m: float = 0.12,
    ) -> None:
        """Draw scene with object cloud, camera polygon(frustrum) and both RGB axes.

        - Object cloud: CAD point cloud transformed by `t_tooltip_cad` (gray)
        - Camera polygon: frustum wireframe (yellow)
        - Object axis: transformed by `t_tooltip_cad` (RGB)
        - Camera axis: transformed by `t_camera_tooltip` (RGB)
        """
        item = self.get_item(item_or_index, debug=False) if isinstance(item_or_index, int) else item_or_index

        t_tooltip_cad    = np.asarray(item["t_tooltip_cad"], dtype=np.float64)
        t_camera_tooltip = np.asarray(item["t_camera_tooltip"],  dtype=np.float64)
        t_camera_cad     = np.asarray(item["t_camera_cad"], dtype=np.float64)

        # object to be manipulated, in the world coordinate system (same as camera), transformed by tooltip→CAD
        object_pcd       = copy.deepcopy(item["cad_pcd"])
        #object_pcd.transform(t_tooltip_cad)
        object_pcd.paint_uniform_color([0.6, 0.6, 0.6])

        # Infer image size from vertices filename when available.
        image_size = (1280, 720)
        cam_vertices_path = item.get("camera_vertices_path", "")
        if isinstance(cam_vertices_path, str) and len(cam_vertices_path) > 0:
            try:
                image_size = parse_resolution(Path(cam_vertices_path).name)
            except Exception:
                pass

        camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_size_m))
        #camera_axis.transform(t_camera_tooltip)
        camera_axis.transform(t_camera_cad)

        object_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_size_m))
        #object_axis.transform(t_tooltip_cad)

        frustum = self.create_camera_frustum_lineset(
            image_size=image_size,
            frustum_depth_m=frustum_depth_m,
            camera_transform=t_camera_cad, #t_camera_tooltip,
            color=(1.0, 1.0, 0.0),
        )

        geometries: list[o3d.geometry.Geometry] = [object_pcd, frustum, camera_axis, object_axis]

        if show_depth_cloud and "depth_pcd_raw" in item and item["depth_pcd_raw"] is not None:
            depth_pcd = copy.deepcopy(item["depth_pcd_raw"])
            depth_pcd.paint_uniform_color([1.0, 0.0, 1.0])
            geometries.append(depth_pcd)

        o3d.visualization.draw(geometries)
    
    def show_subset(self, img_list, ttl_list, vmin=None, vmax=None, save_path='', fig_name=''):
        """Display a list of images in a grid."""
        img_num = len(img_list)
        col_num = min(img_num, 3)
        row_num = int(np.ceil(img_num / col_num))
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
        axes = np.array(axes).reshape(row_num, col_num)
        for k in range(img_num):
            ri, ci = k // col_num, k % col_num
            if img_list[k] is None: continue
            vmax = 50 if k in [0,1] else vmax
            axes[ri, ci].imshow(img_list[k], vmin=vmin, vmax=vmax)
            axes[ri, ci].set_title(ttl_list[k])
        for k in range(img_num, row_num * col_num):
            axes[k // col_num, k % col_num].axis('on')
        if save_path and os.path.exists(save_path):
            fig.savefig(os.path.join(save_path, fig_name + ".png"))
        plt.show(block=False)    
    
    def save_to_ply(self, points: np.ndarray, filename: str):
        """Save a point cloud to a PLY file for visualization."""
        if not self.save_point_cloud:
            return
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for x, y, z in points:
                f.write(f'{x} {y} {z}\n')
        log.info(f"Saved point cloud to {filename}")    


class TestDataSource(unittest.TestCase):
    """Basic tests for class-based scene data manager."""

    def test_init_directory(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)

    def test_get_item(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)

        out = source.get_item(4, debug=True)
        self.assertIn("depth_pcd_raw", out)
        self.assertIn("cad_pcd_aligned", out)
        self.assertEqual(out["t_camera_cad"].shape, (4, 4))
        self.assertTrue(len(out["depth_pcd_raw"].points) > 0)

    def test_show_images(self):
        p       = DataSource()
        img_num = p.init_directory()
        if img_num == 0:
            log.warning("No images found.")
            return
        for k in np.random.randint(0, img_num, size=min(2, img_num)):
            out = p.get_item(int(k), debug=False)
            self.assertTrue(len(out["left_img"]) > 0)
            p.show_subset([out["left_img"], out["right_img"],  out["depth_rs_img"]],
                          ['left (RS)', 'right (RS)', 'depth RS (mm)'])

        plt.show()

    def test_draw_scene(self):
        source = DataSource()
        count = source.init_directory()
        self.assertTrue(count > 0)
        item = source.get_item(0, debug=False)
        source.draw_scene(item, show_depth_cloud=False)

    def test_get_item_projected(self):
        source = DataSource()
        scene_id, item_id = 4, 15
        count = source.init_directory(scene_id)
        self.assertTrue(count > 0)
        out = source.get_item_projected(item_id, debug=True)
        self.assertIn("depth_cad_projected", out)
        self.assertEqual(out["depth_cad_projected"].shape, out["depth_rs_img"].shape)

    def test_show_icp_alignment(self):
        source = DataSource()
        scene_id, item_id = 3, 95
        count = source.init_directory(scene_id)
        self.assertTrue(count > 0)
        item = source.get_item_with_icp(item_id, debug=True)

    def test_get_item_icp_projected(self):
        source = DataSource()
        scene_id, item_id = 4, 12 # 42, 55-ok
        count = source.init_directory(scene_id)
        self.assertTrue(count > 0)
        out = source.get_item_icp_projected(item_id, debug=True)
        self.assertIn("depth_cad_projected", out)
        self.assertEqual(out["depth_cad_projected"].shape, out["depth_rs_img"].shape)        

    def test_get_grid_coordinates(self):
        source = DataSource()
        scene_id, item_id = 4, 3
        count = source.init_directory(scene_id)
        self.assertTrue(count > 0)
        item = source.get_item(item_id, debug=False)
        coordinates = source.get_grid_coordinates(item["left_img"])
        self.assertTrue(len(coordinates) > 0)

    def test_match_grid_to_cad(self):
        source = DataSource()
        scene_id, item_id = 4, 2
        count                   = source.init_directory(scene_id)
        self.assertTrue(count > 0)
        item                    = source.get_item(item_id, debug=False)
        rvec, tvec              = source.match_grid_to_cad(item["left_img"])
        # transfrom rvec tvec to T_camera_cad homogenious transformation matrix
        R, _                    = cv2.Rodrigues(rvec)
        T_camera_cad = np.eye(4, dtype=np.float64)
        T_camera_cad[:3, :3]    = R
        T_camera_cad[:3, 3]     = tvec/1000.0 # convert mm to m

        # use ICP results
        T_camera_cad_icp        = source.load_csv_with_icp_results(item_id)
        T_camera_cad_icp        = T_camera_cad_icp #@ T_flip_z_direction

        print("Estimated T_camera_cad from grid matching:"  )
        print(T_camera_cad)
        print("ICP refined T_camera_cad:"  )
        print(T_camera_cad_icp)


        plt.show()
        self.assertEqual(rvec.shape, (3,))
        self.assertEqual(tvec.shape, (3,))

def RunTest():
    tst = TestDataSource()
    #tst.test_init_directory()
    #tst.test_get_item()
    #tst.test_show_images()
    #tst.test_draw_scene() # ok
    #tst.test_get_item_projected()
    #tst.test_show_icp_alignment()
    #tst.test_get_item_icp_projected()
    #tst.test_get_grid_coordinates()
    tst.test_match_grid_to_cad()

if __name__ == '__main__':
    RunTest()
