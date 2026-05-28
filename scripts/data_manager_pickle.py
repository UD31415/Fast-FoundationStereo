"""
Dataset management for Pickle scene capture dataset.

Refactored to class style (similar to data_manager_inbolt.py):
- `DataSource.init_directory(...)` loads scene JSON and preloads all captures.
- `DataSource.get_item(index)` returns one sample.
- Includes unittest test functions and a small RunTest helper.
"""

from __future__ import annotations

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


log.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log.INFO)

# Helper functions for loading and processing data items.

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

    xyz_flat = np.frombuffer(raw[:pixel_data_bytes], dtype=np.float32).copy()
    xyz = xyz_flat.reshape(height * width, 3)

    valid = xyz[:, 2] != 0.0
    xyz = xyz[valid] * 0.001  # mm -> m

    # Match previous convention used by this script.
    xyz[:, 1] *= -1
    xyz[:, 2] *= -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
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

        log.info("DataSource is defined")

    def __len__(self) -> int:
        return len(self.captures)

    def init_directory(self, scene_json_path: str = "") -> int:
        """Load scene metadata and optionally preload all captures into memory."""

        if len(scene_json_path) < 3:
            json_path = Path(__file__).with_name("scene.json")
        else:
            json_path = Path(scene_json_path)

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

        cad_pcd = cad_mesh.sample_points_poisson_disk(
            number_of_points=10000,
            use_triangle_normal=True,
        )
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
            fbpp                = 24            
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

    def load_item_data(self, index: int) -> dict[str, Any]:
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
        t_camera_cad        = compose_t_camera_cad(t_camera_tooltip, t_tooltip_cad)
        depth_pcd_raw       = load_vertices_to_pcd(Path(vertices_path))
        cad_pcd_aligned     = copy.deepcopy(self.cad_pcd)
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
        }

    def get_item(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Return one loaded capture item by index."""
        if index < 0 or index >= len(self.captures):
            raise IndexError(f"Capture index out of range: {index}")

        item = self.load_item_data(index)
        if item is None:
            raise RuntimeError(f"Failed to build item at index {index}")

        if debug:
            cad_pcd = copy.deepcopy(item["cad_pcd"])
            cad_pcd_aligned = copy.deepcopy(item["cad_pcd_aligned"])
            depth_pcd_raw = copy.deepcopy(item["depth_pcd_raw"])

            cad_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            cad_pcd_aligned.paint_uniform_color([0.0, 0.0, 1.0])
            depth_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0])

            o3d.visualization.draw([cad_pcd, cad_pcd_aligned, depth_pcd_raw])

        return item

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

        out = source.get_item(0, debug=True)
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




def RunTest():
    tst = TestDataSource()
    #tst.test_init_directory()
    #tst.test_get_item()
    #tst.test_show_images()
    tst.test_draw_scene()


if __name__ == '__main__':
    RunTest()
