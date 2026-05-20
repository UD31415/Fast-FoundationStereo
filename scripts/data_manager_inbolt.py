'''

Dataset management for Inbolt stereo dataset.

Loads RealSense IR stereo pairs (mono0/mono1) as left/right images
and Zivid depthmap as ground-truth depth.

Expected directory layout (one or more session folders under root):
    <root>/
      <session>/
        realsense/
          <idx>/
            mono0.png          # left IR image  (uint8 or uint16)
            mono1.png          # right IR image (uint8 or uint16)
        zivid/
          <idx>/
            depthmap_mm.png    # GT depth in mm (uint16)
            color.png          # optional RGB

Only samples that have BOTH a realsense pair AND a matching zivid depth are
included. Sessions that lack a zivid subfolder (e.g. freedrive-only captures)
are silently skipped.

Output dict keys (same as faro_data_manager for compatibility):
    left        : numpy array  (H, W)   uint8/uint16 IR
    right       : numpy array  (H, W)   uint8/uint16 IR
    depth_faro  : numpy array  (H, W)   float32, mm  ← Zivid GT
    depth_rs    : numpy array  (H, W)   float32, mm  (empty / zeros if absent)
    rgb         : numpy array  (H, W, 3) uint8        (Zivid color, or empty)

'''

from typing import Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import unittest
import logging as log
import yaml

# format logger
log.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log.INFO)

# --------------------------------
def dataset_2_camera_info():
    CAMERA_MATRIX_RS = np.array([
        [385.5098876953125, 0, 328.31732177734375],
        [0, 385.5098876953125, 235.6382141113281],
        [0, 0, 1]
    ])

    DIST_COEFFS_RS = np.array([
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0
    ])

    CAMERA_MATRIX_ZIVID = np.array([
        [1241.853637, 0, 609.9444419],
        [0, 1241.853637, 513.6974808515621],
        [0, 0, 1]
    ])
    DIST_COEFFS_ZIVID = np.array([
        - 0.04514386132359505,
        - -0.03609563037753105,
        - -6.156915333122015e-05,
        - 0.00015102965699043125,
        - -0.17297066748142242
    ])
    return CAMERA_MATRIX_RS, DIST_COEFFS_RS, CAMERA_MATRIX_ZIVID, DIST_COEFFS_ZIVID

def dataset_3_camera_info():
    CAMERA_MATRIX_RS = np.array([
        [642.4910888671875, 0, 647.5538330078125],
        [0, 642.4910888671875, 358.3725891113281],
        [0, 0, 1]
    ])

    DIST_COEFFS_RS = np.array([0.0, -0.0, -0.0,  0.0, -0.0])

    CAMERA_MATRIX_ZIVID = np.array([
        [1241.8536376953125, 0, 609.9756664537141],
        [0, 1241.6390380859375, 513.9514436913539],
        [0, 0, 1]
    ])
    DIST_COEFFS_ZIVID = np.array([
     0.04514386132359505,
    -0.03609563037753105,
    -6.156915333122015e-05,
    -0.00015102965699043125,
    -0.17297066748142242
    ])
    return CAMERA_MATRIX_RS, DIST_COEFFS_RS, CAMERA_MATRIX_ZIVID, DIST_COEFFS_ZIVID


CAMERA_MATRIX_RS, DIST_COEFFS_RS, CAMERA_MATRIX_ZIVID, DIST_COEFFS_ZIVID = dataset_3_camera_info()

# --------------------------------
#%% Data source
class DataSource:

    def __init__(self, train_mode = True):
        self.gray_scale_input = False
        self.imgs = []   # list of dicts: {left, right, depth_zivid, rgb}
        self.train_mode = train_mode
        self.save_point_cloud = False
        log.info('Source is defined')

    def init_directory(self, input_rectified='', gray_scale_input=False, sub_indexes=None):
        """Scan root for (realsense, zivid) sample pairs and populate self.imgs."""
        if len(input_rectified) < 3:
            input_rectified = (
                r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/'
                r'Inbolt_datasets/Data Collection-20260322T091926Z-1-001/Data Collection'
                r'C:\Work\Data\Depth\Data Collection-03'
            )

        self.gray_scale_input = gray_scale_input
        self.imgs = []

        #IGNORED_SESSIONS = {'dataset_y16_freedrive', 'dataset_y8_freedrive'}
        #IGNORED_SESSIONS = {'dataset_y16_freedrive','dataset_depth_bias'}
        #IGNORED_SESSIONS = {'20260414_142239'}  # set 2 include all sessions by default; manually exclude any bad ones here
        IGNORED_SESSIONS = {'20260513_074626'}  # set 3
        
        # Each immediate sub-directory is a session
        try:
            if self.train_mode:
                sessions = sorted([
                    os.path.join(input_rectified, d)
                    for d in os.listdir(input_rectified)
                    if os.path.isdir(os.path.join(input_rectified, d))
                    and d not in IGNORED_SESSIONS
                ])
            else:
                sessions = sorted([
                    os.path.join(input_rectified, d)
                    for d in os.listdir(input_rectified)
                    if os.path.isdir(os.path.join(input_rectified, d))
                    and d in IGNORED_SESSIONS
                ])                

        except FileNotFoundError:
            log.error(f"Directory not found: {input_rectified}")
            return 0

        for session in sessions:
            rs_root    = os.path.join(session, 'realsense')
            zivid_root = os.path.join(session, 'zivid')

            if not os.path.isdir(rs_root) or not os.path.isdir(zivid_root):
                continue  # session has no stereo+GT pair

            # Find all left images; match by index folder name
            left_paths = sorted(glob.glob(os.path.join(rs_root, '*', 'mono0.png')))
            for left_path in left_paths:
                idx             = os.path.basename(os.path.dirname(left_path))
                right_path      = os.path.join(rs_root, idx, 'mono1.png')
                depth_rs_path    = os.path.join(rs_root, idx, 'depthmap_mm.png')
                depth_zivid_path = os.path.join(zivid_root, idx, 'depthmap_mm.png')
                rgb_path        = os.path.join(zivid_root, idx, 'color.png')

                if not os.path.isfile(depth_rs_path) or not os.path.isfile(depth_zivid_path):
                    continue  # skip incomplete samples

                rs_metadata_path = os.path.join(rs_root, idx, 'metadata.yaml')
                zv_metadata_path = os.path.join(zivid_root, idx, 'metadata.yaml')

                self.imgs.append({
                    'left':  left_path,
                    'right': right_path,
                    'depth_rs': depth_rs_path,
                    'depth_zivid': depth_zivid_path,
                    'rgb':   rgb_path if os.path.isfile(rgb_path) else None,
                    'metadata_rs': rs_metadata_path if os.path.isfile(rs_metadata_path) else None,
                    'metadata_zv': zv_metadata_path if os.path.isfile(zv_metadata_path) else None,
                })

        if sub_indexes is not None:
            self.imgs = [self.imgs[i] for i in sub_indexes]

        log.info(f"DataSource: found {len(self.imgs)} samples in {input_rectified}")
        return len(self.imgs)

    def get_item(self, index: int, debug: bool = False):
        """Return one sample as a dict with left, right, depth_faro, depth_rs, rgb."""
        output_str = {"left": [], "right": [], "depth_zivid": [], "depth_rs": [], "rgb": [], "metadata_rs": None, "metadata_zv": None}

        entry = self.imgs[index]

        left_img  = cv2.imread(entry['left'],  cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(entry['right'], cv2.IMREAD_UNCHANGED)
        depth_rs_img = cv2.imread(entry['depth_rs'], cv2.IMREAD_UNCHANGED)
        depth_zivid_img = cv2.imread(entry['depth_zivid'], cv2.IMREAD_UNCHANGED)

        if left_img is None or right_img is None or depth_rs_img is None or depth_zivid_img is None:
            log.warning(f"Failed to load sample {index}: {entry}")
            return output_str

        rgb_img = np.array([], dtype=np.uint8)
        if entry['rgb'] is not None:
            rgb_img = cv2.imread(entry['rgb'], cv2.IMREAD_COLOR)
            if rgb_img is None:
                rgb_img = np.array([], dtype=np.uint8)

        depth_rs = depth_rs_img.astype(np.float32)
        depth_zivid = depth_zivid_img.astype(np.float32)   # uint16 mm → float32 mm

        metadata_rs = None
        if entry.get('metadata_rs') is not None:
            with open(entry['metadata_rs'], 'r') as f:
                metadata_rs = yaml.safe_load(f)

        metadata_zv = None
        if entry.get('metadata_zv') is not None:
            with open(entry['metadata_zv'], 'r') as f:
                metadata_zv = yaml.safe_load(f)

        output_str["left"]        = left_img
        output_str["right"]       = right_img
        output_str["depth_zivid"] = depth_zivid   # Zivid GT
        output_str["depth_rs"]    = depth_rs
        output_str["rgb"]         = rgb_img
        output_str["metadata_rs"] = metadata_rs
        output_str["metadata_zv"] = metadata_zv

        if debug:
            img_list = [left_img, right_img, depth_rs, depth_zivid]
            ttl_list = ['left (RS)', 'right (RS)', 'depth RS (mm)', 'depth Zivid (mm)']
            if rgb_img.size > 0:
                img_list.append(rgb_img)
                ttl_list.append('rgb (Zivid)')
            self.show_subset(img_list, ttl_list)

        return output_str
    

    def project_camera_to_3d(self, depth_img_mm: np.ndarray, cam_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        """Project 2D pixel coordinates with depth to 3D points in camera space."""
        h, w = depth_img_mm.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')

        # OpenCV expects Nx1x2 contiguous float32/float64 image points in (x, y) order.
        distorted_points = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        undistorted_points = cv2.undistortPoints(distorted_points,  cam_matrix.astype(np.float32),  dist_coeffs.astype(np.float32) )

        uv = undistorted_points.reshape(-1, 2)
        Z = depth_img_mm.reshape(-1).astype(np.float32)
        valid = np.isfinite(Z) & (Z > 0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        uv      = uv[valid]
        Z       = Z[valid]
        X       = uv[:, 0] * Z
        Y       = uv[:, 1] * Z

        # save to ply point cloud for visualization
        XYZ     = np.stack([X, Y, Z], axis=1).astype(np.float32)

        return XYZ

    def project_3d_to_camera(self, points_3d: np.ndarray, cam_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_size = (480,640)) -> np.ndarray:
        """Project 3D points in camera space back to 2D pixel coordinates."""
        if points_3d.shape[1] != 3:
            raise ValueError("Input points_3d must have shape (N, 3)")
        projected_pts, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            cam_matrix.astype(np.float32),
            dist_coeffs.astype(np.float32),
        )

        uv_rs = projected_pts.reshape(-1, 2)
        u_idx = np.rint(uv_rs[:, 0]).astype(np.int32)
        v_idx = np.rint(uv_rs[:, 1]).astype(np.int32)

        h_rs, w_rs = frame_size
        in_bounds = (u_idx >= 0) & (u_idx < w_rs) & (v_idx >= 0) & (v_idx < h_rs)
        if not np.any(in_bounds):
            return np.zeros((h_rs, w_rs), dtype=np.float32)

        u_idx = u_idx[in_bounds]
        v_idx = v_idx[in_bounds]
        z_vals = points_3d[in_bounds, 2]  # Z values of the valid points

        # Rasterize by nearest pixel; if multiple points hit a pixel, keep the closest depth.
        lin             = v_idx * w_rs + u_idx
        depth_buffer    = np.full(h_rs * w_rs, np.inf, dtype=np.float32)
        np.minimum.at(depth_buffer, lin, z_vals)
        depth_projected = depth_buffer.reshape(h_rs, w_rs)
        depth_projected[~np.isfinite(depth_projected)] = 0.0
        return depth_projected    
    
    def get_camera_to_base_transformation(self, metadata):
        """Extract camera extrinsics from metadata if available."""
        H = np.eye(4, dtype=np.float32)  # default to identity if no extrinsics found
        if metadata is None:
            return H

        # Example: extract rotation and translation from metadata
        # (actual keys depend on how the metadata is structured)
        try:
            #HBF = np.array(metadata['base_T_flange']).reshape(4, 4).astype(np.float32)  # assuming metadata stores transformation as a flat list in row-major order
            HFC = np.array(metadata['base_T_camera']).reshape(4, 4).astype(np.float32) 
            H   = HFC #HBF @ HFC
            # transform meters to mm
            #H[:,3] = H[:,3] * 1000.0
            # rotate the matrix
            #H[:3,:3] = H[:3,:3].T  # transpose rotation to convert from camera-to-base to base-to-camera
            #H = np.linalg.inv(H)  # invert to get camera-to-base if needed
            return H
        except KeyError:
            log.warning("Extrinsics not found in metadata")
            return H
        
    # project from zivid depth patrix to point cloud and back to depth matrix with rs intrinsics and distortion to get "zivid GT as seen by RealSense" for pixel-level comparison
    def project_depth_zivid_to_rs(self,depth_zivid_mm: np.ndarray, depth_rs_mm: np.ndarray, finx = 0) -> np.ndarray:
        "Projects cameras aligned by focal centers"
        # create 3D point cloud from zivid depth
        XYZ = self.project_camera_to_3d(depth_zivid_mm, CAMERA_MATRIX_ZIVID, DIST_COEFFS_ZIVID)  # (N, 3) array of 3D points in Zivid camera space
        # save to ply point cloud for visualization
        #self.save_to_ply(XYZ/1000, f'zivid_original_points_{finx:03d}.ply') # save in meters for visualization

        # project back on imaage RS
        depth_zivid_projected_mm = self.project_3d_to_camera(XYZ, CAMERA_MATRIX_RS, DIST_COEFFS_RS, frame_size = depth_rs_mm.shape)  # (H, W) depth map of Zivid points projected into RealSense pixel space

        XYZ_RS = self.project_camera_to_3d(depth_zivid_projected_mm, CAMERA_MATRIX_RS, DIST_COEFFS_RS)
        # save to ply point cloud for visualization
        #self.save_to_ply(XYZ_RS/1000, f'zivid_projected_points_{finx:03d}.ply') # save in meters for visualization
        return depth_zivid_projected_mm    

    # project from zivid depth patrix to point cloud and back to depth matrix with rs intrinsics and distortion to get "zivid GT as seen by RealSense" for pixel-level comparison
    def transform_project_depth_zivid_to_rs(self,transform_zivid: np.ndarray, depth_zivid_mm: np.ndarray, transform_rs: np.ndarray, depth_rs_mm: np.ndarray, finx = 0) -> np.ndarray:
        "Using tranformations align point clouds first and then project"
        # create 3D point cloud from zivid depth
        XYZ                         = self.project_camera_to_3d(depth_zivid_mm, CAMERA_MATRIX_ZIVID, DIST_COEFFS_ZIVID)  # (N, 3) array of 3D points in Zivid camera space
        XYZ1                        = np.hstack([XYZ, np.ones((XYZ.shape[0], 1), dtype=np.float32)])  # (N, 4) homogeneous coordinates
        XYZ1B_ZIVID                 = XYZ1 @ transform_zivid.T # @ (transform_rs @ np.linalg.inv(transform_zivid) @ XYZ1)
        # save to ply point cloud for visualization
        self.save_to_ply(XYZ1B_ZIVID[:,:3]/1000, f'zivid_original_points_{finx:03d}.ply') # save in meters for visualization

        # create 3D point cloud from real sense depth
        XYZ                         = self.project_camera_to_3d(depth_rs_mm, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in Zivid camera space
        XYZ1                        = np.hstack([XYZ, np.ones((XYZ.shape[0], 1), dtype=np.float32)])  # (N, 4) homogeneous coordinates
        XYZ1B_RS                    = XYZ1 @ transform_rs.T # @ (transform_rs @ np.linalg.inv(transform_zivid) @ XYZ1)
        # save to ply point cloud for visualization
        self.save_to_ply(XYZ1B_RS[:,:3]/1000, f'rs_original_points_{finx:03d}.ply') # save in meters for visualization

        # project back on imaage RS
        XYZ1_RS_projected           = XYZ1B_ZIVID @ np.linalg.inv(transform_rs).T # transform zivid points to rs frame before projecting
        depth_zivid_projected_mm    = self.project_3d_to_camera(XYZ1_RS_projected[:,:3], CAMERA_MATRIX_RS, DIST_COEFFS_RS, frame_size = depth_rs_mm.shape)  # (H, W) depth map of Zivid points projected into RealSense pixel space
        XYZ_RS                      = self.project_camera_to_3d(depth_zivid_projected_mm, CAMERA_MATRIX_RS, DIST_COEFFS_RS)
        
        # return back to base frame for comparison
        XYZ1                        = np.hstack([XYZ_RS, np.ones((XYZ_RS.shape[0], 1), dtype=np.float32)])  # (N, 4) homogeneous coordinates
        XYZ1B_RS                    = XYZ1 @ transform_rs.T # @ (transform_rs @ np.linalg.inv(transform_zivid) @ XYZ1)
        
        # save to ply point cloud for visualization
        self.save_to_ply(XYZ1B_RS[:,:3]/1000, f'zivid_projected_points_{finx:03d}.ply') # save in meters for visualization
        return depth_zivid_projected_mm    

    def reproject_depth(self,
        zivid_depth_mm: np.ndarray,
        zivid_K: np.ndarray,  # 3x3 Zivid intrinsics
        base_T_zivid: np.ndarray,  # 4x4 Zivid camera pose in base frame
        rs_K: np.ndarray,  # 3x3 RealSense intrinsics
        base_T_rs: np.ndarray,  # 4x4 RS camera pose in base frame
        rs_shape: Tuple[int, int],  # (H, W) of output RS image
        zivid_dist: Optional[np.ndarray] = None,  # Zivid distortion coeffs
        rs_dist: Optional[np.ndarray] = None,  # RS distortion coeffs
    ) -> np.ndarray:
        """Reproject Zivid depth map into the RealSense camera frame.
    
        Uses OpenCV undistortPoints / projectPoints for full distortion handling
        when distortion coefficients are provided.
    
        Returns uint16 depth in mm at RS resolution, 0 = invalid.
        """
        H_rs, W_rs = rs_shape
    
        valid = zivid_depth_mm > 0
        ys, xs = np.where(valid)
        Z = zivid_depth_mm[valid].astype(np.float64) / 1000.0  # → metres
    
        if len(Z) == 0:
            return np.zeros((H_rs, W_rs), dtype=np.uint16)
    
        # --- Unproject Zivid pixels → 3D (with distortion correction) ---
        zv_dist = zivid_dist if zivid_dist is not None else np.zeros(5)
        pts_2d = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=-1).reshape(-1, 1, 2)
        # undistortPoints returns normalized camera coords (x/z, y/z)
        norm = cv2.undistortPoints(pts_2d, zivid_K, zv_dist).reshape(-1, 2)
        X = norm[:, 0] * Z
        Y = norm[:, 1] * Z
    
        # --- Zivid frame → base frame → RS frame ---
        pts_zivid = np.vstack([X, Y, Z, np.ones_like(Z)])  # 4 x N
        pts_base = base_T_zivid @ pts_zivid
        rs_T_base = np.linalg.inv(base_T_rs)
        pts_rs = rs_T_base @ pts_base
    
        Xr, Yr, Zr = pts_rs[0], pts_rs[1], pts_rs[2]
        in_front = Zr > 0.01
    
        # --- Project onto RS image plane (with distortion) ---
        rs_d = rs_dist if rs_dist is not None else np.zeros(5)
        pts_3d = np.stack([Xr[in_front], Yr[in_front], Zr[in_front]], axis=-1).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(
            pts_3d,
            np.zeros(3),
            np.zeros(3),
            rs_K.astype(np.float64),
            rs_d,
        )
        uv = proj.reshape(-1, 2)
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        z_vals = Zr[in_front]
    
        # Filter to in-bounds
        in_bounds = (u >= 0) & (u < W_rs) & (v >= 0) & (v < H_rs)
        u, v, z_vals = u[in_bounds], v[in_bounds], z_vals[in_bounds]
    
        # Vectorized z-buffer: for duplicate pixels, keep the closest depth
        depth_mm_vals = z_vals * 1000.0
        flat_idx = v * W_rs + u
        out_flat = np.full(H_rs * W_rs, np.inf)
        np.minimum.at(out_flat, flat_idx, depth_mm_vals)
        out_flat[out_flat == np.inf] = 0.0
    
        return out_flat.reshape(H_rs, W_rs).astype(np.uint16)
 
    def reproject_depth_with_icp(self,
        zivid_depth_mm: np.ndarray,
        zivid_K: np.ndarray,  # 3x3 Zivid intrinsics
        base_T_zivid: np.ndarray,  # 4x4 Zivid camera pose in base frame
        rs_depth_mm: np.ndarray,
        rs_K: np.ndarray,  # 3x3 RealSense intrinsics
        base_T_rs: np.ndarray,  # 4x4 RS camera pose in base frame
        zivid_dist: Optional[np.ndarray] = None,  # Zivid distortion coeffs
        rs_dist: Optional[np.ndarray] = None,  # RS distortion coeffs
    ) -> np.ndarray:
        """Reproject Zivid depth map into the RealSense camera frame.
    
        Uses OpenCV undistortPoints / projectPoints for full distortion handling
        when distortion coefficients are provided.
    
        Returns uint16 depth in mm at RS resolution, 0 = invalid.
        """
        H_rs, W_rs = rs_depth_mm.shape
    
        # --- Unproject Zivid pixels → 3D (with distortion correction) ---
        valid = zivid_depth_mm > 0
        ys, xs = np.where(valid)
        Z = zivid_depth_mm[valid].astype(np.float64) / 1000.0  # → metres
    
        if len(Z) == 0:
            return np.zeros((H_rs, W_rs), dtype=np.uint16)
    
        zv_dist = zivid_dist if zivid_dist is not None else np.zeros(5)
        pts_2d = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=-1).reshape(-1, 1, 2)
        # undistortPoints returns normalized camera coords (x/z, y/z)
        norm = cv2.undistortPoints(pts_2d, zivid_K, zv_dist).reshape(-1, 2)
        X = norm[:, 0] * Z
        Y = norm[:, 1] * Z
        pts_zivid = np.vstack([X, Y, Z, np.ones_like(Z)])  # 4 x N
        pts_zivid_base = base_T_zivid @ pts_zivid

        # --- Unproject RS pixels → 3D (with distortion correction) ---
        valid = rs_depth_mm > 0
        ys, xs = np.where(valid)
        Z = rs_depth_mm[valid].astype(np.float64) / 1000.0  # → metres
    
        if len(Z) == 0:
            return np.zeros((H_rs, W_rs), dtype=np.uint16)
    
        rs_dist = rs_dist if rs_dist is not None else np.zeros(5)
        pts_2d = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=-1).reshape(-1, 1, 2)
        # undistortPoints returns normalized camera coords (x/z, y/z)
        norm = cv2.undistortPoints(pts_2d, rs_K, rs_dist).reshape(-1, 2)
        X = norm[:, 0] * Z
        Y = norm[:, 1] * Z        
        pts_rs = np.vstack([X, Y, Z, np.ones_like(Z)])  # 4 x N
        pts_rs_base = base_T_rs @ pts_rs


        # ---- ICP on the data
        # improving matching
        T_zivid_rs, info = self.point_cloud_matching(pts_zivid_base[:3,:], pts_rs_base[:3,:], voxel_size_m=0.005, max_correspondence_distance_m=0.01,   debug=True)        

        # --- Zivid frame → base frame → RS frame ---
        rs_T_base = np.linalg.inv(base_T_rs)
        pts_rs = rs_T_base @ T_zivid_rs @ pts_zivid_base  # transform zivid points to rs frame using icp result before projecting
    
        
        Xr, Yr, Zr = pts_rs[0], pts_rs[1], pts_rs[2]
        in_front = Zr > 0.01
    
        # --- Project onto RS image plane (with distortion) ---
        rs_d = rs_dist if rs_dist is not None else np.zeros(5)
        pts_3d = np.stack([Xr[in_front], Yr[in_front], Zr[in_front]], axis=-1).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(pts_3d, np.zeros(3),  np.zeros(3),  rs_K.astype(np.float64),  rs_d)
        uv = proj.reshape(-1, 2)
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        z_vals = Zr[in_front]
    
        # Filter to in-bounds
        in_bounds = (u >= 0) & (u < W_rs) & (v >= 0) & (v < H_rs)
        u, v, z_vals = u[in_bounds], v[in_bounds], z_vals[in_bounds]
    
        # Vectorized z-buffer: for duplicate pixels, keep the closest depth
        depth_mm_vals = z_vals * 1000.0
        flat_idx = v * W_rs + u
        out_flat = np.full(H_rs * W_rs, np.inf)
        np.minimum.at(out_flat, flat_idx, depth_mm_vals)
        out_flat[out_flat == np.inf] = 0.0
    
        return out_flat.reshape(H_rs, W_rs).astype(np.uint16)
 

    def get_item_projected(self, index: int, debug: bool = False):
        """Return one sample as a dict with left, right, depth_faro, depth_rs, rgb."""
        output_str = {"left": [], "right": [], "depth_zivid": [], "depth_rs": [], "rgb": [], "metadata_rs": None, "metadata_zv": None}

        entry           = self.imgs[index]

        left_img        = cv2.imread(entry['left'],  cv2.IMREAD_UNCHANGED)
        right_img       = cv2.imread(entry['right'], cv2.IMREAD_UNCHANGED)
        depth_rs_img    = cv2.imread(entry['depth_rs'], cv2.IMREAD_UNCHANGED)
        depth_zivid_img = cv2.imread(entry['depth_zivid'], cv2.IMREAD_UNCHANGED)

        if left_img is None or right_img is None or depth_rs_img is None or depth_zivid_img is None:
            log.warning(f"Failed to load sample {index}: {entry}")
            return output_str

        rgb_img = np.array([], dtype=np.uint8)
        if entry['rgb'] is not None:
            rgb_img = cv2.imread(entry['rgb'], cv2.IMREAD_COLOR)
            if rgb_img is None:
                rgb_img = np.array([], dtype=np.uint8)

        depth_rs    = depth_rs_img.astype(np.float32)
        depth_zivid = depth_zivid_img.astype(np.float32)   # uint16 mm → float32 mm

        zivid_projected_path = entry['depth_zivid'].replace('.png', '_projected.png')  # for debug visualization of projected depth maps
        if os.path.exists(zivid_projected_path):
            depth_zivid_projected = cv2.imread(zivid_projected_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        else:
            depth_zivid_projected  = self.project_depth_zivid_to_rs(depth_zivid, depth_rs, finx = index)
            cv2.imwrite(zivid_projected_path, depth_zivid_projected.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])  # save projected depth for visualization  

        metadata_rs = None
        if entry.get('metadata_rs') is not None:
            with open(entry['metadata_rs'], 'r') as f:
                metadata_rs = yaml.safe_load(f)

        metadata_zv = None
        if entry.get('metadata_zv') is not None:
            with open(entry['metadata_zv'], 'r') as f:
                metadata_zv = yaml.safe_load(f)

        output_str["left"]        = left_img
        output_str["right"]       = right_img
        output_str["depth_zivid"] = depth_zivid_projected   # Zivid GT
        output_str["depth_rs"]    = depth_rs
        output_str["rgb"]         = rgb_img
        output_str["metadata_rs"] = metadata_rs
        output_str["metadata_zv"] = metadata_zv

        if debug:
            img_list = [left_img, right_img, depth_rs, depth_zivid_projected]
            ttl_list = ['left (RS)', 'right (RS)', 'depth RS (mm)', 'depth Zivid (mm)']
            # if rgb_img.size > 0:
            #     img_list.append(rgb_img)
            #     ttl_list.append('rgb (Zivid)')
            self.show_subset(img_list, ttl_list)

            # create point cloud  & save to ply point cloud for visualization
            #XYZ = self.project_camera_to_3d(depth_zivid_projected, CAMERA_MATRIX_ZIVID, DIST_COEFFS_ZIVID)
            XYZ = self.project_camera_to_3d(depth_zivid_projected, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in Zivid camera space
            zivid_path = entry['depth_zivid'].replace('.png', f'.ply')
            #self.save_to_ply(XYZ/1000, zivid_path) # save in meters for visualization

            XYZ = self.project_camera_to_3d(depth_rs, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in RS camera space
            rs_path = entry['depth_rs'].replace('.png', f'.ply')
            #self.save_to_ply(XYZ/1000, rs_path) 

        return output_str    

    def get_item_transformed_and_projected(self, index: int, debug: bool = False):
        """Return one sample as a dict with left, right, depth_faro, depth_rs, rgb."""
        output_str = {"left": [], "right": [], "depth_zivid": [], "depth_rs": [], "rgb": [], "depth_zivid_original": [],"metadata_rs": None, "metadata_zv": None}

        entry           = self.imgs[index]

        left_img        = cv2.imread(entry['left'],  cv2.IMREAD_UNCHANGED)
        right_img       = cv2.imread(entry['right'], cv2.IMREAD_UNCHANGED)
        depth_rs_img    = cv2.imread(entry['depth_rs'], cv2.IMREAD_UNCHANGED)
        depth_zivid_img = cv2.imread(entry['depth_zivid'], cv2.IMREAD_UNCHANGED)

        if left_img is None or right_img is None or depth_rs_img is None or depth_zivid_img is None:
            log.warning(f"Failed to load sample {index}: {entry}")
            return output_str

        rgb_img = np.array([], dtype=np.uint8)
        if entry['rgb'] is not None:
            rgb_img = cv2.imread(entry['rgb'], cv2.IMREAD_COLOR)
            if rgb_img is None:
                rgb_img = np.array([], dtype=np.uint8)

        depth_rs    = depth_rs_img.astype(np.float32)
        depth_zivid = depth_zivid_img.astype(np.float32)   # uint16 mm → float32 mm
        metadata_rs = None
        if entry.get('metadata_rs') is not None:
            with open(entry['metadata_rs'], 'r') as f:
                metadata_rs = yaml.safe_load(f)

            CAMERA_MATRIX_RS = np.array(metadata_rs["intrinsics"]["depthmap_mm"]["camera_matrix"], dtype=np.float32)  
            DIST_COEFFS_RS   = np.array(metadata_rs["intrinsics"]["depthmap_mm"]["dist_coeffs"], dtype=np.float32)  

        metadata_zv = None
        if entry.get('metadata_zv') is not None:
            with open(entry['metadata_zv'], 'r') as f:
                metadata_zv = yaml.safe_load(f)   
            
            CAMERA_MATRIX_ZIVID = np.array(metadata_zv["intrinsics"]["depthmap_mm"]["camera_matrix"], dtype=np.float32)  
            DIST_COEFFS_ZIVID   = np.array(metadata_zv["intrinsics"]["depthmap_mm"]["dist_coeffs"], dtype=np.float32)   


        zivid_projected_path = entry['depth_zivid'].replace('.png', '_projected.png')  # for debug visualization of projected depth maps
        if os.path.exists(zivid_projected_path):
            depth_zivid_projected = cv2.imread(zivid_projected_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        else:

            # depth_zivid_projected = self.reproject_depth(
            #                             zivid_depth_mm = depth_zivid,
            #                             zivid_K = CAMERA_MATRIX_ZIVID,  # 3x3 Zivid intrinsics
            #                             base_T_zivid = self.get_camera_to_base_transformation(metadata_zv),  # 4x4 Zivid camera pose in base frame
            #                             rs_K = CAMERA_MATRIX_RS,  # 3x3 RealSense intrinsics
            #                             base_T_rs = self.get_camera_to_base_transformation(metadata_rs),  # 4x4 RS camera pose in base frame
            #                             rs_shape = depth_rs.shape,  # (H, W) of output RS image
            #                             zivid_dist = DIST_COEFFS_ZIVID,  # Zivid distortion coeffs
            #                             rs_dist = DIST_COEFFS_RS,  # RS distortion coeffs
            #                         )
            depth_zivid_projected = self.reproject_depth_with_icp(
                                        zivid_depth_mm = depth_zivid,
                                        zivid_K = CAMERA_MATRIX_ZIVID,  # 3x3 Zivid intrinsics
                                        base_T_zivid = self.get_camera_to_base_transformation(metadata_zv),  # 4x4 Zivid camera pose in base frame
                                        rs_depth_mm = depth_rs,
                                        rs_K = CAMERA_MATRIX_RS,  # 3x3 RealSense intrinsics
                                        base_T_rs = self.get_camera_to_base_transformation(metadata_rs),  # 4x4 RS camera pose in base frame
                                        zivid_dist = DIST_COEFFS_ZIVID,  # Zivid distortion coeffs
                                        rs_dist = DIST_COEFFS_RS,  # RS distortion coeffs
                                    )                

            #cv2.imwrite(zivid_projected_path, depth_zivid_projected.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])  # save projected depth for visualization  

        # # improving matching
        # pcd_zivid = self.depth_to_pointcloud_base(
        #     depth_mm=depth_zivid,
        #     cam_matrix=CAMERA_MATRIX_ZIVID,
        #     base_T_camera=self.get_camera_to_base_transformation(metadata_zv),
        #     rgb=None
        # )
        # pcd_rs = self.depth_to_pointcloud_base(
        #     depth_mm=depth_rs,
        #     cam_matrix=CAMERA_MATRIX_RS,
        #     base_T_camera=self.get_camera_to_base_transformation(metadata_rs),
        #     rgb=None
        # )
        # T_zivid_rs, info = self.point_cloud_matching(pcd_zivid, pcd_rs, voxel_size_m=0.005, max_correspondence_distance_m=0.02,   debug=True)
        
        # print("Estimated transform from Zivid to RS:\n", T_zivid_rs)


        output_str["left"]        = left_img
        output_str["right"]       = right_img
        output_str["depth_zivid"] = depth_zivid_projected   # Zivid GT
        output_str["depth_zivid_original"] = depth_zivid   # Original Zivid depth
        output_str["depth_rs"]    = depth_rs
        output_str["rgb"]         = rgb_img
        output_str["metadata_rs"] = metadata_rs
        output_str["metadata_zv"] = metadata_zv

        if debug:
            img_list = [left_img, right_img, depth_rs, depth_zivid_projected]
            ttl_list = ['left (RS)', 'right (RS)', 'depth RS (mm)', 'depth Zivid (mm)']
            # if rgb_img.size > 0:
            #     img_list.append(rgb_img)
            #     ttl_list.append('rgb (Zivid)')
            self.show_subset(img_list, ttl_list)

            # create point cloud  & save to ply point cloud for visualization
            #XYZ = self.project_camera_to_3d(depth_zivid_projected, CAMERA_MATRIX_ZIVID, DIST_COEFFS_ZIVID)
            XYZ = self.project_camera_to_3d(depth_zivid_projected, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in Zivid camera space
            zivid_path = entry['depth_zivid'].replace('.png', f'.ply')
            #self.save_to_ply(XYZ/1000, zivid_path) # save in meters for visualization

            XYZ = self.project_camera_to_3d(depth_rs, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in RS camera space
            rs_path = entry['depth_rs'].replace('.png', f'.ply')
            #self.save_to_ply(XYZ/1000, rs_path) 

        return output_str    

    def get_item_and_show_open3d(self, index: int, debug: bool = False):
        """Return one sample as a dict with left, right, depth_faro, depth_rs, rgb."""
        output_str = {"left": [], "right": [], "depth_zivid": [], "depth_rs": [], "rgb": [], "metadata_rs": None, "metadata_zv": None}

        entry           = self.imgs[index]

        left_img        = cv2.imread(entry['left'],  cv2.IMREAD_UNCHANGED)
        right_img       = cv2.imread(entry['right'], cv2.IMREAD_UNCHANGED)
        depth_rs_img    = cv2.imread(entry['depth_rs'], cv2.IMREAD_UNCHANGED)
        depth_zivid_img = cv2.imread(entry['depth_zivid'], cv2.IMREAD_UNCHANGED)

        if left_img is None or right_img is None or depth_rs_img is None or depth_zivid_img is None:
            log.warning(f"Failed to load sample {index}: {entry}")
            return output_str

        rgb_img = None #np.array([], dtype=np.uint8)
        if entry['rgb'] is not None:
            rgb_img = cv2.imread(entry['rgb'], cv2.IMREAD_COLOR)
            if rgb_img is None:
                rgb_img = None

        depth_rs    = depth_rs_img.astype(np.float32)
        depth_zivid = depth_zivid_img.astype(np.float32)   # uint16 mm → float32 mm
        metadata_rs = None
        if entry.get('metadata_rs') is not None:
            with open(entry['metadata_rs'], 'r') as f:
                metadata_rs = yaml.safe_load(f)

        metadata_zv = None
        if entry.get('metadata_zv') is not None:
            with open(entry['metadata_zv'], 'r') as f:
                metadata_zv = yaml.safe_load(f)        


        # zivid_projected_path = entry['depth_zivid'].replace('.png', '_projected.png')  # for debug visualization of projected depth maps
        # if os.path.exists(zivid_projected_path):
        #     depth_zivid_projected = cv2.imread(zivid_projected_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # else:
        #     tranform_zivid = self.get_camera_to_base_transformation(metadata_zv)
        #     tranform_rs    = self.get_camera_to_base_transformation(metadata_rs)
        #     depth_zivid_projected  = self.transform_project_depth_zivid_to_rs(tranform_zivid, depth_zivid, tranform_rs, depth_rs, finx = index)
        #     #cv2.imwrite(zivid_projected_path, depth_zivid_projected.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])  # save projected depth for visualization  

        H = np.array(metadata_zv['base_T_camera']).reshape(4, 4).astype(np.float32)
        #H = np.linalg.inv(H)  # invert to get camera-to-base if needed
        pcd_zivid = self.depth_to_pointcloud_base(
            depth_mm=depth_zivid,
            cam_matrix=CAMERA_MATRIX_ZIVID,
            base_T_camera=H,
            rgb=rgb_img
        )

        H = np.array(metadata_rs['base_T_camera']).reshape(4, 4).astype(np.float32)
        #H = np.linalg.inv(H)  # invert to get camera-to-base if needed
        pcd_rs = self.depth_to_pointcloud_base(
            depth_mm=depth_rs,
            cam_matrix=CAMERA_MATRIX_RS,
            base_T_camera=H,
            rgb=None
        )        

        # Show both clouds together in one Open3D window (zivid=red, rs=green).
        self.show_pointclouds_open3d(
            [pcd_zivid, pcd_rs],
            colors=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            window_name=f"Sample {index}: Zivid (red) vs RealSense (blue)",
        )


        depth_zivid_projected     = depth_rs
        output_str["left"]        = left_img
        output_str["right"]       = right_img
        output_str["depth_zivid"] = depth_zivid_projected   # Zivid GT
        output_str["depth_rs"]    = depth_rs
        output_str["rgb"]         = rgb_img
        output_str["metadata_rs"] = metadata_rs
        output_str["metadata_zv"] = metadata_zv

        if debug:
            img_list = [left_img, right_img, depth_rs, depth_zivid_projected]
            ttl_list = ['left (RS)', 'right (RS)', 'depth RS (mm)', 'depth Zivid (mm)']
            # if rgb_img.size > 0:
            #     img_list.append(rgb_img)
            #     ttl_list.append('rgb (Zivid)')
            self.show_subset(img_list, ttl_list)

            # create point cloud  & save to ply point cloud for visualization
            #XYZ = self.project_camera_to_3d(depth_zivid_projected, CAMERA_MATRIX_ZIVID, DIST_COEFFS_ZIVID)
            XYZ = self.project_camera_to_3d(depth_zivid_projected, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in Zivid camera space
            zivid_path = entry['depth_zivid'].replace('.png', f'.ply')
            #self.save_to_ply(XYZ/1000, zivid_path) # save in meters for visualization

            XYZ = self.project_camera_to_3d(depth_rs, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in RS camera space
            rs_path = entry['depth_rs'].replace('.png', f'.ply')
            #self.save_to_ply(XYZ/1000, rs_path) 

        return output_str    

    def compute_depth_error(self, depth_pred, depth_gt, depth_mask=None):
        """Compute absolute depth error between prediction and GT."""
        depth_pred = depth_pred.astype(np.float32)
        depth_gt   = depth_gt.astype(np.float32)
        depth_error = np.zeros_like(depth_pred)
        mask = np.ones_like(depth_pred, dtype=bool) if depth_mask is None else depth_mask
        valid = np.logical_and(depth_gt > 0, mask)
        valid = np.logical_and(depth_pred > 0, valid)
        depth_error[valid] = (depth_pred[valid] - depth_gt[valid])
        return depth_error

    def show_subset(self, img_list, ttl_list, vmin=None, vmax=None, save_path='', fig_name=''):
        """Display a list of images in a grid."""
        img_num = len(img_list)
        col_num = min(img_num, 3)
        row_num = (img_num + col_num - 1) // col_num
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
        axes = np.array(axes).reshape(row_num, col_num)
        for k in range(img_num):
            ri, ci = k // col_num, k % col_num
            axes[ri, ci].imshow(img_list[k], vmin=vmin, vmax=vmax)
            axes[ri, ci].set_title(ttl_list[k])
        for k in range(img_num, row_num * col_num):
            axes[k // col_num, k % col_num].axis('on')
        if save_path and os.path.exists(save_path):
            fig.savefig(os.path.join(save_path, fig_name + ".png"))
        plt.show(block=False)

    def save_data_to_folder(self, output_str, output_directory):
        """Save sample dict to PNG files on disk."""
        os.makedirs(output_directory, exist_ok=True)

        paths = {
            "img_left.png":        output_str["left"],
            "img_right.png":       output_str["right"],
            "img_depth_zivid.png": output_str["depth_zivid"].astype(np.uint16),
            "img_depth_rs.png":    output_str["depth_rs"].astype(np.uint16),
        }
        success = True
        for fname, img in paths.items():
            out = cv2.imwrite(os.path.join(output_directory, fname), img,
                              [cv2.IMWRITE_PNG_COMPRESSION, 0])
            success = success and out

        if output_str["rgb"] is not None and np.asarray(output_str["rgb"]).size > 0:
            cv2.imwrite(os.path.join(output_directory, "img_rgb.png"),
                        output_str["rgb"], [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return success
    
    def depth_to_pointcloud_base(self,
                                  depth_mm: np.ndarray,
                                  cam_matrix: np.ndarray,
                                  base_T_camera: np.ndarray,
                                  rgb: np.ndarray = None,
                                  depth_scale_to_m: float = 1000.0,
                                  depth_trunc_m: float = 10.0):
        """Build an Open3D point cloud from a depth map using camera intrinsics
        and transform it into the robot base frame using ``base_T_camera``.

        Parameters
        ----------
        depth_mm : (H, W) np.ndarray
            Depth image in millimetres (uint16 or float).
        cam_matrix : (3, 3) np.ndarray
            Pinhole camera intrinsics ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
        base_T_camera : (4, 4) np.ndarray
            Homogeneous transform from camera frame to robot base frame.
            Translation is assumed to be in metres.
        rgb : (H, W, 3) np.ndarray, optional
            Colour image aligned with ``depth_mm``. If provided, the resulting
            cloud carries per-point colours.
        depth_scale_to_m : float
            Factor that converts depth values to metres (1000 for mm input).
        depth_trunc_m : float
            Maximum valid depth (metres). Points beyond are discarded.

        Returns
        -------
        open3d.geometry.PointCloud
            Point cloud expressed in the robot base frame (metres).
        """
        import open3d as o3d

        if depth_mm.ndim != 2:
            raise ValueError("depth_mm must be a 2D array")
        if cam_matrix.shape != (3, 3):
            raise ValueError("cam_matrix must be 3x3")
        if base_T_camera.shape != (4, 4):
            raise ValueError("base_T_camera must be 4x4")

        h, w = depth_mm.shape
        fx, fy = float(cam_matrix[0, 0]), float(cam_matrix[1, 1])
        cx, cy = float(cam_matrix[0, 2]), float(cam_matrix[1, 2])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_mm.astype(np.float32)))

        if rgb is not None:
            if rgb.shape[:2] != depth_mm.shape:
                raise ValueError("rgb and depth_mm must have the same H, W")
            color = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
            color_o3d = o3d.geometry.Image(color)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=depth_scale_to_m,
                depth_trunc=depth_trunc_m,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d, intrinsic,
                depth_scale=depth_scale_to_m,
                depth_trunc=depth_trunc_m,
            )

        # Transform from camera frame to robot base frame.
        pcd.transform(np.asarray(base_T_camera, dtype=np.float64))
        log.info(f"Built point cloud with {len(pcd.points)} points in base frame")
        return pcd

    def point_cloud_matching(self,
                              pcd_zivid,
                              pcd_rs,
                              voxel_size_m: float = 0.005,
                              max_correspondence_distance_m: float = 0.02,
                              init_transform: Optional[np.ndarray] = None,
                              use_global_registration: bool = True,
                              use_point_to_plane: bool = True,
                              max_iter: int = 100,
                              debug: bool = False):
        """Estimate the 4x4 rigid transform that aligns the Zivid cloud onto
        the RealSense cloud using Open3D ICP (with optional FPFH-based global
        pre-alignment for robustness when no initial guess is provided).

        Both inputs may be ``open3d.geometry.PointCloud`` instances or ``(N, 3)``
        numpy arrays of points expressed in metres. The returned transform
        ``T_rs_zivid`` is such that::

            pcd_zivid_aligned = T_rs_zivid @ pcd_zivid

        Parameters
        ----------
        pcd_zivid : open3d.geometry.PointCloud | np.ndarray
            Source cloud to align (Zivid).
        pcd_rs : open3d.geometry.PointCloud | np.ndarray
            Target / reference cloud (RealSense).
        voxel_size_m : float
            Downsampling voxel size in metres. Also drives the FPFH feature
            radii and the global RANSAC distance threshold.
        max_correspondence_distance_m : float
            ICP inlier threshold in metres.
        init_transform : (4, 4) np.ndarray, optional
            Initial guess for the source→target transform. If provided,
            global registration is skipped.
        use_global_registration : bool
            When ``init_transform`` is None, run a fast FPFH/RANSAC step to
            produce an initial alignment before ICP.
        use_point_to_plane : bool
            If True, use point-to-plane ICP (estimates normals on the target).
            Otherwise use point-to-point ICP.
        max_iter : int
            Maximum ICP iterations.
        debug : bool
            If True, visualize the result and log fitness/RMSE.

        Returns
        -------
        T : (4, 4) np.ndarray
            Estimated transform aligning ``pcd_zivid`` onto ``pcd_rs``.
        info : dict
            ``{'fitness': float, 'inlier_rmse': float, 'num_points_src': int,
            'num_points_tgt': int, 'init_transform': np.ndarray}``.
        """
        import open3d as o3d

        def _as_pcd(p):
            if isinstance(p, o3d.geometry.PointCloud):
                return p
            arr = np.asarray(p, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 3:
                if arr.shape[0] == 3:
                    arr = arr.T
                else:
                    raise ValueError("point cloud array must have shape (N, 3)")
            out = o3d.geometry.PointCloud()
            out.points = o3d.utility.Vector3dVector(arr)
            return out

        target = _as_pcd(pcd_rs)
        source = _as_pcd(pcd_zivid)

        if len(target.points) == 0 or len(source.points) == 0:
            log.warning("point_cloud_matching: empty input cloud(s); returning identity")
            return np.eye(4, dtype=np.float64), {
                'fitness': 0.0, 'inlier_rmse': float('inf'),
                'num_points_src': len(source.points),
                'num_points_tgt': len(target.points),
                'init_transform': np.eye(4, dtype=np.float64),
            }
        # pcd_zivid defines the working voume and is typically smaller, 
        # cut all the points in pcd_rs that far more than 0.2m from any nearest point in pcd_zivid to speed up matching and avoid spurious matches on faraway points
        dists = target.compute_point_cloud_distance(source)
        mask = np.array(dists) < 0.2
        target = target.select_by_index(np.where(mask)[0])

        # # --- Downsample + estimate normals ------------------------------------
        # src_down = source.voxel_down_sample(voxel_size_m)
        # tgt_down = target.voxel_down_sample(voxel_size_m)

        # normal_radius = voxel_size_m * 2.0
        # src_down.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        # tgt_down.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

        # # --- Initial transform: user-supplied or FPFH/RANSAC global reg --------
        # if init_transform is not None:
        #     init = np.asarray(init_transform, dtype=np.float64).reshape(4, 4)
        # elif use_global_registration:
        #     feature_radius = voxel_size_m * 5.0
        #     src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        #         src_down,
        #         o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100))
        #     tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        #         tgt_down,
        #         o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100))

        #     distance_threshold = voxel_size_m * 1.5
        #     ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        #         src_down, tgt_down, src_fpfh, tgt_fpfh,
        #         mutual_filter=True,
        #         max_correspondence_distance=distance_threshold,
        #         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        #         ransac_n=4,
        #         checkers=[
        #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        #         ],
        #         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        #     )
        #     init = ransac.transformation
        #     log.info(f"point_cloud_matching: global RANSAC fitness={ransac.fitness:.3f} "
        #              f"rmse={ransac.inlier_rmse:.4f}")
        # else:
        #     init = np.eye(4, dtype=np.float64)

        # --- ICP refinement ---------------------------------------------------
        if use_point_to_plane:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        # 2. CRITICAL STEP: Estimate normals for the target point cloud
        # (We search within a 10cm radius, constraining max NN to 30 for speed)
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
        )

        # 3. Define the maximum distance threshold for matching points
        threshold = 0.02  # 2 centimeters

        # 4. Set an initial rough alignment guess (Identity matrix means they are already close)
        trans_init = np.identity(4)

        # 5. Instantiate the Point-to-Plane estimation object
        #estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        # 6. Run the ICP registration
        icp = o3d.pipelines.registration.registration_icp(
            source, 
            target, 
            threshold, 
            trans_init,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )

        print("Transformation Matrix:")
        print(icp.transformation)            


        T = np.asarray(icp.transformation, dtype=np.float64)
        info = {
            'fitness': float(icp.fitness),
            'inlier_rmse': float(icp.inlier_rmse),
            'num_points_src': len(source.points),
            'num_points_tgt': len(target.points),
            'init_transform': trans_init,
        }
        log.info(f"point_cloud_matching: ICP fitness={info['fitness']:.3f} "
                 f"rmse={info['inlier_rmse']:.4f}")

        if debug:
            import copy
            src_aligned = copy.deepcopy(source).transform(T)
            self.show_pointclouds_open3d(
                [target, source, src_aligned],
                colors=[(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                window_name="ICP: RS (blue) vs Zivid (red) vs Aligned (green)",
            )

        return T, info

    def show_pointclouds_open3d(self, pcds, colors=None, window_name: str = "PointClouds", show_frame: bool = True):
        """Show one or more Open3D point clouds in the same viewer window.

        Parameters
        ----------
        pcds : open3d.geometry.PointCloud | list[open3d.geometry.PointCloud]
            Point cloud(s) to display.
        colors : list[tuple[float, float, float]] | None
            Optional list of RGB tuples (0..1), one per cloud. If a cloud already
            has per-point colours and the matching entry is ``None``, those
            colours are kept. Use this to distinguish clouds (e.g. red vs green).
        window_name : str
            Title of the visualization window.
        show_frame : bool
            If True, also draws a coordinate frame at the base origin.
        """
        import open3d as o3d
        import copy

        if not isinstance(pcds, (list, tuple)):
            pcds = [pcds]

        geometries = []
        for i, pcd in enumerate(pcds):
            if pcd is None or len(pcd.points) == 0:
                log.warning(f"show_pointclouds_open3d: skipping empty cloud {i}")
                continue
            disp = copy.deepcopy(pcd)
            if colors is not None and i < len(colors) and colors[i] is not None:
                disp.paint_uniform_color(colors[i])
            geometries.append(disp)

        if show_frame:
            # Scale frame roughly to scene extent.
            all_pts = np.concatenate([np.asarray(g.points) for g in geometries], axis=0) \
                if geometries else np.zeros((1, 3))
            size = float(np.linalg.norm(all_pts.max(0) - all_pts.min(0))) * 0.1 if len(all_pts) > 1 else 0.1
            size = max(size, 0.05)
            geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0]))

        if not geometries:
            log.warning("show_pointclouds_open3d: nothing to display")
            return

        o3d.visualization.draw_geometries(geometries, window_name=window_name)

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

    def show_projection(self, rs_map, zv_map, zv_valid, idx):
        fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(8,4))
        axes[0].imshow(rs_map, vmin=-10, vmax=1000),axes[0].set_title(f"RealSense Depth Diff (mm)"),
        axes[1].imshow(zv_map, vmin=-10, vmax=1000),axes[1].set_title(f"Zivid Projected Depth Diff (mm)"),
        axes[2].imshow(zv_valid, cmap='gray'),axes[2].set_title(f"Valid Mask (Zivid Projection)"),
        plt.suptitle(f"Sample {idx:03d} Depth Difference Maps and Valid Mask", fontsize=16)
        plt.tight_layout()
        plt.show()


# --------------------------------
#%% Tests
class TestDataSource(unittest.TestCase):

    def test_init_directory(self):
        p       = DataSource()
        img_num = p.init_directory()
        self.assertTrue(img_num > 0)

    def test_get_item(self):
        p       = DataSource()
        img_num = p.init_directory()
        self.assertTrue(img_num > 0)
        out = p.get_item(0, debug=True)
        self.assertTrue(len(out["left"]) > 0)

    def test_show_images(self):
        p       = DataSource()
        img_num = p.init_directory(r'C:\Work\Data\Depth\Data Collection-02')
        if img_num == 0:
            log.warning("No images found.")
            return
        for k in np.random.randint(0, img_num, size=min(8, img_num)):
            out = p.get_item(int(k), debug=True)
            self.assertTrue(len(out["left"]) > 0)
            p.show_subset([out["left"], out["right"], out["depth_zivid"], out["depth_rs"], out["rgb"]],
                          ['left (RS)', 'right (RS)', 'depth Zivid (mm)', 'depth RS (mm)', 'rgb (Zivid)'])

        plt.show()

    def test_get_item_projected(self):
        p       = DataSource()
        #img_num = p.init_directory(r'C:\Work\Data\Depth\Data Collection-02')
        img_num = p.init_directory(r'C:\Work\Data\Depth\Data Collection-03')
        self.assertTrue(img_num > 0)
        for k in np.random.randint(0, img_num, size=min(6, img_num)):
        #for k in range(0, img_num):
            out = p.get_item_projected(int(k), debug=False)
            err = p.compute_depth_error(out["depth_rs"], out["depth_zivid"])
            self.assertTrue(len(out["left"]) > 0)
            p.show_subset([out["left"], out["right"], out["depth_zivid"], out["depth_rs"], err, out["rgb"]],
                          ['left (RS)', 'right (RS)', 'depth Zivid (mm)', 'depth RS (mm)', 'error (mm)', 'rgb (Zivid)'])
        plt.show()


    def test_get_item_transformed_and_projected(self):
        p                  = DataSource()
        p.save_point_cloud = True  # enable saving point clouds for visualization
        #img_num = p.init_directory(r'C:\Work\Data\Depth\Data Collection-02')
        img_num = p.init_directory(r'C:\Work\Data\Depth\Data Collection-03')
        self.assertTrue(img_num > 0)
        for k in np.random.randint(0, img_num, size=min(4, img_num)):
        #for k in range(0, img_num):
            out = p.get_item_transformed_and_projected(int(k), debug=False)
            err = p.compute_depth_error(out["depth_rs"], out["depth_zivid"])
            self.assertTrue(len(out["left"]) > 0)
            p.show_subset([out["left"], out["right"], out["depth_zivid"], out["depth_rs"], err, out["depth_zivid_original"]],
                          ['left (RS)', 'right (RS)', 'depth Zivid (mm)', 'depth RS (mm)', 'error (mm)', 'Original Zivid (mm)'])
        plt.show()

    def test_get_item_transformed_and_projected_using_open3d(self):
        p       = DataSource()
        #img_num = p.init_directory(r'C:\Work\Data\Depth\Data Collection-02')
        img_num = p.init_directory(r'C:\Work\Data\Depth\Data Collection-03')
        self.assertTrue(img_num > 0)
        for k in np.random.randint(0, img_num, size=min(6, img_num)):
        #for k in range(0, img_num):
            out = p.get_item_and_show_open3d(int(k), debug=False)
            err = p.compute_depth_error(out["depth_rs"], out["depth_zivid"])
            self.assertTrue(len(out["left"]) > 0)
            p.show_subset([out["left"], out["right"], out["depth_zivid"], out["depth_rs"], err],
                          ['left (RS)', 'right (RS)', 'depth Zivid (mm)', 'depth RS (mm)', 'error (mm)'])
        plt.show()        

    def test_display_png_files(self):
        p          = DataSource()
        indexes    = [12]
        dir_path = r"C:\Users\udubin\Downloads\20260512_134607"
        #dir_path = r"C:\Work\Code\Fast-FoundationStereo"
        for index in indexes:

            files = [
                os.path.join(dir_path, f"{index:03d}\ir_left.png"),
                os.path.join(dir_path, f"{index:03d}\zivid_depth_mm.png"),
                os.path.join(dir_path, f"{index:03d}\depth_rs_mm.png"),
                os.path.join(dir_path, f"{index:03d}\FS_depth_mm.png"),   
                os.path.join(dir_path, f"{index:03d}\FFS_original_depth_mm.png"),
                os.path.join(dir_path, f"{index:03d}\FFS_inbolt_new_depth_mm.png")     

            ]
            out = []; names = []
            for f in files:
                if os.path.exists(f):
                    fname, ext = os.path.splitext(f)
                    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                    out.append(img)
                    names.append(fname.split('\\')[-1])
                else:
                    log.warning(f"File not found: {f}")
                    out.append(np.array([]))  # placeholder for missing file

            p.show_subset(out, names)

        plt.show()

# --------------------------------
#%% Run Test
def RunTest():
    tst = TestDataSource()
    #tst.test_get_item()
    #tst.test_show_images()
    #tst.test_get_item_projected()
    tst.test_get_item_transformed_and_projected()
    #tst.test_get_item_transformed_and_projected_using_open3d()
    #tst.test_display_png_files()


if __name__ == '__main__':
    RunTest()
