#%% Data extraction from scene capturing json file:
# Get the camera point cloud (ply) generated from vertices (raw data)
# Get the aligned CAD into the camera frame

import open3d as o3d
import re
from pathlib import Path
import numpy as np
import json
import pandas as pd

def compose_t_camera_cad(
    t_camera_tooltip: np.ndarray,
    t_tooltip_cad_raw: np.ndarray,
) -> np.ndarray:
    """Chain t_camera_tooltip and t_tooltip_cad into a single transform.

    Recomposes t_tooltip_cad as T_rot @ T_trans (translate-then-rotate) so
    that the offset translation moves the CAD in the tooltip frame first,
    and the rotation is then applied around the original tooltip axes.
      Standard [R|t]: p' = R*p + t  (rotate-then-translate)
      Desired  [R|R@t]: p' = R*(p+t)  (translate-then-rotate)
    """
    T_rot = np.eye(4)
    T_rot[:3, :3] = t_tooltip_cad_raw[:3, :3]
    T_trans = np.eye(4)
    T_trans[:3, 3] = t_tooltip_cad_raw[:3, 3]
    t_tooltip_cad = T_rot @ T_trans
    print(f"T_tooltip_cad (translate-then-rotate):\n{t_tooltip_cad}")
    t_camera_cad =t_camera_tooltip @ t_tooltip_cad
    print(f"Calculated T_camera_cad:\n{t_camera_cad}")
    return t_camera_cad #Cad frame to camera frame transformation 


def load_vertices_to_pcd(path: Path) -> o3d.geometry.PointCloud:
    """
    Load a Vertices .bin file and return an (N, 3) float32 array of valid
    XYZ points (in mm).  Invalid/missing points (Z == 0) are removed.
    """
    def _parse_resolution(filename: str) -> tuple[int, int]:
        """Return (width, height) extracted from a filename like '…_1280x720_…'."""
        match = re.search(r"(\d+)x(\d+)", filename)
        if not match:
            raise ValueError(
                f"Cannot parse resolution from filename: {filename!r}. "
                "Expected a pattern like '1280x720'."
            )
        return int(match.group(1)), int(match.group(2))

    width, height = _parse_resolution(path.name)

    pixel_data_bytes = width * height * 3 * 4  # float32

    raw = path.read_bytes()
    if len(raw) < pixel_data_bytes:
        raise ValueError(
            f"File too small: expected at least {pixel_data_bytes} bytes "
            f"for {width}×{height} vertices, got {len(raw)}."
        )

    # Trim off any trailing metadata before parsing
    xyz_flat = np.frombuffer(raw[:pixel_data_bytes], dtype=np.float32).copy()
    xyz = xyz_flat.reshape(height * width, 3)  # (N, 3)

    # Remove invalid points (RealSense marks missing depth as Z=0)
    valid = xyz[:, 2] != 0.0
    xyz = xyz[valid] * 0.001 # convert from mm to meters


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print(f"Loaded vertices point cloud with {len(pcd.points)} valid points and {len(pcd.normals)} normals")
    return pcd

#% Scene Data:
#Load scene (captures) json file: 
 
address = r"\\svm.realsenseai.com\RealSense_Validation\VIDB\Public\Stavush\Pickle\Data\Data for model training\data_14_6_26.xlsx"

root_df = pd.read_excel(address)

root_df_stl = root_df[root_df['Label']=='raw data (json file)']
root_df_stl = root_df_stl[root_df_stl['STL name']=='cube_100x100x100']

json_address = root_df_stl.loc[1,"Data Path"]

with open(json_address) as f:
    data = json.load(f)    
    
cad_path = data['cad_path'] #stl path - Remain constant for all captures

t_camera_to_tooltip = np.array(data['t_camera_tooltip']) # hand eye - camera to tooltip

# intrinsics
fx = data['intrinsics']['fx']
fy = data['intrinsics']['fy']
px = data['intrinsics']['ppx']
py = data['intrinsics']['ppy']

# Here can wrap in loop as necessary

n =7 #captures iteration number

images = data['captures'][n]['images']

camera_vertices_path = None
for img in images:
    if img.get('image_type') == 'Vertices':
        camera_vertices_path = img.get('path') #Vertices path for iteration n
    elif img.get('image_type') == 'Depth':
        camera_depth_path = img.get('path') #depth path for iteration n 
    elif img.get('image_type') == 'IR':
        camera_ir_path = img.get('path') #left ir path for iteration n    
    elif img.get('image_type') == 'RightIR':
        camera_rightir_path = img.get('path') #right ir path for iteration n     
        
# Robot positions:
robot_pos = data['captures'][n]['robot_position'] #Robot position
robot_pos_x = data['captures'][n]['robot_position']['x'] #Robot position
robot_pos_y = data['captures'][n]['robot_position']['y'] #Robot position
robot_pos_z = data['captures'][n]['robot_position']['z'] #Robot position
robot_pos_rx = data['captures'][n]['robot_position']['rx'] #Robot position
robot_pos_ry = data['captures'][n]['robot_position']['ry'] #Robot position
robot_pos_rz = data['captures'][n]['robot_position']['rz'] #Robot position


t_cad_to_tooltip = np.array(data['captures'][n]['t_tooltip_cad']) #Cad to tooltip transformation for iteration n


#% Vertices to Point Cloud, CAD alignment to camera frame:
depth_pcd_raw = load_vertices_to_pcd(Path(camera_vertices_path))

# Load CAD mesh (STL)
cad_mesh = o3d.io.read_triangle_mesh(
    cad_path
)

# STL units are in mm, convert to meters:
cad_mesh.scale(0.001, center=(0, 0, 0))

# Ensure mesh has normals
if not cad_mesh.has_vertex_normals():
    cad_mesh.compute_vertex_normals()

# Sample points from mesh to create point cloud
cad_pcd = cad_mesh.sample_points_poisson_disk(
    number_of_points=10000,
    use_triangle_normal=True
)      


# CAD after alignment:
cad_pcd_aligned = o3d.geometry.PointCloud(cad_pcd)

R_fix = o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0])

T_fix = np.eye(4)
T_fix[:3, :3] = R_fix

t_cad_to_tooltip = T_fix @ t_cad_to_tooltip

R = t_camera_to_tooltip[:3, :3]
t = t_camera_to_tooltip[:3, 3]

R_inv = R.T
t_inv = -R_inv @ t

T_tooltip_to_camera = np.eye(4)
T_tooltip_to_camera[:3, :3] = R_inv
T_tooltip_to_camera[:3, 3] = t_inv



T_camera_cad = compose_t_camera_cad(T_tooltip_to_camera, t_cad_to_tooltip)

cad_pcd_aligned.transform(T_camera_cad)

cad_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # gray
cad_pcd_aligned.paint_uniform_color([0, 0, 1]) # blue
depth_pcd_raw.paint_uniform_color([1, 0, 0]) # red

# Visualize:
o3d.visualization.draw([
     cad_pcd,
     cad_pcd_aligned,
     depth_pcd_raw,     
 ])
#%% ICP stage:
import pandas as pd
import ast

root_df_icp = root_df[root_df['Label']=='ICP']
root_df_icp = root_df_icp[root_df_icp['STL name']=='cube_100x100x100']

icp_address = root_df_icp.loc[0,"Data Path"]

df_icp = pd.read_csv(icp_address)

T_icp_matrix = np.array(
    ast.literal_eval(df_icp.loc[n, 'icp_matrix'])
)

T_camera_cad = np.array(
    ast.literal_eval(df_icp.loc[n, 't_camera_cad'])
)

T_camera_cad_icp = T_icp_matrix@T_camera_cad

cad_pcd_aligned_icp = o3d.geometry.PointCloud(cad_pcd)
cad_pcd_aligned_icp.transform(T_camera_cad_icp)

cad_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # gray
cad_pcd_aligned.paint_uniform_color([0, 0, 1]) # blue
depth_pcd_raw.paint_uniform_color([1, 0, 0]) # red
cad_pcd_aligned_icp.paint_uniform_color([0, 0, 0]) 
# Visualize:
o3d.visualization.draw([
     cad_pcd,
     cad_pcd_aligned,
     depth_pcd_raw,     
     cad_pcd_aligned_icp,
 ])


