

#%% Data extraction from scene capturing json file:
# Get the camera point cloud (ply) generated from vertices (raw data)
# Get the aligned CAD into the camera frame
 
import open3d as o3d
import re
from pathlib import Path
import numpy as np
import json
import pandas as pd
 
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
address =r"\\svm.realsenseai.com\RealSense_Validation\VIDB\IQ_AUTO\IQLab0\2026_06\yg_pickle\2026-06-16--15-12-54\Pickle_Scene_Capture_336222073841\data path.xlsx"
 
root_df = pd.read_excel(address)
 
root_df_stl = root_df[root_df['Label']=='raw data (json file)']
root_df_stl = root_df_stl[root_df_stl['STL name']=='cube_100x100x100']
 
json_address = root_df_stl.loc[1,"Data Path"]
 
with open(json_address) as f:
    data = json.load(f)    
cad_path = data['cad_path'] #stl path - Remain constant for all captures
 
t_camera_to_tooltip = np.array(data['t_camera_tooltip']) # hand eye - camera to tooltip
 
T_CAD_to_user = np.array([[ 9.99999959e-01,  9.09367280e-05, -2.72211424e-04,
         4.26479000e-01],
       [ 9.07571176e-05, -9.99999778e-01, -6.59759117e-04,
        -5.75152000e-01],
       [-2.72271360e-04,  6.59734385e-04, -9.99999745e-01,
         4.60300000e-03],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
 
T_user_to_base = np.array([[ 9.99943033e-01,  5.68845315e-03,  9.03171840e-03,
         4.52634000e-01],
       [ 5.68252948e-03, -9.99983622e-01,  6.81399922e-04,
        -5.87378000e-01],
       [ 9.03544659e-03, -6.30038099e-04, -9.99958981e-01,
        -4.18047000e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
n = 10 # capture now
robot_pos_x = data['captures'][n]['robot_position']['x'] #Robot position
robot_pos_y = data['captures'][n]['robot_position']['y'] #Robot position
robot_pos_z = data['captures'][n]['robot_position']['z'] #Robot position
robot_pos_rx = data['captures'][n]['robot_position']['rx'] #Robot position
robot_pos_ry = data['captures'][n]['robot_position']['ry'] #Robot position
robot_pos_rz = data['captures'][n]['robot_position']['rz'] #Robot position
 
t_be = np.array([robot_pos_x, robot_pos_y, robot_pos_z]) / 1000.0
 
rx = np.deg2rad(robot_pos_rx) 
ry = np.deg2rad(robot_pos_ry) 
rz = np.deg2rad(robot_pos_rz)
 
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(rx), -np.sin(rx)],
    [0, np.sin(rx),  np.cos(rx)]
])
 
Ry = np.array([
    [ np.cos(ry), 0, np.sin(ry)],
    [0,           1, 0],
    [-np.sin(ry), 0, np.cos(ry)]
])
 
Rz = np.array([
    [np.cos(rz), -np.sin(rz), 0],
    [np.sin(rz),  np.cos(rz), 0],
    [0, 0, 1]
])
 
R_be = Rz @ Ry @ Rx
 
T_tool_to_base = np.eye(4)
T_tool_to_base[:3, :3] = R_be
T_tool_to_base[:3, 3] = t_be
 
T_base_to_tool = np.linalg.inv(T_tool_to_base)

 
n = 20 #captures iteration number
 
robot_pos = data['captures'][n]['robot_position'] #Robot position
robot_pos_x = data['captures'][n]['robot_position']['x'] #Robot position
robot_pos_y = data['captures'][n]['robot_position']['y'] #Robot position
robot_pos_z = data['captures'][n]['robot_position']['z'] #Robot position
robot_pos_rx = data['captures'][n]['robot_position']['rx'] #Robot position
robot_pos_ry = data['captures'][n]['robot_position']['ry'] #Robot position
robot_pos_rz = data['captures'][n]['robot_position']['rz'] #Robot position
 
t_be = np.array([robot_pos_x, robot_pos_y, robot_pos_z]) / 1000.0
 
rx = np.deg2rad(robot_pos_rx) 
ry = np.deg2rad(robot_pos_ry) 
rz = np.deg2rad(robot_pos_rz)
 
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(rx), -np.sin(rx)],
    [0, np.sin(rx),  np.cos(rx)]
])
 
Ry = np.array([
    [ np.cos(ry), 0, np.sin(ry)],
    [0,           1, 0],
    [-np.sin(ry), 0, np.cos(ry)]
])
 
Rz = np.array([
    [np.cos(rz), -np.sin(rz), 0],
    [np.sin(rz),  np.cos(rz), 0],
    [0, 0, 1]
])
 
R_be = Rz @ Ry @ Rx
 
T_tool_to_base = np.eye(4)
T_tool_to_base[:3, :3] = R_be
T_tool_to_base[:3, 3] = t_be
 
T_base_to_tool = np.linalg.inv(T_tool_to_base)
 
 
# intrinsics
fx = data['intrinsics']['fx']
fy = data['intrinsics']['fy']
px = data['intrinsics']['ppx']
py = data['intrinsics']['ppy']
 
# Here can wrap in loop as necessary
 
 
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
 
R = t_camera_to_tooltip[:3, :3]
t = t_camera_to_tooltip[:3, 3]
 
R_inv = R.T
t_inv = -R_inv @ t
 
T_tooltip_to_camera = np.eye(4)
T_tooltip_to_camera[:3, :3] = R_inv
T_tooltip_to_camera[:3, 3] = t_inv
 
 
T_camera_cad = T_tooltip_to_camera @ T_base_to_tool @ T_user_to_base @ T_CAD_to_user
 
 
cad_pcd_aligned.transform(T_camera_cad)
 
cad_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # gray
cad_pcd_aligned.paint_uniform_color([0, 0, 1]) # blue
 
 
points = np.asarray(depth_pcd_raw.points)
 
# Example: keep only points within a max distance from origin
max_dist = 0.7
 
mask = np.linalg.norm(points, axis=1) < max_dist
filtered_points = points[mask]
 
depth_pcd_raw.points = o3d.utility.Vector3dVector(filtered_points)
depth_pcd_raw.paint_uniform_color([1, 0, 0]) # red
 
# Visualize:
o3d.visualization.draw([
     cad_pcd,
     cad_pcd_aligned,
     depth_pcd_raw,     
])