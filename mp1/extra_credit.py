import numpy as np
import open3d as o3d
import cv2
from pathlib import Path

def get_velodyne_rays():
  v_min = -24.9
  v_max = 2.0
  total_lasers = 64
  vertical_radians = np.radians(np.linspace(v_min, v_max, total_lasers))

  horizontal_step = 0.5
  horizontal_radians = np.radians(np.arange(0, 360, horizontal_step))


  ray_directions = []
  for v in vertical_radians:
    for h in horizontal_radians:
      x = np.cos(v) * np.cos(h)
      y = np.cos(v) * np.sin(h)
      z = np.sin(v)
      ray_directions.append([x, y, z])
          
  return np.array(ray_directions, dtype=np.float32)

def extra_credit():
  current_dir = Path(__file__).parent
  output_dir = current_dir / "outputs"
  cache_dir = output_dir / "cache"
  extra_credit_output_dir = output_dir / "extra_credit"
  extra_credit_output_dir.mkdir(parents=True, exist_ok=True)
  
  print("Running Meshifying and LiDAR Simulation")

  landmark_file = cache_dir / "landmarks_observations.npz"
  if not landmark_file.exists():
     raise FileNotFoundError("Landmarks file missing. Run the  SLAM code first by python run.py all")

  data = np.load(landmark_file)
  landmarks_2d = data["landmarks_xy"]
  
  metrics_file = output_dir / "metrics" / "metrics.json"

  center_x, center_y = np.mean(landmarks_2d, axis=0)

  trajectory_poses = []
  if metrics_file.exists():
     for t in np.linspace(0, 2 * np.pi, 100):
      tx = center_x + 20 * np.cos(t)
      ty = center_y + 20 * np.sin(t)
      theta = t + np.pi / 2
      trajectory_poses.append([tx, ty, theta])
  else:
      for t in np.linspace(0, 2 * np.pi, 100):
         trajectory_poses.append([center_x + 15 * np.cos(t), center_y + 15 * np.sin(t), t])
      
  print(f"Loaded {len(landmarks_2d)} optimized landmarks")

  print("Turning 2D landmarks into 3D walls")
  
  z_jitter = np.random.normal(0, 0.001, size=(len(landmarks_2d), 1))
  
  height_layers = [0.0, 0.5, 1.0, 1.5, 2.5]
  all_wall_points = []
  for h in height_layers:
    points_h = np.column_stack((landmarks_2d, z_jitter + h))
    all_wall_points.append(points_h)

  pcd_walls = o3d.geometry.PointCloud()
  pcd_walls.points = o3d.utility.Vector3dVector(np.vstack(all_wall_points))
  pcd_walls.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
  pcd_walls.orient_normals_consistent_tangent_plane(k=15)

  print("Running Poisson Reconstruction")
  wall_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_walls, depth=10)
  wall_mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.15))
  wall_mesh.paint_uniform_color([0.5, 0.5, 0.5]) 

  mesh_out = extra_credit_output_dir / "slam_mesh.ply"
  o3d.io.write_triangle_mesh(str(mesh_out), wall_mesh)
  print(f"Mesh saved here {mesh_out}")

  print("Simulating Velodyne HDL-64E on mesh")
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wall_mesh))
  
  local_rays = get_velodyne_rays()
  video_file = str(extra_credit_output_dir / "lidar_sim.mp4")
  video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (800, 600))

  for i, pose in enumerate(trajectory_poses):
    tx, ty, theta = pose
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
      ])
    
    world_directions = local_rays @ rotation_matrix.T
    origins = np.tile([tx, ty, 1.2], (len(world_directions), 1))
      
    rays = np.hstack([origins, world_directions]).astype(np.float32)
    result = scene.cast_rays(rays)
    dist = result['t_hit'].numpy()
    
    mask = (dist > 0) & (dist < 50)
    hits = origins[mask] + world_directions[mask] * dist[mask][:, None]
      
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    for point in hits:
      pixel_x = int((point[0] - tx) * 12 + 400)
      pixel_y = int((point[1] - ty) * 12 + 300)
      
      if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
         frame[pixel_y, pixel_x] = [0, 255, 0]
    
    cv2.putText(frame, f"HDL-64E Frame {i}", (10, 30), 1, 1.5, (255,255,255), 2)
    video.write(frame)

  video.release()
  print(f"Video saved to {video_file}")

  print("Capturing screenshot")
  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name="MP1", width=1280, height=720)
  vis.add_geometry(wall_mesh)
  
  ctr = vis.get_view_control()
  vis.reset_view_point(True)
  ctr.set_up(np.array([0, 0, 1]))
  ctr.set_front(np.array([-0.8, -0.8, 0.4]))
  ctr.set_zoom(0.7)

  vis.get_render_option().mesh_show_wireframe = True
  
  for _ in range(25):
      vis.poll_events()
      vis.update_renderer()
  
  screenshot_file = str(extra_credit_output_dir / "3d_reconstruction.png")
  vis.capture_screen_image(screenshot_file)
  print(f"Screenshot saved: {screenshot_file}")
  
  vis.run()
  vis.destroy_window()

if __name__ == "__main__":
  extra_credit()