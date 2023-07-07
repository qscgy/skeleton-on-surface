import open3d as o3d
import numpy as np
import time
import open3d.visualization.gui as gui

mesh_file = "/Users/sam/Downloads/mesh_Auto_A_Aug18_09-06-42.mov_001.obj"
mesh = o3d.io.read_triangle_mesh(mesh_file)
print(mesh)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
while True:
    # Rotation 
    R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi / 32))
    mesh.rotate(R, center=(0, 0, 0))

    # Update geometry
    gui.Application.instance.post_to_main_thread(window, update_geometry)

    time.sleep(0.05)