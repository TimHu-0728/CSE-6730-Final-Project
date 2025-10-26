import numpy as np
import pyvista as pv
np.bool = np.bool_

# x, y, z = ...  # your arrays
t = np.linspace(0, 2*np.pi, 1200)
r_orbit = 6778.0
x = r_orbit*np.cos(t); y = r_orbit*np.sin(t); z = 300*np.sin(3*t)

pts = np.column_stack([x, y, z])            # (N, 3)

# ✅ Create a connected polyline from points
line = pv.lines_from_points(pts, close=False)

# Give it thickness for nicer rendering
traj = line.tube(radius=20.0)

# Earth
R_E = 6378.1363
earth = pv.Sphere(radius=R_E, theta_resolution=360, phi_resolution=180)
atmo  = pv.Sphere(radius=R_E*1.02, theta_resolution=360, phi_resolution=180)

plotter = pv.Plotter(window_size=(1000, 800))
plotter.set_background("black")
plotter.add_mesh(earth, color="royalblue", smooth_shading=True, specular=0.1)
plotter.add_mesh(atmo,  color="deepskyblue", opacity=0.08, smooth_shading=True)
plotter.add_mesh(traj,  color="gold", smooth_shading=True)
plotter.add_mesh(pv.Sphere(radius=R_E*0.01, center=pts[-1]), color="orange")

plotter.add_axes(line_width=2)
plotter.show_grid(color="gray", grid="back", location="outer")
plotter.camera.position = (0, -3.5*R_E, 1.5*R_E)
plotter.camera.focal_point = (0, 0, 0)
plotter.camera.up = (0, 0, 1)
plotter.show()



##



