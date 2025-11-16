# # PYVISTA FOR VISUALIZATION

# import numpy as np
# import pyvista as pv
# from vtkmodules.vtkRenderingCore import vtkAssembly
# import subprocess, shutil
# from pathlib import Path

# def jwstVisualizationFixed(x_fixed, y_fixed, z_fixed, x_Earth, y_Earth, z_Earth, r_12, number_of_years, 
#                          jwstModelPath="./assets/models/JWST/scene.gltf", cubeMapPath='./assets/cubemaps',
#                          save_movie=None, framerate=30, step=5, window_size=(1280, 720)):
#     pv.global_theme.allow_empty_mesh = True

#     num_stars = 40000
#     spaceRadius = 10 * r_12
#     starsCords = np.random.uniform(-spaceRadius, spaceRadius, size=(num_stars, 3))
#     starPoints = pv.PolyData(starsCords)

#     jwstOrbit = pv.PolyData()
#     jwstSamples = len(x_fixed)

#     earthOrbit = pv.PolyData()
#     earthSamples = len(x_Earth)

#     # REASONABLE SCALE FOR RADIUS BUT NOT IN SCALE

#     sunRad = 0.1 * r_12
#     earthRad = 0.05 * r_12

#     sun = pv.Sphere(radius = sunRad, center = (0, 0, 0))
#     earth = pv.Sphere(radius = earthRad * 0.3, center = (0, 0, 0))

#     off = bool(save_movie)
#     pl = pv.Plotter(window_size=window_size if save_movie else [3000, 2700], off_screen=off)
#     pl.set_background('black')

#     pl.import_gltf(jwstModelPath, set_camera = False)
#     actors = pl.renderer.GetActors()
#     actors.InitTraversal()
#     jwstActors = []
#     for _ in range(actors.GetNumberOfItems()):
#         actor = actors.GetNextActor()
#         jwstActors.append(actor)
#     jwstAssembly = vtkAssembly()
#     for actor in jwstActors:
#         jwstAssembly.AddPart(actor)

#     jwstAssembly.SetScale(0.001 * r_12)

#     envTex = pv.cubemap(cubeMapPath, 'space-', '.png')
#     pl.set_environment_texture(envTex)

#     pl.add_mesh(jwstOrbit, color = 'cyan', line_width = 2, label = 'JWST Orbit')
#     pl.add_mesh(earthOrbit, color = 'blue', line_width = 1, label = 'Earth Orbit')

#     pl.add_mesh(sun, color='yellow', label='Sun')
#     earthActor = pl.add_mesh(earth, color='deepskyblue', label='Earth')

#     pl.add_mesh(starPoints, color='white', point_size=2 , render_points_as_spheres=True)

#     light = pv.Light(
#         position=(1, 1, 1),  
#         positional=True,                   
#         color='white',
#         intensity=10.0,
#         focal_point=(0, 0, 0)
#     )
#     pl.add_light(light)
#     pl.renderer.ResetCameraClippingRange()

#     pl.add_legend(bcolor=None, face=None, border=False, )
#     pl.add_text(f"JWST Halo Orbit ({number_of_years} Years) - Fixed Frame",
#                     position='upper_left', font_size=12, color='white')
#     pl.add_actor(jwstAssembly)

#     pl.add_axes()

#     def callback(frame):
#         earthPos = np.array([x_Earth[frame], y_Earth[frame], z_Earth[frame]])
#         vecNorm = earthPos / np.linalg.norm(earthPos)
#         cameraDistance = 1 * r_12
#         cameraPos = earthPos + cameraDistance * vecNorm

#         heightOffset = 0.5 * r_12
#         cameraPos[2] += heightOffset

#         freq = 2 * np.pi / 500
#         phase = frame * freq

#         up = np.array([0, 0, 1])
#         right = np.cross(vecNorm, up)
#         right /= np.linalg.norm(right)
        
#         forward = np.cross(right, vecNorm)
        
#         cameraPos += earthRad * 0.3 * np.sin(phase) * right
#         cameraPos += earthRad * 0.3 * np.cos(phase) * forward

#         cam = pl.renderer.GetActiveCamera()
#         cam.SetPosition(*cameraPos)     
#         cam.SetFocalPoint(0, 0, 0)    
#         pl.renderer.ResetCameraClippingRange()

#         lag = max(0, frame-200)
#         xs = x_fixed[lag:frame]
#         ys = y_fixed[lag:frame]
#         zs = z_fixed[lag:frame]
#         jwstOrbit.points = np.column_stack((xs, ys, zs))
#         n = len(xs)
#         if n > 1:
#             lines = np.arange(0, n)
#             lines = np.insert(lines, 0, n)
#             jwstOrbit.lines = lines

#         earthOrbit.points = np.column_stack((x_Earth[lag:frame], y_Earth[lag:frame], z_Earth[lag:frame]))
#         nE = len(x_Earth[lag:frame])
#         if nE > 1:
#             lines = np.arange(0, nE)
#             lines = np.insert(lines, 0, nE)
#             earthOrbit.lines = lines

#         jwstAssembly.SetPosition(x_fixed[frame], y_fixed[frame], z_fixed[frame])
#         earthActor.position = [x_Earth[frame], y_Earth[frame], z_Earth[frame]]

#         pl.render()
    
#     if save_movie:
#         out = Path(save_movie)
#         out.parent.mkdir(parents=True, exist_ok=True)
#         tmp_dir = out.parent / (out.stem + "_frames")
#         tmp_dir.mkdir(parents=True, exist_ok=True)

#         # ensure ffmpeg exists
#         assert shutil.which("ffmpeg"), "ffmpeg not found on PATH"

#         # off-screen render for consistent frames
#         pl.off_screen = True
#         pl.window_size = window_size

#         frame_ids = range(0, jwstSamples, max(1, step))
#         for k, frame in enumerate(frame_ids):
#             callback(frame)  # your existing update logic
#             pl.render()
#             # save PNG frame_000000.png, frame_000001.png, ...
#             pl.screenshot(filename=str(tmp_dir / f"frame_{k:06d}.png"),
#                         return_img=False, window_size=window_size)

#         pl.close()

#         # stitch PNGs -> MP4 (QuickTime-friendly)
#         cmd = [
#             "ffmpeg",
#             "-y",
#             "-framerate", str(framerate),
#             "-i", str(tmp_dir / "frame_%06d.png"),
#             "-vcodec", "libx264",
#             "-pix_fmt", "yuv420p",
#             "-movflags", "+faststart",
#             str(out)
#         ]
#         subprocess.run(cmd, check=True)

#         # optional: clean up frames
#         for p in tmp_dir.glob("frame_*.png"):
#             p.unlink()
#         tmp_dir.rmdir()
#     else:
#         pl.add_timer_event(max_steps=jwstSamples, duration=500, callback=callback)
#         pl.show()

# PYVISTA FOR VISUALIZATION

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkAssembly
import subprocess, shutil
from pathlib import Path

def jwstVisualizationFixed(
    x_fixed, y_fixed, z_fixed,
    x_Earth, y_Earth, z_Earth,
    r_12, number_of_years,
    jwstModelPath="./assets/models/JWST/scene.gltf",
    cubeMapPath='./assets/cubemaps',
    save_movie=None,
    framerate=30,
    step=5,
    window_size=(1280, 720),
    title="JWST Orbit"         # <-- 唯一新增参数
):
    pv.global_theme.allow_empty_mesh = True

    num_stars = 40000
    spaceRadius = 10 * r_12
    starsCords = np.random.uniform(-spaceRadius, spaceRadius, size=(num_stars, 3))
    starPoints = pv.PolyData(starsCords)

    jwstOrbit = pv.PolyData()
    jwstSamples = len(x_fixed)

    earthOrbit = pv.PolyData()
    earthSamples = len(x_Earth)

    sunRad = 0.1 * r_12
    earthRad = 0.05 * r_12

    sun = pv.Sphere(radius=sunRad, center=(0, 0, 0))
    earth = pv.Sphere(radius=earthRad * 0.3, center=(0, 0, 0))

    off = bool(save_movie)
    pl = pv.Plotter(window_size=window_size if save_movie else [3000, 2700], off_screen=off)
    pl.set_background('black')

    pl.import_gltf(jwstModelPath, set_camera=False)
    actors = pl.renderer.GetActors()
    actors.InitTraversal()
    jwstActors = []
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        jwstActors.append(actor)
    jwstAssembly = vtkAssembly()
    for actor in jwstActors:
        jwstAssembly.AddPart(actor)

    jwstAssembly.SetScale(0.001 * r_12)

    envTex = pv.cubemap(cubeMapPath, 'space-', '.png')
    pl.set_environment_texture(envTex)

    pl.add_mesh(jwstOrbit, color='cyan', line_width=2, label='JWST Orbit')
    pl.add_mesh(earthOrbit, color='blue', line_width=1, label='Earth Orbit')

    pl.add_mesh(sun, color='yellow', label='Sun')
    earthActor = pl.add_mesh(earth, color='deepskyblue', label='Earth')

    pl.add_mesh(starPoints, color='white', point_size=2, render_points_as_spheres=True)

    light = pv.Light(
        position=(1, 1, 1),
        positional=True,
        color='white',
        intensity=10.0,
        focal_point=(0, 0, 0)
    )
    pl.add_light(light)
    pl.renderer.ResetCameraClippingRange()

    pl.add_legend(bcolor=None, face=None, border=False)
    pl.add_text(
        f"{title} ({number_of_years} Years) - Fixed Frame",    # <-- 修改点
        position='upper_left',
        font_size=12,
        color='white'
    )
    pl.add_actor(jwstAssembly)

    pl.add_axes()

    def callback(frame):
        earthPos = np.array([x_Earth[frame], y_Earth[frame], z_Earth[frame]])
        vecNorm = earthPos / np.linalg.norm(earthPos)
        cameraDistance = 1 * r_12
        cameraPos = earthPos + cameraDistance * vecNorm

        heightOffset = 0.5 * r_12
        cameraPos[2] += heightOffset

        freq = 2 * np.pi / 500
        phase = frame * freq

        up = np.array([0, 0, 1])
        right = np.cross(vecNorm, up)
        right /= np.linalg.norm(right)

        forward = np.cross(right, vecNorm)

        cameraPos += earthRad * 0.3 * np.sin(phase) * right
        cameraPos += earthRad * 0.3 * np.cos(phase) * forward

        cam = pl.renderer.GetActiveCamera()
        cam.SetPosition(*cameraPos)
        cam.SetFocalPoint(0, 0, 0)
        pl.renderer.ResetCameraClippingRange()

        lag = max(0, frame - 200)
        xs = x_fixed[lag:frame]
        ys = y_fixed[lag:frame]
        zs = z_fixed[lag:frame]
        jwstOrbit.points = np.column_stack((xs, ys, zs))
        n = len(xs)
        if n > 1:
            lines = np.arange(0, n)
            lines = np.insert(lines, 0, n)
            jwstOrbit.lines = lines

        earthOrbit.points = np.column_stack((x_Earth[lag:frame], y_Earth[lag:frame], z_Earth[lag:frame]))
        nE = len(x_Earth[lag:frame])
        if nE > 1:
            lines = np.arange(0, nE)
            lines = np.insert(lines, 0, nE)
            earthOrbit.lines = lines

        jwstAssembly.SetPosition(x_fixed[frame], y_fixed[frame], z_fixed[frame])
        earthActor.position = [x_Earth[frame], y_Earth[frame], z_Earth[frame]]

        pl.render()

    if save_movie:
        out = Path(save_movie)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = out.parent / (out.stem + "_frames")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        assert shutil.which("ffmpeg"), "ffmpeg not found on PATH"

        pl.off_screen = True
        pl.window_size = window_size

        frame_ids = range(0, jwstSamples, max(1, step))
        for k, frame in enumerate(frame_ids):
            callback(frame)
            pl.render()
            pl.screenshot(
                filename=str(tmp_dir / f"frame_{k:06d}.png"),
                return_img=False,
                window_size=window_size
            )

        pl.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(framerate),
            "-i", str(tmp_dir / "frame_%06d.png"),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out)
        ]
        subprocess.run(cmd, check=True)

        for p in tmp_dir.glob("frame_*.png"):
            p.unlink()
        tmp_dir.rmdir()
    else:
        pl.add_timer_event(max_steps=jwstSamples, duration=500, callback=callback)
        pl.show()
