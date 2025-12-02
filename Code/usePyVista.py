# # PYVISTA FOR VISUALIZATION

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
        pl.remove_actor(actor)

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

    pl.add_legend(bcolor=None, face=None, border=False)
    pl.add_text(
        f"{title} ({number_of_years} Years) - Fixed Frame",    # <-- 修改点
        position='upper_left',
        font_size=12,
        color='white'
    )
    pl.add_actor(jwstAssembly)

    pl.add_axes()

    pl.renderer.ResetCameraClippingRange()
    cameraPos = np.array([r_12 * 0.1, r_12 * 0.1, r_12 * 0.1])
    cam = pl.renderer.GetActiveCamera()
    cam.SetPosition(*cameraPos)
    cam.SetFocalPoint(0, 0, 0)
    def callback(frame):
        # earthPos = np.array([x_Earth[frame], y_Earth[frame], z_Earth[frame]])
        # vecNorm = earthPos / np.linalg.norm(earthPos)
        # cameraDistance = 1 * r_12
        # cameraPos = earthPos + cameraDistance * vecNorm

        # heightOffset = 0.5 * r_12
        # cameraPos[2] += heightOffset

        # freq = 2 * np.pi / 500
        # phase = frame * freq

        # up = np.array([0, 0, 1])
        # right = np.cross(vecNorm, up)
        # right /= np.linalg.norm(right)

        # forward = np.cross(right, vecNorm)

        # cameraPos += earthRad * 0.3 * np.sin(phase) * right
        # cameraPos += earthRad * 0.3 * np.cos(phase) * forward


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
        pl.show(cpos = cameraPos)


def jwstVisualizationRot(
    x_JWST, y_JWST, z_JWST,
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
    spaceRadius = 0.05
    starsCords = np.random.uniform(-spaceRadius, spaceRadius, size=(num_stars, 3))
    starsCords += np.array([x_Earth, y_Earth, z_Earth])
    starPoints = pv.PolyData(starsCords)

    floor = pv.Plane(center=(x_Earth,y_Earth,z_Earth), direction=(0,0,1), 
                    i_size=0.25, j_size=0.25, 
                    i_resolution=100, j_resolution=100)
    jwstOrbit = pv.PolyData()
    jwstSamples = len(x_JWST)

    earthRad = 0.0005 * 0.3
    earth = pv.Sphere(center=(0,0,0), radius=earthRad, theta_resolution=120, phi_resolution=120, start_theta=270.001, end_theta=270,)
    earth.active_texture_coordinates = np.zeros((earth.points.shape[0], 2))
    for i in range(earth.points.shape[0]):
        earth.active_texture_coordinates[i] = [
            0.5 + np.arctan2(-earth.points[i, 0] / earthRad, earth.points[i, 1] / earthRad) / (2 * np.pi),
            0.5 + np.arcsin(earth.points[i, 2] / earthRad) / np.pi,
        ]
    earth.translate((x_Earth, y_Earth, z_Earth), inplace=True)
    off = bool(save_movie)
    pl = pv.Plotter(window_size=window_size if save_movie else [3500, 2700], off_screen=off)
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
        pl.remove_actor(actor)

    jwstAssembly.SetScale(0.000003)

    envTex = pv.cubemap(cubeMapPath, 'space-', '.png')
    pl.set_environment_texture(envTex)

    earthTexture = pv.read_texture("./assets/textures/planets/earth.jpg")

    # pl.add_mesh(floor, color='white', style = 'wireframe', show_edges=True, opacity=0.5, label="XY Plane")
    pl.add_mesh(earth, texture=earthTexture, smooth_shading=True)

    pl.add_mesh(jwstOrbit, color='cyan', line_width=2, label='JWST Orbit')
    pl.add_mesh(starPoints, color='white', point_size=2, render_points_as_spheres=True)

    light = pv.Light(
        position=(0, 0, 0.09),
        positional=True,
        color='white',
        intensity=10.0,
        focal_point=(0, 0, 0)
    )
    pl.add_light(light)

    pl.add_legend(bcolor=None, face=None, border=False)
    pl.add_text(
        f"{title} ({number_of_years} Years) - Rotating Frame",    # <-- 修改点
        position='upper_left',
        font_size=12,
        color='white'
    )
    pl.add_actor(jwstAssembly)

    pl.add_axes()

    def callback(frame):

        earth.rotate_z(1.5, point=(x_Earth, y_Earth, z_Earth), inplace=True)

        lag = max(0, frame - 500)
        xs = x_JWST[lag:frame] / r_12
        ys = y_JWST[lag:frame] / r_12
        zs = z_JWST[lag:frame] / r_12
        jwstOrbit.points = np.column_stack((xs, ys, zs))
        n = len(xs)
        if n > 1:
            lines = np.arange(0, n)
            lines = np.insert(lines, 0, n)
            jwstOrbit.lines = lines

        dx = x_Earth - x_JWST[frame] / r_12
        dy = y_Earth - y_JWST[frame] / r_12
        dz = z_Earth - z_JWST[frame] / r_12
        yaw = np.degrees(np.arctan2(dy, dx))
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        pitch = np.degrees(np.arcsin(dz / distance))
        jwstAssembly.SetPosition(x_JWST[frame] / r_12, y_JWST[frame] / r_12, z_JWST[frame] / r_12)
        jwstAssembly.SetOrientation(0, -pitch, yaw + 90)
   

    if save_movie:
        out = Path(save_movie)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = out.parent / (out.stem + "_frames")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        assert shutil.which("ffmpeg"), "ffmpeg not found on PATH"

        pl.camera.zoom(205.8) 
        pl.off_screen = True
        pl.window_size = window_size
        callback(0)
        cam = pl.renderer.GetActiveCamera()
        
        initial_cam_pos = np.array([0.5, 0.5, 0.5])
        cam.SetPosition(*initial_cam_pos)
        cam.SetFocalPoint(x_Earth, y_Earth, z_Earth)
        pl.render() 
        pl.renderer.ResetCameraClippingRange()

        zoom_start = 333
        zoom_end = 333 + 501
        zoom_steps = zoom_end - zoom_start
        target_pos_offset = np.array([-0.1, -0.1, -0.15])
        prev_progress = 0

        print(f"Rendering {jwstSamples} frames to {tmp_dir}...")

        for frame in range(jwstSamples):
            callback(frame)
            if zoom_start < frame < zoom_end:
                t = (frame - zoom_start) / zoom_steps
                
                progress = t * t * (3 - 2 * t)
                delta_p = progress - prev_progress
                prev_progress = progress

                cam.SetFocalPoint(x_Earth, y_Earth, z_Earth + 0.0052 * progress)

                jwstAssembly.SetScale(0.000003 + progress * 0.00003)

                new_pos = initial_cam_pos + (target_pos_offset * progress)
                cam.SetPosition(*new_pos)

                pl.camera.zoom(0.1 ** delta_p)
                pl.update() 

            pl.renderer.ResetCameraClippingRange()
            pl.render()
            pl.screenshot(
                filename=str(tmp_dir / f"frame_{frame:06d}.png"),
                return_img=False,
                window_size=window_size
            )

        pl.close()

        print("Encoding video...")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(framerate),
            "-i", str(tmp_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p", 
            "-crf", "18", 
            "-movflags", "+faststart", 
            str(out)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Movie saved to: {out}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed: {e.stderr.decode()}")
        finally:
            # cleanup
            for p in tmp_dir.glob("frame_*.png"):
                p.unlink()
            tmp_dir.rmdir()
    else:
        pl.camera.zoom(205.8)
        pl.show(interactive_update=True)
        
        frame = 0
        import time

        zoomStartFrame = 333
        zoomEndFrame = 333 + 501
        zoomSteps = zoomEndFrame - zoomStartFrame
        cam = pl.renderer.GetActiveCamera()
        cam.SetFocalPoint(x_Earth, y_Earth, z_Earth)
        cameraPos = np.array([0.5, 0.5, 0.5])
        cam.SetPosition(*cameraPos)
        pl.renderer.ResetCameraClippingRange()

        current_duration = 0
        prevProgress = 0
        while frame < jwstSamples:
            if 0 < frame < zoomStartFrame:
                current_duration = 0.01 
            elif zoomStartFrame < frame < zoomEndFrame: 
                t = (frame - zoomStartFrame) / zoomSteps
                progress = t * t * (3 - 2 * t) 
                deltaP = progress - prevProgress
                prevProgress = progress
                cam.SetFocalPoint(x_Earth, y_Earth, z_Earth + 0.0052 * progress)
                jwstAssembly.SetScale(0.000003 + progress * 0.00003)
                cameraPos = np.array([0.5 + (-0.1) * progress, 0.5 + (-0.1) * progress, 0.5 + (-0.15) * progress])
                cam.SetPosition(*cameraPos)
                pl.camera.zoom(0.1 ** deltaP)
                current_duration = 0.001
            if frame == zoomEndFrame + 1:
                current_duration = 0.001
            callback(frame)
            pl.update() 
            
            pl.renderer.ResetCameraClippingRange()
            pl.render()
            time.sleep(current_duration)
            
            frame += 1