# from typing import Tuple
# from math import *
# import numpy as np
# import control as ct
# from scipy.spatial.transform import Rotation as Rot
# import plotTraj
# from classes import *
# from usePyVista import jwstVisualizationFixed 
# from pathlib import Path

# OUTPUT_DIR = Path("results/animation"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Instantiate Three body system and its compositions. 
# TB = ThreeBodySystem()

# # Set the initial states for JWST in Rotating frame and nondimensional unit
# # Unstable Halo Orbit
# TB.JamesWebb.X0 = np.array([1.0062010416592476,             # x
#                             -9.8074369820712423e-16,       # xdot
#                             -2.1523078518259630e-23,       # y
#                             -1.3253477924660511e-2,        # ydot
#                             1.2380311201349303e-2,        # z
#                             -1.0956603459061133e-14])      # zdot

# # Stable Halo Orbit
# # TB.JamesWebb.X0 = np.array([1.0065075924336142E+0,             # x
# #                             2.6228553664868397E-16,       # xdot
# #                             -2.1016745594119651E-23,       # y
# #                             -1.3729810632559413E-2,        # ydot
# #                             1.2329298254423008E-2,        # z
# #                             -2.4057058092832893E-15])      # zdot

# # Define parameters used in JWST dynamics
# JWST_params = {'pi_1':TB.pi_1,'pi_2':TB.pi_2}

# # Define the Input/Output System
# JWST_IO = ct.nlsys(TB.JamesWebb.JWST_update_nondim, TB.JamesWebb.JWST_output,
#                     states=['x','xdot','y','ydot','z','zdot'], name = 'JWST',
#                     inputs=['ux','uy','uz'], outputs=['x','xdot','y','ydot','z','zdot'],
#                     params=JWST_params)

# # Simulation
# number_of_years = 2
# timepts_year = 1000
# timepts = np.linspace(0,number_of_years*TB.year,number_of_years*timepts_year)       # Nondimensinal time points
# u = [np.zeros_like(timepts),np.zeros_like(timepts),np.zeros_like(timepts)]
# response = ct.input_output_response(JWST_IO, timepts=timepts, inputs=u, initial_state=TB.JamesWebb.X0,
#                                     solve_ivp_kwargs={'rtol':1e-9,'atol':1e-12}, solve_ivp_method='RK45')
# time, outputs, inputs = response.time, response.outputs, response.inputs

# # Retrive JWST trajectories in rotation frame and dimentionalize them
# x = TB.r_12 * outputs['x']
# y = TB.r_12 * outputs['y']
# z = TB.r_12 * outputs['z']
# xdot = TB.V_C * outputs['xdot']
# ydot = TB.V_C * outputs['ydot']
# zdot = TB.V_C * outputs['zdot']
# t = TB.t_C * time

# # Plot the JWST Trajectory
# plotTraj.Plot_static_RF(x,y,z,t,TB.r_12,TB.x_L2,TB.Earth.x, save_path=OUTPUT_DIR / "jwst_rf_static.png")
# plotTraj.Animation_RF(x,y,z,t,TB.r_12,TB.x_L2,TB.Earth.x, save_path=OUTPUT_DIR / "jwst_rf.mp4", fps=30, dpi=200)

# # Transfer to Fixed Frame
# XYZ = TB.RotToFixed(np.array([x,y,z]).T, TB.Omega, t)
# x_fixed = XYZ[:,0]
# y_fixed = XYZ[:,1]
# z_fixed = XYZ[:,2]
# x_Earth = 0.97*TB.r_12*np.cos(TB.Omega*t)
# y_Earth = 0.97*TB.r_12*np.sin(TB.Omega*t)
# z_Earth = np.zeros_like(x_Earth)

# plotTraj.Animation_FF(x_fixed,y_fixed,z_fixed,x_Earth,y_Earth,z_Earth,t,TB.r_12, save_path=OUTPUT_DIR / "jwst_ff.mp4", fps=30, dpi=200)

# # PYVISTA FOR VISUALIZATION
# jwstVisualizationFixed(x_fixed, y_fixed, z_fixed, x_Earth, y_Earth, z_Earth, TB.r_12, number_of_years, 
#                          jwstModelPath="./assets/models/JWST/scene.gltf", cubeMapPath='./assets/cubemaps/space',
#                          save_movie=OUTPUT_DIR / "jwst_pv_fixed.mp4")

from typing import Tuple
from math import *
import numpy as np
import control as ct
from scipy.spatial.transform import Rotation as Rot
import plotTraj
from classes import *
from usePyVista import jwstVisualizationFixed 
from pathlib import Path

OUTPUT_DIR = Path("results/animation"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TB = ThreeBodySystem()

TB.JamesWebb.X0 = np.array([
    1.0062010416592476,
    -2.1523078518259630e-23,
    1.2380311201349303e-2,
    -9.8074369820712423e-16,
    -1.3253477924660511e-2,
    -1.0956603459061133e-14
])

JWST_params = {'pi_1':TB.pi_1,'pi_2':TB.pi_2}

JWST_IO = ct.nlsys(
    TB.JamesWebb.JWST_update_nondim, TB.JamesWebb.JWST_output,
    states=['x','y','z','xdot','ydot','zdot'],
    name='JWST',
    inputs=['ux','uy','uz'],
    outputs=['x','y','z','xdot','ydot','zdot'],
    params=JWST_params
)

number_of_years = 2
timepts_year = 1000
timepts = np.linspace(0,number_of_years*TB.year,number_of_years*timepts_year)
u = [np.zeros_like(timepts),np.zeros_like(timepts),np.zeros_like(timepts)]

response = ct.input_output_response(
    JWST_IO, timepts=timepts, inputs=u, initial_state=TB.JamesWebb.X0,
    solve_ivp_kwargs={'rtol':1e-9,'atol':1e-12}, solve_ivp_method='RK45'
)

time, outputs, inputs = response.time, response.outputs, response.inputs

x = TB.r_12 * outputs['x']
y = TB.r_12 * outputs['y']
z = TB.r_12 * outputs['z']
xdot = TB.V_C * outputs['xdot']
ydot = TB.V_C * outputs['ydot']
zdot = TB.V_C * outputs['zdot']
t = TB.t_C * time

plotTraj.Plot_static_RF(
    x,y,z,t,TB.r_12,TB.x_L2,TB.Earth.x,
    save_path=OUTPUT_DIR / "jwst_rf_static.png"
)

plotTraj.Animation_RF(
    x,y,z,t,TB.r_12,TB.x_L2,TB.Earth.x,
    save_path=OUTPUT_DIR / "jwst_rf.mp4",
    fps=30, dpi=200
)

XYZ = TB.RotToFixed(np.array([x,y,z]).T, TB.Omega, t)
x_fixed = XYZ[:,0]
y_fixed = XYZ[:,1]
z_fixed = XYZ[:,2]

x_Earth = 0.97*TB.r_12*np.cos(TB.Omega*t)
y_Earth = 0.97*TB.r_12*np.sin(TB.Omega*t)
z_Earth = np.zeros_like(x_Earth)

plotTraj.Animation_FF(
    x_fixed,y_fixed,z_fixed,
    x_Earth,y_Earth,z_Earth,
    t,TB.r_12,
    save_path=OUTPUT_DIR / "jwst_ff.mp4",
    fps=30, dpi=200
)

jwstVisualizationFixed(
    x_fixed, y_fixed, z_fixed,
    x_Earth, y_Earth, z_Earth,
    TB.r_12, number_of_years,
    jwstModelPath="./assets/models/JWST/scene.gltf",
    cubeMapPath='./assets/cubemaps/space',
    save_movie=OUTPUT_DIR / "jwst_pv_fixed.mp4"
)

TB.JamesWebb.X0_LEO = np.array([
    1.0002052721269323E+0,
    -3.6778816842854946E-22,
    -5.9605625108140006E-26,
    -2.1985243974346765E-11,
    -1.2128984679501061E-1,
    4.4030232820030225E-23
])

timepts_leo = np.linspace(0, 1*TB.year, 1*timepts_year)

time_LEO, outputs_LEO, inputs_LEO = TB.JamesWebb.JWST_propagate(
    JWST_IO,
    TB.JamesWebb.X0_LEO,
    timepts=timepts_leo
)

x_LEO = TB.r_12*outputs_LEO['x']
y_LEO = TB.r_12*outputs_LEO['y']
z_LEO = TB.r_12*outputs_LEO['z']
t_LEO = TB.t_C*time_LEO

plotTraj.Plot_static_RF(
    x_LEO,y_LEO,z_LEO,t_LEO,TB.r_12,TB.x_L2,TB.Earth.x,
    save_path=OUTPUT_DIR / "leo_rf_static.png"
)

plotTraj.Animation_RF(
    x_LEO,y_LEO,z_LEO,t_LEO,TB.r_12,TB.x_L2,TB.Earth.x,
    save_path=OUTPUT_DIR / "leo_rf.mp4",
    fps=30, dpi=200
)

XYZ_LEO = TB.RotToFixed(np.array([x_LEO,y_LEO,z_LEO]).T, TB.Omega, t_LEO)
x_LEO_fixed = XYZ_LEO[:,0]
y_LEO_fixed = XYZ_LEO[:,1]
z_LEO_fixed = XYZ_LEO[:,2]

x_Earth_LEO = 0.97*TB.r_12*np.cos(TB.Omega*t_LEO)
y_Earth_LEO = 0.97*TB.r_12*np.sin(TB.Omega*t_LEO)
z_Earth_LEO = np.zeros_like(x_Earth_LEO)

plotTraj.Animation_FF(
    x_LEO_fixed,y_LEO_fixed,z_LEO_fixed,
    x_Earth_LEO,y_Earth_LEO,z_Earth_LEO,
    t_LEO,TB.r_12,
    save_path=OUTPUT_DIR / "leo_ff.mp4",
    fps=30, dpi=200
)

jwstVisualizationFixed(
    x_LEO_fixed, y_LEO_fixed, z_LEO_fixed,
    x_Earth_LEO, y_Earth_LEO, z_Earth_LEO,
    TB.r_12, 1,
    jwstModelPath="./assets/models/JWST/scene.gltf",
    cubeMapPath='./assets/cubemaps/space',
    save_movie=OUTPUT_DIR / "leo_pv_fixed.mp4",
    title="JWST LEO Orbit"
)