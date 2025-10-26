from typing import Tuple
from math import *
import numpy as np
import control as ct
from scipy.spatial.transform import Rotation as Rot
import PlotTraj
from Classes import *

# Instantiate Three body system and its compositions. 
TB = ThreeBodySystem()

# Set the initial states for JWST in Rotating frame and nondimensional unit
# Unstable Halo Orbit
TB.JamesWebb.X0 = np.array([1.0062010416592476,             # x
                            -9.8074369820712423e-16,       # xdot
                            -2.1523078518259630e-23,       # y
                            -1.3253477924660511e-2,        # ydot
                            1.2380311201349303e-2,        # z
                            -1.0956603459061133e-14])      # zdot

# Stable Halo Orbit
# TB.JamesWebb.X0 = np.array([1.0065075924336142E+0,             # x
#                             2.6228553664868397E-16,       # xdot
#                             -2.1016745594119651E-23,       # y
#                             -1.3729810632559413E-2,        # ydot
#                             1.2329298254423008E-2,        # z
#                             -2.4057058092832893E-15])      # zdot

# Define parameters used in JWST dynamics
JWST_params = {'pi_1':TB.pi_1,'pi_2':TB.pi_2}

# Define the Input/Output System
JWST_IO = ct.nlsys(TB.JamesWebb.JWST_update_nondim, TB.JamesWebb.JWST_output,
                    states=['x','xdot','y','ydot','z','zdot'], name = 'JWST',
                    inputs=['ux','uy','uz'], outputs=['x','xdot','y','ydot','z','zdot'],
                    params=JWST_params)

# Simulation
number_of_years = 20
timepts_year = 1000
timepts = np.linspace(0,number_of_years*TB.year,number_of_years*timepts_year)       # Nondimensinal time points
u = [np.zeros_like(timepts),np.zeros_like(timepts),np.zeros_like(timepts)]
response = ct.input_output_response(JWST_IO, timepts=timepts, inputs=u, initial_state=TB.JamesWebb.X0,
                                    solve_ivp_kwargs={'rtol':1e-9,'atol':1e-12}, solve_ivp_method='RK45')
time, outputs, inputs = response.time, response.outputs, response.inputs

# Retrive JWST trajectories in rotation frame and dimentionalize them
x = TB.r_12 * outputs['x']
y = TB.r_12 * outputs['y']
z = TB.r_12 * outputs['z']
xdot = TB.V_C * outputs['xdot']
ydot = TB.V_C * outputs['ydot']
zdot = TB.V_C * outputs['zdot']
t = TB.t_C * time

# Plot the JWST Trajectory
PlotTraj.Plot_static_RF(x,y,z,t,TB.r_12,TB.x_L2,TB.Earth.x)
PlotTraj.Animation_RF(x,y,z,t,TB.r_12,TB.x_L2,TB.Earth.x)

# Transfer to Fixed Frame
XYZ = TB.RotToFixed(np.array([x,y,z]).T, TB.Omega, t)
x_fixed = XYZ[:,0]
y_fixed = XYZ[:,1]
z_fixed = XYZ[:,2]
x_Earth = 0.97*TB.r_12*np.cos(TB.Omega*t)
y_Earth = 0.97*TB.r_12*np.sin(TB.Omega*t)
z_Earth = np.zeros_like(x_Earth)

PlotTraj.Animation_FF(x_fixed,y_fixed,z_fixed,x_Earth,y_Earth,z_Earth,t,TB.r_12)
