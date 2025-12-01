from typing import Tuple
from math import *
import numpy as np
import control as ct
import control.optimal as opt
from scipy.spatial.transform import Rotation as Rot
import PlotTraj
from Classes import *

# Instantiate Three body system and its compositions. 
TB = ThreeBodySystem()

# Set the initial states for JWST in Rotating frame and nondimensional unit
TB.JamesWebb.X0_Halo = np.array([1.0034949114516427,
                                -0.0064032911223132535,
                                0.007161610846661841, 
                                -0.007802784290524786, 
                                -0.0037330786370690866, 
                                -0.016701355598003984])

# LEO initial state vector
TB.JamesWebb.X0_LEO = np.array([1.0002052721269323E+0,
                                -3.6778816842854946E-22,
                                -5.9605625108140006E-26,	
                                -2.1985243974346765E-11,	
                                -1.2128984679501061E-1,	
                                4.4030232820030225E-23])


# Define the Input/Output System
JWST_IO_Earthcentered = ct.nlsys(TB.JamesWebb.JWST_update_nondim_Earthcentered, TB.JamesWebb.JWST_output,
                                states=['x','y','z','xdot','ydot','zdot'], name = 'JWST',
                                inputs=['ux','uy','uz'], outputs=['x','y','z','xdot','ydot','zdot'],
                                params={'pi_1':TB.pi_1,'pi_2':TB.pi_2})

# Define time span
LEO_years = 1
HALO_years = 10
timepts_year = 1000
timepts1 = np.linspace(0, LEO_years*TB.year, LEO_years*timepts_year)                             # Nondimensinal time points for LEO 
# timepts2 = np.linspace(timepts1[-1], timepts1[-1] + TO_years*TB.year, TO_years*10)     # Nondimensinal time points for Transfer Orbit
# timepts3 = np.linspace(timepts2[-1], timepts2[-1] + HALO_years*TB.year, HALO_years*timepts_year) # Nondimensinal time points for Halo Orbit
timepts3 = np.linspace(0, HALO_years*TB.year, HALO_years*timepts_year) # Nondimensinal time points for Halo Orbit

# Simulation of a LEO 
time_LEO, output_LEO, input_LEO = TB.JamesWebb.JWST_propagate(JWST_IO_Earthcentered,TB.Earth_centered(TB.JamesWebb.X0_LEO),timepts=timepts1)
output_LEO = TB.Earth_centered_inverse(output_LEO)
x_LEO, y_LEO, z_LEO, xdot_LEO, ydot_LEO, zdot_LEO, t_LEO = TB.Dimensionalize(output_LEO,time_LEO)

# Solve Optimal Control Problem for transfer orbit
opt_data = load_optimization_result("Optimization_Result")
T_guess = opt_data['T_opt']
X_guess = opt_data['Xs']
u_guess = opt_data['Us']

if False:
  X0_TO = TB.Earth_centered(output_LEO[:,-1])
  Xf_TO = TB.Earth_centered(TB.JamesWebb.X0_Halo)
  T_opt, X_opt, u_opt, J_opt = TB.JamesWebb.optimal_transfer_orbit(X0_TO,Xf_TO,X_guess,uf=[0,0,0],u_guess=u_guess,
                                                            N=600, params={'pi_1':TB.pi_1,'pi_2':TB.pi_2},
                                                            Q=np.diag([10.,10.,10.,1.,1.,1.]),
                                                            R=np.diag([2.,2.,2.]),
                                                            Qf=np.diag([1e2,1e2,1e2,1e2,1e2,1e2]),
                                                            Rf=np.diag([1.,1.,1.]),
                                                            beta = 10.0,
                                                            opt_max_iter = 10000,
                                                            save_result=True)
else:
  T_opt = T_guess
  X_opt = X_guess
  u_opt = u_guess

x_TO, y_TO, z_TO, xdot_TO, ydot_TO, zdot_TO, t_TO = TB.Dimensionalize(TB.Earth_centered_inverse(X_opt))

# Simulation of Halo Orbit
time_Halo, output_Halo, input_Halo = TB.JamesWebb.JWST_propagate(JWST_IO_Earthcentered,TB.Earth_centered(TB.JamesWebb.X0_Halo),timepts=timepts3)
# Retrive JWST trajectories in rotation frame and dimentionalize them
output_Halo = TB.Earth_centered_inverse(output_Halo)
x_Halo, y_Halo, z_Halo, xdot_Halo, ydot_Halo, zdot_Halo, t_Halo = TB.Dimensionalize(output_Halo,time_Halo)

# Build a proper time vector for the TO so it matches the states
t_TO = np.linspace(
    t_LEO[-1],                   # start right after LEO ends
    t_LEO[-1] + T_opt*TB.year,   # end after T_opt years (dimensional)
    len(x_TO)
)

# Merge the Orbits
x = np.concatenate((x_LEO,x_TO,x_Halo))
y = np.concatenate((y_LEO,y_TO,y_Halo))
z = np.concatenate((z_LEO,z_TO,z_Halo))
t = np.concatenate((t_LEO,t_TO,t_Halo))

# Plot the JWST Trajectory
# PlotTraj.Plot_static_RF(x,y,z,TB.r_12,TB.x_L2,TB.Earth.x)
PlotTraj.Animation_RF(x,y,z,t,TB.r_12,TB.x_L2,TB.Earth.x)

# Transfer to Fixed Frame
# XYZ = TB.RotToFixed(np.array([x,y,z]).T, TB.Omega, t)
# x_fixed = XYZ[:,0]
# y_fixed = XYZ[:,1]
# z_fixed = XYZ[:,2]
# x_Earth = 0.97*TB.r_12*np.cos(TB.Omega*t)
# y_Earth = 0.97*TB.r_12*np.sin(TB.Omega*t)
# z_Earth = np.zeros_like(x_Earth)

# PlotTraj.Animation_FF(x_fixed,y_fixed,z_fixed,x_Earth,y_Earth,z_Earth,t,TB.r_12)
