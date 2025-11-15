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
# Unstable Halo Orbit initial state vector
TB.JamesWebb.X0_Halo = np.array([1.0062010416592476,            # x
                            -2.1523078518259630e-23,       # y
                            1.2380311201349303e-2,         # z
                            -9.8074369820712423e-16,       # xdot
                            -1.3253477924660511e-2,        # ydot
                            -1.0956603459061133e-14])      # zdot

# LEO initial state vector
TB.JamesWebb.X0_LEO = np.array([1.0002052721269323E+0,
                                -3.6778816842854946E-22,
                                -5.9605625108140006E-26,	
                                -2.1985243974346765E-11,	
                                -1.2128984679501061E-1,	
                                4.4030232820030225E-23])
# TB.JamesWebb.X0_LEO = np.array([1.0002764020258976E+0,	-5.1724847280753026E-10,	8.1826919794728482E-28,	1.9301945290673018E-7,	1.0426358207786048E-1,	-8.7879486946121682E-26])

# Stable Halo Orbit
# TB.JamesWebb.X0 = np.array([1.0065075924336142E+0,             # x
#                             2.6228553664868397E-16,       # xdot
#                             -2.1016745594119651E-23,       # y
#                             -1.3729810632559413E-2,        # ydot
#                             1.2329298254423008E-2,        # z
#                             -2.4057058092832893E-15])      # zdot

# Butterfly Orbit
# TB.JamesWebb.X0 = np.array([9.9478400099057562E-1,             # x
#                             -3.7820695702053078E-16,       # xdot
#                             6.9675881786374951E-21,       # y
#                             -2.7018843432500038E-3,        # ydot
#                             9.3307317319370944E-3,        # z
#                             -8.9428405556110216E-15])      # zdot


# Define the Input/Output System
JWST_IO = ct.nlsys(TB.JamesWebb.JWST_update_nondim, TB.JamesWebb.JWST_output,
                    states=['x','y','z','xdot','ydot','zdot'], name = 'JWST',
                    inputs=['ux','uy','uz'], outputs=['x','y','z','xdot','ydot','zdot'],
                    params={'pi_1':TB.pi_1,'pi_2':TB.pi_2})


# Define time span
LEO_years = 1
TO_years = 2
HALO_years = 5
timepts_year = 1000
timepts1 = np.linspace(0, LEO_years*TB.year, LEO_years*timepts_year)                             # Nondimensinal time points for LEO 
timepts2 = np.linspace(timepts1[-1], timepts1[-1] + TO_years*TB.year, TO_years*timepts_year)     # Nondimensinal time points for Transfer Orbit
timepts3 = np.linspace(timepts2[-1], timepts2[-1] + HALO_years*TB.year, HALO_years*timepts_year) # Nondimensinal time points for Halo Orbit

# Simulation of a LEO 
time_LEO, output_LEO, input_LEO = TB.JamesWebb.JWST_propagate(JWST_IO,TB.JamesWebb.X0_LEO,timepts=timepts1)
x_LEO, y_LEO, z_LEO, xdot_LEO, ydot_LEO, zdot_LEO, t_LEO = TB.Dimensionalize(output_LEO,time_LEO)

# Solve optimal control problem for the transfer orbit
X0_trans = output_LEO[:,-1]
Xf_trans = TB.JamesWebb.X0_Halo
u0_trans = np.array([0., 0. ,0.])
uf_trans = np.array([0., 0. ,0.])



# Simulation of Halo Orbit
time_Halo, output_Halo, input_Halo = TB.JamesWebb.JWST_propagate(JWST_IO,TB.JamesWebb.X0_Halo,timepts=timepts3)

# Retrive JWST trajectories in rotation frame and dimentionalize them
x_Halo, y_Halo, z_Halo, xdot_Halo, ydot_Halo, zdot_Halo, t_Halo = TB.Dimensionalize(output_Halo,time_Halo)


# Merge the Orbits
x = np.concatenate((x_LEO,x_Halo))
y = np.concatenate((y_LEO,y_Halo))
z = np.concatenate((z_LEO,z_Halo))

# Plot the JWST Trajectory
PlotTraj.Plot_static_RF(x,y,z,TB.r_12,TB.x_L2,TB.Earth.x)
# PlotTraj.Animation_RF(x,y,z,t,TB.r_12,TB.x_L2,TB.Earth.x)

# Transfer to Fixed Frame
# XYZ = TB.RotToFixed(np.array([x,y,z]).T, TB.Omega, t)
# x_fixed = XYZ[:,0]
# y_fixed = XYZ[:,1]
# z_fixed = XYZ[:,2]
# x_Earth = 0.97*TB.r_12*np.cos(TB.Omega*t)
# y_Earth = 0.97*TB.r_12*np.sin(TB.Omega*t)
# z_Earth = np.zeros_like(x_Earth)

# PlotTraj.Animation_FF(x_fixed,y_fixed,z_fixed,x_Earth,y_Earth,z_Earth,t,TB.r_12)
