from typing import Tuple
from math import *
import numpy as np
import control as ct
from scipy.spatial.transform import Rotation as Rot


# Defining classes
class Celestial:
  def __init__(self,
               mass:float,                            # [kg]
               pos:Tuple[float,float,float],          # [km]
               ang_vel:float):     # [rad/s]
    self.mass = mass
    self.x,self.y,self.z = pos
    self.w = ang_vel
    self.X0 = None

class JWST:
  def __init__(self,
               mass:float,
               pos:Tuple[float,float,float]):
    self.mass = mass
    self.x,self.y,self.z = pos
    self.X0 = None

  @staticmethod
  # dynamic of JWST in rotating frame
  def JWST_update_nondim(t,X,u,params):
    pi_1 = params.get('pi_1')
    pi_2 = params.get('pi_2')

    sigma = np.sqrt(np.sum(np.square([X[0] + pi_2, X[2] , X[4]])))
    psi = np.sqrt(np.sum(np.square([X[0] - pi_1, X[2] , X[4]])))

    return np.array([
     X[1],
     -pi_1/sigma**3*(X[0] + pi_2) - pi_2/psi**3*(X[0] - pi_1) + 2*X[3] + X[0] + u[0],
     X[3],
     -pi_1/sigma**3*X[2] - pi_2/psi**3*X[2] - 2*X[1] + X[2] + u[1],
     X[5],
     -pi_1/sigma**3*X[4] - pi_2/psi**3*X[4] + u[2]
    ])

  @staticmethod
  # Output of the system
  def JWST_output(t,X,u,params):
    return X
  
class ThreeBodySystem:                     
  def __init__(self,
               m_Sun:float = 1.98847e30,
               m_Earth:float = 5.9722e24,
               m_JWST:float = 6500,
               Period:float = 31556926,   # Angular Velocity in rad/s of the rotating frame [rad/s] 
               r_12:float = 1.495978707e8):          # Earth-Sun mean distance [km]
    self.G = 6.674e-20                            # Gravitational constant [km^3/kg/s^2]
    self.m_Sun = m_Sun
    self.m_Earth = m_Earth                        # Characteristic Mass in [kg]
    self.M = m_Earth + m_Sun
    self.mu = self.G*self.M
    self.pi_1 = m_Sun / (m_Earth + m_Sun)
    self.pi_2 = m_Earth / (m_Earth + m_Sun)
    self.Period = Period
    self.Omega = 2 * pi / Period
    self.r_12 = r_12                              # Characteristic Length in [km]
    self.t_C =  5022635                           # Characteristic Time in [s]
    self.V_C = r_12/self.t_C                      # Characteristic Velocity in [km/s]
    self.year = Period / self.t_C
    self.w_Earth = 7.292115e-5                     # Earth's Rotation speed in rad/s in  ECEF
    self.n = 1.991e-7                              # rad/s mean motion of the Sun-Earth line in Ecliptic
    self.x_L2 = 1.01009044
    
    # Instantiate 3 bodies as components
    self.Sun = Celestial(mass = m_Sun, pos = (-self.pi_2 ,0,0), ang_vel = 0)
    self.Earth = Celestial(mass = m_Earth, pos = (self.pi_1 ,0,0), ang_vel = self.w_Earth)
    self.JamesWebb = JWST(mass = m_JWST, pos = (self.Earth.x + 2000000/r_12,0,0))

  @staticmethod
  def RotToFixed(r,W,t):
    Rz = Rot.from_euler('z',W * t,degrees=False)
    return Rz.apply(r)