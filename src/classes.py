# from typing import Tuple
# from math import *
# import numpy as np
# import control as ct
# from scipy.spatial.transform import Rotation as Rot


# # Defining classes
# class Celestial:
#   def __init__(self,
#                mass:float,                            # [kg]
#                pos:Tuple[float,float,float],          # [km]
#                ang_vel:float,                         # [rad/s]
#                radius:float):                         # [km]
#     self.mass = mass
#     self.x,self.y,self.z = pos
#     self.w = ang_vel
#     self.X0 = None
#     self.radius = radius

# class JWST:
#   def __init__(self,
#                mass:float,
#                pos:Tuple[float,float,float]):
#     self.mass = mass
#     self.x,self.y,self.z = pos
#     self.X0 = None

#   @staticmethod
#   # dynamic of JWST in rotating frame
#   def JWST_update_nondim(t,X,u,params):
#     pi_1 = params.get('pi_1')
#     pi_2 = params.get('pi_2')

#     sigma = np.sqrt(np.sum(np.square([X[0] + pi_2, X[2] , X[4]])))
#     psi = np.sqrt(np.sum(np.square([X[0] - pi_1, X[2] , X[4]])))

#     return np.array([
#      X[1],
#      -pi_1/sigma**3*(X[0] + pi_2) - pi_2/psi**3*(X[0] - pi_1) + 2*X[3] + X[0] + u[0],
#      X[3],
#      -pi_1/sigma**3*X[2] - pi_2/psi**3*X[2] - 2*X[1] + X[2] + u[1],
#      X[5],
#      -pi_1/sigma**3*X[4] - pi_2/psi**3*X[4] + u[2]
#     ])

#   @staticmethod
#   # Output of the system
#   def JWST_output(t,X,u,params):
#     return X
  
# class ThreeBodySystem:                     
#   def __init__(self,
#                m_Sun:float = 1.98847e30,
#                m_Earth:float = 5.9722e24,
#                m_JWST:float = 6500,
#                Period:float = 31556926,           # Seconds per year [rad/s] 
#                r_12:float = 1.49597871e8):          # Earth-Sun mean distance [km]
#     self.G = 6.674e-20                            # Gravitational constant [km^3/kg/s^2]
#     self.m_Sun = m_Sun
#     self.m_Earth = m_Earth                        # Characteristic Mass in [kg]
#     self.M = m_Earth + m_Sun
#     self.mu = self.G*self.M
#     self.pi_2 = 3.0542e-6                         # m_Sun / (m_Earth + m_Sun)
#     self.pi_1 = 1 - self.pi_2                     # m_Earth / (m_Earth + m_Sun)
#     self.Period = Period
#     self.Omega = 2 * pi / Period
#     self.r_12 = r_12                              # Characteristic Length in [km]
#     self.t_C =  5022635                           # Characteristic Time in [s]
#     self.V_C = r_12/self.t_C                      # Characteristic Velocity in [km/s]
#     self.year = Period / self.t_C
#     self.w_Earth = 7.292115e-5                     # Earth's Rotation speed in rad/s in  ECEF
#     self.n = 1.991e-7                              # rad/s mean motion of the Sun-Earth line in Ecliptic
#     self.x_L2 = 1.01009044

#     # Instantiate 3 bodies as components
#     self.Sun = Celestial(mass = m_Sun, pos = (-self.pi_2 ,0,0), ang_vel = 0, radius=695700)
#     self.Earth = Celestial(mass = m_Earth, pos = (self.pi_1 ,0,0), ang_vel = self.w_Earth, radius=6378)
#     self.JamesWebb = JWST(mass = m_JWST, pos = (self.Earth.x + 2000000/r_12,0,0))

#   @staticmethod
#   def RotToFixed(r,W,t):
#     Rz = Rot.from_euler('z',W * t,degrees=False)
#     return Rz.apply(r)


from typing import Tuple
from math import *
import numpy as np
import control as ct
import control.optimal as opt
from scipy.spatial.transform import Rotation as Rot

class Celestial:
    def __init__(self,
                 mass:float,
                 pos:Tuple[float,float,float],
                 ang_vel:float,
                 radius:float):
        self.mass = mass
        self.x,self.y,self.z = pos
        self.w = ang_vel
        self.X0 = None
        self.radius = radius

class JWST:
    def __init__(self,
                 mass:float,
                 pos:Tuple[float,float,float]):
        self.mass = mass
        self.x,self.y,self.z = pos
        self.X0 = None
        self.X0_Halo = None
        self.X0_LEO = None

    @staticmethod
    def JWST_update_nondim(t,X,u,params):
        pi_1 = params.get('pi_1')
        pi_2 = params.get('pi_2')

        sigma = np.sqrt(np.sum(np.square([X[0] + pi_2, X[1] , X[2]])))
        psi = np.sqrt(np.sum(np.square([X[0] - pi_1, X[1] , X[2]])))

        return np.array([
            X[3],
            X[4],
            X[5],
            -pi_1/sigma**3*(X[0] + pi_2) - pi_2/psi**3*(X[0] - pi_1) + 2*X[4] + X[0] + u[0],
            -pi_1/sigma**3*X[1] - pi_2/psi**3*X[1] - 2*X[3] + X[1] + u[1],
            -pi_1/sigma**3*X[2] - pi_2/psi**3*X[2] + u[2]
        ])

    @staticmethod
    def JWST_output(t,X,u,params):
        return X

    @staticmethod
    def JWST_propagate(JWST_IO:ct.NonlinearIOSystem,
                       X0,
                       timepts,
                       inputs=None):
        if inputs is None:
            inputs = [np.zeros_like(timepts),np.zeros_like(timepts),np.zeros_like(timepts)]

        response = ct.input_output_response(
            JWST_IO,
            timepts=timepts,
            inputs=inputs,
            initial_state=X0,
            solve_ivp_kwargs={'rtol':1e-9,'atol':1e-12},
            solve_ivp_method='RK45'
        )
        return response.time, response.outputs, response.inputs

    @staticmethod
    def optimal_transfer_orbit(JWST_IO:ct.NonlinearIOSystem,X0,Xf,u0,uf,
                               timepts,params,cost='Quadratic',
                               Q=None,R=None,P=None):
        a_C = params.get('a_C')
        M = params.get('M')
        if cost == 'Quadratic':
            traj_cost = opt.quadratic_cost(JWST_IO,Q,R,x0 = Xf, u0 = uf)
            term_cost = opt.quadratic_cost(JWST_IO,P,0,x0 = Xf)
            constraints = [opt.input_range_constraint(JWST_IO,
                        -100/M/a_C*np.array([1,1,1]),
                         100/M/a_C*np.array([1,1,1]))]
        else:
            print('Cost function is undefined.')

        result = opt.solve_optimal_trajectory(
            JWST_IO,
            timepts,
            X0,
            traj_cost,
            constraints,
            terminal_cost=term_cost,
            initial_guess=u0,
            log=True,
            trajectory_method='collocation'
        )
        return result

class ThreeBodySystem:                     
    def __init__(self,
                 m_Sun:float = 1.98847e30,
                 m_Earth:float = 5.9722e24,
                 m_JWST:float = 6500,
                 Period:float = 31556926,
                 r_12:float = 1.49597871e8):
        self.G = 6.674e-20
        self.m_Sun = m_Sun
        self.m_Earth = m_Earth
        self.M = m_Earth + m_Sun
        self.mu = self.G*self.M
        self.pi_2 = 3.0542e-6
        self.pi_1 = 1 - self.pi_2
        self.Period = Period
        self.Omega = 2 * pi / Period
        self.r_12 = r_12
        self.t_C =  5022635
        self.V_C = r_12/self.t_C
        self.a_C = r_12/self.t_C**2
        self.year = Period / self.t_C
        self.w_Earth = 7.292115e-5
        self.n = 1.991e-7
        self.x_L2 = 1.01009044

        self.Sun = Celestial(mass = m_Sun, pos = (-self.pi_2 ,0,0), ang_vel = 0, radius=695700)
        self.Earth = Celestial(mass = m_Earth, pos = (self.pi_1 ,0,0), ang_vel = self.w_Earth, radius=6378)
        self.JamesWebb = JWST(mass = m_JWST, pos = (self.Earth.x + 2000000/r_12,0,0))

    @staticmethod
    def RotToFixed(r,W,t):
        Rz = Rot.from_euler('z',W * t,degrees=False)
        return Rz.apply(r)

    @staticmethod
    def FixedToRot(r, W, t):
        Rz = Rot.from_euler('z', -W * t, degrees=False)
        return Rz.apply(r)

    def Dimensionalize(self,outputs,time):
        x = self.r_12 * outputs['x']
        y = self.r_12 * outputs['y']
        z = self.r_12 * outputs['z']
        xdot = self.V_C * outputs['xdot'] if 'xdot' in outputs else None
        ydot = self.V_C * outputs['ydot'] if 'ydot' in outputs else None
        zdot = self.V_C * outputs['zdot'] if 'zdot' in outputs else None
        t = self.t_C * time
        return x, y, z, xdot, ydot, zdot, t
