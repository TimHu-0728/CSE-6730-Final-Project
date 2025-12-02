from typing import Tuple
from math import *
import numpy as np
import control as ct
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d
import casadi as ca
import time
import pathlib

# Defining classes
class Celestial:
  def __init__(self,
               mass:float,                            # [kg]
               pos:Tuple[float,float,float],          # [km]
               ang_vel:float,                         # [rad/s]
               radius:float):                         # [km]
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
    self.X0_Halo = None
    self.X0_LEO = None

  @staticmethod
  # dynamic of JWST in rotating frame
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
     -pi_1/sigma**3*X[2] - pi_2/psi**3*X[2] + u[2]])
  
  @staticmethod
  def JWST_update_nondim_Earthcentered(t,X,u,params):
    pi_1 = params.get('pi_1')
    pi_2 = params.get('pi_2') 

    sigma = np.linalg.norm([X[0]+1,X[1],X[2]])
    psi = np.linalg.norm([X[0],X[1],X[2]])

    return np.array([
      X[3],
      X[4],
      X[5],
     -pi_1/sigma**3*(X[0] + 1) - pi_2/psi**3*X[0] + 2*X[4] + X[0] + pi_1 + u[0],
     -pi_1/sigma**3*X[1] - pi_2/psi**3*X[1] - 2*X[3] + X[1] + u[1],
     -pi_1/sigma**3*X[2] - pi_2/psi**3*X[2] + u[2]])

  @staticmethod
  # Output of the system
  def JWST_output(t,X,u,params):
    return X
  
  @staticmethod
  def JWST_propagate(JWST_IO:ct.NonlinearIOSystem,
                     X0,
                     timepts,
                     u=None):
      if u is None:
        u = [np.zeros_like(timepts),np.zeros_like(timepts),np.zeros_like(timepts)]

      response = ct.input_output_response(JWST_IO, timepts=timepts, inputs=u, initial_state=X0,
                                    solve_ivp_kwargs={'rtol':1e-9,'atol':1e-12},
                                    solve_ivp_method='RK45')
      return response.time, response.outputs, response.inputs
  
  
  # Solving trajectory optimization proble using CasADi (direct multiple shooting method)
  def optimal_transfer_orbit(self,X0,Xf,X_guess,u0=None,uf=None,u_guess=None,
                             N=200,params=None,
                             Q=np.diag([1.,1.,1.,0.1,0.1,0.1]),
                             R=np.diag([1.,1.,1.]),
                             Qf=np.diag([1e4,1e4,1e4,1e2,1e2,1e2]),
                             Rf=np.diag([1e2,1e2,1e2]),
                             beta = 1.,
                             opt_max_iter = 1000,
                             save_result=True):
    
    pi_1 = params.get('pi_1')
    pi_2 = params.get('pi_2') 

    # Interpolate guessed trajectory to have N timepts
    fX = interp1d(np.linspace(0.0,1.0,np.size(X_guess,axis=1)), X_guess, kind="linear", axis=1)
    X_guess = fX(np.linspace(0.0,1.0,N))
    fu = interp1d(np.linspace(0.0,1.0,np.size(u_guess,axis=1)), u_guess, kind="linear", axis=1)
    u_guess = fu(np.linspace(0.0,1.0,N))

    # Define symbolic variable used in CasADi 
    T = ca.MX.sym('T')
    X = ca.MX.sym('X',6)
    u = ca.MX.sym('u',3)

    # Define symbolic Earth-centered CR3BP dynamic used in multiple shooting method
    sigma = ca.norm_2(ca.vertcat(X[0]+1,X[1],X[2]))
    psi = ca.norm_2(ca.vertcat(X[0],X[1],X[2]))
    Xdot = ca.vertcat(
        X[3],
        X[4],
        X[5],
        -pi_1/sigma**3*(X[0] + 1) - pi_2/psi**3*X[0] + 2*X[4] + X[0] + pi_1 + u[0],
        -pi_1/sigma**3*X[1] - pi_2/psi**3*X[1] - 2*X[3] + X[1] + u[1],
        -pi_1/sigma**3*X[2] - pi_2/psi**3*X[2] + u[2]
    )

    Xf = ca.DM(Xf)
    uf = ca.DM(uf)
    Q = ca.DM(Q)     # running state error penalty weight
    R = ca.DM(R)     # running input penalty weight
    Qf = ca.DM(Qf)   # final state error penalty weight
    beta = ca.DM(beta)

    # Running cost: quadratic state error, quadratic input, and time penalty 
    L = ((X - Xf).T @ Q @ (X - Xf) + u.T @ R @ u)[0,0] + beta*(T/N)

    # Define cvodes integrator from Sundials
    dae = {'x':X,'p':ca.vertcat(u,T),'ode':(T/N)*Xdot,'quad':(T/N)*L}
    # dae = {'x':X,'p':ca.vertcat(u,T),'ode':Xdot,'quad':(T/N)*L}
    opts_int = {'tf': 1.0, 'abstol': 1e-8, 'reltol': 1e-8}
    cr3bp = ca.integrator("cr3bp",'cvodes',dae,opts_int)

    # Form the NLP  
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    w += [T]
    lbw += [0.1]
    ubw += [20.0]
    w0 += [5.0]

    # Initial state constraint
    Xk = ca.MX.sym('X0',6)
    w += [Xk]
    lbw += list(X0)
    ubw += list(X0)
    w0 += list(X0)

    
    U_syms = []
    for k in range(N):
        # Define New input variables and constraint
        Uk = ca.MX.sym('U_' + str(k),3)
        w += [Uk]
        lbw += [-1.]*3
        ubw += [1.]*3
        if u_guess is None:
          w0 += [0.0,0.0,0.0]
        else:
          w0 += list(u_guess[:,k]) 
        U_syms.append(Uk)

        Fk = cr3bp(x0=Xk, p=ca.vertcat(Uk, T))
        Xk_end = Fk['xf']
        J = J + Fk['qf']

        # Define New states variables and constraint
        Xk = ca.MX.sym('X_'+str(k+1),6)
        w += [Xk]
        lbw += [-ca.inf]*6
        ubw += [ca.inf]*6
        w0 += list(X_guess[:,k])

        g += [Xk_end - Xk]
        lbg += [0]*6
        ubg += [0]*6

    # Final state constraint
    g += [Xk - Xf]
    lbg += [0]*6
    ubg += [0]*6

    # Final input constraint
    g += [Uk - uf]
    lbg += [0]*3 
    ubg += [0]*3

    # Final state penalty 
    # J = J + ((Xk - Xf).T @ Qf @ (Xk - Xf))[0,0] + ((Uk - uf).T @ Rf @ (Uk - uf))[0,0]
    # best_solution = None
    # best_objective = float('inf')

    prob = {'f':J, 'x':ca.vertcat(*w),'g':ca.vertcat(*g)}
    ipopt_opts = {
        'tol':                     1e-6,
        'dual_inf_tol':            1e-6,
        'constr_viol_tol':         1e-6,
        'compl_inf_tol':           1e-6,
        'acceptable_tol':          1e-4,
        'acceptable_constr_viol_tol': 1e-4,

        'mu_strategy':             'adaptive',
        'mu_init':                 1e-1,

        'linear_solver':           'mumps',          # if available

        'hessian_approximation':   'limited-memory', # try this if exact Hessian is too slow
        'limited_memory_max_history': 50,

        'nlp_scaling_method':      'gradient-based',
        'print_level':             5,                # 0–12; 5 is moderate
        'max_iter':                opt_max_iter
    }
    opts = {'ipopt':ipopt_opts}
    solver = ca.nlpsol('solver','ipopt',prob,opts) 
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    # Use the best solution found during iterations
    w_opt = sol['x'].full().flatten()
    T_opt = w_opt[0]
    J_opt = sol['f'].full().flatten()
    T_opt, Xs, Us = self.Extract_optimal(w_opt,N)

    if save_result:
      save_optimization_result(T_opt, Xs, Us, J_opt)

    return T_opt, Xs, Us, J_opt
  
  @staticmethod
  def Extract_optimal(w_opt,N):
     # Extract time Horizon T, state trajectories Xs and input Us 
     T_opt = w_opt[0]
     Xs = []
     Us = []
     offset = 1  # Skip T
     for k in range(N+1):
          Xs.append(w_opt[offset + k*9 : offset + k*9 + 6])
          if k < N:
              Us.append(w_opt[offset + k*9 + 6 : offset + k*9 + 9])
     Xs = np.array(Xs).T
     Us = np.array(Us).T
     return T_opt, Xs, Us

  
class ThreeBodySystem:                     
  def __init__(self,
               m_Sun:float = 1.98847e30,
               m_Earth:float = 5.9722e24,
               m_JWST:float = 6500,
               Period:float = 31556926,           # Seconds per year [rad/s] 
               r_12:float = 1.49597871e8):          # Earth-Sun mean distance [km]
    self.G = 6.674e-20                            # Gravitational constant [km^3/kg/s^2]
    self.m_Sun = m_Sun
    self.m_Earth = m_Earth                        
    self.M = m_Earth + m_Sun                      # Characteristic Mass in [kg]
    self.mu = self.G*self.M
    self.pi_2 = 3.0542e-6                         # m_Sun / (m_Earth + m_Sun)
    self.pi_1 = 1 - self.pi_2                     # m_Earth / (m_Earth + m_Sun)
    self.Period = Period
    self.Omega = 2 * pi / Period
    self.r_12 = r_12                              # Characteristic Length in [km]
    self.t_C =  5022635                           # Characteristic Time in [s]
    self.V_C = r_12/self.t_C                      # Characteristic Velocity in [km/s]
    self.a_C = r_12/self.t_C**2                   # Characteristic Acceleration in [km/s^2]
    self.year = Period / self.t_C
    self.w_Earth = 7.292115e-5                     # Earth's Rotation speed in rad/s in  ECEF
    self.n = 1.991e-7                              # rad/s mean motion of the Sun-Earth line in Ecliptic
    self.x_L2 = 1.01009044

    # Instantiate 3 bodies as components
    self.Sun = Celestial(mass = m_Sun, pos = (-self.pi_2 ,0,0), ang_vel = 0, radius=695700)
    self.Earth = Celestial(mass = m_Earth, pos = (self.pi_1 ,0,0), ang_vel = self.w_Earth, radius=6378)
    self.JamesWebb = JWST(mass = m_JWST, pos = (self.Earth.x + 2000000/r_12,0,0))

  @staticmethod
  def RotToFixed(r,W,t):
    Rz = Rot.from_euler('z',W * t,degrees=False)
    return Rz.apply(r)
  
  # @staticmethod
  def Earth_centered(self,X):
    if X.ndim == 1:
      X = X[:, None]      
    return X - np.array([self.pi_1,0,0,0,0,0]).reshape(6,1)

  def Earth_centered_inverse(self,X):
    if X.ndim == 1:
      X = X[:, None]     
    return X + np.array([self.pi_1,0,0,0,0,0]).reshape(6,1)


  def Dimensionalize(self,X,time=None):
    x = self.r_12 * X[0,:]
    y = self.r_12 * X[1,:]
    z = self.r_12 * X[2,:]
    xdot = self.V_C * X[3,:]
    ydot = self.V_C * X[4,:]
    zdot = self.V_C * X[5,:]
    if time is not None:
      t = self.t_C * time
    else:
      t = None
    return x, y, z, xdot, ydot, zdot, t
  
  def States_at_k(self,X,k=None):
    x = X['x'][k]
    y = X['y'][k]
    z = X['z'][k]
    xdot = X['xdot'][k]
    ydot = X['ydot'][k]
    zdot = X['zdot'][k]    
    return x, y, z, xdot, ydot, zdot
  


def save_optimization_result(T_opt, Xs, Us, J_opt, folder='Optimization_Result', name=None):
    """
    save optimization result to folder/name (.npz)。
    - T_opt: scalar
    - Xs: numpy array (N+1, 6)
    - Us: numpy array (N, 3)
    return saved file path（string）
    """
    folder_path = pathlib.Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = time.strftime('result_%Y%m%d_%H%M%S.npz')
    out_path = folder_path / name
    # 确保 numpy 数组类型
    Tv = float(np.array(T_opt).item()) if np.ndim(T_opt) == 0 or np.shape(T_opt) == () else float(T_opt)
    Jv = float(np.array(J_opt).item()) if np.ndim(J_opt) == 0 or np.shape(J_opt) == () else float(J_opt)
    Xs_arr = np.array(Xs)
    Us_arr = np.array(Us)
    np.savez_compressed(out_path, T_opt=Tv, Xs=Xs_arr, Us=Us_arr, J_opt=Jv)
    print(f"Saved optimization result to: {out_path}")
    return str(out_path)

def load_optimization_result(path_or_folder):
    """
    load_optimization_result
    return Dict: {'T_opt': float, 'Xs': np.ndarray, 'Us': np.ndarray}
    """
    p = pathlib.Path(path_or_folder)
    if p.is_dir():
        files = sorted(p.glob('*.npz'))
        if not files:
            raise FileNotFoundError(f"No .npz files found in folder: {p}")
        p = files[-1]  # latest
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    data = np.load(p, allow_pickle=True)
    return {'T_opt': float(data['T_opt'].item() if hasattr(data['T_opt'], 'item') else data['T_opt']),
            # 'J_opt': float(data['J_opt'].item() if hasattr(data['J_opt'], 'item') else data['J_opt']),
            'Xs': data['Xs'],
            'Us': data['Us']}
