import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Animate trajectory in Rotation frame around L2
def Animation_RF(x,y,z,t,r_12,x_L2,x_Earth):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  ax.axis('equal')
  Traj = ax.plot([], [], [], 'r-', label='Halo Orbit',lw=1)[0]
  Head = ax.scatter([],[],[], c='tab:brown',marker='o',s=1,label='JWST')
  ax.scatter(x_L2*r_12,0,0,c='tab:purple',marker='*',label='L_2',s=80)
  ax.scatter(x_Earth*r_12,0,0,c='tab:blue',marker='o',label='Earth',s=1)
  ax.set_title('JWST Trajectory in Rotating Frame')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_xlim([1.49e8,1.515e8])
  ax.set_ylim([-1.5e6,1.5e6])
  ax.set_zlim([-1e6,2e6])
  ax.legend()

  def update(frame):
    Traj.set_data_3d(x[:frame],y[:frame],z[:frame])
    Head._offsets3d = ([x[frame]],[y[frame]],[z[frame]])
    return Traj, Head,

  ani = FuncAnimation(fig=fig,func=update,frames=len(t),interval=0.1,blit=False,repeat=False)
  # ani.save('Animation_Rotation_Frame.gif',writer=PillowWriter(fps=100))
  plt.show()
  

# Plot the Earth with Halo Orbits in Rotation frame
def Plot_static_RF(x,y,z,r_12,x_L2,x_Earth):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  ax.axis('equal')
  ax.set_title('JWST Trajectory in Rotating Frame (CR3BP)',fontsize=20)
  ax.set_xlabel('x',fontsize=20)
  ax.set_ylabel('y',fontsize=20)
  ax.set_zlabel('z',fontsize=20)
  ax.set_xlim([1.49e8,1.515e8])
  ax.set_ylim([-1.5e6,1.5e6])
  ax.set_zlim([-1e6,2e6])
  
  ax.plot(x, y, z, 'r-', label='JWST Orbit',lw=1)
  ax.scatter(x_L2*r_12,0,0,c='tab:brown',marker='*',label='L_2',s=30)
  ax.scatter(x_Earth*r_12,0,0,c='tab:blue',marker='o',label='Earth',s=1)

  plt.legend()
  # plt.savefig('Halo_Orbit.jpg')
  plt.show()

# Animation trajectory in Fixed frame
def Animation_FF(x_JWST,y_JWST,z_JWST,x_Earth,y_Earth,z_Earth,t,r_12):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  ax.axis('equal')
  Traj_JWST = ax.plot([], [], [], 'r-', label='Halo Orbit',lw=1)[0]
  Head_JWST = ax.scatter([],[],[], c='tab:brown',marker='o',s=10,label='JWST')
  Traj_Earth = ax.plot([], [], [], 'b-', label='Earth Orbit',lw=1)[0]
  Head_Earth = ax.scatter([],[],[],c='tab:blue',marker='o',label='Earth',s=40)
  Sun = ax.scatter(0,0,0,c='tab:orange',marker='o',label='Sun',s=150)

  ax.set_title('JWST Trajectory in Fixed Frame')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_xlim([-1.1*r_12,1.1*r_12])
  ax.set_ylim([-1.1*r_12,1.1*r_12])
  ax.set_zlim([-0.1*r_12,0.1*r_12])
  ax.legend()

  def update(frame):
    lag = max(0,frame-200)
    Traj_JWST.set_data_3d(x_JWST[lag:frame],y_JWST[lag:frame],z_JWST[lag:frame])
    Head_JWST._offsets3d = ([x_JWST[frame]],[y_JWST[frame]],[z_JWST[frame]])
    Traj_Earth.set_data_3d(x_Earth[:frame],y_Earth[:frame],z_Earth[:frame])
    Head_Earth._offsets3d = ([x_Earth[frame]],[y_Earth[frame]],[z_Earth[frame]])
    return Traj_JWST, Head_JWST, Traj_Earth, Head_Earth

  ani = FuncAnimation(fig=fig,func=update,frames=len(t),interval=0.1,blit=False,repeat=False)
  # ani.save('Animation_Fixed_Frame.gif',writer=PillowWriter(fps=100))
  plt.show()
  
  
