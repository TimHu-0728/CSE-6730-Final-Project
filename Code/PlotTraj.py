import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Plot static trajectory in Rotation frame
def Plot_static_RF(x,y,z,t,r_12,x_L2):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  ax.plot(x, y, z, 'r-', label='Halo Orbit',lw=1)
  ax.set_title('JWST Trajectory in Rotating Frame (CR3BP)')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_xlim([1.495e8,1.515e8])
  ax.set_ylim([-1.5e6,1.5e6])
  ax.set_zlim([-1e6,2e6])
  ax.plot(x_L2*r_12,0,0,'b*',label='L_2',ms=9)
  plt.legend()
  plt.show()

# Animate trajectory in Rotation frame
def Animation_RF(x,y,z,t,r_12,x_L2):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  Traj = ax.plot([], [], [], 'r-', label='Halo Orbit',lw=1)[0]
  Head = ax.scatter([],[],[], 'ro',s=40,label='JWST')
  ax.plot(x_L2*r_12,0,0,'b*',label='L_2',ms=9)
  ax.set_title('JWST Trajectory in Rotating Frame')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_xlim([1.495e8,1.515e8])
  ax.set_ylim([-1.5e6,1.5e6])
  ax.set_zlim([-1e6,2e6])
  ax.legend()

  def update(frame):
    Traj.set_data_3d(x[:frame],y[:frame],z[:frame])
    Head._offsets3d = ([x[frame]],[y[frame]],[z[frame]])
    return Traj, Head,

  ani = FuncAnimation(fig=fig,func=update,frames=len(t),interval=0.1,blit=False,repeat=False)
  plt.show()
