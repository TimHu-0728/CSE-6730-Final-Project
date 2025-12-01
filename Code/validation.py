import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


#   Jacobi Constant Computation in CR3BP
def compute_jacobi_constant(TB, x, y, z, xdot, ydot, zdot):
    """
    Computes Jacobi constant C = 2Ω(x,y,z) - (xdot^2 + ydot^2 + zdot^2)
    in nondimensional CR3BP rotating frame.
    """
    pi_1 = TB.pi_1
    pi_2 = TB.pi_2

    r1 = np.sqrt((x + pi_2)**2 + y**2 + z**2)
    r2 = np.sqrt((x - pi_1)**2 + y**2 + z**2)

    Omega = 0.5 * (x**2 + y**2) + pi_1 / r1 + pi_2 / r2
    C = 2 * Omega - (xdot**2 + ydot**2 + zdot**2)
    return C

#   Jacobi Constant Validation
def validate_jacobi(TB, outputs, time, save_dir="results/validation"):
    """
    Check if Jacobi constant stays nearly constant.
    Outputs a plot and prints drift statistics.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x = outputs['x']
    y = outputs['y']
    z = outputs['z']
    xdot = outputs['xdot']
    ydot = outputs['ydot']
    zdot = outputs['zdot']

    C = compute_jacobi_constant(TB, x, y, z, xdot, ydot, zdot)
    C0 = C[0]
    drift = np.abs(C - C0) / abs(C0)

    eps_C = np.max(drift)

    # Plot Jacobi Drift
    plt.figure(figsize=(7,4))
    plt.plot(time, C - C0, lw=1)
    plt.xlabel("Time (nondimensional)")
    plt.ylabel("C(t) - C(0)")
    plt.title("Jacobi Constant Drift")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "jacobi_drift.png", dpi=200)
    plt.close()

    print("\n===== Jacobi Constant Validation =====")
    print(f"Initial Jacobi constant C0 = {C0:.6e}")
    print(f"Max relative drift ε_C = {eps_C:.3e}")
    print(f"Plot saved to  {save_dir/'jacobi_drift.png'}")

    return eps_C, C


#   Halo Orbit Boundedness (Distance to L2)
def validate_halo_boundedness(TB, x, y, z, save_dir="results/validation"):
    """
    Check if halo orbit stays in bounded region around L2.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    L2_pos = np.array([TB.x_L2 * TB.r_12, 0, 0])

    r = np.sqrt((x - L2_pos[0])**2 + y**2 + z**2)
    r0 = r[0]

    r_min, r_max = np.min(r), np.max(r)

    # relative bounds
    rel_min = r_min / r0
    rel_max = r_max / r0

    # Plot r(t)
    plt.figure(figsize=(7,4))
    plt.plot(r, lw=1)
    plt.axhline(r0, color='k', ls='--', lw=1, label="initial radius")
    plt.xlabel("Time index")
    plt.ylabel("Distance to L2 [km]")
    plt.title("Halo Orbit Boundedness Check")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "halo_boundedness.png", dpi=200)
    plt.close()

    print("\n===== Halo Orbit Boundedness Validation =====")
    print(f"r0       = {r0:.3e} km")
    print(f"r_min    = {r_min:.3e} km   →  {rel_min:.3f} r0")
    print(f"r_max    = {r_max:.3e} km   →  {rel_max:.3f} r0")
    print(f"Plot saved to  {save_dir/'halo_boundedness.png'}")

    return (r_min, r_max, r0)
