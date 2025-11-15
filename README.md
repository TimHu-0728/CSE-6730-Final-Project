# JWST Halo Orbit Simulation

This repository contains the code for my **CSE 6730 (Modeling & Simulation)** course project. The goal is to model and simulate the motion of the **James Webb Space Telescope (JWST)** in the Earth–Sun Circular Restricted Three-Body Problem (CR3BP) and visualize its halo orbit around the L2 point.

---

## Team Members
Team 15:
- **Shiqi Fan**
- **Tianyang Hu**
- **Haoran Yan**
- **Mark Zhang**

---

## Overview
This project models the James Webb Space Telescope (JWST) in the Earth–Sun Circular Restricted Three-Body Problem (CR3BP) and simulates its halo orbit around the L2 point. The aim is to demonstrate the full workflow used in computational modeling:

- **Modeling:** Formulating the CR3BP equations of motion and representing the Sun–Earth–JWST system in a rotating reference frame.
- **Simulation:** Propagating JWST’s trajectory using a nonlinear state-space model and integrating it over long timescales.
- **Visualization:** Rendering the orbit in both rotating and inertial frames using Matplotlib and a 3D PyVista scene with a JWST model.

This serves as a compact example of integrating physics-based modeling with numerical simulation and 3D visualization.

## Key Files
```
main.py               # Top-level script: set initial state, run simulation, call visualizers
classes.py            # Defines Celestial, JWST, ThreeBodySystem; contains JWST_update_nondim, JWST_output, RotToFixed
plotTraj.py           # Matplotlib 3D plots & animations: Plot_static_RF, Animation_RF, Animation_FF
usePyVista.py         # PyVista interactive visualization: jwstVisualizationFixed (GLTF model, cubemap, camera callback)
assets/               # 3D resources (JWST model + cubemap)
requirements.txt      # Python dependencies
```

---

## How to run

### 1. Reproduce simulations and animations

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python main.py
```
This will run the nonlinear propagation (via `control.nlsys` + `ct.input_output_response`), produce rotating-frame plots and fixed-frame animations, and optionally launch the PyVista viewer if `usePyVista.jwstVisualizationFixed` is called.


### 2. Run the visualization UI

1. From the project root, start a simple HTTP server:
```bash
python3 -m http.server 8000
```

2. Open a browser and go to:
```bash
http://localhost:8000/ui/
```
---

## Notes
- Two initial-condition vectors are included in `main.py`: one for a stable halo orbit and one for an unstable halo orbit.
- Numerical integration uses `solve_ivp` (RK45) with tight tolerances in the simulation call.
- The PyVista visualization is not to-scale for the spacecraft and planetary radii (scaled for visual clarity).
- Intended as the final project for Georgia Tech's CSE 6730 course.

---

*Note: This markdown was written with assistance from AI resources.*

