# JWST Halo Orbit Simulation

This repository contains the code for the **CSE 6730 – Modeling & Simulation** course project. The goal is to model and simulate the motion of the **James Webb Space Telescope (JWST)** in the Earth–Sun Circular Restricted Three-Body Problem (CR3BP) and to visualize its halo orbit around the L2 point, along with the supporting 3D animations and web-based video gallery.

---

## Team Members - Group 15

- **Shiqi Fan**
- **Tianyang Hu**
- **Haoran Yan**
- **Mark Zhang**

---

## Overview

This project models the James Webb Space Telescope (JWST) in the Earth–Sun Circular Restricted Three-Body Problem (CR3BP) and simulates its halo orbit around the L2 point. The main goal is to demonstrate an end-to-end workflow for computational modeling and visualization:

- **Modeling:** Formulate the CR3BP equations of motion and represent the Sun–Earth–JWST system in a rotating reference frame.
- **Simulation:** Propagate JWST’s trajectory using a nonlinear state-space model and numerical integration over long timescales.
- **Visualization:** Render the orbit in both rotating and inertial frames using Matplotlib and a 3D PyVista scene with a textured JWST model.
- **Front-end UI:** Expose the generated animations through a lightweight web UI (`ui/`) that loads `.mp4` files from the `results/` directory as an interactive video gallery.

Together, these components provide a compact example of integrating physics-based modeling, numerical simulation, and 3D visualization with a simple user interface for exploring the results.

---

## Project Structure & Key Files

The repository is organized into modular components that separate simulation logic, 3D assets, generated results, and the front-end UI:

```text
CSE-6730-FINAL-PROJECT/
│
├── assets/                     # 3D resources for visualization
│   ├── cubemaps/               # Environment maps for PyVista scenes
│   ├── models/                 # JWST GLTF model and related meshes
│   └── textures/               # Planet and background textures
│
├── results/                    # Generated outputs (ignored by git if large)
│   ├── animation/              # .mp4 orbit visualizations
│   ├── figures/                # Static plots and rendered frames
│   └── reports/                # Exported PDFs and milestone write-ups
│
├── scratch/                    # Experimental notebooks and temporary assets
│   ├── planets/                # Prototype planet models / tests
│   └── source/                 # Miscellaneous test scripts
│
├── src/                        # Core simulation and visualization code
│   ├── classes.py              # Celestial, JWST, ThreeBodySystem classes; CR3BP dynamics
│   ├── main.py                 # Top-level script: set ICs, run simulation, call visualizers
│   ├── plotTraj.py             # Matplotlib 3D plots and animations (rotating / fixed frames)
│   └── usePyVista.py           # PyVista scenes (JWST model, lighting, camera paths)
│
├── ui/                         # Front-end video gallery for simulation results
│   ├── index.html              # Entry page for browsing animations
│   ├── app.js                  # Simple JS logic to load and play videos
│   ├── style.css               # Layout and styling for the gallery
│   └── imgs/                   # Thumbnails and static images used in the UI
│
├── requirements.txt            # Python dependencies for running the simulations
├── README.md
└── .gitignore
```

---

## How to run

### 1. Reproduce simulations and animations

#### Step 1. Install dependencies:

```bash
pip install -r requirements.txt
```

#### Step 2. Run the main script:

```bash
cd src
python main.py
```

This will:

- Integrate the CR3BP dynamics for JWST,
- Generate rotating-frame and fixed-frame plots/animations,
- Save the resulting `.mp4` files and figures under `../results/`
  (e.g., `results/animation/`, `results/figures/`),

### 2. Run the visualization UI

#### Step 1. From the project root `CSE-6730-FINAL-PROJECT/`, start a simple HTTP server:

```bash
python3 -m http.server 8000
```

#### Step 2. Open a browser and go to:

```bash
http://localhost:8000/ui/
```
The UI will load `.mp4` animations from the `results/` directory and present them as a simple video gallery. Refresh the page after rerunning simulations to see newly generated videos.

---

## Notes
- Two initial-condition vectors are included in `main.py`: one for a stable halo orbit and one for an unstable halo orbit.
- Numerical integration uses `solve_ivp` (RK45) with tight tolerances in the simulation call.
- The PyVista visualization is not to-scale for the spacecraft and planetary radii (scaled for visual clarity).
- Intended as the final project for Georgia Tech's CSE 6730 course.

---

### JWST Halo Orbit – Fixed Frame
![Fixed Frame](results/animation/jwst_ff.gif)

### JWST Halo Orbit – Rotating Frame
![Rotating Frame](results/animation/jwst_rf.gif)

*Note: This markdown was written with assistance from AI resources.*
