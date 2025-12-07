# Bézier Motion Inbetweening (Toy Keyjoint Control)

This repository demonstrates a lightweight, animator-aligned approach to motion
inbetweening using **cubic Bézier curves** for sparse keyjoint trajectories.

The goal is to provide a clean prototype of:
- **low-dimensional motion control**
- **trajectory fitting**
- **simple constraint editing**

This mirrors modern ideas in controllable motion synthesis where sparse, intuitive
controls are modeled first and later expanded to full-body motion.

## Features
- Fit a cubic Bézier curve to a 3D keyjoint trajectory.
- Optional piecewise Bézier fitting.
- Constraint demo: pull the curve toward a user-defined target at a chosen time.

## Install
```bash
pip install -r requirements.txt
pip install -e .
