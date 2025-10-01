# Ball-in-Cup (2D) — Anitescu CCP (QP via OSQP)

A minimal, modular 2D contact simulation: a ball falls under gravity into a “cup” (three walls). Each timestep solves **Anitescu’s convex time-stepping CCP** as a **QP** with **OSQP**, then integrates semi-implicitly. Designed to be small, readable, and hackable for learning contact dynamics (KKT/LCP/CCP).

## Features
- 2D rigid body (ball) with mass/inertia and semi-implicit Euler integration
- Cup geometry from three line segments (with inward normals)
- Contact detection: circle vs segment
- Contact Jacobians per contact (normal & tangent rows)
- Per-step **QP**:
  - objective: (1/2) λᵀ G λ + bᵀ λ where G = J M^{-1} Jᵀ
  - constraints (2D friction cone): λ_n ≥ 0,  |λ_t| ≤ μ λ_n
  - bias term for stabilization/restition at velocity level
- Matplotlib animation (with optional blitting)
- Clean module boundaries so you can swap solvers or geometry

## Directory Structure

Repo/
├─ simulation/
│  ├─ __init__.py
│  ├─ engine/
│  │  ├─ __init__.py
│  │  ├─ solver_ccp_qp.py       # build & solve QP (OSQP)
│  │  └─ integrator.py          # one physics step (detect → QP → integrate)
│  ├─ physics/
│  │  ├─ __init__.py
│  │  └─ rigid_body2d.py        # RigidBody2D (m, I, q, v, M)
│  ├─ geometry/
│  │  ├─ __init__.py
│  │  ├─ primitives.py          # Segment with inward normal n and tangent t
│  │  └─ cup_builder.py         # build_cup(...) -> [left, bottom, right]
│  ├─ collision/
│  │  ├─ __init__.py
│  │  ├─ detect2d.py            # circle-vs-segment contact
│  │  └─ jacobian2d.py          # Jn, Jt rows for a contact
│  └─ demo/
│     ├─ __init__.py
│     └─ animate_ball_in_cup.py # animation using the modules above
└─ (optional) .vscode/launch.json

All __init__.py files can be empty (they just mark packages).

## Setup
1) Activate env & install deps:
   source /home/juneil/venvs/dev/bin/activate
   python -m pip install -U numpy scipy osqp matplotlib

2) Run from repo root:
   python -m simulation.demo.animate_ball_in_cup

(Use -m so Python treats simulation/ as a package.)

## VS Code launch (optional)
Place this in .vscode/launch.json:
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Animate: ball in cup",
      "type": "python",
      "request": "launch",
      "module": "simulation.demo.animate_ball_in_cup",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "python": "/home/juneil/venvs/dev/bin/python"
    }
  ]
}

## Notes
- Set polish=False in solver_ccp_qp.py to silence OSQP polish logs.
- Tweak mu/beta/restitution in the demo to control friction, stabilization, and bounce.
- If the ball doesn't render, try blit=False in the FuncAnimation call or force TkAgg backend.
