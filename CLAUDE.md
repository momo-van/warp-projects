# Warp Examples — Claude Context

## Repo

`C:\Vibe Coding\warp-examples` — a growing collection of GPU-accelerated examples built on NVIDIA Warp,
focused on computational physics and geometry processing.

GitHub: `momo-van/warp-examples` (private)
Owner: mmohajerani@nvidia.com (NVIDIA)

---

## Purpose

This repo is a hands-on exploration space for:
- Writing custom Warp CUDA kernels for physics solvers
- Geometry processing algorithms (mesh operations, SDFs, parametric surfaces)
- Bridging Warp with visualization tools (Rerun, USD, matplotlib)
- Eventually: surrogate models (PhysicsNeMo) + Warp kernel integration

It is **not** tied to Newton (that lives in `newton-examples`). Warp is used here standalone or
with lightweight scientific Python stacks.

---

## Stack

| Layer | Library |
|---|---|
| GPU kernels | [Warp](https://github.com/NVIDIA/warp) (`warp-lang`) |
| Numerics | numpy, scipy |
| 3-D geometry | trimesh, Shapely, open3d (as needed) |
| Visualization | Rerun (`rerun-sdk`), matplotlib, polyscope (as needed) |
| Scene format | USD (`usd-core`, `pxr`) — Z-up, metres (when needed) |
| Tests | pytest — stub-based where GPU not available |

Core install: `pip install warp-lang numpy`

---

## Repo layout

```
examples/
  <example_name>/      # one folder per example
    *.py               # simulation / geometry scripts
    README.md          # per-example description + run instructions
    tests/
      test_*.py        # pytest unit tests (stub-based, no GPU required)
```

---

## Conventions

- All units SI (metres, radians, seconds) unless problem domain dictates otherwise
- Warp kernels: `@wp.kernel`, launched with `wp.launch(...)`
- Keep GPU-free logic separable so tests run without CUDA
- Test stubs live inside the test file — no separate conftest stubs per example
- Visualisation: prefer Rerun for 3-D, matplotlib for 2-D plots

---

## Domain focus areas

| Domain | Notes |
|---|---|
| Computational fluid dynamics | SPH, grid-based, lattice Boltzmann |
| Solid mechanics / FEM | Elasticity, contact, plasticity |
| Geometry processing | SDF construction, mesh smoothing, remeshing, Booleans |
| Heat transfer | Conduction, convection kernels |
| Wave physics | Acoustic / elastic wave propagation |
| ML-physics coupling | PhysicsNeMo surrogate injection points |

---

## Running

```powershell
# Run a specific example
cd examples\<example_name>
python <script>.py

# Run all tests (no GPU needed)
python -m pytest examples\ -v
```
