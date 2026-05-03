# Warp Examples

A collection of GPU-accelerated examples built on [NVIDIA Warp](https://github.com/NVIDIA/warp) — exploring **computational physics** and **geometry processing** through custom CUDA kernels authored in Python.

Each example lives in its own subfolder under `examples/` with its own code, assets, and tests.

---

## Focus Areas

- **Computational physics** — fluid dynamics, continuum mechanics, thermodynamics, elasticity, wave propagation
- **Geometry processing** — mesh generation, remeshing, signed distance fields, Boolean operations, parametric surfaces
- **Simulation kernels** — custom Warp kernels for solvers that go beyond off-the-shelf physics engines

---

## Prerequisites

### System

- **OS**: Windows 10/11 or Linux
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.5 (Turing or newer)
- **CUDA Toolkit**: 12.x
- **Driver**: ≥ 525.60 (Windows) / ≥ 520.61 (Linux)
- **Python**: 3.10 or newer

### Install Warp

```powershell
pip install warp-lang numpy
```

Additional per-example dependencies are listed in each example's own `README.md`.

---

## Examples

| Example | Domain | Description |
|---|---|---|
| *(coming soon)* | — | First example in progress |

---

## Running tests

Each example has its own test suite. To run all tests across every example:

```powershell
python -m pytest examples/ -v
```

Or for a specific example:

```powershell
python -m pytest examples/<example_name>/tests/ -v
```

---

## Contributing a new example

1. Create a subfolder: `examples/<your_example>/`
2. Add your Python files and a `tests/` directory
3. Include a `README.md` inside the subfolder describing what the example demonstrates and how to run it
4. Keep all units SI (metres, radians, seconds) unless the problem domain dictates otherwise

---

## License

Apache 2.0 — see individual file headers.
