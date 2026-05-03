# warplabs-fluids — Claude Context

## What this is

Experimental Warp-native compressible flow solver, living at
`C:\Vibe Coding\warp-examples\examples\warplabs_fluids\`.

Goal: port a JAX-based CFD solver to Warp, prove behavioral equivalence via V&V,
then benchmark to show GPU throughput advantage. Long-term reference: JaxFluids.
Short-term reference: hand-written JAX solver with identical numerics (`benchmarks/jax_euler.py`).

---

## Phase 1 — COMPLETE

**Scope:** 1-D compressible Euler (inviscid), WENO3-HLLC-SSP-RK2, float32.

### Numerical scheme

| Component | Choice | Notes |
|---|---|---|
| Equations | Compressible Euler 1D | [rho, rho*u, E] |
| Reconstruction | WENO3 on primitives | Jiang-Shu (1996) smoothness indicators |
| Riemann solver | HLLC | Toro (2009), Einfeldt wave speeds |
| Time integration | SSP-RK2 | strong-stability-preserving |
| Ghost cells | ng=2 | embedded in state array |
| CFL | 0.4 default | |
| Precision | float32 | matches JAX default |

### Kernel architecture

Draw kernel boundaries at global-memory write points. `@wp.func` = in-register logic.

```
bc_kernel          1 launch   fill ng=2 ghost cells (outflow or periodic)
compute_flux_1d    1 launch   WENO3 + HLLC fused — Q_L/Q_R in registers, only F written
update_rk_1d       1 launch   SSP-RK2 stage: Q_out = alpha*Q0 + beta*Q_in + coeff*dt*RHS
──────────────────────────────
3 launches × 2 RK stages = 6 kernel launches per timestep
```

### File layout

```
warplabs_fluids/
  kernels/
    primitives.py    @wp.func  cons<->prim conversion
    reconstruct.py   @wp.func  weno3_left, weno3_right
    riemann.py       @wp.func  hllc_flux_1d -> wp.vec3f
    bc.py            @wp.kernel bc_outflow_1d, bc_periodic_1d
    flux.py          @wp.kernel compute_flux_1d  (fused WENO3+HLLC)
    update.py        @wp.kernel update_rk_1d
  solver.py          WarpEuler1D class
  utils.py           prim_to_cons, cons_to_prim, l1/l2/linf error (numpy)
cases/
  sod.py             Sod IC + exact Riemann solver (scipy brentq)
benchmarks/
  jax_euler.py       JaxEuler1D — identical scheme in JAX (reference/comparison)
  compare_sod.py     Runs all backends, prints accuracy+throughput, saves PNGs
  scaling_benchmark.py  N-sweep 256..32768, finds GPU crossover point
tests/
  test_primitives.py  roundtrip, energy sign
  test_weno3.py       constant/linear/discontinuity bias
  test_hllc.py        rest state, supersonic, Sod mass flux sign
  test_sod.py         L1 vs exact Riemann (N=256,512)
  test_conservation.py  mass+momentum conserved <1e-5 over 1000 steps (periodic)
```

### V&V results (N=512, t=0.2)

All three backends (JAX CPU, Warp CPU, Warp CUDA) produce **identical** output:
- L1(rho) = 1.73e-3
- L1(u)   = 3.19e-3
- L1(p)   = 1.18e-3

Convergence rate vs N is ~O(N^-0.8), consistent with WENO3 near discontinuities.

### Benchmark results (RTX 5000 Ada + Intel Core Ultra 9)

Fixed 200 steps, float32, median of 5 runs:

| N | JAX CPU | Warp CPU | Warp CUDA |
|---|---|---|---|
| 256 | 13.0 | 2.4 | 1.7 |
| 512 | 15.4 | 3.9 | 3.4 |
| 1024 | 19.1 | 5.6 | 6.8 |
| 2048 | 21.2 | 7.1 | 13.9 |
| 4096 | 17.7 | 8.4 | 26.4 |
| 8192 | 18.1 | 9.2 | 54.9 |
| 16384 | 15.1 | 9.5 | 110.7 |
| 32768 | 38.0 | 10.0 | **208.6** |

All Mcell-updates/s. Crossovers:
- Warp CUDA > Warp CPU: N ~ 1024
- Warp CUDA > JAX CPU:  N ~ 4096

### Key findings

1. **Accuracy:** All backends agree to float32 precision — same algorithm, same numerics.
2. **Warp CPU is slower than JAX CPU** at all N. Warp's CPU backend uses single-threaded LLVM, no AVX. JAX XLA emits AVX2 vector ops over the full array.
3. **Warp CUDA scales linearly with N** — still rising at N=32768 (not yet bandwidth-limited on the GPU).
4. **JAX GPU unavailable on Windows** — CUDA jaxlib wheels are Linux-only. GPU comparison requires Linux or WSL2.
5. **The compute_dt() CFL call** reads GPU state to CPU every step — significant overhead at small N. Future optimization: GPU-side reduction.

---

## Phase 2 — PLANNED

- 2-D compressible Euler
- Strang dimensional splitting (x/2 → y → x/2) for 2nd-order accuracy
- Kelvin-Helmholtz instability (V&V: qualitative + energy spectrum vs JAX)
- 2-D scaling benchmark (NxN grids)
- `WarpEuler2D` class reusing 1-D kernels per axis

## Phase 3 — PLANNED

- 2-D compressible Navier-Stokes (add viscous flux kernel)
- Taylor-Green vortex V&V

---

## Known issues / future work

- `compute_dt()` does CPU readback every step — replace with a Warp reduction kernel for Phase 2
- Warp CPU single-threaded — no parallel CPU path available in Warp 1.12
- JAX GPU comparison deferred until Linux/WSL2 environment available
- PNG plots in `benchmarks/` are committed via local `.gitignore` override (`!*.png`)

---

## Running

```powershell
# From examples/warplabs_fluids/

# Tests (CPU only, no GPU needed)
python -m pytest tests/ -v

# Sod comparison: accuracy table + 2 PNG plots
python benchmarks/compare_sod.py

# N-scaling throughput benchmark
python benchmarks/scaling_benchmark.py
```
