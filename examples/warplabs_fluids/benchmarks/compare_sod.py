"""
Sod shock tube: warplabs-fluids vs JAX reference.

Produces:
  sod_profiles.png   — density / velocity / pressure profiles + exact solution
  sod_benchmark.png  — throughput comparison (Mcell-updates / second)

JAX CUDA is enabled automatically when running on Linux with jax[cuda12].

Run from examples/warplabs_fluids/:
  python benchmarks/compare_sod.py
"""

import sys, time, statistics, contextlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, prim_to_cons, cons_to_prim, l1_error, l2_error
from cases.sod import ic as sod_ic, exact as sod_exact
from benchmarks.jax_euler import JaxEuler1D

# ─────────────────────────--config ──────────────────────────────────────────

N       = 512
GAMMA   = 1.4
DX      = 1.0 / N
T_END   = 0.2
CFL     = 0.4
N_BENCH = 5      # timed repetitions (after warmup)
N_WARM  = 3      # warmup runs

# ─────────────────────────--helpers ─────────────────────────────────────────

def _run_warp(solver, Q0, n_warm, n_bench):
    """Benchmark a WarpEuler1D solver."""
    for _ in range(n_warm):
        solver.initialize(Q0)
        n_steps = solver.run(T_END, CFL)
    wp.synchronize()

    times = []
    for _ in range(n_bench):
        solver.initialize(Q0)
        t0      = time.perf_counter()
        n_steps = solver.run(T_END, CFL)
        wp.synchronize()
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    throughput = N * n_steps / med / 1e6
    return solver.state, n_steps, med, throughput


def _run_jax(solver, Q0, n_warm, n_bench, jax_device=None):
    """Benchmark a JaxEuler1D solver on a given JAX device (None = default)."""
    import jax

    ctx = jax.default_device(jax_device) if jax_device else contextlib.nullcontext()

    with ctx:
        for _ in range(n_warm):
            solver.initialize(Q0)
            n_steps = solver.run(T_END, CFL)
        jax.block_until_ready(solver._Q)

        times = []
        for _ in range(n_bench):
            solver.initialize(Q0)
            t0      = time.perf_counter()
            n_steps = solver.run(T_END, CFL)
            jax.block_until_ready(solver._Q)
            times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    throughput = N * n_steps / med / 1e6
    return solver.state, n_steps, med, throughput


def _jax_gpu_device():
    try:
        import jax
        gpus = jax.devices('gpu')
        return gpus[0] if gpus else None
    except Exception:
        return None


# ─────────────────────────--run all solvers ─────────────────────────────────

def main():
    import jax

    wp.init()
    Q0, x = sod_ic(N, GAMMA)

    jax_gpu = _jax_gpu_device()
    n_solvers = 3 + (1 if jax_gpu else 0)

    if jax_gpu:
        print(f"[info] JAX GPU detected: {jax_gpu} — running {n_solvers}-backend comparison")
    else:
        print(f"[info] No JAX GPU — running {n_solvers}-backend comparison")

    results = {}
    step = 0

    # --- JAX CPU ---
    step += 1
    print(f"\n[{step}/{n_solvers}] JAX CPU   (N={N}, warmup={N_WARM}, bench={N_BENCH}) ...", flush=True)
    cpu_dev = jax.devices('cpu')[0]
    jax_solver = JaxEuler1D(N, DX, gamma=GAMMA, bc="outflow")
    Q_jax, n_jax, t_jax, tp_jax = _run_jax(jax_solver, Q0, N_WARM, N_BENCH, jax_device=cpu_dev)
    results["JAX CPU"] = dict(Q=Q_jax, n=n_jax, t=t_jax, tp=tp_jax, color="#e07b00", ls="--")
    print(f"   {n_jax} steps  {t_jax*1000:.1f} ms median  {tp_jax:.2f} Mcell/s")

    # --- JAX CUDA (Linux + jax[cuda12] only) ---
    if jax_gpu:
        step += 1
        print(f"\n[{step}/{n_solvers}] JAX CUDA  (N={N}) ...", flush=True)
        jax_solver_gpu = JaxEuler1D(N, DX, gamma=GAMMA, bc="outflow")
        Q_jg, n_jg, t_jg, tp_jg = _run_jax(jax_solver_gpu, Q0, N_WARM, N_BENCH, jax_device=jax_gpu)
        results["JAX CUDA"] = dict(Q=Q_jg, n=n_jg, t=t_jg, tp=tp_jg, color="#d55e00", ls="--")
        print(f"   {n_jg} steps  {t_jg*1000:.1f} ms median  {tp_jg:.2f} Mcell/s")

    # --- Warp CPU ---
    step += 1
    print(f"\n[{step}/{n_solvers}] Warp CPU  (N={N}) ...", flush=True)
    wcpu = WarpEuler1D(N, DX, gamma=GAMMA, bc="outflow", device="cpu")
    Q_wcpu, n_wcpu, t_wcpu, tp_wcpu = _run_warp(wcpu, Q0, N_WARM, N_BENCH)
    results["Warp CPU"] = dict(Q=Q_wcpu, n=n_wcpu, t=t_wcpu, tp=tp_wcpu, color="#0072b2", ls="-")
    print(f"   {n_wcpu} steps  {t_wcpu*1000:.1f} ms median  {tp_wcpu:.2f} Mcell/s")

    # --- Warp CUDA ---
    step += 1
    print(f"\n[{step}/{n_solvers}] Warp CUDA (N={N}) ...", flush=True)
    try:
        wcuda = WarpEuler1D(N, DX, gamma=GAMMA, bc="outflow", device="cuda")
        Q_wcuda, n_wcuda, t_wcuda, tp_wcuda = _run_warp(wcuda, Q0, N_WARM, N_BENCH)
        results["Warp CUDA"] = dict(Q=Q_wcuda, n=n_wcuda, t=t_wcuda, tp=tp_wcuda, color="#009e73", ls="-")
        print(f"   {n_wcuda} steps  {t_wcuda*1000:.1f} ms median  {tp_wcuda:.2f} Mcell/s")
    except Exception as e:
        print(f"   CUDA unavailable: {e}")

    # Exact solution
    rho_ex, u_ex, p_ex = sod_exact(T_END, x, GAMMA)

    # accuracy table
    sep = "-" * 50
    print("\n-- Accuracy vs exact Riemann solution ----------------------")
    print(f"{'Solver':<14}  {'L1(rho)':>10}  {'L1(u)':>10}  {'L1(p)':>10}")
    print(sep)
    for name, r in results.items():
        rho, u, p = cons_to_prim(np.asarray(r["Q"]), GAMMA)
        e_rho = l1_error(rho, rho_ex, DX)
        e_u   = l1_error(u,   u_ex,   DX)
        e_p   = l1_error(p,   p_ex,   DX)
        print(f"{name:<14}  {e_rho:>10.3e}  {e_u:>10.3e}  {e_p:>10.3e}")

    # throughput table
    print("\n-- Throughput -----------------------------------------------")
    print(f"{'Solver':<14}  {'Steps':>7}  {'Time (ms)':>10}  {'Mcell/s':>10}")
    print(sep)
    for name, r in results.items():
        print(f"{name:<14}  {r['n']:>7}  {r['t']*1000:>10.1f}  {r['tp']:>10.2f}")

    # profile plots
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Sod shock tube  |  N={N}  |  t={T_END}", fontsize=12, fontweight="bold")

    fields   = ["density", "velocity", "pressure"]
    exact_v  = [rho_ex, u_ex, p_ex]
    Q0_np    = Q0.copy()
    rho0, u0, p0 = cons_to_prim(Q0_np, GAMMA)
    ic_vals  = [rho0, u0, p0]

    for ax, fname, ex_v, ic_v in zip(axes, fields, exact_v, ic_vals):
        ax.plot(x, ic_v, color="0.75", lw=1.2, ls=":", label="initial cond.")
        ax.plot(x, ex_v, color="k",    lw=1.6, ls="-",  label="exact", zorder=5)

        for name, r in results.items():
            rho, u, p = cons_to_prim(np.asarray(r["Q"]), GAMMA)
            vals = [rho, u, p]
            idx  = fields.index(fname)
            ax.plot(x, vals[idx], color=r["color"], lw=1.4, ls=r["ls"],
                    label=name, alpha=0.9)

        ax.set_xlabel("x")
        ax.set_title(fname)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, lw=0.4, alpha=0.5)

    plt.tight_layout()
    out_profiles = ROOT / "benchmarks" / "sod_profiles.png"
    fig.savefig(out_profiles, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_profiles}")

    # benchmark bar chart
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    names  = list(results.keys())
    tps    = [results[n]["tp"] for n in names]
    colors = [results[n]["color"] for n in names]

    bars = ax2.bar(names, tps, color=colors, width=0.5, edgecolor="k", linewidth=0.7)
    ax2.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    ax2.set_ylabel("Throughput  (Mcell-updates / s)")
    ax2.set_title(f"Sod  N={N}  |  WENO3-HLLC-RK2  |  median of {N_BENCH} runs")
    ax2.set_ylim(0, max(tps) * 1.3)
    ax2.grid(True, axis="y", lw=0.4, alpha=0.5)

    plt.tight_layout()
    out_bench = ROOT / "benchmarks" / "sod_benchmark.png"
    fig2.savefig(out_bench, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_bench}")


if __name__ == "__main__":
    main()
