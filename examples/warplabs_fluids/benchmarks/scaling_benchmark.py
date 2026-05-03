"""
N-scaling throughput benchmark: JAX CPU vs Warp CPU vs Warp CUDA.

Uses fixed N_STEPS steps with a CFL-stable dt at each N,
so the result is purely kernel throughput (no adaptive-dt overhead).
JIT / kernel compilation excluded via one warmup run per (solver, N).

Run from examples/warplabs_fluids/:
  python benchmarks/scaling_benchmark.py
"""

import sys, time, statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, prim_to_cons
from cases.sod import ic as sod_ic
from benchmarks.jax_euler import JaxEuler1D

# ─────────────────────────── config ──────────────────────────────────────────

GRID_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
GAMMA      = 1.4
CFL        = 0.4
N_STEPS    = 200   # fixed steps per timed run (excludes adaptive-dt cost)
N_BENCH    = 5     # timed repetitions after warmup
MAX_WALL_S = 30.0  # skip a (solver, N) combo if warmup exceeds this

# ─────────────────────────── helpers ─────────────────────────────────────────

def _stable_dt(N, gamma, cfl):
    dx = 1.0 / N
    a_max = np.sqrt(gamma * 1.0 / 1.0)  # left-state sound speed (Sod IC)
    return cfl * dx / a_max


def _bench_warp(device, N):
    dx  = 1.0 / N
    dt  = _stable_dt(N, GAMMA, CFL)
    Q0, _ = sod_ic(N, GAMMA)

    solver = WarpEuler1D(N, dx, gamma=GAMMA, bc="outflow", device=device)
    solver.initialize(Q0)

    # Warmup (triggers kernel compilation)
    t_warm = time.perf_counter()
    for _ in range(N_STEPS):
        solver.step(dt)
    wp.synchronize()
    if time.perf_counter() - t_warm > MAX_WALL_S:
        return None

    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            solver.step(dt)
        wp.synchronize()
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    return N * N_STEPS / med / 1e6


def _bench_jax(N):
    import jax
    dx  = 1.0 / N
    dt  = _stable_dt(N, GAMMA, CFL)
    Q0, _ = sod_ic(N, GAMMA)

    solver = JaxEuler1D(N, dx, gamma=GAMMA, bc="outflow")
    solver.initialize(Q0)

    # Warmup (triggers JIT compilation for this N)
    t_warm = time.perf_counter()
    for _ in range(N_STEPS):
        solver.step(dt)
    jax.block_until_ready(solver._Q)
    if time.perf_counter() - t_warm > MAX_WALL_S:
        return None

    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            solver.step(dt)
        jax.block_until_ready(solver._Q)
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    return N * N_STEPS / med / 1e6


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    wp.init()

    solvers = {
        "JAX CPU":   (lambda N: _bench_jax(N),                  "#e07b00", "o--"),
        "Warp CPU":  (lambda N: _bench_warp("cpu",  N),          "#0072b2", "s-"),
        "Warp CUDA": (lambda N: _bench_warp("cuda", N),          "#009e73", "^-"),
    }

    results = {name: {"N": [], "tp": []} for name in solvers}

    for N in GRID_SIZES:
        print(f"\nN = {N:>6}", flush=True)
        for name, (fn, *_) in solvers.items():
            tp = fn(N)
            if tp is None:
                print(f"  {name:<12}  skipped (>{MAX_WALL_S}s warmup)")
            else:
                results[name]["N"].append(N)
                results[name]["tp"].append(tp)
                print(f"  {name:<12}  {tp:7.2f} Mcell/s")

    # ── table ──
    print("\n-- Throughput (Mcell/s) by N --------------------------------")
    header = f"{'N':>7}" + "".join(f"  {n:>12}" for n in solvers)
    print(header)
    print("-" * len(header))
    for N in GRID_SIZES:
        row = f"{N:>7}"
        for name, d in results.items():
            if N in d["N"]:
                idx = d["N"].index(N)
                row += f"  {d['tp'][idx]:>12.2f}"
            else:
                row += f"  {'--':>12}"
        print(row)

    # ── plot ──
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, (fn, color, marker_ls) in solvers.items():
        d = results[name]
        if not d["N"]:
            continue
        marker = marker_ls[0]
        ls     = marker_ls[1:]
        ax.plot(d["N"], d["tp"], marker=marker, ls=ls, color=color,
                lw=1.8, ms=7, label=name)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Grid size  N", fontsize=11)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=11)
    ax.set_title(f"1-D Euler  |  WENO3-HLLC-RK2  |  {N_STEPS} steps, median of {N_BENCH} runs",
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)

    xticks = [N for N in GRID_SIZES if N in results["Warp CUDA"]["N"] or N in results["JAX CPU"]["N"]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(n) for n in xticks], rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    out = ROOT / "benchmarks" / "sod_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
