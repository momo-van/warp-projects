"""
Sod shock tube — throughput + GPU memory N-scaling benchmark.
Warp WENO5-Z+HLLC+RK3 (f32)  and  optional JaxFluids (WENO5-Z, f64).

Saves (to benchmarks/sod/):
  sod_scaling.png      — throughput vs N (log-log)
  sod_memory.png       — GPU memory vs N (log-log)
  throughput_scaling.csv
  memory_scaling.csv

Run from examples/warplabs_fluids/ on WSL2:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/sod/throughput_memory.py
"""

import csv, gc, json, os, subprocess, sys, statistics, tempfile, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D
from cases.sod import ic as sod_ic

GRID_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
GAMMA      = 1.4
CFL        = 0.4
N_STEPS    = 200
N_BENCH    = 5
MAX_WALL_S = 60.0
_A_MAX     = float(np.sqrt(GAMMA))


def _nvml_mem_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def _theory_warp_mib(N):
    # WENO5-Z RK3: 3 state arrays, ng=3 → N_ext = N+6
    return 3 * 3 * (N + 6) * 4 / (1024 ** 2)


def _stable_dt(N):
    return CFL * (1.0 / N) / _A_MAX


def bench_warp(device, N):
    dt = _stable_dt(N)
    Q0, _ = sod_ic(N, GAMMA)
    wp.synchronize()
    mem0   = _nvml_mem_mib()
    solver = WarpEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow", device=device, scheme="weno5z-rk3")
    solver.initialize(Q0)
    t0w = time.perf_counter()
    for _ in range(N_STEPS): solver.step(dt)
    wp.synchronize()
    if time.perf_counter() - t0w > MAX_WALL_S:
        del solver; gc.collect(); return None, None
    mem_delta = max(0.0, _nvml_mem_mib() - mem0)
    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0 = time.perf_counter()
        for _ in range(N_STEPS): solver.step(dt)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    tp = N * N_STEPS / statistics.median(times) / 1e6
    del solver; gc.collect()
    return tp, mem_delta


def bench_jaxfluids(N, case_tmpl, num_path, tmp_dir):
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    import jax
    run_dir = tmp_dir / f"sod_N{N}"; run_dir.mkdir(exist_ok=True)
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(run_dir)
    case["general"]["save_dt"]   = 999.0
    cp = run_dir / "case.json"; cp.write_text(json.dumps(case))
    mem0 = _nvml_mem_mib()
    im  = InputManager(str(cp), str(num_path))
    ini = InitializationManager(im); sim = SimulationManager(im)
    buf = ini.initialization()
    t0w = time.perf_counter(); sim.simulate(buf); jax.block_until_ready(buf)
    if time.perf_counter() - t0w > MAX_WALL_S: return None, None
    mem_delta = max(0.0, _nvml_mem_mib() - mem0)
    times = []
    for _ in range(N_BENCH):
        buf = ini.initialization(); t0 = time.perf_counter()
        sim.simulate(buf); jax.block_until_ready(buf)
        times.append(time.perf_counter() - t0)
    try:
        cfl_jxf = json.load(open(num_path))["conservatives"]["time_integration"].get("CFL", 0.5)
    except Exception:
        cfl_jxf = 0.5
    n_steps = max(1, int(round(GAMMA * (1.0/N) / _A_MAX * cfl_jxf)))
    n_steps = max(1, int(round(0.2 / (cfl_jxf * (1.0/N) / _A_MAX))))
    tp = N * n_steps / statistics.median(times) / 1e6
    return tp, mem_delta


def main():
    wp.init()

    jxf_ok = False; case_tmpl = None; num_path = None; tmp_dir = None
    try:
        case_tmpl = json.load(open(JXF_EX / "sod.json"))
        num_setup = json.load(open(JXF_EX / "numerical_setup.json"))
        tmp_dir   = Path(tempfile.mkdtemp(prefix="tp_sod_"))
        num_path  = tmp_dir / "numerical_setup.json"
        num_path.write_text(json.dumps(num_setup))
        jxf_ok = True
        print(f"[info] JaxFluids templates loaded")
    except Exception as e:
        print(f"[info] JaxFluids not available ({e}) — Warp-only run")

    solvers = {
        "Warp CPU  (WENO5-Z, f32)":  (lambda N: bench_warp("cpu",  N), "#0072b2", "s--", False),
        "Warp CUDA (WENO5-Z, f32)":  (lambda N: bench_warp("cuda", N), "#009e73", "^-",  True),
    }
    if jxf_ok:
        solvers["JaxFluids (WENO5-Z, f64)"] = (
            lambda N: bench_jaxfluids(N, case_tmpl, num_path, tmp_dir),
            "#e07b00", "D-", True)

    results = {name: {"N": [], "tp": [], "mem": []} for name in solvers}

    for N in GRID_SIZES:
        print(f"\nN = {N:>7}", flush=True)
        for name, (fn, *_) in solvers.items():
            try:
                tp, mem = fn(N)
            except Exception as e:
                print(f"  {name:<38}  ERROR: {e}"); continue
            if tp is None:
                print(f"  {name:<38}  skipped (>{MAX_WALL_S}s)")
            else:
                results[name]["N"].append(N)
                results[name]["tp"].append(tp)
                results[name]["mem"].append(mem)
                mem_str = f"  mem Δ={mem:.1f} MiB" if mem and mem > 0.01 else ""
                print(f"  {name:<38}  {tp:8.2f} Mcell/s{mem_str}")

    if tmp_dir:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    all_N = sorted({n for d in results.values() for n in d["N"]})

    print("\n-- Throughput (Mcell/s) --")
    hdr = f"{'N':>8}" + "".join(f"  {n:>30}" for n in solvers)
    print(hdr); print("-" * len(hdr))
    for N in all_N:
        row = f"{N:>8}"
        for d in results.values():
            row += f"  {d['tp'][d['N'].index(N)]:>30.2f}" if N in d["N"] else f"  {'--':>30}"
        print(row)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    with open(OUT / "throughput_scaling.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["solver", "N", "throughput_Mcells"])
        for name, d in results.items():
            for N, tp in zip(d["N"], d["tp"]): w.writerow([name, N, tp])
    print(f"\nSaved -> {OUT/'throughput_scaling.csv'}")

    with open(OUT / "memory_scaling.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["solver", "N", "mem_MiB"])
        for name, d in results.items():
            for N, mem in zip(d["N"], d["mem"]): w.writerow([name, N, mem])
    print(f"Saved -> {OUT/'memory_scaling.csv'}")

    subtitle = (
        "1-D Euler  ·  Sod shock tube  ·  WENO5-Z+HLLC+RK3 (fused)  ·  200 steps, median 5 runs\n"
        "WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(subtitle, fontsize=8, color="0.4")
    for name, (fn, color, mls, _) in solvers.items():
        d = results[name]
        if not d["N"]: continue
        m, ls = mls[0], mls[1:]
        ax.plot(d["N"], d["tp"], marker=m, ls=ls, color=color, lw=1.8, ms=7, label=name)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title("Sod  —  throughput vs grid size", fontsize=13, pad=10)
    ax.legend(fontsize=10); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(all_N); ax.set_xticklabels([f"{n:,}" for n in all_N], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sod_scaling.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {OUT/'sod_scaling.png'}")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(subtitle, fontsize=8, color="0.4")
    has_mem = False
    for name, (fn, color, mls, show_mem) in solvers.items():
        d = results[name]
        if not show_mem or not d["N"]: continue
        m, ls = mls[0], mls[1:]
        if d["mem"] and any(v > 0.01 for v in d["mem"]):
            ax.plot(d["N"], d["mem"], marker=m, ls=ls, color=color, lw=1.8, ms=7,
                    label=f"{name} (measured)")
            has_mem = True
    N_th = np.array(all_N, dtype=float)
    ax.plot(N_th, [_theory_warp_mib(n) for n in all_N], color="0.5", ls=":", lw=1.4,
            marker=".", label="theory  3×3×(N+6)×4 B  (Warp RK3)")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Peak GPU memory  Δ (MiB)", fontsize=12)
    ax.set_title("Sod  —  GPU memory vs grid size\n(XLA_PYTHON_CLIENT_PREALLOCATE=false)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=10); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(all_N); ax.set_xticklabels([f"{n:,}" for n in all_N], rotation=35, ha="right", fontsize=8)
    if not has_mem:
        ax.text(0.5, 0.5, "nvidia-smi not available\n(theoretical line only)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="0.5")
    fig.tight_layout()
    fig.savefig(OUT / "sod_memory.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {OUT/'sod_memory.png'}")


if __name__ == "__main__":
    main()
