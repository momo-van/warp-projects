"""
JaxFluids vs Warp CUDA — Shu-Osher shock-density interaction.

Runs the Shu-Osher problem (Mach-3 shock + sin density) with:
  - JaxFluids  (WENO5-Z + HLLC + RK3, double precision — production solver)
  - Our JAX    (WENO3   + HLLC + RK2, float32 — hand-written reference)
  - Warp CUDA  (WENO3   + HLLC + RK2, float32 — fused kernel)

No exact solution — compares profiles and throughput head-to-head.

Saves:
  jaxfluids_profiles.png   — density/velocity/pressure at t=1.8 (N=512)
  jaxfluids_throughput.png — throughput vs N

Run from examples/warplabs_fluids/ inside the JaxFluids venv:
  source /root/venv-jf/bin/activate
  python benchmarks/shu_osher/bench_jaxfluids.py
"""

import json, sys, time, tempfile, shutil, gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/05_shock_density_interaction")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, cons_to_prim
from cases.shu_osher import ic as shu_ic, L, T_END, GAMMA, RHO_L, U_L, P_L
from benchmarks.jax_euler import JaxEuler1D

GRID_SIZES = [256, 512, 1024, 2048, 4096]
CFL_WARP   = 0.4
N_BENCH    = 1
N_WARM     = 1

_c_L   = float(np.sqrt(GAMMA * P_L / RHO_L))
A_MAX  = abs(U_L) + _c_L      # ≈ 4.57


def _load_jxf_templates():
    with open(JXF_EX / "shock_density_interaction.json") as f:
        case = json.load(f)
    with open(JXF_EX / "numerical_setup.json") as f:
        num_setup = json.load(f)
    return case, num_setup


def _patch_case(case_tmpl, N, run_dir):
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(run_dir)
    case["general"]["save_dt"]   = 999.0
    case_path = run_dir / "case.json"
    case_path.write_text(json.dumps(case))
    return case_path


def bench_jaxfluids(N, case_tmpl, num_path, tmp_dir):
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    import jax

    run_dir = tmp_dir / f"run_N{N}"
    run_dir.mkdir(exist_ok=True)
    case_path = _patch_case(case_tmpl, N, run_dir)

    input_mgr = InputManager(str(case_path), str(num_path))
    init_mgr  = InitializationManager(input_mgr)
    sim_mgr   = SimulationManager(input_mgr)
    buffers   = init_mgr.initialization()

    t0 = time.perf_counter()
    sim_mgr.simulate(buffers)
    jax.block_until_ready(buffers)
    t_warm = time.perf_counter() - t0
    print(f"    JaxFluids warmup: {t_warm:.2f}s", flush=True)

    times = []
    for _ in range(N_BENCH):
        buffers = init_mgr.initialization()
        t0 = time.perf_counter()
        sim_mgr.simulate(buffers)
        jax.block_until_ready(buffers)
        times.append(time.perf_counter() - t0)

    t_med = float(np.median(times))

    try:
        with open(num_path) as f:
            ns_json = json.load(f)
        cfl_jxf = ns_json["conservatives"]["time_integration"].get("CFL", 0.5)
    except Exception:
        cfl_jxf = 0.5
    dt_est  = cfl_jxf * (L / N) / A_MAX
    n_steps = max(1, int(round(T_END / dt_est)))
    mcell_s = N * n_steps / t_med / 1e6

    rho = u = p = None
    try:
        import glob, h5py
        case_name = "shock_density_interaction"
        h5_files  = sorted(glob.glob(str(run_dir / case_name / "domain" / "data_*.h5")))
        h5_final  = max(h5_files, key=lambda f: float(Path(f).stem.replace("data_", "")))
        with h5py.File(h5_final, "r") as f:
            rho = np.array(f["primitives/density"][0, 0, :])
            u   = np.array(f["primitives/velocity"][0, 0, :, 0])
            p   = np.array(f["primitives/pressure"][0, 0, :])
    except Exception as e:
        print(f"    [warn] could not read JaxFluids HDF5: {e}")

    return mcell_s, rho, u, p, n_steps, t_med


def bench_warp(N):
    import warp as wp
    dx = L / N
    Q0, x = shu_ic(N, GAMMA)
    solver = WarpEuler1D(N, dx, gamma=GAMMA, bc="outflow", device="cuda")
    solver.initialize(Q0)
    for _ in range(N_WARM):
        solver.initialize(Q0)
        n = solver.run(T_END, CFL_WARP)
    wp.synchronize()
    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0 = time.perf_counter()
        n  = solver.run(T_END, CFL_WARP)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    t_med   = float(np.median(times))
    mcell_s = N * n / t_med / 1e6
    rho, u, p = cons_to_prim(solver.state, GAMMA)
    del solver; gc.collect()
    return mcell_s, rho, u, p, n, t_med, x


def bench_jax_ref(N):
    import jax
    dx = L / N
    Q0, _ = shu_ic(N, GAMMA)
    gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
    with jax.default_device(gpu):
        solver = JaxEuler1D(N, dx, gamma=GAMMA, bc="outflow")
        solver.initialize(Q0)
        for _ in range(N_WARM):
            solver.initialize(Q0)
            n = solver.run(T_END, CFL_WARP)
        jax.block_until_ready(solver._Q)
        times = []
        for _ in range(N_BENCH):
            solver.initialize(Q0)
            t0 = time.perf_counter()
            n  = solver.run(T_END, CFL_WARP)
            jax.block_until_ready(solver._Q)
            times.append(time.perf_counter() - t0)
    t_med   = float(np.median(times))
    mcell_s = N * n / t_med / 1e6
    Q = np.asarray(solver._Q)
    rho, u, p = cons_to_prim(Q, GAMMA)
    del solver; gc.collect()
    return mcell_s, rho, u, p, n, t_med


def main():
    import warp as wp
    wp.init()

    try:
        case_tmpl, num_setup = _load_jxf_templates()
        jxf_ok = True
        print(f"[info] JaxFluids templates loaded from {JXF_EX}")
    except Exception as e:
        print(f"[warn] JaxFluids templates not found: {e}")
        jxf_ok = False

    tmp_dir = Path(tempfile.mkdtemp(prefix="jxf_shu_"))
    if jxf_ok:
        num_path = tmp_dir / "numerical_setup.json"
        num_path.write_text(json.dumps(num_setup))

    results = {k: {"N": [], "tp": [], "rho": None, "u": None, "p": None}
               for k in ["JaxFluids", "JAX CUDA (ours)", "Warp CUDA"]}
    x_ref = None

    for N in GRID_SIZES:
        print(f"\nN = {N:>6}", flush=True)

        if jxf_ok:
            try:
                tp, rho, u, p, ns, tm = bench_jaxfluids(N, case_tmpl, num_path, tmp_dir)
                results["JaxFluids"]["N"].append(N)
                results["JaxFluids"]["tp"].append(tp)
                if N == 512 and rho is not None:
                    results["JaxFluids"]["rho"] = rho
                    results["JaxFluids"]["u"]   = u
                    results["JaxFluids"]["p"]   = p
                print(f"  JaxFluids       {tp:8.2f} Mcell/s  ({ns} steps est, {tm:.2f}s)")
            except Exception as e:
                print(f"  JaxFluids       ERROR: {e}")

        try:
            tp, rho, u, p, ns, tm = bench_jax_ref(N)
            results["JAX CUDA (ours)"]["N"].append(N)
            results["JAX CUDA (ours)"]["tp"].append(tp)
            if N == 512:
                results["JAX CUDA (ours)"]["rho"] = rho
                results["JAX CUDA (ours)"]["u"]   = u
                results["JAX CUDA (ours)"]["p"]   = p
            print(f"  JAX CUDA (ours)  {tp:8.2f} Mcell/s  ({ns} steps, {tm:.2f}s)")
        except Exception as e:
            print(f"  JAX CUDA (ours)  ERROR: {e}")

        try:
            tp, rho, u, p, ns, tm, x = bench_warp(N)
            results["Warp CUDA"]["N"].append(N)
            results["Warp CUDA"]["tp"].append(tp)
            if N == 512:
                results["Warp CUDA"]["rho"] = rho
                results["Warp CUDA"]["u"]   = u
                results["Warp CUDA"]["p"]   = p
                x_ref = x
            print(f"  Warp CUDA        {tp:8.2f} Mcell/s  ({ns} steps, {tm:.2f}s)")
        except Exception as e:
            print(f"  Warp CUDA        ERROR: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── throughput table ──────────────────────────────────────────────────────
    print("\n-- Throughput (Mcell/s) -----------------------------------------")
    hdr = f"{'N':>7}" + "".join(f"  {n:>20}" for n in results)
    print(hdr); print("-" * len(hdr))
    for N in GRID_SIZES:
        row = f"{N:>7}"
        for d in results.values():
            if N in d["N"]:
                row += f"  {d['tp'][d['N'].index(N)]:>20.2f}"
            else:
                row += f"  {'--':>20}"
        print(row)

    # ── cross-solver profile agreement at N=512 ───────────────────────────────
    print("\n-- Profile agreement at N=512 (L1 difference between solvers) ----")
    print(f"{'Pair':<38}  {'L1(rho)':>10}  {'L1(u)':>10}  {'L1(p)':>10}")
    print("-" * 75)
    dx = L / 512
    solvers_with_data = [(k, v) for k, v in results.items() if v["rho"] is not None]
    for i, (n1, d1) in enumerate(solvers_with_data):
        for n2, d2 in solvers_with_data[i+1:]:
            pair = f"{n1}  vs  {n2}"
            e_r = float(np.mean(np.abs(d1["rho"] - d2["rho"])))
            e_u = float(np.mean(np.abs(d1["u"]   - d2["u"])))
            e_p = float(np.mean(np.abs(d1["p"]   - d2["p"])))
            print(f"{pair:<38}  {e_r:>10.3e}  {e_u:>10.3e}  {e_p:>10.3e}")

    # ── plots ─────────────────────────────────────────────────────────────────
    colors = {
        "JaxFluids":      "#e07b00",
        "JAX CUDA (ours)":"#d55e00",
        "Warp CUDA":      "#009e73",
    }
    prof_style = {
        "JaxFluids":      dict(ls="-",  lw=2.0),
        "JAX CUDA (ours)":dict(ls="--", lw=1.5),
        "Warp CUDA":      dict(ls="-.", lw=1.5),
    }
    tp_style = {
        "JaxFluids":      dict(marker="D", ls="-",  lw=1.8, ms=7),
        "JAX CUDA (ours)":dict(marker="o", ls="--", lw=1.8, ms=7),
        "Warp CUDA":      dict(marker="^", ls="-.", lw=1.8, ms=7),
    }

    # throughput
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in results.items():
        if not d["N"]: continue
        ax.plot(d["N"], d["tp"], color=colors[name], **tp_style[name], label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(
        "JaxFluids vs Warp CUDA  —  Shu-Osher throughput\n"
        "JaxFluids: WENO5-Z+HLLC+RK3 (f64)   |   Warp/JAX ours: WENO3+HLLC+RK2 (f32)",
        fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(GRID_SIZES)
    ax.set_xticklabels([f"{n:,}" for n in GRID_SIZES], rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    out = OUT / "jaxfluids_throughput.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {out}")

    # profiles at N=512
    if x_ref is None:
        _, x_ref = shu_ic(512, GAMMA)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Shu-Osher shock-density interaction  |  N=512  |  t={T_END}\n"
        "Mach-3 shock + sinusoidal density  ·  profiles at t=1.8",
        fontsize=10, fontweight="bold")
    field_names = ["density  ρ", "velocity  u", "pressure  p"]
    field_keys  = ["rho", "u", "p"]

    for name, d in results.items():
        if d["rho"] is None:
            continue
        x_plot = x_ref if name != "JaxFluids" else np.linspace(
            L / (2 * 512), L - L / (2 * 512), 512)
        for ax, fname, fkey in zip(axes, field_names, field_keys):
            ax.plot(x_plot, d[fkey], color=colors[name],
                    **prof_style[name], label=name, alpha=0.9)

    # IC overlay (density only)
    Q0_ic, x_ic = shu_ic(512, GAMMA)
    from warplabs_fluids.utils import cons_to_prim as c2p
    rho0, u0, p0 = c2p(Q0_ic, GAMMA)
    for ax, ic_val in zip(axes, [rho0, u0, p0]):
        ax.plot(x_ic, ic_val, color="0.72", lw=1.0, ls=":", label="t=0 (IC)", zorder=1)

    for ax, fname in zip(axes, field_names):
        ax.set_xlabel("x", fontsize=10)
        ax.set_title(fname, fontsize=11)
        ax.set_xlim(0, L)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)

    plt.tight_layout()
    out2 = OUT / "jaxfluids_profiles.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out2}")


if __name__ == "__main__":
    main()
