"""
JaxFluids vs Warp WENO5-Z — Sod shock tube head-to-head.

Both solvers use WENO5-Z + HLLC + RK3.  JaxFluids runs float64; Warp runs float32.
Measures throughput (Mcell/s) and L1 accuracy vs exact Riemann solution.

Saves (to benchmarks/sod/):
  jaxfluids_profiles.png      — density/velocity/pressure at N=512
  jaxfluids_throughput.png    — throughput vs N
  bench_jaxfluids_throughput.csv
  bench_jaxfluids_profiles_N512.csv
  bench_jaxfluids_accuracy.csv

Run from examples/warplabs_fluids/ inside the JaxFluids venv:
  source /root/venv-jf/bin/activate
  python benchmarks/sod/bench_jaxfluids.py
"""

import csv, json, sys, time, tempfile, shutil, gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, cons_to_prim, l1_error
from cases.sod import ic as sod_ic, exact as sod_exact

GRID_SIZES = [256, 512, 1024, 2048, 4096]
GAMMA      = 1.4
T_END      = 0.2
CFL_WARP   = 0.4
N_BENCH    = 1
N_WARM     = 1
_A_MAX     = float(np.sqrt(GAMMA))   # Sod: a_max ≈ sqrt(γ) at left state (p=1,rho=1)


def _patch_case(case_tmpl, N, run_dir):
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(run_dir)
    case["general"]["save_dt"]   = 999.0
    p = run_dir / "case.json"
    p.write_text(json.dumps(case))
    return p


def bench_jaxfluids(N, case_tmpl, num_path, tmp_dir):
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    import jax
    run_dir = tmp_dir / f"run_N{N}"; run_dir.mkdir(exist_ok=True)
    case_path = _patch_case(case_tmpl, N, run_dir)
    im  = InputManager(str(case_path), str(num_path))
    ini = InitializationManager(im)
    sim = SimulationManager(im)
    buf = ini.initialization()
    t0  = time.perf_counter(); sim.simulate(buf); jax.block_until_ready(buf)
    print(f"    JaxFluids warmup: {time.perf_counter()-t0:.2f}s", flush=True)
    times = []
    for _ in range(N_BENCH):
        buf = ini.initialization(); t0 = time.perf_counter()
        sim.simulate(buf); jax.block_until_ready(buf)
        times.append(time.perf_counter() - t0)
    t_med = float(np.median(times))
    try:
        ns = json.load(open(num_path))
        cfl_jxf = ns["conservatives"]["time_integration"].get("CFL", 0.5)
    except Exception:
        cfl_jxf = 0.5
    n_steps = max(1, int(round(T_END / (cfl_jxf * (1.0/N) / _A_MAX))))
    mcell_s = N * n_steps / t_med / 1e6
    rho = u = p = None
    try:
        import glob, h5py
        h5s = sorted(glob.glob(str(run_dir / "sod" / "domain" / "data_*.h5")))
        with h5py.File(max(h5s, key=lambda f: float(Path(f).stem.replace("data_", ""))), "r") as f:
            rho = np.array(f["primitives/density"][0, 0, :])
            u   = np.array(f["primitives/velocity"][0, 0, :, 0])
            p   = np.array(f["primitives/pressure"][0, 0, :])
    except Exception as e:
        print(f"    [warn] HDF5 read failed: {e}")
    return mcell_s, rho, u, p, n_steps, t_med


def bench_warp(N):
    import warp as wp
    Q0, x = sod_ic(N, GAMMA)
    solver = WarpEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow", device="cuda", scheme="weno5z-rk3")
    solver.initialize(Q0)
    for _ in range(N_WARM):
        solver.initialize(Q0); n = solver.run(T_END, CFL_WARP)
    wp.synchronize()
    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0); t0 = time.perf_counter()
        n = solver.run(T_END, CFL_WARP); wp.synchronize()
        times.append(time.perf_counter() - t0)
    t_med   = float(np.median(times))
    mcell_s = N * n / t_med / 1e6
    rho, u, p = cons_to_prim(solver.state, GAMMA)
    del solver; gc.collect()
    return mcell_s, rho, u, p, n, t_med, x


def main():
    import warp as wp
    wp.init()

    try:
        case_tmpl = json.load(open(JXF_EX / "sod.json"))
        num_setup = json.load(open(JXF_EX / "numerical_setup.json"))
        jxf_ok = True; print(f"[info] JaxFluids templates loaded from {JXF_EX}")
    except Exception as e:
        print(f"[warn] JaxFluids not found: {e}"); jxf_ok = False

    tmp_dir = Path(tempfile.mkdtemp(prefix="jxf_sod_"))
    if jxf_ok:
        num_path = tmp_dir / "numerical_setup.json"
        num_path.write_text(json.dumps(num_setup))

    res = {k: {"N": [], "tp": [], "rho": None, "u": None, "p": None}
           for k in ["JaxFluids (WENO5-Z, f64)", "Warp WENO5-Z (f32)"]}
    x512 = None

    for N in GRID_SIZES:
        print(f"\nN = {N:>6}", flush=True)

        if jxf_ok:
            try:
                tp, rho, u, p, ns, tm = bench_jaxfluids(N, case_tmpl, num_path, tmp_dir)
                res["JaxFluids (WENO5-Z, f64)"]["N"].append(N)
                res["JaxFluids (WENO5-Z, f64)"]["tp"].append(tp)
                if N == 512 and rho is not None:
                    res["JaxFluids (WENO5-Z, f64)"]["rho"] = rho
                    res["JaxFluids (WENO5-Z, f64)"]["u"]   = u
                    res["JaxFluids (WENO5-Z, f64)"]["p"]   = p
                print(f"  JaxFluids       {tp:8.2f} Mcell/s  ({ns} steps est, {tm:.2f}s)")
            except Exception as e:
                print(f"  JaxFluids       ERROR: {e}")

        try:
            tp, rho, u, p, ns, tm, x = bench_warp(N)
            res["Warp WENO5-Z (f32)"]["N"].append(N)
            res["Warp WENO5-Z (f32)"]["tp"].append(tp)
            if N == 512:
                res["Warp WENO5-Z (f32)"]["rho"] = rho
                res["Warp WENO5-Z (f32)"]["u"]   = u
                res["Warp WENO5-Z (f32)"]["p"]   = p
                x512 = x
            print(f"  Warp WENO5-Z    {tp:8.2f} Mcell/s  ({ns} steps, {tm:.2f}s)")
        except Exception as e:
            print(f"  Warp WENO5-Z    ERROR: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── accuracy vs exact at N=512 ────────────────────────────────────────────
    if x512 is None:
        _, x512 = sod_ic(512, GAMMA)
    rho_ex, u_ex, p_ex = sod_exact(T_END, x512, GAMMA)
    dx = 1.0 / 512
    print("\n-- Accuracy vs exact Riemann (N=512, t=0.2) --")
    print(f"{'Solver':<30}  {'L1(rho)':>10}  {'L1(u)':>10}  {'L1(p)':>10}")
    print("-" * 68)
    acc_rows = []
    for name, d in res.items():
        if d["rho"] is None:
            print(f"{name:<30}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")
            continue
        e_r = l1_error(d["rho"], rho_ex, dx)
        e_u = l1_error(d["u"],   u_ex,   dx)
        e_p = l1_error(d["p"],   p_ex,   dx)
        acc_rows.append([name, 512, e_r, e_u, e_p])
        print(f"{name:<30}  {e_r:>10.3e}  {e_u:>10.3e}  {e_p:>10.3e}")

    # ── throughput table ──────────────────────────────────────────────────────
    print("\n-- Throughput (Mcell/s) --")
    hdr = f"{'N':>7}" + "".join(f"  {n:>30}" for n in res)
    print(hdr); print("-" * len(hdr))
    for N in GRID_SIZES:
        row = f"{N:>7}"
        for d in res.values():
            row += f"  {d['tp'][d['N'].index(N)]:>30.2f}" if N in d["N"] else f"  {'--':>30}"
        print(row)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    with open(OUT / "bench_jaxfluids_throughput.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "N", "throughput_Mcells"])
        for name, d in res.items():
            for N, tp in zip(d["N"], d["tp"]):
                w.writerow([name, N, tp])
    print(f"\nSaved -> {OUT/'bench_jaxfluids_throughput.csv'}")

    with open(OUT / "bench_jaxfluids_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "N", "L1_rho", "L1_u", "L1_p"])
        for row in acc_rows:
            w.writerow(row)
    print(f"Saved -> {OUT/'bench_jaxfluids_accuracy.csv'}")

    if x512 is not None:
        with open(OUT / "bench_jaxfluids_profiles_N512.csv", "w", newline="") as f:
            w = csv.writer(f)
            hdr_cols = ["x", "rho_exact", "u_exact", "p_exact"]
            for name in res:
                tag = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                hdr_cols += [f"rho_{tag}", f"u_{tag}", f"p_{tag}"]
            w.writerow(hdr_cols)
            for j in range(512):
                row = [x512[j], rho_ex[j], u_ex[j], p_ex[j]]
                for d in res.values():
                    if d["rho"] is not None:
                        row += [d["rho"][j], d["u"][j], d["p"][j]]
                    else:
                        row += ["", "", ""]
                w.writerow(row)
        print(f"Saved -> {OUT/'bench_jaxfluids_profiles_N512.csv'}")

    # ── plots ─────────────────────────────────────────────────────────────────
    colors = {"JaxFluids (WENO5-Z, f64)": "#e07b00", "Warp WENO5-Z (f32)": "#009e73"}
    styles = {"JaxFluids (WENO5-Z, f64)": "D-",      "Warp WENO5-Z (f32)": "^-"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in res.items():
        if not d["N"]: continue
        m, ls = styles[name][0], styles[name][1:]
        ax.plot(d["N"], d["tp"], marker=m, ls=ls, color=colors[name], lw=1.8, ms=7, label=name)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(
        "JaxFluids vs Warp  —  Sod shock tube throughput\n"
        "Both: WENO5-Z + HLLC + RK3  |  JaxFluids: f64  |  Warp: f32",
        fontsize=10)
    ax.legend(fontsize=11); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(GRID_SIZES)
    ax.set_xticklabels([f"{n:,}" for n in GRID_SIZES], rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "jaxfluids_throughput.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {OUT/'jaxfluids_throughput.png'}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Sod shock tube  |  N=512  |  t={T_END}\n"
        "WENO5-Z+HLLC+RK3  |  JaxFluids f64 vs Warp f32",
        fontsize=10, fontweight="bold")
    fields = ["density  ρ", "velocity  u", "pressure  p"]
    exact_v = [rho_ex, u_ex, p_ex]
    for ax, fname, ev in zip(axes, fields, exact_v):
        ax.plot(x512, ev, "k-", lw=2.0, label="exact Riemann", zorder=5)
        for name, d in res.items():
            if d["rho"] is None: continue
            vals = [d["rho"], d["u"], d["p"]]
            ax.plot(x512, vals[fields.index(fname)], color=colors[name],
                    lw=1.4, ls=styles[name][1:], label=name, alpha=0.85)
        ax.set_xlabel("x"); ax.set_title(fname)
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUT / "jaxfluids_profiles.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {OUT/'jaxfluids_profiles.png'}")


if __name__ == "__main__":
    main()
