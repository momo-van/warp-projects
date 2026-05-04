"""
JaxFluids vs Warp WENO5-Z — Shu-Osher shock-density interaction.

Both solvers use WENO5-Z + HLLC + RK3.  JaxFluids runs float64; Warp runs float32.
No exact solution — compares profiles and throughput.

Saves (to benchmarks/shu_osher/):
  jaxfluids_profiles.png
  jaxfluids_throughput.png
  bench_jaxfluids_throughput.csv
  bench_jaxfluids_profiles_N512.csv

Run from examples/warplabs_fluids/ inside the JaxFluids venv:
  source /root/venv-jf/bin/activate
  python benchmarks/shu_osher/bench_jaxfluids.py
"""

import csv, json, sys, time, tempfile, shutil, gc
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

GRID_SIZES = [256, 512, 1024, 2048, 4096]
CFL_WARP   = 0.4
N_BENCH    = 1
N_WARM     = 1
_c_L       = float(np.sqrt(GAMMA * P_L / RHO_L))
A_MAX      = abs(U_L) + _c_L    # ≈ 4.57


def _patch_case(case_tmpl, N, run_dir):
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(run_dir)
    case["general"]["save_dt"]   = 999.0
    p = run_dir / "case.json"; p.write_text(json.dumps(case))
    return p


def bench_jaxfluids(N, case_tmpl, num_path, tmp_dir):
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    import jax
    run_dir = tmp_dir / f"run_N{N}"; run_dir.mkdir(exist_ok=True)
    cp  = _patch_case(case_tmpl, N, run_dir)
    im  = InputManager(str(cp), str(num_path))
    ini = InitializationManager(im); sim = SimulationManager(im)
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
        cfl_jxf = json.load(open(num_path))["conservatives"]["time_integration"].get("CFL", 0.5)
    except Exception:
        cfl_jxf = 0.5
    n_steps = max(1, int(round(T_END / (cfl_jxf * (L / N) / A_MAX))))
    mcell_s = N * n_steps / t_med / 1e6
    rho = u = p = None
    try:
        import glob, h5py
        h5s = sorted(glob.glob(str(run_dir / "shock_density_interaction" / "domain" / "data_*.h5")))
        with h5py.File(max(h5s, key=lambda f: float(Path(f).stem.replace("data_", ""))), "r") as f:
            rho = np.array(f["primitives/density"][0, 0, :])
            u   = np.array(f["primitives/velocity"][0, 0, :, 0])
            p   = np.array(f["primitives/pressure"][0, 0, :])
    except Exception as e:
        print(f"    [warn] HDF5 read failed: {e}")
    return mcell_s, rho, u, p, n_steps, t_med


def bench_warp(N):
    import warp as wp
    Q0, x = shu_ic(N, GAMMA)
    solver = WarpEuler1D(N, L/N, gamma=GAMMA, bc="outflow", device="cuda", scheme="weno5z-rk3")
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
        case_tmpl = json.load(open(JXF_EX / "shock_density_interaction.json"))
        num_setup = json.load(open(JXF_EX / "numerical_setup.json"))
        jxf_ok = True; print(f"[info] JaxFluids templates loaded from {JXF_EX}")
    except Exception as e:
        print(f"[warn] JaxFluids not found: {e}"); jxf_ok = False

    tmp_dir = Path(tempfile.mkdtemp(prefix="jxf_shu_"))
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

    print("\n-- Throughput (Mcell/s) --")
    hdr = f"{'N':>7}" + "".join(f"  {n:>30}" for n in res)
    print(hdr); print("-" * len(hdr))
    for N in GRID_SIZES:
        row = f"{N:>7}"
        for d in res.values():
            row += f"  {d['tp'][d['N'].index(N)]:>30.2f}" if N in d["N"] else f"  {'--':>30}"
        print(row)

    # Profile L1 comparison at N=512 (self: Warp vs JaxFluids)
    if x512 is None:
        _, x512 = shu_ic(512, GAMMA)
    if res["JaxFluids (WENO5-Z, f64)"]["rho"] is not None and res["Warp WENO5-Z (f32)"]["rho"] is not None:
        e_r = float(np.mean(np.abs(res["JaxFluids (WENO5-Z, f64)"]["rho"] - res["Warp WENO5-Z (f32)"]["rho"])))
        e_u = float(np.mean(np.abs(res["JaxFluids (WENO5-Z, f64)"]["u"]   - res["Warp WENO5-Z (f32)"]["u"])))
        e_p = float(np.mean(np.abs(res["JaxFluids (WENO5-Z, f64)"]["p"]   - res["Warp WENO5-Z (f32)"]["p"])))
        print(f"\n-- Profile agreement at N=512 (L1 diff JaxFluids vs Warp) --")
        print(f"  L1(rho)={e_r:.3e}  L1(u)={e_u:.3e}  L1(p)={e_p:.3e}")

    # ── save CSVs ─────────────────────────────────────────────────────────────
    with open(OUT / "bench_jaxfluids_throughput.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["solver", "N", "throughput_Mcells"])
        for name, d in res.items():
            for N, tp in zip(d["N"], d["tp"]): w.writerow([name, N, tp])
    print(f"\nSaved -> {OUT/'bench_jaxfluids_throughput.csv'}")

    if x512 is not None:
        N = 512
        with open(OUT / "bench_jaxfluids_profiles_N512.csv", "w", newline="") as f:
            w = csv.writer(f)
            hdr_cols = ["x"]
            for name in res:
                tag = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                hdr_cols += [f"rho_{tag}", f"u_{tag}", f"p_{tag}"]
            w.writerow(hdr_cols)
            for j in range(N):
                row = [x512[j]]
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
        "JaxFluids vs Warp  —  Shu-Osher throughput\n"
        "Both: WENO5-Z + HLLC + RK3  |  JaxFluids: f64  |  Warp: f32",
        fontsize=10)
    ax.legend(fontsize=11); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(GRID_SIZES)
    ax.set_xticklabels([f"{n:,}" for n in GRID_SIZES], rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "jaxfluids_throughput.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {OUT/'jaxfluids_throughput.png'}")

    # IC overlay
    Q0_ic, x_ic = shu_ic(512, GAMMA)
    from warplabs_fluids.utils import cons_to_prim as c2p
    rho0, u0, p0 = c2p(Q0_ic, GAMMA)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Shu-Osher shock-density interaction  |  N=512  |  t={T_END}\n"
        "WENO5-Z+HLLC+RK3  |  JaxFluids f64 vs Warp f32",
        fontsize=10, fontweight="bold")
    field_names = ["density  ρ", "velocity  u", "pressure  p"]
    for ax, fname in zip(axes, field_names):
        ax.plot(x_ic, [rho0, u0, p0][field_names.index(fname)],
                color="0.72", lw=1.0, ls=":", label="t=0 (IC)", zorder=1)
    for name, d in res.items():
        if d["rho"] is None: continue
        for ax, vals in zip(axes, [d["rho"], d["u"], d["p"]]):
            ax.plot(x512, vals, color=colors[name], lw=1.4, ls=styles[name][1:],
                    label=name, alpha=0.9)
    for ax, fname in zip(axes, field_names):
        ax.set_xlabel("x", fontsize=10); ax.set_title(fname, fontsize=11)
        ax.set_xlim(0, L); ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUT / "jaxfluids_profiles.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {OUT/'jaxfluids_profiles.png'}")


if __name__ == "__main__":
    main()
