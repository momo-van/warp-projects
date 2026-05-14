"""
Precision comparison — Sod shock tube.
Runs JaxFluids fp32 and fp64 (in isolated subprocesses) plus Warp fp32.
All three use WENO5-Z+HLLC+RK3 at N=512.  Accuracy vs exact Riemann.

Saves:
  precision_profiles_N512.csv
  precision_comparison.png

Run:
  source /root/venv-jf/bin/activate
  python benchmarks/sod/bench_precision.py
"""

import argparse, csv, gc, json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")
sys.path.insert(0, str(ROOT))

GAMMA = 1.4
T_END = 0.2
CFL   = 0.4
N     = 512

COLORS = {
    "JaxFluids (WENO5-Z, f32)": "#1a7abd",
    "JaxFluids (WENO5-Z, f64)": "#e07b00",
    "Warp WENO5-Z (f32)":       "#009e73",
}
STYLES = {
    "JaxFluids (WENO5-Z, f32)": ("s", "--"),
    "JaxFluids (WENO5-Z, f64)": ("D", "-"),
    "Warp WENO5-Z (f32)":       ("^", "-"),
}


# ── subprocess helper (called internally, not by user) ────────────────────────

def _run_jxf_subprocess(prec, outfile):
    import jax
    jax.config.update("jax_enable_x64", prec == "f64")
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    import glob, h5py

    case_tmpl = json.load(open(JXF_EX / "sod.json"))
    num_setup = json.load(open(JXF_EX / "numerical_setup.json"))

    with tempfile.TemporaryDirectory(prefix=f"prec_sod_{prec}_") as td:
        td = Path(td)
        case = json.loads(json.dumps(case_tmpl))
        case["domain"]["x"]["cells"] = N
        case["general"]["save_path"] = str(td)
        case["general"]["save_dt"]   = 999.0
        (td / "case.json").write_text(json.dumps(case))
        (td / "numerical_setup.json").write_text(json.dumps(num_setup))
        im  = InputManager(str(td / "case.json"), str(td / "numerical_setup.json"))
        buf = InitializationManager(im).initialization()
        SimulationManager(im).simulate(buf)
        jax.block_until_ready(buf)
        h5s = sorted(glob.glob(str(td / "sod" / "domain" / "data_*.h5")))
        with h5py.File(max(h5s, key=lambda f: float(Path(f).stem.replace("data_", ""))), "r") as f:
            rho = np.array(f["primitives/density"][0, 0, :])
            u   = np.array(f["primitives/velocity"][0, 0, :, 0])
            p   = np.array(f["primitives/pressure"][0, 0, :])

    np.savez(outfile, rho=rho, u=u, p=p)
    print(f"[jxf-{prec}] saved to {outfile}", flush=True)


def _spawn_jxf(prec):
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    try:
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        r = subprocess.run(
            [sys.executable, __file__, "--_jxf-prec", prec, "--_out", tmp.name],
            env=env, timeout=600)
        if r.returncode != 0:
            print(f"  [warn] JaxFluids {prec} subprocess failed (rc={r.returncode})")
            return None
        d = np.load(tmp.name)
        return d["rho"], d["u"], d["p"]
    except Exception as e:
        print(f"  [warn] JaxFluids {prec}: {e}")
        return None
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import warp as wp
    wp.init()
    from warplabs_fluids import WarpEuler1D, cons_to_prim, l1_error
    from cases.sod import ic as sod_ic, exact as sod_exact

    _, x = sod_ic(N, GAMMA)
    rho_ex, u_ex, p_ex = sod_exact(T_END, x, GAMMA)
    dx = 1.0 / N
    res = {}

    print("Running JaxFluids fp32 (subprocess)...", flush=True)
    out = _spawn_jxf("f32")
    if out:
        res["JaxFluids (WENO5-Z, f32)"] = out
        print(f"  L1(rho)={l1_error(out[0], rho_ex, dx):.3e}")

    print("Running JaxFluids fp64 (subprocess)...", flush=True)
    out = _spawn_jxf("f64")
    if out:
        res["JaxFluids (WENO5-Z, f64)"] = out
        print(f"  L1(rho)={l1_error(out[0], rho_ex, dx):.3e}")

    print("Running Warp fp32...", flush=True)
    try:
        Q0, x = sod_ic(N, GAMMA)
        solver = WarpEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow", device="cuda", scheme="weno5z-rk3")
        solver.initialize(Q0); solver.run(T_END, CFL); wp.synchronize()
        rho, u, p = cons_to_prim(solver.state, GAMMA)
        del solver; gc.collect()
        res["Warp WENO5-Z (f32)"] = (rho, u, p)
        print(f"  L1(rho)={l1_error(rho, rho_ex, dx):.3e}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\n-- Accuracy vs exact Riemann (N={N}, t={T_END}) --")
    print(f"{'Solver':<35}  {'L1(rho)':>10}  {'L1(u)':>10}  {'L1(p)':>10}")
    for name, (rho, u, p) in res.items():
        print(f"  {name:<33}  {l1_error(rho,rho_ex,dx):>10.3e}"
              f"  {l1_error(u,u_ex,dx):>10.3e}  {l1_error(p,p_ex,dx):>10.3e}")

    # save CSV
    with open(OUT / "precision_profiles_N512.csv", "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["x", "rho_exact", "u_exact", "p_exact"]
        for name in res:
            tag = name.replace(" ", "_").replace("(","").replace(")","").replace(",","")
            hdr += [f"rho_{tag}", f"u_{tag}", f"p_{tag}"]
        w.writerow(hdr)
        for j in range(N):
            row = [x[j], rho_ex[j], u_ex[j], p_ex[j]]
            for rho, u, p in res.values():
                row += [rho[j], u[j], p[j]]
            w.writerow(row)
    print(f"\nSaved -> {OUT/'precision_profiles_N512.csv'}")

    # plot
    fields = ["density  rho", "velocity  u", "pressure  p"]
    exact_v = [rho_ex, u_ex, p_ex]
    fkeys   = ["rho", "u", "p"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Sod shock tube  |  N={N}  |  t={T_END}  |  WENO5-Z+HLLC+RK3\n"
        "Precision comparison: JaxFluids fp32 vs fp64 vs Warp fp32",
        fontsize=10, fontweight="bold")
    for ax, fname, ev, fk in zip(axes, fields, exact_v, fkeys):
        ax.plot(x, ev, "k-", lw=2.0, label="exact Riemann", zorder=5)
        for name, (rho, u, p) in res.items():
            vals = {"rho": rho, "u": u, "p": p}
            m, ls = STYLES.get(name, ("o", "-"))
            ax.plot(x, vals[fk], color=COLORS.get(name, "#555"),
                    lw=1.4, ls=ls, label=name, alpha=0.85)
        ax.set_xlabel("x"); ax.set_title(fname)
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUT / "precision_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT/'precision_comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_jxf-prec", dest="jxf_prec", default=None)
    parser.add_argument("--_out", dest="out", default=None)
    args, _ = parser.parse_known_args()
    if args.jxf_prec:
        _run_jxf_subprocess(args.jxf_prec, args.out)
    else:
        main()
