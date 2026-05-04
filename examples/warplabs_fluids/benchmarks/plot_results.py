"""
Plot throughput and memory from the last WSL2 benchmark run.
Saves two independent PNGs to benchmarks/.

Run from examples/warplabs_fluids/:
  python benchmarks/plot_results.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "benchmarks"

# ── data: WSL2 native ext4, fused kernel, RTX 5000 Ada, 2026-05-03 ───────────
CELLS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

throughput = {
    "JAX CPU":   [9.45,  12.37, 14.12, 12.74, 15.02, 17.64,  22.49,  25.84,  31.84],
    "Warp CPU":  [3.35,   4.39,  4.99,  5.41,  5.75,  5.98,   5.88,   6.11,   6.14],
    "JAX CUDA":  [1.96,   3.93,  7.40, 13.81, 30.76, 57.82, 119.38, 227.72, 298.08],
    "Warp CUDA": [5.07,   9.27, 19.98, 37.62, 81.68,130.81, 211.02, 382.13, 482.74],
}

# nvidia-smi delta in MiB; None = below 1 MiB resolution (actual < 1 MiB)
memory_mib = {
    "JAX CUDA":  [2.0, None, None, None, None, None,  4.0,  8.0, 16.0],
    "Warp CUDA": [34., 32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.],
}

# theoretical minimum: 2 arrays × 3 vars × N cells × 4 bytes
theory_mib = [2 * 3 * n * 4 / (1024**2) for n in CELLS]

style = {
    "JAX CPU":   dict(color="#e07b00", marker="o", ls="--", lw=1.8, ms=7),
    "Warp CPU":  dict(color="#0072b2", marker="s", ls="--", lw=1.8, ms=7),
    "JAX CUDA":  dict(color="#d55e00", marker="D", ls="-",  lw=1.8, ms=7),
    "Warp CUDA": dict(color="#009e73", marker="^", ls="-",  lw=1.8, ms=7),
}

subtitle = (
    "1-D Euler  ·  WENO3-HLLC-RK2 (fused)  ·  200 steps, median of 5 runs  ·  "
    "WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1  ·  JAX 0.6.2"
)

# ── throughput ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle(subtitle, fontsize=8, color="0.4")

for name, tp in throughput.items():
    s = style[name]
    ax.plot(CELLS, tp, **s, label=name)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("Number of cells", fontsize=12)
ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
ax.set_title("Throughput  vs  number of cells", fontsize=13, pad=10)
ax.legend(fontsize=11)
ax.grid(True, which="both", lw=0.4, alpha=0.5)
ax.set_xticks(CELLS)
ax.set_xticklabels([f"{n:,}" for n in CELLS], rotation=35, ha="right", fontsize=8)

fig.tight_layout()
out = OUT / "plot_throughput.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)

# ── memory ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle(subtitle, fontsize=8, color="0.4")

for name, mems in memory_mib.items():
    s = style[name]
    # split into measured points vs below-resolution points
    N_meas, m_meas = [], []
    N_low,  m_low  = [], []
    for n, m in zip(CELLS, mems):
        if m is not None:
            N_meas.append(n); m_meas.append(m)
        else:
            N_low.append(n);  m_low.append(0.5)   # plot at 0.5 MiB as upper-bound marker

    ax.plot(N_meas, m_meas, **s, label=f"{name} (measured)")
    if N_low:
        ax.scatter(N_low, m_low, marker="v", color=s["color"],
                   s=40, zorder=5, label=f"{name} (< 1 MiB)")

ax.plot(CELLS, theory_mib, color="0.55", ls=":", lw=1.4, marker=".",
        ms=5, label="theory  2 arr × 3 var × N × 4 B")

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("Number of cells", fontsize=12)
ax.set_ylabel("Peak GPU memory  Δ (MiB)", fontsize=12)
ax.set_title(
    "GPU memory footprint  vs  number of cells\n"
    "(JAX with XLA_PYTHON_CLIENT_PREALLOCATE=false)",
    fontsize=12, pad=10,
)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, which="both", lw=0.4, alpha=0.5)
ax.set_xticks(CELLS)
ax.set_xticklabels([f"{n:,}" for n in CELLS], rotation=35, ha="right", fontsize=8)

fig.tight_layout()
out = OUT / "plot_memory.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)
