"""Quick script to inspect JaxFluidsBuffers attribute names after a run."""
import json, sys, tempfile
from pathlib import Path

JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")
sys.path.insert(0, "/root/warp-examples-native/examples/warplabs_fluids")

from jaxfluids import InputManager, InitializationManager, SimulationManager

with open(JXF_EX / "sod.json")             as f: case = json.load(f)
with open(JXF_EX / "numerical_setup.json") as f: ns   = json.load(f)

tmp = Path(tempfile.mkdtemp())
case["domain"]["x"]["cells"] = 64
case["general"]["save_path"]  = str(tmp)
case["general"]["save_dt"]    = 999.0
(tmp / "case.json").write_text(json.dumps(case))
(tmp / "ns.json").write_text(json.dumps(ns))

im   = InputManager(str(tmp / "case.json"), str(tmp / "ns.json"))
init = InitializationManager(im)
sim  = SimulationManager(im)
buf  = init.initialization()
sim.simulate(buf)

import jax
jax.block_until_ready(buf)

print("\n=== JaxFluidsBuffers public attributes ===")
attrs = [a for a in dir(buf) if not a.startswith("_")]
for a in attrs:
    v = getattr(buf, a)
    shape = getattr(v, "shape", type(v).__name__)
    print(f"  {a}: {shape}")
