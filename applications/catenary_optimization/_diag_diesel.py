"""TEMP diagnostic: reproduce the diesel _variant_metrics_df failure in isolation
using the already-cached conventional/no_catenary sim bundle."""
import os
import traceback

os.environ["ENERGIZATION_RESULTS_DIR"] = os.path.join(
    os.path.dirname(__file__),
    "outputs", "energization_results", "sweep_20260801_001256")

import energization_optimization_corridor as m  # noqa: E402

bundle = m._load_sim_bundle("conventional", "no_catenary")
print("bundle:", None if bundle is None else f"{len(bundle['sims'])} sims, "
      f"keys={sorted(bundle.keys())}")

for energy_lvl in ("low", "med", "high"):
    try:
        df = m._variant_metrics_df(
            "diesel", "Diesel",
            {"Diesel": lambda _y: m._defaults.DIESEL_LOCO_COST_USD}, bundle,
            None, [], {m.GRID_REGION_EAST: 1.0}, arch="none",
            demand_lvl="central", demand_mult=1.0,
            infra_lvl="na", infra_mult=1.0, energy_lvl=energy_lvl)
        print(f"OK [{energy_lvl}] rows={df.height} cols={df.columns[:6]}...")
    except Exception:
        print(f"FAILED [{energy_lvl}]:")
        traceback.print_exc()
        break
