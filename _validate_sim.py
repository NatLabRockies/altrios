"""Quick validation of the simulation step wiring (not the full sweep).

Imports the corridor demo (reusing its network + substations), runs ONE coarse
(fast) optimization, applies line losses from the built substations, and runs a
single train sim to confirm the APIs + pydict energy keys work.
"""
import importlib.util
import time
import traceback
from pathlib import Path

import altrios as alt
from altrios import energization as en

_OUT = open(r"c:\Users\mbruchon\Documents\Repos\NREL\altrios-applied\_validate_out.txt", "w")


def _p(msg):
    print(msg, flush=True)
    _OUT.write(str(msg) + "\n")
    _OUT.flush()


demo_path = Path(
    r"c:\Users\mbruchon\Documents\Repos\NREL\altrios-applied\applications"
    r"\energization_optimization_corridor.py"
)
spec = importlib.util.spec_from_file_location("corr", demo_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

try:
    _p(">> coarse optimization for a quick plan ...")
    t0 = time.perf_counter()
    plan = en.energize_optimized(
        mod.network, mod.substations, objective="max_benefit", costs=mod.RULE_COSTS,
        budget=150, high_grade_benefit_weight=5.0, grade_threshold=0.02,
        substation_feed_distance_miles=30.0, min_energized_length_miles=2.0,
        uniform_power_limit_watts=50e6, structure_spacing_miles=5.0,  # coarse = fast
        corridor_width_miles=0.05, mip_gap=0.1, time_limit_seconds=120, tee=False,
    )
    _p(f"   plan: status={str(plan.status).split('.')[-1]} "
       f"miles={plan.energized_track_miles:.0f} "
       f"subs={len(plan.built_substation_ids)} ({time.perf_counter()-t0:.1f}s)")

    built = set(plan.built_substation_ids)
    built_subs = [s for s in mod.substations
                  if getattr(s, "existing", False) or s.id in built]
    _p(f">> apply_line_losses with {len(built_subs)} feeding substations ...")
    net_cat, meta = en.apply_line_losses(
        plan.network, built_subs, system_voltage_volts=25_000.0,
        electrification_type="ac_25kv_1x25", base_power_limit_watts=50e6,
    )
    _p(f"   line losses applied; meta substations n={len(meta)}")

    _p(">> energize_uniform (full-catenary network for first pass) ...")
    net_full = en.energize_uniform(
        mod.network, uniform_power_limit_watts=50e6)
    n_cat = sum(len(l.get("cat_power_limits") or [])
                for l in net_full.to_pydict()[1][1:])
    _p(f"   full-catenary intervals: {n_cat}")

    _p(">> build+walk one sim, then per-link energy aggregation ...")
    loc = alt.import_locations(str(mod.SIM_LOCATIONS_CSV))
    net_cat_cal = mod._calibrate_speeds(net_cat)
    ts = mod._build_and_walk(net_cat_cal, loc, "validate")
    dj, ej = mod._per_link_energy_from_sim(ts)
    fuel_total = ts.get_energy_fuel_joules(False)
    _p(f"   per-link diesel sum = {sum(dj.values())/1e9:.2f} GJ vs fuel getter "
       f"{fuel_total/1e9:.2f} GJ | per-link elec sum = {sum(ej.values())/1e9:.3f} GJ "
       f"| n_links diesel={len(dj)} elec={len(ej)}")
    _p(">> OK")
except Exception:
    _p(">> FAILED:\n" + traceback.format_exc())
finally:
    _OUT.close()
