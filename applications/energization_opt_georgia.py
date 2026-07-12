"""End-to-end test/demo of the energization placement optimizer on the Heart of
Georgia network (real WGS84 lat/lon geometry).

Runs the MILP in structure-site (corridor) mode: every link is densified into
uniform along-track pieces, parallel tracks/sidings that share a location are
clustered into shared structure sites (masts/foundations paid once per site),
and the optimizer selects which pieces to electrify and which candidate
transmission substations to build -- subject to a capital budget and a
substation feeding/spacing constraint -- while prioritizing high-grade track.

The corridor clustering is purely geometric (lat/lon); it needs no EstTimeNet,
centerline, or siding labels, so it is safe on arbitrary imported networks.

Run:
    python applications/energization_opt_georgia.py
"""

from __future__ import annotations

import altrios as alt
from altrios.optimization import energization_placement as cp

MI_PER_M = 0.000621371

network = alt.Network.from_file(
    str(alt.resources_root() / "networks" / "Heart_of_Georgia" / "Heart_of_Georgia.yaml")
)
links = network.to_pydict()[1][1:]


def midpoint(link):
    hs = link.get("headings") or []
    if not hs:
        return None
    m = hs[len(hs) // 2]
    if m.get("lat") is None or m.get("lon") is None:
        return None
    return float(m["lat"]), float(m["lon"])


# Order links roughly along the (E-W) corridor by longitude, then place
# substations at fractional positions along it.
pts = [(midpoint(L), L["length_meters"]) for L in links]
pts = [(p, ln) for p, ln in pts if p is not None]
pts.sort(key=lambda t: t[0][1])  # sort by longitude
total_miles = sum(L["length_meters"] * MI_PER_M for L in links)


def at_fraction(frac: float):
    idx = min(len(pts) - 1, int(frac * (len(pts) - 1)))
    return pts[idx][0]


west_end = at_fraction(0.0)
substations = [
    # A pre-existing (free, always-built) substation at the west terminus.
    cp.Substation(id="west_terminus", lat=west_end[0], lon=west_end[1], existing=True),
    # Candidate substations to build along the corridor.
    cp.Substation(id="cand_1/3", lat=at_fraction(1 / 3)[0], lon=at_fraction(1 / 3)[1]),
    cp.Substation(id="cand_2/3", lat=at_fraction(2 / 3)[0], lon=at_fraction(2 / 3)[1]),
    cp.Substation(id="cand_east", lat=at_fraction(1.0)[0], lon=at_fraction(1.0)[1]),
]

costs = cp.EnergizationOptCosts(
    structure_cost_per_mile=2.5e6,  # $2.5M/mi shared structure (paid once per site)
    energization_cost_per_mile=0.5e6,   # $0.5M/mi marginal per electrified track
    substation_fixed_cost=20.0e6,   # $20M per built substation
)

# Budget allows only a fraction of the corridor to be electrified, forcing the
# optimizer to prioritize high-grade track and choose substations wisely.
budget = 150.0e6

inputs = cp.EnergizationPlacementInputs(
    network=network,
    substations=substations,
    costs=costs,
    budget=budget,
    uniform_power_limit_watts=50.0e6,
    grade_threshold=0.02,
    high_grade_benefit_weight=5.0,
    substation_feed_distance_miles=30.0,
    # Structure-site (corridor) mode: ~1 mi along-track pieces, parallel tracks
    # within ~80 m (0.05 mi) share a structure.
    structure_spacing_miles=1.0,
    corridor_width_miles=0.05,
    # Forbid tiny isolated energized slivers: any interior run must be >= 2 mi.
    min_energized_length_miles=2.0,
)

print(f"Network: {len(links)} links, {total_miles:.1f} track miles")
print(f"Budget: ${budget/1e6:.0f}M | structure ${costs.structure_cost_per_mile/1e6:.1f}M/mi"
      f" | energization ${costs.energization_cost_per_mile/1e6:.1f}M/mi"
      f" | substation ${costs.substation_fixed_cost/1e6:.0f}M each")

result = cp.optimize_energization_placement(inputs, mip_gap=0.01)

print("\n=== RESULT ===")
print("status:", result.status)
print("objective (priority-weighted miles):", round(result.objective_value or 0.0, 1))
print("energized links:", len(result.energized_link_idxs))
print("energized track miles:", round(result.energized_track_miles or 0.0, 1))
print("installed structure sites:", len(result.installed_structure_cell_ids))
print("built substations:", result.built_substation_ids)
print("total cost ($M):", round((result.total_cost or 0.0) / 1e6, 1))

# Confirm the returned network actually carries energization on energized links.
applied = sum(
    1
    for L in (result.network.to_pydict()[1][1:] if result.network else [])
    if L.get("cat_power_limits")
)
print("links with cat_power_limits applied:", applied)

# Demonstrate the corridor minimum-run-length constraint binds: re-solve with a
# large minimum and confirm the electrified footprint changes (longer, fewer runs).
big = cp.EnergizationPlacementInputs(
    **{**vars(inputs), "min_energized_length_miles": 25.0}
)
big_result = cp.optimize_energization_placement(big, mip_gap=0.01)
print("\n=== min run 25 mi (corridor) ===")
print("energized track miles:", round(big_result.energized_track_miles or 0.0, 1))
print("installed structure sites:", len(big_result.installed_structure_cell_ids))
print("objective:", round(big_result.objective_value or 0.0, 1))
