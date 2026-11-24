# %%
"""
Catenary demo: run a dual-mode locomotive with and without overhead power, and
let the energization optimizer decide where the wire goes under a budget.

The same train is simulated three times over the Heart of Georgia network, which
carries real geography so substation reach can be measured:

1. no catenary -- the dual-mode locomotive runs on its engine throughout;
2. the whole network energized with ``energization.energize_uniform``;
3. catenary placed by the mixed-integer program in
   ``altrios.optimization.energization_placement`` under a capital budget, with
   candidate substations spaced along the route, via
   ``energization.energize_optimized``.

The script prints fuel and catenary energy for each case.
"""
import polars as pl

import altrios as alt
from altrios import energization
from altrios.optimization.energization_placement import EnergizationOptCosts, Substation

SAVE_INTERVAL = 60
# Power the overhead line can deliver to one train, applied to every energized segment.
CATENARY_POWER_LIMIT_WATTS = 6.0e6
# Spacing of candidate substations along the route, costs, and the capital budget.
SUBSTATION_SPACING_MILES = 10.0
COST_PER_MILE_USD = 2.0e6
SUBSTATION_COST_USD = 5.0e6
BUDGET_USD = 80.0e6
METERS_PER_MILE = 1609.344

# %%
print("Loading network, locations and rail vehicles")
network_dir = alt.resources_root() / "networks/Heart_of_Georgia"
network = alt.Network.from_file(network_dir / "Heart_of_Georgia.msgpack")
location_map = alt.import_locations(network_dir / "Heart_of_Georgia_locations.csv")
rail_vehicles = [
    alt.RailVehicle.from_file(alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml"),
    alt.RailVehicle.from_file(alt.resources_root() / "rolling_stock/Manifest_Empty.yaml"),
]
train_config = alt.TrainConfig(
    rail_vehicles=rail_vehicles,
    n_cars_by_type={"Manifest_Loaded": 30, "Manifest_Empty": 30},
    train_length_meters=None,
    train_mass_kilograms=None,
)


def build_train_sim() -> alt.SpeedLimitTrainSim:
    """Build one dual-mode locomotive hauling the manifest train.

    Returns:
        Train simulation from West_Terminal to East_Terminal.
    """
    loco = alt.Locomotive.default_dual_mode_loco()
    loco.set_save_interval(SAVE_INTERVAL)
    tsb = alt.TrainSimBuilder(
        train_id="0",
        origin_id="West_Terminal",
        destination_id="East_Terminal",
        train_config=train_config,
        loco_con=alt.Consist([loco], SAVE_INTERVAL),
    )
    train_sim = tsb.make_speed_limit_train_sim(
        location_map=location_map, save_interval=SAVE_INTERVAL
    )
    train_sim.set_save_interval(SAVE_INTERVAL)
    return train_sim


def simulate(net: alt.Network) -> tuple[dict[str, float], alt.EstTimeNet]:
    """Dispatch and walk the train over a network.

    Args:
        net: Network to simulate on.

    Returns:
        Fuel [GJ], catenary energy [GJ] and trip time [h], plus the
        estimated-time network.
    """
    train_sim = build_train_sim()
    est_time_net, _ = alt.make_est_times(train_sim, net)
    timed_path = next(
        iter(
            alt.run_dispatch(
                net, alt.SpeedLimitTrainSimVec([train_sim]), [est_time_net], False, False
            )
        )
    )
    train_sim.walk_timed_path(network=net, timed_path=timed_path)
    sim = train_sim.to_pydict()
    cat_int = sim["loco_con"]["loco_vec"][0]["loco_type"]["DualModeLoco"]["cat_int"]
    totals = {
        "fuel_GJ": sim["loco_con"]["state"]["energy_fuel_joules"] / 1e9,
        "catenary_GJ": cat_int["state"]["energy_elec_in_joules"] / 1e9,
        "trip_hours": sim["state"]["time_seconds"] / 3600.0,
    }
    return totals, est_time_net


# %% 1. No catenary
print("Simulating without catenary")
no_cat, est_time_net = simulate(network)

# %% 2. Everything energized
print("Simulating with the whole network energized")
all_cat, _ = simulate(energization.energize_uniform(network, CATENARY_POWER_LIMIT_WATTS))

# %% 3. Optimized placement under a budget
# Candidate substations every SUBSTATION_SPACING_MILES along the links the train
# actually traverses, taken in travel order from the est-time net.
links = network.to_pydict()[1]
path_link_idxs = (
    pl.DataFrame(est_time_net.to_pydict()["val"])
    .unnest("link_event")
    .filter(pl.col("est_type") == "Clear")
    .sort("time_sched")
    .get_column("link_idx")
    .to_list()
)
substations, miles_since_last = [], SUBSTATION_SPACING_MILES
for idx in dict.fromkeys(i for i in path_link_idxs if i > 0):
    if miles_since_last >= SUBSTATION_SPACING_MILES:
        substations.append(Substation(id=f"sub_{len(substations)}", link_idx=idx))
        miles_since_last = 0.0
    miles_since_last += links[idx]["length_meters"] / METERS_PER_MILE
print(f"Optimizing catenary placement with {len(substations)} candidate substations")
result = energization.energize_optimized(
    network,
    substations,
    CATENARY_POWER_LIMIT_WATTS,
    costs=EnergizationOptCosts(cost_per_mile=COST_PER_MILE_USD, substation_fixed_cost=SUBSTATION_COST_USD),
    budget=BUDGET_USD,
    est_time_nets=[est_time_net],
)
print(
    f"  solver status: {result.status}; energized {result.energized_track_miles:.1f} track-miles, "
    f"built {len(result.built_substation_ids)} substations, "
    f"spent ${result.total_cost / 1e6:.1f}M of ${BUDGET_USD / 1e6:.0f}M"
)
print("Simulating with the optimized placement")
opt_cat, _ = simulate(result.network)

# %% Summary
print()
print(f"{'case':<22}{'fuel [GJ]':>12}{'catenary [GJ]':>16}{'trip [h]':>10}")
for name, t in (("no catenary", no_cat), ("all energized", all_cat), ("optimized, budgeted", opt_cat)):
    print(f"{name:<22}{t['fuel_GJ']:>12.1f}{t['catenary_GJ']:>16.1f}{t['trip_hours']:>10.2f}")

assert no_cat["catenary_GJ"] == 0.0
assert all_cat["fuel_GJ"] < no_cat["fuel_GJ"]
assert no_cat["fuel_GJ"] >= opt_cat["fuel_GJ"] >= all_cat["fuel_GJ"]
assert result.total_cost <= BUDGET_USD * (1 + 1e-6)
