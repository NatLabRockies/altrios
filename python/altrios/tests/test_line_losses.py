"""Tests for AC catenary line-loss modeling (energization line losses)."""

from pathlib import Path

import altrios as alt
from altrios import energization as en
from altrios.optimization.energization_placement import Substation

NETWORK_FILE = (
    alt.resources_root()
    / "networks"
    / "Heart_of_Georgia"
    / "Heart_of_Georgia.msgpack"
)

SAVE_INTERVAL = 60


def _two_substations(network):
    """Place two substations at spread-out mainline positions on the network."""
    d = network.to_pydict()
    arc_start, mainline_set, _project = en._build_corridor_arc(d, 200.0)
    link_by_idx = {l["idx_curr"]: l for l in d[1][1:]}
    # Choose the mainline links nearest to 25% and 75% of the corridor length.
    ordered = sorted(mainline_set, key=lambda i: arc_start[i])
    max_arc = max(arc_start.values())

    def _mid(link):
        hs = link.get("headings") or []
        if not hs:
            return None
        m = hs[len(hs) // 2]
        return m.get("lat"), m.get("lon")

    def _pick(frac):
        target = frac * max_arc
        best = min(ordered, key=lambda i: abs(arc_start[i] - target))
        return _mid(link_by_idx[best])

    a = _pick(0.25)
    b = _pick(0.75)
    return [
        Substation(id="subA", lat=a[0], lon=a[1], max_power_watts=15e6,
                   grid_emissions_region="SRSO"),
        Substation(id="subB", lat=b[0], lon=b[1], max_power_watts=15e6,
                   grid_emissions_region="SRSO"),
    ]


def test_apply_line_losses_stamps_and_derates():
    network = alt.Network.from_file(str(NETWORK_FILE))
    net_e = en.energize_uniform(network, uniform_power_limit_watts=10e6)
    subs = _two_substations(net_e)

    net_ll, meta = en.apply_line_losses(
        net_e, subs, system_voltage_volts=25_000.0,
        electrification_type="ac_25kv_1x25", base_power_limit_watts=10e6,
    )
    d = net_ll.to_pydict()
    assert d[0]["system_voltage_volts"] == 25_000.0
    assert set(meta) == {"subA", "subB"}

    n_intervals = 0
    n_with_district = 0
    n_two_sided = 0
    max_power_seen = 0.0
    for link in d[1][1:]:
        for c in (link.get("cat_power_limits") or []):
            n_intervals += 1
            assert c["resistance_to_source_ohms"] >= 0.0
            assert c["voltage_volts"] == 25_000.0
            assert c["power_limit_watts"] <= 10e6 + 1.0  # derated from base
            max_power_seen = max(max_power_seen, c["power_limit_watts"])
            if c.get("district_id"):
                n_with_district += 1
            frac = c.get("feed_fraction")
            if c.get("district_id_secondary") is not None and frac is not None:
                n_two_sided += 1
                assert 0.5 <= frac <= 1.0  # primary is the nearer -> >= half
    assert n_intervals > 0
    assert n_with_district > 0
    assert n_two_sided > 0  # interior zones are fed from both sides


def test_higher_voltage_reduces_derating():
    """50 kV should deliver more power than 25 kV (loss scales as 1/V**2)."""
    network = alt.Network.from_file(str(NETWORK_FILE))
    net_e = en.energize_uniform(network, uniform_power_limit_watts=10e6)
    subs = _two_substations(net_e)

    def _min_power(volts):
        net_ll, _ = en.apply_line_losses(
            net_e, subs, system_voltage_volts=volts,
            electrification_type="ac_25kv_1x25", base_power_limit_watts=10e6,
        )
        d = net_ll.to_pydict()
        return min(
            c["power_limit_watts"]
            for link in d[1][1:]
            for c in (link.get("cat_power_limits") or [])
        )

    p25 = _min_power(25_000.0)
    p50 = _min_power(50_000.0)
    assert p50 > p25


def _run_catenary_sim():
    rail_vehicle_loaded = alt.RailVehicle.from_file(
        alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml"
    )
    rail_vehicle_empty = alt.RailVehicle.from_file(
        alt.resources_root() / "rolling_stock/Manifest_Empty.yaml"
    )
    train_config = alt.TrainConfig(
        rail_vehicles=[rail_vehicle_loaded, rail_vehicle_empty],
        n_cars_by_type={"Manifest_Loaded": 30, "Manifest_Empty": 30},
        train_length_meters=None,
        train_mass_kilograms=None,
    )
    dml = alt.Locomotive.default_dual_mode_loco()
    dml.set_save_interval(SAVE_INTERVAL)
    loco_con = alt.Consist([dml.copy()], SAVE_INTERVAL)
    tsb = alt.TrainSimBuilder(
        train_id="0", origin_id="Superior", destination_id="Minneapolis",
        train_config=train_config, loco_con=loco_con,
    )
    location_map = alt.import_locations(
        alt.resources_root() / "networks/default_locations.csv"
    )
    network = alt.Network.from_file(
        alt.resources_root() / "networks/Taconite-NoBalloon.msgpack"
    )
    # Taconite lacks lat/lon, so stamp line-loss params directly for the runtime test.
    network_cat = en.energize_uniform(network, 50e6)
    d = network_cat.to_pydict()
    for link in d[1][1:]:
        for c in (link.get("cat_power_limits") or []):
            c["resistance_to_source_ohms"] = 2.0
            c["reactance_to_source_ohms"] = 0.0
            c["voltage_volts"] = 25_000.0
            c["district_id"] = "subMain"
    d[0]["system_voltage_volts"] = 25_000.0
    network_cat = alt.Network.from_pydict(d)

    train_sim = tsb.make_speed_limit_train_sim(
        location_map=location_map, save_interval=SAVE_INTERVAL
    )
    train_sim.set_save_interval(SAVE_INTERVAL)
    est_time_net, _ = alt.make_est_times(train_sim, network_cat)
    timed_link_path = next(iter(alt.run_dispatch(
        network_cat, alt.SpeedLimitTrainSimVec([train_sim]),
        [est_time_net], False, False)))
    train_sim.walk_timed_path(network=network_cat, timed_path=timed_link_path)
    return train_sim


def test_runtime_line_loss_and_energy_balance():
    train_sim = _run_catenary_sim()
    ts = train_sim.to_pydict()
    ch = ts["loco_con"]["loco_vec"][0]["loco_type"]["DualModeLoco"]["cat_int"]["history"]

    # Per-step loss matches R * (P/V)**2.
    pin = ch["pwr_elec_in_watts"]
    pll = ch["pwr_line_loss_watts"]
    R, V = 2.0, 25_000.0
    for p, loss in zip(pin, pll):
        assert abs(loss - R * (p / V) ** 2) <= max(1.0, 1e-6 * abs(loss))

    # Cumulative loss energy is positive and a plausible fraction of catenary energy.
    meta = {"subMain": {"grid_emissions_region": "MROW", "max_power_watts": 50e6}}
    df = en.compute_line_losses(train_sim, meta)
    total = df.filter(df["district_id"] == "__total__")
    loss_j = total["line_loss_energy_joules"][0]
    cat_j = total["catenary_energy_joules"][0]
    assert loss_j > 0.0
    assert 0.0 < loss_j / cat_j < 0.2

    # The total equals the cumulative history value.
    assert abs(loss_j - float(ch["energy_line_loss_joules"][-1])) <= 1.0

    # Grid region attribution.
    sub_row = df.filter(df["district_id"] == "subMain")
    assert sub_row["grid_emissions_region"][0] == "MROW"


def test_substation_load_curves():
    train_sim = _run_catenary_sim()
    curves = en.substation_load_curves(train_sim)
    assert "time_seconds" in curves.columns
    assert "load_watts__subMain" in curves.columns
    assert "load_watts__total" in curves.columns
    assert curves["load_watts__subMain"].max() > 0.0
