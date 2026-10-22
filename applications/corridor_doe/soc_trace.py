"""One representative train per corridor, re-walked with state-of-charge history.

The DoE's own saved history deliberately records almost nothing -- train time,
speed and offset, consist fuel, locomotive aux -- because saved history is the
dominant memory cost of a walked plan and it is what made Winnipeg-Chicago
simulable at all. The battery's own history records NOTHING, so there is no
state-of-charge trace anywhere in `out/sims/`.

This re-runs a single train per (corridor, powertrain) with the battery series
switched on, which costs seconds rather than the memory a whole plan would. The
train is dispatched and walked exactly as the sweep does, so its speed profile
is a dispatched one rather than a free run.

Usage:
    python soc_trace.py --powertrain catenary_bel --out <dir>
    python soc_trace.py --powertrain hybrid --battery-mwh 1.0 --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import polars as pl
import altrios as alt
from altrios import history
from altrios.train_planner import data_prep
import run_sweep as rs
from speed_calibration import CORRIDORS, NETWORKS, load_rail_vehicles

# Fine enough to show a pack filling on a descent; still trivial for one train.
INTERVAL = 10
# `res` is a child of `loco_vec`, and a component only records on steps where
# every ancestor also records, so the parent has to be switched on too -- with
# the same interval, since intervals compose as a least common multiple down the
# tree. `configure` rejects the inconsistent spec rather than silently saving
# nothing, which is how this was caught.
SPEC = {
    "<root>": dict(interval=INTERVAL,
                   keep=["time_seconds", "speed_meters_per_second", "offset_meters"]),
    "loco_con": dict(interval=INTERVAL, keep=["energy_fuel_joules"]),
    "loco_vec": dict(interval=INTERVAL, keep=["energy_aux_joules"]),
    "res": dict(interval=INTERVAL, keep=["soc"]),
}


def elevation_by_offset(network) -> pl.DataFrame:
    """Cumulative distance -> elevation along the mainline, for the backdrop."""
    links = network.to_pydict()[1]
    by_idx = {L["idx_curr"]: L for L in links}
    idx, seen, dist, rows, seen_order = 1, set(), 0.0, [], []
    while idx and idx not in seen:
        seen.add(idx); seen_order.append(idx)
        L = by_idx[idx]
        for e in (L.get("elevs") or []):
            rows.append({"offset_m": dist + e["offset_meters"], "elev_m": e["elev_meters"]})
        dist += L["length_meters"]
        idx = L["idx_next"]
    return pl.DataFrame(rows).unique(subset=["offset_m"]).sort("offset_m"), seen_order, dist


def energized_intervals(network) -> pl.DataFrame:
    """Contiguous energized stretches along the mainline, as (start, end) metres.

    Same construction as the deep-dive's elevation figure: flag each mainline
    link as energized or not, then collapse runs of like values into intervals,
    so the chart draws a handful of rectangles instead of one per link. A link
    is energized when it carries any `cat_power_limits` entry, which is what the
    layout optimizer writes when it puts wire over a section.
    """
    links = network.to_pydict()[1]
    by_idx = {L["idx_curr"]: L for L in links}
    idx, seen, dist, runs = 1, set(), 0.0, []
    while idx and idx not in seen:
        seen.add(idx)
        L = by_idx[idx]
        hot = bool(L.get("cat_power_limits"))
        end = dist + L["length_meters"]
        if runs and runs[-1][2] == hot:
            runs[-1][1] = end
        else:
            runs.append([dist, end, hot])
        dist = end
        idx = L["idx_next"]
    hot_runs = [r for r in runs if r[2]]
    if not hot_runs:
        return pl.DataFrame({"x1_km": [], "x2_km": []},
                            schema={"x1_km": pl.Float64, "x2_km": pl.Float64})
    return pl.DataFrame({"x1_km": [r[0] / 1000 for r in hot_runs],
                         "x2_km": [r[1] / 1000 for r in hot_runs]})


# Which train each direction is represented by; set from --selection.
SELECTION = "median"


def pick_trains(plan: pl.DataFrame, helper_free=None, selection="median") -> list:
    """One representative train per DIRECTION.

    Both directions, because on these corridors they are the interesting
    contrast -- Spokane-Lethbridge runs 3.6x the tonnage one way as the other,
    and a pack that fills on the loaded descent drains on the empty climb.
    Heaviest direction first, so the rows read loaded-then-empty.

    `selection` picks WHICH train in each direction:

    * ``median`` -- the middle train by tonnage. A typical trip.
    * ``heaviest`` -- the heaviest train. For a battery-electric consist this is
      combined with the helper-free filter below, so it is the hardest trip the
      pack is asked to make ON ITS OWN, and therefore the one that comes closest
      to the floor. The plain median understates how hard the pack works: the
      heavy tail of this corridor is exactly the trains the helper ladder gives
      conventional units to, so excluding them and then taking the median
      removes the whole upper tail of pack demand.
    """
    t = plan.unique(subset=["Train_ID"])
    order = (t.group_by("Origin_ID").agg(pl.col("Train_Weight_tons").mean().alias("m"))
             .sort("m", descending=True))
    out = []
    for origin in order["Origin_ID"].to_list():
        side = t.filter(pl.col("Origin_ID") == origin).sort("Train_Weight_tons")
        if helper_free is not None:
            # Illustrate a train that actually runs on its pack: exclude any
            # train the helper ladder had to give conventional units to.
            unaided = side.filter(pl.col("Train_ID").is_in(helper_free))
            if unaided.height:
                side = unaided
        row = side.row(side.height - 1 if selection == "heaviest"
                       else side.height // 2, named=True)
        out.append(dict(row, _n_locos=plan.filter(pl.col("Train_ID") == row["Train_ID"]).height))
    return out


def trace(slug: str, powertrain: str, battery, layout_row) -> pl.DataFrame:
    corridor = CORRIDORS[slug]
    network = rs.network_for(layout_row)
    location_map = alt.import_locations(str(NETWORKS / corridor.locations))
    plan = pl.read_csv(corridor.sample_plan_paths[0])
    # Battery-electric consists are NOT pure: the sweep's helper ladder swaps
    # conventional units into the trains that cannot cross on the pack alone,
    # and on this corridor the heavy direction is exactly those trains. Building
    # a pure pack consist strands the representative heavy train at the 5% floor
    # before it gets anywhere, which is what happened the first time. Use the
    # same assignment the sweep does, at its base (un-escalated) level.
    if powertrain == "catenary_bel":
        plan = rs.assign_diesel_helpers(plan, layout_row, network, location_map, 0.0)
    rv0 = load_rail_vehicles()
    helper_free = None
    if powertrain == "catenary_bel":
        helped = set(plan.filter(pl.col("Locomotive_Type") == rs.HELPER_LOCO_TYPE)
                     ["Train_ID"].to_list())
        helper_free = [t for t in plan["Train_ID"].unique().to_list() if t not in helped]
    wire = energized_intervals(network) if powertrain.startswith("catenary") else None
    return [_one(slug, powertrain, battery, network, location_map, rv0, tr, plan, wire)
            for tr in pick_trains(plan, helper_free, SELECTION)]


def _one(slug, powertrain, battery, network, location_map, rv0, tr, plan,
         wire=None) -> pl.DataFrame:
    # One locomotive per plan row, of the type that row names: a helper row is a
    # conventional unit, everything else is this cell's powertrain.
    types = (plan.filter(pl.col("Train_ID") == tr["Train_ID"])
             .sort("Locomotive_ID")["Locomotive_Type"].to_list())
    locos = [rs.build_locomotive("diesel", None) if t == rs.HELPER_LOCO_TYPE
             else rs.build_locomotive(powertrain, battery) for t in types]
    n_help = sum(1 for t in types if t == rs.HELPER_LOCO_TYPE)
    car_type = rv0[0].to_pydict()["car_type"]
    n_cars = {car_type: int(tr["Cars_Loaded"]) + int(tr["Cars_Empty"])}
    rv = data_prep.scale_rv_mass_and_length(
        [rv0[0]], n_cars, locos,
        float(tr["Train_Weight_tons"]) * alt.utilities.KG_PER_TON,
        float(tr["Train_Length_meters"]))
    rv = data_prep.raise_rv_speed_max(rv, tr.get("Train_Speed_Max_Meters_Per_Second"))
    tsb = alt.TrainSimBuilder(
        train_id=str(tr["Train_ID"]), origin_id=tr["Origin_ID"],
        destination_id=tr["Destination_ID"],
        train_config=alt.TrainConfig(rail_vehicles=rv, n_cars_by_type=n_cars,
                                     train_type=alt.TrainType.Freight),
        loco_con=alt.Consist(locos))
    ts = tsb.make_speed_limit_train_sim(location_map=location_map, save_interval=INTERVAL)
    ts = history.configure(ts, SPEC)
    etn, _ = alt.make_est_times(ts, network)
    tp = alt.run_dispatch(network, alt.SpeedLimitTrainSimVec([ts]), [etn], False, False)[0]
    vec = alt.SpeedLimitTrainSimVec([ts])
    vec.walk_timed_paths(network, [tp], False)
    d = vec.to_pydict()[0]
    h = d["history"]
    out = pl.DataFrame({"offset_m": h["offset_meters"],
                        "speed_mps": h["speed_meters_per_second"],
                        "time_s": h["time_seconds"]})
    socs = []
    for L in d["loco_con"]["loco_vec"]:
        body = next(iter(L["loco_type"].values()))
        hh = (body.get("res") or {}).get("history") or {}
        if hh.get("soc"):
            socs.append(hh["soc"])
    if socs:
        n = min(len(out), min(len(s) for s in socs))
        out = out.head(n).with_columns(
            pl.Series("soc", [sum(s[i] for s in socs) / len(socs) for i in range(n)]))
    elev, order, corridor_m = elevation_by_offset(network)
    # `offset_m` is the TRAIN's own distance from ITS origin, always increasing,
    # while `elev` and the energized intervals are indexed on the corridor walk,
    # which runs one fixed way. A train starting at the far end therefore has to
    # read both series BACKWARDS. Without this both directions were drawn against
    # the same profile, so the loaded Lethbridge train appeared to climb the
    # grade it actually descends -- and its wire bands sat in the wrong places.
    origin_link = None
    for loc in location_map.get(tr["Origin_ID"], []):
        origin_link = loc.to_pydict()["Link Index"]
        break
    reverse = (origin_link is not None and origin_link in order
               and order.index(origin_link) > len(order) // 2)
    out = out.with_columns(
        ((pl.lit(corridor_m) - pl.col("offset_m")) if reverse
         else pl.col("offset_m")).alias("_corridor_m"))
    out = (out.sort("_corridor_m")
           .join_asof(elev.sort("offset_m"), left_on="_corridor_m", right_on="offset_m")
           .sort("offset_m"))
    # Energized flag per sample, for the orange underlay. Carried here rather
    # than as a separate table because a faceted Altair chart cannot layer two
    # data sources -- the underlay must come from the same frame as the lines.
    if wire is not None and wire.height:
        lo, hi = wire['x1_km'].to_list(), wire['x2_km'].to_list()
        hot = [any(a <= v < b for a, b in zip(lo, hi)) for v in (out['_corridor_m'] / 1000)]
        out = out.with_columns(pl.Series('energized', hot))
    else:
        out = out.with_columns(pl.lit(False).alias('energized'))
    return out.with_columns(pl.lit(slug).alias("corridor"),
                            pl.lit(powertrain).alias("powertrain"),
                            pl.lit(n_help).alias("diesel_helpers"),
                            pl.lit(len(types)).alias("n_locos"),
                            pl.lit(str(tr["Origin_ID"])).alias("origin"),
                            pl.lit(float(tr["Train_Weight_tons"])).alias("tons"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--powertrain", required=True)
    ap.add_argument("--battery-mwh", type=float)
    ap.add_argument("--architecture", default="ac_25kv",
                    help="catenary architecture to illustrate (default ac_25kv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--selection", choices=("median", "heaviest"), default="median",
                    help="which train represents each direction. `heaviest` is "
                         "combined with the helper-free filter, so it is the "
                         "hardest trip the pack makes on its own")
    args = ap.parse_args(argv)
    exp = pl.read_csv(HERE / "experiments.csv", infer_schema_length=None)
    sel = exp.filter(pl.col("Powertrain") == args.powertrain)
    if args.battery_mwh is not None:
        sel = sel.filter(pl.col("Battery_MWh") == args.battery_mwh)
    frames, wires = [], []
    for slug in sorted(sel["Corridor"].unique().to_list()):
        rows = sel.filter(pl.col("Corridor") == slug)
        # The most ELECTRIFIED buildout for a catenary arm, chosen by measured
        # grid draw rather than by name. Sorting fingerprints alphabetically
        # picked "nobuild" -- the cell where the optimizer found no feasible
        # layout at all -- which is the one buildout that must never be the
        # illustration.
        if args.powertrain.startswith("catenary"):
            # `ac_25kv` by default: the conventional single-phase 25 kV
            # transformer-fed architecture, which is the sensible reference
            # rather than the autotransformer variant or -- emphatically -- the
            # "nobuild" cell where the optimizer found no feasible layout.
            pref = rows.filter(pl.col("Architecture") == args.architecture)
            if not pref.height:
                raise SystemExit(f"{slug}: no {args.architecture} buildout")
            row = pref.row(0, named=True)
        else:
            row = rows.row(0, named=True)
        print(f"{slug}: {row['Sim_Key']}", flush=True)
        frames.extend(trace(slug, args.powertrain, args.battery_mwh, row))
        if args.powertrain.startswith("catenary"):
            iv = energized_intervals(rs.network_for(row))
            if iv.height:
                wires.append(iv.with_columns(pl.lit(slug).alias("corridor")))
    global SELECTION
    SELECTION = args.selection
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.powertrain + (f"_{args.battery_mwh:g}" if args.battery_mwh else "")
    # The median keeps the historic filename so existing figures do not move;
    # other selections are suffixed.
    suffix = "" if args.selection == "median" else f"_{args.selection}"
    path = args.out / f"soc_{tag}{suffix}.parquet"
    pl.concat(frames, how="diagonal_relaxed").write_parquet(path)
    print("wrote", path)
    if wires:
        wpath = args.out / f"energized_{tag}{suffix}.parquet"
        pl.concat(wires, how="diagonal_relaxed").write_parquet(wpath)
        print("wrote", wpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
