"""Terminal (stationary) charging for battery-carrying locomotives: the energy
each locomotive takes at a terminal before departing, and the charger capital
that serves it.

**Energy.** The DoE walks every train inside ``sim_manager``, which produces no
charging sessions, so until now grid electricity for battery locomotives was
the net pack drawdown divided by the charger efficiency -- a fleet-level true-up
with no location and no per-locomotive deficit. Here each locomotive's trips are
followed in departure order through the consist plan: at every departure it is
charged from what it ARRIVED with on its previous trip up to what it DEPARTS
with on this one. Only deficits are charged; a locomotive that arrives above its
departure SOC is not discharged (it simply leaves fuller than the simulation
assumed, which is conservative). Where the previous arrival is unknown -- the
locomotive's first measured trip, or after an unwalked warm-up/cool-down trip,
or a previous trip that did not end where this one starts -- the mean measured
arrival SOC at that terminal, for that locomotive type, stands in (or over all
terminals, if no measured train arrived at that one).

The departure SOC is whatever the walk started from: the planner's
``Departure_SOC_J`` for battery-electric units (the planner charges them to its
target), or the tuned per-group value for hybrids (``hybrid_controls_io``).

**Chargers.** Battery-electric units keep the planner's own refuelers
(``refuelers_per_incoming_corridor`` ports per terminal). Hybrids, which need
far less energy, get the battery-electric charger count at each terminal scaled
by the ratio of the hybrid's to the battery-electric's terminal charging energy
there; charger cost is linear in installed power, so a fractional port count is
an installed-power equivalent. Where a corridor has no battery-electric cell,
the reference is the battery-electric installed ports per unit of weekly
terminal charging energy from the reference corridor.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import polars as pl

# Deficits below this are numerical noise (a pack that arrived full under the
# wire), not a charging session.
MIN_SESSION_J = 1.0e6

# A reference terminal that takes less than this share of the reference's total
# terminal charging is too small to scale from; the corridor-total ratio is used.
MIN_REFERENCE_SHARE = 0.05

SESSION_SCHEMA = {
    "Locomotive_ID": pl.UInt32,
    "Locomotive_Type": pl.Utf8,
    "Fuel_Type": pl.Utf8,
    "Node": pl.Utf8,
    "Refuel_Energy_J": pl.Float64,
    "Battery_Energy_J": pl.Float64,
    "Start_Hr": pl.Float64,
    "End_Hr": pl.Float64,
}


def _res_of(loco_pd: dict) -> Optional[dict]:
    body = next(iter(loco_pd["loco_type"].values())) or {}
    return body.get("res")


def charging_sessions(plan: pl.DataFrame, sim_pydicts: list,
                      charger_efficiency: float,
                      departure_soc_fn: Optional[Callable[[dict], Optional[float]]] = None,
                      charger_power_w: float = 750e3,
                      ) -> pl.DataFrame:
    """One row per (locomotive, departure) that needed charge: ``Node`` is the
    departure terminal, ``Refuel_Energy_J`` the GRID energy (battery energy over
    the charger efficiency), ``Battery_Energy_J`` the energy into the pack.

    ``plan`` is the consist plan (one row per locomotive per train) with
    ``TrainSimVec_Index`` set for walked trains; ``sim_pydicts`` the walked sims
    in vector order. Within a train the sim's locomotive vector is ordered by
    ``Locomotive_ID`` (the planner sorts the dispatched locomotives by ID before
    building the consist). ``departure_soc_fn(sim_pydict)`` overrides the
    planner's departure SOC (fraction) when it returns a value.

    ``Start_Hr``/``End_Hr`` time each session for charger sizing: it starts when
    the locomotive arrived from its previous trip and runs at ``charger_power_w``
    (into the pack, as the planner applies the charger rate) until done or the departure, whichever is first; without a known
    previous arrival it is placed to end at the departure.
    """
    time_col = ("Departure_Time_Actual_Hr" if "Departure_Time_Actual_Hr" in plan.columns
                else "Departure_Time_Planned_Hr")
    arr_col = ("Arrival_Time_Actual_Hr" if "Arrival_Time_Actual_Hr" in plan.columns
               else "Arrival_Time_Planned_Hr")
    walked = []
    for idx, sim in enumerate(sim_pydicts):
        rows = (plan.filter(pl.col("TrainSimVec_Index") == idx)
                .sort("Locomotive_ID").to_dicts())
        locos = sim["loco_con"]["loco_vec"]
        if len(rows) != len(locos):
            raise ValueError(f"train sim {idx}: {len(locos)} locomotives in the sim, "
                             f"{len(rows)} in the consist plan")
        override = departure_soc_fn(sim) if departure_soc_fn else None
        for r, loco in zip(rows, locos):
            res = _res_of(loco)
            if not res:
                continue
            cap = float(res["energy_capacity_joules"])
            e_dep = (float(override) * cap if override is not None
                     else float(r["Departure_SOC_J"]))
            walked.append(dict(
                Locomotive_ID=int(r["Locomotive_ID"]), Locomotive_Type=str(r["Locomotive_Type"]),
                Train_ID=r["Train_ID"], Origin=str(r["Origin_ID"]),
                Destination=str(r["Destination_ID"]), t_dep=float(r[time_col]),
                t_arr=float(r[arr_col]) if r.get(arr_col) is not None else None,
                cap=cap, e_dep=e_dep, e_arr=float(res["state"]["soc"]) * cap))
    if not walked:
        return pl.DataFrame(schema=SESSION_SCHEMA)
    w = pl.DataFrame(walked)
    # Fallback arrival SOC (fraction) at each terminal, per locomotive type;
    # then over all terminals, for a terminal no measured train arrived at.
    fallback = {(t, n): s for t, n, s in
                w.group_by("Locomotive_Type", "Destination")
                .agg((pl.col("e_arr") / pl.col("cap")).mean().alias("soc"))
                .iter_rows()}
    fallback_any = {t: s for t, s in
                    w.group_by("Locomotive_Type")
                    .agg((pl.col("e_arr") / pl.col("cap")).mean().alias("soc"))
                    .iter_rows()}
    # Each locomotive's FULL trip sequence (walked or not), to know whether the
    # trip before a walked departure was itself walked.
    seq = (plan.select("Locomotive_ID", "Train_ID", "Origin_ID", "Destination_ID",
                       pl.col(time_col).alias("t_dep"),
                       pl.col("TrainSimVec_Index").is_not_null().alias("walked"))
           .unique(subset=["Locomotive_ID", "Train_ID"])
           .sort("Locomotive_ID", "t_dep"))
    prev_train = {}
    for loco_id, g in seq.group_by("Locomotive_ID", maintain_order=True):
        trains = g["Train_ID"].to_list()
        for k, t in enumerate(trains):
            prev_train[(int(loco_id[0]), t)] = trains[k - 1] if k else None
    arr_by = {(r["Locomotive_ID"], r["Train_ID"]): r for r in walked}
    out = []
    for r in walked:
        p = prev_train.get((r["Locomotive_ID"], r["Train_ID"]))
        prev = arr_by.get((r["Locomotive_ID"], p)) if p is not None else None
        start = None
        if prev is not None and prev["Destination"] == r["Origin"]:
            e_prev = prev["e_arr"]
            start = prev["t_arr"]
        else:
            soc = fallback.get((r["Locomotive_Type"], r["Origin"]),
                               fallback_any.get(r["Locomotive_Type"]))
            e_prev = soc * r["cap"]
        deficit = max(0.0, r["e_dep"] - e_prev)
        if deficit > MIN_SESSION_J:
            grid_j = deficit / charger_efficiency
            # Rate applies to energy INTO the pack, as the planner's
            # Refuel_Duration_Hr = (SOC_Target_J - SOC_J) / Refueler_J_Per_Hr.
            dur_hr = deficit / charger_power_w / 3600.0
            if start is None or start > r["t_dep"]:
                start = r["t_dep"] - dur_hr
            out.append(dict(Locomotive_ID=r["Locomotive_ID"],
                            Locomotive_Type=r["Locomotive_Type"],
                            Fuel_Type="Electricity", Node=r["Origin"],
                            Refuel_Energy_J=grid_j, Battery_Energy_J=deficit,
                            Start_Hr=start, End_Hr=min(start + dur_hr, r["t_dep"])))
    if not out:
        return pl.DataFrame(schema=SESSION_SCHEMA)
    return pl.DataFrame(out, schema=SESSION_SCHEMA)


def energy_by_node(sessions: pl.DataFrame) -> Dict[str, float]:
    """Grid joules per terminal."""
    if sessions is None or sessions.height == 0:
        return {}
    return {n: float(v) for n, v in
            sessions.group_by("Node").agg(pl.col("Refuel_Energy_J").sum()).iter_rows()}


def peak_concurrent_sessions(sessions: pl.DataFrame) -> Dict[str, int]:
    """Most charging sessions in progress at once, per terminal: the port count
    at which no locomotive waits for a charger."""
    out: Dict[str, int] = {}
    if sessions is None or sessions.height == 0:
        return out
    for (node,), g in sessions.group_by(["Node"]):
        events = sorted([(s, 1) for s in g["Start_Hr"]] + [(e, -1) for e in g["End_Hr"]],
                        key=lambda t: (t[0], t[1]))
        cur = peak = 0
        for _, k in events:
            cur += k
            peak = max(peak, cur)
        out[str(node)] = peak
    return out


def sized_chargers(template: pl.DataFrame, sessions: pl.DataFrame) -> Optional[pl.DataFrame]:
    """Electric chargers sized to the peak concurrent sessions at each terminal
    (no locomotive waits), from one charger-template row (cost, rate, lifespan)."""
    peaks = peak_concurrent_sessions(sessions)
    if not peaks or template is None or template.height == 0:
        return None
    tpl = template.head(1).to_dicts()[0]
    return pl.DataFrame([{**tpl, "Node": n, "Port_Count": float(k)}
                         for n, k in sorted(peaks.items()) if k > 0])


def bel_charger_rows(refuelers: pl.DataFrame) -> pl.DataFrame:
    """The electric chargers among the planner's refuelers."""
    return refuelers.filter(pl.col("Fuel_Type").cast(pl.Utf8) == "Electricity")


def scaled_chargers(template: pl.DataFrame, own_j_by_node: Dict[str, float],
                    ref_j_by_node: Dict[str, float],
                    ref_ports_by_node: Dict[str, float],
                    same_corridor: bool) -> pl.DataFrame:
    """Charger facilities for a hybrid cell, scaled from a battery-electric
    reference. ``template`` is one electric-charger row (cost, lifespan, rate).

    Same corridor: ports at each terminal = reference ports there x
    (own terminal energy / reference terminal energy). Other corridor, or a
    reference terminal with under MIN_REFERENCE_SHARE of the reference's
    charging: ports = own terminal energy x (total reference ports / total
    reference energy).
    Port counts are fractional installed-power equivalents (cost is linear in MW).
    """
    tpl = template.head(1).to_dicts()[0]
    rows = []
    tot_ports = sum(ref_ports_by_node.values())
    tot_j = sum(ref_j_by_node.values())
    for node, j in own_j_by_node.items():
        if j <= 0:
            continue
        if (same_corridor and tot_j > 0
                and ref_j_by_node.get(node, 0.0) >= MIN_REFERENCE_SHARE * tot_j):
            ports = ref_ports_by_node.get(node, 0.0) * j / ref_j_by_node[node]
        elif tot_j > 0:
            ports = tot_ports * j / tot_j
        else:
            continue
        rows.append({**tpl, "Node": node, "Port_Count": float(ports)})
    if not rows:
        return template.head(0).with_columns(pl.col("Port_Count").cast(pl.Float64))
    return pl.DataFrame(rows).with_columns(pl.col("Port_Count").cast(pl.Float64))
