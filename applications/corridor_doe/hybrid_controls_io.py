"""Optional tuned controller settings for the hybrid arm.

The DoE's hybrid runs ALTRIOS's rule-based `RGWDB` controller. By default it
runs the library values; if `in/hybrid_controls.csv` exists, the values there
replace them. The file is written by the genetic-algorithm sweep in the
analysis repo (`Scripts/hybrid_controls/run_sweep_ga.py` -> `chosen_controls.json`).

Two scopes per corridor, `fwd` and `rev`, so a locomotive can carry a different
rule set for each direction of travel. That is the software-defined-controls
question: the tuning corridor's two directions differ enormously in tonnage
(Spokane-Lethbridge is 14,867 t mean out of Lethbridge against 4,226 t out of
Spokane), and a single compromise setting need not be best for either.

The mapping from a corridor's two directions onto `fwd`/`rev` is by MEAN TRAIN
TONNAGE, not by anything in the tuning corridor's geography: the heavier
direction of each corridor takes the set tuned on the tuning corridor's heavier
direction. The chosen `Origin_ID` is written into the file per corridor and
scope so the mapping is explicit rather than implied. On Toronto-Battle Creek
the two directions are within 6% of each other in tonnage, so which set a
direction receives there is close to arbitrary and should be reported as such.

ABSENT FILE = library defaults, which is the pre-tuning behaviour, so nothing
downstream has to special-case "untuned".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import polars as pl

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
CONTROLS_PATH = HERE / "in" / "hybrid_controls.csv"

# The RGWDB fields the sweep tunes. Anything else in the file is ignored, and a
# field missing from the file keeps its library value.
RGWDB_FIELDS = (
    "speed_soc_regen_buffer_meters_per_second",
    "speed_soc_disch_buffer_meters_per_second",
    "frac_of_max_pwr_to_run_fc",
    "frac_pwr_demand_fc_forced_on",
)
SCOPES = ("fwd", "rev")

# The sweep names its decision variables short; the model field names are long.
SHORT_TO_RGWDB = {
    "regen_buffer_mps": "speed_soc_regen_buffer_meters_per_second",
    "disch_buffer_mps": "speed_soc_disch_buffer_meters_per_second",
    "frac_run_fc": "frac_of_max_pwr_to_run_fc",
    "frac_forced_on": "frac_pwr_demand_fc_forced_on",
}


def load(path: Optional[Path] = None) -> Optional[Dict[str, dict]]:
    """``{corridor: {"scopes": {scope: {field: value}}, "origins": {origin_id: scope}}}``
    or None when no file is present."""
    path = Path(path) if path else CONTROLS_PATH
    if not path.exists():
        return None
    frame = pl.read_csv(path, infer_schema_length=None)
    missing = [c for c in ("Corridor", "Scope", "Origin_ID") if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}")
    out: Dict[str, dict] = {}
    for row in frame.iter_rows(named=True):
        scope = str(row["Scope"]).strip()
        if scope not in SCOPES:
            raise ValueError(f"{path}: Scope must be one of {SCOPES}, got {scope!r}")
        entry = out.setdefault(str(row["Corridor"]), {"scopes": {}, "origins": {}})
        entry["scopes"][scope] = {
            f: float(row[f]) for f in RGWDB_FIELDS
            if f in frame.columns and row[f] is not None
        }
        entry["origins"][str(row["Origin_ID"]).strip()] = scope
    logger.info("hybrid controls: %s -> %s", path.name,
                {c: sorted(v["scopes"]) for c, v in out.items()})
    return out


def scope_for(controls: dict, corridor: str, origin_id: str) -> Optional[str]:
    """Which tuned set a train departing `origin_id` uses, or None if untuned.

    Unknown origins fall back to `fwd` with a warning rather than failing the
    cell: a corridor whose plan gains a new origin should still simulate, and a
    silent mis-assignment is what the logged warning exists to prevent.
    """
    entry = (controls or {}).get(corridor)
    if not entry:
        return None
    scope = entry["origins"].get(str(origin_id).strip())
    if scope is None:
        logger.warning("hybrid controls: %s has no scope for origin %r; using fwd",
                       corridor, origin_id)
        return "fwd"
    return scope


def values_for(controls: dict, corridor: str, scope: Optional[str]) -> Dict[str, float]:
    """The RGWDB overrides for one corridor and scope; empty dict when untuned."""
    entry = (controls or {}).get(corridor)
    if not entry or scope is None:
        return {}
    return dict(entry["scopes"].get(scope) or {})


def apply_to_locomotive(loco, values: Dict[str, float]):
    """Stamp `values` onto a hybrid locomotive's RGWDB controller.

    A no-op for every other powertrain, so callers need not branch: only
    `HybridLoco` carries an RGWDB block.
    """
    if not values:
        return loco
    import altrios as alt

    d = loco.to_pydict()
    body = d.get("loco_type", {}).get("HybridLoco")
    if not body:
        return loco
    rgwdb = body["pt_cntrl"]["RGWDB"]
    for field, value in values.items():
        if field not in rgwdb:
            raise KeyError(f"RGWDB has no field {field!r}")
        rgwdb[field] = float(value)
    return alt.Locomotive.from_pydict(d)


def fingerprint_bytes(path: Optional[Path] = None) -> bytes:
    """What `run_sweep._inputs_fingerprint` should hash so a controls change
    invalidates the hybrid cells. Stable when the file is absent."""
    path = Path(path) if path else CONTROLS_PATH
    return path.read_bytes() if path.exists() else b"no-hybrid-controls"


def write_from_chosen(chosen_params: Dict[str, Dict[str, float]],
                      origins_by_corridor: Dict[str, Dict[str, str]],
                      source: str,
                      path: Optional[Path] = None) -> Path:
    """Write `in/hybrid_controls.csv` from the sweep's `chosen_controls.json`.

    `chosen_params` is the sweep's `{scope: {short_name: value}}`;
    `origins_by_corridor` is `{corridor: {scope: origin_id}}`, computed from
    mean train tonnage per origin. `source` records which GA run produced it.
    """
    rows = []
    for corridor, by_scope in origins_by_corridor.items():
        for scope, origin in by_scope.items():
            row = {"Corridor": corridor, "Scope": scope, "Origin_ID": origin,
                   "Source": source}
            for short, value in (chosen_params.get(scope) or {}).items():
                row[SHORT_TO_RGWDB[short]] = float(value)
            rows.append(row)
    path = Path(path) if path else CONTROLS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)
    return path


# ---------------------------------------------------------------------------
# Per-group regen calibration (2026-10-18): `in/hybrid_regen_controls.csv`
#
# Written by `Scripts/hybrid_controls/regen_calibration.py`. One row per
# (corridor, pack size, origin, load class). Unlike the per-direction file
# above, the values are stamped onto each TRAIN just before it is walked
# (`sim_manager`'s `configure_sim` hook), because the planner reuses one
# locomotive type across both directions and both load classes. That is the
# same stamping the calibration's fixed-path re-walk does, which reproduces the
# DoE's walk exactly.
#
# `Soc_Dep` is the departure state of charge the calibration chose for that
# group. It is NOT stamped from here: departure SOC is set in exactly one place,
# the train planner, via `network_charging_guidelines`
# (`data_prep.dispatch_headroom_j`). `headroom_guidelines` below converts these
# rows into that table. Stamping it here as well would have the planner set a
# departure SOC, this hook overwrite it, and terminal charging bill the
# overwritten one. Load class is by tonnage per locomotive against
# `Tons_Per_Loco_Split` (heavy at or above it).
#
# ABSENT FILE = library controller and no headroom, i.e. depart as full as the
# pack arrived (the previous behaviour).
REGEN_CONTROLS_PATH = HERE / "in" / "hybrid_regen_controls.csv"
REGEN_FIELDS = (
    "speed_soc_regen_buffer_coeff",
    "speed_soc_regen_buffer_meters_per_second",
    "frac_of_max_pwr_to_run_fc",
    "speed_soc_disch_buffer_meters_per_second",
)


def load_regen(path: Optional[Path] = None) -> Optional[pl.DataFrame]:
    path = Path(path) if path else REGEN_CONTROLS_PATH
    if not path.exists():
        return None
    return pl.read_csv(path, infer_schema_length=None)


def regen_fingerprint_bytes(path: Optional[Path] = None) -> bytes:
    path = Path(path) if path else REGEN_CONTROLS_PATH
    return path.read_bytes() if path.exists() else b"no-hybrid-regen-controls"


def _regen_group_lookup(table: Optional[pl.DataFrame], corridor: str,
                        battery_mwh: Optional[float]):
    """``sim_pydict -> row`` for one corridor x pack cell (the tuned row for the
    train's origin and load class), or None when the table has no rows for it."""
    if table is None or battery_mwh is None:
        return None
    rows = table.filter((pl.col("Corridor") == corridor)
                        & ((pl.col("Battery_MWh") - float(battery_mwh)).abs() < 1e-9))
    if rows.height == 0:
        logger.warning("hybrid regen controls: no rows for %s %.2f MWh", corridor, battery_mwh)
        return None
    by_key = {(r["Origin_ID"], r["Load_Class"]): r for r in rows.iter_rows(named=True)}
    split = {r["Origin_ID"]: float(r["Tons_Per_Loco_Split"]) for r in rows.iter_rows(named=True)}

    def lookup(d: dict):
        origin = d["origs"][0]["Location ID"]
        if origin not in split:
            logger.warning("hybrid regen controls: %s has no rows for origin %r", corridor, origin)
            return None
        locos = d["loco_con"]["loco_vec"]
        tons_per_loco = float(d["state"]["mass_static_kilograms"]) / 1000.0 / max(len(locos), 1)
        cls = "heavy" if tons_per_loco >= split[origin] else "light"
        return by_key.get((origin, cls)) or by_key.get((origin, "all"))

    return lookup


def regen_configurer(table: Optional[pl.DataFrame], corridor: str,
                     battery_mwh: Optional[float]):
    """A `sim -> sim` stamper for one corridor x pack cell, or None when the
    table is absent or has no rows for the cell."""
    lookup = _regen_group_lookup(table, corridor, battery_mwh)
    if lookup is None:
        return None

    def configure(sim):
        import altrios as alt

        d = sim.to_pydict()
        row = lookup(d)
        if row is None:
            return sim
        for loco in d["loco_con"]["loco_vec"]:
            body = loco["loco_type"].get("HybridLoco")
            if not body:
                continue
            rg = body["pt_cntrl"]["RGWDB"]
            for f in REGEN_FIELDS:
                if row.get(f) is not None:
                    rg[f] = float(row[f])
            # Departure SOC deliberately untouched -- see the note above.
        return alt.SpeedLimitTrainSim.from_pydict(d)

    return configure


def headroom_guidelines(table: Optional[pl.DataFrame],
                        corridor: Optional[str] = None,
                        battery_mwh: Optional[float] = None,
                        max_soc: float = 0.95) -> Optional[pl.DataFrame]:
    """The calibration's departure SOCs as a `network_charging_guidelines` table.

    The planner caps departure SOC at ``SOC_Max_J - headroom``, so a target SOC
    of `Soc_Dep` on a pack of `Battery_MWh` is a headroom of
    ``(max_soc - Soc_Dep) * Battery_MWh`` MWh. Negative values (a target above
    the pack's own ceiling) clamp to zero.

    One row per (origin, destination, load class, locomotive type, pack size).
    The calibration tunes by ORIGIN, so every destination out of that origin
    gets the same row -- `Destination` is carried as a wildcard rather than
    enumerated, which keeps the table honest about what was actually measured
    while leaving the column in place for when it is tuned per direction.
    Likewise `Locomotive_Type` is a wildcard: the caller already narrows the
    table to one corridor and pack size, so there is nothing left to
    distinguish. Battery-electric terminals are simply absent from the
    calibration, which the planner reads as no headroom -- depart as full as
    the pack arrived -- until they are tuned in their own right.
    """
    if table is None or table.height == 0:
        return None
    needed = {"Corridor", "Battery_MWh", "Origin_ID", "Load_Class", "Soc_Dep"}
    missing = needed - set(table.columns)
    if missing:
        raise ValueError(f"regen controls table is missing {sorted(missing)}")
    # Terminal names repeat across corridors, so the cell must be selected here:
    # the planner keys on origin alone and would otherwise see another
    # corridor's rows for the same name.
    if corridor is not None:
        table = table.filter(pl.col("Corridor") == corridor)
    if battery_mwh is not None:
        table = table.filter(
            (pl.col("Battery_MWh") - float(battery_mwh)).abs() < 1e-9
        )
    if table.height == 0:
        return None
    out = table.select(
        pl.col("Origin_ID").cast(pl.Utf8).alias("Origin"),
        pl.lit("all").alias("Destination"),
        pl.col("Load_Class").cast(pl.Utf8),
        pl.col("Tons_Per_Loco_Split").cast(pl.Float64)
        if "Tons_Per_Loco_Split" in table.columns
        else pl.lit(None, dtype=pl.Float64).alias("Tons_Per_Loco_Split"),
        pl.lit(None, dtype=pl.Utf8).alias("Locomotive_Type"),
        pl.col("Battery_MWh").cast(pl.Float64),
        pl.max_horizontal(
            pl.lit(0.0),
            (pl.lit(float(max_soc)) - pl.col("Soc_Dep").cast(pl.Float64))
            * pl.col("Battery_MWh").cast(pl.Float64),
        ).alias("Allowable_Battery_Headroom_MWh"),
    )
    return out.unique()
