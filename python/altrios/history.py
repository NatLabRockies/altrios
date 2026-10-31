"""Select which history series a simulation records, per component.

`configure` takes a spec mapping each component of a `SpeedLimitTrainSim` or
`SetSpeedTrainSim` to a recording `interval` (time steps) and the series to
`keep`; components not in the spec record nothing::

    sim = history.configure(sim, {
        "<root>":   dict(interval=120, keep=["time_seconds", "offset_meters"]),
        "loco_con": dict(interval=120, keep=["pwr_fuel_watts"]),
    })

Components: `"<root>"` (the train; required), `"fric_brake"` (speed-limit
sims only), `"loco_con"`, `"loco_vec"`, `"fc"`, `"gen"`, `"edrv"`, `"res"`,
`"cat_int"`. Omit `keep` to record every series. A child's interval must be a
multiple of its parent's, and the step counter `i` is always recorded. Call
before the walk; from `sim_manager`, use the `configure_sim` hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional

if TYPE_CHECKING:
    import altrios as alt

__all__ = ["configure", "COMPONENTS", "ROOT"]

ROOT = "<root>"

# Each component's gating ancestor. `fric_brake` hangs off the train sim
# directly; the powertrain sub-components hang off each locomotive. The
# intermediate `ConventionalLoco`/`DualModeLoco`/`HybridLoco`/
# `BatteryElectricLoco` wrappers own no `save_interval`, so they add no gate and
# do not appear here.
_PARENT: Dict[str, Optional[str]] = {
    ROOT: None,
    "fric_brake": ROOT,
    "loco_con": ROOT,
    "loco_vec": "loco_con",
    "fc": "loco_vec",
    "gen": "loco_vec",
    "edrv": "loco_vec",
    "res": "loco_vec",
    "cat_int": "loco_vec",
}

COMPONENTS = tuple(_PARENT)

_POWERTRAIN_PARTS = ("fc", "gen", "edrv", "res", "cat_int")


def _components(sim_dict: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Group a sim pydict's history-bearing components by name.

    Per-locomotive components contribute one dict per unit, all configured together.

    Args:
        sim_dict: Train sim pydict, as returned by `to_pydict()`.

    Returns:
        Component name to the list of matching component dicts.
    """
    found: Dict[str, List[Dict[str, Any]]] = {ROOT: [sim_dict]}

    fric_brake = sim_dict.get("fric_brake")
    if isinstance(fric_brake, dict):
        found["fric_brake"] = [fric_brake]

    loco_con = sim_dict.get("loco_con")
    if not isinstance(loco_con, dict):
        return found
    found["loco_con"] = [loco_con]

    locos = [lo for lo in (loco_con.get("loco_vec") or []) if isinstance(lo, dict)]
    if locos:
        found["loco_vec"] = locos

    for part in _POWERTRAIN_PARTS:
        matches = [
            variant[part]
            for loco in locos
            for variant in (loco.get("loco_type") or {}).values()
            if isinstance(variant, dict) and isinstance(variant.get(part), dict)
        ]
        if matches:
            found[part] = matches
    return found


def _check_intervals(intervals: Mapping[str, Optional[int]]) -> None:
    """Validate save intervals against the component tree.

    Args:
        intervals: Component name to save interval in time steps, or None.

    Raises:
        ValueError: When an interval is not a positive int, an ancestor records
            nothing, or an interval is not a multiple of its ancestor's.
    """
    for name, interval in intervals.items():
        if interval is None:
            continue
        if not isinstance(interval, int) or interval < 1:
            raise ValueError(
                f"interval for {name!r} must be a positive whole number of time "
                f"steps or None, got {interval!r}"
            )
        ancestor = _PARENT[name]
        while ancestor is not None:
            parent_interval = intervals.get(ancestor)
            if parent_interval is None:
                raise ValueError(
                    f"{name!r} is set to record every {interval} steps, but its "
                    f"ancestor {ancestor!r} is not recording at all. A component "
                    f"only records on steps where every ancestor does, so this "
                    f"would save nothing."
                )
            if interval % parent_interval:
                raise ValueError(
                    f"{name!r} records every {interval} steps and {ancestor!r} "
                    f"every {parent_interval}; a component only records where "
                    f"its ancestors do, so the effective interval would be "
                    f"lcm({interval}, {parent_interval}) -- coarser than either. "
                    f"Make {name!r} a multiple of {parent_interval}."
                )
            ancestor = _PARENT[ancestor]


def _apply_mask(
    component: Dict[str, Any], name: str, keep: Optional[Iterable[str]]
) -> None:
    """Restrict a component's recorded history series to `keep`.

    Args:
        component: Component pydict carrying a `history.save_fields` mask.
        name: Component name, used in error messages.
        keep: Series names to record; None leaves the mask unchanged.

    Raises:
        RuntimeError: When the component has no `history.save_fields` mask.
        ValueError: When `keep` names a series the component does not have.
    """
    if keep is None:
        return
    history = component.get("history")
    if not isinstance(history, dict):
        return
    mask = history.get("save_fields")
    if not isinstance(mask, dict):
        raise RuntimeError(
            f"{name!r} has no `history.save_fields`, so this build of altrios "
            "predates per-field history masking; rebuild the Rust extension "
            "(`maturin develop --release`)."
        )
    keep = list(keep)
    # `i` is always recorded and deliberately absent from the mask; naming it is
    # harmless and arguably clearer, so accept it rather than rejecting it.
    unknown = [field for field in keep if field != "i" and field not in mask]
    if unknown:
        raise ValueError(
            f"{name!r} has no history series {unknown}; available: "
            + ", ".join(sorted(mask))
        )
    for field in mask:
        mask[field] = field in keep


def configure(
    sim: alt.SpeedLimitTrainSim | alt.SetSpeedTrainSim,
    spec: Mapping[str, Mapping[str, Any]],
) -> alt.SpeedLimitTrainSim | alt.SetSpeedTrainSim:
    """Return `sim` recording only what `spec` asks for.

    Components absent from `spec` record nothing; `"<root>"` must be present.

    Args:
        sim: `SpeedLimitTrainSim` or `SetSpeedTrainSim`, before it is walked.
        spec: Component name to `{"interval": int | None, "keep": [series, ...]}`.
            `interval` is in time steps (one second); omitting `keep` records
            every series. Names: `"<root>"` (the train), `"fric_brake"`,
            `"loco_con"`, `"loco_vec"` (all locomotives) and the powertrain
            parts `"fc"`, `"gen"`, `"edrv"`, `"res"` and `"cat_int"`.

    Returns:
        A new sim of the same type with the intervals and masks applied.

    Raises:
        ValueError: When `spec` names an unknown component or omits `"<root>"`.
    """
    unknown = sorted(set(spec) - set(_PARENT))
    if unknown:
        raise ValueError(
            f"unknown component(s) {unknown}; expected any of "
            + ", ".join(repr(c) for c in COMPONENTS)
        )
    if ROOT not in spec:
        raise ValueError(
            f"{ROOT!r} must be in the spec: it gates every other component, so "
            "leaving it out would record nothing anywhere."
        )

    _check_intervals({name: cfg.get("interval") for name, cfg in spec.items()})

    sim_dict = sim.to_pydict()
    for name, components in _components(sim_dict).items():
        cfg = spec.get(name)
        for component in components:
            if cfg is None:
                component["save_interval"] = None
                continue
            component["save_interval"] = cfg.get("interval")
            _apply_mask(component, name, cfg.get("keep"))
    return type(sim).from_pydict(sim_dict)
