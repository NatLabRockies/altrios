"""Choose what a simulation records, per component and per series.

Saved history is the dominant memory cost of a walked train plan, and studies
differ enormously in how much of it they read. ALTRIOS has always let each
component carry its own ``save_interval``; what it could not do until now is
record only *some* series within a component, because ``save_interval`` gates
the whole ``HistoryVec`` push. A study that needs one number off the consist --
fuel power, say -- paid for all twenty of its series.

``configure`` sets both levers at once, from one total specification::

    from altrios import history

    sim = history.configure(sim, {
        "<root>":   dict(interval=120, keep=["time_seconds",
                                             "speed_meters_per_second",
                                             "offset_meters"]),
        "loco_con": dict(interval=120, keep=["pwr_fuel_watts",
                                             "energy_fuel_joules"]),
        "loco_vec": dict(interval=120, keep=["pwr_aux_watts"]),
    })

Components not named are switched off entirely. That is deliberate: a spec is a
complete statement of what a study reads, so adding a metric that needs a new
series fails loudly at configuration time rather than yielding a null column
several steps downstream.

Two rules the model imposes, both checked here rather than discovered later.

**Intervals compose down the tree.** ``save_state`` recurses into children
*inside* the parent's interval check, so a component only records on steps where
every ancestor also records -- the effective interval is the LCM of the chain.
Name an interval that is not a multiple of its parent's and you get something
coarser than either value you wrote, silently. ``configure`` rejects that
instead.

**The step counter is always recorded.** ``i`` has no mask entry and cannot be
switched off. It is what makes histories at different intervals joinable: a
coarser component's samples can be aligned to the train's by ``i``, where
aligning by position stops meaning anything as soon as two intervals differ.

Applies to a ``SpeedLimitTrainSim``. Call it through ``sim_manager``'s
``configure_sim`` hook, which runs after ``set_save_interval`` (which recurses
and would otherwise undo this) and before the walk, while history is still empty
and the round-trip is cheap.
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
        sim_dict: `SpeedLimitTrainSim` pydict, as returned by `to_pydict()`.

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
    sim: alt.SpeedLimitTrainSim, spec: Mapping[str, Mapping[str, Any]]
) -> alt.SpeedLimitTrainSim:
    """Return `sim` recording only what `spec` asks for.

    Components absent from `spec` record nothing; `"<root>"` must be present.

    Args:
        sim: `SpeedLimitTrainSim`, before it is walked.
        spec: Component name to `{"interval": int | None, "keep": [series, ...]}`.
            `interval` is in time steps (one second); omitting `keep` records
            every series. Names: `"<root>"` (the train), `"fric_brake"`,
            `"loco_con"`, `"loco_vec"` (all locomotives) and the powertrain
            parts `"fc"`, `"gen"`, `"edrv"`, `"res"` and `"cat_int"`.

    Returns:
        A new `SpeedLimitTrainSim` with the intervals and masks applied.

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
