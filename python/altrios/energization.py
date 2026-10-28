"""Utilities for placing energization (e.g. catenary or wireless power transfer) power limits on a network."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, List, Dict, Union

# local imports
import altrios as alt

if TYPE_CHECKING:
    import polars as pl

    from altrios.optimization.energization_placement import (
        EconomicParameters,
        EnergizationOptCosts,
        EnergizationPlacementResult,
        PieceBenefitContext,
        Substation,
    )


#: Default maximum real power (W) a substation can deliver when a
#: :class:`Substation` does not set its own ``max_power_watts``. 20 MVA is a
#: representative traction substation rating.
DEFAULT_SUBSTATION_MAX_POWER_WATTS = 20.0e6

#: Fraction of nominal voltage below which the pantograph voltage may not fall,
#: used to derate voltage-limited deliverable power. 0.7 corresponds to the EN
#: 50163 minimum of 17.5 kV on a 25 kV nominal system.
DEFAULT_MIN_PANTOGRAPH_VOLTAGE_FRACTION = 0.7

#: Default substation conversion efficiency (grid -> catenary) when a
#: :class:`Substation` does not set its own ``efficiency``. 0.99 is representative
#: of an AC traction substation (a step-down transformer, ~1% loss). DC substations
#: additionally rectify AC to DC and are typically a few percent less efficient
#: (~0.97); set ``Substation.efficiency`` accordingly for DC feeds.
DEFAULT_SUBSTATION_EFFICIENCY = 0.99

#: Literature (Route A) loop series impedance defaults per electrification type,
#: given as ``(resistance_ohms_per_km, reactance_ohms_per_km)`` of the catenary
#: loop (contact + messenger + return). Resistance drives energy loss; reactance
#: only affects voltage drop (hence voltage-limited capacity). Users may override
#: either value directly. ``ac_2x25_at`` values are the *effective* per-km
#: impedance of a 2x25 kV autotransformer feed (current largely runs in the
#: feeder, roughly halving effective impedance). ``ac_50kv`` shares the 25 kV
#: single-track geometry; its benefit comes from the ``1/V**2`` loss scaling.
#: The ``dc_*`` entries are DC systems: reactance is zero (no AC). ``dc_1500v`` and
#: ``dc_3kv`` are conventional low-voltage DC railways, whose heavier conductors /
#: conductor rail give low loop resistance. ``dc_9kv`` is a high-voltage DC (HVDC)
#: railway feed: the higher voltage carries the same power at lower current, so it
#: shares the lighter, standard-catenary single-track geometry (resistance near the
#: 25 kV AC value) while its ``1/V**2`` loss scaling and DC (zero-reactance) feed
#: allow much wider substation spacing. Pair each ``dc_*`` type with a matching DC
#: ``system_voltage_volts`` (1500, 3000, or 9000) and a lower ``Substation.efficiency``
#: to account for rectifier conversion loss.
ELECTRIFICATION_IMPEDANCE_DEFAULTS: Dict[str, Tuple[float, float]] = {
    "ac_25kv_1x25": (0.20, 0.50),
    "ac_2x25_at": (0.075, 0.15),
    "ac_50kv": (0.20, 0.50),
    "dc_1500v": (0.03, 0.0),
    "dc_3kv": (0.04, 0.0),
    "dc_9kv": (0.20, 0.0),
}

#: Default design per-train peak catenary power draw (W) used to size feed
#: distance and substation ratings. ~10 MW is a representative peak for a
#: partially electrified dual-mode freight consist (a handful of ~3.3 MW electric
#: locomotives); the ALTRIOS catenary demo uses 50 MW as an effectively unlimited
#: cap. Override per corridor/traffic assumption.
DEFAULT_DESIGN_TRAIN_POWER_WATTS = 10.0e6


@dataclass
class ElectrificationConfig:
    """A named electrification configuration (voltage + impedance + limits).

    Bundles the electrical assumptions that distinguish, e.g., 9 kV HVDC from
    25 kV or 50 kV AC. Impedance is stored per meter (loop series R/X). These are
    the physical inputs consumed by the line-loss model and by
    :func:`suggest_optimization_inputs`.
    """

    #: Human-readable configuration name.
    name: str
    #: Nominal catenary voltage (V).
    system_voltage_volts: float
    #: Key into :data:`ELECTRIFICATION_IMPEDANCE_DEFAULTS` (also passed straight
    #: through to :func:`apply_line_losses` as ``electrification_type``).
    electrification_type: str
    #: Loop series resistance per meter (ohm/m).
    resistance_ohms_per_meter: float
    #: Loop series reactance per meter (ohm/m); ``0.0`` for DC.
    reactance_ohms_per_meter: float
    #: Grid -> catenary substation conversion efficiency (transformer, plus
    #: rectifier for DC).
    substation_efficiency: float = DEFAULT_SUBSTATION_EFFICIENCY
    #: On-board FRONT-END (pantograph -> traction DC link) conversion efficiency,
    #: i.e. the stage UPSTREAM of the traction inverter/motor (which ALTRIOS models
    #: separately as ``ElectricDrivetrain.eta``, so it must NOT be included here to
    #: avoid double counting). AC locomotives carry a step-down transformer plus
    #: rectifier (~1.5% + ~2.0% loss => ~0.965); DC needs only a chopper/filter
    #: (~1.0% loss => ~0.99). The core sim uses a flat 0.99 regardless of type;
    #: this field lets the illustrative energy/opex model reflect the AC-vs-DC
    #: front-end difference.
    onboard_efficiency: float = 0.99
    #: Minimum allowed pantograph voltage as a fraction of nominal.
    min_pantograph_voltage_fraction: float = DEFAULT_MIN_PANTOGRAPH_VOLTAGE_FRACTION


def _config_from_defaults(
    name: str,
    voltage: float,
    etype: str,
    *,
    efficiency: float,
    onboard_efficiency: float,
    min_voltage_fraction: float,
) -> ElectrificationConfig:
    """Build an electrification config from the impedance table.

    Args:
        name: Configuration name.
        voltage: Nominal catenary voltage (V).
        etype: Key into :data:`ELECTRIFICATION_IMPEDANCE_DEFAULTS`.
        efficiency: Substation grid-to-catenary conversion efficiency.
        onboard_efficiency: On-board front-end conversion efficiency.
        min_voltage_fraction: Minimum pantograph voltage as a fraction of nominal.

    Returns:
        Config with the table's per-km impedances converted to ohm/m.
    """
    r_km, x_km = ELECTRIFICATION_IMPEDANCE_DEFAULTS[etype]
    return ElectrificationConfig(
        name=name,
        system_voltage_volts=voltage,
        electrification_type=etype,
        resistance_ohms_per_meter=r_km / 1000.0,
        reactance_ohms_per_meter=x_km / 1000.0,
        substation_efficiency=efficiency,
        onboard_efficiency=onboard_efficiency,
        min_pantograph_voltage_fraction=min_voltage_fraction,
    )


#: Built-in electrification presets keyed by a short name. AC systems use a 0.99
#: (transformer-only) substation efficiency, a ~0.965 on-board FRONT-END efficiency
#: (step-down transformer ~1.5% + rectifier ~2.0%), and the EN 50163 0.7
#: minimum-voltage fraction; DC systems use a lower substation efficiency (AC->DC
#: rectifier loss), a higher ~0.99 on-board front-end efficiency (chopper/filter,
#: no on-board transformer), and the ~0.67 DC minimum-voltage fraction. The on-board
#: values exclude the traction inverter/motor (modeled by ``ElectricDrivetrain``).
ELECTRIFICATION_CONFIGS: Dict[str, ElectrificationConfig] = {
    "ac_25kv": _config_from_defaults(
        "ac_25kv", 25_000.0, "ac_25kv_1x25",
        efficiency=0.99, onboard_efficiency=0.965, min_voltage_fraction=0.7,
    ),
    "ac_2x25": _config_from_defaults(
        "ac_2x25", 25_000.0, "ac_2x25_at",
        efficiency=0.99, onboard_efficiency=0.965, min_voltage_fraction=0.7,
    ),
    "ac_50kv": _config_from_defaults(
        "ac_50kv", 50_000.0, "ac_50kv",
        efficiency=0.99, onboard_efficiency=0.965, min_voltage_fraction=0.7,
    ),
    "dc_9kv": _config_from_defaults(
        "dc_9kv", 9_000.0, "dc_9kv",
        efficiency=0.97, onboard_efficiency=0.99, min_voltage_fraction=0.67,
    ),
    "dc_3kv": _config_from_defaults(
        "dc_3kv", 3_000.0, "dc_3kv",
        efficiency=0.975, onboard_efficiency=0.99, min_voltage_fraction=0.67,
    ),
    "dc_1500v": _config_from_defaults(
        "dc_1500v", 1_500.0, "dc_1500v",
        efficiency=0.975, onboard_efficiency=0.99, min_voltage_fraction=0.67,
    ),
}


@dataclass
class SuggestedOptimizationInputs:
    """Config-derived defaults for the energization optimizer and line-loss model.

    Produced by :func:`suggest_optimization_inputs`. The first block feeds the
    placement optimizer (:func:`energize_optimized` /
    ``EnergizationPlacementInputs``); the electrical block feeds
    :func:`apply_line_losses` and populates each ``Substation``.
    """

    #: -- Placement-optimizer inputs --
    #: Per-train power limit (W) to stamp on every energized segment.
    uniform_power_limit_watts: float
    #: Suggested substation feeding/spacing radius (miles).
    substation_feed_distance_miles: float
    #: -- Substation / line-loss inputs --
    #: Suggested substation power rating (W).
    substation_max_power_watts: float
    #: Suggested substation conversion efficiency.
    substation_efficiency: float
    #: On-board (pantograph -> traction) conversion efficiency (AC vs DC).
    onboard_efficiency: float
    #: Nominal catenary voltage (V).
    system_voltage_volts: float
    #: Electrification-type key for ``apply_line_losses``.
    electrification_type: str
    #: Loop series resistance per meter (ohm/m).
    resistance_ohms_per_meter: float
    #: Loop series reactance per meter (ohm/m).
    reactance_ohms_per_meter: float
    #: Minimum pantograph voltage fraction.
    min_pantograph_voltage_fraction: float
    #: The design per-train peak power (W) used to derive the values above.
    design_train_power_watts: float

    def optimizer_kwargs(self) -> dict[str, float]:
        """Return keyword arguments for :func:`energize_optimized`.

        Returns:
            ``uniform_power_limit_watts`` and ``substation_feed_distance_miles``.
        """
        return {
            "uniform_power_limit_watts": self.uniform_power_limit_watts,
            "substation_feed_distance_miles": self.substation_feed_distance_miles,
        }

    def line_loss_kwargs(self) -> dict[str, float | str]:
        """Return keyword arguments for :func:`apply_line_losses`.

        Returns:
            Voltage, electrification type, impedances, substation defaults and the
            minimum pantograph voltage fraction.
        """
        return {
            "system_voltage_volts": self.system_voltage_volts,
            "electrification_type": self.electrification_type,
            "resistance_ohms_per_meter": self.resistance_ohms_per_meter,
            "reactance_ohms_per_meter": self.reactance_ohms_per_meter,
            "default_substation_max_power_watts": self.substation_max_power_watts,
            "default_substation_efficiency": self.substation_efficiency,
            "min_pantograph_voltage_fraction": self.min_pantograph_voltage_fraction,
        }


def max_gap_miles_from_battery(
    usable_energy_kwh: float,
    avg_consumption_kwh_per_mile: float,
    *,
    reserve_fraction: float = 0.0,
) -> float:
    """Convert battery size into a maximum unelectrified gap.

    Assumes constant consumption per mile and a full recharge on any energized track.

    Args:
        usable_energy_kwh: Battery energy (kWh) available for traction.
        avg_consumption_kwh_per_mile: Average traction draw per corridor-mile
            (kWh/mi).
        reserve_fraction: Fraction of the pack withheld as reserve, in ``[0, 1)``.

    Returns:
        ``usable_energy_kwh * (1 - reserve_fraction) / avg_consumption_kwh_per_mile``
        (miles), for ``max_energized_gap_miles``.

    Raises:
        ValueError: If the consumption is not positive, the energy is negative or
            the reserve fraction is outside ``[0, 1)``.
    """
    if avg_consumption_kwh_per_mile <= 0.0:
        raise ValueError("avg_consumption_kwh_per_mile must be positive.")
    if usable_energy_kwh < 0.0:
        raise ValueError("usable_energy_kwh must be non-negative.")
    if not (0.0 <= reserve_fraction < 1.0):
        raise ValueError("reserve_fraction must be in [0, 1).")
    return usable_energy_kwh * (1.0 - reserve_fraction) / avg_consumption_kwh_per_mile


def max_gap_kwh_from_battery(
    usable_energy_kwh: float,
    *,
    reserve_fraction: float = 0.0,
) -> float:
    """Return the usable battery traction energy between charges.

    Args:
        usable_energy_kwh: Battery energy (kWh) available for traction.
        reserve_fraction: Fraction of the pack withheld as reserve, in ``[0, 1)``.

    Returns:
        ``usable_energy_kwh * (1 - reserve_fraction)`` (kWh), for
        ``max_energized_gap_kwh``.

    Raises:
        ValueError: If the energy is negative or the reserve fraction is outside
            ``[0, 1)``.
    """
    if usable_energy_kwh < 0.0:
        raise ValueError("usable_energy_kwh must be non-negative.")
    if not (0.0 <= reserve_fraction < 1.0):
        raise ValueError("reserve_fraction must be in [0, 1).")
    return usable_energy_kwh * (1.0 - reserve_fraction)


def min_recharge_miles_from_battery(
    usable_energy_kwh: float,
    net_charge_kwh_per_mile: float,
    *,
    recharge_fraction: float = 1.0,
    reserve_fraction: float = 0.0,
) -> float:
    """Size the energized run length that recharges the pack.

    Args:
        usable_energy_kwh: Usable battery capacity (kWh).
        net_charge_kwh_per_mile: Net charge banked per energized corridor-mile
            (kWh/mi), charge power minus traction draw; must be positive.
        recharge_fraction: Fraction of the usable pack the run must restore.
        reserve_fraction: Fraction of the pack withheld as reserve, in ``[0, 1)``.

    Returns:
        ``usable_energy_kwh * (1 - reserve_fraction) * recharge_fraction /
        net_charge_kwh_per_mile`` (miles), for ``min_energized_length_miles``.

    Raises:
        ValueError: If the net charge rate is not positive, the energy or the
            recharge fraction is negative, or the reserve fraction is outside
            ``[0, 1)``.
    """
    if net_charge_kwh_per_mile <= 0.0:
        raise ValueError("net_charge_kwh_per_mile must be positive.")
    if usable_energy_kwh < 0.0:
        raise ValueError("usable_energy_kwh must be non-negative.")
    if not (0.0 <= reserve_fraction < 1.0):
        raise ValueError("reserve_fraction must be in [0, 1).")
    if recharge_fraction < 0.0:
        raise ValueError("recharge_fraction must be non-negative.")
    return (
        usable_energy_kwh * (1.0 - reserve_fraction) * recharge_fraction
        / net_charge_kwh_per_mile
    )


def suggest_optimization_inputs(
    config: str | ElectrificationConfig,
    *,
    design_train_power_watts: float = DEFAULT_DESIGN_TRAIN_POWER_WATTS,
    concurrent_trains_per_substation: float = 2.0,
    min_feed_distance_miles: float | None = 1.0,
    max_feed_distance_miles: float | None = None,
) -> SuggestedOptimizationInputs:
    """Derive optimizer and line-loss inputs from an electrification choice.

    Feed distance is the single-sided voltage-drop limit ``V**2 (1 - f) / (|Z| P_d)``.

    Args:
        config: Preset key in :data:`ELECTRIFICATION_CONFIGS` (case-insensitive)
            or an explicit :class:`ElectrificationConfig`.
        design_train_power_watts: Peak per-train catenary draw (W) sustained at
            the coverage boundary; also the per-segment power limit.
        concurrent_trains_per_substation: Design-power trains one substation
            feeds at once; scales the substation rating.
        min_feed_distance_miles: Lower clamp on the feed distance (miles).
        max_feed_distance_miles: Upper clamp on the feed distance (miles).

    Returns:
        Suggested power limit, feed distance, substation rating and efficiency,
        plus the config's electrical assumptions.

    Raises:
        KeyError: If ``config`` names an unknown preset.
        ValueError: If ``design_train_power_watts`` is not positive.
    """
    if isinstance(config, str):
        key = config.lower()
        if key not in ELECTRIFICATION_CONFIGS:
            raise KeyError(
                f"Unknown electrification config {config!r}; choose from "
                f"{sorted(ELECTRIFICATION_CONFIGS)} or pass an ElectrificationConfig."
            )
        cfg = ELECTRIFICATION_CONFIGS[key]
    else:
        cfg = config

    if design_train_power_watts <= 0.0:
        raise ValueError("design_train_power_watts must be positive.")

    v = float(cfg.system_voltage_volts)
    f = float(cfg.min_pantograph_voltage_fraction)
    z_mag = math.hypot(
        float(cfg.resistance_ohms_per_meter), float(cfg.reactance_ohms_per_meter)
    )
    p_d = float(design_train_power_watts)

    if z_mag > 0.0:
        feed_m = v * v * (1.0 - f) / (z_mag * p_d)
    else:
        feed_m = float("inf")
    feed_mi = feed_m * alt.utilities.MI_PER_M
    if max_feed_distance_miles is not None:
        feed_mi = min(feed_mi, float(max_feed_distance_miles))
    if min_feed_distance_miles is not None:
        feed_mi = max(feed_mi, float(min_feed_distance_miles))

    return SuggestedOptimizationInputs(
        uniform_power_limit_watts=p_d,
        substation_feed_distance_miles=feed_mi,
        substation_max_power_watts=p_d * float(concurrent_trains_per_substation),
        substation_efficiency=float(cfg.substation_efficiency),
        onboard_efficiency=float(cfg.onboard_efficiency),
        system_voltage_volts=v,
        electrification_type=cfg.electrification_type,
        resistance_ohms_per_meter=float(cfg.resistance_ohms_per_meter),
        reactance_ohms_per_meter=float(cfg.reactance_ohms_per_meter),
        min_pantograph_voltage_fraction=f,
        design_train_power_watts=p_d,
    )




def energize_uniform(
    network: alt.Network,
    uniform_power_limit_watts: float,
) -> alt.Network:
    """Electrify every link at one power limit.

    Args:
        network: Network to electrify; a modified copy is returned.
        uniform_power_limit_watts: Power limit (W) applied over each full link.

    Returns:
        New network with a full-length ``cat_power_limits`` entry on every link.
    """
    network_dict = network.to_pydict()
    # Iterate through all links in the dictionary
    # network_dict is a tuple: (NetworkErrTol dict, [links])
    # Skip index 0 which is the dummy link
    for i in range(1, len(network_dict[1])):
        link = network_dict[1][i]
        # Add energization power limit spanning the entire link
        link['cat_power_limits'] = [{
            'offset_start_meters': 0.0,
            'offset_end_meters': link['length_meters'],
            'power_limit_watts': uniform_power_limit_watts,
            'district_id': None
        }]

    return alt.Network.from_pydict(network_dict)


def energize_high_grade(
    network: alt.Network,
    uniform_power_limit_watts: float,
    grade_threshold: float
) -> alt.Network:
    """Electrify links whose maximum grade meets a threshold.

    Args:
        network: Network to electrify; a modified copy is returned.
        uniform_power_limit_watts: Power limit (W) applied over each qualifying
            link.
        grade_threshold: Minimum value of a link's maximum grade (rise/run) for
            it to be electrified.

    Returns:
        New network with full-length ``cat_power_limits`` on qualifying links.
    """
    network_dict = network.to_pydict()
    for i in range(1, len(network_dict[1])):
        link = network_dict[1][i]
        if len(link['elevs']) >= 2:
            max_grade = -float('inf')
            for j in range(len(link['elevs']) - 1):
                grade = (link["elevs"][j + 1]['elev_meters'] - link["elevs"][j]['elev_meters']) / (link["elevs"][j + 1]['offset_meters'] - link["elevs"][j]['offset_meters'])
                max_grade = max(max_grade, grade)
            if max_grade >= grade_threshold:
                link['cat_power_limits'] = [{
                    'offset_start_meters': 0.0,
                    'offset_end_meters': link['length_meters'],
                    'power_limit_watts': uniform_power_limit_watts,
                    'district_id': None
                }]

    return alt.Network.from_pydict(network_dict)


def energize_optimized(
    network: alt.Network,
    substations: list[Substation],
    uniform_power_limit_watts: float,
    *,
    costs: EnergizationOptCosts | None = None,
    budget: float | None = None,
    grade_threshold: float = 0.01,
    high_grade_benefit_weight: float = 10.0,
    substation_feed_distance_miles: float = 20.0,
    require_high_grade_coverage: bool = False,
    est_time_nets: list[alt.EstTimeNet] | None = None,
    excluded_link_idxs: list[int] | None = None,
    structure_spacing_miles: float | None = None,
    corridor_width_miles: float = 0.05,
    electrify_whole_site: bool = False,
    min_energized_length_miles: float | None = None,
    max_energized_gap_miles: float | None = None,
    max_energized_gap_kwh: float | None = None,
    gap_energy_kwh_per_mile_by_link: dict[int, float] | None = None,
    charge_energy_kwh_per_mile_by_link: dict[int, float] | None = None,
    regen_energy_kwh_per_mile_by_link: dict[int, float] | None = None,
    battery_soc_state_model: bool = False,
    objective: str = "max_benefit",
    economics: EconomicParameters | None = None,
    piece_opex_fn: Callable[[PieceBenefitContext], float] | None = None,
    piece_benefit_fn: Callable[[PieceBenefitContext], float] | None = None,
    time_limit_seconds: float | None = None,
    mip_gap: float | None = None,
    tee: bool = False,
    solver_threads: int | None = None,
    warmstart: bool = False,
) -> EnergizationPlacementResult:
    """Optimize energization placement with a MILP.

    Selects track pieces and substations under a budget and feeding constraints.

    Args:
        network: Network on which energization is placed; a copy is returned in
            the result.
        substations: Candidate and existing transmission substations.
        uniform_power_limit_watts: Power limit (W) on every energized segment.
        costs: Cost coefficients; ``None`` means all zero.
        budget: Capital budget cap ($); ``None`` leaves spending unconstrained.
        grade_threshold: Grade (rise/run) at or above which track is high-grade.
        high_grade_benefit_weight: Objective weight of high-grade track relative
            to 1.0 for other track.
        substation_feed_distance_miles: Maximum track distance (miles) from an
            energized segment to a built substation.
        require_high_grade_coverage: Whether every high-grade segment must be
            energized.
        est_time_nets: Train paths that restrict candidates to traversed links.
        excluded_link_idxs: Link indices removed from candidacy.
        structure_spacing_miles: Piece length (miles) for structure-site mode;
            ``None`` treats each link as one piece.
        corridor_width_miles: Lateral distance (miles) within which parallel
            pieces share one structure site.
        electrify_whole_site: Whether energizing a site wires every piece at it,
            replacing per-piece binaries with site binaries.
        min_energized_length_miles: Minimum interior energized run (corridor
            miles); requires structure-site mode.
        max_energized_gap_miles: Maximum unelectrified mainline run (corridor
            miles); requires structure-site mode with a contiguous mainline.
        max_energized_gap_kwh: Usable battery energy (kWh) between charges;
            bounds each unelectrified window's drain and overrides the miles gap.
        gap_energy_kwh_per_mile_by_link: Traction drain per corridor-mile
            (kWh/mi) by link index; required with ``max_energized_gap_kwh``.
        charge_energy_kwh_per_mile_by_link: Net charge per energized
            corridor-mile (kWh/mi) by link index; enables charge-credited gaps.
        regen_energy_kwh_per_mile_by_link: Regenerative recovery per
            corridor-mile (kWh/mi) by link index, netted against drain.
        battery_soc_state_model: Whether to use the explicit state-of-charge
            recursion instead of covering gap rows.
        objective: ``"max_benefit"`` or ``"min_cost"`` (capital plus discounted
            operating cost).
        economics: Horizon and discount rate for ``"min_cost"``.
        piece_opex_fn: Annual net operating cost ($/yr) per piece; required for
            ``"min_cost"``. See :func:`make_piece_opex_fn`.
        piece_benefit_fn: Per-piece benefit overriding the grade heuristic in
            ``"max_benefit"``.
        time_limit_seconds: Solver wall-clock limit (s); ``None`` for no limit.
        mip_gap: Relative MIP gap tolerance; ``None`` uses the solver default.
        tee: Whether to stream solver output.
        solver_threads: Thread cap for HiGHS; ``None`` uses all cores.
        warmstart: Whether to seed a fully wired feasible MIP start.

    Returns:
        Result with the energized network, built substations, costs and solver
        diagnostics; ``network`` is ``None`` when no incumbent was found.
    """
    from altrios.optimization import energization_placement as cp

    inputs = cp.EnergizationPlacementInputs(
        network=network,
        substations=list(substations),
        costs=costs if costs is not None else cp.EnergizationOptCosts(),
        budget=budget,
        uniform_power_limit_watts=uniform_power_limit_watts,
        grade_threshold=grade_threshold,
        high_grade_benefit_weight=high_grade_benefit_weight,
        substation_feed_distance_miles=substation_feed_distance_miles,
        require_high_grade_coverage=require_high_grade_coverage,
        est_time_nets=list(est_time_nets) if est_time_nets is not None else [],
        excluded_link_idxs=(list(excluded_link_idxs)
                            if excluded_link_idxs is not None else None),
        structure_spacing_miles=structure_spacing_miles,
        corridor_width_miles=corridor_width_miles,
        electrify_whole_site=electrify_whole_site,
        min_energized_length_miles=min_energized_length_miles,
        max_energized_gap_miles=max_energized_gap_miles,
        max_energized_gap_kwh=max_energized_gap_kwh,
        gap_energy_kwh_per_mile_by_link=gap_energy_kwh_per_mile_by_link,
        charge_energy_kwh_per_mile_by_link=charge_energy_kwh_per_mile_by_link,
        regen_energy_kwh_per_mile_by_link=regen_energy_kwh_per_mile_by_link,
        battery_soc_state_model=battery_soc_state_model,
        objective=objective,
        economics=economics,
        piece_opex_fn=piece_opex_fn,
        piece_benefit_fn=piece_benefit_fn,
    )
    return cp.optimize_energization_placement(
        inputs,
        time_limit_seconds=time_limit_seconds,
        mip_gap=mip_gap,
        tee=tee,
        solver_threads=solver_threads,
        warmstart=warmstart,
    )


PriceByLink = Union[float, Dict[int, float]]


def _price_lookup(price: PriceByLink) -> Callable[[int], float]:
    """Build a link-index-to-price lookup.

    Args:
        price: One corridor-wide price or a price per link index.

    Returns:
        Function returning the link's price; unmapped links take the map's mean.
    """
    if isinstance(price, dict):
        default = (sum(price.values()) / len(price)) if price else 0.0
        return lambda idx: float(price.get(idx, default))
    value = float(price)
    return lambda idx: value


def make_piece_opex_fn(
    *,
    diesel_gallons_per_track_mile_by_link: dict[int, float],
    electricity_kwh_per_track_mile_by_link: dict[int, float],
    diesel_price_usd_per_gallon: PriceByLink,
    electricity_price_usd_per_kwh: PriceByLink,
    default_diesel_gallons_per_track_mile: float = 0.0,
    default_electricity_kwh_per_track_mile: float = 0.0,
) -> Callable[[PieceBenefitContext], float]:
    """Build a per-piece annual operating-cost function.

    Opex is electricity bought minus diesel displaced; negative values are savings.

    Args:
        diesel_gallons_per_track_mile_by_link: Annual diesel displaced per
            track-mile (gal/mi), keyed by link index.
        electricity_kwh_per_track_mile_by_link: Annual grid-side electricity
            bought per track-mile (kWh/mi), keyed by link index.
        diesel_price_usd_per_gallon: Diesel price ($/gal), corridor-wide or per
            link index.
        electricity_price_usd_per_kwh: Electricity price ($/kWh), corridor-wide
            or per link index.
        default_diesel_gallons_per_track_mile: Diesel density for unmapped links.
        default_electricity_kwh_per_track_mile: Electricity density for unmapped
            links.

    Returns:
        Function mapping a :class:`PieceBenefitContext` to annual net operating
        cost ($/yr), for ``objective="min_cost"``.
    """
    diesel_price = _price_lookup(diesel_price_usd_per_gallon)
    elec_price = _price_lookup(electricity_price_usd_per_kwh)

    def _opex(ctx: PieceBenefitContext) -> float:
        """Price one piece's annual electricity minus displaced diesel."""
        gal_per_mi = diesel_gallons_per_track_mile_by_link.get(
            ctx.link_idx, default_diesel_gallons_per_track_mile
        )
        kwh_per_mi = electricity_kwh_per_track_mile_by_link.get(
            ctx.link_idx, default_electricity_kwh_per_track_mile
        )
        gallons = gal_per_mi * ctx.length_miles
        kwh = kwh_per_mi * ctx.length_miles
        return elec_price(ctx.link_idx) * kwh - diesel_price(ctx.link_idx) * gallons

    return _opex


def make_bel_piece_opex_fn(
    *,
    traction_kwh_per_track_mile_by_link: dict[int, float],
    electricity_price_usd_per_kwh: PriceByLink,
    battery_roundtrip_efficiency: float,
    default_traction_kwh_per_track_mile: float = 0.0,
) -> Callable[[PieceBenefitContext], float]:
    """Build a per-piece operating-cost function for battery-electric fleets.

    Energizing a piece avoids battery round-trip loss on its traction energy.

    Args:
        traction_kwh_per_track_mile_by_link: Annual traction energy per
            track-mile (kWh/mi), keyed by link index.
        electricity_price_usd_per_kwh: Electricity price ($/kWh), corridor-wide
            or per link index.
        battery_roundtrip_efficiency: Pack round-trip efficiency in ``(0, 1]``.
        default_traction_kwh_per_track_mile: Traction density for unmapped links.

    Returns:
        Function mapping a :class:`PieceBenefitContext` to
        ``-price * (1 / efficiency - 1) * annual_kwh`` ($/yr, a saving).

    Raises:
        ValueError: If ``battery_roundtrip_efficiency`` is outside ``(0, 1]``.
    """
    if not 0.0 < battery_roundtrip_efficiency <= 1.0:
        raise ValueError("battery_roundtrip_efficiency must be in (0, 1].")
    loss_factor = (1.0 / battery_roundtrip_efficiency) - 1.0
    elec_price = _price_lookup(electricity_price_usd_per_kwh)

    def _opex(ctx: PieceBenefitContext) -> float:
        """Return the avoided round-trip loss cost of one piece."""
        kwh_per_mi = traction_kwh_per_track_mile_by_link.get(
            ctx.link_idx, default_traction_kwh_per_track_mile
        )
        e_p = kwh_per_mi * ctx.length_miles
        return -elec_price(ctx.link_idx) * loss_factor * e_p

    return _opex


def _compute_energized_intervals(
    start_pos_in_cycle: float,
    link_length: float,
    gap_meters: float,
    cat_meters: float,
) -> tuple[list[tuple[float, float]], float]:
    """Lay a repeating gap/energized pattern across one link.

    Args:
        start_pos_in_cycle: Position (m) within the gap/energized cycle at the
            link start.
        link_length: Link length (m).
        gap_meters: Unelectrified length (m) of each cycle.
        cat_meters: Electrified length (m) of each cycle.

    Returns:
        Energized ``(offset_start, offset_end)`` pairs (m from the link start)
        and the cycle position at the link end.
    """
    cycle_length = gap_meters + cat_meters
    intervals: List[Tuple[float, float]] = []
    pos_in_cycle = start_pos_in_cycle % cycle_length
    link_offset = 0.0
    while link_offset < link_length - 1e-9:
        if pos_in_cycle < gap_meters:
            advance = min(gap_meters - pos_in_cycle, link_length - link_offset)
            link_offset += advance
            pos_in_cycle += advance
            if pos_in_cycle >= gap_meters - 1e-9:
                pos_in_cycle = gap_meters
        else:
            advance = min(cycle_length - pos_in_cycle, link_length - link_offset)
            if advance > 1e-9:
                intervals.append((link_offset, link_offset + advance))
            link_offset += advance
            pos_in_cycle += advance
            if pos_in_cycle >= cycle_length - 1e-9:
                pos_in_cycle = 0.0
    return intervals, pos_in_cycle


def _longest_mainline_path(
    link_by_idx: dict[int, dict], candidates: set[int]
) -> list[int]:
    """Return the longest through-route found by following ``idx_next``.

    Args:
        link_by_idx: Link dicts keyed by ``idx_curr``.
        candidates: Link indices eligible for the mainline.

    Returns:
        Ordered link indices of the longest walk starting at a link that no
        candidate points to, or at any candidate when every link is pointed to.
    """
    is_target: set = set()
    for a in candidates:
        link = link_by_idx[a]
        for key in ("idx_next", "idx_next_alt"):
            t = link.get(key, 0)
            if t in candidates:
                is_target.add(t)
    heads = [a for a in candidates if a not in is_target] or list(candidates)
    best: List[int] = []
    best_len = -1.0
    for start in heads:
        path: List[int] = []
        seen: set = set()
        cur = start
        total = 0.0
        while cur in candidates and cur not in seen:
            seen.add(cur)
            path.append(cur)
            total += float(link_by_idx[cur]["length_meters"])
            cur = link_by_idx[cur].get("idx_next", 0)
        if total > best_len:
            best_len = total
            best = path
    return best


def energize_even_spacing(
    network: alt.Network,
    uniform_power_limit_watts: float,
    spacing_meters: tuple[float, float],
    *,
    sample_spacing_meters: float = 200.0,
    corridor_match_meters: float = 1000.0,
) -> alt.Network:
    """Place energization in a repeating gap/energized corridor pattern.

    Parallel tracks and sidings take the pattern phase of the nearest mainline point.

    Args:
        network: Network to electrify; a modified copy is returned. Links need
            lat/lon geometry.
        uniform_power_limit_watts: Power limit (W) on every energized interval.
        spacing_meters: ``(gap_meters, energized_meters)`` of the repeating
            pattern, measured along the corridor.
        sample_spacing_meters: Along-track piece length (m) used to reduce the
            network to virtual links.
        corridor_match_meters: Maximum lateral distance (m) from the mainline for
            a link to receive the pattern.

    Returns:
        New network with energized sub-link intervals applied.

    Raises:
        ValueError: If no link has lat/lon geometry or no mainline can be built.
    """
    import math
    import numpy as np
    from scipy.spatial import cKDTree
    from altrios.optimization.energization_placement import _densify_links

    gap_meters, cat_meters = spacing_meters
    cycle = gap_meters + cat_meters
    network_dict = network.to_pydict()
    link_by_idx: Dict[int, dict] = {
        link["idx_curr"]: link for link in network_dict[1][1:]
    }
    candidate_idxs = list(link_by_idx.keys())

    # 1) Reduce to uniform virtual-link pieces (shared representation).
    p_link, p_off0, p_off1, p_lat, p_lon, p_len = _densify_links(
        network_dict, candidate_idxs, sample_spacing_meters
    )
    n = len(p_link)
    if n == 0:
        raise ValueError(
            "energize_even_spacing requires lat/lon geometry on the network links."
        )
    link_pieces: Dict[int, List[int]] = {}
    for i in range(n):
        link_pieces.setdefault(p_link[i], []).append(i)

    # 2) Reference mainline = longest idx_next through-route; give its pieces
    #    cumulative corridor arc-length.
    mainline_links = _longest_mainline_path(link_by_idx, set(candidate_idxs))
    mainline_pieces: List[int] = []
    for link_idx in mainline_links:
        mainline_pieces.extend(link_pieces.get(link_idx, []))
    if not mainline_pieces:
        raise ValueError("Could not build a mainline corridor from the network.")
    arc_start = [0.0] * n
    on_mainline = [False] * n
    arc = 0.0
    for i in mainline_pieces:
        arc_start[i] = arc
        on_mainline[i] = True
        arc += p_len[i]

    # 3) Project midpoints to a local meter frame; KD-tree of mainline pieces so
    #    every other track inherits the nearest mainline arc-length.
    lat_arr = np.asarray(p_lat)
    lon_arr = np.asarray(p_lon)
    lat0 = float(lat_arr.mean())
    lon0 = float(lon_arr.mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    xy = np.column_stack(
        [(lon_arr - lon0) * m_per_deg_lon, (lat_arr - lat0) * m_per_deg_lat]
    )
    main_xy = xy[mainline_pieces]
    main_tree = cKDTree(main_xy)
    main_arc = [arc_start[i] for i in mainline_pieces]
    mainline_link_set = set(mainline_links)

    # 4) For each link, find the corridor arc-length at its start, then compute the
    #    exact energized intervals over the whole link in ONE continuous pass. This
    #    keeps each link's intervals contiguous (the cycle phase flows along the
    #    link), so parallel tracks/sidings inherit the corridor pattern and no link
    #    gets artificially fragmented zones. Mainline links use their own cumulative
    #    arc; every other track inherits the nearest mainline arc by lat/lon.
    link_intervals: Dict[int, List[Tuple[float, float]]] = {}
    for link_idx, piece_list in link_pieces.items():
        i0 = piece_list[0]  # first piece, near the link start (offset order)
        if link_idx in mainline_link_set:
            arc0 = arc_start[i0]
        else:
            dist, k = main_tree.query(xy[i0])
            if dist > corridor_match_meters:
                continue  # too far from the corridor -> leave un-energized
            arc0 = main_arc[k]
        length = float(link_by_idx[link_idx]["length_meters"])
        segs, _ = _compute_energized_intervals(
            arc0 % cycle, length, gap_meters, cat_meters
        )
        if segs:
            link_intervals[link_idx] = segs

    # 5) Apply the (already contiguous, sorted) intervals to a copy of the network.
    for i in range(1, len(network_dict[1])):
        link = network_dict[1][i]
        link_idx = link["idx_curr"]
        if link_idx not in link_intervals:
            continue
        link["cat_power_limits"] = [
            {
                "offset_start_meters": start,
                "offset_end_meters": end,
                "power_limit_watts": uniform_power_limit_watts,
                "district_id": None,
            }
            for start, end in link_intervals[link_idx]
        ]

    return alt.Network.from_pydict(network_dict)


def _latlon_at_offset(
    link: dict, offset_meters: float
) -> tuple[float, float] | None:
    """Interpolate a link's (lat, lon) at an along-track offset.

    Args:
        link: Link dict with a ``headings`` geometry list.
        offset_meters: Offset from the link start (m); clamped to the geometry.

    Returns:
        Interpolated (lat, lon), or ``None`` when the link has no coordinates.
    """
    headings = link.get("headings") or []
    pts = [
        (float(h["offset_meters"]), float(h["lat"]), float(h["lon"]))
        for h in headings
        if h.get("lat") is not None and h.get("lon") is not None
    ]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0][1], pts[0][2]
    if offset_meters <= pts[0][0]:
        return pts[0][1], pts[0][2]
    if offset_meters >= pts[-1][0]:
        return pts[-1][1], pts[-1][2]
    for (o0, la0, lo0), (o1, la1, lo1) in zip(pts[:-1], pts[1:]):
        if o0 <= offset_meters <= o1:
            span = o1 - o0
            frac = 0.0 if span == 0 else (offset_meters - o0) / span
            return la0 + frac * (la1 - la0), lo0 + frac * (lo1 - lo0)
    return pts[-1][1], pts[-1][2]


def _build_corridor_arc(
    network_dict: tuple,
    sample_spacing_meters: float,
) -> tuple[dict[int, float], set[int], Callable[[float, float], float]]:
    """Build a corridor arc-length reference from network geometry.

    Args:
        network_dict: ``network.to_pydict()`` output.
        sample_spacing_meters: Along-track piece length (m) for densification.

    Returns:
        Arc position (m) at the start of each mainline link, the mainline link
        set, and a function mapping (lat, lon) to the nearest mainline arc
        position (m).

    Raises:
        ValueError: If no link has lat/lon geometry or no mainline can be built.
    """
    import math
    import numpy as np
    from scipy.spatial import cKDTree
    from altrios.optimization.energization_placement import _densify_links

    link_by_idx: Dict[int, dict] = {
        link["idx_curr"]: link for link in network_dict[1][1:]
    }
    candidate_idxs = list(link_by_idx.keys())
    p_link, p_off0, _p_off1, p_lat, p_lon, p_len = _densify_links(
        network_dict, candidate_idxs, sample_spacing_meters
    )
    n = len(p_link)
    if n == 0:
        raise ValueError(
            "Line-loss modeling requires lat/lon geometry on the network links."
        )
    link_pieces: Dict[int, List[int]] = {}
    for i in range(n):
        link_pieces.setdefault(p_link[i], []).append(i)

    mainline_links = _longest_mainline_path(link_by_idx, set(candidate_idxs))
    mainline_link_set = set(mainline_links)

    # Cumulative corridor arc-length along the mainline, per piece and per link.
    arc_start_of_link: Dict[int, float] = {}
    mainline_pieces: List[int] = []
    arc = 0.0
    for link_idx in mainline_links:
        arc_start_of_link[link_idx] = arc
        for i in link_pieces.get(link_idx, []):
            mainline_pieces.append(i)
        arc += float(link_by_idx[link_idx]["length_meters"])
    if not mainline_pieces:
        raise ValueError("Could not build a mainline corridor from the network.")

    # Local equirectangular projection for KD-tree distance in meters.
    lat_arr = np.asarray(p_lat)
    lon_arr = np.asarray(p_lon)
    lat0 = float(lat_arr.mean())
    lon0 = float(lon_arr.mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    def _to_xy(lat: float, lon: float) -> tuple[float, float]:
        """Project (lat, lon) to local equirectangular meters."""
        return ((lon - lon0) * m_per_deg_lon, (lat - lat0) * m_per_deg_lat)

    main_xy = np.array([_to_xy(p_lat[i], p_lon[i]) for i in mainline_pieces])
    main_tree = cKDTree(main_xy)
    # Arc-length at each mainline piece midpoint = link arc start + piece start.
    main_arc = []
    for i in mainline_pieces:
        main_arc.append(arc_start_of_link[p_link[i]] + float(p_off0[i]))

    def project_fn(lat: float, lon: float) -> float:
        """Return the nearest mainline arc position (m)."""
        _dist, k = main_tree.query(_to_xy(lat, lon))
        return float(main_arc[k])

    return arc_start_of_link, mainline_link_set, project_fn


def apply_line_losses(
    network: alt.Network,
    substations: list[Substation],
    *,
    system_voltage_volts: float = 25_000.0,
    resistance_ohms_per_meter: float | None = None,
    reactance_ohms_per_meter: float | None = None,
    electrification_type: str = "ac_25kv_1x25",
    default_substation_max_power_watts: float = DEFAULT_SUBSTATION_MAX_POWER_WATTS,
    default_substation_efficiency: float = DEFAULT_SUBSTATION_EFFICIENCY,
    min_pantograph_voltage_fraction: float = DEFAULT_MIN_PANTOGRAPH_VOLTAGE_FRACTION,
    base_power_limit_watts: float | None = None,
    sample_spacing_meters: float = 200.0,
    corridor_match_meters: float = 1000.0,
) -> tuple[alt.Network, dict[str, dict]]:
    """Derate energized intervals for catenary line losses.

    Each interval is fed from the nearest substations along the corridor arc.

    Args:
        network: Network already carrying ``cat_power_limits``; a modified copy
            is returned.
        substations: Feeding substations; each needs lat/lon or a ``link_idx``.
        system_voltage_volts: Nominal catenary voltage (V) for substations that
            set none; also written to the network config.
        resistance_ohms_per_meter: Loop series resistance (ohm/m); ``None``
            takes the ``electrification_type`` default.
        reactance_ohms_per_meter: Loop series reactance (ohm/m); ``None`` takes
            the ``electrification_type`` default.
        electrification_type: Key into :data:`ELECTRIFICATION_IMPEDANCE_DEFAULTS`.
        default_substation_max_power_watts: Rating (W) for substations without
            ``max_power_watts``.
        default_substation_efficiency: Grid-to-catenary efficiency for
            substations without ``efficiency``.
        min_pantograph_voltage_fraction: Minimum pantograph voltage as a
            fraction of nominal, for the voltage-drop power limit.
        base_power_limit_watts: Pre-loss rating (W) to derate from; ``None``
            uses each interval's existing ``power_limit_watts``.
        sample_spacing_meters: Along-track piece length (m) for the corridor arc.
        corridor_match_meters: Lateral match tolerance (m) for off-mainline
            intervals; currently unused.

    Returns:
        The derated network, whose intervals carry ``resistance_to_source_ohms``,
        ``reactance_to_source_ohms``, ``voltage_volts``, ``district_id``,
        ``district_id_secondary`` and ``feed_fraction`` and whose power limit is
        the least of the base rating, the substation-capacity limit and the
        voltage-drop limit; and substation metadata keyed by district id with
        ``grid_emissions_region``, ``max_power_watts``, ``voltage_volts``,
        ``efficiency`` and ``arc_meters``.

    Raises:
        ValueError: If ``electrification_type`` is unknown while an impedance is
            ``None``, or a substation has no resolvable location.
    """
    import math
    from altrios.optimization.energization_placement import _link_midpoint_latlon

    if resistance_ohms_per_meter is None or reactance_ohms_per_meter is None:
        if electrification_type not in ELECTRIFICATION_IMPEDANCE_DEFAULTS:
            raise ValueError(
                f"Unknown electrification_type {electrification_type!r}; "
                f"choose from {sorted(ELECTRIFICATION_IMPEDANCE_DEFAULTS)} or pass "
                "resistance_ohms_per_meter and reactance_ohms_per_meter explicitly."
            )
        r_km, x_km = ELECTRIFICATION_IMPEDANCE_DEFAULTS[electrification_type]
        if resistance_ohms_per_meter is None:
            resistance_ohms_per_meter = r_km / 1000.0
        if reactance_ohms_per_meter is None:
            reactance_ohms_per_meter = x_km / 1000.0
    r_per_m = float(resistance_ohms_per_meter)
    x_per_m = float(reactance_ohms_per_meter)

    network_dict = network.to_pydict()
    link_by_idx: Dict[int, dict] = {
        link["idx_curr"]: link for link in network_dict[1][1:]
    }

    arc_start_of_link, mainline_link_set, project_fn = _build_corridor_arc(
        network_dict, sample_spacing_meters
    )

    # Resolve each substation to a corridor arc-length and collect metadata.
    sub_records: List[dict] = []
    substation_meta: Dict[str, dict] = {}
    for sub in substations:
        if sub.lat is not None and sub.lon is not None:
            pos: Optional[Tuple[float, float]] = (float(sub.lat), float(sub.lon))
        elif sub.link_idx is not None and sub.link_idx in link_by_idx:
            pos = _link_midpoint_latlon(link_by_idx[sub.link_idx])
        else:
            pos = None
        if pos is None:
            raise ValueError(
                f"Substation {sub.id!r} has no resolvable location "
                "(provide lat/lon or a valid link_idx)."
            )
        arc = project_fn(pos[0], pos[1])
        did = str(sub.id)
        max_p = (
            sub.max_power_watts
            if sub.max_power_watts is not None
            else default_substation_max_power_watts
        )
        volt = sub.voltage_volts if sub.voltage_volts is not None else system_voltage_volts
        eff = (
            sub.efficiency
            if getattr(sub, "efficiency", None) is not None
            else default_substation_efficiency
        )
        rec = {
            "id": did,
            "arc": arc,
            "max_power_watts": float(max_p),
            "voltage_volts": float(volt),
            "efficiency": float(eff),
            "grid_emissions_region": sub.grid_emissions_region,
        }
        sub_records.append(rec)
        substation_meta[did] = {
            "grid_emissions_region": sub.grid_emissions_region,
            "max_power_watts": float(max_p),
            "voltage_volts": float(volt),
            "efficiency": float(eff),
            "arc_meters": arc,
        }
    sub_records.sort(key=lambda s: s["arc"])
    sub_arcs = [s["arc"] for s in sub_records]

    def _feed(x: float) -> tuple[dict | None, dict | None]:
        """Return the left and right feeding substations at ``x``."""
        import bisect

        left = None
        right = None
        j = bisect.bisect_right(sub_arcs, x)
        if j > 0:
            left = sub_records[j - 1]
        if j < len(sub_records):
            right = sub_records[j]
        # If the position coincides with a substation, treat it as directly fed.
        return left, right

    v_min_frac = float(min_pantograph_voltage_fraction)

    for i in range(1, len(network_dict[1])):
        link = network_dict[1][i]
        cpls = link.get("cat_power_limits") or []
        if not cpls:
            continue
        link_idx = link["idx_curr"]
        on_mainline = link_idx in mainline_link_set
        for cpl in cpls:
            start = float(cpl["offset_start_meters"])
            end = float(cpl["offset_end_meters"])
            mid = 0.5 * (start + end)
            if on_mainline:
                x = arc_start_of_link[link_idx] + mid
            else:
                ll = _latlon_at_offset(link, mid)
                if ll is None:
                    continue
                x = project_fn(ll[0], ll[1])

            left, right = _feed(x)
            r_eff = 0.0
            x_eff = 0.0
            feeding = None
            secondary = None
            frac_primary = 1.0
            if left is not None and right is not None:
                d1 = max(x - left["arc"], 0.0)
                d2 = max(right["arc"] - x, 0.0)
                span = d1 + d2
                if span > 0.0:
                    r_eff = r_per_m * d1 * d2 / span
                    x_eff = x_per_m * d1 * d2 / span
                    # Current divides inversely to path resistance (distance), so
                    # the nearer substation supplies the larger share. Primary =
                    # nearer; its share = distance to the *other* substation / D.
                    if d1 <= d2:
                        feeding, secondary = left, right
                        frac_primary = d2 / span
                    else:
                        feeding, secondary = right, left
                        frac_primary = d1 / span
                else:
                    feeding = left if d1 <= d2 else right
            elif left is not None:
                d1 = max(x - left["arc"], 0.0)
                r_eff = r_per_m * d1
                x_eff = x_per_m * d1
                feeding = left
            elif right is not None:
                d2 = max(right["arc"] - x, 0.0)
                r_eff = r_per_m * d2
                x_eff = x_per_m * d2
                feeding = right

            base = (
                float(base_power_limit_watts)
                if base_power_limit_watts is not None
                else float(cpl.get("power_limit_watts", 0.0))
            )
            if feeding is None:
                # No reachable substation: keep base rating, model no loss.
                cpl["resistance_to_source_ohms"] = 0.0
                cpl["reactance_to_source_ohms"] = 0.0
                cpl["voltage_volts"] = system_voltage_volts
                continue

            v = feeding["voltage_volts"]
            p_sub_max = feeding["max_power_watts"]
            # Substation-capacity-limited pantograph power: solve P + k P^2 = Pmax.
            k = r_eff / (v * v) if v > 0.0 else 0.0
            if k > 0.0:
                p_pow = (-1.0 + math.sqrt(1.0 + 4.0 * k * p_sub_max)) / (2.0 * k)
            else:
                p_pow = p_sub_max
            # Voltage-drop-limited power: V - (P/V)*|Z| >= V_min.
            z_eff = math.hypot(r_eff, x_eff)
            if z_eff > 0.0:
                p_volt = v * (v - v_min_frac * v) / z_eff
            else:
                p_volt = float("inf")

            new_power = min(base, p_pow, p_volt)
            cpl["power_limit_watts"] = max(new_power, 0.0)
            cpl["resistance_to_source_ohms"] = r_eff
            cpl["reactance_to_source_ohms"] = x_eff
            cpl["voltage_volts"] = v
            cpl["district_id"] = feeding["id"]
            if secondary is not None:
                cpl["district_id_secondary"] = secondary["id"]
                cpl["feed_fraction"] = frac_primary
            else:
                cpl["district_id_secondary"] = None
                cpl["feed_fraction"] = None

    network_dict[0]["system_voltage_volts"] = float(system_voltage_volts)
    return alt.Network.from_pydict(network_dict), substation_meta


def compute_line_losses(
    train_sim: alt.SpeedLimitTrainSim | dict,
    substation_meta: dict[str, dict] | None = None,
) -> pl.DataFrame:
    """Aggregate catenary line losses from a solved train simulation.

    Exact cumulative totals are apportioned to districts by sampled power history.

    Args:
        train_sim: Solved train simulation, or its ``to_pydict()``, saved with a
            per-step history.
        substation_meta: Metadata from :func:`apply_line_losses`, keyed by
            district id, giving each district's region and efficiency.

    Returns:
        One row per district with ``district_id``, ``grid_emissions_region``,
        ``line_loss_energy_joules``, ``catenary_energy_joules`` and
        ``substation_energy_joules`` (pantograph energy plus loss, divided by
        efficiency), ending with a ``"__total__"`` row.

    Raises:
        ValueError: If the history lacks train offsets or time steps.
    """
    import polars as pl

    ts = train_sim.to_pydict() if hasattr(train_sim, "to_pydict") else train_sim

    # Per-step time deltas and train position (down-sampled by save_interval).
    hist = ts.get("history", {})
    dt_s = hist.get("dt_seconds")
    offsets = hist.get("offset_meters")
    if offsets is None:
        offsets = hist.get("offset_back_meters")
    if offsets is None:
        raise ValueError(
            "Train sim history lacks 'offset_meters'; run with a save_interval so "
            "per-step history is recorded."
        )
    n_steps = len(offsets)
    if dt_s is None:
        # Fall back to differencing cumulative time.
        times = hist.get("time_seconds")
        if times is None:
            raise ValueError(
                "Train sim history lacks 'dt_seconds' and 'time_seconds'."
            )
        dt_s = [0.0] + [float(times[t]) - float(times[t - 1]) for t in range(1, len(times))]


    # Flattened path cat_power_limits carry district_id along the path offset axis.
    path_cpls = ts.get("path_tpc", {}).get("cat_power_limits", []) or []
    cpl_bounds = [
        (
            float(c["offset_start_meters"]),
            float(c["offset_end_meters"]),
            c.get("district_id"),
            c.get("district_id_secondary"),
            c.get("feed_fraction"),
        )
        for c in path_cpls
    ]

    def _feed_at(offset: float) -> tuple[str | None, str | None, float]:
        """Return (primary_id, secondary_id, frac_primary) at a path offset."""
        for s, e, did, did2, frac in cpl_bounds:
            if offset < s:
                break
            if offset <= e:
                fp = 1.0 if (did2 is None or frac is None) else float(frac)
                return did, did2, fp
        return None, None, 1.0

    # Exact network-wide totals from the cumulative (every-step) energy fields,
    # plus down-sampled per-step power for apportioning across substations. When
    # a zone is fed from two sides, each step's power is split between the two
    # feeding substations by the current-division fraction (nearer feeds more).
    loco_vec = ts.get("loco_con", {}).get("loco_vec", []) or []
    total_loss = 0.0
    total_cat = 0.0
    pwr_line_loss = [0.0] * n_steps
    pwr_cat_in = [0.0] * n_steps
    for loco in loco_vec:
        lt = loco.get("loco_type", {})
        dml = lt.get("DualModeLoco")
        if dml is None:
            continue
        ch = dml.get("cat_int", {}).get("history", {})
        ell = ch.get("energy_line_loss_joules")
        ein = ch.get("energy_elec_in_joules")
        if ell:
            total_loss += float(ell[-1])
        if ein:
            total_cat += float(ein[-1])
        pll = ch.get("pwr_line_loss_watts")
        pin = ch.get("pwr_elec_in_watts")
        if pll is not None:
            for t in range(min(n_steps, len(pll))):
                pwr_line_loss[t] += float(pll[t])
        if pin is not None:
            for t in range(min(n_steps, len(pin))):
                pwr_cat_in[t] += float(pin[t])

    # Per-substation weights from the sampled power history, split two-sided.
    loss_w: Dict[Optional[str], float] = {}
    cat_w: Dict[Optional[str], float] = {}
    for t in range(n_steps):
        primary, secondary, fp = _feed_at(float(offsets[t]))
        dt = float(dt_s[t]) if t < len(dt_s) else 0.0
        loss_e = pwr_line_loss[t] * dt
        cat_e = pwr_cat_in[t] * dt
        loss_w[primary] = loss_w.get(primary, 0.0) + loss_e * fp
        cat_w[primary] = cat_w.get(primary, 0.0) + cat_e * fp
        if secondary is not None and fp < 1.0:
            loss_w[secondary] = loss_w.get(secondary, 0.0) + loss_e * (1.0 - fp)
            cat_w[secondary] = cat_w.get(secondary, 0.0) + cat_e * (1.0 - fp)
    loss_w_sum = sum(loss_w.values())
    cat_w_sum = sum(cat_w.values())

    districts = sorted(set(loss_w) | set(cat_w), key=lambda d: (d is None, d))
    rows = []
    total_substation_grid = 0.0
    for did in districts:
        region = None
        efficiency = 1.0
        if substation_meta is not None and did in substation_meta:
            region = substation_meta[did].get("grid_emissions_region")
            eff = substation_meta[did].get("efficiency")
            if eff:
                efficiency = float(eff)
        # Apportion the exact totals by the sampled-power weights.
        loss_frac = (loss_w.get(did, 0.0) / loss_w_sum) if loss_w_sum > 0 else 0.0
        cat_frac = (cat_w.get(did, 0.0) / cat_w_sum) if cat_w_sum > 0 else 0.0
        loss_j = total_loss * loss_frac
        cat_j = total_cat * cat_frac
        # Grid-side substation input includes catenary line loss and is scaled up
        # by the substation's conversion (transformer/rectifier) efficiency.
        substation_grid_j = (cat_j + loss_j) / efficiency if efficiency > 0 else cat_j + loss_j
        total_substation_grid += substation_grid_j
        rows.append(
            {
                "district_id": did,
                "grid_emissions_region": region,
                "line_loss_energy_joules": loss_j,
                "catenary_energy_joules": cat_j,
                "substation_energy_joules": substation_grid_j,
            }
        )
    rows.append(
        {
            "district_id": "__total__",
            "grid_emissions_region": None,
            "line_loss_energy_joules": total_loss,
            "catenary_energy_joules": total_cat,
            "substation_energy_joules": total_substation_grid,
        }
    )
    return pl.DataFrame(rows)


def substation_load_curves(
    train_sim: alt.SpeedLimitTrainSim | dict,
    substation_meta: dict[str, dict] | None = None,
) -> pl.DataFrame:
    """Build per-substation load curves from a solved train simulation.

    Each step's pantograph power plus line loss is split by feed fraction.

    Args:
        train_sim: Solved train simulation, or its ``to_pydict()``, saved with a
            per-step history.
        substation_meta: Metadata from :func:`apply_line_losses`; currently
            unused.

    Returns:
        Columns ``time_seconds``, one ``load_watts__<district_id>`` per
        substation, and ``load_watts__total``.

    Raises:
        ValueError: If the history lacks train offsets or times.
    """
    import polars as pl

    ts = train_sim.to_pydict() if hasattr(train_sim, "to_pydict") else train_sim
    hist = ts.get("history", {})
    offsets = hist.get("offset_meters") or hist.get("offset_back_meters")
    times = hist.get("time_seconds")
    if offsets is None or times is None:
        raise ValueError(
            "Train sim history lacks 'offset_meters'/'time_seconds'; run with a "
            "save_interval so per-step history is recorded."
        )
    n_steps = len(offsets)

    path_cpls = ts.get("path_tpc", {}).get("cat_power_limits", []) or []
    cpl_bounds = [
        (
            float(c["offset_start_meters"]),
            float(c["offset_end_meters"]),
            c.get("district_id"),
            c.get("district_id_secondary"),
            c.get("feed_fraction"),
        )
        for c in path_cpls
    ]

    def _feed_at(offset: float) -> tuple[str | None, str | None, float]:
        """Return (primary_id, secondary_id, frac_primary) at a path offset."""
        for s, e, did, did2, frac in cpl_bounds:
            if offset < s:
                break
            if offset <= e:
                fp = 1.0 if (did2 is None or frac is None) else float(frac)
                return did, did2, fp
        return None, None, 1.0

    loco_vec = ts.get("loco_con", {}).get("loco_vec", []) or []
    p_sub_total = [0.0] * n_steps
    for loco in loco_vec:
        lt = loco.get("loco_type", {})
        # Dual-mode locomotives always draw catenary; battery-electric ones do so
        # only when fitted with a pantograph (``cat_int`` present).
        cat_int = (lt.get("DualModeLoco") or lt.get("BatteryElectricLoco") or {}).get(
            "cat_int"
        )
        if cat_int is None:
            continue
        ch = cat_int.get("history", {})
        pll = ch.get("pwr_line_loss_watts")
        pin = ch.get("pwr_elec_in_watts")
        for t in range(n_steps):
            if pin is not None and t < len(pin):
                p_sub_total[t] += float(pin[t])
            if pll is not None and t < len(pll):
                p_sub_total[t] += float(pll[t])

    # Collect the set of substations that appear.
    subs: List[str] = []
    seen = set()
    for _s, _e, did, did2, _f in cpl_bounds:
        for d in (did, did2):
            if d is not None and d not in seen:
                seen.add(d)
                subs.append(d)

    series: Dict[str, List[float]] = {d: [0.0] * n_steps for d in subs}
    total_series = [0.0] * n_steps
    for t in range(n_steps):
        primary, secondary, fp = _feed_at(float(offsets[t]))
        total_series[t] = p_sub_total[t]
        if primary is not None:
            series.setdefault(primary, [0.0] * n_steps)[t] += p_sub_total[t] * fp
        if secondary is not None and fp < 1.0:
            series.setdefault(secondary, [0.0] * n_steps)[t] += p_sub_total[t] * (1.0 - fp)

    data = {"time_seconds": [float(x) for x in times[:n_steps]]}
    for d in sorted(series):
        data[f"load_watts__{d}"] = series[d]
    data["load_watts__total"] = total_series
    return pl.DataFrame(data)


def __getattr__(name: str) -> type[Substation]:
    """Lazily re-export :class:`Substation` from the optimizer module.

    Args:
        name: Attribute name being looked up.

    Returns:
        The :class:`Substation` class.

    Raises:
        AttributeError: If ``name`` is anything else.
    """
    if name == "Substation":
        from altrios.optimization.energization_placement import Substation

        return Substation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

