"""Utilities for placing energization (e.g. catenary or wireless power transfer) power limits on a network."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, List, Dict, Union

# local imports
import altrios as alt

if TYPE_CHECKING:
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
    """Build an :class:`ElectrificationConfig` using the per-km impedance table."""
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

    def optimizer_kwargs(self) -> Dict[str, float]:
        """Keyword args for :func:`energize_optimized` (power limit + feed distance)."""
        return {
            "uniform_power_limit_watts": self.uniform_power_limit_watts,
            "substation_feed_distance_miles": self.substation_feed_distance_miles,
        }

    def line_loss_kwargs(self) -> Dict[str, float]:
        """Keyword args for :func:`apply_line_losses` (electrical assumptions)."""
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
    """Convert a battery-electric locomotive's battery size into a maximum
    unelectrified-gap distance (miles) for ``energize_optimized(max_energized_gap_miles=...)``.

    Under the optimizer's simplifying assumptions -- any electrified segment
    recharges the battery, and traction energy is drawn at a constant average
    rate per mile -- a battery that can usefully deliver ``E`` kWh bridges a gap
    of ``E / e`` miles, where ``e`` is the average consumption in kWh per mile.
    Holding back a ``reserve_fraction`` of the pack (e.g. to protect state of
    charge or cover accessory/HVAC loads) reduces the usable energy accordingly,
    so the "kWh of energy lost between segments" is
    ``usable_energy_kwh * (1 - reserve_fraction)`` and

    .. math::

        \\text{max gap (mi)} = \\frac{E_\\text{usable}\\,(1 - \\text{reserve})}{e}.

    Parameters
    ----------
    usable_energy_kwh : float
        Battery energy (kWh) available for traction between charges (i.e. the
        usable, not nameplate, capacity).
    avg_consumption_kwh_per_mile : float
        Average traction energy drawn per corridor-mile (kWh/mi). A convenient
        source is the mean of the per-link ``electricity_kwh_per_track_mile``
        densities used to build :func:`make_piece_opex_fn` from a first-pass
        simulation.
    reserve_fraction : float, default 0.0
        Fraction of ``usable_energy_kwh`` withheld as a reserve (0 <= f < 1).

    Returns
    -------
    float
        Maximum contiguous unelectrified gap (miles).

    Raises
    ------
    ValueError
        If ``avg_consumption_kwh_per_mile`` is not positive, ``usable_energy_kwh``
        is negative, or ``reserve_fraction`` is outside ``[0, 1)``.
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
    """Usable battery traction energy (kWh) for ``max_energized_gap_kwh``.

    The ENERGY-based analogue of :func:`max_gap_miles_from_battery`: instead of
    pre-dividing by an average per-mile consumption, it returns the raw kWh a BEL
    may spend between charges, ``usable_energy_kwh * (1 - reserve_fraction)``. The
    optimizer then bounds each unelectrified corridor window by its per-link drain
    density (``gap_energy_kwh_per_mile_by_link``) instead of a flat distance, so
    the wire is required where the battery actually depletes fastest.

    Parameters
    ----------
    usable_energy_kwh : float
        Battery energy (kWh) available for traction between charges.
    reserve_fraction : float, default 0.0
        Fraction withheld as reserve (0 <= f < 1).
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
    """Minimum energized-run length (miles) to recharge the pack (Tier 0 recharge).

    Sizes ``min_energized_length_miles`` from battery/charge physics so an
    energized run is long enough to actually restore charge, replacing the
    optimizer's optimistic "any energized segment fully recharges" assumption with
    a physically sized minimum run. Under a roughly constant net charge rate
    (charge power minus draw, per corridor-mile) the run needed to bank a fraction
    ``recharge_fraction`` of the usable pack is

    .. math::

        L_\\text{recharge} = \\frac{E_\\text{usable}(1 - \\text{reserve})\\,
        \\text{recharge\\_fraction}}{c_\\text{net}},

    where ``c_net`` is the net charge in kWh per mile. This is the Tier-0 recharge
    formulation: it keeps the minimum-run constraint measured in CORRIDOR MILES
    (its matrix is unchanged), so it is structurally free; the "energy" only sizes
    the threshold. Take the max with any anti-sliver minimum run.

    Parameters
    ----------
    usable_energy_kwh : float
        Usable (not nameplate) battery capacity (kWh).
    net_charge_kwh_per_mile : float
        Net charge delivered per energized corridor-mile (catenary charge power
        minus traction draw, converted to energy per mile at line speed). Must be
        positive.
    recharge_fraction : float, default 1.0
        Fraction of the usable pack the run must restore (e.g. 1.0 to refill a
        fully depleted pack).
    reserve_fraction : float, default 0.0
        Fraction of the pack withheld as reserve (0 <= f < 1).
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
    config: Union[str, ElectrificationConfig],
    *,
    design_train_power_watts: float = DEFAULT_DESIGN_TRAIN_POWER_WATTS,
    concurrent_trains_per_substation: float = 2.0,
    min_feed_distance_miles: Optional[float] = 1.0,
    max_feed_distance_miles: Optional[float] = None,
) -> SuggestedOptimizationInputs:
    """Derive reasonable optimizer/line-loss inputs from an electrification choice.

    Given a configuration (e.g. ``"dc_9kv"``, ``"ac_25kv"``, ``"ac_50kv"``), this
    pre-processing step computes a physically motivated substation feeding radius
    and per-train power limit, plus matching substation rating and electrical
    assumptions, so the optimizer and the line-loss model can be configured
    consistently for that voltage class.

    Feed distance is sized from the AC/DC voltage-drop limit (the same model used
    by :func:`apply_line_losses`). For a train drawing the design power ``P_d`` at
    the *single-sided* coverage boundary (distance ``d`` from its only feeding
    substation), the pantograph voltage stays at or above the minimum fraction
    ``f`` of nominal ``V`` when

    .. math::

        V\\,(1 - f)\\,V \\;\\ge\\; |Z|\\,d\\,\\frac{P_d}{V}
        \\quad\\Longrightarrow\\quad
        d \\;\\le\\; \\frac{V^2 (1 - f)}{|Z|\\,P_d},

    where :math:`|Z| = \\sqrt{R^2 + X^2}` is the per-meter loop impedance. The
    suggested ``substation_feed_distance_miles`` is this ``d`` (interior track fed
    from two substations gets roughly double the capability, so this is
    conservative). Because ``d`` scales as ``V**2``, higher voltages permit much
    wider substation spacing; DC's zero reactance further reduces ``|Z|``.

    Parameters
    ----------
    config : str or ElectrificationConfig
        A key into :data:`ELECTRIFICATION_CONFIGS` (case-insensitive) or an
        explicit :class:`ElectrificationConfig`.
    design_train_power_watts : float, default 10 MW
        Peak per-train catenary power the system must sustain at the coverage
        boundary. Also used as the per-segment ``uniform_power_limit_watts``.
    concurrent_trains_per_substation : float, default 2.0
        Number of design-power trains a substation should feed simultaneously;
        sets ``substation_max_power_watts = design_power * this``.
    min_feed_distance_miles, max_feed_distance_miles : float, optional
        Optional clamps on the derived feed distance. ``min`` defaults to 1 mile
        (avoids a degenerate near-zero radius); ``max`` is unset by default.

    Returns
    -------
    SuggestedOptimizationInputs
        The derived values, with ``optimizer_kwargs()`` / ``line_loss_kwargs()``
        helpers for wiring into :func:`energize_optimized` and
        :func:`apply_line_losses`.

    Raises
    ------
    KeyError
        If ``config`` is an unknown preset name.
    ValueError
        If ``design_train_power_watts`` is not positive.
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
    """
    Apply a uniform energization power limit spanning every link in the network.

    Each non-dummy link is given a single ``cat_power_limits`` entry covering its
    full length at the specified power limit, electrifying the entire network.

    Parameters
    ----------
    network : alt.Network
        Network to electrify. Not mutated; a new network is returned.
    uniform_power_limit_watts : float
        Energization power limit (W) applied to every link.

    Returns
    -------
    alt.Network
        A new network with energization power limits applied to all links.
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
    """
    Apply energization only to links whose maximum grade meets a threshold.

    For each link with at least two elevation points, the maximum grade is
    computed from consecutive ``elevs`` samples; if it is greater than or equal to
    ``grade_threshold`` the whole link is electrified at the given power limit.

    Parameters
    ----------
    network : alt.Network
        Network to electrify. Not mutated; a new network is returned.
    uniform_power_limit_watts : float
        Energization power limit (W) applied to each qualifying link.
    grade_threshold : float
        Minimum maximum-grade (rise/run, unitless) at or above which a link is
        electrified.

    Returns
    -------
    alt.Network
        A new network with energization power limits applied to high-grade links.
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
    substations: List["Substation"],
    uniform_power_limit_watts: float,
    *,
    costs: Optional["EnergizationOptCosts"] = None,
    budget: Optional[float] = None,
    grade_threshold: float = 0.01,
    high_grade_benefit_weight: float = 10.0,
    substation_feed_distance_miles: float = 20.0,
    require_high_grade_coverage: bool = False,
    est_time_nets: Optional[List[alt.EstTimeNet]] = None,
    excluded_link_idxs: Optional[List[int]] = None,
    structure_spacing_miles: Optional[float] = None,
    corridor_width_miles: float = 0.05,
    electrify_whole_site: bool = False,
    min_energized_length_miles: Optional[float] = None,
    max_energized_gap_miles: Optional[float] = None,
    max_energized_gap_kwh: Optional[float] = None,
    gap_energy_kwh_per_mile_by_link: Optional[Dict[int, float]] = None,
    charge_energy_kwh_per_mile_by_link: Optional[Dict[int, float]] = None,
    regen_energy_kwh_per_mile_by_link: Optional[Dict[int, float]] = None,
    battery_soc_state_model: bool = False,
    objective: str = "max_benefit",
    economics: Optional["EconomicParameters"] = None,
    piece_opex_fn: Optional[Callable[["PieceBenefitContext"], float]] = None,
    piece_benefit_fn: Optional[Callable[["PieceBenefitContext"], float]] = None,
    time_limit_seconds: Optional[float] = None,
    mip_gap: Optional[float] = None,
    tee: bool = False,
    solver_threads: Optional[int] = None,
    warmstart: bool = False,
) -> "EnergizationPlacementResult":
    """
    Optimize energization placement with a MILP, prioritizing high-grade track.

    Thin convenience wrapper around
    :mod:`altrios.optimization.energization_placement`. It jointly selects which track
    segments to electrify and which candidate transmission substations to build,
    subject to a capital ``budget`` and substation feeding/spacing constraints,
    while prioritizing high-grade regions.

    If ``structure_spacing_miles`` is set, structure-site (corridor) mode is used:
    every link is densified into uniform along-track pieces and parallel
    tracks/sidings that share a location are clustered so they share a single
    structure cost (see ``EnergizationOptCosts``). Clustering is purely geometric and
    requires networks with lat/lon geometry, but needs no centerline, topology, or
    est-time nets.

    Parameters
    ----------
    network : alt.Network
        Network on which energization is placed. Not mutated; the optimized network is
        returned inside the result.
    substations : list of Substation
        Candidate (and optionally pre-existing) transmission substations.
    uniform_power_limit_watts : float
        Energization power limit (W) applied to every energized segment.
    costs : EnergizationOptCosts, optional
        Cost coefficients. Defaults to a zero-cost ``EnergizationOptCosts``.
    budget : float, optional
        Capital budget cap ($). ``None`` (default) leaves spending unconstrained.
    grade_threshold : float, default 0.01
        Grade (rise/run) at or above which a segment is treated as high priority.
    high_grade_benefit_weight : float, default 10.0
        Objective multiplier applied to high-grade segments relative to 1.0 for
        ordinary segments.
    substation_feed_distance_miles : float, default 20.0
        Maximum distance (miles) from an energized segment to its nearest built
        substation (the feeding/spacing constraint).
    require_high_grade_coverage : bool, default False
        If True, force every high-grade segment to be energized rather than merely
        prioritizing it in the objective.
    est_time_nets : list of alt.EstTimeNet, optional
        In per-link mode, restricts candidate segments to traversed links.
        Not required by structure-site (corridor) mode.
    structure_spacing_miles : float, optional
        If set, use structure-site (corridor) mode with along-track pieces of this
        length (miles); otherwise (default) each link is treated as its own
        segment.
    corridor_width_miles : float, default 0.05
        Lateral distance (miles) within which parallel-track pieces are clustered
        into one shared structure site (structure-site mode only).
    electrify_whole_site : bool, default False
        If True, use the WHOLE-SITE formulation: energizing a structure site wires
        every track piece at it (``x == s``), collapsing the per-piece binaries
        onto the site indicators. Far fewer integer variables and a stronger LP
        relaxation, so the optimality gap closes much faster; the trade-off is no
        partial-cross-section electrification (parallel tracks/sidings at a site
        are wired together).
    min_energized_length_miles : float, optional
        Minimum contiguous energized run length in CORRIDOR miles (parallel
        tracks/sidings in a site count once). Prevents energizing tiny isolated
        slivers even when structure/substation costs are zero; the run may start
        at any position. Requires structure-site mode
        (``structure_spacing_miles``); ``None`` (default) disables it.
    max_energized_gap_miles : float, optional
        Maximum contiguous UNelectrified gap (CORRIDOR miles) along the mainline
        -- the range limit for a battery-electric locomotive, which must reach
        the next electrified segment before its battery depletes. Derive it from
        battery size with :func:`max_gap_miles_from_battery`. Can FORCE
        electrification (a feasibility pre-check raises ``ValueError`` if a gap
        cannot be closed by any reachable substation). Requires structure-site
        mode with a contiguous mainline path; ``None`` (default) disables it.
    max_energized_gap_kwh : float, optional
        ENERGY-based range limit (kWh): the usable battery traction energy a BEL
        may spend between charges (see :func:`max_gap_kwh_from_battery`). When set,
        the gap constraint bounds each unelectrified window's cumulative DRAIN
        energy (from ``gap_energy_kwh_per_mile_by_link``) instead of its distance,
        so wire is required where the pack depletes fastest. Takes precedence over
        ``max_energized_gap_miles``. ``None`` (default) disables it.
    gap_energy_kwh_per_mile_by_link : dict[int, float], optional
        Per-link single-locomotive traction DRAIN per corridor-mile (kWh/mi).
        Required when ``max_energized_gap_kwh`` is set; also weights the
        energy-priority objective term.
    charge_energy_kwh_per_mile_by_link : dict[int, float], optional
        Per-link NET CHARGE delivered per energized corridor-mile (kWh/mi). When
        supplied with ``max_energized_gap_kwh`` this enables the charge-credited
        (recharge Tier 1) gap family, which no longer assumes a single energized
        bucket fully recharges the pack. ``None`` (default) uses the simpler
        ``sum u >= 1`` energy-gap family.
    regen_energy_kwh_per_mile_by_link : dict[int, float], optional
        Per-link regenerative-braking recovery per corridor-mile (kWh/mi). When
        supplied, the gap/SOC constraints use NET drain (``drain - regen``) so
        regen extends the modeled battery range. ``None`` (default) treats regen
        as zero.
    battery_soc_state_model : bool, default False
        If True, use the explicit SOC-recursion (Tier 2) range model instead of the
        covering gap families: a continuous per-bucket state of charge with a
        curtailment slack (linear, no big-M) that tracks charge/drain/regen exactly
        and forbids depletion. Faithful (subsumes the covering gaps) but heavier
        (continuous state, not TU). Requires ``max_energized_gap_kwh`` +
        ``gap_energy_kwh_per_mile_by_link``.
    objective : str, default "max_benefit"
        ``"max_benefit"`` maximizes priority-weighted (grade) benefit subject to
        ``budget`` -- the rule-based formulation. ``"min_cost"`` instead minimizes
        lifecycle cost = capital + discounted operating cost, and requires
        ``economics`` and ``piece_opex_fn``.
    economics : EconomicParameters, optional
        Horizon + discount rate for ``"min_cost"`` mode (see
        :class:`~altrios.optimization.energization_placement.EconomicParameters`).
    piece_opex_fn : callable, optional
        ``PieceBenefitContext -> float`` giving each piece's **annual net operating
        cost** ($/yr, electricity minus displaced diesel; negative = saving).
        Required for ``"min_cost"``. Build it from a first-pass dual-mode
        simulation and prices via :func:`make_piece_opex_fn`.
    piece_benefit_fn : callable, optional
        ``PieceBenefitContext -> float`` overriding the default grade-priority
        benefit in ``"max_benefit"`` mode.
    time_limit_seconds : float, optional
        Solver wall-clock time limit (s). ``None`` (default) means no limit.
    mip_gap : float, optional
        Relative MIP optimality gap tolerance. ``None`` (default) uses the solver
        default.
    tee : bool, default False
        If True, stream solver output.
    solver_threads : int, optional
        Cap the HiGHS solver's parallelism to this many threads. ``None``
        (default) uses the solver default (all available cores).
    warmstart : bool, default False
        Seed a feasible fully-wired MIP start so HiGHS always has an incumbent
        (useful for the heavy Tier-1/SOC models). A no-incumbent time-limit hit
        returns a result carrying only ``status`` (``network`` is ``None``) rather
        than raising.

    Returns
    -------
    EnergizationPlacementResult
        Result whose ``network`` field holds the network with the optimized
        energization applied, along with the chosen substations, costs, and
        diagnostics.
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


def _price_lookup(price: "PriceByLink") -> Callable[[int], float]:
    """``link_idx -> price`` for a scalar price or a per-link price map. A link
    missing from the map takes the map's mean, so a partially mapped network
    still prices every piece."""
    if isinstance(price, dict):
        default = (sum(price.values()) / len(price)) if price else 0.0
        return lambda idx: float(price.get(idx, default))
    value = float(price)
    return lambda idx: value


def make_piece_opex_fn(
    *,
    diesel_gallons_per_track_mile_by_link: Dict[int, float],
    electricity_kwh_per_track_mile_by_link: Dict[int, float],
    diesel_price_usd_per_gallon: "PriceByLink",
    electricity_price_usd_per_kwh: "PriceByLink",
    default_diesel_gallons_per_track_mile: float = 0.0,
    default_electricity_kwh_per_track_mile: float = 0.0,
) -> Callable[["PieceBenefitContext"], float]:
    """Build a per-piece annual operating-cost function for ``objective="min_cost"``.

    Returns a callable suitable for ``energize_optimized(piece_opex_fn=...)`` /
    ``EnergizationPlacementInputs.piece_opex_fn``. For each piece it computes the
    **annual net operating cost delta** of electrifying that piece:

    ``opex = electricity_price * kwh - diesel_price * gallons``

    where ``kwh`` and ``gallons`` are the piece's annual electricity purchased and
    annual diesel displaced (both = per-track-mile density * piece length_miles).
    A negative result is a saving. Because the optimizer only sees this scalar per
    piece, all traffic/loading assumptions live here, in the densities:

    The per-link densities encode **all the traffic assumptions** -- trains per
    day, tonnage/loading, consist makeup (how many locomotives, how many dual-mode)
    -- because they are meant to be produced by aggregating first-pass dual-mode
    simulations over an annual train set (e.g. via the ALTRIOS demand /
    train-planner / sim_manager pipeline and the ``annualize`` convention). The
    electricity density should already include catenary line loss and substation
    conversion loss (grid-side energy); the diesel density is the fuel the
    dual-mode consist would otherwise burn on that segment.

    Parameters
    ----------
    diesel_gallons_per_track_mile_by_link : dict[int, float]
        Annual diesel gallons displaced per track-mile, keyed by link index.
    electricity_kwh_per_track_mile_by_link : dict[int, float]
        Annual grid-side electricity (kWh) purchased per track-mile, keyed by link
        index.
    diesel_price_usd_per_gallon, electricity_price_usd_per_kwh : float or dict[int, float]
        Fuel and electricity prices, either one corridor-wide value or a price per
        link index (regional price zones); unmapped links take the map's mean.
    default_diesel_gallons_per_track_mile, default_electricity_kwh_per_track_mile : float
        Fallback densities for links absent from the maps (default 0).

    Returns
    -------
    callable
        ``PieceBenefitContext -> float`` annual net operating cost ($/yr).
    """
    diesel_price = _price_lookup(diesel_price_usd_per_gallon)
    elec_price = _price_lookup(electricity_price_usd_per_kwh)

    def _opex(ctx: "PieceBenefitContext") -> float:
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
    traction_kwh_per_track_mile_by_link: Dict[int, float],
    electricity_price_usd_per_kwh: "PriceByLink",
    battery_roundtrip_efficiency: float,
    default_traction_kwh_per_track_mile: float = 0.0,
) -> Callable[["PieceBenefitContext"], float]:
    """Per-piece annual operating-cost function for a BATTERY-ELECTRIC fleet.

    A BEL never burns mainline diesel, so the dual-mode "electricity minus diesel
    displaced" opex (:func:`make_piece_opex_fn`) does not apply. To first order the
    total grid energy a BEL draws is fixed by the route's traction demand
    regardless of WHERE the wire hangs, so the electricity bill is nearly
    layout-independent and the BEL problem is capital-dominated. The one
    layout-dependent operating cost is the battery **round-trip loss** on the
    energy that must pass through the pack: energizing a piece lets its traction
    energy be drawn straight from the wire instead of via the battery, avoiding
    that loss. The per-piece opex is therefore a SAVING (negative):

    ``opex_p = - price_e * (1/eta_rt - 1) * E_p``

    where ``E_p`` = annual traction kWh on the piece (per-track-mile density *
    ``length_miles``). Minimizing ``capital + PVAF * sum_p opex_p * x_p`` thus
    rewards energizing the highest-energy (steep/heavy) track first -- i.e.
    "prioritize the segments with the largest energy requirement per mile" -- which
    is exactly the gradient of the round-trip-loss term, not an ad hoc heuristic.
    The range-extender diesel term (for a SOFT range limit) is intentionally
    omitted; add it when soft gaps are modeled.

    Parameters
    ----------
    traction_kwh_per_track_mile_by_link : dict[int, float]
        Annual traction (at-the-wheel) energy per track-mile, keyed by link index.
        Grade/speed/mass scaling and traffic assumptions live here, as in
        :func:`make_piece_opex_fn`.
    electricity_price_usd_per_kwh : float or dict[int, float]
        Grid electricity price, corridor-wide or per link index (price zones).
    battery_roundtrip_efficiency : float
        Pack charge->discharge round-trip efficiency ``eta_rt`` in ``(0, 1]``.
    default_traction_kwh_per_track_mile : float, default 0.0
        Fallback density for links absent from the map.

    Returns
    -------
    callable
        ``PieceBenefitContext -> float`` annual net operating cost ($/yr, negative
        = saving).
    """
    if not 0.0 < battery_roundtrip_efficiency <= 1.0:
        raise ValueError("battery_roundtrip_efficiency must be in (0, 1].")
    loss_factor = (1.0 / battery_roundtrip_efficiency) - 1.0
    elec_price = _price_lookup(electricity_price_usd_per_kwh)

    def _opex(ctx: "PieceBenefitContext") -> float:
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
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Compute the energization sub-link intervals for one link in a repeating pattern.

    Walks from ``start_pos_in_cycle`` through the repeating gap/energization cycle
    (``gap_meters`` of no energization followed by ``cat_meters`` of energization) across
    ``link_length`` meters, producing precise ``(offset_start, offset_end)`` pairs
    relative to the start of the link. The ending cycle position is returned so
    consecutive links can be chained without floating-point drift.

    Parameters
    ----------
    start_pos_in_cycle : float
        Position (meters) within the gap/energization cycle at the start of the link.
    link_length : float
        Length of the link (meters).
    gap_meters : float
        Length (meters) of the un-electrified gap portion of the cycle.
    cat_meters : float
        Length (meters) of the electrified portion of the cycle.

    Returns
    -------
    tuple of (list of tuple of (float, float), float)
        The energized ``(offset_start, offset_end)`` intervals within the link and
        the ending position within the cycle.
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
    link_by_idx: Dict[int, dict], candidates: set
) -> List[int]:
    """
    Return the longest through-route (ordered link idxs) by following ``idx_next``.

    Follows each source link's ``idx_next`` chain -- the primary/through pointer, so
    it continues straight through junctions while sidings diverge via
    ``idx_next_alt`` -- and returns the walk covering the greatest total length. This
    is a best-effort continuous mainline centerline; the opposite direction and
    sidings are associated separately by geometric proximity.

    Parameters
    ----------
    link_by_idx : dict
        Mapping of link index to link dict.
    candidates : set
        Link indices eligible to be part of the mainline.

    Returns
    -------
    list of int
        Ordered link indices of the longest through-route.
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
    spacing_meters: Tuple[float, float],
    *,
    sample_spacing_meters: float = 200.0,
    corridor_match_meters: float = 1000.0,
) -> alt.Network:
    """
    Place energization in an even gap/energized pattern measured in CORRIDOR distance.

    ``spacing_meters`` is ``(gap_meters, energized_meters)``: a repeating pattern of
    ``gap_meters`` of no energization followed by ``energized_meters`` of energization,
    laid out along the corridor's linear distance. All tracks that make up the corridor
    at a given location -- both directions of a double-track line plus adjacent sidings --
    inherit the SAME pattern state, so the pattern is measured once in corridor distance
    (not multiplied per parallel track).

    The corridor is reduced to uniform "virtual link" pieces (see ``_densify_links``);
    the longest through-route (followed via ``idx_next``) is the reference mainline whose
    cumulative arc-length drives the pattern, and every other piece (opposite direction,
    sidings, branches) inherits the pattern from the nearest mainline arc-length by
    lat/lon proximity. This replaces the old est-time-net mainline heuristic, giving
    exact intervals (the cycle phase is continuous along the corridor) and robust,
    geometry-based siding handling.

    Parameters
    ----------
    network : alt.Network
        Network to electrify. Not mutated; a new network is returned. Requires lat/lon
        geometry on the links.
    uniform_power_limit_watts : float
        Energization power limit (W) applied to every energized interval.
    spacing_meters : tuple of (float, float)
        ``(gap_meters, energized_meters)`` of the repeating corridor pattern.
    sample_spacing_meters : float, default 200.0
        Along-track piece length (m) for the reduced representation; smaller values give
        finer siding-to-corridor matching at some extra cost.
    corridor_match_meters : float, default 1000.0
        Maximum lateral distance (m) from a piece to the reference mainline for it to be
        treated as part of the corridor; pieces farther than this are left un-energized.

    Returns
    -------
    alt.Network
        A new network with evenly spaced energization sub-link intervals applied.
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


def _latlon_at_offset(link: dict, offset_meters: float) -> Optional[Tuple[float, float]]:
    """Interpolate a link's (lat, lon) at a given along-track offset (meters)."""
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
):
    """Build a corridor arc-length reference from the network geometry.

    Reuses the shared densify + longest-mainline machinery to produce, for every
    link: the cumulative corridor arc-length at the link start (for mainline
    links) plus a KD-tree that maps any lat/lon to the nearest mainline
    arc-length (for sidings / opposite-direction / branch tracks).

    Returns
    -------
    tuple
        ``(arc_start_of_link, mainline_link_set, project_fn)`` where
        ``project_fn(lat, lon) -> arc_meters`` returns the nearest mainline
        arc-length, and ``arc_start_of_link[idx]`` is the corridor arc at the
        start of mainline link ``idx`` (absent for non-mainline links).
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

    def _to_xy(lat: float, lon: float):
        return ((lon - lon0) * m_per_deg_lon, (lat - lat0) * m_per_deg_lat)

    main_xy = np.array([_to_xy(p_lat[i], p_lon[i]) for i in mainline_pieces])
    main_tree = cKDTree(main_xy)
    # Arc-length at each mainline piece midpoint = link arc start + piece start.
    main_arc = []
    for i in mainline_pieces:
        main_arc.append(arc_start_of_link[p_link[i]] + float(p_off0[i]))

    def project_fn(lat: float, lon: float) -> float:
        _dist, k = main_tree.query(_to_xy(lat, lon))
        return float(main_arc[k])

    return arc_start_of_link, mainline_link_set, project_fn


def apply_line_losses(
    network: alt.Network,
    substations: "List[Substation]",
    *,
    system_voltage_volts: float = 25_000.0,
    resistance_ohms_per_meter: Optional[float] = None,
    reactance_ohms_per_meter: Optional[float] = None,
    electrification_type: str = "ac_25kv_1x25",
    default_substation_max_power_watts: float = DEFAULT_SUBSTATION_MAX_POWER_WATTS,
    default_substation_efficiency: float = DEFAULT_SUBSTATION_EFFICIENCY,
    min_pantograph_voltage_fraction: float = DEFAULT_MIN_PANTOGRAPH_VOLTAGE_FRACTION,
    base_power_limit_watts: Optional[float] = None,
    sample_spacing_meters: float = 200.0,
    corridor_match_meters: float = 1000.0,
) -> "Tuple[alt.Network, Dict[str, dict]]":
    """Precompute AC catenary line losses and derate energized power limits.

    For every energized ``cat_power_limits`` interval already present on the
    network, this computes the effective series resistance/reactance from the
    interval to its feeding substation network along the rail corridor, then:

    - stamps ``resistance_to_source_ohms`` / ``reactance_to_source_ohms`` /
      ``voltage_volts`` / ``district_id`` on the interval so the simulation can
      compute resistive line loss (``loss = resistance * (pwr / voltage)**2``)
      each time step, and
    - derates the interval ``power_limit_watts`` to the minimum of its base
      rating, the substation-capacity-limited power (accounting for loss), and
      the voltage-drop-limited power (minimum pantograph voltage).

    The corridor is treated as linear in graph space (branching neglected); a
    train between two adjacent substations is fed from both sides in parallel, so
    the effective resistance at corridor position ``x`` between substations at
    ``a < b`` is ``r * (x - a) * (b - x) / (b - a)`` (half the single-sided loss
    at mid-span); beyond the outermost substation it is ``r * |x - a|``.

    Parameters
    ----------
    network : alt.Network
        A network that already carries ``cat_power_limits`` (e.g. from
        :func:`energize_uniform`, :func:`energize_even_spacing`, or the
        optimizer). Not mutated; a new network is returned.
    substations : list of Substation
        Feeding substations (user-provided or optimizer-produced). Each must have
        a resolvable location (``lat``/``lon`` or ``link_idx``). ``max_power_watts``,
        ``grid_emissions_region``, and ``voltage_volts`` are used if set.
    system_voltage_volts : float, default 25000
        Global nominal catenary voltage. Used where a substation does not
        override it; also written to the network config.
    resistance_ohms_per_meter, reactance_ohms_per_meter : float, optional
        Loop series resistance/reactance per meter. If ``None``, taken from
        ``electrification_type`` via :data:`ELECTRIFICATION_IMPEDANCE_DEFAULTS`.
    electrification_type : str, default "ac_25kv_1x25"
        Key into :data:`ELECTRIFICATION_IMPEDANCE_DEFAULTS` for default impedance.
    default_substation_max_power_watts : float
        Fallback substation power rating when a substation omits ``max_power_watts``.
    default_substation_efficiency : float, default 0.99
        Fallback grid->catenary conversion efficiency when a substation omits
        ``efficiency``. 0.99 suits AC (transformer only); use a lower value (~0.97)
        for DC substations to capture AC->DC rectifier loss.
    min_pantograph_voltage_fraction : float, default 0.7
        Minimum allowed pantograph voltage as a fraction of nominal, for the
        voltage-drop-limited capacity.
    base_power_limit_watts : float, optional
        Base (pre-loss) power rating to derate from. If ``None``, each interval's
        existing ``power_limit_watts`` is used as its base rating.
    sample_spacing_meters : float, default 200.0
        Along-track piece length for building the corridor arc-length reference.
    corridor_match_meters : float, default 1000.0
        Maximum lateral distance for matching an off-mainline interval to the
        corridor; ignored (no loss modeled) beyond this.

    Returns
    -------
    tuple of (alt.Network, dict)
        The derated network and a ``substation_meta`` dict keyed by string
        district id, each with ``grid_emissions_region``, ``max_power_watts``,
        ``voltage_volts``, ``efficiency``, and corridor ``arc_meters``. Pass the
        metadata to :func:`compute_line_losses` for grid-region attribution and
        conversion-loss scaling.
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

    def _feed(x: float):
        """Return (left_record, right_record) feeding substations by corridor arc."""
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
    train_sim,
    substation_meta: Optional[Dict[str, dict]] = None,
):
    """Aggregate catenary line losses from a solved train simulation.

    Reads the per-time-step resistive line loss the simulation recorded in each
    dual-mode locomotive's catenary interface history, attributes it to the
    feeding substation/district (by mapping the train position to the energized
    ``cat_power_limits`` interval that carries the ``district_id``), and, if
    ``substation_meta`` is provided, to each substation's grid-emissions region.

    Parameters
    ----------
    train_sim : SpeedLimitTrainSim or dict
        A solved train simulation (or its ``to_pydict()``), saved with a history
        (``save_interval`` set).
    substation_meta : dict, optional
        The metadata returned by :func:`apply_line_losses`, keyed by district id.

    Returns
    -------
    polars.DataFrame
        One row per district with ``district_id``, ``grid_emissions_region``,
        ``line_loss_energy_joules``, ``catenary_energy_joules`` (pantograph
        energy drawn), and ``substation_energy_joules`` (grid-side substation
        input = pantograph + line loss, scaled up by ``1/efficiency`` when the
        substation's conversion ``efficiency`` is known from ``substation_meta``).
        A final ``"__total__"`` row carries the network-wide totals.

    Notes
    -----
    Network-wide totals come from each locomotive's *cumulative*
    ``energy_line_loss`` / ``energy_elec_in`` (integrated every simulation step,
    so exact regardless of ``save_interval``). When more than one district feeds
    the train, the totals are apportioned across districts using the
    (down-sampled) per-step power history as weights; the split is therefore
    approximate but the totals are exact.
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

    def _feed_at(offset: float):
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
    train_sim,
    substation_meta: Optional[Dict[str, dict]] = None,
):
    """Per-substation load curves (power vs. time) from a solved train sim.

    Splits each recorded time step's substation load (pantograph power drawn plus
    line loss) between the one or two feeding substations by the same
    current-division fractions used in :func:`compute_line_losses`, giving a load
    time series per substation. Resolution follows the simulation ``save_interval``.

    Parameters
    ----------
    train_sim : SpeedLimitTrainSim or dict
        A solved train simulation (or its ``to_pydict()``) with history.
    substation_meta : dict, optional
        Metadata from :func:`apply_line_losses` (adds a ``grid_emissions_region``
        column is not applied here; regions are available via that mapping).

    Returns
    -------
    polars.DataFrame
        Columns ``time_seconds`` and ``load_watts__<district_id>`` (one per
        substation), plus ``load_watts__total``.
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

    def _feed_at(offset: float):
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


def __getattr__(name: str):
    """Lazily re-export :class:`Substation` (source-independent) without eagerly
    importing the optimizer's heavy dependencies (pyomo, etc.)."""
    if name == "Substation":
        from altrios.optimization.energization_placement import Substation

        return Substation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

