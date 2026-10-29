"""Mixed-integer linear program (MILP) for optimal energization placement.

The optimizer decides, subject to a capital budget, which track segments to
energize (e.g. with catenary or wireless power transfer) and which transmission
substations to build. It prioritizes high-grade regions (where electrification
yields the largest operational benefit) while respecting substation-feeding
(spacing) constraints.

Decision variables:

- `x[p] in {0, 1}`: electrify track piece `p` (a link sub-interval).
- `s[c] in {0, 1}`: install shared energization structure in corridor cell `c`.
- `y[j] in {0, 1}`: build candidate substation `j` (lump-sum capital).

Structure sites and parallel tracks: when `structure_spacing_miles` is set, every
link is densified into uniform along-track **pieces**, and pieces whose midpoints
fall within `corridor_width_miles` of one another are clustered into shared
**structure sites** `s[c]` (masts, foundations). Both directions of a
double-track line and any sidings at the same location fall into one site and
share a single structure cost; each individual track then pays only a marginal
per-mile electrification cost via `x[p]`. This makes it economical to wire every
parallel track once the structure is built. Clustering is purely geometric
(lat/lon), needing no centerline, topology, or est-time nets, so it is safe on
arbitrary imported networks. When `structure_spacing_miles` is `None`, each link
is treated as its own single-piece cell (no parallel grouping).

Data-schema assumption: substation candidates and track segments are assumed to
expose geographic positions. A `Substation` may be located either directly by
`(lat, lon)` or by `link_idx` (in which case the location is taken from that
link's geometry). Segment positions are derived from each link's `headings`
(lat/lon) geometry.

The optimizer targets `pyomo` for modeling and the HiGHS solver (`highspy`) via
Pyomo's APPSI interface.

Notes:
    - Feeding uses the STRONG disaggregated covering constraint
      `x[p] <= sum_{j in cover(p)} y[j]`; structure sharing likewise uses the
      strong per-piece linking `x[p] <= s[c]` rather than an aggregated Big-M row.
      Both yield a tight LP relaxation (near totally unimodular). The only row
      that breaks total unimodularity is the single budget knapsack constraint,
      so the MILP solves quickly.
    - Costs are linear in the decisions, so the whole program is a MILP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers.highs import Highs
import polars as pl

# local imports
import altrios as alt
from altrios.defaults import DISCOUNT_RATE

#: Miles per meter.
MILES_PER_METER = 0.000621371
#: Mean Earth radius in miles (for haversine distance).
_EARTH_RADIUS_MILES = 3958.7613
#: Charge-credited (Tier 1) gap horizon: cap net-drain windows at this many full
#: usable packs. Longer over-drains are either implied by shorter windows or
#: physically un-closeable, so bounding the lookahead keeps the row count linear.
#: Tier 1 is a PROTOTYPE: with fine buckets it emits ~(this factor) windows per
#: start, i.e. many weighted-covering rows -- heavier than the TU ``sum u >= 1``
#: energy gap. Lower it (or coarsen buckets) to trade fidelity for speed.
_GAP_WINDOW_KWH_FACTOR = 2.0
#: Upper bound (miles) on the shared run/gap bucket width. Buckets target
#: ~param/8 but are capped here so a LONG run/gap (e.g. a 36 mi battery recharge
#: run) does not force sub-mile buckets -> a quadratic blow-up in the min-run
#: difference rows. 2 mi keeps run/gap boundaries within ~2 mi while cutting the
#: min-run row count ~30x on long-range BEL problems.
_BUCKET_MAX_MILES = 2.0


@dataclass
class EconomicParameters:
    """Time-value parameters for the lifecycle (min-cost) objective.

    The horizon and discount rate enter the objective only through a single
    present-value annuity factor (:meth:`pvaf`) multiplying the per-piece annual
    operating cost, so the program stays linear.
    """

    #: Project analysis horizon (years).
    project_lifespan_years: int
    #: Annual discount rate (fraction). Defaults to the ALTRIOS-wide
    #: :data:`altrios.defaults.DISCOUNT_RATE`.
    discount_rate: float = DISCOUNT_RATE

    def pvaf(self) -> float:
        """Return the present-value annuity factor.

        Returns:
            ``sum_{t=1}^{N} (1+r)^{-t}`` for horizon ``N`` and rate ``r``; equals
            ``N`` when the rate is zero.
        """
        r = float(self.discount_rate)
        n = int(self.project_lifespan_years)
        if r == 0.0:
            return float(n)
        return (1.0 - (1.0 + r) ** (-n)) / r


@dataclass
class Substation:
    """A candidate transmission substation the optimizer may build.

    Location is given either directly via ``(lat, lon)`` or by ``link_idx``,
    from which the position is derived using that link's geometry.
    """

    id: Hashable
    link_idx: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    #: Lump-sum capital cost ($) to build this substation. If ``None``, the
    #: global :attr:`EnergizationOptCosts.substation_fixed_cost` is used.
    fixed_cost: Optional[float] = None
    #: Transmission/interconnection distance (miles) used with the per-mile
    #: substation cost coefficient.
    connection_miles: float = 0.0
    #: If True, the substation already exists: it is treated as always-built at
    #: zero cost, so it provides feeding coverage for free and is not a decision
    #: variable. Use this for termini/yards that are already electrified.
    existing: bool = False
    #: Maximum real power (W) this substation can deliver to the catenary. Used by
    #: the line-loss model to derate deliverable pantograph power. ``None`` falls
    #: back to a global default.
    max_power_watts: Optional[float] = None
    #: Cambium (or other) grid-emissions region this substation draws from, used
    #: in postprocessing to attribute catenary energy and line losses to the
    #: correct grid. ``None`` leaves attribution unresolved.
    grid_emissions_region: Optional[str] = None
    #: Optional per-substation nominal catenary voltage (V) override. ``None``
    #: uses the global system voltage passed to the line-loss model.
    voltage_volts: Optional[float] = None
    #: Conversion efficiency (grid -> catenary) of this substation, in ``(0, 1]``.
    #: For AC this is essentially the step-down transformer efficiency (~0.99);
    #: for DC it also includes AC->DC rectifier loss (typically ~0.97). Used by
    #: the line-loss postprocessing to scale grid-side (substation input) energy
    #: up from the delivered catenary energy. ``None`` falls back to a global
    #: default.
    efficiency: Optional[float] = None


@dataclass
class EnergizationOptCosts:
    """Cost coefficients for the objective/budget.

    Energization cost is split into a shared per-corridor **structure** cost (piling,
    masts, foundations -- installed once per corridor cell regardless of how many
    parallel tracks are wired) and a per-track **marginal** electrification cost
    (contact wire, feeders, etc.). This split makes it economical to wire every
    parallel track in a corridor once the shared structure is paid for.

    A single ``cost_per_mile`` coefficient, when provided, applies on a per-mile
    basis to BOTH energization (marginal, per track) and substations (times each
    substation's ``connection_miles``). More specific coefficients override it.
    """

    #: Shared per-mile coefficient applied to both energization and substations.
    cost_per_mile: Optional[float] = None
    #: Marginal energization cost per mile of *each* electrified track ($/mi).
    #: Falls back to ``cost_per_mile``.
    energization_cost_per_mile: Optional[float] = None
    #: Shared corridor structure cost per mile ($/mi), charged once per corridor
    #: cell when any track in it is electrified. Defaults to 0 (no shared
    #: structure -> each track priced independently).
    structure_cost_per_mile: float = 0.0
    #: Substation per-mile interconnection cost ($/mi). Falls back to
    #: ``cost_per_mile``.
    substation_cost_per_mile: Optional[float] = None
    #: Default lump-sum substation cost ($) when a ``Substation`` does not set
    #: its own ``fixed_cost``.
    substation_fixed_cost: float = 0.0


@dataclass(frozen=True)
class PieceBenefitContext:
    """Context passed to a custom piece-benefit function (``piece_benefit_fn``).

    A track *piece* is one electrifiable link sub-interval. The fields describe
    that piece so a benefit model can score it; additional fields may be appended
    in the future without breaking existing callables.
    """

    #: Index of the link this piece belongs to.
    link_idx: int
    #: Piece start offset along the link (meters).
    offset_start_meters: float
    #: Piece end offset along the link (meters).
    offset_end_meters: float
    #: Piece length (miles).
    length_miles: float
    #: Whether the piece's corridor cell is high-grade (>= ``grade_threshold``).
    high_grade: bool


@dataclass
class EnergizationPlacementInputs:
    """All data the optimizer needs."""

    #: Network on which energization is being placed.
    network: alt.Network
    #: Candidate substations that may be built.
    substations: List[Substation] = field(default_factory=list)
    #: Cost coefficients.
    costs: EnergizationOptCosts = field(default_factory=EnergizationOptCosts)
    #: Capital budget cap ($). ``None`` means unconstrained.
    budget: Optional[float] = None
    #: Power limit (W) applied to any energized segment.
    uniform_power_limit_watts: float = 0.0
    #: Grade at/above which a segment is treated as "high priority".
    grade_threshold: float = 0.01
    #: Benefit multiplier applied to high-grade segments (relative to 1.0 for
    #: ordinary segments). Higher values prioritize electrifying steep track.
    high_grade_benefit_weight: float = 10.0
    #: Maximum allowed distance (miles) from an energized segment to its nearest
    #: built substation. This is the substation feeding/spacing constraint.
    substation_feed_distance_miles: float = 20.0
    #: If True, high-grade segments are forced to be energized (hard constraint)
    #: rather than merely prioritized via the objective.
    require_high_grade_coverage: bool = False
    #: Est-time nets. In per-link mode, restricts candidate segments to the links
    #: these trains traverse. Not required by structure-site (corridor) mode.
    est_time_nets: List[alt.EstTimeNet] = field(default_factory=list)
    #: Link indices to EXCLUDE from candidacy (a denylist), applied after the
    #: est-time-net / flip candidate selection in both modes. Used to feed a
    #: realized-draw exclusion back into the buildout: links a fleet never drew
    #: catenary from (e.g. a descending/no-traction direction) are removed so the
    #: optimizer neither wires nor pays for them. ``None`` (default) excludes none.
    excluded_link_idxs: Optional[List[int]] = None
    #: If set, enable structure-site (corridor) mode: densify every link into
    #: uniform along-track pieces of this length (miles) and cluster parallel
    #: tracks that share a location so they share one structure cost. Requires
    #: lat/lon geometry but no centerline/topology (safe on arbitrary networks).
    #: If ``None`` (default), each link is treated as its own single-piece cell.
    structure_spacing_miles: Optional[float] = None
    #: Lateral distance (miles) within which pieces from parallel tracks are
    #: clustered into the same shared structure site (structure-site mode only).
    corridor_width_miles: float = 0.05
    #: If True, use the WHOLE-SITE (per-cell) formulation: energizing a structure
    #: site wires every track piece at it (``x == s``), collapsing the per-piece
    #: electrification binaries onto the site indicators ``s[c]``. This shrinks the
    #: model from ~one binary per (short) link to ~one per corridor site and
    #: strengthens the LP relaxation, so the optimality gap closes far faster; the
    #: modeling cost is that parallel tracks/sidings at a site are wired together
    #: (no partial-cross-section electrification). ``False`` (default) keeps the
    #: fine per-piece formulation.
    electrify_whole_site: bool = False
    #: Minimum contiguous energized run length in CORRIDOR miles (parallel tracks
    #: and sidings in a structure site count once, so both directions of a
    #: double-track energized for 5 mi count as 5, not 10). Once an interior run
    #: of energized corridor starts it must span at least this distance, which
    #: forbids tiny isolated slivers even when structure/substation costs are
    #: zero. Runs may start at any site boundary (nothing is pre-grouped). Runs
    #: that reach a corridor end/junction are exempt (they may continue there).
    #: Requires structure-site mode (``structure_spacing_miles``). ``None``
    #: (default) disables the constraint.
    min_energized_length_miles: Optional[float] = None
    #: Maximum contiguous UNelectrified run ("gap") length in CORRIDOR miles along
    #: the mainline. No stretch of un-energized corridor may exceed this, which is
    #: the range constraint for a battery-electric locomotive that must recharge
    #: from the wire before its battery depletes. Physically it is the distance a
    #: full battery can bridge; derive it from battery size with
    #: :func:`altrios.energization.max_gap_miles_from_battery`
    #: (``usable_kWh * (1 - reserve) / avg_kWh_per_mile``). Simplifying
    #: assumptions: any electrified segment recharges the battery, consumption is a
    #: constant average per mile, and the battery is full at each corridor end
    #: (yard charging), so gaps at the ends obey the same limit (no exemption).
    #: Unlike the minimum-run constraint this can FORCE electrification, so a
    #: feasibility pre-check runs and raises ``ValueError`` if a gap cannot be
    #: closed by any reachable substation. Requires structure-site mode with a
    #: contiguous mainline path. ``None`` (default) disables the constraint.
    max_energized_gap_miles: Optional[float] = None
    #: ENERGY-based maximum-gap limit (kWh): the usable battery traction energy a
    #: BEL can spend between charges. When set (with
    #: ``gap_energy_kwh_per_mile_by_link``), the gap constraint limits the
    #: cumulative *drain energy* of any unelectrified corridor window rather than
    #: its distance, so wire is required where the battery actually depletes
    #: fastest (steep/heavy track) instead of on a flat per-mile budget. Takes
    #: precedence over ``max_energized_gap_miles``. ``None`` (default) disables it.
    max_energized_gap_kwh: Optional[float] = None
    #: Per-link single-locomotive traction DRAIN per corridor-mile (kWh/mi),
    #: keyed by link index. Supplies the drain weights ``delta`` for the
    #: energy-based gap (``max_energized_gap_kwh``) and the energy-priority
    #: objective term. Required whenever ``max_energized_gap_kwh`` is set.
    gap_energy_kwh_per_mile_by_link: Optional[Dict[int, float]] = None
    #: Per-link NET CHARGE delivered per corridor-mile of ENERGIZED track
    #: (kWh/mi), keyed by link index. When supplied alongside
    #: ``max_energized_gap_kwh`` this enables the **charge-credited** gap family
    #: (recharge Tier 1): an energized bucket credits only ``charge + drain`` (not
    #: an unconditional full recharge), so short slivers that cannot refill the
    #: pack no longer "reset" a gap. ``None`` (default) uses the simpler
    #: ``sum u >= 1`` energy-gap family (any single energized bucket resets the
    #: gap), which is totally unimodular but assumes any energized segment fully
    #: recharges.
    charge_energy_kwh_per_mile_by_link: Optional[Dict[int, float]] = None
    #: Per-link REGEN recovery per corridor-mile (kWh/mi), keyed by link index.
    #: When supplied, the gap/SOC constraints use NET drain (``drain - regen``) so
    #: regenerative braking extends the battery range through gaps. Regen buckets
    #: should carry ``charge = 0`` (no catenary charging value where the train is
    #: braking). ``None`` (default) treats regen as zero.
    regen_energy_kwh_per_mile_by_link: Optional[Dict[int, float]] = None
    #: If True, replace the covering gap family with the explicit SOC-recursion
    #: (Tier 2) range model: a continuous per-bucket state of charge evolves along
    #: the mainline as ``soc[b] = soc[b-1] + charge*u - net_drain - curtail`` with a
    #: free curtailment slack ``curtail >= 0`` (the linear, big-M-free encoding of
    #: charge saturation) and is held in ``[0, max_energized_gap_kwh]``, forbidding
    #: depletion. Faithful and order-aware (subsumes the covering gaps) but heavier
    #: (continuous state, not totally unimodular). Requires ``max_energized_gap_kwh``
    #: + ``gap_energy_kwh_per_mile_by_link``. ``False`` (default) uses the covering.
    battery_soc_state_model: bool = False
    #: Optional custom objective-benefit model. When provided, it is called once
    #: per track piece to return that piece's objective coefficient ``b_p``,
    #: replacing the built-in grade-priority-weighted mileage. This is the seam
    #: for future, less grade-specific objectives -- e.g. gallons of diesel
    #: displaced by a dual-mode locomotive, computed from a first-pass
    #: simulation and mapped onto pieces. The optimizer itself is agnostic to how
    #: ``b_p`` is produced (it maximizes ``sum_p b_p * x[p]``), so no constraint
    #: or solver change is needed to swap objectives. ``None`` (default) uses the
    #: grade heuristic ``miles * (high_grade_benefit_weight if high_grade else 1)``.
    piece_benefit_fn: Optional[Callable[["PieceBenefitContext"], float]] = None
    #: Objective mode. ``"max_benefit"`` (default) maximizes priority-weighted
    #: benefit subject to the budget (the rule-based/grade formulation).
    #: ``"min_cost"`` instead minimizes lifecycle cost = capital + discounted
    #: operating cost, and requires ``economics`` and ``piece_opex_fn``.
    objective: str = "max_benefit"
    #: Time-value parameters (horizon + discount rate) for ``"min_cost"`` mode.
    #: Ignored by ``"max_benefit"``.
    economics: Optional[EconomicParameters] = None
    #: Per-piece **annual net operating cost delta** ($/yr) of electrifying a
    #: piece: electricity purchased (incl. line + conversion loss) minus displaced
    #: diesel; negative means a saving. Called once per piece (like
    #: ``piece_benefit_fn``). Populate it from a first-pass dual-mode simulation
    #: and fuel/electricity prices (see ``energization.make_piece_opex_fn``).
    #: Required for ``"min_cost"``; ignored otherwise.
    piece_opex_fn: Optional[Callable[["PieceBenefitContext"], float]] = None


@dataclass
class EnergizationPlacementResult:
    """Optimizer outputs."""

    #: Network with the optimized energization placement applied.
    network: Optional[alt.Network] = None
    #: Link indices that were energized (any sub-interval).
    energized_link_idxs: List[int] = field(default_factory=list)
    #: Per-link energized sub-intervals (meters): link_idx -> [(start, end), ...].
    energized_link_intervals: Dict[int, List[Tuple[float, float]]] = field(
        default_factory=dict
    )
    #: Substation ids that were built.
    built_substation_ids: List[Hashable] = field(default_factory=list)
    #: Corridor cells where shared energization structure was installed.
    installed_structure_cell_ids: List[Hashable] = field(default_factory=list)
    #: Total capital cost of the chosen plan ($).
    total_cost: Optional[float] = None
    #: Total electrified track length (miles), summed across parallel tracks.
    energized_track_miles: Optional[float] = None
    #: Shared-structure length (miles) actually charged: the along-track length of
    #: every installed structure cell, counted ONCE however many parallel tracks
    #: share it. This is the mileage ``structure_cost_per_mile`` multiplied, and the
    #: right basis for any downstream capital accounting -- ``energized_track_miles``
    #: over-charges structure on multi-track corridors.
    structure_route_miles: Optional[float] = None
    #: Objective (priority-weighted energized miles in ``"max_benefit"`` mode, or
    #: lifecycle cost $ in ``"min_cost"`` mode).
    objective_value: Optional[float] = None
    #: Annual net operating cost ($/yr) of the chosen plan (sum of energized
    #: pieces' ``piece_opex_fn``). ``None`` outside ``"min_cost"`` mode.
    annual_opex_cost: Optional[float] = None
    #: Discounted lifetime operating cost ($) = ``pvaf * annual_opex_cost``.
    #: ``None`` outside ``"min_cost"`` mode.
    discounted_opex_cost: Optional[float] = None
    #: Total lifecycle cost ($) = ``total_cost + discounted_opex_cost`` (the
    #: minimized objective). ``None`` outside ``"min_cost"`` mode.
    lifecycle_cost: Optional[float] = None
    #: Solver termination condition / status string.
    status: Optional[str] = None
    #: Best objective bound (dual bound) reported by the solver -- the proven
    #: bound on the optimum. ``None`` if unavailable.
    objective_bound: Optional[float] = None
    #: Absolute optimality gap = ``|objective_value - objective_bound|`` (same
    #: units as the objective: lifecycle $ in min-cost mode). ``None`` if no bound.
    optimality_gap_abs: Optional[float] = None
    #: Relative optimality gap = ``optimality_gap_abs / |objective_value|`` (the
    #: MIP gap; 0.0 means proven optimal). ``None`` if no bound/incumbent.
    optimality_gap_rel: Optional[float] = None


@dataclass
class _ProblemData:
    """Pre-computed problem data shared between build and extract steps.

    The model is expressed over corridor *cells* (shared structure) and track
    *pieces* (a link sub-interval that can be individually electrified). In
    per-link mode each link is one cell containing one whole-link piece.
    """

    cell_ids: List[Hashable]
    struct_cost: Dict[Hashable, float]
    #: Along-track corridor length (miles) of each cell/structure site. Used by
    #: the minimum-run-length constraint to measure distance in CORRIDOR miles
    #: (parallel tracks in a site do not multiply it).
    cell_len_miles: Dict[Hashable, float]
    cell_high_grade: Dict[Hashable, bool]
    piece_ids: List[Hashable]
    piece_cell: Dict[Hashable, Hashable]
    piece_link: Dict[Hashable, int]
    piece_offsets: Dict[Hashable, Tuple[float, float]]
    piece_miles: Dict[Hashable, float]
    piece_benefit: Dict[Hashable, float]
    piece_marginal_cost: Dict[Hashable, float]
    piece_annual_opex: Dict[Hashable, float]
    piece_is_mainline: Dict[Hashable, bool]
    piece_cover: Dict[Hashable, List[Hashable]]
    piece_auto_covered: Dict[Hashable, bool]
    substation_ids: List[Hashable]
    sub_cost: Dict[Hashable, float]
    network_dict: tuple
    #: Structure-cell ids in contiguous corridor order along the mainline path,
    #: used to order sites for the minimum-run-length constraint (robust to
    #: sidings/junctions, unlike link-chain ordering). ``None`` -> fall back to
    #: link chains.
    mainline_site_seq: Optional[List[Hashable]] = None
    #: Along-corridor arc position (meters) of each cell in ``mainline_site_seq``
    #: (aligned), used to bucket the corridor for the min-run-length constraint.
    mainline_site_arcs: Optional[List[float]] = None
    #: Per-cell single-traversal traction DRAIN energy (kWh) = per-mile drain
    #: density * cell corridor length. Weights the energy-based gap constraint.
    #: Empty when no drain densities were supplied.
    cell_drain_kwh: Dict[Hashable, float] = field(default_factory=dict)
    #: Per-cell NET CHARGE energy (kWh) delivered if the cell is energized =
    #: per-mile charge density * cell corridor length. Empty unless the
    #: charge-credited (Tier 1) gap family is enabled.
    cell_charge_kwh: Dict[Hashable, float] = field(default_factory=dict)
    #: Per-cell REGEN recovery energy (kWh) over one corridor traversal =
    #: per-mile regen density * cell corridor length. Empty unless regen supplied;
    #: subtracted from drain to form NET drain in the gap/SOC constraints.
    cell_regen_kwh: Dict[Hashable, float] = field(default_factory=dict)


def _first_not_none(*values: float | None) -> float:
    """Return the first value that is not ``None``.

    Args:
        *values: Candidate values in priority order.

    Returns:
        The first non-``None`` value, or ``0.0`` when all are ``None``.
    """
    for value in values:
        if value is not None:
            return value
    return 0.0


def _cell_energy_from_density(
    density_by_link: dict[int, float] | None,
    cell_ids: list[Hashable],
    piece_ids: list[Hashable],
    piece_cell: dict[Hashable, Hashable],
    piece_link: dict[Hashable, int],
    cell_len_miles: dict[Hashable, float],
) -> dict[Hashable, float]:
    """Convert per-link energy densities to per-cell energies.

    A cell's rate is the mean density over its pieces with nonzero density.

    Args:
        density_by_link: Energy per corridor-mile (kWh/mi), keyed by link index.
        cell_ids: Corridor cell ids.
        piece_ids: Track piece ids.
        piece_cell: Cell id of each piece.
        piece_link: Link index of each piece.
        cell_len_miles: Corridor length (miles) of each cell.

    Returns:
        Rate times cell length (kWh) per cell; ``{}`` when no density is given.
    """
    if not density_by_link:
        return {}
    # Average over the pieces whose link the fleet actually USED. A cell groups
    # parallel tracks at one location, and an untravelled track carries density
    # 0, so averaging over every piece reported a fraction of the travelled
    # track's rate -- on dc_9kv battery-electric, a median of 0.336 across 99.1%
    # of cells. A train traverses the corridor on ONE track; the rate that
    # belongs to the cell is that track's.
    sums: Dict[Hashable, float] = {c: 0.0 for c in cell_ids}
    counts: Dict[Hashable, int] = {c: 0 for c in cell_ids}
    all_sums: Dict[Hashable, float] = {c: 0.0 for c in cell_ids}
    all_counts: Dict[Hashable, int] = {c: 0 for c in cell_ids}
    for p in piece_ids:
        c = piece_cell[p]
        d = float(density_by_link.get(piece_link[p], 0.0))
        all_sums[c] += d
        all_counts[c] += 1
        if d != 0.0:
            sums[c] += d
            counts[c] += 1

    def _rate(c: Hashable) -> float:
        """Return the mean piece density (kWh/mi) of cell ``c``."""
        if counts[c]:
            return sums[c] / counts[c]
        # No piece in this cell carries traffic: fall back to the all-piece mean
        # (zero, or whatever proxy the caller supplied) rather than divide by 0.
        return all_sums[c] / all_counts[c] if all_counts[c] else 0.0

    return {c: _rate(c) * cell_len_miles.get(c, 0.0) for c in cell_ids}



def _compute_piece_benefit(
    inputs: EnergizationPlacementInputs,
    *,
    link_idx: int,
    off0: float,
    off1: float,
    miles: float,
    high_grade: bool,
) -> float:
    """Return the objective coefficient ``b_p`` of one piece.

    Args:
        inputs: Optimizer inputs; ``piece_benefit_fn`` overrides the grade rule.
        link_idx: Index of the link the piece belongs to.
        off0: Piece start offset along the link (m).
        off1: Piece end offset along the link (m).
        miles: Piece length (miles).
        high_grade: Whether the piece's cell is high-grade.

    Returns:
        ``piece_benefit_fn`` value when set; otherwise ``miles`` scaled by
        ``high_grade_benefit_weight`` for high-grade pieces.
    """
    if inputs.piece_benefit_fn is not None:
        return float(
            inputs.piece_benefit_fn(
                PieceBenefitContext(
                    link_idx=int(link_idx),
                    offset_start_meters=float(off0),
                    offset_end_meters=float(off1),
                    length_miles=float(miles),
                    high_grade=bool(high_grade),
                )
            )
        )
    weight = inputs.high_grade_benefit_weight if high_grade else 1.0
    return miles * weight


def _compute_piece_annual_opex(
    inputs: EnergizationPlacementInputs,
    *,
    link_idx: int,
    off0: float,
    off1: float,
    miles: float,
    high_grade: bool,
) -> float:
    """Return one piece's annual net operating cost delta.

    Args:
        inputs: Optimizer inputs supplying ``piece_opex_fn``.
        link_idx: Index of the link the piece belongs to.
        off0: Piece start offset along the link (m).
        off1: Piece end offset along the link (m).
        miles: Piece length (miles).
        high_grade: Whether the piece's cell is high-grade.

    Returns:
        ``piece_opex_fn`` value ($/yr, negative for a saving); ``0.0`` when no
        opex model is supplied.
    """
    if inputs.piece_opex_fn is None:
        return 0.0
    return float(
        inputs.piece_opex_fn(
            PieceBenefitContext(
                link_idx=int(link_idx),
                offset_start_meters=float(off0),
                offset_end_meters=float(off1),
                length_miles=float(miles),
                high_grade=bool(high_grade),
            )
        )
    )


def _haversine_miles(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Return the great-circle distance between two points.

    Args:
        lat1: Latitude of the first point (degrees).
        lon1: Longitude of the first point (degrees).
        lat2: Latitude of the second point (degrees).
        lon2: Longitude of the second point (degrees).

    Returns:
        Distance (miles).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return _EARTH_RADIUS_MILES * 2.0 * math.asin(math.sqrt(a))


def _link_midpoint_latlon(link: dict) -> tuple[float, float] | None:
    """Return a representative (lat, lon) for a link.

    Args:
        link: Link dict with a ``headings`` geometry list.

    Returns:
        (lat, lon) of the middle heading vertex, or ``None`` without geometry.
    """
    headings = link.get("headings") or []
    if not headings:
        return None
    mid = headings[len(headings) // 2]
    lat, lon = mid.get("lat"), mid.get("lon")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def _link_max_grade(link: dict) -> float:
    """Return the maximum grade along a link.

    Args:
        link: Link dict with an ``elevs`` elevation profile.

    Returns:
        Largest rise/run between consecutive samples with nonzero run; ``0.0``
        when fewer than two samples exist.
    """
    elevs = link.get("elevs") or []
    if len(elevs) < 2:
        return 0.0
    max_grade = -float("inf")
    for j in range(len(elevs) - 1):
        run = elevs[j + 1]["offset_meters"] - elevs[j]["offset_meters"]
        if run == 0:
            continue
        rise = elevs[j + 1]["elev_meters"] - elevs[j]["elev_meters"]
        max_grade = max(max_grade, rise / run)
    return max_grade if max_grade != -float("inf") else 0.0


def _ordered_path_link_idxs(est_time_net: alt.EstTimeNet) -> list[int]:
    """Return the link indices an est-time net traverses.

    Args:
        est_time_net: Estimated-time network of one train's path.

    Returns:
        Link indices ordered by scheduled clear time, consecutive repeats removed.
    """
    ordered = (
        pl.DataFrame(est_time_net.to_pydict()["val"])
        .lazy()
        .unnest("link_event")
        .with_columns(
            pl.col("est_type")
            .replace({0: "Arrive", 1: "Clear", 2: "Fake"})
            .cast(pl.Categorical),
            pl.selectors.contains("idx").replace({0: None}).cast(pl.UInt32),
        )
        .filter(pl.col("est_type") == pl.lit("Clear"))
        .select("link_idx", "time_sched")
        .sort("time_sched")
        .select("link_idx")
        .collect()
        .get_column("link_idx")
        .drop_nulls()
        .to_list()
    )
    deduped: List[int] = []
    for idx in ordered:
        if not deduped or deduped[-1] != idx:
            deduped.append(int(idx))
    return deduped


def _candidate_link_idxs(
    inputs: EnergizationPlacementInputs, link_by_idx: dict[int, dict]
) -> list[int]:
    """Return the link indices eligible for electrification.

    With est-time nets, keep traversed links and their ``idx_flip`` counterparts.

    Args:
        inputs: Optimizer inputs (``est_time_nets``, ``excluded_link_idxs``).
        link_by_idx: Link dicts keyed by ``idx_curr``.

    Returns:
        Candidate link indices with ``excluded_link_idxs`` removed; every link is
        a candidate when no est-time nets are given.
    """
    if not inputs.est_time_nets:
        candidates = list(link_by_idx.keys())
    else:
        seen: set = set()
        ordered: List[int] = []
        for etn in inputs.est_time_nets:
            for idx in _ordered_path_link_idxs(etn):
                if idx not in link_by_idx:
                    continue
                flip = link_by_idx[idx].get("idx_flip", 0)
                for j in (idx, flip):
                    if j and j in link_by_idx and j not in seen:
                        seen.add(j)
                        ordered.append(j)
        candidates = ordered
    # Denylist: drop links a realized-draw pass flagged as never energized in use.
    if inputs.excluded_link_idxs:
        excl = set(inputs.excluded_link_idxs)
        candidates = [i for i in candidates if i not in excl]
    return candidates


def _merge_intervals(
    intervals: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent intervals.

    Args:
        intervals: ``(start, end)`` pairs in any order.

    Returns:
        Sorted, disjoint ``(start, end)`` pairs.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: List[List[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _resolve_substations(
    inputs: EnergizationPlacementInputs, link_by_idx: dict[int, dict]
) -> tuple[
    list[Hashable],
    dict[Hashable, float],
    dict[Hashable, tuple[float, float]],
    list[tuple[float, float]],
]:
    """Resolve substation positions and capital costs.

    Args:
        inputs: Optimizer inputs supplying ``substations`` and ``costs``.
        link_by_idx: Link dicts keyed by ``idx_curr``, for ``link_idx`` locations.

    Returns:
        Candidate substation ids, their capital cost ($, fixed plus per-mile
        connection cost), their (lat, lon), and the (lat, lon) of existing
        substations.

    Raises:
        ValueError: If a substation has neither lat/lon nor a valid ``link_idx``.
    """
    sub_per_mile = _first_not_none(
        inputs.costs.substation_cost_per_mile, inputs.costs.cost_per_mile
    )
    sub_positions: Dict[Hashable, Tuple[float, float]] = {}
    sub_cost: Dict[Hashable, float] = {}
    substation_ids: List[Hashable] = []
    existing_positions: List[Tuple[float, float]] = []
    for sub in inputs.substations:
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
        if sub.existing:
            existing_positions.append(pos)
            continue
        substation_ids.append(sub.id)
        sub_positions[sub.id] = pos
        fixed = _first_not_none(sub.fixed_cost, inputs.costs.substation_fixed_cost)
        sub_cost[sub.id] = fixed + sub_per_mile * sub.connection_miles
    return substation_ids, sub_cost, sub_positions, existing_positions


def _piece_coverage(
    pos: tuple[float, float] | None,
    existing_positions: list[tuple[float, float]],
    sub_positions: dict[Hashable, tuple[float, float]],
    feed_dist: float,
) -> tuple[list[Hashable], bool]:
    """Find substations within great-circle feed range of a point.

    Args:
        pos: (lat, lon) of the point, or ``None`` when unknown.
        existing_positions: (lat, lon) of already-built substations.
        sub_positions: (lat, lon) of candidate substations, keyed by id.
        feed_dist: Maximum feeding distance (miles).

    Returns:
        Candidate ids in range, and whether an existing substation already covers
        the point (the id list is then empty).
    """
    if pos is None:
        return [], False
    if any(
        _haversine_miles(pos[0], pos[1], ep[0], ep[1]) <= feed_dist
        for ep in existing_positions
    ):
        return [], True
    covering = [
        sid
        for sid, sp in sub_positions.items()
        if _haversine_miles(pos[0], pos[1], sp[0], sp[1]) <= feed_dist
    ]
    return covering, False


class _ArcCoverage:
    """Feeding coverage by along-corridor TRACK distance (matches line-loss model).

    Projects every substation and query point onto the mainline corridor arc via
    the same ``_build_corridor_arc`` machinery :func:`altrios.energization.apply_line_losses`
    uses, so a point is "within range" of a substation when their arc-length
    separation -- i.e. distance travelled along the track -- is ``<=`` the feed
    distance, NOT the great-circle ("crow-flies") distance. Falls back to
    great-circle distance only if a corridor arc cannot be built (e.g. a network
    with no discernible mainline).
    """

    def __init__(
        self,
        network_dict: tuple,
        existing_positions: list[tuple[float, float]],
        sub_positions: dict[Hashable, tuple[float, float]],
        feed_dist_miles: float,
        sample_spacing_meters: float = 200.0,
    ) -> None:
        """Project substations onto the mainline corridor arc.

        Falls back to great-circle distance when no corridor arc can be built.

        Args:
            network_dict: ``network.to_pydict()`` output.
            existing_positions: (lat, lon) of already-built substations.
            sub_positions: (lat, lon) of candidate substations, keyed by id.
            feed_dist_miles: Maximum feeding distance (miles).
            sample_spacing_meters: Along-track sample spacing (m) for the arc.
        """
        self._existing = list(existing_positions)
        self._subs = dict(sub_positions)
        self._feed_miles = float(feed_dist_miles)
        self._feed_m = self._feed_miles / MILES_PER_METER
        self._project = None
        self._arc_start: Dict[int, float] = {}
        try:
            from altrios.energization import _build_corridor_arc

            _arc_start, _main_set, project = _build_corridor_arc(
                network_dict, sample_spacing_meters
            )
            self._project = project
            self._arc_start = _arc_start
            self._ex_arcs = [project(p[0], p[1]) for p in self._existing if p]
            self._sub_arcs = {
                sid: project(pos[0], pos[1]) for sid, pos in self._subs.items()
            }
        except Exception:  # noqa: BLE001 - no mainline -> crow-flies fallback
            self._project = None

    def ordered_mainline_links(self) -> list[int]:
        """Return mainline link indices in corridor order.

        Returns:
            Link indices sorted by the arc position of their start; empty when no
            arc was built.
        """
        return sorted(self._arc_start, key=lambda k: self._arc_start[k])

    def arc(self, lat: float | None, lon: float | None) -> float | None:
        """Return the along-corridor arc position of a point.

        Args:
            lat: Latitude (degrees).
            lon: Longitude (degrees).

        Returns:
            Arc position (m), or ``None`` without an arc or coordinates.
        """
        if self._project is None or lat is None or lon is None:
            return None
        return self._project(lat, lon)

    def coverage(
        self, lat: float | None, lon: float | None
    ) -> tuple[list[Hashable], bool]:
        """Find substations within track-distance feed range of a point.

        Args:
            lat: Latitude (degrees).
            lon: Longitude (degrees).

        Returns:
            Candidate ids whose arc separation is within range, and whether an
            existing substation already covers the point.
        """
        if lat is None or lon is None:
            return [], False
        if self._project is None:  # fallback: great-circle distance
            return _piece_coverage(
                (lat, lon), self._existing, self._subs, self._feed_miles
            )
        x = self._project(lat, lon)
        if any(abs(x - ea) <= self._feed_m for ea in self._ex_arcs):
            return [], True
        cover = [
            sid for sid, sa in self._sub_arcs.items() if abs(x - sa) <= self._feed_m
        ]
        return cover, False


def link_coverage(
    network: alt.Network,
    substations: list[Substation] | None,
    built_substation_ids: list[Hashable] | None = None,
    *,
    substation_feed_distance_miles: float = 20.0,
    sample_spacing_meters: float = 200.0,
) -> dict[int, bool]:
    """Report which links lie within substation feed range.

    Range is measured along the corridor arc, with a great-circle fallback.

    Args:
        network: Network whose links are checked.
        substations: Substations; existing ones and built ones provide coverage.
        built_substation_ids: Ids of candidate substations treated as built.
        substation_feed_distance_miles: Maximum feeding distance (miles).
        sample_spacing_meters: Along-track sample spacing (m) for the arc.

    Returns:
        ``True`` per link index when the link midpoint is within range.
    """
    built = set(built_substation_ids or [])
    network_dict = network.to_pydict()
    link_by_idx: Dict[int, dict] = {
        link["idx_curr"]: link for link in network_dict[1][1:]
    }
    existing: List[Tuple[float, float]] = []
    sub_pos: Dict[Hashable, Tuple[float, float]] = {}
    for sub in substations or []:
        lat = getattr(sub, "lat", None)
        lon = getattr(sub, "lon", None)
        link_idx = getattr(sub, "link_idx", None)
        if lat is not None and lon is not None:
            pos: Optional[Tuple[float, float]] = (float(lat), float(lon))
        elif link_idx is not None and link_idx in link_by_idx:
            pos = _link_midpoint_latlon(link_by_idx[link_idx])
        else:
            pos = None
        if pos is None:
            continue
        if getattr(sub, "existing", False):
            existing.append(pos)
        elif getattr(sub, "id", None) in built:
            sub_pos[getattr(sub, "id")] = pos
    cov = _ArcCoverage(
        network_dict, existing, sub_pos, substation_feed_distance_miles,
        sample_spacing_meters,
    )
    out: Dict[int, bool] = {}
    for idx, link in link_by_idx.items():
        ll = _link_midpoint_latlon(link)
        if ll is None:
            out[idx] = False
            continue
        c, a = cov.coverage(ll[0], ll[1])
        out[idx] = bool(a or c)
    return out


def _prepare(inputs: EnergizationPlacementInputs) -> _ProblemData:
    """Build problem data for the selected formulation mode.

    Args:
        inputs: Optimizer inputs.

    Returns:
        Structure-site problem data when ``structure_spacing_miles`` is set,
        otherwise per-link problem data.
    """
    network_dict = inputs.network.to_pydict()
    link_by_idx: Dict[int, dict] = {
        link["idx_curr"]: link for link in network_dict[1][1:]
    }
    substation_ids, sub_cost, sub_positions, existing_positions = (
        _resolve_substations(inputs, link_by_idx)
    )
    common = (substation_ids, sub_cost, sub_positions, existing_positions)
    # Feeding coverage by along-corridor TRACK distance (same metric as the
    # line-loss model), replacing the former great-circle (crow-flies) rule.
    arc_cov = _ArcCoverage(
        network_dict, existing_positions, sub_positions,
        inputs.substation_feed_distance_miles,
    )
    common = (substation_ids, sub_cost, sub_positions, existing_positions, arc_cov)
    if inputs.structure_spacing_miles is not None:
        return _prepare_sites(inputs, network_dict, link_by_idx, *common)
    return _prepare_per_link(inputs, network_dict, link_by_idx, *common)


def _prepare_per_link(
    inputs: EnergizationPlacementInputs,
    network_dict: tuple,
    link_by_idx: dict[int, dict],
    substation_ids: list[Hashable],
    sub_cost: dict[Hashable, float],
    sub_positions: dict[Hashable, tuple[float, float]],
    existing_positions: list[tuple[float, float]],
    arc_cov: _ArcCoverage,
) -> _ProblemData:
    """Build problem data with one whole-link piece per cell.

    Args:
        inputs: Optimizer inputs.
        network_dict: ``network.to_pydict()`` output.
        link_by_idx: Link dicts keyed by ``idx_curr``.
        substation_ids: Candidate substation ids.
        sub_cost: Capital cost ($) of each candidate substation.
        sub_positions: (lat, lon) of each candidate substation.
        existing_positions: (lat, lon) of already-built substations.
        arc_cov: Feeding-coverage evaluator.

    Returns:
        Problem data in which every candidate link is its own cell and piece.
    """
    cat_per_mile = _first_not_none(
        inputs.costs.energization_cost_per_mile, inputs.costs.cost_per_mile
    )
    struct_per_mile = inputs.costs.structure_cost_per_mile

    candidate_idxs = _candidate_link_idxs(inputs, link_by_idx)

    cell_ids: List[Hashable] = []
    struct_cost: Dict[Hashable, float] = {}
    cell_len_miles: Dict[Hashable, float] = {}
    cell_high_grade: Dict[Hashable, bool] = {}
    piece_ids: List[Hashable] = []
    piece_cell: Dict[Hashable, Hashable] = {}
    piece_link: Dict[Hashable, int] = {}
    piece_offsets: Dict[Hashable, Tuple[float, float]] = {}
    piece_miles: Dict[Hashable, float] = {}
    piece_benefit: Dict[Hashable, float] = {}
    piece_marginal_cost: Dict[Hashable, float] = {}
    piece_annual_opex: Dict[Hashable, float] = {}
    piece_is_mainline: Dict[Hashable, bool] = {}
    piece_cover: Dict[Hashable, List[Hashable]] = {}
    piece_auto_covered: Dict[Hashable, bool] = {}

    for idx in candidate_idxs:
        link = link_by_idx[idx]
        length_m = float(link["length_meters"])
        miles = length_m * MILES_PER_METER
        high = _link_max_grade(link) >= inputs.grade_threshold

        cell_ids.append(idx)
        struct_cost[idx] = struct_per_mile * miles
        cell_len_miles[idx] = miles
        cell_high_grade[idx] = high

        piece_ids.append(idx)
        piece_cell[idx] = idx
        piece_link[idx] = idx
        piece_offsets[idx] = (0.0, length_m)
        piece_miles[idx] = miles
        piece_benefit[idx] = _compute_piece_benefit(
            inputs, link_idx=idx, off0=0.0, off1=length_m, miles=miles, high_grade=high
        )
        piece_marginal_cost[idx] = cat_per_mile * miles
        piece_annual_opex[idx] = _compute_piece_annual_opex(
            inputs, link_idx=idx, off0=0.0, off1=length_m, miles=miles, high_grade=high
        )
        piece_is_mainline[idx] = True

        cover, auto = arc_cov.coverage(
            *(_link_midpoint_latlon(link) or (None, None))
        )
        piece_cover[idx] = cover
        piece_auto_covered[idx] = auto

    cell_drain_kwh = _cell_energy_from_density(
        inputs.gap_energy_kwh_per_mile_by_link,
        cell_ids, piece_ids, piece_cell, piece_link, cell_len_miles,
    )
    cell_charge_kwh = _cell_energy_from_density(
        inputs.charge_energy_kwh_per_mile_by_link,
        cell_ids, piece_ids, piece_cell, piece_link, cell_len_miles,
    )
    cell_regen_kwh = _cell_energy_from_density(
        inputs.regen_energy_kwh_per_mile_by_link,
        cell_ids, piece_ids, piece_cell, piece_link, cell_len_miles,
    )

    return _ProblemData(
        cell_ids=cell_ids,
        struct_cost=struct_cost,
        cell_len_miles=cell_len_miles,
        cell_high_grade=cell_high_grade,
        piece_ids=piece_ids,
        piece_cell=piece_cell,
        piece_link=piece_link,
        piece_offsets=piece_offsets,
        piece_miles=piece_miles,
        piece_benefit=piece_benefit,
        piece_marginal_cost=piece_marginal_cost,
        piece_annual_opex=piece_annual_opex,
        piece_is_mainline=piece_is_mainline,
        piece_cover=piece_cover,
        piece_auto_covered=piece_auto_covered,
        substation_ids=substation_ids,
        sub_cost=sub_cost,
        network_dict=network_dict,
        cell_drain_kwh=cell_drain_kwh,
        cell_charge_kwh=cell_charge_kwh,
        cell_regen_kwh=cell_regen_kwh,
    )


def _network_vertex_df(
    network_dict: tuple, candidate_idxs: list[int] | None = None
) -> pl.DataFrame:
    """Explode link geometry into a per-vertex dataframe.

    Args:
        network_dict: ``network.to_pydict()`` output; the dummy link is skipped.
        candidate_idxs: Link indices to keep; ``None`` keeps every link.

    Returns:
        One row per heading vertex with ``idx_curr``, ``length_meters``,
        ``offset_meters``, ``lat`` and ``lon``, sorted by link then offset; rows
        lacking coordinates are dropped.
    """
    rows = [
        {
            "idx_curr": int(link["idx_curr"]),
            "length_meters": float(link["length_meters"]),
            "headings": link.get("headings") or [],
        }
        for link in network_dict[1][1:]
    ]
    df = pl.DataFrame(rows)
    if candidate_idxs is not None:
        df = df.filter(pl.col("idx_curr").is_in(list(candidate_idxs)))
    return (
        df.explode("headings")
        .with_columns(
            pl.col("headings").struct.field("offset_meters").alias("offset_meters"),
            pl.col("headings").struct.field("lat").alias("lat"),
            pl.col("headings").struct.field("lon").alias("lon"),
        )
        .drop("headings")
        .drop_nulls(["offset_meters", "lat", "lon"])
        .sort("idx_curr", "offset_meters")
    )


def _densify_links(
    network_dict: tuple,
    candidate_idxs: list[int],
    spacing_m: float,
) -> tuple[
    list[int], list[float], list[float], list[float], list[float], list[float]
]:
    """Split each candidate link into uniform along-track pieces.

    Args:
        network_dict: ``network.to_pydict()`` output.
        candidate_idxs: Link indices to densify.
        spacing_m: Target piece length (m).

    Returns:
        Parallel lists ``(link_idx, off_start_m, off_end_m, mid_lat, mid_lon,
        length_m)`` with one entry per piece; lat/lon are interpolated at the
        piece midpoint.
    """
    import numpy as np

    vdf = _network_vertex_df(network_dict, candidate_idxs)
    link_len = {
        int(link["idx_curr"]): float(link["length_meters"])
        for link in network_dict[1][1:]
    }
    p_link: List[int] = []
    p_off0: List[float] = []
    p_off1: List[float] = []
    p_lat: List[float] = []
    p_lon: List[float] = []
    p_len: List[float] = []
    for part in vdf.partition_by("idx_curr", maintain_order=True):
        idx = int(part["idx_curr"][0])
        length = link_len.get(idx, 0.0)
        if length <= 0.0:
            continue
        voff = part["offset_meters"].to_numpy()
        vlat = part["lat"].to_numpy()
        vlon = part["lon"].to_numpy()
        if voff.size < 1:
            continue
        n = max(1, math.ceil(length / spacing_m))
        edges = np.linspace(0.0, length, n + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        mlat = np.interp(mids, voff, vlat)
        mlon = np.interp(mids, voff, vlon)
        for i in range(n):
            p_link.append(idx)
            p_off0.append(float(edges[i]))
            p_off1.append(float(edges[i + 1]))
            p_lat.append(float(mlat[i]))
            p_lon.append(float(mlon[i]))
            p_len.append(float(edges[i + 1] - edges[i]))
    return p_link, p_off0, p_off1, p_lat, p_lon, p_len


def _prepare_sites(
    inputs: EnergizationPlacementInputs,
    network_dict: tuple,
    link_by_idx: dict[int, dict],
    substation_ids: list[Hashable],
    sub_cost: dict[Hashable, float],
    sub_positions: dict[Hashable, tuple[float, float]],
    existing_positions: list[tuple[float, float]],
    arc_cov: _ArcCoverage,
) -> _ProblemData:
    """Build problem data with pieces clustered into structure sites.

    Piece midpoints within ``corridor_width_miles`` of each other share one cell.

    Args:
        inputs: Optimizer inputs.
        network_dict: ``network.to_pydict()`` output.
        link_by_idx: Link dicts keyed by ``idx_curr``.
        substation_ids: Candidate substation ids.
        sub_cost: Capital cost ($) of each candidate substation.
        sub_positions: (lat, lon) of each candidate substation.
        existing_positions: (lat, lon) of already-built substations.
        arc_cov: Feeding-coverage evaluator; also orders the mainline sites.

    Returns:
        Problem data with per-site structure costs, per-piece marginal costs and
        the mainline site sequence used by the run and gap constraints.

    Raises:
        ValueError: If no candidate link carries lat/lon geometry.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    spacing_m = inputs.structure_spacing_miles / MILES_PER_METER
    lateral_m = inputs.corridor_width_miles / MILES_PER_METER
    cat_per_mile = _first_not_none(
        inputs.costs.energization_cost_per_mile, inputs.costs.cost_per_mile
    )
    struct_per_mile = inputs.costs.structure_cost_per_mile

    # Candidate links: all links, optionally restricted to traversed links.
    candidate_idxs = _candidate_link_idxs(inputs, link_by_idx)

    grade_high = {
        idx: _link_max_grade(link_by_idx[idx]) >= inputs.grade_threshold
        for idx in candidate_idxs
    }

    # Densify each link into uniform along-track pieces (shared representation).
    p_link, p_off0, p_off1, p_lat, p_lon, p_len_m = _densify_links(
        network_dict, candidate_idxs, spacing_m
    )
    p_high = [grade_high[idx] for idx in p_link]

    n_pieces = len(p_link)
    if n_pieces == 0:
        raise ValueError(
            "structure-site (corridor) mode requires lat/lon geometry on the "
            "network links; none was found."
        )

    # Project midpoints to a local equirectangular meter frame, then cluster
    # pieces within `lateral_m` into shared structure sites via connected
    # components. Along-track spacing (>> lateral_m) keeps consecutive pieces on
    # one link in separate sites; only laterally-adjacent parallel tracks merge.
    lat_arr = np.asarray(p_lat)
    lon_arr = np.asarray(p_lon)
    lat0 = float(lat_arr.mean())
    lon0 = float(lon_arr.mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    coords = np.column_stack(
        [(lon_arr - lon0) * m_per_deg_lon, (lat_arr - lat0) * m_per_deg_lat]
    )
    parent = list(range(n_pieces))

    def _find(a: int) -> int:
        """Return the union-find root of piece ``a``."""
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if lateral_m > 0.0:
        for a, b in cKDTree(coords).query_pairs(r=lateral_m):
            ra, rb = _find(int(a)), _find(int(b))
            if ra != rb:
                parent[ra] = rb
    site_of = [_find(i) for i in range(n_pieces)]

    # Assemble cells (structure sites) and pieces.
    cell_ids: List[Hashable] = sorted(set(site_of))
    cell_high_grade: Dict[Hashable, bool] = {c: False for c in cell_ids}
    struct_len_miles: Dict[Hashable, float] = {c: 0.0 for c in cell_ids}

    piece_ids: List[Hashable] = list(range(n_pieces))
    piece_cell: Dict[Hashable, Hashable] = {}
    piece_link: Dict[Hashable, int] = {}
    piece_offsets: Dict[Hashable, Tuple[float, float]] = {}
    piece_miles: Dict[Hashable, float] = {}
    piece_is_mainline: Dict[Hashable, bool] = {}
    for i in range(n_pieces):
        c = site_of[i]
        miles = p_len_m[i] * MILES_PER_METER
        piece_cell[i] = c
        piece_link[i] = p_link[i]
        piece_offsets[i] = (p_off0[i], p_off1[i])
        piece_miles[i] = miles
        # No mainline/siding distinction in structure-site mode: every piece is
        # real electrifiable track (so require_high_grade_coverage forces all
        # high-grade pieces on).
        piece_is_mainline[i] = True
        cell_high_grade[c] = cell_high_grade[c] or p_high[i]
        # Structure at a site spans the (uniform) along-track length of its
        # pieces; charge it once regardless of how many parallel tracks join.
        struct_len_miles[c] = max(struct_len_miles[c], miles)

    struct_cost: Dict[Hashable, float] = {
        c: struct_per_mile * struct_len_miles[c] for c in cell_ids
    }

    # Benefit, marginal cost, and feeding coverage per piece.
    piece_benefit: Dict[Hashable, float] = {}
    piece_marginal_cost: Dict[Hashable, float] = {}
    piece_annual_opex: Dict[Hashable, float] = {}
    piece_cover: Dict[Hashable, List[Hashable]] = {}
    piece_auto_covered: Dict[Hashable, bool] = {}
    for i in range(n_pieces):
        c = piece_cell[i]
        miles = piece_miles[i]
        piece_benefit[i] = _compute_piece_benefit(
            inputs,
            link_idx=piece_link[i],
            off0=piece_offsets[i][0],
            off1=piece_offsets[i][1],
            miles=miles,
            high_grade=cell_high_grade[c],
        )
        piece_marginal_cost[i] = cat_per_mile * miles
        piece_annual_opex[i] = _compute_piece_annual_opex(
            inputs,
            link_idx=piece_link[i],
            off0=piece_offsets[i][0],
            off1=piece_offsets[i][1],
            miles=miles,
            high_grade=cell_high_grade[c],
        )
        cover, auto = arc_cov.coverage(p_lat[i], p_lon[i])
        piece_cover[i] = cover
        piece_auto_covered[i] = auto

    # Structure-cell sequence in contiguous corridor order (by the mainline
    # path's cumulative arc), so the minimum-run-length constraint has one
    # unbroken sequence spanning the corridor -- robust to sidings/junctions and
    # free of the spurious gaps that per-cell arc means or link chains produce.
    mainline_site_seq: Optional[List[Hashable]] = None
    mainline_site_arcs: Optional[List[float]] = None
    _link_pieces_ord: Dict[int, List[int]] = {}
    for i in range(n_pieces):
        _link_pieces_ord.setdefault(p_link[i], []).append(i)
    for _v in _link_pieces_ord.values():
        _v.sort(key=lambda i: p_off0[i])
    _arc_start = getattr(arc_cov, "_arc_start", {}) or {}
    _seq: List[Hashable] = []
    _arcs: List[float] = []
    for link_idx in arc_cov.ordered_mainline_links():
        base = _arc_start.get(link_idx, 0.0)
        for i in _link_pieces_ord.get(link_idx, []):
            c = piece_cell[i]
            if not _seq or _seq[-1] != c:
                _seq.append(c)
                _arcs.append(base + p_off0[i])
    if len(_seq) >= 2:
        mainline_site_seq = _seq
        mainline_site_arcs = _arcs

    cell_drain_kwh = _cell_energy_from_density(
        inputs.gap_energy_kwh_per_mile_by_link,
        cell_ids, piece_ids, piece_cell, piece_link, struct_len_miles,
    )
    cell_charge_kwh = _cell_energy_from_density(
        inputs.charge_energy_kwh_per_mile_by_link,
        cell_ids, piece_ids, piece_cell, piece_link, struct_len_miles,
    )
    cell_regen_kwh = _cell_energy_from_density(
        inputs.regen_energy_kwh_per_mile_by_link,
        cell_ids, piece_ids, piece_cell, piece_link, struct_len_miles,
    )

    return _ProblemData(
        cell_ids=cell_ids,
        struct_cost=struct_cost,
        cell_len_miles=struct_len_miles,
        cell_high_grade=cell_high_grade,
        piece_ids=piece_ids,
        piece_cell=piece_cell,
        piece_link=piece_link,
        piece_offsets=piece_offsets,
        piece_miles=piece_miles,
        piece_benefit=piece_benefit,
        piece_marginal_cost=piece_marginal_cost,
        piece_annual_opex=piece_annual_opex,
        piece_is_mainline=piece_is_mainline,
        piece_cover=piece_cover,
        piece_auto_covered=piece_auto_covered,
        substation_ids=substation_ids,
        sub_cost=sub_cost,
        network_dict=network_dict,
        mainline_site_seq=mainline_site_seq,
        mainline_site_arcs=mainline_site_arcs,
        cell_drain_kwh=cell_drain_kwh,
        cell_charge_kwh=cell_charge_kwh,
        cell_regen_kwh=cell_regen_kwh,
    )


def _build_link_chains(
    link_by_idx: dict[int, dict],
    link_pieces: dict[int, list[Hashable]],
) -> list[list[Hashable]]:
    """Chain per-link piece lists into linear physical sequences.

    Links join through ``idx_next`` only where the connection is unambiguous.

    Args:
        link_by_idx: Link dicts keyed by ``idx_curr``.
        link_pieces: Piece ids of each candidate link in increasing-offset order.

    Returns:
        Chains of piece ids in forward-travel order; each chain ends at a
        junction or ambiguous connection.
    """
    cand = set(link_pieces.keys())
    succ: Dict[int, int] = {}
    for a in cand:
        link = link_by_idx.get(a)
        if link is None:
            continue
        nxt = link.get("idx_next", 0)
        nxt_alt = link.get("idx_next_alt", 0)
        if nxt in cand and nxt_alt not in cand:
            succ[a] = nxt
    # Keep an edge only if the successor has a single candidate predecessor.
    pred_count: Dict[int, int] = {c: 0 for c in cand}
    for b in succ.values():
        pred_count[b] += 1
    succ = {a: b for a, b in succ.items() if pred_count[b] == 1}
    has_pred = set(succ.values())

    chains: List[List[Hashable]] = []
    for a in cand:
        if a in has_pred:
            continue  # not a chain head; reached via its predecessor
        chain: List[Hashable] = []
        cur: Optional[int] = a
        seen: set = set()
        while cur is not None and cur in cand and cur not in seen:
            seen.add(cur)
            chain.extend(link_pieces[cur])
            cur = succ.get(cur)
        chains.append(chain)
    return chains


def _install_corridor_buckets(
    model: pyo.ConcreteModel, data: _ProblemData, bucket_mi: float,
    tie_structure_to_x: bool = True,
) -> tuple[int, list[float], dict[Hashable, int]]:
    """Install the shared corridor bucket layer.

    Adds ``structure_used``, one binary ``u[b]`` per bucket and ``run_couple`` rows.

    Args:
        model: Model holding ``C``, ``s`` and, unless whole-site, ``x``; receives
            ``structure_used`` (``s[c] <= sum_{p in c} x[p]``), set ``RunBucket``,
            variables ``u`` and ``run_couple`` (``s[c] == u[bucket(c)]``).
        data: Problem data with ``mainline_site_seq`` and cell lengths.
        bucket_mi: Target bucket length (corridor miles).
        tie_structure_to_x: Whether to add ``structure_used``; ``False`` in
            whole-site mode, where ``s`` is the electrification variable.

    Returns:
        Bucket count, each bucket's length (miles) and the cell-to-bucket map;
        ``(0, [], {})`` when no contiguous mainline sequence exists.
    """
    pieces_in_cell: Dict[Hashable, List[Hashable]] = {}
    for p in data.piece_ids:
        pieces_in_cell.setdefault(data.piece_cell[p], []).append(p)

    # Structure only where at least one track is energized (ties s to x). Skipped
    # in whole-site mode, where s IS the electrification (there is no x).
    if tie_structure_to_x:
        def _structure_used_rule(m: pyo.ConcreteModel, c: Hashable):
            """Bound ``s[c]`` by the electrified pieces in cell ``c``."""
            ps = pieces_in_cell.get(c, [])
            if not ps:
                return m.s[c] == 0
            return m.s[c] <= sum(m.x[p] for p in ps)

        model.structure_used = pyo.Constraint(model.C, rule=_structure_used_rule)

    seq = getattr(data, "mainline_site_seq", None)
    if not seq or len(seq) < 2:
        return 0, [], {}

    # Tile the CONTIGUOUS mainline cell sequence into ~bucket_mi buckets by
    # accumulating cell length (assign a cell to the current bucket, then roll
    # over once the bucket fills). Sequential bucket ids leave NO gaps.
    cell_bucket: Dict[Hashable, int] = {}
    b = 0
    acc = 0.0
    for c in seq:
        if c in cell_bucket:
            continue
        cell_bucket[c] = b
        acc += data.cell_len_miles.get(c, 0.0)
        if acc >= bucket_mi:
            b += 1
            acc = 0.0
    n_buckets = (max(cell_bucket.values()) + 1) if cell_bucket else 0
    if n_buckets < 2:
        return 0, [], {}

    bucket_len_miles = [0.0] * n_buckets
    for c, bkt in cell_bucket.items():
        bucket_len_miles[bkt] += data.cell_len_miles.get(c, 0.0)

    model.RunBucket = pyo.Set(initialize=range(n_buckets), ordered=True)
    model.u = pyo.Var(model.RunBucket, domain=pyo.Binary)

    couple = list(cell_bucket.items())
    model.run_couple_index = pyo.Set(initialize=range(len(couple)), ordered=True)

    def _couple_rule(m: pyo.ConcreteModel, t: int):
        """Tie a cell's ``s[c]`` to its bucket's ``u[b]``."""
        c, bkt = couple[t]
        return m.s[c] == m.u[bkt]

    model.run_couple = pyo.Constraint(model.run_couple_index, rule=_couple_rule)
    return n_buckets, bucket_len_miles, cell_bucket


def _install_soc_state_model(
    model: pyo.ConcreteModel,
    n_buckets: int,
    bucket_net: list[float],
    bucket_charge: list[float],
    pack_kwh: float,
) -> None:
    """Install the explicit state-of-charge battery-range model.

    Adds continuous ``soc`` and ``curtail`` variables with a per-bucket energy balance.

    Args:
        model: Model holding bucket binaries ``u``; receives set ``SocBucket``,
            ``soc[b]`` in ``[0, pack_kwh]``, slack ``curtail[b] >= 0`` and
            ``soc_balance`` (``soc[b] == soc[b-1] + charge*u[b] - net -
            curtail[b]``, starting from a full pack).
        n_buckets: Number of corridor buckets.
        bucket_net: Net drain (drain minus regen) per bucket (kWh).
        bucket_charge: Charge banked per bucket when energized (kWh).
        pack_kwh: Usable pack energy (kWh); the initial and maximum ``soc``.
    """
    model.SocBucket = pyo.Set(initialize=range(n_buckets), ordered=True)
    model.soc = pyo.Var(model.SocBucket, domain=pyo.NonNegativeReals,
                        bounds=(0.0, pack_kwh))
    model.curtail = pyo.Var(model.SocBucket, domain=pyo.NonNegativeReals)

    def _soc_rule(m: pyo.ConcreteModel, b: int):
        """Balance ``soc[b]`` against charge, net drain and curtailment."""
        prev = pack_kwh if b == 0 else m.soc[b - 1]
        return (
            m.soc[b]
            == prev + bucket_charge[b] * m.u[b] - bucket_net[b] - m.curtail[b]
        )

    model.soc_balance = pyo.Constraint(model.SocBucket, rule=_soc_rule)


def build_model(
    inputs: EnergizationPlacementInputs, data: _ProblemData
) -> pyo.ConcreteModel:
    """Construct the energization-placement MILP.

    Args:
        inputs: Optimizer inputs selecting the objective and optional constraints.
        data: Pre-computed cells, pieces, costs and feeding coverage.

    Returns:
        Model with binaries ``s`` (structure per cell), ``x`` (per piece, absent
        in whole-site mode) and ``y`` (per substation); the structure-linking,
        coverage, run-length, gap, high-grade and budget constraints requested;
        and the objective ``obj``.

    Raises:
        ValueError: If a requested constraint lacks its prerequisites, a gap
            cannot be closed by any reachable substation, or ``objective`` is
            unknown.
    """
    model = pyo.ConcreteModel(name="energization_placement")

    model.C = pyo.Set(initialize=data.cell_ids, ordered=True)
    model.P = pyo.Set(initialize=data.piece_ids, ordered=True)
    model.J = pyo.Set(initialize=data.substation_ids, ordered=True)

    # s[c]: install shared corridor structure; x[p]: electrify a track piece;
    # y[j]: build a substation. All binary. In WHOLE-SITE mode there is no per-
    # piece x -- energizing a site (s[c]=1) wires all its pieces, so wire cost /
    # opex / benefit / coverage collapse onto s and the piece binaries vanish.
    _per_cell = bool(inputs.electrify_whole_site)
    model.s = pyo.Var(model.C, domain=pyo.Binary)
    if not _per_cell:
        model.x = pyo.Var(model.P, domain=pyo.Binary)
    model.y = pyo.Var(model.J, domain=pyo.Binary)

    if _per_cell:
        _cell_pieces: Dict[Hashable, List[Hashable]] = {}
        for p in data.piece_ids:
            _cell_pieces.setdefault(data.piece_cell[p], []).append(p)
        cell_marginal = {c: 0.0 for c in data.cell_ids}
        cell_opex = {c: 0.0 for c in data.cell_ids}
        cell_benefit = {c: 0.0 for c in data.cell_ids}
        cell_cover: Dict[Hashable, List[Hashable]] = {}
        cell_auto: Dict[Hashable, bool] = {}
        for c in data.cell_ids:
            ps = _cell_pieces.get(c, [])
            cell_marginal[c] = sum(data.piece_marginal_cost[p] for p in ps)
            cell_opex[c] = sum(data.piece_annual_opex[p] for p in ps)
            cell_benefit[c] = sum(data.piece_benefit[p] for p in ps)
            cov: set = set()
            auto = False
            for p in ps:
                cov.update(data.piece_cover[p])
                auto = auto or data.piece_auto_covered[p]
            cell_cover[c] = list(cov)
            cell_auto[c] = auto

    # Strong per-piece structure linking: a track piece can only be electrified
    # if its corridor cell's shared structure is installed. (Whole-site mode has
    # no per-piece x -- s IS the electrification -- so this is skipped.)
    if not _per_cell:
        def _structure_rule(m: pyo.ConcreteModel, p: Hashable):
            """Allow ``x[p]`` only when its cell's structure is installed."""
            return m.x[p] <= m.s[data.piece_cell[p]]

        model.structure = pyo.Constraint(model.P, rule=_structure_rule)

    # Feeding/coverage: a piece (or, in whole-site mode, a site) may be energized
    # only if an in-range substation is built. Pieces fed by a pre-existing
    # substation are auto-covered (unconstrained); those with no in-range
    # candidate (and no existing feed) are forced off.
    if not _per_cell:
        def _coverage_rule(m: pyo.ConcreteModel, p: Hashable):
            """Allow ``x[p]`` only when an in-range substation is built."""
            if data.piece_auto_covered[p]:
                return pyo.Constraint.Skip
            covering = data.piece_cover[p]
            if not covering:
                return m.x[p] == 0
            return m.x[p] <= sum(m.y[j] for j in covering)

        model.coverage = pyo.Constraint(model.P, rule=_coverage_rule)
    else:
        def _coverage_rule_cell(m: pyo.ConcreteModel, c: Hashable):
            """Allow ``s[c]`` only when an in-range substation is built."""
            if cell_auto[c]:
                return pyo.Constraint.Skip
            cov = cell_cover[c]
            if not cov:
                return m.s[c] == 0
            return m.s[c] <= sum(m.y[j] for j in cov)

        model.coverage = pyo.Constraint(model.C, rule=_coverage_rule_cell)

    # Optional minimum contiguous energized run length, measured in CORRIDOR
    # miles (NOT track-miles: parallel tracks/sidings in a structure site count
    # once). Corridor site-sequences are built from the piece chains (spanning
    # link boundaries via idx_next), then any interior run of energized sites is
    # required to span >= the minimum. Structure is tied to energization
    # (s[c] <= sum_{p in c} x[p]) so s[c] == 1 iff the corridor site is actually
    # energized -- this keeps the guarantee even when structure costs are zero.
    # The run itself may start at ANY site boundary (nothing is pre-grouped), so
    # the optimizer keeps full freedom over where the >= min-length run begins.
    # Optional corridor run-shape constraints: minimum contiguous ENERGIZED run
    # (min_energized_length_miles) and/or maximum contiguous UNelectrified gap
    # (max_energized_gap_miles). Both are measured in CORRIDOR miles (parallel
    # tracks/sidings in a structure site count once) and share one coarse bucket
    # layer along the mainline (u[b], with s[c] == u[bucket(c)]); when both are
    # active the finer bucket width is used so each is enforced at adequate
    # resolution.
    _need_buckets = (
        inputs.min_energized_length_miles is not None
        or inputs.max_energized_gap_miles is not None
        or inputs.max_energized_gap_kwh is not None
    )
    if _need_buckets:
        _widths = []
        if inputs.min_energized_length_miles is not None:
            _widths.append(min(_BUCKET_MAX_MILES,
                               max(inputs.min_energized_length_miles / 8.0, 1e-6)))
        if inputs.max_energized_gap_miles is not None:
            _widths.append(min(_BUCKET_MAX_MILES,
                               max(inputs.max_energized_gap_miles / 8.0, 1e-6)))
        if inputs.max_energized_gap_kwh is not None:
            # Size buckets to resolve the ENERGY budget: convert the kWh cap to an
            # equivalent distance via the average drain density so resolution
            # matches the miles-based gap (~gap/8 per bucket).
            _tot_kwh = sum(data.cell_drain_kwh.values())
            _tot_mi = sum(
                data.cell_len_miles.get(c, 0.0) for c in data.cell_drain_kwh
            )
            _avg_drain = (_tot_kwh / _tot_mi) if _tot_mi > 0 else 0.0
            _gap_equiv_mi = (
                inputs.max_energized_gap_kwh / _avg_drain if _avg_drain > 0 else 0.5
            )
            _widths.append(min(_BUCKET_MAX_MILES, max(_gap_equiv_mi / 8.0, 1e-6)))
        bucket_mi = min(_widths)
        n_buckets, bucket_len_miles, cell_bucket = _install_corridor_buckets(
            model, data, bucket_mi, tie_structure_to_x=not _per_cell
        )
    else:
        n_buckets, bucket_len_miles, cell_bucket = 0, [], {}
    # Kept for the warm-start seed, which must keep s[c] == u[bucket] consistent.
    model._cell_bucket = dict(cell_bucket)

    # Minimum contiguous energized run length. Once an interior run of energized
    # corridor starts it must span >= min_len, which forbids tiny isolated slivers
    # even when structure/substation costs are zero. Runs may start at any bucket
    # boundary (nothing is pre-grouped); runs reaching a corridor end/junction are
    # exempt (they may continue there). Uses the shared bucket layer on the
    # mainline, or falls back to per-cell link chains when no mainline exists.
    if inputs.min_energized_length_miles is not None:
        min_len = inputs.min_energized_length_miles
        if n_buckets >= 2:
            window = max(1, int(round(min_len / bucket_mi)))
            # One contiguous corridor -> one bucket run [0 .. n_buckets-1]; any
            # interior energized run must span >= window buckets (>= min_len).
            triples: set = set()
            for i in range(1, n_buckets):  # i=0 is a corridor end -> exempt
                last = i + window - 1
                if last > n_buckets - 1:
                    continue  # within min_len of the corridor end -> exempt
                for k in range(i + 1, last + 1):
                    triples.add((k, i, i - 1))
            tl = list(triples)
            if tl:
                model.min_run_index = pyo.Set(
                    initialize=range(len(tl)), ordered=True
                )

                def _min_run_rule(m: pyo.ConcreteModel, t: int):
                    """Force ``u[j]`` on when a run starts at ``i``."""
                    j, i, i_prev = tl[t]
                    return m.u[j] >= m.u[i] - m.u[i_prev]

                model.min_run = pyo.Constraint(
                    model.min_run_index, rule=_min_run_rule
                )
        else:
            # Fallback: per-cell link chains (breaks at junctions). structure_used
            # was already installed by _install_corridor_buckets above.
            link_by_idx = {L["idx_curr"]: L for L in data.network_dict[1][1:]}
            link_pieces: Dict[int, List[Hashable]] = {}
            for p in data.piece_ids:
                link_pieces.setdefault(data.piece_link[p], []).append(p)
            for lp in link_pieces.values():
                lp.sort(key=lambda p: data.piece_offsets[p][0])
            triples = set()
            for chain in _build_link_chains(link_by_idx, link_pieces):
                cseq: List[Hashable] = []
                for p in chain:
                    c = data.piece_cell[p]
                    if not cseq or cseq[-1] != c:
                        cseq.append(c)
                if len(cseq) < 2:
                    continue
                lengths = [data.cell_len_miles.get(c, 0.0) for c in cseq]
                n_seq = len(cseq)
                for i in range(1, n_seq):
                    cum = 0.0
                    reach = None
                    for k in range(i, n_seq):
                        cum += lengths[k]
                        if cum >= min_len - 1e-9:
                            reach = k
                            break
                    if reach is None:
                        continue
                    for k in range(i + 1, reach + 1):
                        triples.add((cseq[k], cseq[i], cseq[i - 1]))
            tl = list(triples)
            if tl:
                model.min_run_index = pyo.Set(
                    initialize=range(len(tl)), ordered=True
                )

                def _min_run_rule(m: pyo.ConcreteModel, t: int):
                    """Force ``s[j]`` on when a run starts at ``i``."""
                    j, i, i_prev = tl[t]
                    return m.s[j] >= m.s[i] - m.s[i_prev]

                model.min_run = pyo.Constraint(
                    model.min_run_index, rule=_min_run_rule
                )

    # Maximum-gap (battery-electric range) constraint. Three variants share the
    # bucket layer and all forbid a BEL from depleting between charges:
    #   * DISTANCE (max_energized_gap_miles): no unelectrified window longer than
    #     the cap; sum_{b in window} u[b] >= 1. Interval matrix -> TU.
    #   * ENERGY (max_energized_gap_kwh, no charge densities): same sum u >= 1 but
    #     windows are bounded by cumulative DRAIN ENERGY exceeding the usable
    #     battery kWh, so wire is required where the pack drains fastest (steep/
    #     heavy track) rather than on a flat per-mile budget. Still TU. Assumes any
    #     single energized bucket fully recharges the pack (resets the gap).
    #   * CHARGE-CREDITED / recharge Tier 1 (energy + charge densities): drops the
    #     "any bucket fully recharges" assumption. For every contiguous window S
    #     the NET drain (drain on un-energized buckets minus charge banked on
    #     energized ones) may not exceed the usable pack:
    #         sum_{b in S} (drain_b + charge_b) u[b] >= sum_{b in S} drain_b - usable
    #     One row per window; all-positive u weights over CONTIGUOUS support
    #     (weighted covering). Emitting it for every window ending at each bucket
    #     covers the unknown "last full charge" point, so it is SUFFICIENT for SOC
    #     feasibility with no continuous state, no saturation, no big-M -- weighted
    #     C1P support keeps the LP tight though not strictly TU. The ENERGY form
    #     above is exactly this with charge_b = 0 and each row relaxed to
    #     sum u >= 1. Windows are capped at ``_GAP_WINDOW_KWH_FACTOR`` full packs
    #     of drain (a prototype horizon: longer over-drains are either handled by
    #     shorter windows or physically un-closeable).
    _energy_gap = inputs.max_energized_gap_kwh is not None
    if inputs.max_energized_gap_miles is not None or _energy_gap:
        if n_buckets < 2:
            raise ValueError(
                "max_energized_gap_miles/kwh requires structure-site (corridor) "
                "mode with a contiguous mainline path, but none was found. Set "
                "structure_spacing_miles and ensure the network has lat/lon "
                "geometry."
            )
        if _energy_gap and not data.cell_drain_kwh:
            raise ValueError(
                "max_energized_gap_kwh requires gap_energy_kwh_per_mile_by_link to "
                "supply per-link drain densities."
            )
        if inputs.battery_soc_state_model and not _energy_gap:
            raise ValueError(
                "battery_soc_state_model requires max_energized_gap_kwh (pack size) "
                "and gap_energy_kwh_per_mile_by_link (drain densities)."
            )

        # Per-bucket drain / charge / regen energy (kWh); bucket miles for
        # reporting. NET drain = drain - regen (regen extends range through gaps);
        # the covering / SOC constraints all use net drain.
        bucket_drain = [0.0] * n_buckets
        bucket_charge = [0.0] * n_buckets
        bucket_regen = [0.0] * n_buckets
        for c, bkt in cell_bucket.items():
            bucket_drain[bkt] += data.cell_drain_kwh.get(c, 0.0)
            bucket_charge[bkt] += data.cell_charge_kwh.get(c, 0.0)
            bucket_regen[bkt] += data.cell_regen_kwh.get(c, 0.0)
        bucket_net = [bucket_drain[b] - bucket_regen[b] for b in range(n_buckets)]
        prefix_mi = [0.0]
        for L in bucket_len_miles:
            prefix_mi.append(prefix_mi[-1] + L)

        # A bucket can be energized iff some cell it contains has a piece that is
        # auto-covered by an existing substation or reachable by a candidate.
        cell_coverable: Dict[Hashable, bool] = {}
        for p in data.piece_ids:
            if data.piece_auto_covered[p] or data.piece_cover[p]:
                cell_coverable[data.piece_cell[p]] = True
        bucket_coverable = [False] * n_buckets
        for c, bkt in cell_bucket.items():
            if cell_coverable.get(c):
                bucket_coverable[bkt] = True

        if inputs.battery_soc_state_model:
            _install_soc_state_model(
                model, n_buckets, bucket_net, bucket_charge,
                float(inputs.max_energized_gap_kwh),
            )
            _tier1 = False
        else:
            _tier1 = _energy_gap and bool(data.cell_charge_kwh)
        if inputs.battery_soc_state_model:
            pass  # SOC recursion installed above; skip the covering families
        elif _tier1:
            budget = float(inputs.max_energized_gap_kwh)
            window_cap = budget * _GAP_WINDOW_KWH_FACTOR
            # (window buckets, rhs) rows; rhs = cumulative NET drain - usable pack.
            rows: List[Tuple[Tuple[int, ...], float]] = []
            for i in range(n_buckets):
                cum_drain = 0.0
                for j in range(i, n_buckets):
                    cum_drain += bucket_net[j]
                    rhs = cum_drain - budget
                    if rhs > 1e-9:
                        w = tuple(range(i, j + 1))
                        rows.append((w, rhs))
                        # A window can be satisfied only if its coverable buckets
                        # can bank enough (net drain + charge); else the gap is
                        # un-closeable regardless of the solver.
                        max_bank = sum(
                            max(bucket_net[b] + bucket_charge[b], 0.0)
                            for b in w
                            if bucket_coverable[b]
                        )
                        if max_bank + 1e-9 < rhs:
                            raise ValueError(
                                f"max_energized_gap_kwh={budget:.0f} is infeasible: "
                                f"the corridor from mile {prefix_mi[i]:.1f} to "
                                f"{prefix_mi[j + 1]:.1f} would over-drain the battery "
                                f"by {rhs - max_bank:.0f} kWh even if every reachable "
                                f"segment in it were energized. Add a substation in "
                                f"range or increase substation_feed_distance_miles / "
                                f"battery size."
                            )
                    if cum_drain > window_cap:
                        break
            if rows:
                model.max_gap_index = pyo.Set(
                    initialize=range(len(rows)), ordered=True
                )

                def _max_gap_rule(m: pyo.ConcreteModel, t: int):
                    """Require window charge credit to cover its over-drain."""
                    w, rhs = rows[t]
                    # Floor per-bucket credit at 0: a regen bucket (net<0) with no
                    # charge cannot make the covering easier by being energized.
                    return (
                        sum(
                            max(bucket_net[b] + bucket_charge[b], 0.0) * m.u[b]
                            for b in w
                        )
                        >= rhs
                    )

                model.max_gap = pyo.Constraint(
                    model.max_gap_index, rule=_max_gap_rule
                )
        else:
            if _energy_gap:
                budget = float(inputs.max_energized_gap_kwh)
                weight = bucket_net
                label = f"max_energized_gap_kwh={budget:.0f}"
                unit = "kWh"
            else:
                budget = float(inputs.max_energized_gap_miles)
                weight = bucket_len_miles
                label = f"max_energized_gap_miles={budget}"
                unit = "mi"

            # Minimal window per start i: smallest [i..j] whose weight exceeds
            # the budget (distance or NET drain energy).
            windows: List[Tuple[int, ...]] = []
            for i in range(n_buckets):
                cum = 0.0
                j = i
                while j < n_buckets and cum + weight[j] <= budget:
                    cum += weight[j]
                    j += 1
                if j < n_buckets:  # else: tail to corridor end within budget -> ok
                    windows.append(tuple(range(i, j + 1)))

            # Feasibility pre-check: a binding window with no coverable bucket can
            # never satisfy sum u >= 1, so the gap is un-closeable.
            for w in windows:
                if not any(bucket_coverable[b] for b in w):
                    raise ValueError(
                        f"{label} ({unit}) is infeasible: the corridor from mile "
                        f"{prefix_mi[w[0]]:.1f} to {prefix_mi[w[-1] + 1]:.1f} exceeds "
                        f"the maximum gap yet contains no track reachable by any "
                        f"candidate or existing substation, so the gap cannot be "
                        f"closed. Add a substation in range or increase "
                        f"substation_feed_distance_miles."
                    )

            wl = list(dict.fromkeys(windows))  # dedup, preserve order
            if wl:
                model.max_gap_index = pyo.Set(
                    initialize=range(len(wl)), ordered=True
                )

                def _max_gap_rule(m: pyo.ConcreteModel, t: int):
                    """Require at least one energized bucket in the window."""
                    return sum(m.u[b] for b in wl[t]) >= 1

                model.max_gap = pyo.Constraint(
                    model.max_gap_index, rule=_max_gap_rule
                )

    # Optional hard requirement: electrify mainline track in every high-grade
    # cell (this pulls in the shared structure via the linking constraint).
    if inputs.require_high_grade_coverage:
        req = [
            p
            for p in data.piece_ids
            if data.piece_is_mainline[p] and data.cell_high_grade[data.piece_cell[p]]
        ]
        uncovered = [
            p
            for p in req
            if not data.piece_auto_covered[p] and not data.piece_cover[p]
        ]
        if uncovered:
            bad_links = sorted({data.piece_link[p] for p in uncovered})
            raise ValueError(
                "require_high_grade_coverage=True but these high-grade mainline "
                f"links have no substation within {inputs.substation_feed_distance_miles} "
                f"miles: {bad_links}. Add candidate substations or increase "
                "substation_feed_distance_miles."
            )

        req_set = set(req)

        if _per_cell:
            req_cells = {data.piece_cell[p] for p in req_set}

            def _require_cell(m: pyo.ConcreteModel, c: Hashable):
                """Force ``s[c]`` on for high-grade cells."""
                return m.s[c] == 1 if c in req_cells else pyo.Constraint.Skip

            model.require_high_grade = pyo.Constraint(model.C, rule=_require_cell)
        else:
            def _require_rule(m: pyo.ConcreteModel, p: Hashable):
                """Force ``x[p]`` on for high-grade mainline pieces."""
                if p in req_set:
                    return m.x[p] == 1
                return pyo.Constraint.Skip

            model.require_high_grade = pyo.Constraint(model.P, rule=_require_rule)

    # Single budget knapsack row (the only non-TU constraint): shared structure
    # + marginal per-track electrification + substation capital.
    if inputs.budget is not None:
        _wire_budget = (
            sum(cell_marginal[c] * model.s[c] for c in model.C) if _per_cell
            else sum(data.piece_marginal_cost[p] * model.x[p] for p in model.P)
        )
        model.budget = pyo.Constraint(
            expr=(
                sum(data.struct_cost[c] * model.s[c] for c in model.C)
                + _wire_budget
                + sum(data.sub_cost[j] * model.y[j] for j in model.J)
                <= inputs.budget
            )
        )

    if inputs.objective == "min_cost":
        # Minimize lifecycle cost = capital + discounted operating cost. The
        # horizon/discount enter only via the scalar PVAF multiplying the annual
        # per-piece operating cost, so the program stays linear.
        if inputs.economics is None:
            raise ValueError(
                "objective='min_cost' requires `economics` (an EconomicParameters)."
            )
        if inputs.piece_opex_fn is None:
            raise ValueError(
                "objective='min_cost' requires `piece_opex_fn` to supply per-piece "
                "annual operating cost."
            )
        pvaf = inputs.economics.pvaf()
        if _per_cell:
            _wire = sum(cell_marginal[c] * model.s[c] for c in model.C)
            _opex = sum(cell_opex[c] * model.s[c] for c in model.C)
        else:
            _wire = sum(data.piece_marginal_cost[p] * model.x[p] for p in model.P)
            _opex = sum(data.piece_annual_opex[p] * model.x[p] for p in model.P)
        model.obj = pyo.Objective(
            expr=(
                sum(data.struct_cost[c] * model.s[c] for c in model.C)
                + _wire
                + sum(data.sub_cost[j] * model.y[j] for j in model.J)
                + pvaf * _opex
            ),
            sense=pyo.minimize,
        )
    elif inputs.objective == "max_benefit":
        # Maximize total priority-weighted electrified track miles (each parallel
        # track contributes, so wiring both directions is rewarded once structure
        # is paid for). High-grade cells are up-weighted.
        _benefit = (
            sum(cell_benefit[c] * model.s[c] for c in model.C) if _per_cell
            else sum(data.piece_benefit[p] * model.x[p] for p in model.P)
        )
        model.obj = pyo.Objective(expr=_benefit, sense=pyo.maximize)
    else:
        raise ValueError(
            f"Unknown objective {inputs.objective!r}; expected 'max_benefit' or "
            "'min_cost'."
        )

    return model


def solve_model(
    model: pyo.ConcreteModel,
    *,
    time_limit_seconds: float | None = None,
    mip_gap: float | None = None,
    tee: bool = False,
    solver_threads: int | None = None,
    warmstart: bool = False,
) -> tuple[pyo.ConcreteModel, str, bool, float | None, float | None]:
    """Solve the model with HiGHS through Pyomo APPSI.

    Variable values are loaded only when the solver found a feasible incumbent.

    Args:
        model: Model from :func:`build_model`.
        time_limit_seconds: Solver wall-clock limit (s); ``None`` for no limit.
        mip_gap: Relative MIP gap tolerance; ``None`` uses the solver default.
        tee: Whether to stream solver output.
        solver_threads: Thread cap for HiGHS; ``None`` uses all cores.
        warmstart: Whether to pass pre-seeded variable values as a MIP start.

    Returns:
        The model, the termination condition, whether a solution was loaded, the
        best objective bound and the incumbent objective (both ``None`` when
        unavailable).
    """
    solver = Highs()
    if time_limit_seconds is not None:
        solver.config.time_limit = time_limit_seconds
    if mip_gap is not None:
        solver.config.mip_gap = mip_gap
    if solver_threads is not None:
        # Cap HiGHS parallelism; its default (0) uses all available cores.
        solver.highs_options["threads"] = int(solver_threads)
    # More heuristic effort (default 0.05) to find good incumbents faster on the
    # large piece-count MIP, where the LP bound is weak and closing the gap relies
    # on a strong incumbent.
    solver.highs_options["mip_heuristic_effort"] = 0.2
    solver.config.stream_solver = tee
    # Do not auto-load (and thus do not raise) when no incumbent exists.
    solver.config.load_solution = False
    if warmstart:
        solver.config.warmstart = True

    results = solver.solve(model)
    has_solution = results.best_feasible_objective is not None
    if has_solution:
        results.solution_loader.load_vars()
    return (
        model, str(results.termination_condition), has_solution,
        getattr(results, "best_objective_bound", None),
        results.best_feasible_objective,
    )


def _seed_warmstart(model: pyo.ConcreteModel, data: _ProblemData) -> None:
    """Seed a fully wired feasible MIP start.

    Builds every substation, energizes every coverable piece and holds a full pack.

    Args:
        model: Model from :func:`build_model`.
        data: Problem data giving each piece's coverage and cell.
    """
    for j in model.J:
        model.y[j].set_value(1, skip_validation=True)
    energized_cells = set()
    _has_x = hasattr(model, "x")
    for p in model.P:
        on = 1 if (data.piece_auto_covered[p] or data.piece_cover[p]) else 0
        if _has_x:
            model.x[p].set_value(on, skip_validation=True)
        if on:
            energized_cells.add(data.piece_cell[p])
    for c in model.C:
        model.s[c].set_value(1 if c in energized_cells else 0, skip_validation=True)
    if hasattr(model, "u"):
        # A bucket is on only when every cell in it is energized in the seed;
        # otherwise its cells are switched off so the seed satisfies s[c] == u[b].
        bucket_cells: dict[int, list[Hashable]] = {}
        for c, b in getattr(model, "_cell_bucket", {}).items():
            bucket_cells.setdefault(b, []).append(c)
        pieces_in_cell: dict[Hashable, list[Hashable]] = {}
        for p in model.P:
            pieces_in_cell.setdefault(data.piece_cell[p], []).append(p)
        for b in model.RunBucket:
            cells = bucket_cells.get(b, [])
            on = 1 if cells and all(c in energized_cells for c in cells) else 0
            model.u[b].set_value(on, skip_validation=True)
            if on:
                continue
            for c in cells:
                model.s[c].set_value(0, skip_validation=True)
                if _has_x:
                    for p in pieces_in_cell.get(c, []):
                        model.x[p].set_value(0, skip_validation=True)
    if hasattr(model, "soc"):
        pack = model.soc[next(iter(model.SocBucket))].ub
        for b in model.SocBucket:
            model.soc[b].set_value(pack, skip_validation=True)
            model.curtail[b].set_value(0.0, skip_validation=True)


def extract_result(
    inputs: EnergizationPlacementInputs,
    data: _ProblemData,
    model: pyo.ConcreteModel,
    status: str,
) -> EnergizationPlacementResult:
    """Translate the solved model into a result object.

    Args:
        inputs: Optimizer inputs (network, power limit, objective settings).
        data: Problem data used to build the model.
        model: Solved model with variable values loaded.
        status: Solver termination condition string.

    Returns:
        Result holding the energized network, per-link energized intervals, built
        substations, installed cells, costs, mileages and the objective value.
    """
    if inputs.electrify_whole_site:
        # Whole-site: energizing a site wires every piece at it (no per-piece x).
        _on_cells = {
            c for c in data.cell_ids if pyo.value(model.s[c]) > 0.5
        }
        energized_pieces: List[Hashable] = [
            p for p in data.piece_ids if data.piece_cell[p] in _on_cells
        ]
    else:
        energized_pieces = [
            p for p in data.piece_ids if pyo.value(model.x[p]) > 0.5
        ]
    built: List[Hashable] = [
        j for j in data.substation_ids if pyo.value(model.y[j]) > 0.5
    ]
    installed_cells: List[Hashable] = [
        c for c in data.cell_ids if pyo.value(model.s[c]) > 0.5
    ]

    total_cost = (
        sum(data.struct_cost[c] for c in installed_cells)
        + sum(data.piece_marginal_cost[p] for p in energized_pieces)
        + sum(data.sub_cost[j] for j in built)
    )
    energized_track_miles = sum(data.piece_miles[p] for p in energized_pieces)
    structure_route_miles = sum(data.cell_len_miles[c] for c in installed_cells)
    objective_value = float(pyo.value(model.obj))

    # Lifecycle economics (only meaningful in min-cost mode).
    annual_opex_cost: Optional[float] = None
    discounted_opex_cost: Optional[float] = None
    lifecycle_cost: Optional[float] = None
    if inputs.objective == "min_cost" and inputs.economics is not None:
        annual_opex_cost = sum(
            data.piece_annual_opex[p] for p in energized_pieces
        )
        discounted_opex_cost = inputs.economics.pvaf() * annual_opex_cost
        lifecycle_cost = total_cost + discounted_opex_cost

    # Collect energized sub-intervals per link and merge overlaps.
    raw_intervals: Dict[int, List[Tuple[float, float]]] = {}
    for p in energized_pieces:
        raw_intervals.setdefault(data.piece_link[p], []).append(
            data.piece_offsets[p]
        )
    link_intervals: Dict[int, List[Tuple[float, float]]] = {
        li: _merge_intervals(ivs) for li, ivs in raw_intervals.items()
    }

    # Apply energization sub-intervals to a copy of the network.
    network_dict = inputs.network.to_pydict()
    for link in network_dict[1][1:]:
        li = link["idx_curr"]
        if li not in link_intervals:
            continue
        link["cat_power_limits"] = [
            {
                "offset_start_meters": start,
                "offset_end_meters": end,
                "power_limit_watts": inputs.uniform_power_limit_watts,
                "district_id": None,
            }
            for start, end in link_intervals[li]
        ]
    network = alt.Network.from_pydict(network_dict)

    return EnergizationPlacementResult(
        network=network,
        energized_link_idxs=sorted(link_intervals.keys()),
        energized_link_intervals=link_intervals,
        built_substation_ids=built,
        installed_structure_cell_ids=installed_cells,
        total_cost=total_cost,
        energized_track_miles=energized_track_miles,
        structure_route_miles=structure_route_miles,
        objective_value=objective_value,
        annual_opex_cost=annual_opex_cost,
        discounted_opex_cost=discounted_opex_cost,
        lifecycle_cost=lifecycle_cost,
        status=status,
    )


def optimize_energization_placement(
    inputs: EnergizationPlacementInputs,
    *,
    time_limit_seconds: float | None = None,
    mip_gap: float | None = None,
    tee: bool = False,
    solver_threads: int | None = None,
    warmstart: bool = False,
) -> EnergizationPlacementResult:
    """Prepare, build, solve and extract in one call.

    Args:
        inputs: Optimizer inputs.
        time_limit_seconds: Solver wall-clock limit (s); ``None`` for no limit.
        mip_gap: Relative MIP gap tolerance; ``None`` uses the solver default.
        tee: Whether to stream solver output.
        solver_threads: Thread cap for HiGHS; ``None`` uses all cores.
        warmstart: Whether to seed a fully wired MIP start.

    Returns:
        Full result with optimality-gap fields, or one carrying only ``status``
        and ``objective_bound`` when the solver found no feasible incumbent.
    """
    data = _prepare(inputs)
    model = build_model(inputs, data)
    if warmstart:
        _seed_warmstart(model, data)
    model, status, has_solution, bound, incumbent = solve_model(
        model,
        time_limit_seconds=time_limit_seconds,
        mip_gap=mip_gap,
        tee=tee,
        solver_threads=solver_threads,
        warmstart=warmstart,
    )
    if not has_solution:
        return EnergizationPlacementResult(status=status, objective_bound=bound)
    result = extract_result(inputs, data, model, status)
    result.objective_bound = bound
    obj = result.objective_value
    if bound is not None and obj is not None:
        result.optimality_gap_abs = abs(obj - bound)
        result.optimality_gap_rel = result.optimality_gap_abs / max(abs(obj), 1e-12)
    return result
