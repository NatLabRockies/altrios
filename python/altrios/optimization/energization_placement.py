"""Mixed-integer linear program (MILP) for optimal energization placement.

The optimizer decides, subject to a capital budget, which track segments to
energize (e.g. with catenary or wireless power transfer) and which transmission
substations to build. It prioritizes high-grade regions (where electrification
yields the largest operational benefit) while respecting substation-feeding
(spacing) constraints.

Decision variables
------------------
- ``x[p] in {0, 1}``  -- electrify track piece ``p`` (a link sub-interval).
- ``s[c] in {0, 1}``  -- install shared energization structure in corridor cell ``c``.
- ``y[j] in {0, 1}``  -- build candidate substation ``j`` (lump-sum capital).

Structure sites and parallel tracks
-----------------------------------
When ``structure_spacing_miles`` is set, every link is densified into uniform
along-track **pieces**, and pieces whose midpoints fall within
``corridor_width_miles`` of one another are clustered into shared **structure
sites** ``s[c]`` (masts, foundations). Both directions of a double-track line and
any sidings at the same location fall into one site and share a single structure
cost; each individual track then pays only a marginal per-mile electrification
cost via ``x[p]``. This makes it economical to wire every parallel track once the
structure is built. Clustering is purely geometric (lat/lon), needing no
centerline, topology, or est-time nets, so it is safe on arbitrary imported
networks. When ``structure_spacing_miles`` is ``None``, each link is treated as
its own single-piece cell (no parallel grouping).

Formulation / best-practice notes
----------------------------------
- Feeding uses the STRONG disaggregated covering constraint
  ``x[p] <= sum_{j in cover(p)} y[j]``; structure sharing likewise uses the
  strong per-piece linking ``x[p] <= s[c]`` rather than an aggregated Big-M row.
  Both yield a tight LP relaxation (near totally unimodular). The only row that
  breaks total unimodularity is the single budget knapsack constraint, so the
  MILP solves quickly.
- Costs are linear in the decisions, so the whole program is a MILP.

Data-schema assumption
----------------------
Substation candidates and track segments are assumed to expose geographic
positions. A ``Substation`` may be located either directly by ``(lat, lon)`` or
by ``link_idx`` (in which case the location is taken from that link's geometry).
Segment positions are derived from each link's ``headings`` (lat/lon) geometry.

The optimizer targets ``pyomo`` for modeling and the HiGHS solver (``highspy``)
via Pyomo's APPSI interface.
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
        """Present-value annuity factor ``sum_{t=1}^{N} (1+r)^{-t}``.

        Equals ``N`` when the discount rate is zero.
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
    #: If set, enable structure-site (corridor) mode: densify every link into
    #: uniform along-track pieces of this length (miles) and cluster parallel
    #: tracks that share a location so they share one structure cost. Requires
    #: lat/lon geometry but no centerline/topology (safe on arbitrary networks).
    #: If ``None`` (default), each link is treated as its own single-piece cell.
    structure_spacing_miles: Optional[float] = None
    #: Lateral distance (miles) within which pieces from parallel tracks are
    #: clustered into the same shared structure site (structure-site mode only).
    corridor_width_miles: float = 0.05
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


def _first_not_none(*values: Optional[float]) -> float:
    """Return the first non-``None`` value, or ``0.0`` if all are ``None``."""
    for value in values:
        if value is not None:
            return value
    return 0.0


def _compute_piece_benefit(
    inputs: EnergizationPlacementInputs,
    *,
    link_idx: int,
    off0: float,
    off1: float,
    miles: float,
    high_grade: bool,
) -> float:
    """Objective coefficient ``b_p`` for one piece.

    Delegates to ``inputs.piece_benefit_fn`` when supplied (the seam for custom,
    non-grade objectives such as diesel gallons displaced); otherwise applies the
    default grade heuristic ``miles * (high_grade_benefit_weight if high_grade
    else 1)``.
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
    """Annual net operating cost delta ($/yr) for electrifying one piece.

    Delegates to ``inputs.piece_opex_fn`` (electricity purchased minus displaced
    diesel; negative = saving). Returns ``0.0`` when no opex model is supplied.
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
    """Great-circle distance in miles between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return _EARTH_RADIUS_MILES * 2.0 * math.asin(math.sqrt(a))


def _link_midpoint_latlon(link: dict) -> Optional[Tuple[float, float]]:
    """Representative (lat, lon) for a link, taken from its geometry."""
    headings = link.get("headings") or []
    if not headings:
        return None
    mid = headings[len(headings) // 2]
    lat, lon = mid.get("lat"), mid.get("lon")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def _link_max_grade(link: dict) -> float:
    """Maximum grade along a link (0.0 if geometry is insufficient)."""
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


def _ordered_path_link_idxs(est_time_net: alt.EstTimeNet) -> List[int]:
    """Ordered, de-duplicated link indices traversed by an est-time net."""
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
    inputs: EnergizationPlacementInputs, link_by_idx: Dict[int, dict]
) -> List[int]:
    """Link indices eligible for electrification.

    With ``est_time_nets``, restrict candidates to links the trains actually
    traverse, and include the opposite-direction link (``idx_flip``) of each so a
    physical track carrying traffic *in either direction* is eligible (a train
    running one way generates traffic on one directional link; its flip is the
    same rail). Links no train ever uses in either direction are dropped, so the
    optimizer cannot electrify trafficless spurs/branches purely for their grade.
    Without ``est_time_nets`` every link is a candidate (unchanged behavior).
    """
    if not inputs.est_time_nets:
        return list(link_by_idx.keys())
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
    return ordered


def _merge_intervals(
    intervals: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Merge overlapping/adjacent (start, end) intervals."""
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
    inputs: EnergizationPlacementInputs, link_by_idx: Dict[int, dict]
) -> Tuple[
    List[Hashable],
    Dict[Hashable, float],
    Dict[Hashable, Tuple[float, float]],
    List[Tuple[float, float]],
]:
    """Resolve substation positions/costs, splitting existing vs candidate."""
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
    pos: Optional[Tuple[float, float]],
    existing_positions: List[Tuple[float, float]],
    sub_positions: Dict[Hashable, Tuple[float, float]],
    feed_dist: float,
) -> Tuple[List[Hashable], bool]:
    """Return (candidate covering substation ids, auto_covered by existing)."""
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
        existing_positions: List[Tuple[float, float]],
        sub_positions: Dict[Hashable, Tuple[float, float]],
        feed_dist_miles: float,
        sample_spacing_meters: float = 200.0,
    ) -> None:
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

    def ordered_mainline_links(self) -> List[int]:
        """Mainline link indices in corridor order (by cumulative arc length).

        This is the exact, contiguous corridor ordering of the mainline path --
        unlike per-cell mean arc positions, it never leaves spurious gaps -- so it
        gives the minimum-run-length constraint a single unbroken site sequence.
        """
        return sorted(self._arc_start, key=lambda k: self._arc_start[k])

    def arc(self, lat: Optional[float], lon: Optional[float]) -> Optional[float]:
        """Along-corridor arc position (meters) of ``(lat, lon)``, or ``None``."""
        if self._project is None or lat is None or lon is None:
            return None
        return self._project(lat, lon)

    def coverage(
        self, lat: Optional[float], lon: Optional[float]
    ) -> Tuple[List[Hashable], bool]:
        """Return ``(covering candidate ids, auto_covered_by_existing)``."""
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
    network,
    substations,
    built_substation_ids=None,
    *,
    substation_feed_distance_miles: float = 20.0,
    sample_spacing_meters: float = 200.0,
) -> Dict[int, bool]:
    """Per-link feeding coverage (bool) by along-corridor TRACK distance.

    Returns ``{link_idx: covered}`` where ``covered`` is ``True`` iff the link's
    midpoint is within ``substation_feed_distance_miles`` **track miles** (corridor
    arc distance -- the same metric the optimizer and ``apply_line_losses`` use) of
    any EXISTING substation or any substation whose id is in
    ``built_substation_ids``. This mirrors the optimizer's feeding rule, so a map's
    "within range" shading matches what actually governed the optimization.
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
    """Extract problem data; dispatch to per-link or corridor-cell mode."""
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
    link_by_idx: Dict[int, dict],
    substation_ids: List[Hashable],
    sub_cost: Dict[Hashable, float],
    sub_positions: Dict[Hashable, Tuple[float, float]],
    existing_positions: List[Tuple[float, float]],
    arc_cov: "_ArcCoverage",
) -> _ProblemData:
    """Per-link mode: each link is one cell containing one whole-link piece."""
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
    )


def _network_vertex_df(
    network_dict: tuple, candidate_idxs: Optional[List[int]] = None
) -> pl.DataFrame:
    """Explode link geometry into a per-vertex Polars dataframe.

    Provides the ``Network -> pydict -> Polars`` bridge for the geometry (GIS)
    steps of structure-site mode. Returns one row per heading vertex with columns
    ``idx_curr``, ``length_meters``, ``offset_meters``, ``lat``, ``lon`` (rows
    lacking lat/lon are dropped), sorted by link then offset.

    Parameters
    ----------
    network_dict : tuple
        ``network.to_pydict()`` output (``[config, [links]]``); index 0 of the
        links list is the dummy link and is skipped.
    candidate_idxs : list of int, optional
        If given, keep only these link indices.

    Returns
    -------
    pl.DataFrame
        Per-vertex geometry table.
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
    candidate_idxs: List[int],
    spacing_m: float,
) -> Tuple[
    List[int], List[float], List[float], List[float], List[float], List[float]
]:
    """Split each candidate link into uniform along-track pieces.

    This is the shared "reduced virtual links" representation used by both the
    optimizer and the even-spacing placer: each link is cut into pieces of about
    ``spacing_m`` meters, with lat/lon interpolated at each piece midpoint.

    Parameters
    ----------
    network_dict : tuple
        ``network.to_pydict()`` output.
    candidate_idxs : list of int
        Link indices to densify.
    spacing_m : float
        Target along-track piece length (meters).

    Returns
    -------
    tuple of lists
        ``(link_idx, off_start_m, off_end_m, mid_lat, mid_lon, length_m)`` with
        one entry per piece.
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
    link_by_idx: Dict[int, dict],
    substation_ids: List[Hashable],
    sub_cost: Dict[Hashable, float],
    sub_positions: Dict[Hashable, Tuple[float, float]],
    existing_positions: List[Tuple[float, float]],
    arc_cov: "_ArcCoverage",
) -> _ProblemData:
    """Structure-site mode: geometry-only shared-structure clustering.

    Densifies every candidate link into uniform along-track pieces, then clusters
    piece midpoints that fall within ``corridor_width_miles`` of each other into
    shared **structure sites**. Parallel tracks (both directions, sidings) at the
    same location cluster into one site and share a single structure cost; each
    track then pays only a marginal per-mile electrification cost. Requires lat/lon
    geometry but NO centerline, topology, or est-time nets, so it is safe on
    arbitrary imported networks (e.g. OpenRailwayMap).
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
    )


def _build_link_chains(
    link_by_idx: Dict[int, dict],
    link_pieces: Dict[int, List[Hashable]],
) -> List[List[Hashable]]:
    """Concatenate per-link ordered pieces into maximal linear physical chains.

    Links are chained forward via ``idx_next`` only where the connection is
    unambiguous (the successor is a candidate, the current link has no candidate
    ``idx_next_alt``, and the successor has exactly one candidate predecessor), so
    chains are simple sequences that break at junctions. Each returned chain is a
    list of piece ids in physical (increasing-offset, forward-travel) order. This
    lets the minimum-run-length constraint span link boundaries so a short link in
    the middle of a corridor is treated as interior track, not an exempt boundary.
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
    model: pyo.ConcreteModel, data: "_ProblemData", bucket_mi: float
) -> Tuple[int, List[float], Dict[Hashable, int]]:
    """Install the shared corridor bucket layer for the run-length / gap constraints.

    First ties structure to actual electrification (``s[c] <= sum_{p in c} x[p]``)
    so each ``s[c]`` -- and hence each bucket ``u[b]`` -- is a faithful "corridor
    site energized" indicator even when structure costs are zero. Then, when a
    contiguous mainline corridor exists, tiles it into uniform ~``bucket_mi``
    buckets (accumulating cell length so sequential bucket ids 0,1,2,... leave no
    gaps) and couples ``s[c] == u[bucket(c)]``. This coarse granularity keeps the
    MILP small (one binary per bucket instead of one per ~250 ft link) while
    preserving the run/gap guarantees.

    Both the minimum-run-length and maximum-gap constraints share this single
    bucket layer; when both are active the caller passes the finer ``bucket_mi``
    so each is enforced at adequate resolution.

    Returns ``(n_buckets, bucket_len_miles, cell_bucket)``: the bucket count, each
    bucket's corridor length (miles), and the cell -> bucket map. Returns
    ``(0, [], {})`` when no mainline corridor exists (only ``structure_used`` is
    installed); callers needing buckets must then fall back or raise.
    """
    pieces_in_cell: Dict[Hashable, List[Hashable]] = {}
    for p in data.piece_ids:
        pieces_in_cell.setdefault(data.piece_cell[p], []).append(p)

    # Structure only where at least one track is energized (ties s to x).
    def _structure_used_rule(m, c):
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

    def _couple_rule(m, t):
        c, bkt = couple[t]
        return m.s[c] == m.u[bkt]

    model.run_couple = pyo.Constraint(model.run_couple_index, rule=_couple_rule)
    return n_buckets, bucket_len_miles, cell_bucket


def build_model(
    inputs: EnergizationPlacementInputs, data: _ProblemData
) -> pyo.ConcreteModel:
    """Construct the energization-placement MILP."""
    model = pyo.ConcreteModel(name="energization_placement")

    model.C = pyo.Set(initialize=data.cell_ids, ordered=True)
    model.P = pyo.Set(initialize=data.piece_ids, ordered=True)
    model.J = pyo.Set(initialize=data.substation_ids, ordered=True)

    # s[c]: install shared corridor structure; x[p]: electrify a track piece;
    # y[j]: build a substation. All binary.
    model.s = pyo.Var(model.C, domain=pyo.Binary)
    model.x = pyo.Var(model.P, domain=pyo.Binary)
    model.y = pyo.Var(model.J, domain=pyo.Binary)

    # Strong per-piece structure linking: a track piece can only be electrified
    # if its corridor cell's shared structure is installed.
    def _structure_rule(m, p):
        return m.x[p] <= m.s[data.piece_cell[p]]

    model.structure = pyo.Constraint(model.P, rule=_structure_rule)

    # Strong disaggregated feeding/coverage: a piece may be energized only if an
    # in-range substation is built. Pieces fed by a pre-existing substation are
    # unconstrained; pieces with no in-range candidate (and no existing feed)
    # are forced off.
    # Feeding/coverage: a piece may be energized only if an in-range substation
    # is built. Pieces fed by a pre-existing substation are auto-covered
    # (unconstrained); pieces with no in-range candidate (and no existing feed)
    # are forced off.
    def _coverage_rule(m, p):
        if data.piece_auto_covered[p]:
            return pyo.Constraint.Skip
        covering = data.piece_cover[p]
        if not covering:
            return m.x[p] == 0
        return m.x[p] <= sum(m.y[j] for j in covering)

    model.coverage = pyo.Constraint(model.P, rule=_coverage_rule)

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
    )
    if _need_buckets:
        _widths = []
        if inputs.min_energized_length_miles is not None:
            _widths.append(min(0.5, max(inputs.min_energized_length_miles / 8.0, 1e-6)))
        if inputs.max_energized_gap_miles is not None:
            _widths.append(min(0.5, max(inputs.max_energized_gap_miles / 8.0, 1e-6)))
        bucket_mi = min(_widths)
        n_buckets, bucket_len_miles, cell_bucket = _install_corridor_buckets(
            model, data, bucket_mi
        )
    else:
        n_buckets, bucket_len_miles, cell_bucket = 0, [], {}

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

                def _min_run_rule(m, t):
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

                def _min_run_rule(m, t):
                    j, i, i_prev = tl[t]
                    return m.s[j] >= m.s[i] - m.s[i_prev]

                model.min_run = pyo.Constraint(
                    model.min_run_index, rule=_min_run_rule
                )

    # Maximum contiguous UNelectrified gap length (battery-electric range limit).
    # No stretch of un-energized corridor may exceed max_gap: in every window of
    # consecutive buckets whose cumulative length EXCEEDS max_gap, at least one
    # bucket must be energized (sum_{b in window} u[b] >= 1). The minimal such
    # window per start index suffices (longer windows are implied). Windows that
    # touch a corridor end are NOT exempt: the battery is assumed full at each end
    # (yard charging), so end gaps obey the same limit. Because this can FORCE
    # electrification (unlike the minimum run), a feasibility pre-check first
    # verifies each binding window contains at least one coverable bucket.
    if inputs.max_energized_gap_miles is not None:
        gap_max = inputs.max_energized_gap_miles
        if n_buckets < 2:
            raise ValueError(
                "max_energized_gap_miles requires structure-site (corridor) mode "
                "with a contiguous mainline path, but none was found. Set "
                "structure_spacing_miles and ensure the network has lat/lon "
                "geometry."
            )

        # Prefix miles at each bucket boundary, for region reporting.
        prefix = [0.0]
        for L in bucket_len_miles:
            prefix.append(prefix[-1] + L)

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

        # Minimal window per start i: smallest [i..j] with sum(len) > gap_max.
        windows: List[Tuple[int, ...]] = []
        for i in range(n_buckets):
            cum = 0.0
            j = i
            while j < n_buckets and cum + bucket_len_miles[j] <= gap_max:
                cum += bucket_len_miles[j]
                j += 1
            if j < n_buckets:  # else: tail from i to corridor end <= gap_max -> ok
                windows.append(tuple(range(i, j + 1)))

        # Feasibility pre-check: a binding window with no coverable bucket can
        # never satisfy sum u >= 1, so the gap is un-closeable -> infeasible.
        for w in windows:
            if not any(bucket_coverable[b] for b in w):
                raise ValueError(
                    f"max_energized_gap_miles={gap_max} is infeasible: the corridor "
                    f"from mile {prefix[w[0]]:.1f} to {prefix[w[-1] + 1]:.1f} spans "
                    f"more than the maximum gap yet contains no track reachable by "
                    f"any candidate or existing substation, so the gap cannot be "
                    f"closed. Add a substation in range or increase "
                    f"substation_feed_distance_miles."
                )

        wl = list(dict.fromkeys(windows))  # dedup, preserve order
        if wl:
            model.max_gap_index = pyo.Set(
                initialize=range(len(wl)), ordered=True
            )

            def _max_gap_rule(m, t):
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

        def _require_rule(m, p):
            if p in req_set:
                return m.x[p] == 1
            return pyo.Constraint.Skip

        model.require_high_grade = pyo.Constraint(model.P, rule=_require_rule)

    # Single budget knapsack row (the only non-TU constraint): shared structure
    # + marginal per-track electrification + substation capital.
    if inputs.budget is not None:
        model.budget = pyo.Constraint(
            expr=(
                sum(data.struct_cost[c] * model.s[c] for c in model.C)
                + sum(data.piece_marginal_cost[p] * model.x[p] for p in model.P)
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
        model.obj = pyo.Objective(
            expr=(
                sum(data.struct_cost[c] * model.s[c] for c in model.C)
                + sum(data.piece_marginal_cost[p] * model.x[p] for p in model.P)
                + sum(data.sub_cost[j] * model.y[j] for j in model.J)
                + pvaf
                * sum(data.piece_annual_opex[p] * model.x[p] for p in model.P)
            ),
            sense=pyo.minimize,
        )
    elif inputs.objective == "max_benefit":
        # Maximize total priority-weighted electrified track miles (each parallel
        # track contributes, so wiring both directions is rewarded once structure
        # is paid for). High-grade cells are up-weighted.
        model.obj = pyo.Objective(
            expr=sum(data.piece_benefit[p] * model.x[p] for p in model.P),
            sense=pyo.maximize,
        )
    else:
        raise ValueError(
            f"Unknown objective {inputs.objective!r}; expected 'max_benefit' or "
            "'min_cost'."
        )

    return model


def solve_model(
    model: pyo.ConcreteModel,
    *,
    time_limit_seconds: Optional[float] = None,
    mip_gap: Optional[float] = None,
    tee: bool = False,
    solver_threads: Optional[int] = None,
) -> Tuple[pyo.ConcreteModel, str]:
    """Solve the model with the HiGHS solver via Pyomo APPSI.

    Returns the solved model and a termination-condition string.
    """
    solver = Highs()
    if time_limit_seconds is not None:
        solver.config.time_limit = time_limit_seconds
    if mip_gap is not None:
        solver.config.mip_gap = mip_gap
    if solver_threads is not None:
        # Cap HiGHS parallelism; its default (0) uses all available cores.
        solver.highs_options["threads"] = int(solver_threads)
    solver.config.stream_solver = tee

    results = solver.solve(model)
    return model, str(results.termination_condition)


def extract_result(
    inputs: EnergizationPlacementInputs,
    data: _ProblemData,
    model: pyo.ConcreteModel,
    status: str,
) -> EnergizationPlacementResult:
    """Translate the solved model into a :class:`EnergizationPlacementResult`."""
    energized_pieces: List[Hashable] = [
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
        objective_value=objective_value,
        annual_opex_cost=annual_opex_cost,
        discounted_opex_cost=discounted_opex_cost,
        lifecycle_cost=lifecycle_cost,
        status=status,
    )


def optimize_energization_placement(
    inputs: EnergizationPlacementInputs,
    *,
    time_limit_seconds: Optional[float] = None,
    mip_gap: Optional[float] = None,
    tee: bool = False,
    solver_threads: Optional[int] = None,
) -> EnergizationPlacementResult:
    """End-to-end entry point: prepare data, build, solve, and extract."""
    data = _prepare(inputs)
    model = build_model(inputs, data)
    model, status = solve_model(
        model,
        time_limit_seconds=time_limit_seconds,
        mip_gap=mip_gap,
        tee=tee,
        solver_threads=solver_threads,
    )
    return extract_result(inputs, data, model, status)
