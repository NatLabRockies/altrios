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
from typing import Dict, Hashable, List, Optional, Tuple

import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers.highs import Highs
import polars as pl

# local imports
import altrios as alt

#: Miles per meter.
MILES_PER_METER = 0.000621371
#: Mean Earth radius in miles (for haversine distance).
_EARTH_RADIUS_MILES = 3958.7613


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
    #: Objective (total priority-weighted energized miles).
    objective_value: Optional[float] = None
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
    piece_is_mainline: Dict[Hashable, bool]
    piece_cover: Dict[Hashable, List[Hashable]]
    piece_auto_covered: Dict[Hashable, bool]
    substation_ids: List[Hashable]
    sub_cost: Dict[Hashable, float]
    network_dict: tuple


def _first_not_none(*values: Optional[float]) -> float:
    """Return the first non-``None`` value, or ``0.0`` if all are ``None``."""
    for value in values:
        if value is not None:
            return value
    return 0.0


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
) -> _ProblemData:
    """Per-link mode: each link is one cell containing one whole-link piece."""
    cat_per_mile = _first_not_none(
        inputs.costs.energization_cost_per_mile, inputs.costs.cost_per_mile
    )
    struct_per_mile = inputs.costs.structure_cost_per_mile
    feed = inputs.substation_feed_distance_miles

    if inputs.est_time_nets:
        candidate_idxs: List[int] = []
        seen = set()
        for etn in inputs.est_time_nets:
            for idx in _ordered_path_link_idxs(etn):
                if idx in link_by_idx and idx not in seen:
                    seen.add(idx)
                    candidate_idxs.append(idx)
    else:
        candidate_idxs = list(link_by_idx.keys())

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
    piece_is_mainline: Dict[Hashable, bool] = {}
    piece_cover: Dict[Hashable, List[Hashable]] = {}
    piece_auto_covered: Dict[Hashable, bool] = {}

    for idx in candidate_idxs:
        link = link_by_idx[idx]
        length_m = float(link["length_meters"])
        miles = length_m * MILES_PER_METER
        high = _link_max_grade(link) >= inputs.grade_threshold
        weight = inputs.high_grade_benefit_weight if high else 1.0

        cell_ids.append(idx)
        struct_cost[idx] = struct_per_mile * miles
        cell_len_miles[idx] = miles
        cell_high_grade[idx] = high

        piece_ids.append(idx)
        piece_cell[idx] = idx
        piece_link[idx] = idx
        piece_offsets[idx] = (0.0, length_m)
        piece_miles[idx] = miles
        piece_benefit[idx] = miles * weight
        piece_marginal_cost[idx] = cat_per_mile * miles
        piece_is_mainline[idx] = True

        cover, auto = _piece_coverage(
            _link_midpoint_latlon(link), existing_positions, sub_positions, feed
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
    feed = inputs.substation_feed_distance_miles

    # Candidate links: all links, optionally restricted to traversed links.
    if inputs.est_time_nets:
        candidate_idxs: List[int] = []
        seen = set()
        for etn in inputs.est_time_nets:
            for idx in _ordered_path_link_idxs(etn):
                if idx in link_by_idx and idx not in seen:
                    seen.add(idx)
                    candidate_idxs.append(idx)
    else:
        candidate_idxs = list(link_by_idx.keys())

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
    piece_cover: Dict[Hashable, List[Hashable]] = {}
    piece_auto_covered: Dict[Hashable, bool] = {}
    for i in range(n_pieces):
        c = piece_cell[i]
        miles = piece_miles[i]
        weight = inputs.high_grade_benefit_weight if cell_high_grade[c] else 1.0
        piece_benefit[i] = miles * weight
        piece_marginal_cost[i] = cat_per_mile * miles
        cover, auto = _piece_coverage(
            (p_lat[i], p_lon[i]), existing_positions, sub_positions, feed
        )
        piece_cover[i] = cover
        piece_auto_covered[i] = auto

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
        piece_is_mainline=piece_is_mainline,
        piece_cover=piece_cover,
        piece_auto_covered=piece_auto_covered,
        substation_ids=substation_ids,
        sub_cost=sub_cost,
        network_dict=network_dict,
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
    if inputs.min_energized_length_miles is not None:
        min_len = inputs.min_energized_length_miles

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

        # Order structure sites along each corridor via the piece chains.
        link_by_idx = {L["idx_curr"]: L for L in data.network_dict[1][1:]}
        link_pieces: Dict[int, List[Hashable]] = {}
        for p in data.piece_ids:
            link_pieces.setdefault(data.piece_link[p], []).append(p)
        for lp in link_pieces.values():
            lp.sort(key=lambda p: data.piece_offsets[p][0])
        site_seqs: List[List[Hashable]] = []
        for chain in _build_link_chains(link_by_idx, link_pieces):
            seq: List[Hashable] = []
            for p in chain:
                c = data.piece_cell[p]
                if not seq or seq[-1] != c:
                    seq.append(c)
            if len(seq) >= 2:
                site_seqs.append(seq)

        # (j, i, i_prev) site triples forcing s[j] >= s[i] - s[i_prev]. Dedup,
        # since parallel tracks yield the same corridor site-sequence.
        triples: set = set()
        for seq in site_seqs:
            lengths = [data.cell_len_miles[c] for c in seq]
            n_seq = len(seq)
            for i in range(1, n_seq):  # interior starts only (i=0 = corridor end)
                cum = 0.0
                reach = None
                for k in range(i, n_seq):
                    cum += lengths[k]
                    if cum >= min_len - 1e-9:
                        reach = k
                        break
                if reach is None:
                    continue  # remaining corridor too short -> boundary-exempt
                for k in range(i + 1, reach + 1):
                    triples.add((seq[k], seq[i], seq[i - 1]))

        min_run_triples = list(triples)
        if min_run_triples:
            model.min_run_index = pyo.Set(
                initialize=range(len(min_run_triples)), ordered=True
            )

            def _min_run_rule(m, t):
                j, i, i_prev = min_run_triples[t]
                return m.s[j] >= m.s[i] - m.s[i_prev]

            model.min_run = pyo.Constraint(model.min_run_index, rule=_min_run_rule)

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

    # Maximize total priority-weighted electrified track miles (each parallel
    # track contributes, so wiring both directions is rewarded once structure
    # is paid for). High-grade cells are up-weighted.
    model.obj = pyo.Objective(
        expr=sum(data.piece_benefit[p] * model.x[p] for p in model.P),
        sense=pyo.maximize,
    )

    return model


def solve_model(
    model: pyo.ConcreteModel,
    *,
    time_limit_seconds: Optional[float] = None,
    mip_gap: Optional[float] = None,
    tee: bool = False,
) -> Tuple[pyo.ConcreteModel, str]:
    """Solve the model with the HiGHS solver via Pyomo APPSI.

    Returns the solved model and a termination-condition string.
    """
    solver = Highs()
    if time_limit_seconds is not None:
        solver.config.time_limit = time_limit_seconds
    if mip_gap is not None:
        solver.config.mip_gap = mip_gap
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
        status=status,
    )


def optimize_energization_placement(
    inputs: EnergizationPlacementInputs,
    *,
    time_limit_seconds: Optional[float] = None,
    mip_gap: Optional[float] = None,
    tee: bool = False,
) -> EnergizationPlacementResult:
    """End-to-end entry point: prepare data, build, solve, and extract."""
    data = _prepare(inputs)
    model = build_model(inputs, data)
    model, status = solve_model(
        model,
        time_limit_seconds=time_limit_seconds,
        mip_gap=mip_gap,
        tee=tee,
    )
    return extract_result(inputs, data, model, status)
