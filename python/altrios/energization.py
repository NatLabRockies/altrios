"""Utilities for placing energization (e.g. catenary or wireless power transfer) power limits on a network."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict

# local imports
import altrios as alt

if TYPE_CHECKING:
    from altrios.optimization.energization_placement import (
        EnergizationOptCosts,
        EnergizationPlacementResult,
        Substation,
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
    structure_spacing_miles: Optional[float] = None,
    corridor_width_miles: float = 0.05,
    min_energized_length_miles: Optional[float] = None,
    time_limit_seconds: Optional[float] = None,
    mip_gap: Optional[float] = None,
    tee: bool = False,
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
    min_energized_length_miles : float, optional
        Minimum contiguous energized run length in CORRIDOR miles (parallel
        tracks/sidings in a site count once). Prevents energizing tiny isolated
        slivers even when structure/substation costs are zero; the run may start
        at any position. Requires structure-site mode
        (``structure_spacing_miles``); ``None`` (default) disables it.
    time_limit_seconds : float, optional
        Solver wall-clock time limit (s). ``None`` (default) means no limit.
    mip_gap : float, optional
        Relative MIP optimality gap tolerance. ``None`` (default) uses the solver
        default.
    tee : bool, default False
        If True, stream solver output.

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
        structure_spacing_miles=structure_spacing_miles,
        corridor_width_miles=corridor_width_miles,
        min_energized_length_miles=min_energized_length_miles,
    )
    return cp.optimize_energization_placement(
        inputs,
        time_limit_seconds=time_limit_seconds,
        mip_gap=mip_gap,
        tee=tee,
    )


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
