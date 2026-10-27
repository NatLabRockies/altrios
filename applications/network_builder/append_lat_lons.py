"""
Append per-point lat/lon to an existing ALTRIOS network, changing nothing else.

Networks built before ``include_lat_lons`` was enabled (or serialized by a
version that dropped the fields) have ``Heading.lat`` / ``Heading.lon`` unset.
Rebuilding from OSM would restore them, but also discards everything done to
the network since -- elevation smoothing, grade-spike filtering, speed sets.
This module instead recovers the coordinates for a network as it stands and
writes them into the existing ``headings`` entries, leaving every other field
byte-identical.

The recovery is an exact inversion of ``build_altrios_with_sidings``, not a
fit. That builder emits one link per OSM *edge* (a consecutive node pair u->v
within a way), with::

    length_meters   = max(haversine_distance(u, v), 1.0)
    heading_radians = radians(get_bearing(u, v))          # same at both points
    headings        = [pt(0.0, heading, u), pt(length, heading, v)]

so re-fetching each way's node geometry and recomputing those two quantities
for every candidate node pair identifies which pair a link came from. Both are
continuous floats, so the match is effectively unique within a way, and the
residuals are reported to prove it.

Links that carry no OSM way id -- synthetic sidings, the endpoint buffers --
are then dead-reckoned from whichever neighbours did match, using the bearing
and length already stored on the link. That is exact too: the builder derived
those headings from real node coordinates in the first place.

Usage:
    python append_lat_lons.py --network path/to/network.msgpack [--out path.msgpack]
    python append_lat_lons.py --network ... --geom-cache ways.json --offline
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import altrios as alt

# Reuse the builder's own distance/bearing helpers rather than restating them:
# any divergence here would silently corrupt the match.
from build_altrios_with_sidings import R_EARTH, get_bearing, haversine_distance

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
Coord = Tuple[float, float]  # (lat, lon)

# Match tolerances. Lengths and bearings round-trip through YAML at full
# precision, so a link built from unmodified OSM data matches to ~1e-9; the
# defaults are loose enough to absorb float noise and tight enough that a way
# edited upstream since the build fails to match instead of matching wrongly.
LENGTH_TOL_M = 1e-3
HEADING_TOL_RAD = 1e-6

# Two edges are a genuine tie only when they fit *equally* well. Testing
# "both inside the tolerance" instead would flag every second edge of a long
# straight stretch, where neighbouring edges differ by far less than the
# tolerance but the correct one still fits exactly.
TIE_EPS = 1e-9

# A link's end node is its successor's start node, so recovered coordinates
# must agree there. Half a metre is far above float noise and far below the
# shortest real link, making this a decisive check rather than a tunable one.
CONTINUITY_TOL_M = 0.5


@dataclass
class MatchReport:
    """Outcome of a lat/lon recovery pass, for logging and assertion."""

    # "osm" recovers coordinates by matching links to OSM way edges;
    # "profile" recovers them by arc length along a route profile. Only the
    # wording of the report differs.
    mode: str = "osm"
    total_links: int = 0
    skipped_no_headings: int = 0
    matched_osm: int = 0
    matched_dead_reckoned: int = 0
    unmatched: List[int] = field(default_factory=list)
    ambiguous: List[int] = field(default_factory=list)
    missing_ways: List[str] = field(default_factory=list)
    max_length_residual_m: float = 0.0
    max_heading_residual_rad: float = 0.0

    @property
    def resolved(self) -> int:
        return self.matched_osm + self.matched_dead_reckoned

    def log(self) -> None:
        primary, secondary = (
            ("matched via OSM way    ", "dead-reckoned          ")
            if self.mode == "osm"
            else ("placed along profile   ", "sidings interpolated   ")
        )
        logger.info("links total              : %d", self.total_links)
        logger.info("  sentinel (no headings) : %d", self.skipped_no_headings)
        logger.info("  %s : %d", primary, self.matched_osm)
        logger.info("  %s : %d", secondary, self.matched_dead_reckoned)
        logger.info("  UNRESOLVED             : %d", len(self.unmatched))
        if self.ambiguous:
            logger.warning("  ambiguous matches      : %d", len(self.ambiguous))
        if self.missing_ways:
            logger.warning("  ways not returned      : %d", len(self.missing_ways))
        logger.info(
            "max residual: length %.3e m, heading %.3e rad",
            self.max_length_residual_m,
            self.max_heading_residual_rad,
        )


# -- OSM way geometry --------------------------------------------------------


def way_ids_in(links: Sequence[dict]) -> List[str]:
    """The distinct numeric OSM way ids referenced by ``links``.

    Non-numeric ``osm_id`` values (``siding_synth``, ``start_buffer_fwd``, the
    ``0`` sentinel) are excluded -- those links have no OSM geometry and are
    dead-reckoned instead.
    """
    ids = {
        str(link["osm_id"])
        for link in links
        if str(link.get("osm_id")).isdigit() and str(link["osm_id"]) != "0"
    }
    return sorted(ids, key=int)


def fetch_way_geometry(
    way_ids: Sequence[str],
    *,
    url: str = OVERPASS_URL,
    batch_size: int = 120,
    retries: int = 4,
    pause_s: float = 2.0,
) -> Dict[str, List[Coord]]:
    """Fetch ``[(lat, lon), ...]`` node geometry per OSM way id from Overpass.

    Batched and retried -- the public endpoint returns 429/504 under load.
    """
    geoms: Dict[str, List[Coord]] = {}
    for start in range(0, len(way_ids), batch_size):
        batch = way_ids[start : start + batch_size]
        query = f"[out:json][timeout:180];way(id:{','.join(batch)});out geom;"
        payload = urllib.parse.urlencode({"data": query}).encode()
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"User-Agent": "altrios-applied/network-builder"},
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.load(response)
                break
            except Exception as exc:  # noqa: BLE001 - transient endpoint errors
                logger.warning(
                    "Overpass batch at %d attempt %d failed: %s", start, attempt, exc
                )
                time.sleep(8)
        else:
            raise RuntimeError(f"Overpass failed for batch starting at {start}")

        for element in body.get("elements", []):
            if element.get("type") == "way" and "geometry" in element:
                geoms[str(element["id"])] = [
                    (point["lat"], point["lon"]) for point in element["geometry"]
                ]
        logger.info("fetched %d/%d ways", len(geoms), len(way_ids))
        time.sleep(pause_s)
    return geoms


# -- Geometry helpers --------------------------------------------------------


def _edge_length_and_heading(u: Coord, v: Coord) -> Tuple[float, float]:
    """``(length_meters, heading_radians)`` for edge u->v, as the builder computes them."""
    length = max(haversine_distance(u[1], u[0], v[1], v[0]), 1.0)
    heading = math.radians(get_bearing(u[0], u[1], v[0], v[1]))
    return length, heading


def _angular_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in radians."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def rhumb_destination(origin: Coord, heading_rad: float, distance_m: float) -> Coord:
    """Point reached from ``origin`` on a constant bearing -- inverse of ``get_bearing``.

    ``get_bearing`` returns a rhumb-line (Mercator) bearing, so the matching
    direct formula is the rhumb one, not the great-circle one.
    """
    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    delta = distance_m / R_EARTH
    d_lat = delta * math.cos(heading_rad)
    lat2 = lat1 + d_lat

    d_psi = math.log(
        math.tan(lat2 / 2 + math.pi / 4) / math.tan(lat1 / 2 + math.pi / 4)
    )
    # q degenerates as the track approaches due east/west; fall back to the
    # parallel-of-latitude form there.
    q = d_lat / d_psi if abs(d_psi) > 1e-12 else math.cos(lat1)
    lon2 = lon1 + delta * math.sin(heading_rad) / q

    return (math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180)


def _candidate_edges(
    geometry: Sequence[Coord],
) -> List[Tuple[float, float, Coord, Coord]]:
    """Every traversable edge of a way as ``(length, heading, start, end)``.

    Both directions are emitted: the builder makes a forward and a reverse link
    for each OSM edge, and either may be the one being matched.
    """
    candidates: List[Tuple[float, float, Coord, Coord]] = []
    for u, v in zip(geometry, geometry[1:]):
        length, heading = _edge_length_and_heading(u, v)
        candidates.append((length, heading, u, v))
        candidates.append((length, (heading + math.pi) % (2 * math.pi), v, u))
    return candidates


# -- Core --------------------------------------------------------------------


def _match_against_way(
    link: dict,
    candidates: Sequence[Tuple[float, float, Coord, Coord]],
    *,
    length_tol: float,
    heading_tol: float,
) -> Tuple[Optional[Tuple[Coord, Coord]], float, float, bool]:
    """Best ``(start, end)`` for ``link`` among a way's edges.

    Returns ``(coords or None, length_residual, heading_residual, ambiguous)``.
    ``ambiguous`` is set when a second, geometrically distinct edge also falls
    inside both tolerances -- a genuine tie we should not silently resolve.
    """
    link_length = link["length_meters"]
    link_heading = link["headings"][0]["heading_radians"]

    scored = []
    for length, heading, start, end in candidates:
        d_length = abs(length - link_length)
        d_heading = _angular_delta(heading, link_heading)
        scored.append((d_heading, d_length, start, end))
    if not scored:
        return None, math.inf, math.inf, False

    scored.sort(key=lambda row: (row[0], row[1]))
    d_heading, d_length, start, end = scored[0]
    if d_length > length_tol or d_heading > heading_tol:
        return None, d_length, d_heading, False

    ambiguous = any(
        abs(other_dh - d_heading) <= TIE_EPS
        and abs(other_dl - d_length) <= TIE_EPS
        and (other_start, other_end) != (start, end)
        for other_dh, other_dl, other_start, other_end in scored[1:]
    )
    return (start, end), d_length, d_heading, ambiguous


def _assign(
    link: dict, start: Optional[Coord] = None, end: Optional[Coord] = None
) -> None:
    """Write coordinates into a link's first and last heading points.

    Either end may be omitted, leaving that point's optional lat/lon unset.
    """
    headings = link["headings"]
    if start is not None:
        headings[0]["lat"], headings[0]["lon"] = start
    if end is not None:
        headings[-1]["lat"], headings[-1]["lon"] = end


def is_synthetic_buffer(link: dict) -> bool:
    """True for the approach links the builder appends at each corridor end.

    They carry a named ``osm_id`` (``start_buffer_fwd`` and friends) and a
    placeholder bearing rather than a surveyed one, so their outer end cannot
    be projected from what the link itself records.
    """
    return str(link.get("osm_id", "")).endswith(("buffer_fwd", "buffer_rev"))


def _has_placeholder_heading(link: dict) -> bool:
    """True when a link's bearing is a filler value rather than a real one.

    Real bearings come out of ``atan2`` and are irrational; exactly 0 or
    exactly pi means nobody computed one. Projecting along such a heading
    invents a position -- for a 3 km approach link, kilometres off, at right
    angles to the track it is supposed to extend.
    """
    if not link.get("headings"):
        return False
    heading = link["headings"][0]["heading_radians"]
    return heading == 0.0 or abs(heading - math.pi) < 1e-9


def _dead_reckon(links: Sequence[dict], resolved: Dict[int, Tuple[Coord, Coord]]) -> int:
    """Fill in links with no OSM geometry by walking out from resolved neighbours.

    Propagates along the network's own topology -- a link's end node is its
    successor's start node, ``idx_flip`` traverses the same edge reversed --
    and closes the remaining endpoint from the link's stored bearing and length.
    """
    by_idx = {link["idx_curr"]: link for link in links}
    filled = 0
    # Links given the one endpoint a neighbour could supply, whose far end is
    # deliberately left unset. Tracked so the loop terminates and so they are
    # not reported as unresolved.
    partial: set = set()

    changed = True
    while changed:
        changed = False
        for link in links:
            idx = link["idx_curr"]
            if idx in resolved or idx in partial or not link.get("headings"):
                continue

            start: Optional[Coord] = None
            end: Optional[Coord] = None

            # A neighbour that shares a node hands us that node's position.
            flip = resolved.get(link["idx_flip"])
            if flip is not None:
                start, end = flip[1], flip[0]
            else:
                for prev_key in ("idx_prev", "idx_prev_alt"):
                    prev = resolved.get(link.get(prev_key, 0))
                    if prev is not None:
                        start = prev[1]
                        break
                for next_key in ("idx_next", "idx_next_alt"):
                    nxt = resolved.get(link.get(next_key, 0))
                    if nxt is not None:
                        end = nxt[0]
                        break

            if start is None and end is None:
                continue

            heading = link["headings"][0]["heading_radians"]
            length = link["length_meters"]
            if _has_placeholder_heading(link):
                # Take only what a neighbour supplied -- that endpoint is a
                # real shared node. The far end stays unset rather than being
                # projected along a bearing that was never measured.
                _assign(by_idx[idx], start, end)
                filled += 1
                changed = True
                partial.add(idx)
                continue

            if start is not None and end is None:
                end = rhumb_destination(start, heading, length)
            elif end is not None and start is None:
                start = rhumb_destination(
                    end, (heading + math.pi) % (2 * math.pi), length
                )

            resolved[idx] = (start, end)
            _assign(by_idx[idx], start, end)
            filled += 1
            changed = True

    return filled, partial


def append_lat_lons(
    links: List[dict],
    way_geoms: Dict[str, List[Coord]],
    *,
    length_tol: float = LENGTH_TOL_M,
    heading_tol: float = HEADING_TOL_RAD,
) -> MatchReport:
    """Populate ``Heading.lat`` / ``Heading.lon`` on ``links``, in place.

    Only those two fields are written. ``links`` is the link list from
    ``Network.to_pydict()[1]``; pass the result back through
    ``Network.from_pydict`` to serialize.
    """
    report = MatchReport(total_links=len(links))
    candidates_by_way = {
        way_id: _candidate_edges(geometry) for way_id, geometry in way_geoms.items()
    }
    resolved: Dict[int, Tuple[Coord, Coord]] = {}

    for link in links:
        if not link.get("headings"):
            report.skipped_no_headings += 1
            continue

        way_id = str(link.get("osm_id"))
        candidates = candidates_by_way.get(way_id)
        if candidates is None:
            if way_id.isdigit() and way_id != "0":
                report.missing_ways.append(way_id)
            continue

        coords, d_length, d_heading, ambiguous = _match_against_way(
            link, candidates, length_tol=length_tol, heading_tol=heading_tol
        )
        if coords is None:
            continue

        report.max_length_residual_m = max(report.max_length_residual_m, d_length)
        report.max_heading_residual_rad = max(
            report.max_heading_residual_rad, d_heading
        )
        if ambiguous:
            report.ambiguous.append(link["idx_curr"])

        _assign(link, *coords)
        resolved[link["idx_curr"]] = coords
        report.matched_osm += 1

    report.matched_dead_reckoned, partial = _dead_reckon(links, resolved)
    report.unmatched = [
        link["idx_curr"]
        for link in links
        if link.get("headings")
        and link["idx_curr"] not in resolved
        and link["idx_curr"] not in partial
    ]
    if partial:
        logger.info(
            "%d synthetic approach link(s) given only their shared endpoint: %s",
            len(partial),
            sorted(partial),
        )
    report.missing_ways = sorted(set(report.missing_ways))
    return report


# -- Profile-driven recovery -------------------------------------------------
#
# A network built by resampling a route profile (rather than from OSM directly)
# carries no way ids and no real bearings, so the OSM matcher above has nothing
# to work with. What it does carry is the profile's own arc length: the
# mainline is that profile chopped into fixed-length links. Re-reading the
# profile and mapping each link boundary back onto it by cumulative distance
# recovers the coordinates exactly, provided the two total lengths agree.


def load_profile(path: Path) -> List[Tuple[float, float, float]]:
    """Read a route profile CSV into ``[(cumulative_dist_m, lat, lon), ...]``.

    Expects ``cumulative_dist_m``, ``lat`` and ``lon`` columns, sorted by
    distance. Other columns (elevation, grade, curvature, subdivision) are
    ignored -- this only recovers position.
    """
    import csv

    points: List[Tuple[float, float, float]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points.append(
                (
                    float(row["cumulative_dist_m"]),
                    float(row["lat"]),
                    float(row["lon"]),
                )
            )
    points.sort(key=lambda point: point[0])
    return points


def _interpolate_profile(
    profile: Sequence[Tuple[float, float, float]], offset: float
) -> Coord:
    """Coordinate at ``offset`` metres along the profile, linearly interpolated."""
    import bisect

    distances = [point[0] for point in profile]
    i = bisect.bisect_left(distances, offset)
    if i <= 0:
        return (profile[0][1], profile[0][2])
    if i >= len(profile):
        return (profile[-1][1], profile[-1][2])

    d0, lat0, lon0 = profile[i - 1]
    d1, lat1, lon1 = profile[i]
    span = d1 - d0
    if span <= 0:
        return (lat0, lon0)
    t = (offset - d0) / span
    return (lat0 + (lat1 - lat0) * t, lon0 + (lon1 - lon0) * t)


def _mainline_chain(links: Sequence[dict], origin_link: int) -> List[int]:
    """Link indices along ``idx_next`` from ``origin_link``, stopping at the first repeat."""
    by_idx = {link["idx_curr"]: link for link in links}
    idx, seen, chain = origin_link, set(), []
    while idx and idx not in seen:
        seen.add(idx)
        chain.append(idx)
        idx = by_idx[idx]["idx_next"]
    return chain


def append_lat_lons_from_profile(
    links: List[dict],
    profile: Sequence[Tuple[float, float, float]],
    *,
    origin_link: int = 1,
    buffer_factor: float = 10.0,
) -> MatchReport:
    """Populate lat/lon from a route profile, by arc length along the mainline.

    The synthetic approach links at each end of the chain are left without
    coordinates on purpose. They carry placeholder bearings, so projecting
    along them would invent a position tens of kilometres off the real
    alignment -- worse than leaving the optional field unset. They are
    detected as chain-end links longer than ``buffer_factor`` times the median
    link length.

    Sidings do not appear in the profile at all; each is interpolated between
    the mainline junctions it attaches to. A passing siding runs alongside the
    mainline, so that places it within a track's width of the truth -- far
    inside any tolerance that matters downstream.
    """
    report = MatchReport(mode="profile", total_links=len(links))
    by_idx = {link["idx_curr"]: link for link in links}
    chain = _mainline_chain(links, origin_link)

    # Strip the synthetic end buffers so the remaining run aligns with the profile.
    lengths = sorted(by_idx[i]["length_meters"] for i in chain)
    median = lengths[len(lengths) // 2]
    interior = list(chain)
    buffers: List[int] = []
    while interior and by_idx[interior[0]]["length_meters"] > buffer_factor * median:
        buffers.append(interior.pop(0))
    while interior and by_idx[interior[-1]]["length_meters"] > buffer_factor * median:
        buffers.append(interior.pop())

    profile_total = profile[-1][0]
    interior_total = sum(by_idx[i]["length_meters"] for i in interior)
    logger.info(
        "mainline interior %.3f m vs profile %.3f m (difference %.3f m)",
        interior_total,
        profile_total,
        abs(interior_total - profile_total),
    )

    # Offsets of each interior link, then its two heading points.
    offset_of: Dict[int, Tuple[float, float]] = {}
    acc = 0.0
    for i in interior:
        length = by_idx[i]["length_meters"]
        offset_of[i] = (acc, acc + length)
        acc += length

    resolved: Dict[int, Tuple[Coord, Coord]] = {}
    for i in interior:
        start_off, end_off = offset_of[i]
        start = _interpolate_profile(profile, start_off)
        end = _interpolate_profile(profile, end_off)
        _assign(by_idx[i], start, end)
        resolved[i] = (start, end)
        report.matched_osm += 1

        flip_idx = by_idx[i]["idx_flip"]
        flip = by_idx.get(flip_idx)
        if flip is not None and flip_idx and flip.get("headings"):
            _assign(flip, end, start)
            resolved[flip_idx] = (end, start)
            report.matched_osm += 1

    report.matched_dead_reckoned = _interpolate_sidings(
        links, resolved, offset_of, profile
    )

    for link in links:
        if not link.get("headings"):
            report.skipped_no_headings += 1

    # A buffer's reverse link is just as synthetic as the buffer itself.
    excluded = set(buffers) | {
        by_idx[i]["idx_flip"] for i in buffers if by_idx[i]["idx_flip"]
    }
    report.unmatched = [
        link["idx_curr"]
        for link in links
        if link.get("headings")
        and link["idx_curr"] not in resolved
        and link["idx_curr"] not in excluded
    ]
    logger.info(
        "left %d synthetic buffer link(s) without coordinates: %s",
        len(excluded),
        sorted(excluded),
    )
    return report


def _interpolate_sidings(
    links: Sequence[dict],
    resolved: Dict[int, Tuple[Coord, Coord]],
    offset_of: Dict[int, Tuple[float, float]],
    profile: Sequence[Tuple[float, float, float]],
) -> int:
    """Place siding links by interpolating between the mainline junctions they join.

    Each siding is a chain hanging off the mainline at both ends. Walking it
    gives the along-siding distance of every boundary, which is rescaled onto
    the mainline arc length between its two attachment points.
    """
    by_idx = {link["idx_curr"]: link for link in links}
    filled = 0

    starts = [
        link
        for link in links
        if link["idx_curr"] not in resolved
        and link.get("headings")
        and by_idx.get(link["idx_prev"], {}).get("idx_curr") in offset_of
    ]

    for first in starts:
        # Walk the siding to its far attachment, accumulating distance.
        run, idx, seen = [], first["idx_curr"], set()
        while idx and idx not in seen and idx not in offset_of:
            seen.add(idx)
            run.append(idx)
            idx = by_idx[idx]["idx_next"]
        if not run or idx not in offset_of:
            continue  # dangling siding: leave it for the report

        entry_offset = offset_of[first["idx_prev"]][1]
        exit_offset = offset_of[idx][0]
        run_lengths = [by_idx[i]["length_meters"] for i in run]
        run_total = sum(run_lengths)
        if run_total <= 0:
            continue

        acc = 0.0
        for i, length in zip(run, run_lengths):
            if i in resolved:
                acc += length
                continue
            span = exit_offset - entry_offset
            start = _interpolate_profile(profile, entry_offset + span * acc / run_total)
            acc += length
            end = _interpolate_profile(profile, entry_offset + span * acc / run_total)
            _assign(by_idx[i], start, end)
            resolved[i] = (start, end)
            filled += 1

            flip_idx = by_idx[i]["idx_flip"]
            flip = by_idx.get(flip_idx)
            if flip is not None and flip_idx and flip.get("headings") and flip_idx not in resolved:
                _assign(flip, end, start)
                resolved[flip_idx] = (end, start)
                filled += 1

    return filled


# -- Endpoint buffers --------------------------------------------------------


def find_endpoint_buffers(
    links: Sequence[dict], origin_link: int = 1, length_factor: float = 10.0
) -> List[int]:
    """The synthetic approach links at each corridor end.

    The OSM builder names them (``start_buffer_fwd`` and friends), so that is
    checked first. A network resampled from a route profile carries no
    ``osm_id`` at all, and there the buffers are found structurally instead:
    the first and last links of the mainline chain, which are far longer than
    any real link, together with their flips.
    """
    named = [link["idx_curr"] for link in links if is_synthetic_buffer(link)]
    if named:
        return sorted(named)

    by_idx = {link["idx_curr"]: link for link in links}
    chain = _mainline_chain(links, origin_link)
    if not chain:
        return []
    lengths = sorted(by_idx[i]["length_meters"] for i in chain)
    median = lengths[len(lengths) // 2]

    found: List[int] = []
    for end in (chain[0], chain[-1]):
        if by_idx[end]["length_meters"] > length_factor * median:
            found.append(end)
            flip = by_idx[end]["idx_flip"]
            if flip and flip in by_idx:
                found.append(flip)
    return sorted(set(found))


def resize_endpoint_buffers(
    links: Sequence[dict],
    target_length_m: float,
    *,
    longest_train_m: Optional[float] = None,
    braking_margin_m: float = 300.0,
    origin_link: int = 1,
) -> List[int]:
    """Set the synthetic approach link at each corridor end to one length, in place.

    ALTRIOS starts a train with its tail at offset 0 of the origin link, so the
    buffer must exceed the longest train plus its braking distance or the train
    has nowhere to stand. Resizing one is local: the buffer is flat (both
    ``elevs`` share an elevation, so there is no grade to interpolate), its
    speed limit is a single constant span, and no topology or location offset
    depends on its length.

    Size it to the longest train, not as large as possible. Every train travels
    the surplus ``buffer - train_length`` on top of the real corridor, so an
    oversized buffer inflates every run: Winnipeg-Chicago's original 100 km
    buffer added 8.2% to a 1,220 km corridor.

    Growing is always safe. Shrinking is not -- it can leave the longest train
    with nowhere to stand -- so it requires ``longest_train_m`` as evidence
    that the new length still fits.

    Returns the link indices changed.
    """
    by_idx = {link["idx_curr"]: link for link in links}
    buffers = [by_idx[i] for i in find_endpoint_buffers(links, origin_link)]
    if not buffers:
        raise ValueError(
            "no endpoint buffer links found -- neither a named "
            "`*_buffer_fwd`/`*_buffer_rev` osm_id nor an over-long link at "
            "either end of the mainline chain"
        )
    shrinking = [l for l in buffers if target_length_m < l["length_meters"]]
    if shrinking:
        if longest_train_m is None:
            raise ValueError(
                "refusing to shrink a buffer without longest_train_m: pass the "
                "longest train the corridor must dispatch so the new length can "
                "be checked against it"
            )
        floor = longest_train_m + braking_margin_m
        if target_length_m < floor:
            raise ValueError(
                f"buffer of {target_length_m:.0f} m is too short for a "
                f"{longest_train_m:.0f} m train plus {braking_margin_m:.0f} m "
                f"braking margin (need >= {floor:.0f} m)"
            )

    changed: List[int] = []
    for link in buffers:
        old = link["length_meters"]
        if target_length_m == old:
            continue

        link["length_meters"] = target_length_m
        # Flat profile: move the far offset, leave both elevations alone.
        link["elevs"][-1]["offset_meters"] = target_length_m
        link["headings"][-1]["offset_meters"] = target_length_m
        speed_set = link.get("speed_set")
        if speed_set:
            for limit in speed_set["speed_limits"]:
                if limit["offset_end_meters"] == old:
                    limit["offset_end_meters"] = target_length_m
        changed.append(link["idx_curr"])
    return changed


# -- Verification ------------------------------------------------------------


def _without_lat_lons(links: Sequence[dict]) -> List[dict]:
    """A deep copy of ``links`` with every heading's lat/lon blanked."""
    stripped = json.loads(json.dumps(links))
    for link in stripped:
        for point in link.get("headings", []):
            point["lat"] = None
            point["lon"] = None
    return stripped


def assert_only_lat_lons_changed(
    original: Sequence[dict], updated: Sequence[dict], ignore_idxs: Sequence[int] = ()
) -> None:
    """Raise unless ``updated`` differs from ``original`` in lat/lon alone.

    ``ignore_idxs`` exempts links changed deliberately by another operation;
    those are checked by ``assert_buffer_resize_only`` instead.
    """
    skip = set(ignore_idxs)
    before = [l for l in _without_lat_lons(original) if l["idx_curr"] not in skip]
    after = [l for l in _without_lat_lons(updated) if l["idx_curr"] not in skip]
    if before != after:
        raise AssertionError(
            "network changed beyond lat/lon -- refusing to write the result"
        )


def assert_buffer_resize_only(
    original: Sequence[dict], updated: Sequence[dict], idxs: Sequence[int]
) -> None:
    """Raise unless the named links differ only by having been resized.

    Length, the far elevation/heading offsets and the speed-limit span may
    move; elevations, speed values and topology may not.
    """
    before = {l["idx_curr"]: l for l in _without_lat_lons(original)}
    after = {l["idx_curr"]: l for l in _without_lat_lons(updated)}
    for idx in idxs:
        old, new = before[idx], after[idx]
        length = new["length_meters"]
        allowed = json.loads(json.dumps(old))
        allowed["length_meters"] = length
        allowed["elevs"][-1]["offset_meters"] = length
        allowed["headings"][-1]["offset_meters"] = length
        if allowed.get("speed_set"):
            for limit in allowed["speed_set"]["speed_limits"]:
                if limit["offset_end_meters"] == old["length_meters"]:
                    limit["offset_end_meters"] = length
        if allowed != new:
            raise AssertionError(
                f"link {idx} changed beyond a buffer resize -- refusing to write"
            )


def check_continuity(
    links: Sequence[dict], *, tol_m: float = CONTINUITY_TOL_M
) -> List[Tuple[int, str, int, float]]:
    """Distances where a link's end node fails to coincide with its neighbour's start.

    This is the check that makes the whole recovery trustworthy. Each link is
    matched against OSM independently, with no reference to its neighbours, so
    agreement at every shared node across the network is strong evidence that
    the matches are right -- and a misassigned link shows up here as a jump.
    ``idx_flip`` is included: the reverse link must traverse the same edge
    backwards.

    Returns ``(link_idx, relation, other_idx, distance_m)`` for each break.
    """
    by_idx = {link["idx_curr"]: link for link in links}

    def node(link: dict, which: int) -> Optional[Coord]:
        if not link.get("headings"):
            return None
        point = link["headings"][which]
        if point["lat"] is None or point["lon"] is None:
            return None
        return (point["lat"], point["lon"])

    breaks: List[Tuple[int, str, int, float]] = []
    for link in links:
        for relation, which_self, which_other in (
            ("idx_next", -1, 0),
            ("idx_next_alt", -1, 0),
            ("idx_prev", 0, -1),
            ("idx_prev_alt", 0, -1),
            ("idx_flip", 0, -1),
        ):
            other_idx = link.get(relation, 0)
            other = by_idx.get(other_idx)
            if not other_idx or other is None:
                continue
            here, there = node(link, which_self), node(other, which_other)
            if here is None or there is None:
                continue
            distance = haversine_distance(here[1], here[0], there[1], there[0])
            if distance > tol_m:
                breaks.append((link["idx_curr"], relation, other_idx, distance))
    return breaks


# -- CLI ---------------------------------------------------------------------


def _finish(args, network_dict, links, original, report: MatchReport) -> int:
    """Report, verify, and write -- shared by both recovery paths."""
    report.log()

    extended: List[int] = []
    if getattr(args, "set_buffers", None):
        extended = resize_endpoint_buffers(
            links,
            args.set_buffers,
            longest_train_m=getattr(args, "longest_train", None),
        )
        logger.info(
            "resized %d endpoint buffer link(s) to %.0f m: %s",
            len(extended),
            args.set_buffers,
            sorted(extended),
        )

    assert_only_lat_lons_changed(original, links, ignore_idxs=extended)
    if extended:
        assert_buffer_resize_only(original, links, extended)
        logger.info("verified: buffers resized, nothing else on those links")
    logger.info("verified: nothing but lat/lon changed")

    breaks = check_continuity(links)
    if breaks:
        worst = max(breaks, key=lambda row: row[3])
        logger.error(
            "%d continuity breaks; worst %.2f m at link %d --%s--> %d",
            len(breaks),
            worst[3],
            worst[0],
            worst[1],
            worst[2],
        )
        return 1
    logger.info("verified: every shared node agrees within %.2f m", CONTINUITY_TOL_M)

    if report.unmatched:
        logger.error(
            "%d links still have no coordinates (first 10: %s)",
            len(report.unmatched),
            report.unmatched[:10],
        )
        return 1

    if args.dry_run:
        logger.info("--dry-run: not writing")
        return 0

    # Same format as the input unless --out says otherwise.
    out_path = args.out or args.network.with_name(
        f"{args.network.stem}_with_lat_lons{args.network.suffix}"
    )
    alt.Network.from_pydict(network_dict).to_file(str(out_path))
    logger.info("wrote %s", out_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append lat/lon to an existing ALTRIOS network, changing nothing else."
    )
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="output YAML (default: alongside the input, suffixed `_with_lat_lons`)",
    )
    parser.add_argument(
        "--geom-cache",
        type=Path,
        help="JSON cache of way geometry; read when present, written after a fetch",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="fail rather than calling Overpass when the cache misses",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="route profile CSV (cumulative_dist_m, lat, lon) to recover coordinates "
        "by arc length instead of by OSM way match -- for networks resampled from "
        "a profile, which carry no way ids",
    )
    parser.add_argument(
        "--origin-link",
        type=int,
        default=1,
        help="link index the mainline walk starts from, with --profile (default 1)",
    )
    parser.add_argument("--length-tol", type=float, default=LENGTH_TOL_M)
    parser.add_argument("--heading-tol", type=float, default=HEADING_TOL_RAD)
    parser.add_argument(
        "--set-buffers",
        type=float,
        metavar="METERS",
        help="set each synthetic endpoint buffer to METERS. Growing gives long "
        "trains room to stand at the origin; shrinking removes phantom distance "
        "from every run, and requires --longest-train",
    )
    parser.add_argument(
        "--longest-train",
        type=float,
        metavar="METERS",
        help="longest train the corridor must dispatch; required to shrink a "
        "buffer, so the new length can be checked against it",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the match, write nothing"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    network = alt.Network.from_file(str(args.network))
    network_dict = network.to_pydict()
    links = network_dict[1]
    original = json.loads(json.dumps(links))

    if args.profile:
        profile = load_profile(args.profile)
        logger.info(
            "profile: %d points spanning %.3f km",
            len(profile),
            profile[-1][0] / 1000,
        )
        report = append_lat_lons_from_profile(
            links, profile, origin_link=args.origin_link
        )
        return _finish(args, network_dict, links, original, report)

    way_ids = way_ids_in(links)
    logger.info(
        "network has %d links referencing %d OSM ways", len(links), len(way_ids)
    )

    geoms: Dict[str, List[Coord]] = {}
    if args.geom_cache and args.geom_cache.exists():
        geoms = {
            way_id: [tuple(point) for point in points]
            for way_id, points in json.loads(args.geom_cache.read_text()).items()
        }
        logger.info("loaded %d ways from %s", len(geoms), args.geom_cache)

    missing = [way_id for way_id in way_ids if way_id not in geoms]
    if missing:
        if args.offline:
            raise SystemExit(f"--offline set but {len(missing)} ways are not cached")
        geoms.update(fetch_way_geometry(missing))
        if args.geom_cache:
            args.geom_cache.write_text(json.dumps(geoms))
            logger.info("cached %d ways to %s", len(geoms), args.geom_cache)

    report = append_lat_lons(
        links, geoms, length_tol=args.length_tol, heading_tol=args.heading_tol
    )
    return _finish(args, network_dict, links, original, report)


if __name__ == "__main__":
    raise SystemExit(main())
