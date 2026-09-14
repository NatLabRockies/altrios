"""
Re-filter grade spikes in a stored ALTRIOS network, leaving everything else alone.

``build_altrios_with_sidings`` smooths and spike-filters elevations, but only
along the mainline::

    mainline_elevs = filter_grade_spikes(mainline_elevs, mainline_distances, ...)
    for i in range(len(mainline_nodes)):
        elevations[i] = mainline_elevs[i]

Siding nodes are never written back, so they keep raw sampled elevations. The
result is unusable track: Toronto-Battle Creek has siding grades to 90% and
Spokane-Lethbridge to 39%, against mainline 99th percentiles near 1%. That
matters because meet-pass routes trains into sidings, so a train can be sent
onto a grade no locomotive could climb.

The mainline is not clean either. Its filter runs at the build's ``max_grade``
(4% by default, far above any Class 1 ruling grade) and skips the first and
last 500 m of the path entirely, so spikes survive at both ends: Toronto's
mainline reaches 15.85% and Winnipeg-Chicago's 5.83%.

This module re-applies the builder's own filter to every path -- mainline and
each siding -- at a grade cap appropriate to the railroad. Only ``elevs``
values change; offsets, lengths, speed limits, topology and coordinates are
untouched, and a link's reverse twin is kept consistent with it.

Usage:
    python repair_grades.py --network path/to/net.msgpack --max-grade 0.025
    python repair_grades.py --network ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import altrios as alt

# Reuse the builder's own filter so a repaired network matches a freshly built
# one rather than approximating it.
from build_altrios_with_sidings import filter_grade_spikes

logger = logging.getLogger(__name__)

# Ruling grades on the corridors modelled here are near or below 1.5% (CN
# Toronto-Sarnia under 1%, CN north of Superior about 1.3%, CP Crowsnest about
# 2.2% at its worst). 2.5% leaves headroom for the genuinely steep mountain
# sections while removing sampling artifacts, which start well above it.
DEFAULT_MAX_GRADE = 0.025

# The builder's filter ignores the first and last stretch of every path, which
# is why spikes survive at path ends. Repair runs edge-to-edge instead.
EDGE_BUFFER_M = 0.0


@dataclass
class GradeReport:
    paths: int = 0
    links_changed: int = 0
    max_grade_before: float = 0.0
    max_grade_after: float = 0.0
    worst_links: List[Tuple[int, float, float]] = field(default_factory=list)

    def log(self) -> None:
        logger.info("paths filtered        : %d", self.paths)
        logger.info("links with new elevs  : %d", self.links_changed)
        logger.info(
            "max grade             : %.2f%% -> %.2f%%",
            100 * self.max_grade_before,
            100 * self.max_grade_after,
        )
        for idx, before, after in self.worst_links[:8]:
            logger.info(
                "   link %6d  %6.2f%% -> %5.2f%%", idx, 100 * before, 100 * after
            )


def _grade(link: dict) -> float:
    """Absolute grade of a two-point link, or 0 for a degenerate one."""
    elevs = link.get("elevs") or []
    if len(elevs) < 2:
        return 0.0
    run = elevs[-1]["offset_meters"] - elevs[0]["offset_meters"]
    if run <= 0:
        return 0.0
    return abs(elevs[-1]["elev_meters"] - elevs[0]["elev_meters"]) / run


def link_paths(links: Sequence[dict], origin_link: int = 1) -> List[List[int]]:
    """The mainline chain, then every siding chain hanging off it.

    A siding is walked from a link whose predecessor is on the mainline until
    it rejoins. Grades have to be filtered along a path rather than per link,
    because a spike is a relationship between neighbouring nodes.
    """
    by_idx = {link["idx_curr"]: link for link in links}

    idx, seen, mainline = origin_link, set(), []
    while idx and idx not in seen:
        seen.add(idx)
        mainline.append(idx)
        idx = by_idx[idx]["idx_next"]
    on_mainline = set(mainline)

    paths = [mainline]
    claimed = set(mainline)
    for link in links:
        start = link["idx_curr"]
        if start in claimed or not link.get("elevs"):
            continue
        if link["idx_prev"] not in on_mainline:
            continue
        run, walk = [], start
        while walk and walk not in claimed and walk not in on_mainline:
            claimed.add(walk)
            run.append(walk)
            walk = by_idx[walk]["idx_next"]
        if run:
            paths.append(run)
    return paths


def repair_grades(
    links: Sequence[dict],
    max_grade: float = DEFAULT_MAX_GRADE,
    origin_link: int = 1,
) -> GradeReport:
    """Cap grades along every path, in place. Only ``elevs`` values change."""
    by_idx = {link["idx_curr"]: link for link in links}
    report = GradeReport()
    report.max_grade_before = max((_grade(l) for l in links), default=0.0)
    before = {l["idx_curr"]: _grade(l) for l in links}

    for path in link_paths(links, origin_link):
        path = [i for i in path if len(by_idx[i].get("elevs") or []) >= 2]
        # A single-link path still has a grade worth clamping, so only a truly
        # empty one is skipped.
        if not path:
            continue

        # Node elevations along the path: each link contributes its far node,
        # and consecutive links share the node between them.
        nodes = [by_idx[path[0]]["elevs"][0]["elev_meters"]]
        distances = [0.0]
        for i in path:
            nodes.append(by_idx[i]["elevs"][-1]["elev_meters"])
            distances.append(distances[-1] + by_idx[i]["length_meters"])

        # A siding shares its first and last node with the mainline it leaves
        # and rejoins, and ALTRIOS validates that a link's last elevation
        # equals its successor's first -- including across `idx_next_alt`.
        # Filtering the two paths independently moves those shared nodes apart,
        # so the junction elevations are pinned to whatever the mainline holds.
        entry = by_idx.get(by_idx[path[0]]["idx_prev"])
        exit_ = by_idx.get(by_idx[path[-1]]["idx_next"])
        pin_lo = (
            entry["elevs"][-1]["elev_meters"]
            if entry is not None and entry["idx_curr"] not in path and entry.get("elevs")
            else None
        )
        pin_hi = (
            exit_["elevs"][0]["elev_meters"]
            if exit_ is not None and exit_["idx_curr"] not in path and exit_.get("elevs")
            else None
        )
        if pin_lo is not None:
            nodes[0] = pin_lo
        if pin_hi is not None:
            nodes[-1] = pin_hi

        if len(nodes) >= 3:
            filtered = filter_grade_spikes(
                nodes, distances, max_grade=max_grade, edge_buffer_m=EDGE_BUFFER_M
            )
            # The filter walks forward and can move the final node; put the
            # junctions back so the shared elevations still agree.
            if pin_lo is not None:
                filtered[0] = pin_lo
            if pin_hi is not None:
                filtered[-1] = pin_hi
        else:
            # A one-link path has only two nodes, and the builder's filter
            # returns anything shorter than three unchanged -- which is why
            # single-link sidings kept spikes of 50% and more. Clamp directly,
            # moving the far node: the near one is shared with the mainline
            # and is already filtered.
            filtered = list(nodes)
            run = distances[-1] - distances[0]
            rise = filtered[-1] - filtered[0]
            if run >= 1.0 and abs(rise) / run > max_grade:
                filtered[-1] = filtered[0] + math.copysign(max_grade * run, rise)
        report.paths += 1

        for n, i in enumerate(path):
            link = by_idx[i]
            lo, hi = float(filtered[n]), float(filtered[n + 1])
            if (
                link["elevs"][0]["elev_meters"] == lo
                and link["elevs"][-1]["elev_meters"] == hi
            ):
                continue
            link["elevs"][0]["elev_meters"] = lo
            link["elevs"][-1]["elev_meters"] = hi
            report.links_changed += 1

            # The reverse twin traverses the same ground backwards, so it
            # carries the same elevations in the opposite order.
            flip = by_idx.get(link["idx_flip"])
            if flip is not None and link["idx_flip"] and len(flip.get("elevs") or []) >= 2:
                flip["elevs"][0]["elev_meters"] = hi
                flip["elevs"][-1]["elev_meters"] = lo

    report.max_grade_after = max((_grade(l) for l in links), default=0.0)
    report.worst_links = sorted(
        ((i, before[i], _grade(by_idx[i])) for i in before),
        key=lambda row: row[1],
        reverse=True,
    )[:8]
    return report


def check_elevation_junctions(
    links: Sequence[dict], tol_m: float = 1e-9
) -> List[Tuple[int, str, int, float]]:
    """Places where a link's last elevation disagrees with its successor's first.

    ALTRIOS enforces this at load with zero tolerance, across ``idx_next`` and
    ``idx_next_alt`` alike, so a repair that filters the mainline and a siding
    as separate paths will be rejected unless their shared nodes still agree.
    Checked here so the failure is reported before the network is written.
    """
    by_idx = {link["idx_curr"]: link for link in links}
    breaks = []
    for link in links:
        if not link.get("elevs"):
            continue
        for relation in ("idx_next", "idx_next_alt"):
            other_idx = link.get(relation, 0)
            other = by_idx.get(other_idx)
            if not other_idx or other is None or not other.get("elevs"):
                continue
            delta = other["elevs"][0]["elev_meters"] - link["elevs"][-1]["elev_meters"]
            if abs(delta) > tol_m:
                breaks.append((link["idx_curr"], relation, other_idx, delta))
    return breaks


def assert_only_elevs_changed(original: Sequence[dict], updated: Sequence[dict]) -> None:
    """Raise unless the two differ in elevation values alone."""

    def stripped(links: Sequence[dict]) -> List[dict]:
        out = json.loads(json.dumps(links))
        for link in out:
            for point in link.get("elevs") or []:
                point["elev_meters"] = None
        return out

    if stripped(original) != stripped(updated):
        raise AssertionError(
            "network changed beyond elevation values -- refusing to write"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-grade", type=float, default=DEFAULT_MAX_GRADE)
    parser.add_argument("--origin-link", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    network = alt.Network.from_file(str(args.network))
    network_dict = network.to_pydict()
    links = network_dict[1]
    original = json.loads(json.dumps(links))

    report = repair_grades(links, args.max_grade, args.origin_link)
    report.log()
    assert_only_elevs_changed(original, links)
    logger.info("verified: nothing but elevation values changed")

    junctions = check_elevation_junctions(links)
    if junctions:
        worst = max(junctions, key=lambda row: abs(row[3]))
        logger.error(
            "%d elevation junction mismatches; worst %.3f m at link %d --%s--> %d",
            len(junctions), worst[3], worst[0], worst[1], worst[2],
        )
        return 1
    logger.info("verified: every shared elevation node agrees")

    if args.dry_run:
        logger.info("--dry-run: not writing")
        return 0

    out_path = args.out or args.network
    alt.Network.from_pydict(network_dict).to_file(str(out_path))
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
