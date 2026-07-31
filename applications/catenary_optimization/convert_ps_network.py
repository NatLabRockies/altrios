"""Convert a Parallel-Systems-format ALTRIOS network YAML (bare link list with
`length`, list-style `speed_sets`, `Lat`/`Lon` headings) into the schema used by
this repo's altrios-core (the `[config, [links]]` tuple with `length_meters`,
map-style `speed_sets`, and lat/lon headings).

Usage:
    python convert_ps_network.py <source.yaml> <dest.yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

import altrios as alt

# Permissive error tolerances so geometry-only imports pass validation despite
# known grade/heading spikes in the source data (this network is used only for
# the energization optimizer, which reads geometry, not for simulation).
_CONFIG = {
    "max_grade": 5.0,
    "max_curv_radians_per_meter": 100.0,
    "max_heading_step_radians": 6.28,
    "max_elev_step_meters": 1.0e9,
}


def _convert_link(link: dict) -> dict:
    elevs = [
        {"offset_meters": e["offset"], "elev_meters": e["elev"]}
        for e in (link.get("elevs") or [])
    ]
    headings = [
        {
            "offset_meters": h["offset"],
            "heading_radians": h.get("heading", 0.0),
            "lat": h.get("Lat", h.get("lat")),
            "lon": h.get("Lon", h.get("lon")),
        }
        for h in (link.get("headings") or [])
    ]
    speed_sets: dict = {}
    for ss in link.get("speed_sets") or []:
        train_type = ss.get("train_type") or "Freight"
        speed_sets[train_type] = {
            "speed_limits": [
                {
                    "offset_start_meters": sl["offset_start"],
                    "offset_end_meters": sl["offset_end"],
                    "speed_meters_per_second": sl["speed"],
                }
                for sl in (ss.get("speed_limits") or [])
            ],
            "speed_params": ss.get("speed_params", []),
            "is_head_end": ss.get("is_head_end", False),
        }
    length_meters = link.get("length", 0.0)
    # Validation requires each non-dummy link to have either a non-empty
    # speed_sets map or a speed_set. Links lacking speed data (e.g. some
    # sidings/turnouts) get a default speed_set; speed is unused by the optimizer.
    speed_set = None
    if not speed_sets and length_meters > 0.0:
        speed_set = {
            "speed_limits": [
                {
                    "offset_start_meters": 0.0,
                    "offset_end_meters": length_meters,
                    "speed_meters_per_second": 20.0,
                }
            ],
            "speed_params": [],
            "is_head_end": False,
        }
    return {
        "idx_curr": link["idx_curr"],
        "idx_flip": link.get("idx_flip", 0),
        "idx_next": link.get("idx_next", 0),
        "idx_next_alt": link.get("idx_next_alt", 0),
        "idx_prev": link.get("idx_prev", 0),
        "idx_prev_alt": link.get("idx_prev_alt", 0),
        "osm_id": str(link.get("osm_id", "0")),
        "length_meters": length_meters,
        "elevs": elevs,
        "headings": headings,
        "speed_sets": speed_sets,
        "speed_set": speed_set,
        "cat_power_limits": [],
        "link_idxs_lockout": link.get("link_idxs_lockout", []),
    }


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    log = dst.with_suffix(".convert.log")
    try:
        raw = yaml.safe_load(src.read_text())
        links = [_convert_link(link) for link in raw]
        network = alt.Network.from_pydict([_CONFIG, links])
        network.to_file(str(dst))
        log.write_text(f"OK: converted {len(links)} links -> {dst}\n")
        print(f"Converted {len(links)} links -> {dst}")
    except Exception:
        import traceback

        log.write_text("FAILED:\n" + traceback.format_exc())
        raise



if __name__ == "__main__":
    main()
