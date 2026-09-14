"""
Convert stored ALTRIOS networks between serialization formats.

Network YAML is large and slow: the Winnipeg-Chicago corridor is a ~48 MB text
file, and parsing it dominates the startup of anything that loads it. ALTRIOS
already accepts MessagePack through ``SerdeAPI`` (the ``msgpack`` feature is on
by default), so the same network can be stored as a compact binary instead.

Every conversion is verified by reloading the result and comparing it to the
source through ``to_pydict()``. A format change that altered a single speed
limit or elevation would be far worse than a slow load, so nothing is reported
as converted unless the round trip is exact.

Usage:
    python convert_network_format.py --list
    python convert_network_format.py --all --to msgpack
    python convert_network_format.py --network path/to/net.yaml --to msgpack
    python convert_network_format.py --all --to msgpack --delete-source
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import altrios as alt

logger = logging.getLogger(__name__)

# Files under resources/networks that are not whole networks.
NOT_A_NETWORK = ("links_test",)


def discover_networks(root: Optional[Path] = None, suffix: str = ".yaml") -> List[Path]:
    """Every stored network file with ``suffix``, largest first."""
    root = root or (Path(alt.resources_root()) / "networks")
    found = [
        path
        for path in sorted(root.rglob(f"*{suffix}"))
        if path.stem not in NOT_A_NETWORK
    ]
    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)


def _load_timed(path: Path) -> Tuple[object, float]:
    """``(network, seconds)`` for loading ``path``."""
    start = time.perf_counter()
    network = alt.Network.from_file(str(path))
    return network, time.perf_counter() - start


def convert(path: Path, to_format: str = "msgpack") -> dict:
    """Convert one network, verifying the round trip is lossless.

    Returns a record of sizes and load times, or raises if the reloaded network
    does not match the source exactly.
    """
    source, load_source_s = _load_timed(path)
    target = path.with_suffix(f".{to_format}")

    start = time.perf_counter()
    source.to_file(str(target))
    write_s = time.perf_counter() - start

    reloaded, load_target_s = _load_timed(target)
    if source.to_pydict() != reloaded.to_pydict():
        target.unlink(missing_ok=True)
        raise AssertionError(
            f"{path.name}: reloaded {to_format} does not match the source -- "
            "conversion discarded"
        )

    return {
        "name": path.name,
        "source_mb": path.stat().st_size / 1e6,
        "target_mb": target.stat().st_size / 1e6,
        "load_source_s": load_source_s,
        "load_target_s": load_target_s,
        "write_s": write_s,
        "target": target,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--network", type=Path, help="convert a single network")
    parser.add_argument(
        "--all", action="store_true", help="convert every stored network"
    )
    parser.add_argument("--to", default="msgpack", help="target format (default msgpack)")
    parser.add_argument(
        "--list", action="store_true", help="list stored networks and exit"
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="remove the original after a verified conversion",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        for path in discover_networks():
            logger.info("%7.1f MB  %s", path.stat().st_size / 1e6, path)
        return 0

    if args.network:
        targets = [args.network]
    elif args.all:
        targets = discover_networks()
    else:
        parser.error("pass --network, --all, or --list")

    records, failures = [], []
    for path in targets:
        try:
            record = convert(path, args.to)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("FAILED %s: %s", path.name, exc)
            failures.append(path)
            continue
        records.append(record)
        logger.info(
            "%-46s %7.1f -> %6.1f MB (%.0f%% smaller)  load %5.2f -> %5.2f s (%.1fx)",
            record["name"],
            record["source_mb"],
            record["target_mb"],
            100 * (1 - record["target_mb"] / record["source_mb"]),
            record["load_source_s"],
            record["load_target_s"],
            record["load_source_s"] / max(record["load_target_s"], 1e-9),
        )

    if records:
        src = sum(r["source_mb"] for r in records)
        tgt = sum(r["target_mb"] for r in records)
        load_src = sum(r["load_source_s"] for r in records)
        load_tgt = sum(r["load_target_s"] for r in records)
        logger.info(
            "TOTAL %d networks: %.1f -> %.1f MB (%.0f%% smaller), "
            "load %.1f -> %.1f s (%.1fx faster)",
            len(records),
            src,
            tgt,
            100 * (1 - tgt / src),
            load_src,
            load_tgt,
            load_src / max(load_tgt, 1e-9),
        )

    # Deleting is deferred until every conversion has been verified, so a
    # failure part-way through never leaves a network with no readable copy.
    if args.delete_source and not failures:
        for path in targets:
            path.unlink()
            logger.info("removed %s", path.name)
    elif args.delete_source:
        logger.error("not deleting sources: %d conversion(s) failed", len(failures))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
