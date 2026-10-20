"""Locomotive reuse for workflows that dispatch trains WITHOUT the train planner.

The planner itself reuses locomotives when ``TrainPlannerConfig.reuse_locomotives``
is set. Workflows that build and dispatch trains directly (the catenary deep dive)
know each train's departure, arrival and locomotive composition but never track
units, so their fleet was one locomotive per locomotive-trip. This assigns fleet
IDs after the fact with the same rule the planner applies:

1. a unit of the right type already in service and ready at the origin
   (earliest ready first);
2. otherwise one in service and ready at another node -- implicit repositioning,
   counted but not simulated (no energy, no time);
3. otherwise a new unit.

A unit is ready ``max(servicing_hr, extra_hr)`` after it arrives, where
``extra_hr`` is that unit's charging time (zero for diesel), matching the
planner, which runs servicing and charging concurrently from arrival.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def assign_reused_ids(trains: Sequence[dict], servicing_hr: float = 3.0
                      ) -> Tuple[Dict[int, List[int]], dict]:
    """Fleet IDs per train slot.

    ``trains``: one dict per train with ``key`` (any hashable), ``origin``,
    ``destination``, ``dep_hr``, ``arr_hr``, ``types`` (one locomotive type per
    slot, consist order) and optionally ``extra_hr`` (per-slot hours beyond
    arrival before the unit is ready, e.g. charging).

    Returns ``({key: [fleet_id per slot]}, stats)`` with ``stats`` holding
    ``fleet`` (units by type), ``moves`` (implicit repositioning) and
    ``new_units``.
    """
    # ready units per type: list of [ready_hr, node, unit_id]
    ready: Dict[str, List[list]] = {}
    unit_type: Dict[int, str] = {}
    out: Dict = {}
    moves = 0
    for t in sorted(trains, key=lambda r: (r["dep_hr"], str(r["key"]))):
        ids = []
        taken = set()
        for slot, ltype in enumerate(t["types"]):
            pool = ready.setdefault(ltype, [])
            free = [u for u in pool if u[0] <= t["dep_hr"] and u[2] not in taken]
            local = [u for u in free if u[1] == t["origin"]]
            pick = None
            if local:
                pick = min(local, key=lambda u: (u[0], u[2]))
            elif free:
                pick = min(free, key=lambda u: (u[0], u[2]))
                moves += 1
            if pick is not None:
                pool.remove(pick)
                uid = pick[2]
            else:
                uid = len(unit_type) + 1
                unit_type[uid] = ltype
            taken.add(uid)
            ids.append(uid)
        extra = t.get("extra_hr") or [0.0] * len(ids)
        for uid, ltype, ex in zip(ids, t["types"], extra):
            ready.setdefault(ltype, []).append(
                [t["arr_hr"] + max(servicing_hr, float(ex or 0.0)), t["destination"], uid])
        out[t["key"]] = ids
    fleet: Dict[str, int] = {}
    for ltype in unit_type.values():
        fleet[ltype] = fleet.get(ltype, 0) + 1
    return out, {"fleet": fleet, "moves": moves, "new_units": len(unit_type)}
