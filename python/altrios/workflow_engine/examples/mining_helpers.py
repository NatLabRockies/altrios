"""Python helpers for the ``mining_haul.yaml`` example catalog.

Registers exactly one callable: ``mining.dispatch_trucks`` — a
:mod:`schedule builder <altrios.workflow_engine.runner._resolve_schedules>`
that synthesizes truck arrivals from a small site-level spec.

This module is imported by the workflow_engine loader when a catalog
whose ``python_module`` field is ``altrios.workflow_engine.examples.mining_helpers``
is loaded; module-level @register decorators populate the default
:class:`CallableRegistry`.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from altrios.workflow_engine.registry import register


@register("mining.dispatch_trucks")
def dispatch_trucks(
    *,
    schedule: Any,
    env=None,
    state=None,
    config: Mapping[str, Any] = None,
    layout=None,
    rng=None,
    payload_t: float = 220.0,
) -> Iterable[dict]:
    """Synthesize a fixed roster of haul-truck arrivals.

    The site's ``schedules:`` value for this key must be a mapping
    with ``count`` (int) and ``dispatch_interval_hr`` (float):

    .. code-block:: yaml

        schedules:
          haul_cycle.truck_dispatch:
            count: 10
            dispatch_interval_hr: 0.1   # one truck every 6 minutes

    Returns one dict per truck::

        {"kind": "truck", "id": "truck-0", "arrival_time": 0.0,
         "payload_t": 220.0}

    ``payload_t`` defaults to whatever the catalog passes through
    ``schedule_mappings`` (220 tonnes for the shipped example).
    """
    if not isinstance(schedule, Mapping):
        raise ValueError(
            f"mining.dispatch_trucks: schedule must be a mapping with "
            f"'count' and 'dispatch_interval_hr' keys, got {schedule!r}."
        )
    try:
        count = int(schedule["count"])
        interval = float(schedule["dispatch_interval_hr"])
    except KeyError as exc:
        raise ValueError(
            f"mining.dispatch_trucks: schedule missing key {exc}; "
            f"expected 'count' and 'dispatch_interval_hr'."
        ) from None
    if count <= 0:
        return []
    if interval < 0:
        raise ValueError(
            f"mining.dispatch_trucks: dispatch_interval_hr must be >= 0, "
            f"got {interval}."
        )
    return [
        {
            "kind": "truck",
            "id": f"truck-{i}",
            "arrival_time": i * interval,
            "payload_t": float(payload_t),
        }
        for i in range(count)
    ]
