"""Run-scoped output buffers used by the workflow engine.

The interpreter writes simulation outputs into a single
:class:`OutputCollector` per run. The collector accumulates plain dicts
(rows) in three independent lists; downstream code converts them to
polars DataFrames at end of run.

The shape of each row is deliberately schema-loose — the engine writes
whatever the YAML step declares, plus a small set of envelope columns
(``record_timestamp``, ``zone`` if present). Catalog-specific reporting
code (e.g. :mod:`altrios.lifts.terminal`) is responsible for assembling
the wide ``container_data`` / ``resource_log`` / consumption DataFrames
from these rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutputCollector:
    """Three append-only lists of row dicts collected during a run.

    Attributes:
        event_log: One row per ``record_event`` call. Used to assemble
            per-entity wide tables (e.g. container_data).
        resource_log: One row per ``record_resource_event`` call.
            Resource-level status transitions for equipment /
            infrastructure pools.
        consumption_log: One row per ``record_consumption`` call.
            Energy / fuel / emissions accounting; rate × duration is
            computed by the catalog, not by the engine.
    """

    event_log: list[dict[str, Any]] = field(default_factory=list)
    resource_log: list[dict[str, Any]] = field(default_factory=list)
    consumption_log: list[dict[str, Any]] = field(default_factory=list)

    def record_event(self, row: dict[str, Any]) -> None:
        """Append a row to :attr:`event_log`.

        The row is shallow-copied so subsequent caller-side mutation of
        ``row`` does not bleed into the recorded entry.

        Parameters
        ----------
        row : dict
            Per-event payload. Catalog code is responsible for
            schema-loose envelope columns (``record_timestamp``,
            ``zone``, etc.); the engine never inspects keys.
        """
        self.event_log.append(dict(row))

    def record_resource_event(self, row: dict[str, Any]) -> None:
        """Append a row to :attr:`resource_log`.

        Parameters
        ----------
        row : dict
            Per-resource-transition payload (resource name, status,
            timestamp, optional identifiers). Shallow-copied on append.
        """
        self.resource_log.append(dict(row))

    def record_consumption(self, row: dict[str, Any]) -> None:
        """Append a row to :attr:`consumption_log`.

        Parameters
        ----------
        row : dict
            Per-consumption-event payload. Catalogs compute
            ``rate × duration`` themselves before recording; the engine
            does not multiply or otherwise post-process the row.
        """
        self.consumption_log.append(dict(row))
