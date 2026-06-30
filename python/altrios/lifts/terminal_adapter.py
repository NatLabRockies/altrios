"""Phase 3 / Strategy B — read-only ``Terminal`` look-alike.

This module is a **pure attribute proxy** used by the freight catalog
during the B (adapter shim) phase of the workflow-engine migration.
Its sole job is to expose to the existing freight Python helpers
(``train_flow.py``, ``drayage_flow.py``, ``vessel_flow.py``,
``yard_flow.py``, ``utilities.py``, ``consumption.py``) the same
attribute surface the legacy :class:`~altrios.lifts.classes.Terminal`
class exposes, **without modifying any of those helpers**.

The adapter is a deliberate, time-bounded stepping stone:

1. Under B, each freight workflow YAML is a one-step ``python:`` graph
   that calls ``process_train_arrival(env, terminal_adapter, entry)``
   (or the drayage / vessel equivalent). Existing helpers see a
   ``terminal``-shaped object and Just Work.
2. Under A (immediately after B), the helpers are refactored to take
   ``(env, state, config, output, ...)`` directly, freight workflows
   are decomposed into fine-grained YAML graphs, and **this module
   is deleted**.

**Discipline rules (per Decision D7 in WORKFLOW_ENGINE_PLAN.md):**

- :class:`TerminalAdapter` is a pure attribute holder. **No methods
  with logic, no conditional behaviour, no new state.** Anything
  substantive belongs in phase A.
- If a PR tries to add real behaviour to this class, the PR should
  be redirected to phase A instead.
- This file is on the deletion list for phase A.11.

Surface area provided (verified via ``grep terminal\.[a-z_]+``
against the freight Python tree, 2026-06-30):

- ``env``                              — passed through from runner
- ``state``                            — passed through (the runner's
  ``SimpleNamespace`` of pools, augmented with ``container_events``)
- ``config``                           — passed through (merged
  catalog defaults + site config)
- ``energy_use_config``                — alias for
  ``config['energy_use']`` (read by ``utilities.compute_consumption``)
- ``distances``                        — yard-distance table built by
  ``calculate_distances`` (read by ``yard_flow``)
- ``layout``                           — passed through (engine
  :class:`~altrios.workflow_engine.Layout`, or ``None``)
- ``log_level``                        — passed through
- ``log(level, msg)``                  — delegates to
  ``altrios.lifts.utilities.log``
- Eight freight numeric constants exposed as UPPER_SNAKE attributes,
  resolved from the merged config dict at construction time. The
  freight catalog ships them via ``config_defaults:``; sites can
  override.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from altrios.lifts import utilities
from altrios.lifts.distances import calculate_distances


# Public-but-frozen mapping of UPPER_SNAKE attribute name to lowercase
# config key. The catalog's ``config_defaults:`` ships values for every
# config key in the RHS column.
_CONSTANT_CONFIG_KEYS: Mapping[str, str] = {
    "CONTAINERS_PER_CRANE_MOVE_MEAN": "containers_per_crane_move_mean",
    "CRANE_MOVE_DEV_TIME": "crane_move_dev_time",
    "TRUCK_DIESEL_PERCENTAGE": "truck_diesel_percentage",
    "TRUCK_ARRIVAL_MEAN": "truck_arrival_mean",
    "TRUCK_INGATE_TIME": "truck_ingate_time",
    "TRUCK_INGATE_TIME_DEV": "truck_ingate_time_dev",
    "TRUCK_OUTGATE_TIME": "truck_outgate_time",
    "TRUCK_OUTGATE_TIME_DEV": "truck_outgate_time_dev",
}


class TerminalAdapter:
    """Read-only ``Terminal``-shaped facade over ``(env, state, config)``.

    Constructed once per run by the freight catalog's ``state_init``
    helper (:func:`altrios.lifts.python_helpers.build_freight_state`),
    then stashed on ``state.terminal_adapter`` so the catalog's
    one-step ``python:`` graphs can pass it to the unmodified freight
    generators.

    Parameters
    ----------
    env
        The SimPy :class:`simpy.Environment` for the run.
    state
        The runner's :class:`types.SimpleNamespace` of resource pools.
        Must already have a ``container_events: list`` attribute
        attached by the calling ``state_init`` (the freight helpers
        ``.append`` to it directly).
    config
        The merged config dict (catalog ``config_defaults:`` overlaid
        by site ``config:``). Must contain a top-level ``energy_use``
        sub-dict; this is the consumption-rate table the freight
        ``compute_consumption`` helper reads. Must also contain every
        key in :data:`_CONSTANT_CONFIG_KEYS`'s values; missing keys
        produce a :class:`KeyError` at construction with a clear
        message naming the catalog the user should check.
    layout
        The engine :class:`~altrios.workflow_engine.Layout` (or
        ``None``). Currently unused by freight helpers; passed
        through so future helpers can read it without another
        refactor.
    log_level
        Whatever the legacy freight ``loggingLevel`` enum value the
        run is using. Defaults to ``loggingLevel.BASIC`` if not
        supplied. Read by ``terminal.log()`` callers; otherwise
        opaque to the adapter.

    Raises
    ------
    KeyError
        If ``config`` is missing the ``energy_use`` block or any of
        the eight freight numeric-constant config keys. The message
        names the missing key and points at the catalog's
        ``config_defaults`` block.
    """

    def __init__(
        self,
        env: Any,
        state: Any,
        config: Mapping[str, Any],
        *,
        layout: Any = None,
        log_level: Optional[Any] = None,
    ) -> None:
        self.env = env
        self.state = state
        self.config = config
        self.layout = layout

        # Energy/consumption rate table (compute_consumption indexes
        # into this). Required for any freight run that records
        # consumption rows.
        try:
            self.energy_use_config = config["energy_use"]
        except KeyError:
            raise KeyError(
                "TerminalAdapter: config is missing the 'energy_use' "
                "sub-dict (consumption-rate table). The freight catalog "
                "must ship it under config_defaults.energy_use; sites "
                "may override individual entries."
            ) from None

        # Per-event freight constants. Pulled out as UPPER_SNAKE attrs
        # because the legacy helpers read them as `terminal.X` without
        # any indirection.
        missing: list[str] = []
        for attr, cfg_key in _CONSTANT_CONFIG_KEYS.items():
            if cfg_key not in config:
                missing.append(cfg_key)
                continue
            setattr(self, attr, config[cfg_key])
        if missing:
            raise KeyError(
                f"TerminalAdapter: config is missing required freight "
                f"constant key(s) {missing!r}. Add them under the "
                f"catalog's config_defaults: block (or override per "
                f"site)."
            )

        # Yard distance table — freight-specific layout computation
        # done eagerly here so yard_flow's `terminal.distances` reads
        # are cheap.
        self.distances = calculate_distances(
            config=config, config_path=None, actual_railcars=None,
        )
        self.yard_length = self.distances["yard_length"]
        self.track_capacity = self.distances["n_max"]

        # Log threshold. Stored both on self (read by freight helpers
        # that introspect `terminal.log_level`) and on the utilities
        # module (read by `utilities.log` itself, which the helpers
        # call through `self.log`).
        if log_level is None:
            from altrios.lifts.classes import loggingLevel
            log_level = loggingLevel.BASIC
        self.log_level = log_level
        utilities.set_log_level(log_level)

        # Track numbers / gate counts that classes.Terminal pulls flat.
        # Several diagnostics in the existing flow modules read these
        # directly off `terminal.X`; keeping the names intact avoids
        # rewriting them in B.
        yard_cfg = config.get("yard", {}) or {}
        gate_cfg = config.get("gates", {}) or {}
        self.track_number = yard_cfg.get("track_number")
        self.in_gate_numbers = gate_cfg.get("in_gate_numbers")
        self.out_gate_numbers = gate_cfg.get("out_gate_numbers")

    def log(self, level: Any, msg: str) -> None:
        """Forward to :func:`altrios.lifts.utilities.log`. The legacy
        ``Terminal.log`` is a one-liner doing the same; we keep this
        method on the adapter so unmodified helpers (e.g.
        ``train_flow.process_train_arrival``) that call
        ``terminal.log(level, msg)`` still work."""
        utilities.log(level, msg)
