"""LIFTS — intermodal rail terminal discrete-event simulator.

Public API. New code should import from this top-level package rather
than reaching into individual modules, so future internal restructuring
is cheap.

Phase A.8/A.9 removed the legacy ``run_terminal_simulation`` /
``TerminalMode`` registry path; ``altrios.workflow_engine.run_site`` is
now the only supported entry point. Convenience helpers for assembling
freight DataFrames from a ``RunResult`` are in
``altrios.lifts.python_helpers.assemble_outputs``.
"""
from altrios.lifts.classes import (
    Terminal,
    TerminalState,
    loggingLevel,
)

__all__ = [
    "Terminal",
    "TerminalState",
    "loggingLevel",
]
