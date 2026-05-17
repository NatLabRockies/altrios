"""LIFTS — intermodal rail terminal discrete-event simulator.

Public API. New code should import from this top-level package rather than
reaching into individual modules, so future internal restructuring is cheap.
"""
from altrios.lifts.classes import (
    Terminal,
    TerminalState,
    loggingLevel,
)
from altrios.lifts.terminal_sim import (
    TerminalMode,
    get_mode,
    list_modes,
    register_mode,
    run_terminal_simulation,
)

__all__ = [
    "Terminal",
    "TerminalState",
    "TerminalMode",
    "get_mode",
    "list_modes",
    "loggingLevel",
    "register_mode",
    "run_terminal_simulation",
]
