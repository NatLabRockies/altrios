"""LIFTS \u2014 the discrete-event workflow-engine subsystem of ALTRIOS.

This package is an umbrella for three co-located sub-packages:

* :mod:`altrios.lifts.workflow_engine` \u2014 the domain-neutral simulation
  engine (primitives, interpreter, loader, output collector). Knows
  nothing about freight, mining, or any specific domain.
* :mod:`altrios.lifts.terminal` \u2014 the freight intermodal rail-terminal
  catalog (the original LIFTS use case): trains, drayage trucks,
  vessels, containers, RTGs, top-picks, yard tractors.
* :mod:`altrios.lifts.mine` \u2014 a small open-pit mining haul-cycle
  example catalog. Demonstrates the engine is genuinely domain-neutral.

The canonical entry point is :func:`altrios.lifts.workflow_engine.run_site`
against a site YAML in :mod:`altrios.lifts.terminal.sites` (freight)
or :mod:`altrios.lifts.mine.sites` (mining). Convenience helpers for
assembling freight DataFrames from a :class:`RunResult` are in
:func:`altrios.lifts.terminal.python_helpers.assemble_outputs`.
"""
from altrios.lifts.terminal.classes import loggingLevel

__all__ = [
    "loggingLevel",
]
