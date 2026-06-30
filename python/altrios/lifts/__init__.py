"""LIFTS — intermodal rail terminal discrete-event simulator.

Public API. New code should import from this top-level package rather
than reaching into individual modules, so future internal restructuring
is cheap.

The canonical entry point is :func:`altrios.workflow_engine.run_site`
against one of the site files in :mod:`altrios.lifts.sites`.
Convenience helpers for assembling freight DataFrames from a
:class:`RunResult` are in
:func:`altrios.lifts.python_helpers.assemble_outputs`.
"""
from altrios.lifts.classes import loggingLevel

__all__ = [
    "loggingLevel",
]
