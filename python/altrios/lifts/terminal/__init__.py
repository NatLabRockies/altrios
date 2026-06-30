"""`altrios.lifts.terminal` — the freight intermodal rail-terminal catalog.

Original LIFTS use case: discrete-event simulation of trains, drayage
trucks, vessels, containers, rubber-tyred gantries (RTGs), top-picks,
and yard tractors at a multimodal rail/ocean terminal. Drives the
domain-neutral :mod:`altrios.lifts.workflow_engine` via the YAML in
``catalog.yaml`` and the helpers in :mod:`altrios.lifts.terminal.python_helpers`.

Sites live under :mod:`altrios.lifts.terminal.sites` (the Allouez
``allouez_*.yaml`` family). The canonical entry points are
:func:`run` (resolve a bundled site by name and run it) and
:func:`site_path` (just resolve the path).

Example::

    from altrios.lifts import terminal

    result = terminal.run("allouez_truck_rail", seed=42)
"""
from __future__ import annotations

from altrios.lifts.workflow_engine import make_runner, make_site_path

site_path = make_site_path(__file__, kind="terminal")
run = make_runner(site_path)

__all__ = ["run", "site_path"]
