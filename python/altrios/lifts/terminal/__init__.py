"""`altrios.lifts.terminal` — the freight intermodal rail-terminal catalog.

Original LIFTS use case: discrete-event simulation of trains, drayage
trucks, vessels, containers, rubber-tyred gantries (RTGs), top-picks,
and yard tractors at a multimodal rail/ocean terminal. Drives the
domain-neutral :mod:`altrios.lifts.workflow_engine` via the YAML in
``catalog.yaml`` and the helpers in :mod:`altrios.lifts.terminal.python_helpers`.

Sites live under :mod:`altrios.lifts.terminal.sites` (the Allouez
``allouez_*.yaml`` family). The canonical entry point is
:func:`altrios.lifts.workflow_engine.run_site`.
"""
