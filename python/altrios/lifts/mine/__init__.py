"""`altrios.lifts.mine` — open-pit mining haul-cycle example catalog.

Pairs ``mining_haul.yaml`` with a small Python helper module
(``mining_helpers``) registered via the
:func:`altrios.lifts.workflow_engine.registry.register` decorator.
This catalog is intentionally tiny; it exists to demonstrate that the
workflow engine is domain-neutral (the same engine that drives the
freight intermodal terminal in :mod:`altrios.lifts.terminal` also
drives this open-pit mining model) and to exercise the
python-escape-hatch pattern.

Example::

    from altrios.lifts import mine

    result = mine.run("example_mine", seed=42)
"""
from __future__ import annotations

from altrios.lifts.workflow_engine import make_runner, make_site_path

site_path = make_site_path(__file__, kind="mine")
run = make_runner(site_path)

__all__ = ["run", "site_path"]
