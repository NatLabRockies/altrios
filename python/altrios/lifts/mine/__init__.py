"""`altrios.lifts.mine` — open-pit mining haul-cycle example catalog.

Pairs ``mining_haul.yaml`` with a small Python helper module
(``mining_helpers``) registered via the
:func:`altrios.lifts.workflow_engine.registry.register` decorator.
This catalog is intentionally tiny; it exists to demonstrate that the
workflow engine is domain-neutral (the same engine that drives the
freight intermodal terminal in :mod:`altrios.lifts.terminal` also
drives this open-pit mining model) and to exercise the
python-escape-hatch pattern.
"""
