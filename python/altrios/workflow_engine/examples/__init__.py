"""Workflow-engine example catalogs and the helpers they reference.

Each example pairs a YAML catalog (e.g. ``mining_haul.yaml``) with a
small Python helper module (``mining_helpers``) registered via the
:func:`altrios.workflow_engine.registry.register` decorator. The
helpers exist to demonstrate the python-escape-hatch pattern; the
examples themselves are intentionally tiny.
"""
