"""Test fixture module imported by loader tests via ``python_module``.

When :func:`altrios.lifts.workflow_engine.load_catalog` imports this module,
the module-level :func:`~altrios.lifts.workflow_engine.registry.register`
decorators below populate the default callable registry, exposing the
``test_loader.partition_by_kind`` and ``test_loader.initial_items``
names that ``ResourceSpecModel`` fields point at.
"""
from __future__ import annotations

from altrios.lifts.workflow_engine.registry import register


@register("test_loader.partition_by_kind")
def partition_by_kind(item) -> str:
    """Trivial partition function for test catalogs — items partition
    on their ``kind`` attribute (or ``"default"`` if missing)."""
    return getattr(item, "kind", "default")


@register("test_loader.initial_items")
def initial_items() -> list:
    """Trivial initial-items factory for test catalogs."""
    return ["item_a", "item_b"]
