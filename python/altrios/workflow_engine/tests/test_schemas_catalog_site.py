"""Tests for the catalog- and site-level pydantic schemas.

Covers ``EntityKindSpecModel``, ``MetaModel``, ``WorkflowModeModel``,
``CatalogModel``, ``LayoutModel``, ``SiteModel`` plus the engine
``Catalog`` / ``WorkflowMode`` dataclasses built by ``to_engine()``.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from altrios.workflow_engine.catalog import Catalog, WorkflowMode
from altrios.workflow_engine.entities import EntityKindSpec
from altrios.workflow_engine.schemas import (
    SCHEMA_VERSION_V1,
    CatalogModel,
    EntityKindSpecModel,
    LayoutModel,
    LayoutNodeModel,
    MetaModel,
    SiteModel,
    StepGraphModel,
    StepModel,
    WorkflowModeModel,
)


# ---- Helpers --------------------------------------------------------


def _make_graph_dict(name: str, entry: str = "s0") -> dict:
    return {
        "name": name,
        "entry": entry,
        "steps": [{"id": entry, "type": "log", "params": {"message": "hi"}}],
    }


# ---- MetaModel ------------------------------------------------------


def test_meta_v1_ok():
    m = MetaModel(schema_version=1)
    assert m.schema_version == SCHEMA_VERSION_V1


def test_meta_rejects_v2():
    with pytest.raises(ValidationError) as exc:
        MetaModel(schema_version=2)
    assert "must be 1" in str(exc.value)


def test_meta_rejects_extra_fields():
    with pytest.raises(ValidationError):
        MetaModel(schema_version=1, future_flag=True)


# ---- EntityKindSpecModel --------------------------------------------


def test_entity_kind_basic():
    k = EntityKindSpecModel(
        name="container",
        attrs={"weight_t": "float", "origin": "str"},
        description="Intermodal container.",
    )
    eng = k.to_engine()
    assert isinstance(eng, EntityKindSpec)
    assert eng.name == "container"
    assert eng.attrs["weight_t"] == "float"


def test_entity_kind_minimum_just_name():
    k = EntityKindSpecModel(name="train")
    assert k.attrs == {}
    eng = k.to_engine()
    assert eng.name == "train"


def test_entity_kind_rejects_empty_name():
    with pytest.raises(ValidationError):
        EntityKindSpecModel(name="")


def test_entity_kind_rejects_extra_fields():
    with pytest.raises(ValidationError):
        EntityKindSpecModel(name="x", colour="red")


# ---- WorkflowModeModel ----------------------------------------------


def test_mode_basic():
    m = WorkflowModeModel(
        name="truck_rail",
        arrival_routing={"container": "container_flow"},
        graphs=[_make_graph_dict("container_flow")],
        resources=[
            {"name": "crane", "kind": "Resource", "role": "equipment", "capacity": 4}
        ],
    )
    eng = m.to_engine()
    assert isinstance(eng, WorkflowMode)
    assert eng.name == "truck_rail"
    assert eng.arrival_routing["container"] == "container_flow"
    assert "container_flow" in eng.graphs
    assert eng.resource_specs[0].name == "crane"


def test_mode_rejects_routing_to_unknown_graph():
    with pytest.raises(ValidationError) as exc:
        WorkflowModeModel(
            name="m",
            arrival_routing={"container": "does_not_exist"},
            graphs=[_make_graph_dict("real_graph")],
        )
    assert "names no declared graph" in str(exc.value)


def test_mode_rejects_duplicate_graph_names():
    with pytest.raises(ValidationError) as exc:
        WorkflowModeModel(
            name="m",
            graphs=[_make_graph_dict("g1"), _make_graph_dict("g1")],
        )
    assert "Duplicate graph name" in str(exc.value)


def test_mode_rejects_duplicate_resource_names():
    with pytest.raises(ValidationError) as exc:
        WorkflowModeModel(
            name="m",
            graphs=[_make_graph_dict("g1")],
            resources=[
                {"name": "x", "kind": "Resource", "role": "equipment"},
                {"name": "x", "kind": "Store", "role": "queue"},
            ],
        )
    assert "Duplicate resource name" in str(exc.value)


def test_mode_requires_at_least_one_graph():
    with pytest.raises(ValidationError):
        WorkflowModeModel(name="m", graphs=[])


def test_mode_rejects_extra_fields():
    with pytest.raises(ValidationError):
        WorkflowModeModel(
            name="m",
            graphs=[_make_graph_dict("g1")],
            mystery_field=42,
        )


# ---- CatalogModel ---------------------------------------------------


def _catalog_minimum() -> dict:
    return {
        "meta": {"schema_version": 1},
        "name": "freight",
        "entity_kinds": [{"name": "container"}],
        "modes": [
            {
                "name": "truck_rail",
                "arrival_routing": {"container": "g1"},
                "graphs": [_make_graph_dict("g1")],
            }
        ],
    }


def test_catalog_basic():
    cm = CatalogModel.model_validate(_catalog_minimum())
    cat = cm.to_engine()
    assert isinstance(cat, Catalog)
    assert cat.name == "freight"
    assert cat.schema_version == 1
    assert "container" in cat.entity_kinds
    assert cat.mode("truck_rail").arrival_routing["container"] == "g1"


def test_catalog_rejects_v2_meta():
    data = _catalog_minimum()
    data["meta"]["schema_version"] = 2
    with pytest.raises(ValidationError):
        CatalogModel.model_validate(data)


def test_catalog_rejects_duplicate_entity_kinds():
    data = _catalog_minimum()
    data["entity_kinds"] = [{"name": "container"}, {"name": "container"}]
    with pytest.raises(ValidationError) as exc:
        CatalogModel.model_validate(data)
    assert "Duplicate entity_kind" in str(exc.value)


def test_catalog_rejects_duplicate_mode_names():
    data = _catalog_minimum()
    data["modes"].append(dict(data["modes"][0]))
    with pytest.raises(ValidationError) as exc:
        CatalogModel.model_validate(data)
    assert "Duplicate mode name" in str(exc.value)


def test_catalog_engine_rejects_routing_to_undeclared_kind():
    """Routing to an undeclared kind is caught at engine-build time,
    not in pydantic (validator-scope rules)."""
    data = _catalog_minimum()
    data["modes"][0]["arrival_routing"] = {"mystery_kind": "g1"}
    cm = CatalogModel.model_validate(data)
    with pytest.raises(ValueError) as exc:
        cm.to_engine()
    assert "unknown entity kind 'mystery_kind'" in str(exc.value)


def test_catalog_optional_fields_default_empty():
    cm = CatalogModel.model_validate(_catalog_minimum())
    cat = cm.to_engine()
    assert cat.consumption_rates == {}
    assert cat.schedule_mappings == {}
    assert cat.python_module is None


def test_catalog_python_module_carried_through():
    data = _catalog_minimum()
    data["python_module"] = "altrios.lifts.helpers"
    cm = CatalogModel.model_validate(data)
    cat = cm.to_engine()
    assert cat.python_module == "altrios.lifts.helpers"


def test_catalog_rejects_extra_fields():
    data = _catalog_minimum()
    data["surprise"] = "boo"
    with pytest.raises(ValidationError):
        CatalogModel.model_validate(data)


def test_catalog_no_modes_allowed():
    """Catalogs with no modes are syntactically valid (a domain might
    ship pure entity_kinds for later remixing)."""
    data = {
        "meta": {"schema_version": 1},
        "name": "empty",
        "entity_kinds": [{"name": "x"}],
    }
    cm = CatalogModel.model_validate(data)
    cat = cm.to_engine()
    assert cat.modes == ()


# ---- LayoutModel ----------------------------------------------------


def test_layout_node_xy():
    n = LayoutNodeModel(x=10.0, y=20.0)
    assert n.z is None


def test_layout_node_with_z():
    n = LayoutNodeModel(x=10.0, y=20.0, z=3.5)
    assert n.z == 3.5


def test_layout_node_rejects_extra():
    with pytest.raises(ValidationError):
        LayoutNodeModel(x=0, y=0, w=1)


def test_layout_basic():
    lm = LayoutModel(
        nodes={
            "berth_1": {"x": 0, "y": 0},
            "stack_A": {"x": 380, "y": 50},
        }
    )
    assert lm.nodes["stack_A"].x == 380


def test_layout_empty_nodes_ok():
    lm = LayoutModel()
    assert lm.nodes == {}


# ---- SiteModel ------------------------------------------------------


def _site_minimum() -> dict:
    return {
        "meta": {"schema_version": 1},
        "name": "rotterdam",
        "catalog": "altrios.lifts",
    }


def test_site_basic():
    s = SiteModel.model_validate(_site_minimum())
    assert s.name == "rotterdam"
    assert s.catalog == "altrios.lifts"
    assert s.modes == []
    assert s.layout is None
    assert s.seed is None


def test_site_full():
    s = SiteModel.model_validate(
        {
            "meta": {"schema_version": 1},
            "name": "rotterdam",
            "catalog": "altrios.lifts",
            "modes": ["truck_rail", "rail_vessel"],
            "config": {"crane_count": 4},
            "layout": {"nodes": {"berth_1": {"x": 0, "y": 0}}},
            "resource_overrides": {"crane": {"capacity": 8}},
            "schedules": {"truck_rail.train_arrivals": "data/x.csv"},
            "seed": 42,
        }
    )
    assert s.modes == ["truck_rail", "rail_vessel"]
    assert s.config == {"crane_count": 4}
    assert s.layout is not None
    assert s.layout.nodes["berth_1"].x == 0
    assert s.resource_overrides["crane"]["capacity"] == 8
    assert s.seed == 42


def test_site_rejects_duplicate_modes():
    data = _site_minimum()
    data["modes"] = ["truck_rail", "truck_rail"]
    with pytest.raises(ValidationError) as exc:
        SiteModel.model_validate(data)
    assert "Duplicate mode" in str(exc.value)


def test_site_requires_meta():
    data = _site_minimum()
    del data["meta"]
    with pytest.raises(ValidationError):
        SiteModel.model_validate(data)


def test_site_requires_catalog():
    data = _site_minimum()
    del data["catalog"]
    with pytest.raises(ValidationError):
        SiteModel.model_validate(data)


def test_site_rejects_v2_meta():
    data = _site_minimum()
    data["meta"]["schema_version"] = 2
    with pytest.raises(ValidationError):
        SiteModel.model_validate(data)


def test_site_rejects_extra_fields():
    data = _site_minimum()
    data["bonus"] = 1
    with pytest.raises(ValidationError):
        SiteModel.model_validate(data)


def test_site_extends_accepted_as_string():
    """``extends:`` is purely advisory at schema level; the loader
    resolves it before validation."""
    data = _site_minimum()
    data["extends"] = "./base.yaml"
    s = SiteModel.model_validate(data)
    assert s.extends == "./base.yaml"


# ---- Engine Catalog dataclass invariants ----------------------------


def test_engine_catalog_rejects_schema_version_other_than_1():
    with pytest.raises(ValueError) as exc:
        Catalog(
            name="c",
            schema_version=2,
            modes=(),
            entity_kinds={},
        )
    assert "schema_version must be 1" in str(exc.value)


def test_engine_workflow_mode_rejects_routing_to_unknown_graph():
    from altrios.workflow_engine.steps import Step, StepGraph
    g = StepGraph(
        name="g1",
        entry="s0",
        steps={"s0": Step(id="s0", type="log", params={"message": "hi"})},
    )
    with pytest.raises(ValueError) as exc:
        WorkflowMode(
            name="m",
            arrival_routing={"container": "missing"},
            graphs={"g1": g},
            resource_specs=(),
        )
    assert "unknown graph 'missing'" in str(exc.value)


def test_engine_catalog_mode_lookup():
    from altrios.workflow_engine.steps import Step, StepGraph
    g = StepGraph(
        name="g1",
        entry="s0",
        steps={"s0": Step(id="s0", type="log", params={"message": "hi"})},
    )
    mode = WorkflowMode(
        name="truck_rail",
        arrival_routing={},
        graphs={"g1": g},
        resource_specs=(),
    )
    cat = Catalog(
        name="c",
        schema_version=1,
        modes=(mode,),
        entity_kinds={},
    )
    assert cat.mode("truck_rail") is mode
    with pytest.raises(KeyError) as exc:
        cat.mode("nope")
    assert "Available" in str(exc.value)
