"""Tests for :mod:`altrios.lifts.workflow_engine.loader`.

Covers :func:`load_catalog` and :func:`load_site` end-to-end:

* Schema validation propagation.
* ``python_module`` import + ``_python`` resolver wiring.
* Site ``extends:`` single-level deep-merge.
* Catalog reference resolution (filesystem path and Python dotted
  module path).
* Cross-validation that site-activated modes exist in the catalog.

The fixture module :mod:`._loader_helpers` registers the two callable
names that ``ResourceSpecModel.partition_by_python`` /
``init_items_python`` fields point at.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from altrios.lifts.workflow_engine import LoaderError, load_catalog, load_site
from altrios.lifts.workflow_engine.catalog import Catalog
from altrios.lifts.workflow_engine.yaml_loader import YamlLoaderError


# Importing the fixture module here ensures its @register decorators
# have fired by the time the loader tries to resolve the names. (In
# normal use this happens via the catalog's ``python_module:`` field;
# we exercise that path explicitly below as well.)
from altrios.lifts.workflow_engine.tests import _loader_helpers  # noqa: F401


CATALOG_MINIMAL_YAML = """\
meta:
  schema_version: 1
name: test_catalog
entity_kinds:
  - name: container
    attrs:
      weight_t: float
modes:
  - name: truck_rail
    arrival_routing:
      container: handle_container
    graphs:
      - name: handle_container
        entry: log_step
        steps:
          - id: log_step
            type: log
            params:
              message: "Handling container"
"""


SITE_MINIMAL_YAML = """\
meta:
  schema_version: 1
name: test_site
catalog: {catalog_path}
modes: [truck_rail]
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


# ---- load_catalog ---------------------------------------------------


def test_load_catalog_minimal(tmp_path: Path):
    p = _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    cat = load_catalog(p)
    assert isinstance(cat, Catalog)
    assert cat.name == "test_catalog"
    assert cat.schema_version == 1
    assert "container" in cat.entity_kinds
    assert cat.mode("truck_rail").arrival_routing["container"] == "handle_container"


def test_load_catalog_propagates_pydantic_errors(tmp_path: Path):
    bad_yaml = CATALOG_MINIMAL_YAML.replace("schema_version: 1", "schema_version: 2")
    p = _write(tmp_path / "catalog.yaml", bad_yaml)
    with pytest.raises(ValidationError):
        load_catalog(p)


def test_load_catalog_rejects_non_mapping_root(tmp_path: Path):
    p = _write(tmp_path / "catalog.yaml", "- just_a_list\n- of_strings\n")
    with pytest.raises(LoaderError) as exc:
        load_catalog(p)
    assert "must contain a YAML mapping" in str(exc.value)


def test_load_catalog_propagates_yaml_errors(tmp_path: Path):
    p = _write(tmp_path / "catalog.yaml", "meta: {schema_version: 1\nname: x\n")
    with pytest.raises(YamlLoaderError):
        load_catalog(p)


def test_load_catalog_with_python_module(tmp_path: Path):
    yaml = CATALOG_MINIMAL_YAML.replace(
        "name: test_catalog\n",
        "name: test_catalog\npython_module: altrios.lifts.workflow_engine.tests._loader_helpers\n",
    )
    p = _write(tmp_path / "catalog.yaml", yaml)
    cat = load_catalog(p)
    assert cat.python_module == "altrios.lifts.workflow_engine.tests._loader_helpers"


def test_load_catalog_rejects_unimportable_python_module(tmp_path: Path):
    yaml = CATALOG_MINIMAL_YAML.replace(
        "name: test_catalog\n",
        "name: test_catalog\npython_module: not.a.real.module.path.xyz\n",
    )
    p = _write(tmp_path / "catalog.yaml", yaml)
    with pytest.raises(LoaderError) as exc:
        load_catalog(p)
    assert "failed to import" in str(exc.value)


def test_load_catalog_resolves_partition_by_python(tmp_path: Path):
    yaml = """\
meta: {schema_version: 1}
name: with_resources
python_module: altrios.lifts.workflow_engine.tests._loader_helpers
entity_kinds:
  - name: container
modes:
  - name: m1
    arrival_routing: {container: g1}
    graphs:
      - name: g1
        entry: s0
        steps:
          - id: s0
            type: log
            params: {message: "hi"}
    resources:
      - name: stack
        kind: Store
        role: storage
        capacity: 100
        partition_by_python: test_loader.partition_by_kind
        init_items_python: test_loader.initial_items
"""
    p = _write(tmp_path / "catalog.yaml", yaml)
    cat = load_catalog(p)
    spec = cat.mode("m1").resource_specs[0]
    assert spec.name == "stack"
    assert spec.partition_by is not None
    assert callable(spec.partition_by)
    assert spec.init_items is not None


def test_load_catalog_unresolved_python_helper_raises(tmp_path: Path):
    yaml = """\
meta: {schema_version: 1}
name: bad_resources
entity_kinds:
  - name: x
modes:
  - name: m1
    arrival_routing: {x: g1}
    graphs:
      - name: g1
        entry: s0
        steps:
          - id: s0
            type: log
            params: {message: "hi"}
    resources:
      - name: r1
        kind: Resource
        role: equipment
        partition_by_python: never.registered
"""
    p = _write(tmp_path / "catalog.yaml", yaml)
    with pytest.raises(LoaderError) as exc:
        load_catalog(p)
    assert "no such callable" in str(exc.value) or "no callable" in str(exc.value).lower()


def test_load_catalog_converts_expressions(tmp_path: Path):
    """Brace-strings in step params become Expression objects after
    convert_expressions, and pydantic accepts them in params dicts."""
    from altrios.lifts.workflow_engine.expressions import Expression
    yaml = """\
meta: {schema_version: 1}
name: c
entity_kinds:
  - name: x
modes:
  - name: m1
    arrival_routing: {x: g1}
    graphs:
      - name: g1
        entry: s0
        steps:
          - id: s0
            type: timeout
            params:
              duration: "{entity.weight_t / 3.0}"
"""
    p = _write(tmp_path / "catalog.yaml", yaml)
    cat = load_catalog(p)
    step = cat.mode("m1").graphs["g1"].steps["s0"]
    assert isinstance(step.params["duration"], Expression)


def test_load_catalog_with_include(tmp_path: Path):
    """The yaml_loader's !include mechanism survives the full pipeline."""
    (tmp_path / "modes").mkdir()
    _write(
        tmp_path / "modes" / "truck_rail.yaml",
        """\
name: truck_rail
arrival_routing: {container: g1}
graphs:
  - name: g1
    entry: s0
    steps:
      - id: s0
        type: log
        params: {message: "from include"}
""",
    )
    _write(
        tmp_path / "catalog.yaml",
        """\
meta: {schema_version: 1}
name: c
entity_kinds:
  - name: container
modes:
  - !include modes/truck_rail.yaml
""",
    )
    cat = load_catalog(tmp_path / "catalog.yaml")
    assert cat.mode("truck_rail") is not None


# ---- load_site ------------------------------------------------------


def test_load_site_minimal(tmp_path: Path):
    cat_path = _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    site_path = _write(
        tmp_path / "site.yaml",
        SITE_MINIMAL_YAML.format(catalog_path="./catalog.yaml"),
    )
    site, cat = load_site(site_path)
    assert site.name == "test_site"
    assert site.modes == ["truck_rail"]
    assert cat.name == "test_catalog"


def test_load_site_absolute_catalog_path(tmp_path: Path):
    cat_path = _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    site_path = _write(
        tmp_path / "site.yaml",
        SITE_MINIMAL_YAML.format(catalog_path=str(cat_path)),
    )
    site, cat = load_site(site_path)
    assert cat.name == "test_catalog"


def test_load_site_rejects_mode_not_in_catalog(tmp_path: Path):
    cat_path = _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    site_yaml = """\
meta: {schema_version: 1}
name: bad
catalog: ./catalog.yaml
modes: [does_not_exist]
"""
    site_path = _write(tmp_path / "site.yaml", site_yaml)
    with pytest.raises(LoaderError) as exc:
        load_site(site_path)
    assert "not present in catalog" in str(exc.value)


def test_load_site_extends_merges_parent(tmp_path: Path):
    cat_path = _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    base_yaml = """\
meta: {schema_version: 1}
name: base
catalog: ./catalog.yaml
modes: [truck_rail]
config:
  crane_count: 4
  shift_hours: 8
seed: 1
"""
    override_yaml = """\
extends: ./base.yaml
config:
  crane_count: 8
seed: 100
"""
    _write(tmp_path / "base.yaml", base_yaml)
    site_path = _write(tmp_path / "override.yaml", override_yaml)

    site, cat = load_site(site_path)
    assert site.config["crane_count"] == 8       # overridden
    assert site.config["shift_hours"] == 8       # inherited
    assert site.seed == 100                       # overridden
    assert site.modes == ["truck_rail"]           # inherited


def test_load_site_extends_drops_extends_field(tmp_path: Path):
    """SiteModel must not see the ``extends`` key after merge."""
    _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    _write(
        tmp_path / "base.yaml",
        """\
meta: {schema_version: 1}
name: base
catalog: ./catalog.yaml
""",
    )
    site_path = _write(
        tmp_path / "override.yaml",
        "extends: ./base.yaml\nname: override\n",
    )
    site, cat = load_site(site_path)
    # No exception means ``extends`` did not reach the pydantic model
    # as an unknown extra (it would have, since extends_accepted_as_string
    # tested above ALLOWS it, so this test just checks the name override).
    assert site.name == "override"


def test_load_site_extends_resolves_relative_to_child(tmp_path: Path):
    """An extends path is relative to the CHILD file's directory."""
    (tmp_path / "scenarios").mkdir()
    _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    _write(
        tmp_path / "site.yaml",
        """\
meta: {schema_version: 1}
name: base
catalog: ./catalog.yaml
""",
    )
    # Child file lives in scenarios/, parent is ../site.yaml
    site_path = _write(
        tmp_path / "scenarios" / "busy.yaml",
        "extends: ../site.yaml\nname: busy\n",
    )
    site, cat = load_site(site_path)
    assert site.name == "busy"


def test_load_site_extends_rejects_non_mapping_parent(tmp_path: Path):
    _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    _write(tmp_path / "base.yaml", "- not_a_mapping\n")
    site_path = _write(
        tmp_path / "child.yaml",
        "extends: ./base.yaml\n",
    )
    with pytest.raises(LoaderError) as exc:
        load_site(site_path)
    assert "must contain a YAML mapping" in str(exc.value)


def test_load_site_extends_requires_string(tmp_path: Path):
    _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    site_path = _write(
        tmp_path / "site.yaml",
        "extends: [list, not, string]\n",
    )
    with pytest.raises(LoaderError) as exc:
        load_site(site_path)
    assert "extends must be a non-empty string" in str(exc.value)


def test_load_site_layout_block(tmp_path: Path):
    _write(tmp_path / "catalog.yaml", CATALOG_MINIMAL_YAML)
    site_yaml = """\
meta: {schema_version: 1}
name: with_layout
catalog: ./catalog.yaml
layout:
  nodes:
    berth_1: {x: 0, y: 0}
    stack_A: {x: 380, y: 50}
"""
    site_path = _write(tmp_path / "site.yaml", site_yaml)
    site, cat = load_site(site_path)
    assert site.layout is not None
    assert site.layout.nodes["stack_A"].x == 380


# ---- Catalog reference resolution -----------------------------------


def test_resolve_catalog_reference_filesystem(tmp_path: Path):
    """Catalog refs with path separators are filesystem paths."""
    from altrios.lifts.workflow_engine.loader import _resolve_catalog_reference
    # Absolute path returned as-is.
    abs_p = str(tmp_path / "x.yaml")
    assert _resolve_catalog_reference(abs_p) == abs_p
    # Relative path joined to site_dir.
    assert _resolve_catalog_reference(
        "./sub/catalog.yaml", site_dir=str(tmp_path)
    ).endswith("catalog.yaml")


def test_resolve_catalog_reference_yaml_extension():
    """A ref ending in .yaml is treated as a filesystem path even
    without a separator."""
    from altrios.lifts.workflow_engine.loader import _resolve_catalog_reference
    out = _resolve_catalog_reference("catalog.yaml", site_dir="/tmp")
    assert out.endswith("catalog.yaml")


def test_resolve_catalog_reference_unknown_dotted_module():
    """An unimportable dotted-module ref produces a clear error."""
    from altrios.lifts.workflow_engine.loader import _resolve_catalog_reference
    with pytest.raises(LoaderError) as exc:
        _resolve_catalog_reference("totally.not.a.module")
    assert "could not be imported" in str(exc.value)
