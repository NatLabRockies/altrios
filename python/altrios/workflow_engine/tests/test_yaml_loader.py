"""Tests for the YAML loader — ``!include`` resolution
and cycle detection."""
from __future__ import annotations

from pathlib import Path

import pytest

from altrios.workflow_engine.yaml_loader import (
    YamlLoaderError,
    load_yaml_file,
    load_yaml_string,
)


def test_load_yaml_string_basic():
    data = load_yaml_string("a: 1\nb: [2, 3]\n")
    assert data == {"a": 1, "b": [2, 3]}


def test_load_yaml_file_basic(tmp_path: Path):
    p = tmp_path / "doc.yaml"
    p.write_text("greeting: hello\nlist: [1, 2, 3]\n", encoding="utf-8")
    assert load_yaml_file(p) == {"greeting": "hello", "list": [1, 2, 3]}


def test_load_yaml_string_invalid_raises():
    with pytest.raises(YamlLoaderError, match="parse error"):
        load_yaml_string("not: valid: yaml: :")


def test_include_resolves_relative_to_including_file(tmp_path: Path):
    common = tmp_path / "common"
    common.mkdir()
    (common / "shared.yaml").write_text(
        "shared_value: 42\nitems: [a, b, c]\n", encoding="utf-8"
    )
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "name: parent\nincluded: !include common/shared.yaml\n", encoding="utf-8"
    )
    data = load_yaml_file(parent)
    assert data["name"] == "parent"
    assert data["included"] == {"shared_value": 42, "items": ["a", "b", "c"]}


def test_include_can_be_absolute(tmp_path: Path):
    target = tmp_path / "target.yaml"
    target.write_text("x: 1\n", encoding="utf-8")
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        f"sub: !include {target.as_posix()}\n", encoding="utf-8"
    )
    data = load_yaml_file(parent)
    assert data == {"sub": {"x": 1}}


def test_include_recursive_resolves_relative_to_inner_file(tmp_path: Path):
    """An included file's own !include resolves relative to the
    included file's directory, not the original parent."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "leaf.yaml").write_text("leaf_value: 99\n", encoding="utf-8")
    (sub_dir / "mid.yaml").write_text(
        "mid: !include leaf.yaml\n", encoding="utf-8"
    )
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "tree: !include sub/mid.yaml\n", encoding="utf-8"
    )
    assert load_yaml_file(parent) == {
        "tree": {"mid": {"leaf_value": 99}}
    }


def test_include_missing_file_raises_with_context(tmp_path: Path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("sub: !include does_not_exist.yaml\n", encoding="utf-8")
    with pytest.raises(YamlLoaderError) as excinfo:
        load_yaml_file(parent)
    msg = str(excinfo.value)
    assert "does_not_exist.yaml" in msg
    assert "parent.yaml" in msg  # error reports who tried to include it


def test_include_cycle_detected(tmp_path: Path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("from_a: !include b.yaml\n", encoding="utf-8")
    b.write_text("from_b: !include a.yaml\n", encoding="utf-8")
    with pytest.raises(YamlLoaderError, match="Cyclic"):
        load_yaml_file(a)


def test_include_self_cycle_detected(tmp_path: Path):
    a = tmp_path / "a.yaml"
    a.write_text("loop: !include a.yaml\n", encoding="utf-8")
    with pytest.raises(YamlLoaderError, match="Cyclic"):
        load_yaml_file(a)


def test_include_non_scalar_path_rejected(tmp_path: Path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("bad: !include [a, b]\n", encoding="utf-8")
    with pytest.raises(YamlLoaderError, match="scalar"):
        load_yaml_file(parent)


def test_include_empty_path_rejected(tmp_path: Path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("bad: !include ''\n", encoding="utf-8")
    with pytest.raises(YamlLoaderError, match="non-empty"):
        load_yaml_file(parent)


def test_safe_loader_rejects_python_object_construction(tmp_path: Path):
    """Defense in depth: the loader must not allow !!python/object tags."""
    p = tmp_path / "evil.yaml"
    p.write_text(
        "obj: !!python/object/apply:os.system ['echo pwned']\n",
        encoding="utf-8",
    )
    with pytest.raises(YamlLoaderError, match="parse error"):
        load_yaml_file(p)


def test_load_string_with_base_dir_for_includes(tmp_path: Path):
    (tmp_path / "shared.yaml").write_text("x: 1\n", encoding="utf-8")
    data = load_yaml_string("sub: !include shared.yaml\n", base_dir=tmp_path)
    assert data == {"sub": {"x": 1}}


def test_empty_file_returns_none(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_yaml_file(p) is None
