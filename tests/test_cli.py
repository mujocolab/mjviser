"""Tests for the CLI path resolution logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from mjviser.__main__ import _resolve_path


@pytest.fixture
def model_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Create a directory tree with some XML files and cd into it."""
  (tmp_path / "a" / "humanoid").mkdir(parents=True)
  (tmp_path / "b").mkdir()
  (tmp_path / "a" / "humanoid" / "humanoid.xml").write_text("<mujoco/>")
  (tmp_path / "a" / "ant.xml").write_text("<mujoco/>")
  (tmp_path / "b" / "humanoid_100.xml").write_text("<mujoco/>")
  monkeypatch.chdir(tmp_path)
  return tmp_path


def test_exact_path(model_tree: Path) -> None:
  result = _resolve_path(str(model_tree / "a" / "ant.xml"))
  assert result == model_tree / "a" / "ant.xml"


def test_relative_path(model_tree: Path) -> None:
  result = _resolve_path("a/ant.xml")
  assert result == Path("a/ant.xml")


def test_single_match(model_tree: Path) -> None:
  result = _resolve_path("ant")
  assert result.name == "ant.xml"


def test_multiple_matches_picks_first_by_default(
  model_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr("builtins.input", lambda _: "")
  result = _resolve_path("humanoid")
  assert result.name.startswith("humanoid")


def test_multiple_matches_user_selects(
  model_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr("builtins.input", lambda _: "2")
  result = _resolve_path("humanoid")
  assert result.suffix == ".xml"


def test_no_matches(model_tree: Path) -> None:
  with pytest.raises(SystemExit):
    _resolve_path("nonexistent")


def test_glob_passthrough(model_tree: Path) -> None:
  """If the query contains a wildcard, use it as a glob directly."""
  result = _resolve_path("**/ant*")
  assert result.name == "ant.xml"


def test_directory_name_does_not_short_circuit(
  model_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A directory named 'humanoid' should not be returned as-is."""
  monkeypatch.setattr("builtins.input", lambda _: "1")
  result = _resolve_path("humanoid")
  assert result.is_file() or result.suffix == ".xml"


def test_directory_path_searches_inside(
  model_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Passing a directory path should find XMLs inside it."""
  result = _resolve_path(str(model_tree / "a" / "humanoid"))
  assert result.name == "humanoid.xml"
