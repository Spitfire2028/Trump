"""The layer rule, made executable.

DurakFish's central architectural constraint is a single arrow:

    cards → game → information → ai → simulation

Higher layers may import lower ones; the reverse is forbidden. In
particular the rules engine must never learn about information sets,
agents or the driver, so that it stays reusable and so that "the AI cannot
cheat" is a structural fact rather than a convention.

Comments in a design document do not enforce anything. This module parses
the actual import graph, so an upward import fails the build the moment
somebody writes one.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "durakfish"

#: Lower number = lower layer. A module may import its own layer and below.
LAYERS: dict[str, int] = {
    "exceptions": 0,
    "cards": 1,
    "game": 2,
    "information": 3,
    "ai": 4,
    "simulation": 5,
}

#: The package root re-exports from everywhere; it sits above all layers.
ROOT_INIT = "__init__"


def python_files() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def layer_of(path: pathlib.Path) -> int | None:
    """Which layer a source file belongs to, or None for the package root."""
    relative = path.relative_to(SRC)
    head = relative.parts[0]
    if head.endswith(".py"):
        stem = head[:-3]
        if stem == ROOT_INIT:
            return None
        return LAYERS.get(stem)
    return LAYERS.get(head)


def imported_packages(path: pathlib.Path) -> set[str]:
    """First-party subpackages imported by a file, e.g. ``{'cards', 'game'}``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()

    def record(dotted: str) -> None:
        parts = dotted.split(".")
        if parts and parts[0] == "durakfish" and len(parts) > 1:
            found.add(parts[1])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            record(node.module)
    return found


def test_every_source_file_is_assigned_to_a_layer() -> None:
    """A new top-level module must be placed in the hierarchy deliberately."""
    unplaced = [
        p.relative_to(SRC)
        for p in python_files()
        if layer_of(p) is None and p.relative_to(SRC).as_posix() != "__init__.py"
    ]
    assert not unplaced, f"these files belong to no declared layer: {unplaced}"


def test_no_module_imports_from_a_higher_layer() -> None:
    violations: list[str] = []
    for path in python_files():
        layer = layer_of(path)
        if layer is None:
            continue  # the package root may import anything
        for package in imported_packages(path):
            other = LAYERS.get(package)
            if other is None:
                violations.append(
                    f"{path.relative_to(SRC)} imports unknown package {package!r}"
                )
            elif other > layer:
                violations.append(
                    f"{path.relative_to(SRC)} (layer {layer}) imports "
                    f"{package} (layer {other}) — dependencies point downward only"
                )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("forbidden", ["information", "ai", "simulation"])
def test_the_rules_engine_stays_independent_of_everything_above_it(
    forbidden: str,
) -> None:
    """The headline guarantee, stated one package at a time."""
    for path in python_files():
        if path.relative_to(SRC).parts[0] != "game":
            continue
        assert forbidden not in imported_packages(path), (
            f"{path.relative_to(SRC)} imports {forbidden}; the rules engine "
            f"must not depend on layers above it"
        )


def test_the_cards_layer_depends_on_nothing_but_exceptions() -> None:
    for path in python_files():
        if path.relative_to(SRC).parts[0] != "cards":
            continue
        assert imported_packages(path) <= {"cards", "exceptions"}


def test_exceptions_is_a_leaf() -> None:
    assert imported_packages(SRC / "exceptions.py") == set()


def test_the_information_layer_does_not_depend_on_agents_or_the_driver() -> None:
    """Redaction must be usable without anything that consumes it."""
    for path in python_files():
        if path.relative_to(SRC).parts[0] != "information":
            continue
        assert not imported_packages(path) & {"ai", "simulation"}


def test_layers_are_importable_in_isolation() -> None:
    """Importing a lower layer must not drag a higher one in."""
    import subprocess
    import sys

    code = (
        "import sys; import durakfish.game; "
        "loaded = set(sys.modules); "
        "bad = [m for m in loaded if m.startswith('durakfish.') and "
        "m.split('.')[1] in ('information', 'ai', 'simulation')]; "
        "print(bad); assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(SRC.parent.parent),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
