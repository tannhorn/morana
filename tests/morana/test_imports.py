"""Import tests for the current public package modules."""

import importlib
import inspect

import morana
from morana import operators
from morana.solvers import finite_volume

MODULES = [
    "morana",
    "morana.boundary",
    "morana.configuration",
    "morana.execution_reports",
    "morana.material_mesh",
    "morana.materials",
    "morana.normalization",
    "morana.operators",
    "morana.hex_planar_mesh",
    "morana.results",
    "morana.solvers",
    "morana.solvers.finite_volume",
    "morana.sources",
]


def test_modules_import() -> None:
    """All package modules should be importable."""
    for module in MODULES:
        importlib.import_module(module)


def test_exported_callables_have_docstrings() -> None:
    """Every supported class or function should document its public purpose."""
    exports = [
        *(getattr(morana, name) for name in morana.__all__),
        *(getattr(operators, name) for name in operators.__all__),
        *(getattr(finite_volume, name) for name in finite_volume.__all__),
    ]
    undocumented = [
        value.__name__
        for value in exports
        if (inspect.isclass(value) or inspect.isfunction(value))
        and inspect.getdoc(value) is None
    ]
    assert undocumented == []
