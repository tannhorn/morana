"""Tests for fixed-source objects."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from morana import CellSource, MaterialSource, UniformSource


def test_uniform_source_broadcasts_to_selected_layer_active_cells() -> None:
    """Uniform values use explicit axial selection and compact active IDs."""
    source = UniformSource([1.0, 2.0])

    values = source.values(1, {0: "medium", 1: "reflector"})

    assert values.shape == (2, 2)
    np.testing.assert_allclose(values[:, 0], [1.0, 2.0])
    assert not values.flags.writeable


def test_sources_share_explicit_axial_values_interface() -> None:
    """Every source type evaluates one selected axial layer."""
    sources = [
        UniformSource([1.0]),
        MaterialSource({"medium": [2.0]}),
        CellSource(([[3.0]],)),
    ]

    np.testing.assert_allclose(sources[0].values(0, {0: "medium"}), [[1.0]])
    np.testing.assert_allclose(sources[1].values(0, {0: "medium"}), [[2.0]])
    np.testing.assert_allclose(sources[2].values(0, {0: "medium"}), [[3.0]])


def test_empty_layer_returns_valid_group_by_zero_array() -> None:
    """Sources support a selected axial layer without active cells."""
    sources = [
        UniformSource([1.0, 2.0]),
        MaterialSource({"medium": [3.0, 4.0]}),
        CellSource(([[5.0], [6.0]], [[], []])),
    ]

    for source in sources:
        values = source.values(1, {})
        assert values.shape == (2, 0)
        assert not values.flags.writeable


def test_cell_source_owns_ragged_layer_arrays() -> None:
    """Explicit sources own immutable arrays with independent cell counts."""
    lower = np.array([[1.0, 2.0], [3.0, 4.0]])
    upper = np.array([[5.0], [6.0]])
    source = CellSource((lower, upper))
    lower[0, 0] = 9.0
    upper[0, 0] = 9.0

    assert len(source.layers) == 2
    np.testing.assert_allclose(source.layers[0], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(source.layers[1], [[5.0], [6.0]])
    assert not source.layers[0].flags.writeable
    assert not source.layers[1].flags.writeable
    np.testing.assert_allclose(source.values(1, {0: "medium"}), [[5.0], [6.0]])


def test_cell_source_checks_group_counts_and_selected_shape() -> None:
    """Layer arrays have one group structure and match selected active cells."""
    with pytest.raises(ValueError, match="same group count"):
        CellSource(([[1.0]], [[2.0], [3.0]]))
    with pytest.raises(ValueError, match="shape"):
        CellSource(([1.0, 2.0],))

    source = CellSource(([[1.0]],))
    with pytest.raises(ValueError, match="column count"):
        source.values(0, {0: "medium", 1: "reflector"})


def test_material_source_uses_selected_layer_material_names() -> None:
    """Material values follow the selected layer's compact material mapping."""
    source = MaterialSource({"medium": [1.0], "reflector": [0.0]})

    values = source.values(2, {0: "medium", 1: "reflector"})

    np.testing.assert_allclose(values, [[1.0, 0.0]])
    assert not values.flags.writeable


@pytest.mark.parametrize("material", [None, 1, ""])
def test_material_source_requires_nonempty_string_material_names(
    material: object,
) -> None:
    """Material source keys are public material identities at construction."""
    with pytest.raises((TypeError, ValueError), match="material name"):
        MaterialSource({material: [1.0]})  # type: ignore[dict-item]


def test_material_source_rejects_missing_material() -> None:
    """Bulk material source evaluation rejects missing material values."""
    source = MaterialSource({"medium": [1.0]})

    with pytest.raises(ValueError, match="reflector"):
        source.values(0, {0: "medium", 1: "reflector"})


def test_source_values_require_compact_active_ids_and_valid_axial_index() -> None:
    """Source evaluation checks compact IDs and explicit axial indices."""
    source = UniformSource([1.0])

    with pytest.raises(ValueError, match="slice-local active IDs"):
        source.values(0, {1: "medium"})
    with pytest.raises(ValueError, match="axial_index"):
        source.values(-1, {0: "medium"})


@pytest.mark.parametrize(
    "source",
    [
        UniformSource([1.0]),
        MaterialSource({"medium": [1.0]}),
        CellSource(([[1.0]],)),
    ],
)
@pytest.mark.parametrize(
    ("material_by_active_id", "exception"),
    [
        (None, TypeError),
        ({False: "medium"}, TypeError),
        ({0.0: "medium"}, TypeError),
        ({-1: "medium"}, ValueError),
        ({0: None}, TypeError),
        ({0: ""}, ValueError),
    ],
)
def test_source_values_check_mapping_type_ids_and_material_names(
    source: UniformSource | MaterialSource | CellSource,
    material_by_active_id: object,
    exception: type[Exception],
) -> None:
    """Every built-in source validates its shared layer-context contract."""
    with pytest.raises(exception):
        source.values(0, material_by_active_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "source",
    [
        UniformSource([1.0]),
        MaterialSource({"medium": [1.0]}),
        CellSource(([[1.0]],)),
    ],
)
@pytest.mark.parametrize("axial_index", [False, 0.0, "0"])
def test_source_values_reject_noninteger_axial_indices(
    source: UniformSource | MaterialSource | CellSource, axial_index: object
) -> None:
    """Axial layer selection accepts only non-Boolean integer indices."""
    with pytest.raises(TypeError, match="axial_index"):
        source.values(axial_index, {0: "medium"})  # type: ignore[arg-type]


def test_material_source_requires_consistent_group_count() -> None:
    """Material source vectors must all have the same group count."""
    with pytest.raises(ValueError):
        MaterialSource({"medium": [1.0], "reflector": [0.0, 0.0]})


def test_sources_are_immutable() -> None:
    """Validated source definitions must not permit replacement or map edits."""
    uniform = UniformSource([1.0])
    material = MaterialSource({"medium": [2.0]})
    cell = CellSource(([[3.0]],))

    with pytest.raises(FrozenInstanceError):
        uniform.strength = np.array([4.0])
    with pytest.raises(FrozenInstanceError):
        material.values_by_material = {"medium": np.array([4.0])}
    with pytest.raises(TypeError):
        material.values_by_material["medium"] = np.array([4.0])
    with pytest.raises(FrozenInstanceError):
        cell.layers = (np.array([[4.0]]),)
