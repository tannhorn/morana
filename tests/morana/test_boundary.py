"""Tests for fluent boundary-condition assignment and resolution."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from morana import (
    BoundaryCondition,
    BoundaryConditionSet,
    BoundarySelector,
    HexPlanarMesh,
    MaterialMesh,
    MaterialSlice,
)
from morana.boundary import BoundaryAssignment, _resolve_boundary_condition


def test_fluent_boundary_assignments_cover_each_supported_scope() -> None:
    """Conditions should own the user-facing selector construction grammar."""
    boundaries = BoundaryConditionSet(
        BoundaryCondition.reflective().globally(),
        BoundaryCondition.vacuum().on_outer(),
        BoundaryCondition.dirichlet([1.0, 0.0]).on_radial(),
        BoundaryCondition.reflective().on_bottom(),
        BoundaryCondition.vacuum().on_top(),
        BoundaryCondition.reflective().on_excluded(kind="inactive", direction="v-"),
    )
    assert [entry.selector.scope for entry in boundaries.assignments] == [
        "global",
        "outer",
        "radial",
        "bottom",
        "top",
        "to_excluded",
    ]


def test_boundary_selector_factories_define_vocabulary() -> None:
    """Topology selectors retain the locked vocabulary internally."""
    assert BoundarySelector.everywhere().scope == "global"
    assert BoundarySelector.to_excluded(key="0").excluded_key == "0"
    assert BoundarySelector.bottom().scope == "bottom"
    assert BoundarySelector.top().scope == "top"
    assert BoundarySelector.to_excluded(direction="bottom").direction == "bottom"
    assert BoundarySelector.to_excluded(direction="top").direction == "top"


def test_excluded_boundary_selector_rejects_non_string_key() -> None:
    """Excluded-interface selection should require a string material key."""
    with pytest.raises(TypeError, match="material key must be a string"):
        BoundarySelector.to_excluded(key=0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (("scope", 0), ("direction", 0), ("excluded_kind", 0)),
)
def test_boundary_selector_requires_string_fields(field: str, value: object) -> None:
    """Selector labels must not defer type errors to boundary resolution."""
    arguments = {"scope": "to_excluded", field: value}
    with pytest.raises(TypeError):
        BoundarySelector(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ("excluded_key", "excluded_kind"))
def test_boundary_selector_checks_filter_types_before_scope_rules(field: str) -> None:
    """Wrong filter types must not be reported as a scope incompatibility."""
    with pytest.raises(TypeError):
        BoundarySelector(scope="outer", **{field: 0})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ("excluded_key", "excluded_kind"))
@pytest.mark.parametrize("value", ("   ", "void\nport", "void\u200bport"))
def test_boundary_selector_requires_human_readable_filters(
    field: str, value: str
) -> None:
    """Excluded filters must use the shared legible-identifier contract."""
    with pytest.raises(ValueError):
        BoundarySelector(scope="to_excluded", **{field: value})


def test_boundary_assignment_requires_checked_value_objects() -> None:
    """Direct assignments must preserve their typed topology/physics contract."""
    with pytest.raises(TypeError, match="selector"):
        BoundaryAssignment(0, BoundaryCondition.vacuum())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="condition"):
        BoundaryAssignment(BoundarySelector.outer(), 0)  # type: ignore[arg-type]


def test_boundary_set_requires_boundary_assignments() -> None:
    """Boundary-set construction and extension must reject wrong object types."""
    with pytest.raises(TypeError, match="BoundaryAssignment"):
        BoundaryConditionSet(0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BoundaryAssignment"):
        BoundaryConditionSet().with_assignment(0)  # type: ignore[arg-type]


def test_boundary_set_resolution_requires_domain_face() -> None:
    """Boundary resolution must not accept an unvalidated face lookalike."""
    boundaries = BoundaryConditionSet(BoundaryCondition.vacuum().globally())
    with pytest.raises(TypeError, match="DomainFace"):
        boundaries.resolve(0)  # type: ignore[arg-type]


def test_boundary_coverage_requires_material_mesh() -> None:
    """Coverage checks must not accept an unvalidated material-mesh lookalike."""
    with pytest.raises(TypeError, match="MaterialMesh"):
        BoundaryConditionSet().check_coverage(0)  # type: ignore[arg-type]


def test_prescribed_vectors_are_owned_and_zero_dirichlet_is_homogeneous() -> None:
    """Prescribed data must be immutable vectors; zero remains group-independent."""
    values = np.array([1.0, 0.0])
    boundary = BoundaryCondition.dirichlet(values)
    values[0] = 3.0
    np.testing.assert_allclose(boundary.flux, [1.0, 0.0])
    assert boundary.flux.flags.writeable is False
    assert BoundaryCondition.zero_dirichlet().flux is None


def test_direct_boundary_construction_owns_prescribed_vectors() -> None:
    """Direct construction enforces the same vector-ownership invariant."""
    values = np.array([2.0])
    boundary = BoundaryCondition(kind="incoming_current", current=values)
    values[0] = 4.0
    np.testing.assert_allclose(boundary.current, [2.0])
    assert boundary.current.flags.writeable is False


def test_boundary_condition_requires_string_kind() -> None:
    """Direct boundary construction should check its tagged-union discriminator."""
    with pytest.raises(TypeError, match="kind must be a string"):
        BoundaryCondition(kind=0)  # type: ignore[arg-type]


def test_group_resolution_uses_complete_affine_source_vectors() -> None:
    """Solver-facing boundary data has no scalar/current special case."""
    incoming = _resolve_boundary_condition(
        BoundaryCondition.incoming_current([2.0, 3.0]), 2
    )
    partial = _resolve_boundary_condition(
        BoundaryCondition.partial_current_return(0.5, current=[3.0, 6.0]), 2
    )
    homogeneous = _resolve_boundary_condition(BoundaryCondition.zero_dirichlet(), 2)
    assert incoming.alpha == pytest.approx(0.5)
    np.testing.assert_allclose(incoming.source, [4.0, 6.0])
    assert partial.alpha == pytest.approx(1.0 / 6.0)
    np.testing.assert_allclose(partial.source, [4.0, 8.0])
    np.testing.assert_allclose(homogeneous.flux, [0.0, 0.0])
    assert incoming.source.flags.writeable is False


def test_group_resolution_rejects_wrong_prescribed_vector_length() -> None:
    """Boundary spectra are checked once against configuration group count."""
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        _resolve_boundary_condition(BoundaryCondition.dirichlet([1.0]), 2)


def test_group_resolution_owns_prescribed_vectors_separately() -> None:
    """Resolved data must not share mutable storage with boundary inputs."""
    condition = BoundaryCondition.dirichlet([1.0])
    resolved = _resolve_boundary_condition(condition, 1)

    condition.flux.setflags(write=True)
    condition.flux[0] = 2.0

    np.testing.assert_allclose(resolved.flux, [1.0])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BoundaryCondition.dirichlet(1.0),
        lambda: BoundaryCondition.incoming_current([1.0, -1.0]),
        lambda: BoundaryCondition.incoming_current([np.inf]),
        lambda: BoundaryCondition.partial_current_return(0.5, current=[]),
        lambda: BoundaryCondition.partial_current_return(-0.1),
        lambda: BoundaryCondition.partial_current_return(1.1),
        lambda: BoundaryCondition.partial_current_return(True),
        lambda: BoundaryCondition.partial_current_return("0.5"),
        lambda: BoundaryCondition.robin(-0.1),
        lambda: BoundaryCondition.robin(True),
        lambda: BoundaryCondition.robin("0.5"),
    ],
)
def test_boundary_checking_rejects_invalid_data(factory) -> None:
    """Prescribed spectra and scalar response coefficients are checked."""
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_boundary_set_rejects_duplicate_selector_specificity() -> None:
    """Duplicate assignments at one specificity are invalid immediately."""
    with pytest.raises(ValueError, match="duplicate boundary assignment"):
        BoundaryConditionSet(
            BoundaryCondition.reflective().on_radial(),
            BoundaryCondition.vacuum().on_radial(),
        )


def test_boundary_set_is_immutable() -> None:
    """Validated boundary assignment sets must not permit replacement."""
    boundaries = BoundaryConditionSet(BoundaryCondition.reflective().globally())

    with pytest.raises(FrozenInstanceError):
        boundaries.assignments = ()


def test_resolution_uses_excluded_direction_then_global_precedence() -> None:
    """Excluded-face direction rules should override fallback conditions."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice.from_openmc_rings(mesh, [["0"] * 6, ["medium"]], height=1.0),),
    )
    boundaries = BoundaryConditionSet(
        BoundaryCondition.vacuum().globally(),
        BoundaryCondition.reflective().on_excluded(key="0", direction="x+"),
    )
    assert boundaries.resolve(material_mesh.face(0, 0, "x+")).kind == "reflective"


def test_axial_outer_precedence_uses_specific_face_then_outer_then_global() -> None:
    """Physical bottom/top rules refine outer and global fallback conditions."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
            MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
        )
    )
    boundaries = BoundaryConditionSet(
        BoundaryCondition.reflective().globally(),
        BoundaryCondition.vacuum().on_outer(),
        BoundaryCondition.dirichlet([1.0]).on_bottom(),
    )

    assert boundaries.resolve(material_mesh.face(0, 0, "bottom")).kind == "dirichlet"
    assert boundaries.resolve(material_mesh.face(1, 0, "top")).kind == "vacuum"
    assert boundaries.resolve(material_mesh.face(0, 0, "x+")).kind == "vacuum"


def test_axial_excluded_direction_rule_refines_global_fallback() -> None:
    """Axial excluded interfaces accept directional excluded-region selectors."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
            MaterialSlice.from_openmc_rings(mesh, [["0"]], height=1.0),
        )
    )
    boundaries = BoundaryConditionSet(
        BoundaryCondition.vacuum().globally(),
        BoundaryCondition.reflective().on_excluded(key="0", direction="top"),
    )

    assert boundaries.resolve(material_mesh.face(0, 0, "top")).kind == "reflective"


def test_coverage_requires_physical_axial_boundary_conditions() -> None:
    """Every exposed axial face must resolve before finite-volume assembly."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
            MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
        )
    )
    incomplete = BoundaryConditionSet(
        BoundaryCondition.reflective().on_radial(),
        BoundaryCondition.reflective().on_bottom(),
    )

    with pytest.raises(ValueError, match="direction='top'"):
        incomplete.check_coverage(material_mesh)

    BoundaryConditionSet(
        BoundaryCondition.reflective().on_radial(),
        BoundaryCondition.reflective().on_bottom(),
        BoundaryCondition.reflective().on_top(),
    ).check_coverage(material_mesh)
