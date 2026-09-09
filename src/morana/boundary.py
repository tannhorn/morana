"""Boundary-condition models for hexagonal diffusion problems.

Boundary physics is represented separately from topology selectors. Fluent
condition assignments resolve independently for every exposed radial, axial,
or excluded-region face.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from morana._arrays import (
    freeze_owned_float_array,
    readonly_float_array,
    require_finite_nonnegative_array,
)
from morana.material_mesh import (
    AXIAL_DIRECTION_LABELS,
    DOMAIN_FACE_KIND_OUTER,
    DOMAIN_FACE_KIND_TO_EXCLUDED,
    DomainFace,
    EXPOSED_DOMAIN_FACE_KINDS,
    MaterialMesh,
)
from morana._validation import (
    require_finite_real,
    require_human_readable_identifier,
    require_nonempty_string,
    require_optional_string,
    require_positive_integer,
)
from morana.hex_planar_mesh import HEX_DIRECTION_LABELS

_VALID_KINDS = {
    "reflective",
    "vacuum",
    "dirichlet",
    "robin",
    "partial_current",
    "incoming_current",
}
_BOUNDARY_SELECTOR_SCOPES = frozenset(
    {"global", "outer", "radial", "to_excluded", "bottom", "top"}
)
_DIRECTIONAL_SCOPES = {"to_excluded"}
_BOUNDARY_DIRECTION_LABELS = frozenset((*HEX_DIRECTION_LABELS, *AXIAL_DIRECTION_LABELS))


@dataclass(frozen=True)
class BoundaryCondition:
    """Represent immutable diffusion boundary physics for selected faces.

    Prefer the named constructors over direct construction; they make the
    physical convention explicit. Attach the resulting condition to exposed
    faces with ``globally()`` or an ``on_*()`` method. Boundary vectors are
    copied into read-only arrays and are ordered fast to thermal.

    Parameters
    ----------
    kind
        Nonempty direct-construction boundary kind. Supported values are
        ``"reflective"``, ``"vacuum"``, ``"dirichlet"``, ``"robin"``,
        ``"partial_current"``, and ``"incoming_current"``. The selected kind
        determines which remaining fields are accepted.
    flux
        Nonempty, finite, nonnegative one-dimensional prescribed face-flux
        spectrum in ``n / cm^2 / s``. Accepted only for ``"dirichlet"``;
        ``None`` denotes zero flux. Its length must equal the problem energy
        group count when the boundary is resolved.
    alpha
        Finite, nonnegative, dimensionless outward net-current-over-face-flux
        coefficient for ``"robin"`` only.
    beta
        Finite, dimensionless returned-to-outgoing partial-current ratio in
        ``[0, 1]`` for ``"partial_current"`` only.
    current
        Nonempty, finite, nonnegative one-dimensional imposed incoming
        partial-current spectrum in ``n / cm^2 / s``. It is accepted by
        ``"partial_current"`` and ``"incoming_current"`` only, and its length
        must equal the problem energy group count when resolved. ``None`` is
        allowed only for ``"partial_current"`` and denotes no imposed
        incidence.

    Raises
    ------
    TypeError
        If ``kind`` is not a string.
    ValueError
        If ``kind`` is empty or unsupported, or its remaining fields do not
        satisfy the selected boundary convention.

    Notes
    -----
    Reflective, vacuum, zero-Dirichlet, and Robin conditions are homogeneous.
    A Dirichlet condition is homogeneous when its resolved flux vector is
    zero. A partial-current-return condition is homogeneous when its optional
    current is absent or resolves to zero; an incoming-current condition is
    homogeneous only when its current resolves to zero. Only homogeneous
    conditions are accepted by ``morana.solvers.finite_volume.solve_keff()``. See the
    modeling workflow and exposed-boundary theory for selector precedence and
    finite-volume equations.
    """

    kind: str
    flux: np.ndarray | None = None
    alpha: float | None = None
    beta: float | None = None
    current: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Check boundary data."""
        require_nonempty_string("boundary condition kind", self.kind)
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unknown boundary condition kind: {self.kind!r}")
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if value is not None:
                object.__setattr__(self, name, require_finite_real(name, value))
        for name, value in (("flux", self.flux), ("current", self.current)):
            if value is not None:
                object.__setattr__(self, name, _owned_boundary_vector(name, value))

        checkers = {
            "reflective": self._check_parameterless,
            "vacuum": self._check_parameterless,
            "dirichlet": self._check_dirichlet,
            "robin": self._check_robin,
            "partial_current": self._check_partial_current,
            "incoming_current": self._check_incoming_current,
        }
        checkers[self.kind]()

    @classmethod
    def reflective(cls) -> "BoundaryCondition":
        """Return a homogeneous reflective boundary with ``J_out = 0``.

        This condition contributes neither leakage nor a boundary source and
        is equivalent to ``partial_current_return(beta=1.0)`` without imposed
        incidence.
        """
        return cls(kind="reflective")

    @classmethod
    def vacuum(cls) -> "BoundaryCondition":
        """Return a homogeneous Marshak-vacuum boundary.

        Morana uses ``J_out = phi_b / 2`` at the physical face. It is
        equivalent to ``robin(alpha=0.5)`` and to
        ``partial_current_return(beta=0.0)`` without imposed incidence.
        """
        return cls(kind="vacuum")

    @classmethod
    def zero_dirichlet(cls) -> "BoundaryCondition":
        """Return a homogeneous Dirichlet boundary with ``phi_b = 0``.

        This is not the Marshak-vacuum approximation: it prescribes zero
        physical face flux and is the limiting case of a Robin coefficient
        tending to infinity.
        """
        return cls(kind="dirichlet")

    @classmethod
    def dirichlet(cls, flux: np.ndarray | list[float]) -> "BoundaryCondition":
        """Return a prescribed group-resolved face-flux boundary condition.

        Parameters
        ----------
        flux
            Nonempty, finite, nonnegative one-dimensional face-flux spectrum
            in ``n / cm^2 / s``, ordered fast to thermal. Its length must
            equal the problem energy group count when the boundary is
            resolved. The values are copied into a read-only array.

        Notes
        -----
        A nonzero spectrum contributes an inhomogeneous boundary source and
        is therefore unavailable to ``morana.solvers.finite_volume.solve_keff()``. Use
        ``zero_dirichlet()`` for a homogeneous zero-flux boundary.
        """
        return cls(kind="dirichlet", flux=flux)

    @classmethod
    def robin(cls, alpha: float) -> "BoundaryCondition":
        """Return a homogeneous scalar Robin-current boundary.

        Parameters
        ----------
        alpha
            Finite, nonnegative, dimensionless coefficient in
            ``J_out = alpha * phi_b``. ``0`` is reflective and ``0.5`` is the
            Marshak-vacuum value. Values above ``0.5`` are mathematical sinks,
            not passive physical albedos.
        """
        return cls(kind="robin", alpha=alpha)

    @classmethod
    def partial_current_return(
        cls,
        beta: float,
        current: np.ndarray | list[float] | None = None,
    ) -> "BoundaryCondition":
        """Return a partial-current-return boundary with optional incidence.

        Parameters
        ----------
        beta
            Finite returned-to-outgoing partial-current ratio in ``[0, 1]``.
            ``0`` is Marshak vacuum and ``1`` is reflective when ``current``
            is omitted or zero.
        current
            Optional nonempty, finite, nonnegative group-resolved imposed
            incoming partial current in ``n / cm^2 / s``, ordered fast to
            thermal. Its length must equal the problem energy group count when
            resolved. The values are copied into a read-only array.

        Notes
        -----
        Morana applies ``j_minus = beta * j_plus + current`` independently in
        each group. A nonzero ``current`` creates an inhomogeneous boundary
        source and is unavailable to ``morana.solvers.finite_volume.solve_keff()``.
        """
        return cls(
            kind="partial_current",
            beta=beta,
            current=current,
        )

    @classmethod
    def incoming_current(cls, current: np.ndarray | list[float]) -> "BoundaryCondition":
        """Return a pure group-resolved incoming partial-current boundary.

        Parameters
        ----------
        current
            Nonempty, finite, nonnegative imposed incoming partial-current
            spectrum in ``n / cm^2 / s``, ordered fast to thermal. Its length
            must equal the problem energy group count when the boundary is
            resolved. The values are copied into a read-only array.

        Notes
        -----
        This is a Marshak-vacuum response with independent incidence:
        ``J_out = phi_b / 2 - 2 * current``. A nonzero spectrum creates an
        inhomogeneous boundary source and is unavailable to
        ``morana.solvers.finite_volume.solve_keff()``.
        """
        return cls(kind="incoming_current", current=current)

    def _require_only(
        self,
        *,
        flux: bool = False,
        alpha: bool = False,
        beta: bool = False,
        current: bool = False,
    ) -> None:
        """Reject fields that do not belong to this boundary representation."""
        if not flux and self.flux is not None:
            raise ValueError(f"{self.kind} boundary conditions do not accept flux")
        if not alpha and self.alpha is not None:
            raise ValueError(f"{self.kind} boundary conditions do not accept alpha")
        if not beta and self.beta is not None:
            raise ValueError(f"{self.kind} boundary conditions do not accept beta")
        if not current and self.current is not None:
            raise ValueError(f"{self.kind} boundary conditions do not accept current")

    def _check_parameterless(self) -> None:
        """Check a boundary preset with no configurable data."""
        self._require_only()

    def _check_dirichlet(self) -> None:
        """Check prescribed scalar-flux data."""
        self._require_only(flux=True)

    def _check_robin(self) -> None:
        """Check scalar Robin-current data."""
        if self.alpha is None:
            raise ValueError("robin boundary conditions require alpha")
        if self.alpha < 0.0:
            raise ValueError("robin alpha must be non-negative")
        self._require_only(alpha=True)

    def _check_partial_current(self) -> None:
        """Check partial-current return and optional incident current."""
        if self.beta is None:
            raise ValueError("partial-current boundaries require beta")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("partial-current beta must be between zero and one")
        self._require_only(beta=True, current=True)

    def _check_incoming_current(self) -> None:
        """Check pure imposed incoming partial-current data."""
        if self.current is None:
            raise ValueError("incoming-current boundaries require current")
        self._require_only(current=True)

    def globally(self) -> "BoundaryAssignment":
        """Apply this condition to every exposed face as a fallback."""
        return BoundaryAssignment(BoundarySelector.everywhere(), self)

    def on_outer(self) -> "BoundaryAssignment":
        """Apply this condition to every exterior domain face."""
        return BoundaryAssignment(BoundarySelector.outer(), self)

    def on_radial(self) -> "BoundaryAssignment":
        """Apply this condition to lateral exterior faces."""
        return BoundaryAssignment(BoundarySelector.radial(), self)

    def on_bottom(self) -> "BoundaryAssignment":
        """Apply this condition to physical exterior bottom faces."""
        return BoundaryAssignment(BoundarySelector.bottom(), self)

    def on_top(self) -> "BoundaryAssignment":
        """Apply this condition to physical exterior top faces."""
        return BoundaryAssignment(BoundarySelector.top(), self)

    def on_excluded(
        self,
        *,
        key: str | None = None,
        kind: str | None = None,
        direction: str | None = None,
    ) -> "BoundaryAssignment":
        """Apply this condition to faces adjoining excluded material-mesh regions.

        Parameters
        ----------
        key
            Excluded neighbor key to match. The key must be a string.
            Mutually exclusive with ``kind``.
        kind
            Excluded-region kind to match, such as ``"reflector"`` or
            ``"channel"``. Mutually exclusive with ``key``.
        direction
            Optional radial (``"x+"``, ``"u-"``, and so on) or axial
            (``"bottom"`` or ``"top"``) face direction to match.

        Notes
        -----
        Every supplied selector argument must match; an omitted argument
        matches any value. Therefore, ``on_excluded()`` selects every
        excluded-region interface. This selector does not apply to physical
        exterior faces. Compatible excluded-interface assignments resolve by
        key and direction, key, kind and direction, kind, direction, then the
        unqualified excluded-interface rule; the global rule is the final
        fallback.
        """
        return BoundaryAssignment(
            BoundarySelector.to_excluded(
                key=key,
                kind=kind,
                direction=direction,
            ),
            self,
        )


@dataclass(frozen=True)
class _ResolvedBoundaryCondition:
    """Solver-facing boundary data resolved against one energy-group count.

    ``flux`` and ``source`` are immutable vectors ordered fast to thermal.
    For non-Dirichlet conditions, ``source`` makes
    ``J_out = alpha * phi_face - source`` explicit for every group.
    """

    kind: str
    alpha: float
    flux: np.ndarray
    source: np.ndarray


def _resolve_boundary_condition(
    condition: BoundaryCondition,
    groups: int,
) -> _ResolvedBoundaryCondition:
    """Resolve one boundary specification into group-complete solver data."""
    groups = require_positive_integer("groups", groups)
    zeros = _resolved_boundary_vector(None, groups, "source")
    if condition.kind == "dirichlet":
        return _ResolvedBoundaryCondition(
            kind=condition.kind,
            alpha=0.0,
            flux=_resolved_boundary_vector(condition.flux, groups, "flux"),
            source=zeros,
        )
    if condition.kind == "reflective":
        alpha, source = 0.0, zeros
    elif condition.kind == "vacuum":
        alpha, source = 0.5, zeros
    elif condition.kind == "robin":
        alpha, source = condition.alpha, zeros
    elif condition.kind == "incoming_current":
        alpha = 0.5
        source = 2.0 * _resolved_boundary_vector(condition.current, groups, "current")
    elif condition.kind == "partial_current":
        alpha = (1.0 - condition.beta) / (2.0 * (1.0 + condition.beta))
        source = (
            2.0
            * _resolved_boundary_vector(condition.current, groups, "current")
            / (1.0 + condition.beta)
        )
    else:
        raise ValueError(f"unknown boundary condition kind: {condition.kind!r}")
    source = freeze_owned_float_array(source)
    return _ResolvedBoundaryCondition(
        kind=condition.kind,
        alpha=alpha,
        flux=zeros,
        source=source,
    )


def _owned_boundary_vector(
    field_name: str, value: np.ndarray | list[float]
) -> np.ndarray:
    """Return an owned immutable boundary spectrum."""
    vector = readonly_float_array(value)
    _check_boundary_vector(field_name, vector)
    return vector


def _resolved_boundary_vector(
    values: np.ndarray | None,
    groups: int,
    name: str,
) -> np.ndarray:
    """Return an owned group-complete vector, using homogeneous zero if omitted."""
    if values is None:
        return freeze_owned_float_array(np.zeros(groups, dtype=float))
    if values.shape != (groups,):
        raise ValueError(f"boundary {name} must have shape ({groups},)")
    return freeze_owned_float_array(values.copy())


def _check_boundary_vector(field_name: str, value: np.ndarray) -> None:
    """Require a nonempty finite nonnegative one-dimensional spectrum."""
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{field_name} must be a nonempty one-dimensional vector")
    require_finite_nonnegative_array(field_name, value)


@dataclass(frozen=True)
class BoundarySelector:
    """Immutably select exposed boundary faces by topology, not physics.

    Prefer the fluent ``BoundaryCondition`` selection methods in ordinary
    problem definitions. Direct construction is available when constructing a
    ``BoundaryAssignment`` explicitly. A selector matches topology only; its
    associated condition supplies the boundary physics.

    Parameters
    ----------
    scope
        Closed selector vocabulary: ``"global"`` is the fallback for every
        exposed face; ``"outer"`` selects all physical exterior faces;
        ``"radial"`` selects physical lateral exterior faces;
        ``"to_excluded"`` selects faces adjoining excluded material-mesh
        positions; and ``"bottom"`` and ``"top"`` select the corresponding
        physical axial exterior faces.
    direction
        Optional direction filter for ``"to_excluded"`` only. Accepted values
        are ``"x+"``, ``"x-"``, ``"u+"``, ``"u-"``, ``"v+"``, ``"v-"``,
        ``"bottom"``, and ``"top"``.
    excluded_key
        Optional human-readable material-mesh excluded key for
        ``"to_excluded"`` only. It must contain a non-whitespace character
        and only printable characters.
    excluded_kind
        Optional human-readable excluded-region kind for ``"to_excluded"``
        only. It must contain a non-whitespace character and only printable
        characters. Mutually exclusive with ``excluded_key``.

    Raises
    ------
    TypeError
        If ``scope`` is not a string, or a supplied ``direction``,
        ``excluded_key``, or ``excluded_kind`` is not a string.
    ValueError
        If the scope or direction is unknown; direction is used with another
        scope; an excluded filter is used with another scope; both excluded
        filters are supplied; or a supplied identifier is empty,
        whitespace-only, or non-printable.

    Notes
    -----
    ``to_excluded`` filters are combined, and omitted filters match any value.
    Selector precedence is resolved by ``BoundaryConditionSet``; the selector
    does not carry a numeric priority.
    """

    scope: str
    direction: str | None = None
    excluded_key: str | None = None
    excluded_kind: str | None = None

    def __post_init__(self) -> None:
        """Check selector vocabulary and excluded-key type."""
        require_nonempty_string("boundary selector scope", self.scope)
        require_optional_string("boundary direction", self.direction)
        if self.excluded_key is not None:
            require_human_readable_identifier("material key", self.excluded_key)
        if self.excluded_kind is not None:
            require_human_readable_identifier("excluded kind", self.excluded_kind)
        if self.scope not in _BOUNDARY_SELECTOR_SCOPES:
            raise ValueError(f"unknown boundary selector scope: {self.scope!r}")
        if (
            self.direction is not None
            and self.direction not in _BOUNDARY_DIRECTION_LABELS
        ):
            raise ValueError(f"unknown boundary direction: {self.direction!r}")
        if self.direction is not None and self.scope not in _DIRECTIONAL_SCOPES:
            raise ValueError(
                f"scope {self.scope!r} does not support direction selection"
            )
        if self.scope != "to_excluded" and (
            self.excluded_key is not None or self.excluded_kind is not None
        ):
            raise ValueError("excluded key/kind selectors require scope 'to_excluded'")
        if self.excluded_key is not None and self.excluded_kind is not None:
            raise ValueError("select by excluded key or excluded kind, not both")

    @classmethod
    def everywhere(cls) -> "BoundarySelector":
        """Select every exposed face as the global fallback.

        This includes both physical exterior and excluded-interface faces.
        More specific compatible selectors take precedence.
        """
        return cls(scope="global")

    @classmethod
    def outer(cls) -> "BoundarySelector":
        """Select every physical exterior face, radial and axial.

        Excluded-interface faces are not physical exterior faces and are not
        selected. Radial, bottom, and top selectors refine this scope.
        """
        return cls(scope="outer")

    @classmethod
    def radial(cls) -> "BoundarySelector":
        """Select physical lateral exterior faces only.

        This scope excludes bottom, top, and excluded-interface faces. Morana
        does not provide direction-specific physical-exterior radial selectors.
        """
        return cls(scope="radial")

    @classmethod
    def to_excluded(
        cls,
        key: str | None = None,
        kind: str | None = None,
        direction: str | None = None,
    ) -> "BoundarySelector":
        """Select faces adjoining excluded material-mesh positions.

        Parameters
        ----------
        key
            Optional excluded neighbor key. The key must be a string.
            Mutually exclusive with ``kind``.
        kind
            Optional excluded-region kind. Mutually exclusive with ``key``.
        direction
            Optional radial or axial face direction: ``"x+"``, ``"x-"``,
            ``"u+"``, ``"u-"``, ``"v+"``, ``"v-"``, ``"bottom"``, or
            ``"top"``.

        Notes
        -----
        Every supplied filter must match, while an omitted filter matches any
        value. Thus ``to_excluded()`` selects every excluded interface. This
        factory does not select physical exterior faces.
        """
        return cls(
            scope="to_excluded",
            direction=direction,
            excluded_key=key,
            excluded_kind=kind,
        )

    @classmethod
    def bottom(cls) -> "BoundarySelector":
        """Select physical exterior faces at the bottom of the stack.

        Excluded interfaces to a lower in-stack position are selected only by
        ``to_excluded(direction="bottom")``.
        """
        return cls(scope="bottom")

    @classmethod
    def top(cls) -> "BoundarySelector":
        """Select physical exterior faces at the top of the stack.

        Excluded interfaces to an upper in-stack position are selected only by
        ``to_excluded(direction="top")``.
        """
        return cls(scope="top")

    @property
    def specificity_key(self) -> tuple[str, str | None, str | None, str | None]:
        """Return the complete selector identity for duplicate detection.

        This tuple is not a precedence rank. ``BoundaryConditionSet`` rejects
        assignments with equal identities and resolves distinct overlapping
        selectors by its documented topology precedence.
        """
        return (
            self.scope,
            self.direction,
            self.excluded_key,
            self.excluded_kind,
        )


@dataclass(frozen=True)
class BoundaryAssignment:
    """Pair a boundary topology selector with one physics condition.

    Parameters
    ----------
    selector
        Exposed-face topology selected by the assignment.
    condition
        Diffusion boundary physics applied when the selector resolves.

    Notes
    -----
    ``BoundaryAssignment`` is an immutable value record: its selector and
    condition are both immutable, so the assignment safely represents one
    checked topology-to-physics pairing.

    Create assignments through a ``BoundaryCondition.on_*`` method in normal
    use. ``BoundaryConditionSet`` rejects duplicate selectors and resolves
    overlapping assignments using its documented topology precedence.
    """

    selector: BoundarySelector
    condition: BoundaryCondition

    def __post_init__(self) -> None:
        """Require checked topology and physics value objects."""
        if not isinstance(self.selector, BoundarySelector):
            raise TypeError("selector must be a BoundarySelector")
        if not isinstance(self.condition, BoundaryCondition):
            raise TypeError("condition must be a BoundaryCondition")


@dataclass(frozen=True, init=False)
class BoundaryConditionSet:
    """Store selector-bound boundary conditions and resolve exposed faces.

    Parameters
    ----------
    *assignments
        Fluent boundary assignments returned by methods such as
        [`BoundaryCondition.globally`][morana.boundary.BoundaryCondition.globally]
        and [`BoundaryCondition.on_top`][morana.boundary.BoundaryCondition.on_top].
        Every value must be a ``BoundaryAssignment``. Two assignments with the
        same complete selector identity are rejected; overlapping assignments
        with distinct selectors are allowed. The set may remain incomplete
        while a configuration is being assembled.

    Attributes
    ----------
    assignments
        Tuple of immutable assignments in construction order.

    Notes
    -----
    ``BoundaryConditionSet`` is immutable. Use ``with_assignment()`` to
    construct a separate checked set with one additional assignment.

    Resolution is independent of construction order. An excluded interface
    uses key-and-direction, key, kind-and-direction, kind, direction-only,
    any-excluded, then global precedence. A physical exterior face uses its
    matching radial, bottom, or top selector, then outer, then global
    precedence. Call ``check_coverage`` with the current material mesh to
    require a condition for every exposed radial and axial face. Problem
    assembly and solver execution perform the same coverage requirement.
    """

    assignments: tuple[BoundaryAssignment, ...]

    def __init__(self, *assignments: BoundaryAssignment) -> None:
        """Build a possibly incomplete boundary set from fluent assignments.

        Raises
        ------
        TypeError
            If an input is not a ``BoundaryAssignment``.
        ValueError
            If an input duplicates a selector already present in this set.
        """
        assignments = tuple(assignments)
        seen: set[tuple[str, str | None, str | None, str | None]] = set()
        for assignment in assignments:
            if not isinstance(assignment, BoundaryAssignment):
                raise TypeError("boundary conditions must be BoundaryAssignment values")
            key = assignment.selector.specificity_key
            if key in seen:
                raise ValueError(
                    "duplicate boundary assignment for selector "
                    f"{assignment.selector!r}"
                )
            seen.add(key)
        object.__setattr__(self, "assignments", assignments)

    def with_assignment(self, assignment: BoundaryAssignment) -> "BoundaryConditionSet":
        """Return a new set with one additional fluent assignment.

        Parameters
        ----------
        assignment
            Fluent ``BoundaryAssignment`` to append after existing
            assignments. It is checked against every existing selector using
            the same validation as the constructor.

        Returns
        -------
        BoundaryConditionSet
            New set; this set and its assignment tuple are unchanged.

        Raises
        ------
        TypeError
            If ``assignment`` is not a ``BoundaryAssignment``.
        ValueError
            If ``assignment`` duplicates an existing selector.
        """
        return BoundaryConditionSet(*self.assignments, assignment)

    def resolve(self, face: "DomainFace") -> BoundaryCondition:
        """Resolve a condition for one exposed radial or axial face.

        Parameters
        ----------
        face
            Exposed ``DomainFace`` from the material mesh. Internal faces do
            not have boundary conditions and are rejected.

        Returns
        -------
        BoundaryCondition
            Condition selected by the locked topology precedence.

        Raises
        ------
        TypeError
            If ``face`` is not a ``DomainFace``.
        ValueError
            If ``face`` is internal or no assignment covers it.

        Notes
        -----
        Resolution uses selector specificity, not assignment construction
        order. See the class documentation for both physical-exterior and
        excluded-interface precedence chains.
        """
        if not isinstance(face, DomainFace):
            raise TypeError("face must be a DomainFace")
        if face.kind == DOMAIN_FACE_KIND_OUTER:
            if face.direction in AXIAL_DIRECTION_LABELS:
                selectors = (
                    BoundarySelector(face.direction),
                    BoundarySelector.outer(),
                    BoundarySelector.everywhere(),
                )
            else:
                selectors = (
                    BoundarySelector.radial(),
                    BoundarySelector.outer(),
                    BoundarySelector.everywhere(),
                )
        elif face.kind == DOMAIN_FACE_KIND_TO_EXCLUDED:
            selectors = [
                BoundarySelector.to_excluded(
                    key=face.neighbor_key, direction=face.direction
                ),
                BoundarySelector.to_excluded(key=face.neighbor_key),
            ]
            if face.neighbor_key_kind is not None:
                selectors.extend(
                    (
                        BoundarySelector.to_excluded(
                            kind=face.neighbor_key_kind, direction=face.direction
                        ),
                        BoundarySelector.to_excluded(kind=face.neighbor_key_kind),
                    )
                )
            selectors.extend(
                (
                    BoundarySelector.to_excluded(direction=face.direction),
                    BoundarySelector.to_excluded(),
                    BoundarySelector.everywhere(),
                )
            )
        else:
            raise ValueError("boundary resolution requires an exposed face")
        by_selector = {
            assignment.selector: assignment.condition for assignment in self.assignments
        }
        for selector in selectors:
            condition = by_selector.get(selector)
            if condition is not None:
                return condition
        raise ValueError(f"boundary conditions do not cover {face!r}")

    def check_coverage(self, material_mesh: "MaterialMesh") -> None:
        """Require every exposed radial and axial face to resolve.

        Parameters
        ----------
        material_mesh
            Material layout whose active cells and exposed faces are checked.

        Raises
        ------
        TypeError
            If ``material_mesh`` is not a ``MaterialMesh``.
        ValueError
            If one or more exposed faces lack a resolved condition. The error
            lists every uncovered face with its layer, active-cell ID,
            direction, topology kind, and excluded-neighbor identity when
            applicable.

        Notes
        -----
        This method does not mutate the set or the material mesh. Use
        ``ProblemConfiguration.check_boundary_coverage()`` when validating the
        boundary set attached to a complete problem definition.
        """
        if not isinstance(material_mesh, MaterialMesh):
            raise TypeError("material_mesh must be a MaterialMesh")
        uncovered = []
        directions = material_mesh.face_direction_labels
        for axial_index in range(material_mesh.n_axial_layers):
            active_cells = material_mesh.n_active_cells(axial_index)
            for active_id in range(active_cells):
                for direction in directions:
                    face = material_mesh.face(axial_index, active_id, direction)
                    if face.kind in EXPOSED_DOMAIN_FACE_KINDS:
                        try:
                            self.resolve(face)
                        except ValueError:
                            uncovered.append(
                                f"axial_index={axial_index}, "
                                f"active_id={active_id}, "
                                f"direction={face.direction!r}, "
                                f"kind={face.kind!r}, "
                                f"neighbor_key={face.neighbor_key!r}, "
                                f"neighbor_key_kind={face.neighbor_key_kind!r}"
                            )
        if uncovered:
            raise ValueError(
                "boundary conditions do not cover exposed faces: "
                + "; ".join(uncovered)
            )
