# Python reference

This reference is generated from Morana's type annotations and NumPy-style
docstrings. The package and module `__all__` declarations define the supported
module-level surface; underscore-prefixed members are private.

- [Core API](core.md) covers the objects exported from `morana`.
- [Finite-volume diffusion solves](finite_volume.md) covers the method-scoped
  `morana.solvers.finite_volume` functions.
- [Coupled operators](operators.md) covers the independently callable
  finite-volume assembly API in `morana.operators`.

Use the [modeling and solver workflow](../modeling_workflow.md) for
cross-object behavior, result conventions, and capability boundaries.
The [theory and numerical conventions](../theory_references.md) derive the
equations assembled by the operator API.
