# Finite-volume diffusion solves

The functions below execute Morana's cell-centered finite-volume
diffusion method. Pass a complete `ProblemConfiguration` or
`ProblemConfigurationSnapshot` explicitly to each function. A mutable
configuration is captured as an immutable snapshot at solve entry; a supplied
snapshot is used directly. The completed result retains that same snapshot as
provenance.

::: morana.solvers.finite_volume
