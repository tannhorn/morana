# Coupled operators

These functions expose Morana's independently callable finite-volume assembly
surface. Start with `extract_cross_section_data(snapshot)`, then pass the
complete layered data and the same snapshot to the loss, fission, and
right-hand-side assemblers. Each function can also capture a mutable
configuration, but an explicit snapshot keeps several assembly calls on one
stable problem definition. Assemblers return fresh mutable SciPy sparse
matrices or NumPy vectors; extracted compact data are immutable snapshots.
Read [direct finite-volume operator assembly](../operator_assembly.md)
for data flow, layout, and ownership conventions and the
[finite-volume theory](../theory_references.md#finite-volume-discretization)
for the assembled equations.

::: morana.operators
    options:
      show_submodules: false
