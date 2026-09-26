"""Run the complete finite-volume production-path assessment matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from studies.finite_volume_performance.cases import (
    BICGSTAB_JACOBI_CASE_ID,
    DIRECT_CASE_ID,
    GMRES_JACOBI_CASE_ID,
    POWER_ITERATION_ID,
    WIELANDT_ITERATION_ID,
)
from studies.finite_volume_performance.protocol import DEFAULT_STUDY_PROTOCOL
from studies.finite_volume_performance.runner import DEFAULT_OUTPUT_DIR, run


@dataclass(frozen=True)
class ProductionWorkload:
    """Describe one selected workload and its two checked fixed shifts."""

    groups: int
    axial_layers: int
    structured_shift: float
    permuted_shift: float

    def shift(self, placement: str) -> float:
        """Return the checked shift for one maintained placement."""
        if placement == "structured":
            return self.structured_shift
        if placement == "permuted":
            return self.permuted_shift
        raise ValueError("placement must be 'structured' or 'permuted'")


MATRIX_WORKLOADS = (
    ProductionWorkload(36, 6, 1.021339391553522, 0.9900506307635286),
    ProductionWorkload(18, 12, 1.0407799590610955, 1.0045898337826535),
    ProductionWorkload(72, 24, 1.0207685354094909, 0.9907602987675709),
)
MATRIX_PLACEMENTS = ("structured", "permuted")
MATRIX_SOLVERS = (
    DIRECT_CASE_ID,
    GMRES_JACOBI_CASE_ID,
    BICGSTAB_JACOBI_CASE_ID,
)
MATRIX_ITERATIONS = (POWER_ITERATION_ID, WIELANDT_ITERATION_ID)


@dataclass(frozen=True)
class ProductionMatrixRun:
    """Describe one independently resumable matrix entry."""

    workload: ProductionWorkload
    placement: str
    solver: str
    iteration: str

    @property
    def shift_inverse_keff(self) -> float | None:
        """Return the fixed shift for Wielandt iteration, otherwise ``None``."""
        if self.iteration == POWER_ITERATION_ID:
            return None
        return self.workload.shift(self.placement)

    @property
    def repetitions(self) -> int:
        """Return the frozen repetition count for this solver."""
        return DEFAULT_STUDY_PROTOCOL.repetitions(self.solver)

    @property
    def output_name(self) -> str:
        """Return a stable filename containing every matrix axis."""
        workload = self.workload
        return (
            f"production_{self.solver}_{self.placement}_g{workload.groups}_"
            f"z{workload.axial_layers}_{self.iteration}.json"
        )


def matrix_runs() -> tuple[ProductionMatrixRun, ...]:
    """Return the complete deterministic 36-entry production matrix."""
    return tuple(
        ProductionMatrixRun(workload, placement, solver, iteration)
        for placement in MATRIX_PLACEMENTS
        for iteration in MATRIX_ITERATIONS
        for workload in MATRIX_WORKLOADS
        for solver in MATRIX_SOLVERS
    )


def _print_plan(entries: tuple[ProductionMatrixRun, ...], output_dir: Path) -> None:
    """Print the resolved campaign without starting workers."""
    recorded = sum(entry.repetitions + 1 for entry in entries)
    print(
        f"Production matrix: {len(entries)} entries, {recorded} recorded workers, "
        f"{len(entries)} warm-ups"
    )
    for entry in entries:
        shift = entry.shift_inverse_keff
        policy = (
            entry.iteration if shift is None else f"{entry.iteration} shift={shift}"
        )
        print(
            f"  {entry.output_name}: {entry.solver}, {entry.placement}, "
            f"g={entry.workload.groups}, z={entry.workload.axial_layers}, "
            f"{policy}, repetitions={entry.repetitions}, profile=yes"
        )
    print(f"Output directory: {output_dir}")


def run_matrix(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> tuple[Path, ...]:
    """Run or resume every entry in the frozen production assessment matrix."""
    entries = matrix_runs()
    output_paths = tuple(output_dir / entry.output_name for entry in entries)
    if dry_run:
        _print_plan(entries, output_dir)
        return output_paths

    existing = tuple(path for path in output_paths if path.exists())
    if existing and not resume:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Production-matrix outputs already exist: {names}; use --resume or a new "
            "output directory"
        )

    completed = []
    for entry_number, (entry, output_path) in enumerate(
        zip(entries, output_paths, strict=True), start=1
    ):
        print(
            f"Production matrix entry [{entry_number}/{len(entries)}]: "
            f"{entry.output_name}"
        )
        completed.append(
            run(
                "selected",
                output_dir,
                timeout_seconds=DEFAULT_STUDY_PROTOCOL.timeout_seconds,
                address_space_limit_bytes=(
                    DEFAULT_STUDY_PROTOCOL.address_space_limit_bytes
                ),
                resume=resume and output_path.exists(),
                solver_case=entry.solver,
                workloads=((entry.workload.groups, entry.workload.axial_layers),),
                warmup=DEFAULT_STUDY_PROTOCOL.warmup,
                output_name=entry.output_name,
                repetitions=entry.repetitions,
                profile=DEFAULT_STUDY_PROTOCOL.profile,
                iteration_id=entry.iteration,
                shift_inverse_keff=entry.shift_inverse_keff,
                placement=entry.placement,
            )
        )
    return tuple(completed)


def parse_args() -> argparse.Namespace:
    """Parse the complete-matrix command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume existing entries and start matrix entries not yet created",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the complete resolved matrix without starting workers",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete production-path assessment matrix."""
    arguments = parse_args()
    run_matrix(
        arguments.output_dir,
        resume=arguments.resume,
        dry_run=arguments.dry_run,
    )


if __name__ == "__main__":
    main()
