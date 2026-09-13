"""Shared OpenMC-independent MGXS run settings and artifact conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from examples.openmc_comparison.run_settings import (
    EigenvalueRunSettings,
    active_history_label,
)

SUPPORTED_GROUP_STRUCTURES = ("CASMO-8", "CASMO-25", "CASMO-40", "CASMO-70")
TALLIED_GROUP_STRUCTURE = "CASMO-70"
DEFAULT_PARTICLES = 20_000
DEFAULT_BATCHES = 300
DEFAULT_INACTIVE = 100
DEFAULT_GENERATIONS_PER_BATCH = 5
DEFAULT_SEED = 31415
RUNTIME_DATASET = "homogenized_cell"
RUNTIME_TOTAL_FIELD = "openmc_transport"
SCATTER_MATRIX_FORMULATION = "consistent"


@dataclass(frozen=True)
class MgxsRunSettings(EigenvalueRunSettings):
    """Describe one reproducible stochastic MGXS transport run.

    Every run tallies :data:`TALLIED_GROUP_STRUCTURE` and condenses it to the
    coarser structures in :data:`SUPPORTED_GROUP_STRUCTURES`. Keeping the fine
    tally model fixed allows a higher-history request to restart from a
    compatible lower-history statepoint and permits every supported
    condensation to be exported from that statepoint without rerunning
    transport.
    """

    particles: int = DEFAULT_PARTICLES
    batches: int = DEFAULT_BATCHES
    inactive: int = DEFAULT_INACTIVE
    generations_per_batch: int = DEFAULT_GENERATIONS_PER_BATCH
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        """Check OpenMC statistical controls."""

        super().__post_init__()
        if not isinstance(self.generations_per_batch, int) or isinstance(
            self.generations_per_batch, bool
        ):
            raise TypeError("generations_per_batch must be an integer")
        if self.generations_per_batch <= 0:
            raise ValueError("generations_per_batch must be positive")

    @classmethod
    def from_active_histories(
        cls,
        *,
        active_histories: int,
        seed: int,
        particles: int = DEFAULT_PARTICLES,
        inactive: int = DEFAULT_INACTIVE,
        generations_per_batch: int = DEFAULT_GENERATIONS_PER_BATCH,
    ) -> "MgxsRunSettings":
        """Build settings that represent an exact active-history target."""

        for name, value in (
            ("active_histories", active_histories),
            ("particles", particles),
            ("generations_per_batch", generations_per_batch),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        histories_per_batch = particles * generations_per_batch
        active_batches, remainder = divmod(active_histories, histories_per_batch)
        if remainder:
            raise ValueError(
                "active histories must be divisible by particles times "
                "generations per batch"
            )
        return cls(
            particles=particles,
            batches=inactive + active_batches,
            inactive=inactive,
            generations_per_batch=generations_per_batch,
            seed=seed,
        )

    @property
    def active_histories(self) -> int:
        """Return the total number of active source-particle histories."""

        return (
            self.particles * self.generations_per_batch * (self.batches - self.inactive)
        )

    @property
    def artifact_label(self) -> str:
        """Return a collision-free OpenMC artifact label."""

        label = f"{active_history_label(self.active_histories)}_seed{self.seed}"
        if (
            self.particles,
            self.generations_per_batch,
            self.inactive,
        ) != (
            DEFAULT_PARTICLES,
            DEFAULT_GENERATIONS_PER_BATCH,
            DEFAULT_INACTIVE,
        ):
            label += (
                f"_p{self.particles}_g{self.generations_per_batch}" f"_i{self.inactive}"
            )
        return label

    def runtime_library_path(self, output_dir: Path, group_structure: str) -> Path:
        """Return one group structure's expected runtime-MGXS library path."""

        if group_structure not in SUPPORTED_GROUP_STRUCTURES:
            raise ValueError("unsupported MGXS group structure")
        filename = f"{group_structure.lower()}.h5"
        return self.artifact_directory(output_dir) / filename

    def as_record(self) -> dict[str, object]:
        """Return JSON-ready settings used to identify restart compatibility."""

        return {
            "runtime_total_field": RUNTIME_TOTAL_FIELD,
            "scatter_matrix_formulation": SCATTER_MATRIX_FORMULATION,
            "tallied_group_structure": TALLIED_GROUP_STRUCTURE,
            "group_structures": list(SUPPORTED_GROUP_STRUCTURES),
            "particles": self.particles,
            "batches": self.batches,
            "inactive_batches": self.inactive,
            "generations_per_batch": self.generations_per_batch,
            "active_histories": self.active_histories,
            "seed": self.seed,
        }

    def matches_record(self, record: dict[str, object]) -> bool:
        """Return whether ``record`` identifies this exact MGXS sample."""

        return all(
            record.get(name) == value for name, value in self.as_record().items()
        )

    def matches_tally_record(self, record: dict[str, object]) -> bool:
        """Return whether ``record`` identifies this exact fine-group tally.

        Unlike :meth:`matches_record`, this ignores the recorded export set so
        every supported condensation can be recreated from an existing
        CASMO-70 statepoint.
        """

        tally_fields = {
            name: value
            for name, value in self.as_record().items()
            if name != "group_structures"
        }
        return all(record.get(name) == value for name, value in tally_fields.items())

    def is_compatible_restart(self, record: dict[str, object]) -> bool:
        """Return whether ``record`` can continue this run to more batches."""

        previous_batches = record.get("batches")
        return (
            record.get("runtime_total_field") == RUNTIME_TOTAL_FIELD
            and record.get("scatter_matrix_formulation") == SCATTER_MATRIX_FORMULATION
            and record.get("tallied_group_structure") == TALLIED_GROUP_STRUCTURE
            and record.get("group_structures") == list(SUPPORTED_GROUP_STRUCTURES)
            and record.get("particles") == self.particles
            and record.get("inactive_batches") == self.inactive
            and record.get("generations_per_batch") == self.generations_per_batch
            and record.get("seed") == self.seed
            and isinstance(previous_batches, int)
            and not isinstance(previous_batches, bool)
            and self.inactive < previous_batches < self.batches
        )
