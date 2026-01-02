"""
Core type definitions for HoloGrad protocol.

Defines the fundamental data structures used throughout the protocol:
- Task: Published by Coordinator to Workers
- Proof: Submitted by Workers to Coordinator
- VerificationResult: Returned by Verifier
- Actor states for Coordinator and Worker
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class VerificationStatus(Enum):
    """Status of proof verification."""

    PENDING = auto()
    ACCEPTED = auto()
    REJECTED = auto()
    SLASHED = auto()


class WorkerStatus(Enum):
    """Status of a worker in the protocol."""

    IDLE = auto()
    COMPUTING = auto()
    SUBMITTED = auto()
    VERIFIED = auto()
    SLASHED = auto()


@dataclass
class Task:
    """
    Task published by Coordinator to Workers.

    Contains all information needed for a worker to compute
    a valid proof for the current training step.
    """

    # Step number in training
    step: int

    # Worker-specific seed for direction generation
    seed: bytes

    # Commitment to current parameters H(theta_t)
    param_commitment: bytes

    # Commitment to minibatch H(B_t)
    batch_commitment: bytes

    # Commitment to ADC codebook H(U_t) if ADC enabled
    codebook_commitment: Optional[bytes] = None

    # Whether to use ADC directions
    use_adc: bool = False

    # Whether to use momentum-centric mode
    use_momentum: bool = False

    # Unit momentum direction for momentum-centric mode (D-dimensional)
    momentum_direction: Optional[NDArray[np.float32]] = None

    # Additional metadata
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Validate task data."""
        if not isinstance(self.seed, bytes) or len(self.seed) != 32:
            raise ValueError("seed must be 32 bytes (SHA-256 output)")
        if not isinstance(self.param_commitment, bytes) or len(self.param_commitment) != 32:
            raise ValueError("param_commitment must be 32 bytes")
        if not isinstance(self.batch_commitment, bytes) or len(self.batch_commitment) != 32:
            raise ValueError("batch_commitment must be 32 bytes")
        if self.use_momentum and self.momentum_direction is None:
            raise ValueError("momentum_direction required when use_momentum=True")


@dataclass
class Proof:
    """
    Proof submitted by Worker to Coordinator.

    Contains the scalar projection a = <g, v> and the seed
    used to generate direction v, enabling verification.
    """

    # Step number
    step: int

    # Worker ID
    worker_id: int

    # Seed used for direction generation
    seed: bytes

    # Scalar projection a = <g, v>
    scalar: float

    # Timestamp of computation completion
    timestamp: float = 0.0

    # Optional: low-dimensional projection for ADC (z vector)
    adc_projection: Optional[NDArray[np.float32]] = None

    def __post_init__(self) -> None:
        """Validate proof data."""
        if not isinstance(self.seed, bytes) or len(self.seed) != 32:
            raise ValueError("seed must be 32 bytes")
        if not np.isfinite(self.scalar):
            raise ValueError("scalar must be finite")


@dataclass
class VerificationResult:
    """
    Result of proof verification by Verifier.

    Contains the verification decision and supporting data
    for audit and analysis.
    """

    # Original proof being verified
    proof: Proof

    # Verification status
    status: VerificationStatus

    # Recomputed scalar value a*
    recomputed_scalar: Optional[float] = None

    # Absolute difference |a - a*|
    difference: Optional[float] = None

    # Whether proof was sampled for verification
    was_sampled: bool = False

    # Verification timestamp
    timestamp: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if proof was accepted."""
        return self.status == VerificationStatus.ACCEPTED

    @property
    def was_slashed(self) -> bool:
        """Check if worker was slashed."""
        return self.status == VerificationStatus.SLASHED


@dataclass
class WorkerState:
    """
    Internal state of a Worker actor.

    Tracks the worker's current status and accumulated data
    across training steps.
    """

    # Worker identifier
    worker_id: int

    # Current status
    status: WorkerStatus = WorkerStatus.IDLE

    # Current step being processed
    current_step: int = 0

    # Total proofs submitted
    proofs_submitted: int = 0

    # Total proofs verified
    proofs_verified: int = 0

    # Total proofs rejected/slashed
    proofs_rejected: int = 0

    # Accumulated rewards (simulation)
    rewards: float = 0.0

    # Accumulated penalties (simulation)
    penalties: float = 0.0

    # Simulated network delay profile
    delay_mean: float = 0.1
    delay_std: float = 0.05


@dataclass
class CoordinatorState:
    """
    Internal state of the Coordinator actor.

    Tracks aggregation state and protocol progress.
    """

    # Current training step
    current_step: int = 0

    # Proofs collected for current step
    proofs_collected: list[Proof] = field(default_factory=list)

    # Number of proofs needed per step
    target_proof_count: int = 64

    # Total steps completed
    steps_completed: int = 0

    # Total proofs processed across all steps
    total_proofs_processed: int = 0

    # Proofs trimmed by robust aggregation
    total_proofs_trimmed: int = 0

    # Current parameter commitment
    param_commitment: Optional[bytes] = None

    # Current batch commitment
    batch_commitment: Optional[bytes] = None

    # Current codebook commitment (if ADC enabled)
    codebook_commitment: Optional[bytes] = None

    def reset_for_step(self, step: int) -> None:
        """Reset state for a new training step."""
        self.current_step = step
        self.proofs_collected = []

    def add_proof(self, proof: Proof) -> bool:
        """
        Add a proof to the collection.

        Returns True if enough proofs collected.
        """
        if proof.step != self.current_step:
            return False

        self.proofs_collected.append(proof)
        return len(self.proofs_collected) >= self.target_proof_count


@dataclass
class AggregationResult:
    """Result of robust gradient aggregation."""

    # Synthesized gradient (flattened)
    gradient: NDArray[np.float32]

    # Number of proofs used (after trimming)
    proofs_used: int

    # Number of proofs trimmed
    proofs_trimmed: int

    # Scalar values before trimming
    scalars_original: list[float] = field(default_factory=list)

    # Scalar values after trimming
    scalars_trimmed: list[float] = field(default_factory=list)

    # Indices of trimmed proofs (for Byzantine tracking)
    trimmed_indices: list[int] = field(default_factory=list)


@dataclass
class StepMetrics:
    """Metrics collected for a single training step."""

    step: int

    # Training metrics
    loss: float = 0.0

    # Communication metrics
    bytes_received: int = 0
    proofs_received: int = 0

    # Aggregation metrics
    proofs_used: int = 0
    proofs_trimmed: int = 0

    # Verification metrics
    proofs_verified: int = 0
    proofs_accepted: int = 0
    proofs_rejected: int = 0

    # ADC metrics
    captured_energy_ratio: float = 0.0

    # Timing metrics
    step_time: float = 0.0
    collection_time: float = 0.0
    aggregation_time: float = 0.0
    update_time: float = 0.0
