from dataclasses import dataclass, field
from typing import List, Optional, Callable
import time

import numpy as np
from numpy.typing import NDArray

from holograd.core.types import Proof, VerificationResult, VerificationStatus
from holograd.protocol.direction import DirectionGenerator, ADCCodebook


@dataclass
class VerificationStats:
    total_proofs: int = 0
    proofs_sampled: int = 0
    proofs_accepted: int = 0
    proofs_rejected: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.proofs_sampled == 0:
            return 1.0
        return self.proofs_accepted / self.proofs_sampled

    @property
    def rejection_rate(self) -> float:
        if self.proofs_sampled == 0:
            return 0.0
        return self.proofs_rejected / self.proofs_sampled


class Verifier:
    def __init__(
        self,
        dimension: int,
        p_verify: float = 0.05,
        epsilon: float = 1e-4,
        use_adc: bool = False,
        adc_rank: int = 32,
    ):
        if not 0.0 <= p_verify <= 1.0:
            raise ValueError("p_verify must be in [0, 1]")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.dimension = dimension
        self.p_verify = p_verify
        self.epsilon = epsilon
        self.use_adc = use_adc

        self._direction_gen = DirectionGenerator(dimension)
        self._adc_codebook: Optional[ADCCodebook] = None

        if use_adc:
            self._adc_codebook = ADCCodebook(dimension, rank=adc_rank)

        self._rng = np.random.default_rng()
        self._stats = VerificationStats()

    @property
    def stats(self) -> VerificationStats:
        return self._stats

    def set_codebook(self, codebook: ADCCodebook) -> None:
        self._adc_codebook = codebook

    def should_sample(self) -> bool:
        return self._rng.random() < self.p_verify

    def verify_proof(
        self,
        proof: Proof,
        recompute_fn: Callable[[NDArray[np.float32]], float],
        force_verify: bool = False,
    ) -> VerificationResult:
        self._stats.total_proofs += 1

        should_verify = force_verify or self.should_sample()

        if not should_verify:
            return VerificationResult(
                proof=proof,
                status=VerificationStatus.PENDING,
                was_sampled=False,
                timestamp=time.time(),
            )

        self._stats.proofs_sampled += 1

        if proof.adc_projection is not None and self._adc_codebook is not None:
            direction = self._adc_codebook.reconstruct_direction(proof.adc_projection)
        elif self.use_adc and self._adc_codebook is not None:
            result = self._adc_codebook.generate_direction(proof.seed)
            direction = result.direction
        else:
            result = self._direction_gen.generate(proof.seed)
            direction = result.direction

        recomputed_scalar = recompute_fn(direction)
        difference = abs(proof.scalar - recomputed_scalar)

        if difference <= self.epsilon:
            status = VerificationStatus.ACCEPTED
            self._stats.proofs_accepted += 1
        else:
            status = VerificationStatus.SLASHED
            self._stats.proofs_rejected += 1

        return VerificationResult(
            proof=proof,
            status=status,
            recomputed_scalar=recomputed_scalar,
            difference=difference,
            was_sampled=True,
            timestamp=time.time(),
        )

    def verify_batch(
        self,
        proofs: List[Proof],
        recompute_fn: Callable[[NDArray[np.float32]], float],
    ) -> List[VerificationResult]:
        return [self.verify_proof(proof, recompute_fn) for proof in proofs]

    def detection_probability(
        self,
        num_proofs: int,
        invalid_fraction: float,
    ) -> float:
        if invalid_fraction <= 0 or invalid_fraction > 1:
            return 0.0

        num_invalid = int(num_proofs * invalid_fraction)
        if num_invalid == 0:
            return 0.0

        prob_miss_one = 1 - self.p_verify
        prob_miss_all = prob_miss_one**num_invalid

        return 1 - prob_miss_all

    def reset_stats(self) -> None:
        self._stats = VerificationStats()
