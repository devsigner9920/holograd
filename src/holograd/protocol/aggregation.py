from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class AggregationResult:
    gradient: NDArray[np.float32]
    proofs_used: int
    proofs_trimmed: int
    scalars_original: List[float] = field(default_factory=list)
    scalars_trimmed: List[float] = field(default_factory=list)
    trimmed_indices: List[int] = field(default_factory=list)


class RobustAggregator:
    def __init__(
        self,
        tau: float = 0.1,
        method: Literal["trimmed_mean", "median", "mean"] = "trimmed_mean",
    ):
        if not 0.0 <= tau < 0.5:
            raise ValueError("tau must be in [0, 0.5)")

        self.tau = tau
        self.method = method

    def aggregate(
        self,
        scalars: List[float],
        directions: List[NDArray[np.float32]],
        scale_factor: float = 1.0,
        effective_dimension: int | None = None,
    ) -> AggregationResult:
        if len(scalars) != len(directions):
            raise ValueError("scalars and directions must have same length")

        if len(scalars) == 0:
            raise ValueError("cannot aggregate empty list")

        n = len(scalars)
        scalars_array = np.array(scalars, dtype=np.float32)

        # Variance correction: sqrt(K / effective_dim)
        # For random directions: effective_dim = D (full dimension)
        # For ADC: effective_dim = rank (low-rank subspace)
        if effective_dimension is not None and effective_dimension > 0:
            variance_correction = float(np.sqrt(n / effective_dimension))
        else:
            variance_correction = 1.0

        if self.method == "mean":
            return self._aggregate_mean(
                scalars_array, directions, scale_factor, n, variance_correction
            )
        elif self.method == "median":
            return self._aggregate_median(
                scalars_array, directions, scale_factor, n, variance_correction
            )
        else:
            return self._aggregate_trimmed_mean(
                scalars_array, directions, scale_factor, n, variance_correction
            )

    def _aggregate_mean(
        self,
        scalars: NDArray[np.float32],
        directions: List[NDArray[np.float32]],
        scale_factor: float,
        n: int,
        variance_correction: float = 1.0,
    ) -> AggregationResult:
        dimension = directions[0].shape[0]
        gradient = np.zeros(dimension, dtype=np.float32)

        for scalar, direction in zip(scalars, directions):
            gradient += scalar * direction

        gradient = (scale_factor / n) * variance_correction * gradient

        return AggregationResult(
            gradient=gradient,
            proofs_used=n,
            proofs_trimmed=0,
            scalars_original=scalars.tolist(),
            scalars_trimmed=scalars.tolist(),
            trimmed_indices=[],
        )

    def _aggregate_median(
        self,
        scalars: NDArray[np.float32],
        directions: List[NDArray[np.float32]],
        scale_factor: float,
        n: int,
        variance_correction: float = 1.0,
    ) -> AggregationResult:
        dimension = directions[0].shape[0]
        directions_array = np.stack(directions, axis=0)

        weighted = scalars[:, np.newaxis] * directions_array
        gradient = np.median(weighted, axis=0).astype(np.float32)
        gradient = scale_factor * variance_correction * gradient

        return AggregationResult(
            gradient=gradient,
            proofs_used=n,
            proofs_trimmed=0,
            scalars_original=scalars.tolist(),
            scalars_trimmed=scalars.tolist(),
            trimmed_indices=[],
        )

    def _aggregate_trimmed_mean(
        self,
        scalars: NDArray[np.float32],
        directions: List[NDArray[np.float32]],
        scale_factor: float,
        n: int,
        variance_correction: float = 1.0,
    ) -> AggregationResult:
        trim_count = int(n * self.tau)

        if trim_count == 0 or 2 * trim_count >= n:
            return self._aggregate_mean(scalars, directions, scale_factor, n, variance_correction)

        sorted_indices = np.argsort(scalars)
        trimmed_low = sorted_indices[:trim_count].tolist()
        trimmed_high = sorted_indices[-trim_count:].tolist()
        trimmed_indices = trimmed_low + trimmed_high

        keep_indices = sorted_indices[trim_count:-trim_count]

        dimension = directions[0].shape[0]
        gradient = np.zeros(dimension, dtype=np.float32)

        kept_scalars = []
        for idx in keep_indices:
            gradient += scalars[idx] * directions[idx]
            kept_scalars.append(float(scalars[idx]))

        k_prime = len(keep_indices)
        gradient = (scale_factor / k_prime) * variance_correction * gradient

        return AggregationResult(
            gradient=gradient,
            proofs_used=k_prime,
            proofs_trimmed=2 * trim_count,
            scalars_original=scalars.tolist(),
            scalars_trimmed=kept_scalars,
            trimmed_indices=trimmed_indices,
        )

    def trim_scalars(self, scalars: List[float]) -> Tuple[List[float], List[int]]:
        n = len(scalars)
        if n == 0:
            return [], []

        trim_count = int(n * self.tau)
        if trim_count == 0 or 2 * trim_count >= n:
            return scalars, []

        scalars_array = np.array(scalars, dtype=np.float32)
        sorted_indices = np.argsort(scalars_array)

        trimmed_low = sorted_indices[:trim_count].tolist()
        trimmed_high = sorted_indices[-trim_count:].tolist()
        trimmed_indices = trimmed_low + trimmed_high

        keep_indices = sorted_indices[trim_count:-trim_count]
        kept_scalars = [scalars[idx] for idx in keep_indices]

        return kept_scalars, trimmed_indices
