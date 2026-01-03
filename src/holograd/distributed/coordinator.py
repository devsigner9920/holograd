from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

import numpy as np
from numpy.typing import NDArray

from holograd.core.types import Task, Proof, CoordinatorState, AggregationResult
from holograd.protocol.commitment import CommitmentChain
from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.protocol.aggregation import RobustAggregator


@dataclass
class CoordinatorConfig:
    dimension: int
    num_workers: int
    proofs_per_step: int = 64
    global_seed: str = "holograd_v1"
    use_adc: bool = False
    adc_rank: int = 32
    adc_oja_alpha: float = 1e-3
    adc_qr_period: int = 100
    adc_warmup_samples: int = 0
    adc_alpha_decay: float = 1.0
    adc_alpha_min: float = 1e-4
    adc_use_power_iteration: bool = False
    adc_power_iteration_steps: int = 3
    tau: float = 0.1
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    momentum: float = 0.9

    use_momentum_centric: bool = False
    momentum_beta: float = 0.9
    momentum_warmup_steps: int = 10
    grad_norm_ema_alpha: float = 0.1
    device: str = "cpu"


class Coordinator:
    def __init__(self, config: CoordinatorConfig):
        self.config = config
        self.state = CoordinatorState(target_proof_count=config.proofs_per_step)

        self._commitment_chain = CommitmentChain.from_string(config.global_seed)
        self._direction_gen = DirectionGenerator(config.dimension)
        self._aggregator = RobustAggregator(tau=config.tau)

        self._adc_codebook: Optional[ADCCodebook] = None
        if config.use_adc:
            self._adc_codebook = ADCCodebook(
                dimension=config.dimension,
                rank=config.adc_rank,
                oja_alpha=config.adc_oja_alpha,
                qr_period=config.adc_qr_period,
                warmup_samples=config.adc_warmup_samples,
                alpha_decay=config.adc_alpha_decay,
                alpha_min=config.adc_alpha_min,
                use_power_iteration=config.adc_use_power_iteration,
                power_iteration_steps=config.adc_power_iteration_steps,
                device=config.device,
            )

        self._current_params: Optional[NDArray[np.float32]] = None
        self._current_batch_indices: Optional[NDArray[np.int64]] = None
        self._current_batch_seed: int = 0
        self._momentum_buffer: Optional[NDArray[np.float32]] = None

        self._momentum_direction: Optional[NDArray[np.float32]] = None
        self._grad_norm_estimate: float = 1.0

    @property
    def codebook(self) -> Optional[ADCCodebook]:
        return self._adc_codebook

    def set_parameters(self, params: NDArray[np.float32]) -> None:
        self._current_params = params
        self.state.param_commitment = self._commitment_chain.hash_parameters(params)

    def set_batch(self, indices: NDArray[np.int64], seed: int) -> None:
        self._current_batch_indices = indices
        self._current_batch_seed = seed
        self.state.batch_commitment = self._commitment_chain.hash_batch(indices, seed)

    def publish_tasks(self, step: int) -> List[Task]:
        if self.state.param_commitment is None:
            raise ValueError("Parameters must be set before publishing tasks")
        if self.state.batch_commitment is None:
            raise ValueError("Batch must be set before publishing tasks")

        self.state.reset_for_step(step)

        codebook_commitment = None
        if self._adc_codebook is not None:
            codebook_commitment = self._commitment_chain.hash_codebook(
                self._adc_codebook.codebook, step
            )
            self.state.codebook_commitment = codebook_commitment

        use_momentum = (
            self.config.use_momentum_centric
            and step >= self.config.momentum_warmup_steps
            and self._momentum_direction is not None
        )

        # During ADC warmup, use random directions to get better gradient estimates
        adc_warmed_up = self._adc_codebook is not None and self._adc_codebook.is_warmed_up
        use_adc_this_step = self.config.use_adc and adc_warmed_up and not use_momentum

        tasks = []
        for task_id in range(self.config.proofs_per_step):
            seed = self._commitment_chain.get_worker_seed(
                param_commitment=self.state.param_commitment,
                batch_commitment=self.state.batch_commitment,
                step=step,
                worker_id=task_id,
                codebook_commitment=codebook_commitment,
            )

            task = Task(
                step=step,
                seed=seed,
                param_commitment=self.state.param_commitment,
                batch_commitment=self.state.batch_commitment,
                codebook_commitment=codebook_commitment,
                use_adc=use_adc_this_step,
                use_momentum=use_momentum,
                momentum_direction=self._momentum_direction if use_momentum else None,
                timestamp=time.time(),
            )
            tasks.append(task)

        return tasks

    def collect_proof(self, proof: Proof) -> bool:
        if proof.step != self.state.current_step:
            return False

        return self.state.add_proof(proof)

    def aggregate(self) -> Tuple[NDArray[np.float32], AggregationResult]:
        proofs = self.state.proofs_collected

        if len(proofs) == 0:
            raise ValueError("No proofs collected")

        first_proof = proofs[0]
        is_momentum_mode = (
            self.config.use_momentum_centric
            and self._momentum_direction is not None
            and hasattr(first_proof, "seed")
        )

        if is_momentum_mode and self.state.current_step >= self.config.momentum_warmup_steps:
            return self._aggregate_momentum(proofs)
        else:
            return self._aggregate_random(proofs)

    def _aggregate_random(
        self, proofs: List[Proof]
    ) -> Tuple[NDArray[np.float32], AggregationResult]:
        scalars = [p.scalar for p in proofs]

        adc_warmed_up = self._adc_codebook is not None and self._adc_codebook.is_warmed_up
        use_adc_for_reconstruction = self.config.use_adc and adc_warmed_up

        directions = self._reconstruct_directions_batch(proofs, use_adc_for_reconstruction)

        if use_adc_for_reconstruction and self._adc_codebook is not None:
            scale_factor = self._adc_codebook.get_scale_factor()
            effective_dimension = self.config.adc_rank
        else:
            scale_factor = self._direction_gen.scale_factor
            effective_dimension = self.config.dimension

        agg_result = self._aggregator.aggregate(
            scalars, directions, scale_factor, effective_dimension
        )

        self.state.steps_completed += 1
        self.state.total_proofs_processed += len(proofs)
        self.state.total_proofs_trimmed += agg_result.proofs_trimmed

        return agg_result.gradient, agg_result

    def _reconstruct_directions_batch(
        self,
        proofs: List[Proof],
        use_adc: bool,
    ) -> List[NDArray[np.float32]]:
        directions_map: dict[int, NDArray[np.float32]] = {}

        adc_indices: List[int] = []
        adc_projections: List[NDArray[np.float32]] = []

        for i, p in enumerate(proofs):
            if p.adc_projection is not None:
                adc_indices.append(i)
                adc_projections.append(p.adc_projection)

        if adc_projections and self._adc_codebook is not None:
            z_batch = np.stack(adc_projections, axis=1)
            batch_result = self._adc_codebook.reconstruct_directions_batch(z_batch)
            for batch_idx, proof_idx in enumerate(adc_indices):
                directions_map[proof_idx] = batch_result[:, batch_idx]

        for i, p in enumerate(proofs):
            if i in directions_map:
                continue
            if use_adc and self._adc_codebook is not None:
                result = self._adc_codebook.generate_direction(p.seed)
            else:
                result = self._direction_gen.generate(p.seed)
            directions_map[i] = result.direction

        return [directions_map[i] for i in range(len(proofs))]

    def _aggregate_momentum(
        self, proofs: List[Proof]
    ) -> Tuple[NDArray[np.float32], AggregationResult]:
        scalars = [p.scalar for p in proofs]

        trimmed_scalars, trimmed_indices = self._aggregator.trim_scalars(scalars)
        mean_scalar = float(np.mean(trimmed_scalars))

        gradient = mean_scalar * self._grad_norm_estimate * self._momentum_direction

        agg_result = AggregationResult(
            gradient=gradient,
            proofs_used=len(trimmed_scalars),
            proofs_trimmed=len(trimmed_indices),
            scalars_original=scalars,
            scalars_trimmed=trimmed_scalars,
            trimmed_indices=trimmed_indices,
        )

        self.state.steps_completed += 1
        self.state.total_proofs_processed += len(proofs)
        self.state.total_proofs_trimmed += len(trimmed_indices)

        return gradient, agg_result

    def update_parameters(
        self,
        gradient: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        if self._current_params is None:
            raise ValueError("Parameters not set")

        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.config.max_grad_norm:
            gradient = gradient * (self.config.max_grad_norm / grad_norm)
            grad_norm = self.config.max_grad_norm

        if self._momentum_buffer is None:
            self._momentum_buffer = np.zeros_like(gradient)

        self._momentum_buffer = self.config.momentum * self._momentum_buffer + gradient

        self._current_params = (
            self._current_params - self.config.learning_rate * self._momentum_buffer
        )
        self.set_parameters(self._current_params)

        if self._adc_codebook is not None:
            self._adc_codebook.update(gradient)

        if self.config.use_momentum_centric:
            self._update_momentum_direction(gradient, grad_norm)

        return self._current_params

    def _update_momentum_direction(
        self,
        gradient: NDArray[np.float32],
        grad_norm: float,
    ) -> None:
        beta = self.config.momentum_beta
        alpha = self.config.grad_norm_ema_alpha

        if self._momentum_direction is None:
            self._momentum_direction = gradient / (grad_norm + 1e-8)
            self._grad_norm_estimate = grad_norm
        else:
            unnormalized = (
                beta * (self._momentum_direction * self._grad_norm_estimate) + (1 - beta) * gradient
            )
            norm = np.linalg.norm(unnormalized)
            if norm > 1e-8:
                self._momentum_direction = unnormalized / norm
            self._grad_norm_estimate = alpha * grad_norm + (1 - alpha) * self._grad_norm_estimate

    def step(
        self,
        proofs: List[Proof],
    ) -> Tuple[NDArray[np.float32], AggregationResult]:
        for proof in proofs:
            self.collect_proof(proof)

        gradient, agg_result = self.aggregate()
        self.update_parameters(gradient)

        return gradient, agg_result
