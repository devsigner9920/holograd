from dataclasses import dataclass
from typing import Callable, Optional
import time

import numpy as np
from numpy.typing import NDArray

from holograd.core.types import Task, Proof, WorkerState, WorkerStatus
from holograd.protocol.direction import DirectionGenerator, ADCCodebook, DirectionResult


@dataclass
class WorkerConfig:
    worker_id: int
    dimension: int
    use_adc: bool = False
    adc_rank: int = 32


class Worker:
    def __init__(
        self,
        config: WorkerConfig,
        gradient_fn: Optional[Callable[[NDArray[np.float32]], float]] = None,
    ):
        self.config = config
        self.state = WorkerState(worker_id=config.worker_id)

        self._direction_gen = DirectionGenerator(config.dimension)
        self._adc_codebook: Optional[ADCCodebook] = None

        if config.use_adc:
            self._adc_codebook = ADCCodebook(
                dimension=config.dimension,
                rank=config.adc_rank,
            )

        self._gradient_fn = gradient_fn

    def set_gradient_fn(self, fn: Callable[[NDArray[np.float32]], float]) -> None:
        self._gradient_fn = fn

    def set_codebook(self, codebook: ADCCodebook) -> None:
        self._adc_codebook = codebook

    def compute_proof(
        self,
        task: Task,
        gradient: Optional[NDArray[np.float32]] = None,
    ) -> Proof:
        self.state.status = WorkerStatus.COMPUTING
        self.state.current_step = task.step

        adc_projection = None

        if task.use_momentum and task.momentum_direction is not None:
            direction = task.momentum_direction
        elif task.use_adc and self._adc_codebook is not None:
            result = self._adc_codebook.generate_direction(task.seed)
            direction = result.direction
            adc_projection = result.z_projection
        else:
            result = self._direction_gen.generate(task.seed)
            direction = result.direction

        if gradient is not None:
            scalar = float(np.dot(gradient.flatten(), direction))
        elif self._gradient_fn is not None:
            scalar = self._gradient_fn(direction)
        else:
            raise ValueError("Either gradient or gradient_fn must be provided")

        proof = Proof(
            step=task.step,
            worker_id=self.config.worker_id,
            seed=task.seed,
            scalar=scalar,
            timestamp=time.time(),
            adc_projection=adc_projection,
        )

        self.state.status = WorkerStatus.SUBMITTED
        self.state.proofs_submitted += 1

        return proof

    def reset(self) -> None:
        self.state = WorkerState(worker_id=self.config.worker_id)
