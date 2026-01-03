from dataclasses import dataclass
from typing import List, Optional, Callable
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.typing import NDArray

from holograd.core.types import Task, Proof
from holograd.distributed.worker import Worker, WorkerConfig
from holograd.protocol.direction import ADCCodebook


@dataclass
class DelayConfig:
    enabled: bool = True
    distribution: str = "normal"
    mean: float = 0.1
    std: float = 0.05


class WorkerPool:
    def __init__(
        self,
        num_workers: int,
        dimension: int,
        use_adc: bool = False,
        adc_rank: int = 32,
        delay_config: Optional[DelayConfig] = None,
        max_threads: Optional[int] = None,
    ):
        self.num_workers = num_workers
        self.dimension = dimension
        self.delay_config = delay_config or DelayConfig(enabled=False)
        self.max_threads = max_threads or num_workers

        self._workers: List[Worker] = []
        for i in range(num_workers):
            config = WorkerConfig(
                worker_id=i,
                dimension=dimension,
                use_adc=use_adc,
                adc_rank=adc_rank,
            )
            self._workers.append(Worker(config))

        self._rng = np.random.default_rng()

    def set_gradient_fn(self, fn: Callable[[NDArray[np.float32]], float]) -> None:
        for worker in self._workers:
            worker.set_gradient_fn(fn)

    def set_codebook(self, codebook: ADCCodebook) -> None:
        for worker in self._workers:
            worker.set_codebook(codebook)

    def _get_delay(self) -> float:
        if not self.delay_config.enabled:
            return 0.0

        if self.delay_config.distribution == "normal":
            delay = self._rng.normal(self.delay_config.mean, self.delay_config.std)
        elif self.delay_config.distribution == "exponential":
            delay = self._rng.exponential(self.delay_config.mean)
        elif self.delay_config.distribution == "uniform":
            low = max(0, self.delay_config.mean - self.delay_config.std)
            high = self.delay_config.mean + self.delay_config.std
            delay = self._rng.uniform(low, high)
        else:
            delay = self.delay_config.mean

        return max(0.0, delay)

    def _compute_single(
        self,
        worker: Worker,
        task: Task,
        gradient: Optional[NDArray[np.float32]] = None,
    ) -> Proof:
        delay = self._get_delay()
        if delay > 0:
            time.sleep(delay)

        return worker.compute_proof(task, gradient)

    def compute_proofs_sequential(
        self,
        tasks: List[Task],
        gradient: Optional[NDArray[np.float32]] = None,
    ) -> List[Proof]:
        proofs = []
        for task, worker in zip(tasks, self._workers):
            proof = self._compute_single(worker, task, gradient)
            proofs.append(proof)
        return proofs

    def compute_proofs_parallel(
        self,
        tasks: List[Task],
        gradient: Optional[NDArray[np.float32]] = None,
        first_k: Optional[int] = None,
    ) -> List[Proof]:
        proofs = []
        target = first_k or len(tasks)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {}
            for i, task in enumerate(tasks[:target]):
                worker = self._workers[i % self.num_workers]
                future = executor.submit(self._compute_single, worker, task, gradient)
                futures[future] = i

            for future in as_completed(futures):
                proof = future.result()
                proofs.append(proof)

                if len(proofs) >= target:
                    break

        return proofs[:target]

    def get_worker(self, worker_id: int) -> Worker:
        return self._workers[worker_id]
