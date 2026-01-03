import numpy as np
import pytest

from holograd.core.types import Task, WorkerStatus
from holograd.distributed.worker import Worker, WorkerConfig
from holograd.protocol.direction import ADCCodebook


class TestWorker:
    def test_compute_proof_with_gradient(self):
        config = WorkerConfig(worker_id=0, dimension=100)
        worker = Worker(config)

        task = Task(
            step=0,
            seed=b"\x00" * 32,
            param_commitment=b"\x01" * 32,
            batch_commitment=b"\x02" * 32,
        )
        gradient = np.random.randn(100).astype(np.float32)

        proof = worker.compute_proof(task, gradient=gradient)

        assert proof.step == 0
        assert proof.worker_id == 0
        assert proof.seed == task.seed
        assert np.isfinite(proof.scalar)

    def test_compute_proof_with_gradient_fn(self):
        config = WorkerConfig(worker_id=0, dimension=100)
        worker = Worker(config)

        def gradient_fn(direction):
            return float(np.sum(direction))

        worker.set_gradient_fn(gradient_fn)

        task = Task(
            step=0,
            seed=b"\x00" * 32,
            param_commitment=b"\x01" * 32,
            batch_commitment=b"\x02" * 32,
        )

        proof = worker.compute_proof(task)

        assert np.isfinite(proof.scalar)

    def test_state_updates(self):
        config = WorkerConfig(worker_id=5, dimension=100)
        worker = Worker(config)

        task = Task(
            step=3,
            seed=b"\x00" * 32,
            param_commitment=b"\x01" * 32,
            batch_commitment=b"\x02" * 32,
        )
        gradient = np.random.randn(100).astype(np.float32)

        worker.compute_proof(task, gradient=gradient)

        assert worker.state.status == WorkerStatus.SUBMITTED
        assert worker.state.current_step == 3
        assert worker.state.proofs_submitted == 1

    def test_adc_mode(self):
        config = WorkerConfig(
            worker_id=0,
            dimension=100,
            use_adc=True,
            adc_rank=32,
        )
        worker = Worker(config)

        codebook = ADCCodebook(dimension=100, rank=32)
        worker.set_codebook(codebook)

        task = Task(
            step=0,
            seed=b"\x00" * 32,
            param_commitment=b"\x01" * 32,
            batch_commitment=b"\x02" * 32,
            use_adc=True,
        )
        gradient = np.random.randn(100).astype(np.float32)

        proof = worker.compute_proof(task, gradient=gradient)

        assert proof.adc_projection is not None
        assert proof.adc_projection.shape == (32,)

    def test_reset(self):
        config = WorkerConfig(worker_id=0, dimension=100)
        worker = Worker(config)

        task = Task(
            step=0,
            seed=b"\x00" * 32,
            param_commitment=b"\x01" * 32,
            batch_commitment=b"\x02" * 32,
        )
        gradient = np.random.randn(100).astype(np.float32)
        worker.compute_proof(task, gradient=gradient)

        worker.reset()

        assert worker.state.proofs_submitted == 0
        assert worker.state.status == WorkerStatus.IDLE

    def test_no_gradient_raises(self):
        config = WorkerConfig(worker_id=0, dimension=100)
        worker = Worker(config)

        task = Task(
            step=0,
            seed=b"\x00" * 32,
            param_commitment=b"\x01" * 32,
            batch_commitment=b"\x02" * 32,
        )

        with pytest.raises(ValueError):
            worker.compute_proof(task)
