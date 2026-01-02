import numpy as np
import pytest

from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
from holograd.distributed.worker import Worker, WorkerConfig
from holograd.core.types import Proof


class TestMomentumCentricCoordinator:
    @pytest.fixture
    def momentum_config(self):
        return CoordinatorConfig(
            dimension=1000,
            num_workers=4,
            proofs_per_step=4,
            use_momentum_centric=True,
            momentum_warmup_steps=2,
            momentum_beta=0.9,
            grad_norm_ema_alpha=0.1,
        )

    @pytest.fixture
    def coordinator(self, momentum_config):
        coord = Coordinator(momentum_config)
        params = np.random.randn(momentum_config.dimension).astype(np.float32)
        coord.set_parameters(params)
        return coord

    def test_warmup_uses_random_directions(self, coordinator):
        coordinator.set_batch(np.array([1, 2, 3]), seed=42)
        tasks = coordinator.publish_tasks(step=0)

        assert tasks[0].use_momentum is False
        assert tasks[0].momentum_direction is None

    def test_after_warmup_uses_momentum(self, coordinator):
        for step in range(3):
            coordinator.set_batch(np.array([step]), seed=step)
            coordinator.publish_tasks(step=step)
            gradient = np.random.randn(1000).astype(np.float32)
            coordinator.update_parameters(gradient)

        coordinator.set_batch(np.array([10]), seed=100)
        tasks = coordinator.publish_tasks(step=3)

        assert tasks[0].use_momentum is True
        assert tasks[0].momentum_direction is not None
        assert tasks[0].momentum_direction.shape == (1000,)

    def test_momentum_direction_is_normalized(self, coordinator):
        for step in range(3):
            coordinator.set_batch(np.array([step]), seed=step)
            coordinator.publish_tasks(step=step)
            gradient = np.random.randn(1000).astype(np.float32) * 10
            coordinator.update_parameters(gradient)

        coordinator.set_batch(np.array([10]), seed=100)
        tasks = coordinator.publish_tasks(step=3)

        norm = np.linalg.norm(tasks[0].momentum_direction)
        assert abs(norm - 1.0) < 1e-5

    def test_momentum_aggregation_uses_mean_scalar(self, coordinator):
        for step in range(3):
            coordinator.set_batch(np.array([step]), seed=step)
            coordinator.publish_tasks(step=step)
            gradient = np.random.randn(1000).astype(np.float32)
            coordinator.update_parameters(gradient)

        coordinator.set_batch(np.array([10]), seed=100)
        tasks = coordinator.publish_tasks(step=3)

        workers = [Worker(WorkerConfig(worker_id=i, dimension=1000)) for i in range(4)]
        gradient = np.random.randn(1000).astype(np.float32)
        proofs = [w.compute_proof(tasks[i], gradient=gradient) for i, w in enumerate(workers)]

        for proof in proofs:
            coordinator.collect_proof(proof)

        reconstructed, agg_result = coordinator.aggregate()

        assert agg_result.proofs_used == 4
        expected_direction = tasks[0].momentum_direction
        cosine_sim = np.dot(reconstructed, expected_direction) / (
            np.linalg.norm(reconstructed) * np.linalg.norm(expected_direction) + 1e-8
        )
        assert abs(abs(cosine_sim) - 1.0) < 1e-5


class TestMomentumCentricWorker:
    def test_worker_projects_onto_momentum_direction(self):
        dim = 100
        momentum_dir = np.random.randn(dim).astype(np.float32)
        momentum_dir = momentum_dir / np.linalg.norm(momentum_dir)

        coord_config = CoordinatorConfig(
            dimension=dim,
            num_workers=1,
            proofs_per_step=1,
            use_momentum_centric=True,
            momentum_warmup_steps=0,
        )
        coordinator = Coordinator(coord_config)
        coordinator.set_parameters(np.zeros(dim, dtype=np.float32))
        coordinator._momentum_direction = momentum_dir
        coordinator._grad_norm_estimate = 1.0

        coordinator.set_batch(np.array([1]), seed=42)
        tasks = coordinator.publish_tasks(step=0)

        worker = Worker(WorkerConfig(worker_id=0, dimension=dim))
        gradient = np.random.randn(dim).astype(np.float32)
        proof = worker.compute_proof(tasks[0], gradient=gradient)

        expected_scalar = float(np.dot(gradient, momentum_dir))
        assert abs(proof.scalar - expected_scalar) < 1e-5


class TestMomentumCentricEndToEnd:
    def test_training_loop_with_momentum(self):
        dim = 500
        num_workers = 4
        num_steps = 10

        config = CoordinatorConfig(
            dimension=dim,
            num_workers=num_workers,
            proofs_per_step=num_workers,
            use_momentum_centric=True,
            momentum_warmup_steps=2,
        )
        coord = Coordinator(config)
        workers = [Worker(WorkerConfig(worker_id=i, dimension=dim)) for i in range(num_workers)]

        params = np.random.randn(dim).astype(np.float32)
        coord.set_parameters(params)

        initial_loss = np.sum(params**2)

        for step in range(num_steps):
            coord.set_batch(np.array([step]), seed=step)
            tasks = coord.publish_tasks(step=step)

            gradient = 2 * coord._current_params
            proofs = [w.compute_proof(tasks[i], gradient=gradient) for i, w in enumerate(workers)]

            for proof in proofs:
                coord.collect_proof(proof)

            reconstructed, _ = coord.aggregate()
            coord.update_parameters(reconstructed)

        final_loss = np.sum(coord._current_params**2)
        assert final_loss < initial_loss
