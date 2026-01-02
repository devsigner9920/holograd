import numpy as np
import pytest

from holograd.core.types import Proof
from holograd.distributed.coordinator import Coordinator, CoordinatorConfig


class TestCoordinator:
    def test_publish_tasks(self):
        config = CoordinatorConfig(
            dimension=100,
            num_workers=4,
            proofs_per_step=4,
        )
        coord = Coordinator(config)

        params = np.random.randn(100).astype(np.float32)
        coord.set_parameters(params)
        coord.set_batch(np.array([0, 1, 2, 3], dtype=np.int64), seed=42)

        tasks = coord.publish_tasks(step=0)

        assert len(tasks) == 4
        for i, task in enumerate(tasks):
            assert task.step == 0
            assert len(task.seed) == 32
            assert task.param_commitment is not None
            assert task.batch_commitment is not None

    def test_tasks_have_different_seeds(self):
        config = CoordinatorConfig(
            dimension=100,
            num_workers=4,
            proofs_per_step=4,
        )
        coord = Coordinator(config)

        params = np.random.randn(100).astype(np.float32)
        coord.set_parameters(params)
        coord.set_batch(np.array([0, 1, 2, 3], dtype=np.int64), seed=42)

        tasks = coord.publish_tasks(step=0)

        seeds = [t.seed for t in tasks]
        assert len(set(seeds)) == 4

    def test_collect_and_aggregate(self):
        config = CoordinatorConfig(
            dimension=100,
            num_workers=4,
            proofs_per_step=4,
        )
        coord = Coordinator(config)

        params = np.random.randn(100).astype(np.float32)
        coord.set_parameters(params)
        coord.set_batch(np.array([0, 1, 2, 3], dtype=np.int64), seed=42)

        tasks = coord.publish_tasks(step=0)

        for i, task in enumerate(tasks):
            proof = Proof(
                step=0,
                worker_id=i,
                seed=task.seed,
                scalar=float(i),
            )
            coord.collect_proof(proof)

        gradient, agg_result = coord.aggregate()

        assert gradient.shape == (100,)
        assert agg_result.proofs_used > 0

    def test_update_parameters(self):
        config = CoordinatorConfig(
            dimension=100,
            num_workers=4,
            proofs_per_step=4,
            learning_rate=0.1,
            max_grad_norm=100.0,
            momentum=0.0,
        )
        coord = Coordinator(config)

        params = np.ones(100, dtype=np.float32)
        coord.set_parameters(params)

        gradient = np.ones(100, dtype=np.float32)
        new_params = coord.update_parameters(gradient)

        expected = np.ones(100) - 0.1 * np.ones(100)
        np.testing.assert_allclose(new_params, expected)

    def test_wrong_step_proof_rejected(self):
        config = CoordinatorConfig(
            dimension=100,
            num_workers=4,
            proofs_per_step=4,
        )
        coord = Coordinator(config)

        params = np.random.randn(100).astype(np.float32)
        coord.set_parameters(params)
        coord.set_batch(np.array([0, 1, 2, 3], dtype=np.int64), seed=42)
        coord.publish_tasks(step=0)

        wrong_step_proof = Proof(
            step=1,
            worker_id=0,
            seed=b"\x00" * 32,
            scalar=1.0,
        )

        accepted = coord.collect_proof(wrong_step_proof)

        assert accepted is False
        assert len(coord.state.proofs_collected) == 0

    def test_adc_mode(self):
        config = CoordinatorConfig(
            dimension=100,
            num_workers=4,
            proofs_per_step=4,
            use_adc=True,
            adc_rank=16,
        )
        coord = Coordinator(config)

        assert coord.codebook is not None
        assert coord.codebook.rank == 16

        params = np.random.randn(100).astype(np.float32)
        coord.set_parameters(params)
        coord.set_batch(np.array([0, 1, 2, 3], dtype=np.int64), seed=42)

        tasks = coord.publish_tasks(step=0)

        assert all(t.use_adc for t in tasks)
        assert all(t.codebook_commitment is not None for t in tasks)

    def test_full_step(self):
        config = CoordinatorConfig(
            dimension=100,
            num_workers=4,
            proofs_per_step=4,
            learning_rate=0.01,
        )
        coord = Coordinator(config)

        params = np.random.randn(100).astype(np.float32)
        coord.set_parameters(params)
        coord.set_batch(np.array([0, 1, 2, 3], dtype=np.int64), seed=42)

        tasks = coord.publish_tasks(step=0)

        proofs = []
        for i, task in enumerate(tasks):
            proofs.append(
                Proof(
                    step=0,
                    worker_id=i,
                    seed=task.seed,
                    scalar=1.0,
                )
            )

        gradient, agg_result = coord.step(proofs)

        assert coord.state.steps_completed == 1
        assert coord.state.total_proofs_processed == 4
