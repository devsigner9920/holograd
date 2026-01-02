import numpy as np
import pytest

from holograd.core.types import (
    Task,
    Proof,
    VerificationResult,
    VerificationStatus,
    WorkerState,
    WorkerStatus,
    CoordinatorState,
)


class TestTask:
    def test_valid_task(self):
        task = Task(
            step=1,
            seed=b"\x00" * 32,
            param_commitment=b"\x01" * 32,
            batch_commitment=b"\x02" * 32,
        )
        assert task.step == 1
        assert task.use_adc is False

    def test_invalid_seed_length(self):
        with pytest.raises(ValueError, match="seed must be 32 bytes"):
            Task(
                step=1,
                seed=b"\x00" * 16,
                param_commitment=b"\x01" * 32,
                batch_commitment=b"\x02" * 32,
            )

    def test_with_adc(self):
        task = Task(
            step=1,
            seed=b"\x00" * 32,
            param_commitment=b"\x01" * 32,
            batch_commitment=b"\x02" * 32,
            codebook_commitment=b"\x03" * 32,
            use_adc=True,
        )
        assert task.use_adc is True
        assert task.codebook_commitment is not None


class TestProof:
    def test_valid_proof(self):
        proof = Proof(
            step=1,
            worker_id=0,
            seed=b"\x00" * 32,
            scalar=0.5,
        )
        assert proof.scalar == 0.5

    def test_invalid_scalar(self):
        with pytest.raises(ValueError, match="scalar must be finite"):
            Proof(
                step=1,
                worker_id=0,
                seed=b"\x00" * 32,
                scalar=float("nan"),
            )

    def test_with_adc_projection(self):
        projection = np.random.randn(32).astype(np.float32)
        proof = Proof(
            step=1,
            worker_id=0,
            seed=b"\x00" * 32,
            scalar=0.5,
            adc_projection=projection,
        )
        assert proof.adc_projection is not None
        assert proof.adc_projection.shape == (32,)


class TestVerificationResult:
    def test_accepted_proof(self):
        proof = Proof(step=1, worker_id=0, seed=b"\x00" * 32, scalar=0.5)
        result = VerificationResult(
            proof=proof,
            status=VerificationStatus.ACCEPTED,
            recomputed_scalar=0.5,
            difference=0.0,
            was_sampled=True,
        )
        assert result.is_valid is True
        assert result.was_slashed is False

    def test_slashed_proof(self):
        proof = Proof(step=1, worker_id=0, seed=b"\x00" * 32, scalar=0.5)
        result = VerificationResult(
            proof=proof,
            status=VerificationStatus.SLASHED,
            recomputed_scalar=0.9,
            difference=0.4,
            was_sampled=True,
        )
        assert result.is_valid is False
        assert result.was_slashed is True


class TestCoordinatorState:
    def test_reset_for_step(self):
        state = CoordinatorState(current_step=0, target_proof_count=64)
        state.proofs_collected.append(Proof(step=0, worker_id=0, seed=b"\x00" * 32, scalar=0.5))

        state.reset_for_step(1)

        assert state.current_step == 1
        assert len(state.proofs_collected) == 0

    def test_add_proof_returns_true_when_full(self):
        state = CoordinatorState(current_step=0, target_proof_count=2)

        proof1 = Proof(step=0, worker_id=0, seed=b"\x00" * 32, scalar=0.5)
        proof2 = Proof(step=0, worker_id=1, seed=b"\x01" * 32, scalar=0.6)

        assert state.add_proof(proof1) is False
        assert state.add_proof(proof2) is True

    def test_add_proof_wrong_step(self):
        state = CoordinatorState(current_step=0, target_proof_count=2)
        proof = Proof(step=1, worker_id=0, seed=b"\x00" * 32, scalar=0.5)

        assert state.add_proof(proof) is False
        assert len(state.proofs_collected) == 0
