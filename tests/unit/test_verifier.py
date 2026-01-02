import numpy as np
import pytest

from holograd.core.types import Proof, VerificationStatus
from holograd.verification.verifier import Verifier


class TestVerifier:
    def test_verify_valid_proof(self):
        verifier = Verifier(dimension=100, p_verify=1.0, epsilon=1e-4)

        seed = b"\x00" * 32
        direction_result = verifier._direction_gen.generate(seed)
        expected_scalar = 1.5

        def recompute_fn(direction):
            return expected_scalar

        proof = Proof(
            step=0,
            worker_id=0,
            seed=seed,
            scalar=expected_scalar,
        )

        result = verifier.verify_proof(proof, recompute_fn, force_verify=True)

        assert result.status == VerificationStatus.ACCEPTED
        assert result.is_valid
        assert result.difference < 1e-4

    def test_verify_invalid_proof(self):
        verifier = Verifier(dimension=100, p_verify=1.0, epsilon=1e-4)

        seed = b"\x00" * 32

        def recompute_fn(direction):
            return 1.0

        proof = Proof(
            step=0,
            worker_id=0,
            seed=seed,
            scalar=2.0,
        )

        result = verifier.verify_proof(proof, recompute_fn, force_verify=True)

        assert result.status == VerificationStatus.SLASHED
        assert result.was_slashed
        assert result.difference > 1e-4

    def test_sampling_probability(self):
        verifier = Verifier(dimension=100, p_verify=0.5, epsilon=1e-4)

        np.random.seed(42)
        sampled_count = 0
        n_trials = 1000

        for i in range(n_trials):
            if verifier.should_sample():
                sampled_count += 1

        actual_rate = sampled_count / n_trials
        assert 0.4 < actual_rate < 0.6

    def test_stats_tracking(self):
        verifier = Verifier(dimension=100, p_verify=1.0, epsilon=1e-4)

        def recompute_fn(direction):
            return 1.0

        valid_proof = Proof(step=0, worker_id=0, seed=b"\x00" * 32, scalar=1.0)
        invalid_proof = Proof(step=0, worker_id=1, seed=b"\x01" * 32, scalar=5.0)

        verifier.verify_proof(valid_proof, recompute_fn, force_verify=True)
        verifier.verify_proof(invalid_proof, recompute_fn, force_verify=True)

        assert verifier.stats.total_proofs == 2
        assert verifier.stats.proofs_sampled == 2
        assert verifier.stats.proofs_accepted == 1
        assert verifier.stats.proofs_rejected == 1

    def test_detection_probability(self):
        verifier = Verifier(dimension=100, p_verify=0.05, epsilon=1e-4)

        prob = verifier.detection_probability(num_proofs=100, invalid_fraction=0.1)

        expected = 1 - (0.95**10)
        assert prob == pytest.approx(expected, rel=1e-6)

    def test_zero_verify_rate(self):
        verifier = Verifier(dimension=100, p_verify=0.0, epsilon=1e-4)

        proof = Proof(step=0, worker_id=0, seed=b"\x00" * 32, scalar=1.0)

        result = verifier.verify_proof(proof, lambda d: 1.0)

        assert result.status == VerificationStatus.PENDING
        assert not result.was_sampled

    def test_full_verify_rate(self):
        verifier = Verifier(dimension=100, p_verify=1.0, epsilon=1e-4)

        proof = Proof(step=0, worker_id=0, seed=b"\x00" * 32, scalar=1.0)

        result = verifier.verify_proof(proof, lambda d: 1.0)

        assert result.was_sampled

    def test_verify_batch(self):
        verifier = Verifier(dimension=100, p_verify=1.0, epsilon=1e-4)

        proofs = [
            Proof(step=0, worker_id=i, seed=bytes([i]) + b"\x00" * 31, scalar=1.0) for i in range(5)
        ]

        results = verifier.verify_batch(proofs, lambda d: 1.0)

        assert len(results) == 5
        assert all(r.was_sampled for r in results)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            Verifier(dimension=100, p_verify=1.5)

        with pytest.raises(ValueError):
            Verifier(dimension=100, epsilon=-0.1)

    def test_reset_stats(self):
        verifier = Verifier(dimension=100, p_verify=1.0, epsilon=1e-4)

        proof = Proof(step=0, worker_id=0, seed=b"\x00" * 32, scalar=1.0)
        verifier.verify_proof(proof, lambda d: 1.0, force_verify=True)

        assert verifier.stats.total_proofs == 1

        verifier.reset_stats()

        assert verifier.stats.total_proofs == 0
