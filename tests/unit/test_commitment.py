import numpy as np
import pytest

from holograd.protocol.commitment import CommitmentChain


class TestCommitmentChain:
    def test_from_string(self):
        chain = CommitmentChain.from_string("test_seed")
        assert chain.global_seed == b"test_seed"

    def test_worker_seed_deterministic(self):
        chain = CommitmentChain.from_string("test_seed")

        seed1 = chain.get_worker_seed(
            param_commitment=b"\x00" * 32,
            batch_commitment=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        seed2 = chain.get_worker_seed(
            param_commitment=b"\x00" * 32,
            batch_commitment=b"\x01" * 32,
            step=0,
            worker_id=0,
        )

        assert seed1 == seed2
        assert len(seed1) == 32

    def test_different_workers_different_seeds(self):
        chain = CommitmentChain.from_string("test_seed")

        seed0 = chain.get_worker_seed(
            param_commitment=b"\x00" * 32,
            batch_commitment=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        seed1 = chain.get_worker_seed(
            param_commitment=b"\x00" * 32,
            batch_commitment=b"\x01" * 32,
            step=0,
            worker_id=1,
        )

        assert seed0 != seed1

    def test_different_steps_different_seeds(self):
        chain = CommitmentChain.from_string("test_seed")

        seed_step0 = chain.get_worker_seed(
            param_commitment=b"\x00" * 32,
            batch_commitment=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        seed_step1 = chain.get_worker_seed(
            param_commitment=b"\x00" * 32,
            batch_commitment=b"\x01" * 32,
            step=1,
            worker_id=0,
        )

        assert seed_step0 != seed_step1

    def test_with_codebook_commitment(self):
        chain = CommitmentChain.from_string("test_seed")

        seed_without = chain.get_worker_seed(
            param_commitment=b"\x00" * 32,
            batch_commitment=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        seed_with = chain.get_worker_seed(
            param_commitment=b"\x00" * 32,
            batch_commitment=b"\x01" * 32,
            step=0,
            worker_id=0,
            codebook_commitment=b"\x02" * 32,
        )

        assert seed_without != seed_with

    def test_hash_parameters_array(self):
        params = np.random.randn(100).astype(np.float32)

        hash1 = CommitmentChain.hash_parameters(params)
        hash2 = CommitmentChain.hash_parameters(params)

        assert hash1 == hash2
        assert len(hash1) == 32

    def test_hash_parameters_dict(self):
        params = {
            "layer1": np.zeros(10, dtype=np.float32),
            "layer2": np.ones(10, dtype=np.float32),
        }

        hash1 = CommitmentChain.hash_parameters(params)
        hash2 = CommitmentChain.hash_parameters(params)

        assert hash1 == hash2

    def test_hash_batch(self):
        indices = np.array([0, 5, 10, 15], dtype=np.int64)

        hash1 = CommitmentChain.hash_batch(indices, batch_seed=42)
        hash2 = CommitmentChain.hash_batch(indices, batch_seed=42)

        assert hash1 == hash2
        assert len(hash1) == 32

    def test_hash_codebook(self):
        codebook = np.random.randn(100, 32).astype(np.float32)

        hash1 = CommitmentChain.hash_codebook(codebook, step=0)
        hash2 = CommitmentChain.hash_codebook(codebook, step=0)

        assert hash1 == hash2
        assert len(hash1) == 32
