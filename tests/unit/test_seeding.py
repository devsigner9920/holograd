import numpy as np
import pytest

from holograd.utils.seeding import (
    get_deterministic_seed,
    hash_parameters,
    hash_batch,
    hash_codebook,
    seed_to_rng_key,
    set_seed,
)


class TestDeterministicSeed:
    def test_same_inputs_same_output(self):
        seed1 = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        seed2 = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        assert seed1 == seed2

    def test_different_step_different_seed(self):
        seed1 = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        seed2 = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=1,
            worker_id=0,
        )
        assert seed1 != seed2

    def test_different_worker_different_seed(self):
        seed1 = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        seed2 = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=0,
            worker_id=1,
        )
        assert seed1 != seed2

    def test_seed_length(self):
        seed = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        assert len(seed) == 32

    def test_with_codebook_hash(self):
        seed_without = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=0,
            worker_id=0,
        )
        seed_with = get_deterministic_seed(
            global_seed="test_seed",
            param_hash=b"\x00" * 32,
            batch_hash=b"\x01" * 32,
            step=0,
            worker_id=0,
            codebook_hash=b"\x02" * 32,
        )
        assert seed_without != seed_with


class TestHashParameters:
    def test_deterministic(self):
        params = np.random.randn(100).astype(np.float32)
        hash1 = hash_parameters(params)
        hash2 = hash_parameters(params)
        assert hash1 == hash2

    def test_different_params_different_hash(self):
        params1 = np.zeros(100, dtype=np.float32)
        params2 = np.ones(100, dtype=np.float32)
        hash1 = hash_parameters(params1)
        hash2 = hash_parameters(params2)
        assert hash1 != hash2

    def test_dict_params(self):
        params = {
            "layer1": np.zeros(10, dtype=np.float32),
            "layer2": np.ones(10, dtype=np.float32),
        }
        hash1 = hash_parameters(params)
        hash2 = hash_parameters(params)
        assert hash1 == hash2
        assert len(hash1) == 32


class TestHashBatch:
    def test_deterministic(self):
        indices = np.array([0, 5, 10, 15])
        hash1 = hash_batch(indices, batch_seed=42)
        hash2 = hash_batch(indices, batch_seed=42)
        assert hash1 == hash2

    def test_different_seed_different_hash(self):
        indices = np.array([0, 5, 10, 15])
        hash1 = hash_batch(indices, batch_seed=42)
        hash2 = hash_batch(indices, batch_seed=43)
        assert hash1 != hash2


class TestHashCodebook:
    def test_deterministic(self):
        codebook = np.random.randn(100, 32).astype(np.float32)
        hash1 = hash_codebook(codebook, step=0)
        hash2 = hash_codebook(codebook, step=0)
        assert hash1 == hash2

    def test_different_step_different_hash(self):
        codebook = np.random.randn(100, 32).astype(np.float32)
        hash1 = hash_codebook(codebook, step=0)
        hash2 = hash_codebook(codebook, step=1)
        assert hash1 != hash2


class TestSeedToRngKey:
    def test_produces_generator(self):
        seed = b"\x00" * 32
        rng = seed_to_rng_key(seed)
        assert isinstance(rng, np.random.Generator)

    def test_same_seed_same_sequence(self):
        seed = b"\x01\x02\x03" + b"\x00" * 29
        rng1 = seed_to_rng_key(seed)
        rng2 = seed_to_rng_key(seed)

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]
        assert values1 == values2
