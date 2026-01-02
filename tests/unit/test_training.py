import numpy as np
import pytest

from holograd.training.model import SimpleGPT2, ParameterManager
from holograd.training.data import SyntheticDataset, DataLoader, create_synthetic_data


class TestParameterManager:
    def test_flatten_unflatten_roundtrip(self):
        shapes = {"a": (10, 5), "b": (20,), "c": (3, 3, 3)}
        manager = ParameterManager(shapes)

        params = {
            "a": np.random.randn(10, 5).astype(np.float32),
            "b": np.random.randn(20).astype(np.float32),
            "c": np.random.randn(3, 3, 3).astype(np.float32),
        }

        flat = manager.flatten(params)
        recovered = manager.unflatten(flat)

        for key in params:
            np.testing.assert_array_equal(params[key], recovered[key])

    def test_total_parameters(self):
        shapes = {"a": (10, 5), "b": (20,)}
        manager = ParameterManager(shapes)

        assert manager.total_parameters == 50 + 20


class TestSimpleGPT2:
    def test_small_model_creation(self):
        model = SimpleGPT2(size="small", max_seq_len=128)

        assert model.n_layer == 12
        assert model.n_head == 12
        assert model.n_embd == 768
        assert model.num_parameters > 0

    def test_forward_pass(self):
        model = SimpleGPT2(size="small", max_seq_len=128)

        batch_size = 2
        seq_len = 32
        input_ids = np.random.randint(0, model.vocab_size, (batch_size, seq_len))

        logits = model.forward(input_ids)

        assert logits.shape == (batch_size, seq_len, model.vocab_size)

    def test_compute_loss(self):
        model = SimpleGPT2(size="small", max_seq_len=128)

        batch_size = 2
        seq_len = 32
        input_ids = np.random.randint(0, model.vocab_size, (batch_size, seq_len))
        labels = np.random.randint(0, model.vocab_size, (batch_size, seq_len))

        loss = model.compute_loss(input_ids, labels)

        assert np.isfinite(loss)
        assert loss > 0

    def test_flat_params_roundtrip(self):
        model = SimpleGPT2(size="small", max_seq_len=128)

        original_params = model.get_flat_params().copy()

        new_params = np.random.randn(model.num_parameters).astype(np.float32)
        model.set_flat_params(new_params)

        model.set_flat_params(original_params)
        recovered = model.get_flat_params()

        np.testing.assert_array_equal(original_params, recovered)


class TestSyntheticDataset:
    def test_creation(self):
        dataset = SyntheticDataset(
            vocab_size=1000,
            num_samples=100,
            seq_length=64,
        )

        assert len(dataset) == 100

    def test_getitem(self):
        dataset = SyntheticDataset(
            vocab_size=1000,
            num_samples=100,
            seq_length=64,
        )

        input_ids, labels = dataset[0]

        assert input_ids.shape == (64,)
        assert labels.shape == (64,)
        assert input_ids.dtype == np.int64

    def test_get_batch(self):
        dataset = SyntheticDataset(
            vocab_size=1000,
            num_samples=100,
            seq_length=64,
        )

        indices = np.array([0, 5, 10])
        input_ids, labels = dataset.get_batch(indices)

        assert input_ids.shape == (3, 64)
        assert labels.shape == (3, 64)


class TestDataLoader:
    def test_iteration(self):
        dataset = SyntheticDataset(num_samples=100, seq_length=32)
        loader = DataLoader(dataset, batch_size=8)

        batches = list(loader)

        assert len(batches) == 100 // 8
        assert batches[0].input_ids.shape[0] == 8

    def test_deterministic_batch(self):
        dataset = SyntheticDataset(num_samples=100, seq_length=32, seed=42)
        loader = DataLoader(dataset, batch_size=8, seed=42)

        batch1 = loader.get_deterministic_batch(step=5)
        batch2 = loader.get_deterministic_batch(step=5)

        np.testing.assert_array_equal(batch1.indices, batch2.indices)
        np.testing.assert_array_equal(batch1.input_ids, batch2.input_ids)

    def test_different_steps_different_batches(self):
        dataset = SyntheticDataset(num_samples=100, seq_length=32)
        loader = DataLoader(dataset, batch_size=8)

        batch1 = loader.get_deterministic_batch(step=5)
        batch2 = loader.get_deterministic_batch(step=6)

        assert not np.array_equal(batch1.indices, batch2.indices)


class TestCreateSyntheticData:
    def test_creates_loaders(self):
        train_loader, val_loader = create_synthetic_data(
            num_train_samples=100,
            num_val_samples=20,
            batch_size=8,
        )

        assert len(train_loader) == 100 // 8
        assert len(val_loader) == 20 // 8
