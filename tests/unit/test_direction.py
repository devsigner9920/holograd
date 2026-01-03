import numpy as np
import pytest

from holograd.protocol.direction import DirectionGenerator, ADCCodebook


class TestDirectionGenerator:
    def test_generates_unit_norm(self):
        gen = DirectionGenerator(dimension=1000)
        seed = b"\x00" * 32

        result = gen.generate(seed)

        norm = np.linalg.norm(result.direction)
        assert abs(norm - 1.0) < 1e-6

    def test_deterministic_generation(self):
        gen = DirectionGenerator(dimension=1000)
        seed = b"\x01\x02\x03" + b"\x00" * 29

        result1 = gen.generate(seed)
        result2 = gen.generate(seed)

        np.testing.assert_array_equal(result1.direction, result2.direction)

    def test_different_seeds_different_directions(self):
        gen = DirectionGenerator(dimension=1000)

        result1 = gen.generate(b"\x00" * 32)
        result2 = gen.generate(b"\x01" * 32)

        assert not np.allclose(result1.direction, result2.direction)

    def test_sigma_squared(self):
        dimension = 1000
        gen = DirectionGenerator(dimension=dimension)

        assert gen.sigma_squared == pytest.approx(1.0 / dimension)
        assert gen.scale_factor == pytest.approx(dimension)

    def test_chunked_generation_matches_regular(self):
        gen = DirectionGenerator(dimension=10000)
        seed = b"\x05" * 32

        result_regular = gen.generate(seed)
        result_chunked = gen.generate_chunked(seed, chunk_size=1000)

        np.testing.assert_allclose(
            result_regular.direction,
            result_chunked.direction,
            rtol=1e-5,
        )

    def test_isotropic_distribution(self):
        gen = DirectionGenerator(dimension=100)
        n_samples = 1000

        directions = []
        for i in range(n_samples):
            seed = i.to_bytes(4, "big") + b"\x00" * 28
            result = gen.generate(seed)
            directions.append(result.direction)

        directions = np.stack(directions)
        cov = directions.T @ directions / n_samples

        expected = gen.sigma_squared * np.eye(100)
        np.testing.assert_allclose(cov, expected, atol=0.1)


class TestADCCodebook:
    def test_initialization_orthonormal(self):
        codebook = ADCCodebook(dimension=100, rank=32)
        U = codebook.codebook

        UtU = U.T @ U
        np.testing.assert_allclose(UtU, np.eye(32), atol=1e-3)

    def test_generate_direction_returns_z(self):
        codebook = ADCCodebook(dimension=100, rank=32)
        seed = b"\x00" * 32

        result = codebook.generate_direction(seed)

        assert result.z_projection is not None
        assert result.z_projection.shape == (32,)
        assert result.direction.shape == (100,)

    def test_reconstruct_matches_generate(self):
        codebook = ADCCodebook(dimension=100, rank=32)
        seed = b"\x00" * 32

        result = codebook.generate_direction(seed)
        reconstructed = codebook.reconstruct_direction(result.z_projection)

        np.testing.assert_allclose(result.direction, reconstructed, rtol=1e-5)

    def test_update_changes_codebook(self):
        codebook = ADCCodebook(dimension=100, rank=32)
        U_before = codebook.codebook.copy()

        gradient = np.random.randn(100).astype(np.float32)
        codebook.update(gradient)

        assert not np.allclose(codebook.codebook, U_before)

    def test_qr_maintains_orthonormality(self):
        codebook = ADCCodebook(dimension=100, rank=32, qr_period=10)

        for i in range(20):
            gradient = np.random.randn(100).astype(np.float32)
            codebook.update(gradient)

        U = codebook.codebook
        UtU = U.T @ U
        np.testing.assert_allclose(UtU, np.eye(32), atol=1e-3)

    def test_captured_energy_ratio(self):
        codebook = ADCCodebook(dimension=100, rank=32)

        gradient = np.random.randn(100).astype(np.float32)
        ratio = codebook.captured_energy_ratio(gradient)

        assert 0.0 <= ratio <= 1.0

    def test_captured_energy_increases_with_updates(self):
        np.random.seed(42)
        codebook = ADCCodebook(dimension=100, rank=32, qr_period=100)

        gradient_direction = np.random.randn(100).astype(np.float32)
        gradient_direction = gradient_direction / np.linalg.norm(gradient_direction)

        initial_ratio = codebook.captured_energy_ratio(gradient_direction)

        for i in range(50):
            noise = 0.1 * np.random.randn(100).astype(np.float32)
            gradient = gradient_direction + noise
            codebook.update(gradient)

        final_ratio = codebook.captured_energy_ratio(gradient_direction)

        assert final_ratio > initial_ratio

    def test_reset(self):
        codebook = ADCCodebook(dimension=100, rank=32)

        for i in range(10):
            gradient = np.random.randn(100).astype(np.float32)
            codebook.update(gradient)

        assert codebook.step == 10

        codebook.reset(seed=123)

        assert codebook.step == 0

        U = codebook.codebook
        UtU = U.T @ U
        np.testing.assert_allclose(UtU, np.eye(32), atol=1e-3)

    def test_warmup_accumulates_gradients(self):
        codebook = ADCCodebook(dimension=100, rank=32, warmup_samples=5)

        assert not codebook.is_warmed_up

        for i in range(4):
            gradient = np.random.randn(100).astype(np.float32)
            codebook.update(gradient)
            assert not codebook.is_warmed_up

        gradient = np.random.randn(100).astype(np.float32)
        codebook.update(gradient)
        assert codebook.is_warmed_up

    def test_warmup_svd_initialization(self):
        np.random.seed(42)
        codebook = ADCCodebook(dimension=100, rank=8, warmup_samples=10)

        gradient_direction = np.random.randn(100).astype(np.float32)
        gradient_direction = gradient_direction / np.linalg.norm(gradient_direction)

        for i in range(10):
            noise = 0.1 * np.random.randn(100).astype(np.float32)
            gradient = gradient_direction + noise
            codebook.update(gradient)

        assert codebook.is_warmed_up
        energy = codebook.captured_energy_ratio(gradient_direction)
        assert energy > 0.5

    def test_adaptive_alpha_decay(self):
        codebook = ADCCodebook(
            dimension=100,
            rank=32,
            oja_alpha=0.1,
            alpha_decay=0.9,
            alpha_min=0.001,
        )

        initial_alpha = codebook.current_alpha
        assert initial_alpha == 0.1

        for i in range(20):
            gradient = np.random.randn(100).astype(np.float32)
            codebook.update(gradient)

        assert codebook.current_alpha < initial_alpha
        assert codebook.current_alpha >= 0.001

    def test_power_iteration_mode(self):
        np.random.seed(42)
        codebook = ADCCodebook(
            dimension=100,
            rank=8,
            oja_alpha=0.5,
            qr_period=100,
            use_power_iteration=True,
            power_iteration_steps=3,
        )

        gradient_direction = np.random.randn(100).astype(np.float32)
        gradient_direction = gradient_direction / np.linalg.norm(gradient_direction)

        initial_energy = codebook.captured_energy_ratio(gradient_direction)

        for i in range(20):
            noise = 0.1 * np.random.randn(100).astype(np.float32)
            gradient = gradient_direction + noise
            codebook.update(gradient)

        final_energy = codebook.captured_energy_ratio(gradient_direction)
        assert final_energy > initial_energy

    def test_energy_ema_tracking(self):
        codebook = ADCCodebook(dimension=100, rank=32)

        for i in range(10):
            gradient = np.random.randn(100).astype(np.float32)
            codebook.update(gradient)

        assert codebook.energy_ema > 0
