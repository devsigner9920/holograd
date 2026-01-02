import numpy as np
import pytest

from holograd.gradient.backprop import BackpropGradient
from holograd.gradient.jvp import JVPGradient


def quadratic_loss(params, batch):
    return float(np.sum(params**2))


def linear_loss(params, batch):
    weights = batch
    return float(np.dot(params.flatten(), weights.flatten()))


class TestBackpropGradient:
    def test_numerical_gradient_quadratic(self):
        grad_computer = BackpropGradient(framework="numpy")

        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gradient = grad_computer.compute_full_gradient(params, quadratic_loss, None)

        expected = 2 * params
        np.testing.assert_allclose(gradient, expected, rtol=1e-2)

    def test_compute_projection(self):
        grad_computer = BackpropGradient(framework="numpy")

        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        result = grad_computer.compute_projection(params, direction, quadratic_loss, None)

        expected_grad = 2 * params
        expected_scalar = np.dot(expected_grad, direction)

        assert result.scalar == pytest.approx(expected_scalar, rel=1e-2)
        assert result.gradient is not None
        assert result.compute_time > 0

    def test_projection_linear_loss(self):
        grad_computer = BackpropGradient(framework="numpy")

        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        direction = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3)

        result = grad_computer.compute_projection(params, direction, linear_loss, weights)

        expected_scalar = np.dot(weights, direction)
        assert result.scalar == pytest.approx(expected_scalar, rel=1e-2)


class TestJVPGradient:
    def test_numerical_jvp_quadratic(self):
        jvp_computer = JVPGradient(framework="numpy")

        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        result = jvp_computer.compute_projection(params, direction, quadratic_loss, None)

        expected = 2.0
        assert result.scalar == pytest.approx(expected, rel=1e-2)

    def test_jvp_matches_backprop(self):
        backprop = BackpropGradient(framework="numpy")
        jvp = JVPGradient(framework="numpy")

        np.random.seed(42)
        params = np.random.randn(50).astype(np.float32)
        direction = np.random.randn(50).astype(np.float32)
        direction = direction / np.linalg.norm(direction)

        backprop_result = backprop.compute_projection(params, direction, quadratic_loss, None)
        jvp_result = jvp.compute_projection(params, direction, quadratic_loss, None)

        assert jvp_result.scalar == pytest.approx(backprop_result.scalar, rel=0.2)

    def test_jvp_no_full_gradient(self):
        jvp_computer = JVPGradient(framework="numpy")

        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        result = jvp_computer.compute_projection(params, direction, quadratic_loss, None)

        assert result.gradient is None

    def test_jvp_memory_efficient(self):
        backprop = BackpropGradient(framework="numpy")
        jvp = JVPGradient(framework="numpy")

        params = np.random.randn(1000).astype(np.float32)
        direction = np.random.randn(1000).astype(np.float32)

        backprop_result = backprop.compute_projection(params, direction, quadratic_loss, None)
        jvp_result = jvp.compute_projection(params, direction, quadratic_loss, None)

        assert jvp_result.memory_bytes <= backprop_result.memory_bytes


class TestGradientConsistency:
    def test_multiple_directions(self):
        backprop = BackpropGradient(framework="numpy")
        jvp = JVPGradient(framework="numpy")

        np.random.seed(123)
        params = np.random.randn(20).astype(np.float32)

        for _ in range(5):
            direction = np.random.randn(20).astype(np.float32)
            direction = direction / np.linalg.norm(direction)

            bp_result = backprop.compute_projection(params, direction, quadratic_loss, None)
            jvp_result = jvp.compute_projection(params, direction, quadratic_loss, None)

            assert jvp_result.scalar == pytest.approx(bp_result.scalar, rel=0.2)

    def test_zero_direction(self):
        jvp = JVPGradient(framework="numpy")

        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        direction = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        result = jvp.compute_projection(params, direction, quadratic_loss, None)

        assert result.scalar == pytest.approx(0.0, abs=1e-6)
