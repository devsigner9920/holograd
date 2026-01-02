import numpy as np
import pytest

from holograd.protocol.aggregation import RobustAggregator, AggregationResult


class TestRobustAggregator:
    def test_mean_aggregation(self):
        agg = RobustAggregator(tau=0.0, method="mean")

        scalars = [1.0, 2.0, 3.0, 4.0]
        directions = [np.array([1, 0, 0], dtype=np.float32) for _ in range(4)]

        result = agg.aggregate(scalars, directions, scale_factor=1.0)

        expected = np.array([2.5, 0, 0], dtype=np.float32)
        np.testing.assert_allclose(result.gradient, expected)
        assert result.proofs_used == 4
        assert result.proofs_trimmed == 0

    def test_trimmed_mean_removes_outliers(self):
        agg = RobustAggregator(tau=0.2, method="trimmed_mean")

        scalars = [1.0, 2.0, 3.0, 4.0, 100.0]
        directions = [np.array([1, 0], dtype=np.float32) for _ in range(5)]

        result = agg.aggregate(scalars, directions, scale_factor=1.0)

        assert result.proofs_trimmed == 2
        assert result.proofs_used == 3
        assert 100.0 not in result.scalars_trimmed
        assert 1.0 not in result.scalars_trimmed

    def test_scale_factor_applied(self):
        agg = RobustAggregator(tau=0.0, method="mean")

        scalars = [1.0, 1.0]
        directions = [np.array([1, 0], dtype=np.float32) for _ in range(2)]

        result = agg.aggregate(scalars, directions, scale_factor=10.0)

        expected = np.array([10.0, 0], dtype=np.float32)
        np.testing.assert_allclose(result.gradient, expected)

    def test_trimmed_indices_tracked(self):
        agg = RobustAggregator(tau=0.2, method="trimmed_mean")

        scalars = [5.0, 1.0, 3.0, 2.0, 4.0]
        directions = [np.array([1], dtype=np.float32) for _ in range(5)]

        result = agg.aggregate(scalars, directions, scale_factor=1.0)

        assert len(result.trimmed_indices) == 2
        sorted_scalars = sorted(enumerate(scalars), key=lambda x: x[1])
        lowest_idx = sorted_scalars[0][0]
        highest_idx = sorted_scalars[-1][0]
        assert lowest_idx in result.trimmed_indices
        assert highest_idx in result.trimmed_indices

    def test_invalid_tau(self):
        with pytest.raises(ValueError):
            RobustAggregator(tau=0.5)

        with pytest.raises(ValueError):
            RobustAggregator(tau=-0.1)

    def test_empty_list_raises(self):
        agg = RobustAggregator()

        with pytest.raises(ValueError):
            agg.aggregate([], [], scale_factor=1.0)

    def test_mismatched_lengths_raises(self):
        agg = RobustAggregator()

        scalars = [1.0, 2.0]
        directions = [np.array([1], dtype=np.float32)]

        with pytest.raises(ValueError):
            agg.aggregate(scalars, directions, scale_factor=1.0)

    def test_median_aggregation(self):
        agg = RobustAggregator(method="median")

        scalars = [1.0, 2.0, 100.0]
        directions = [np.array([1, 0], dtype=np.float32) for _ in range(3)]

        result = agg.aggregate(scalars, directions, scale_factor=1.0)

        assert result.gradient[0] == pytest.approx(2.0)
