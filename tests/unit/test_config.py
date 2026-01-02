import tempfile
from pathlib import Path

import pytest

from holograd.core.config import (
    HoloGradConfig,
    ProtocolConfig,
    ADCConfig,
    VerificationConfig,
    AggregationConfig,
    get_default_config,
)


class TestProtocolConfig:
    def test_default_values(self):
        config = ProtocolConfig()
        assert config.K == 64
        assert config.learning_rate == 3e-4
        assert config.gradient_method == "jvp"

    def test_custom_values(self):
        config = ProtocolConfig(K=128, learning_rate=1e-4)
        assert config.K == 128
        assert config.learning_rate == 1e-4


class TestADCConfig:
    def test_default_values(self):
        config = ADCConfig()
        assert config.rank == 32
        assert config.oja_alpha == 1e-3
        assert config.qr_period == 100
        assert config.enabled is True

    def test_disabled_adc(self):
        config = ADCConfig(enabled=False)
        assert config.enabled is False


class TestVerificationConfig:
    def test_default_values(self):
        config = VerificationConfig()
        assert config.p_verify == 0.05
        assert config.epsilon == 1e-4


class TestHoloGradConfig:
    def test_default_config(self):
        config = get_default_config()
        assert config.protocol.K == 64
        assert config.adc.rank == 32
        assert config.verification.p_verify == 0.05
        assert config.aggregation.tau == 0.1
        assert config.seed == 42

    def test_yaml_roundtrip(self):
        config = get_default_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            loaded = HoloGradConfig.from_yaml(yaml_path)

            assert loaded.protocol.K == config.protocol.K
            assert loaded.adc.rank == config.adc.rank
            assert loaded.verification.epsilon == config.verification.epsilon
            assert loaded.seed == config.seed

    def test_override(self):
        config = get_default_config()
        new_config = config.override(**{"protocol.K": 128, "adc.rank": 64})

        assert new_config.protocol.K == 128
        assert new_config.adc.rank == 64
        assert config.protocol.K == 64

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            HoloGradConfig.from_yaml("/nonexistent/path.yaml")
