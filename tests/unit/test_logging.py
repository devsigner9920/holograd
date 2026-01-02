import json
import tempfile
from pathlib import Path

import pytest

from holograd.utils.logging import MetricsLogger, setup_logging


class TestMetricsLogger:
    def test_creates_experiment_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_to_tensorboard=False,
            )

            assert (Path(tmpdir) / "test_exp").exists()
            logger.close()

    def test_auto_generates_experiment_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                output_dir=tmpdir,
                log_to_tensorboard=False,
            )

            assert logger.experiment_name.startswith("holograd_")
            logger.close()

    def test_log_step_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_to_csv=True,
                log_to_json=False,
                log_to_tensorboard=False,
            )

            logger.log_step(step=1, metrics={"loss": 0.5, "accuracy": 0.8})
            logger.log_step(step=2, metrics={"loss": 0.4, "accuracy": 0.85})
            logger.close()

            csv_path = Path(tmpdir) / "test_exp" / "metrics.csv"
            assert csv_path.exists()

            content = csv_path.read_text()
            assert "loss" in content
            assert "0.5" in content

    def test_log_step_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_to_csv=False,
                log_to_json=True,
                log_to_tensorboard=False,
            )

            logger.log_step(step=1, metrics={"loss": 0.5})
            logger.close()

            json_path = Path(tmpdir) / "test_exp" / "metrics.json"
            assert json_path.exists()

            data = json.loads(json_path.read_text())
            assert data["experiment_name"] == "test_exp"
            assert len(data["metrics"]) == 1
            assert data["metrics"][0]["loss"] == 0.5

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with MetricsLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_to_tensorboard=False,
            ) as logger:
                logger.log_step(step=1, metrics={"loss": 0.5})

            json_path = Path(tmpdir) / "test_exp" / "metrics.json"
            assert json_path.exists()

    def test_log_with_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_to_csv=False,
                log_to_json=True,
                log_to_tensorboard=False,
            )

            logger.log_step(step=1, metrics={"loss": 0.5}, prefix="train")
            logger.close()

            json_path = Path(tmpdir) / "test_exp" / "metrics.json"
            data = json.loads(json_path.read_text())
            assert "train/loss" in data["metrics"][0]


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging()
        assert logger.name == "holograd"

    def test_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file)

            logger.info("test message")

            assert log_file.exists()
