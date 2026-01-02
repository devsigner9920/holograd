"""
Metrics logging infrastructure for HoloGrad.

Supports multiple output formats:
- CSV: Tabular data for analysis
- JSON: Structured data with full metadata
- TensorBoard: Real-time visualization

Each run gets a unique experiment ID for tracking.
"""

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import asdict

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup Python logging for HoloGrad.

    Args:
        level: Logging level (default INFO)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("holograd")
    logger.setLevel(level)

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """
    Unified metrics logging to CSV, JSON, and TensorBoard.

    Usage:
        logger = MetricsLogger(output_dir="outputs", experiment_name="run_001")
        logger.log_step(step=1, metrics={"loss": 0.5, "accuracy": 0.8})
        logger.log_scalar("learning_rate", 0.001, step=1)
        logger.close()
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs",
        experiment_name: Optional[str] = None,
        log_to_csv: bool = True,
        log_to_json: bool = True,
        log_to_tensorboard: bool = True,
    ):
        """
        Initialize metrics logger.

        Args:
            output_dir: Base directory for outputs
            experiment_name: Name for this experiment (auto-generated if None)
            log_to_csv: Enable CSV logging
            log_to_json: Enable JSON logging
            log_to_tensorboard: Enable TensorBoard logging
        """
        self.output_dir = Path(output_dir)

        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"holograd_{timestamp}"
        self.experiment_name = experiment_name

        # Create experiment directory
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize outputs
        self.log_to_csv = log_to_csv
        self.log_to_json = log_to_json
        self.log_to_tensorboard = log_to_tensorboard and TENSORBOARD_AVAILABLE

        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fields: Optional[list[str]] = None

        self._json_data: list[dict[str, Any]] = []

        self._tb_writer: Optional[Any] = None
        if self.log_to_tensorboard:
            tb_dir = self.experiment_dir / "tensorboard"
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))

        # Setup Python logger
        self._logger = setup_logging(log_file=self.experiment_dir / "training.log")

        # Metadata
        self._start_time = datetime.now()
        self._step_count = 0

    def log_step(
        self,
        step: int,
        metrics: dict[str, Any],
        prefix: str = "",
    ) -> None:
        """
        Log metrics for a training step.

        Args:
            step: Current step number
            metrics: Dictionary of metric name -> value
            prefix: Optional prefix for metric names
        """
        self._step_count = step

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Add step and timestamp
        record = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}

        # CSV logging
        if self.log_to_csv:
            self._log_csv(record)

        # JSON logging (accumulated)
        if self.log_to_json:
            self._json_data.append(record)

        # TensorBoard logging
        if self.log_to_tensorboard and self._tb_writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(name, value, step)

    def log_scalar(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a single scalar value."""
        step = step if step is not None else self._step_count

        if self.log_to_tensorboard and self._tb_writer is not None:
            self._tb_writer.add_scalar(name, value, step)

    def log_histogram(
        self,
        name: str,
        values: Any,
        step: Optional[int] = None,
    ) -> None:
        """Log histogram of values."""
        step = step if step is not None else self._step_count

        if self.log_to_tensorboard and self._tb_writer is not None:
            self._tb_writer.add_histogram(name, values, step)

    def log_config(self, config: Any) -> None:
        """Log configuration to file."""
        config_path = self.experiment_dir / "config.yaml"

        if hasattr(config, "to_yaml"):
            config.to_yaml(config_path)
        elif hasattr(config, "_to_dict"):
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(config._to_dict(), f, default_flow_style=False)
        else:
            # Try to convert dataclass
            try:
                config_dict = asdict(config)
                import yaml

                with open(config_path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            except Exception as e:
                self._logger.warning(f"Could not save config: {e}")

    def log_message(self, message: str, level: str = "info") -> None:
        """Log a text message."""
        log_func = getattr(self._logger, level.lower(), self._logger.info)
        log_func(message)

    def _log_csv(self, record: dict[str, Any]) -> None:
        """Write record to CSV file."""
        # Initialize CSV on first write
        if self._csv_writer is None:
            csv_path = self.experiment_dir / "metrics.csv"
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_fields = list(record.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_fields, extrasaction="ignore"
            )
            self._csv_writer.writeheader()

        # Handle new fields
        new_fields = set(record.keys()) - set(self._csv_fields or [])
        if new_fields:
            self._logger.warning(
                f"New fields not in CSV header: {new_fields}. Consider restarting to include them."
            )

        self._csv_writer.writerow(record)
        self._csv_file.flush()

    def save_json(self) -> None:
        """Save accumulated JSON data to file."""
        if not self.log_to_json or not self._json_data:
            return

        json_path = self.experiment_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "experiment_name": self.experiment_name,
                    "start_time": self._start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_steps": self._step_count,
                    "metrics": self._json_data,
                },
                f,
                indent=2,
            )

    def close(self) -> None:
        """Close all output handles and save final data."""
        # Save JSON
        self.save_json()

        # Close CSV
        if self._csv_file is not None:
            self._csv_file.close()

        # Close TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.close()

        self._logger.info(
            f"Experiment '{self.experiment_name}' completed. Total steps: {self._step_count}"
        )

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
