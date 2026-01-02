from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime

import numpy as np
import yaml

from holograd.core.config import HoloGradConfig
from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data
from holograd.training.trainer import HoloGradTrainer
from holograd.utils.logging import MetricsLogger


@dataclass
class ExperimentConfig:
    name: str
    base_config: HoloGradConfig
    sweep_params: Dict[str, List[Any]] = field(default_factory=dict)
    num_runs_per_config: int = 1
    max_steps: int = 100
    output_dir: Path = field(default_factory=lambda: Path("experiments"))


@dataclass
class ExperimentResult:
    config_id: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    run_time: float


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []
        self._start_time: Optional[float] = None

    def _generate_configs(self) -> List[Dict[str, Any]]:
        if not self.config.sweep_params:
            return [{}]

        configs = [{}]
        for param_name, values in self.config.sweep_params.items():
            new_configs = []
            for config in configs:
                for value in values:
                    new_config = config.copy()
                    new_config[param_name] = value
                    new_configs.append(new_config)
            configs = new_configs

        return configs

    def _apply_params(
        self,
        base_config: HoloGradConfig,
        params: Dict[str, Any],
    ) -> HoloGradConfig:
        if not params:
            return base_config
        return base_config.override(**params)

    def run_single(
        self,
        config: HoloGradConfig,
        config_id: str,
        params: Dict[str, Any],
    ) -> ExperimentResult:
        run_start = time.perf_counter()

        model = SimpleGPT2(
            size=config.training.model_size,
            max_seq_len=config.training.sequence_length * 2,
            seed=config.seed,
        )

        train_loader, val_loader = create_synthetic_data(
            vocab_size=model.vocab_size,
            num_train_samples=1000,
            num_val_samples=100,
            seq_length=config.training.sequence_length,
            batch_size=config.training.batch_size,
            seed=config.seed,
        )

        exp_name = f"{self.config.name}_{config_id}_{datetime.now().strftime('%H%M%S')}"
        logger = MetricsLogger(
            output_dir=self.config.output_dir / "runs",
            experiment_name=exp_name,
            log_to_tensorboard=False,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
        )

        train_results = trainer.train(num_steps=self.config.max_steps)

        logger.close()

        run_time = time.perf_counter() - run_start

        return ExperimentResult(
            config_id=config_id,
            params=params,
            metrics=train_results,
            run_time=run_time,
        )

    def run_all(self) -> List[ExperimentResult]:
        self._start_time = time.perf_counter()
        self.results = []

        param_configs = self._generate_configs()
        total_runs = len(param_configs) * self.config.num_runs_per_config

        run_idx = 0
        for param_idx, params in enumerate(param_configs):
            for run_num in range(self.config.num_runs_per_config):
                config_id = f"config_{param_idx:03d}_run_{run_num:02d}"

                run_config = self._apply_params(self.config.base_config, params)
                run_config = run_config.override(seed=self.config.base_config.seed + run_num)

                result = self.run_single(run_config, config_id, params)
                self.results.append(result)

                run_idx += 1

        return self.results

    def aggregate_results(self) -> Dict[str, Dict[str, float]]:
        if not self.results:
            return {}

        grouped: Dict[str, List[ExperimentResult]] = {}
        for result in self.results:
            param_key = json.dumps(result.params, sort_keys=True)
            if param_key not in grouped:
                grouped[param_key] = []
            grouped[param_key].append(result)

        aggregated = {}
        for param_key, results in grouped.items():
            metrics_lists: Dict[str, List[float]] = {}
            for result in results:
                for metric_name, value in result.metrics.items():
                    if metric_name not in metrics_lists:
                        metrics_lists[metric_name] = []
                    metrics_lists[metric_name].append(value)

            aggregated[param_key] = {
                "params": results[0].params,
                "num_runs": len(results),
            }
            for metric_name, values in metrics_lists.items():
                aggregated[param_key][f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[param_key][f"{metric_name}_std"] = float(np.std(values))

        return aggregated

    def save_results(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self.config.output_dir / f"{self.config.name}_results.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "experiment_name": self.config.name,
            "sweep_params": self.config.sweep_params,
            "results": [
                {
                    "config_id": r.config_id,
                    "params": r.params,
                    "metrics": r.metrics,
                    "run_time": r.run_time,
                }
                for r in self.results
            ],
            "aggregated": self.aggregate_results(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def create_ablation_a_config(base_config: HoloGradConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name="ablation_a_proof_count",
        base_config=base_config,
        sweep_params={"protocol.K": [8, 16, 32, 64, 128]},
        max_steps=50,
    )


def create_ablation_b_config(base_config: HoloGradConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name="ablation_b_codebook_rank",
        base_config=base_config,
        sweep_params={"adc.rank": [8, 16, 32, 64]},
        max_steps=50,
    )


def create_ablation_d_config(base_config: HoloGradConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name="ablation_d_adversary",
        base_config=base_config,
        sweep_params={"distributed.byzantine_fraction": [0.0, 0.1, 0.2]},
        max_steps=50,
    )


def create_ablation_e_config(base_config: HoloGradConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name="ablation_e_verify_rate",
        base_config=base_config,
        sweep_params={"verification.p_verify": [0.0, 0.01, 0.05, 0.1]},
        max_steps=50,
    )
