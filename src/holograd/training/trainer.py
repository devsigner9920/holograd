from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch.autograd.functional import jvp

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    jvp = None  # type: ignore
    TORCH_AVAILABLE = False

from holograd.core.config import HoloGradConfig
from holograd.core.types import StepMetrics
from holograd.protocol.commitment import CommitmentChain
from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.protocol.aggregation import RobustAggregator
from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
from holograd.distributed.worker import Worker, WorkerConfig
from holograd.distributed.simulation import WorkerPool, DelayConfig
from holograd.verification.verifier import Verifier
from holograd.training.model import SimpleGPT2, ParameterManager
from holograd.training.data import DataLoader, BatchData
from holograd.utils.logging import MetricsLogger


@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    total_tokens: int = 0
    best_val_loss: float = float("inf")
    training_losses: List[float] = field(default_factory=list)


class HoloGradTrainer:
    def __init__(
        self,
        config: HoloGradConfig,
        model: SimpleGPT2,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        logger: Optional[MetricsLogger] = None,
        device: Optional[str] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.device: Optional[str] = None
        if TORCH_AVAILABLE:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

        self.state = TrainerState()

        use_momentum_centric = config.protocol.direction_mode == "momentum"

        self._coordinator = Coordinator(
            CoordinatorConfig(
                dimension=model.num_parameters,
                num_workers=config.distributed.num_workers,
                proofs_per_step=config.protocol.K,
                global_seed=config.protocol.global_seed,
                use_adc=config.adc.enabled and not use_momentum_centric,
                adc_rank=config.adc.rank,
                adc_oja_alpha=config.adc.oja_alpha,
                adc_qr_period=config.adc.qr_period,
                adc_warmup_samples=config.adc.warmup_samples,
                adc_alpha_decay=config.adc.alpha_decay,
                adc_alpha_min=config.adc.alpha_min,
                adc_use_power_iteration=config.adc.use_power_iteration,
                adc_power_iteration_steps=config.adc.power_iteration_steps,
                tau=config.aggregation.tau,
                learning_rate=config.protocol.learning_rate,
                max_grad_norm=config.training.max_grad_norm,
                momentum=config.protocol.momentum,
                use_momentum_centric=use_momentum_centric,
                momentum_beta=config.momentum_centric.beta,
                momentum_warmup_steps=config.momentum_centric.warmup_steps,
                grad_norm_ema_alpha=config.momentum_centric.grad_norm_ema_alpha,
            )
        )

        delay_config = DelayConfig(
            enabled=config.distributed.simulate_delays,
            distribution=config.distributed.delay_distribution,
            mean=config.distributed.delay_mean,
            std=config.distributed.delay_std,
        )

        self._worker_pool = WorkerPool(
            num_workers=config.distributed.num_workers,
            dimension=model.num_parameters,
            use_adc=config.adc.enabled,
            adc_rank=config.adc.rank,
            delay_config=delay_config,
        )

        self._verifier = Verifier(
            dimension=model.num_parameters,
            p_verify=config.verification.p_verify,
            epsilon=config.verification.epsilon,
            use_adc=config.adc.enabled,
            adc_rank=config.adc.rank,
        )

        if config.adc.enabled and self._coordinator.codebook is not None:
            self._worker_pool.set_codebook(self._coordinator.codebook)
            self._verifier.set_codebook(self._coordinator.codebook)

        self._current_batch: Optional[BatchData] = None

    def _compute_gradient(self, batch: BatchData) -> NDArray[np.float32]:
        input_ids, labels = batch.input_ids, batch.labels
        params = self.model.get_flat_params()

        eps = 1e-3
        gradient = np.zeros_like(params)

        base_loss = self.model.compute_loss(input_ids, labels)

        sample_size = min(1000, len(params))
        indices = np.random.choice(len(params), sample_size, replace=False)

        for idx in indices:
            params_plus = params.copy()
            params_plus[idx] += eps
            self.model.set_flat_params(params_plus)
            loss_plus = self.model.compute_loss(input_ids, labels)
            gradient[idx] = (loss_plus - base_loss) / eps

        self.model.set_flat_params(params)
        return gradient

    def _create_gradient_fn(
        self,
        batch: BatchData,
    ) -> Callable[[NDArray[np.float32]], float]:
        if TORCH_AVAILABLE:
            return self._create_gradient_fn_torch(batch)
        return self._create_gradient_fn_finite_diff(batch)

    def _create_gradient_fn_torch(
        self,
        batch: BatchData,
    ) -> Callable[[NDArray[np.float32]], float]:
        input_ids_np, labels_np = batch.input_ids, batch.labels
        params_np = self.model.get_flat_params()
        device = self.device

        input_ids_t = torch.tensor(input_ids_np, dtype=torch.long, device=device)
        labels_t = torch.tensor(labels_np, dtype=torch.long, device=device)

        def gradient_fn(direction: NDArray[np.float32]) -> float:
            params_t = torch.tensor(params_np, dtype=torch.float32, device=device)
            direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

            def loss_fn(flat_params: torch.Tensor) -> torch.Tensor:
                params_dict = self.model.flat_params_to_torch_dict(flat_params)
                return self.model.compute_loss_torch(input_ids_t, labels_t, params_dict)

            _, jvp_value = jvp(loss_fn, (params_t,), (direction_t,))
            return float(jvp_value)

        return gradient_fn

    def _create_gradient_fn_finite_diff(
        self,
        batch: BatchData,
    ) -> Callable[[NDArray[np.float32]], float]:
        input_ids, labels = batch.input_ids, batch.labels
        params = self.model.get_flat_params()

        eps = 1e-4

        def gradient_fn(direction: NDArray[np.float32]) -> float:
            params_plus = params + eps * direction
            self.model.set_flat_params(params_plus)
            loss_plus = self.model.compute_loss(input_ids, labels)

            params_minus = params - eps * direction
            self.model.set_flat_params(params_minus)
            loss_minus = self.model.compute_loss(input_ids, labels)

            self.model.set_flat_params(params)
            return (loss_plus - loss_minus) / (2 * eps)

        return gradient_fn

    def _compute_true_gradient(self, batch: BatchData) -> NDArray[np.float32]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for true gradient computation")

        device = self.device
        input_ids_t = torch.tensor(batch.input_ids, dtype=torch.long, device=device)
        labels_t = torch.tensor(batch.labels, dtype=torch.long, device=device)
        params_np = self.model.get_flat_params()
        params_t = torch.tensor(params_np, dtype=torch.float32, device=device, requires_grad=True)

        params_dict = self.model.flat_params_to_torch_dict(params_t)
        loss = self.model.compute_loss_torch(input_ids_t, labels_t, params_dict)
        loss.backward()

        return params_t.grad.cpu().numpy().astype(np.float32)

    def bootstrap_adc(
        self,
        num_steps: int = 10,
        lr: Optional[float] = None,
    ) -> List[float]:
        if self._coordinator.codebook is None:
            return []

        bootstrap_lr = lr or self.config.protocol.learning_rate
        losses = []
        train_iter = iter(self.train_loader)

        for step in range(num_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            gradient = self._compute_true_gradient(batch)
            self._coordinator.codebook.update(gradient)

            params = self.model.get_flat_params()
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > self.config.training.max_grad_norm:
                gradient = gradient * (self.config.training.max_grad_norm / grad_norm)

            new_params = params - bootstrap_lr * gradient
            self.model.set_flat_params(new_params)

            loss = self.model.compute_loss(batch.input_ids, batch.labels)
            losses.append(loss)
            self.state.training_losses.append(loss)
            self.state.step += 1
            self.state.total_tokens += batch.input_ids.size

        return losses

    def train_step(self, batch: BatchData) -> StepMetrics:
        step_start = time.perf_counter()

        self._current_batch = batch
        params = self.model.get_flat_params()
        self._coordinator.set_parameters(params)
        self._coordinator.set_batch(batch.indices, batch.batch_seed)

        tasks = self._coordinator.publish_tasks(self.state.step)

        gradient_fn = self._create_gradient_fn(batch)
        self._worker_pool.set_gradient_fn(gradient_fn)

        collection_start = time.perf_counter()
        proofs = self._worker_pool.compute_proofs_parallel(
            tasks,
            first_k=self.config.protocol.K,
        )
        collection_time = time.perf_counter() - collection_start

        aggregation_start = time.perf_counter()
        for proof in proofs:
            self._coordinator.collect_proof(proof)
        gradient, agg_result = self._coordinator.aggregate()
        aggregation_time = time.perf_counter() - aggregation_start

        update_start = time.perf_counter()
        new_params = self._coordinator.update_parameters(gradient)
        self.model.set_flat_params(new_params)
        update_time = time.perf_counter() - update_start

        loss = self.model.compute_loss(batch.input_ids, batch.labels)
        self.state.training_losses.append(loss)

        captured_energy = 0.0
        if self.config.adc.enabled and self._coordinator.codebook is not None:
            captured_energy = self._coordinator.codebook.captured_energy_ratio(gradient)

        step_time = time.perf_counter() - step_start

        metrics = StepMetrics(
            step=self.state.step,
            loss=loss,
            bytes_received=len(proofs) * 40,
            proofs_received=len(proofs),
            proofs_used=agg_result.proofs_used,
            proofs_trimmed=agg_result.proofs_trimmed,
            captured_energy_ratio=captured_energy,
            step_time=step_time,
            collection_time=collection_time,
            aggregation_time=aggregation_time,
            update_time=update_time,
        )

        self.state.step += 1
        self.state.total_tokens += batch.input_ids.size

        return metrics

    def evaluate(self) -> float:
        if self.val_loader is None:
            return 0.0

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss = self.model.compute_loss(batch.input_ids, batch.labels)
            total_loss += loss
            num_batches += 1

            if num_batches >= 10:
                break

        return total_loss / max(num_batches, 1)

    def train(self, num_steps: Optional[int] = None) -> Dict[str, float]:
        max_steps = num_steps or self.config.training.max_steps

        if self.logger:
            self.logger.log_config(self.config)

        train_iter = iter(self.train_loader)

        while self.state.step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.state.epoch += 1
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch)

            if self.logger and self.state.step % self.config.logging.log_interval == 0:
                self.logger.log_step(
                    step=metrics.step,
                    metrics={
                        "train/loss": metrics.loss,
                        "train/proofs_used": metrics.proofs_used,
                        "train/proofs_trimmed": metrics.proofs_trimmed,
                        "train/step_time": metrics.step_time,
                        "adc/captured_energy": metrics.captured_energy_ratio,
                    },
                )

            if (
                self.val_loader is not None
                and self.state.step % self.config.training.eval_interval == 0
            ):
                val_loss = self.evaluate()

                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss

                if self.logger:
                    self.logger.log_step(
                        step=self.state.step,
                        metrics={"val/loss": val_loss},
                    )

        return {
            "final_train_loss": self.state.training_losses[-1]
            if self.state.training_losses
            else 0.0,
            "best_val_loss": self.state.best_val_loss,
            "total_steps": self.state.step,
            "total_tokens": self.state.total_tokens,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            params=self.model.get_flat_params(),
            step=self.state.step,
            epoch=self.state.epoch,
            best_val_loss=self.state.best_val_loss,
        )

    def load_checkpoint(self, path: Path) -> None:
        data = np.load(path)
        self.model.set_flat_params(data["params"])
        self.state.step = int(data["step"])
        self.state.epoch = int(data["epoch"])
        self.state.best_val_loss = float(data["best_val_loss"])
