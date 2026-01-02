#!/usr/bin/env python3
"""
Benchmark: Compare HoloGrad variants on GPT-2 training.

Variants:
1. Full SGD (baseline)
2. Random HoloGrad (K=64 random directions)
3. ADC HoloGrad (K=64 subspace directions)
4. Momentum-Centric HoloGrad (1 momentum direction)
"""

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using numpy-only mode")

from holograd.core.config import HoloGradConfig
from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data
from holograd.training.trainer import HoloGradTrainer


@dataclass
class BenchmarkResult:
    name: str
    losses: List[float]
    step_times: List[float]
    scalars_per_step: int
    final_loss: float
    total_time: float

    @property
    def loss_reduction(self) -> float:
        if len(self.losses) < 2:
            return 0.0
        return (self.losses[0] - self.losses[-1]) / self.losses[0] * 100

    @property
    def avg_step_time(self) -> float:
        return np.mean(self.step_times) if self.step_times else 0.0


def run_full_sgd(
    model: SimpleGPT2, train_loader, num_steps: int, lr: float = 3e-4
) -> BenchmarkResult:
    """Baseline: Full gradient SGD."""
    print("\n=== Running Full SGD (Baseline) ===")

    params = model.get_flat_params()
    losses = []
    step_times = []

    train_iter = iter(train_loader)
    start_time = time.time()

    for step in range(num_steps):
        step_start = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids, labels = batch.input_ids, batch.labels

        loss = model.compute_loss(input_ids, labels)
        losses.append(loss)

        if TORCH_AVAILABLE:
            params_t = torch.tensor(params, dtype=torch.float32, requires_grad=True)
            params_dict = model.flat_params_to_torch_dict(params_t)
            input_ids_t = torch.tensor(input_ids, dtype=torch.long)
            labels_t = torch.tensor(labels, dtype=torch.long)

            loss_t = model.compute_loss_torch(input_ids_t, labels_t, params_dict)
            loss_t.backward()
            gradient = params_t.grad.numpy()
        else:
            eps = 1e-4
            gradient = np.zeros_like(params)
            sample_indices = np.random.choice(len(params), min(500, len(params)), replace=False)
            for idx in sample_indices:
                params_plus = params.copy()
                params_plus[idx] += eps
                model.set_flat_params(params_plus)
                loss_plus = model.compute_loss(input_ids, labels)
                gradient[idx] = (loss_plus - loss) / eps
            model.set_flat_params(params)

        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1.0:
            gradient = gradient / grad_norm

        params = params - lr * gradient
        model.set_flat_params(params)

        step_times.append(time.time() - step_start)

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{num_steps}: loss={loss:.4f}")

    total_time = time.time() - start_time

    return BenchmarkResult(
        name="Full SGD",
        losses=losses,
        step_times=step_times,
        scalars_per_step=model.num_parameters,
        final_loss=losses[-1] if losses else 0.0,
        total_time=total_time,
    )


def run_holograd_variant(
    name: str,
    config: HoloGradConfig,
    model: SimpleGPT2,
    train_loader,
    val_loader,
    num_steps: int,
) -> BenchmarkResult:
    """Run a HoloGrad variant."""
    print(f"\n=== Running {name} ===")

    model_copy = SimpleGPT2(size="tiny")
    model_copy.set_flat_params(model.get_flat_params().copy())

    trainer = HoloGradTrainer(
        config=config,
        model=model_copy,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    losses = []
    step_times = []
    train_iter = iter(train_loader)
    start_time = time.time()

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step_start = time.time()
        metrics = trainer.train_step(batch)
        step_times.append(time.time() - step_start)
        losses.append(metrics.loss)

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{num_steps}: loss={metrics.loss:.4f}")

    total_time = time.time() - start_time

    if config.protocol.direction_mode == "momentum":
        scalars_per_step = config.distributed.num_workers
    else:
        scalars_per_step = config.protocol.K

    return BenchmarkResult(
        name=name,
        losses=losses,
        step_times=step_times,
        scalars_per_step=scalars_per_step,
        final_loss=losses[-1] if losses else 0.0,
        total_time=total_time,
    )


def print_results(results: List[BenchmarkResult], baseline: BenchmarkResult):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print(
        f"\n{'Method':<25} {'Final Loss':>12} {'Loss Δ%':>10} {'Scalars/Step':>14} {'Efficiency':>12}"
    )
    print("-" * 80)

    baseline_reduction = baseline.loss_reduction

    for r in [baseline] + results:
        if baseline_reduction > 0:
            efficiency = r.loss_reduction / baseline_reduction * 100
        else:
            efficiency = 0.0

        print(
            f"{r.name:<25} {r.final_loss:>12.4f} {r.loss_reduction:>9.2f}% {r.scalars_per_step:>14} {efficiency:>11.1f}%"
        )

    print("\n" + "=" * 80)
    print("COMMUNICATION EFFICIENCY (vs Full SGD)")
    print("=" * 80)

    for r in results:
        comm_reduction = (1 - r.scalars_per_step / baseline.scalars_per_step) * 100
        print(f"{r.name:<25}: {comm_reduction:.2f}% communication reduction")


def main():
    print("=" * 80)
    print("HoloGrad Variant Comparison Benchmark")
    print("=" * 80)

    NUM_STEPS = 200
    K = 64
    NUM_WORKERS = 8

    print(f"\nSettings:")
    print(f"  - Model: GPT-2 tiny")
    print(f"  - Steps: {NUM_STEPS}")
    print(f"  - K (directions): {K}")
    print(f"  - Workers: {NUM_WORKERS}")

    print("\nCreating model and data...")
    model = SimpleGPT2(size="tiny")
    print(f"  - Parameters: {model.num_parameters:,}")

    train_loader, val_loader = create_synthetic_data(
        vocab_size=model.vocab_size,
        seq_length=128,
        num_train_samples=500,
        batch_size=4,
    )

    initial_params = model.get_flat_params().copy()

    baseline = run_full_sgd(model, train_loader, NUM_STEPS)
    model.set_flat_params(initial_params.copy())

    results = []

    config_random = HoloGradConfig()
    config_random.protocol.K = K
    config_random.protocol.direction_mode = "random"
    config_random.adc.enabled = False
    config_random.distributed.num_workers = NUM_WORKERS

    result_random = run_holograd_variant(
        "Random HoloGrad (K=64)",
        config_random,
        model,
        train_loader,
        val_loader,
        NUM_STEPS,
    )
    results.append(result_random)
    model.set_flat_params(initial_params.copy())

    config_adc = HoloGradConfig()
    config_adc.protocol.K = K
    config_adc.protocol.direction_mode = "adc"
    config_adc.adc.enabled = True
    config_adc.adc.rank = 32
    config_adc.distributed.num_workers = NUM_WORKERS

    result_adc = run_holograd_variant(
        "ADC HoloGrad (r=32)",
        config_adc,
        model,
        train_loader,
        val_loader,
        NUM_STEPS,
    )
    results.append(result_adc)
    model.set_flat_params(initial_params.copy())

    config_momentum = HoloGradConfig()
    config_momentum.protocol.K = K
    config_momentum.protocol.direction_mode = "momentum"
    config_momentum.momentum_centric.warmup_steps = 5
    config_momentum.momentum_centric.beta = 0.9
    config_momentum.distributed.num_workers = NUM_WORKERS

    result_momentum = run_holograd_variant(
        "Momentum-Centric (N=8)",
        config_momentum,
        model,
        train_loader,
        val_loader,
        NUM_STEPS,
    )
    results.append(result_momentum)

    print_results(results, baseline)

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    momentum_efficiency = (
        result_momentum.loss_reduction / baseline.loss_reduction * 100
        if baseline.loss_reduction > 0
        else 0
    )
    random_efficiency = (
        result_random.loss_reduction / baseline.loss_reduction * 100
        if baseline.loss_reduction > 0
        else 0
    )

    print(f"""
1. Momentum-Centric uses {result_momentum.scalars_per_step} scalars/step vs {result_random.scalars_per_step} for Random
   -> {result_random.scalars_per_step / result_momentum.scalars_per_step:.0f}x less communication

2. Momentum-Centric achieves {momentum_efficiency:.1f}% of Full SGD efficiency
   Random HoloGrad achieves {random_efficiency:.1f}% of Full SGD efficiency

3. Communication reduction vs Full SGD:
   - Random HoloGrad: {(1 - result_random.scalars_per_step / baseline.scalars_per_step) * 100:.4f}%
   - Momentum-Centric: {(1 - result_momentum.scalars_per_step / baseline.scalars_per_step) * 100:.4f}%
""")


if __name__ == "__main__":
    main()
