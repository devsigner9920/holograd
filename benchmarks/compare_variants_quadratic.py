#!/usr/bin/env python3
"""
Benchmark: Compare HoloGrad variants on quadratic loss minimization.
This provides a clear signal for comparing convergence.
"""

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
from holograd.distributed.worker import Worker, WorkerConfig


@dataclass
class BenchmarkResult:
    name: str
    losses: List[float]
    scalars_per_step: int

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("inf")

    @property
    def loss_reduction_pct(self) -> float:
        if len(self.losses) < 2 or self.losses[0] == 0:
            return 0.0
        return (self.losses[0] - self.losses[-1]) / self.losses[0] * 100


def run_full_sgd(dim: int, num_steps: int, lr: float) -> BenchmarkResult:
    print("\n=== Full SGD (Baseline) ===")
    params = np.random.randn(dim).astype(np.float32)
    losses = []

    for step in range(num_steps):
        loss = float(np.sum(params**2))
        losses.append(loss)

        gradient = 2 * params
        params = params - lr * gradient

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: loss={loss:.4f}")

    return BenchmarkResult(name="Full SGD", losses=losses, scalars_per_step=dim)


def run_holograd_variant(
    name: str,
    dim: int,
    num_steps: int,
    num_workers: int,
    lr: float,
    use_momentum_centric: bool = False,
    use_adc: bool = False,
    K: int = 64,
) -> BenchmarkResult:
    print(f"\n=== {name} ===")

    config = CoordinatorConfig(
        dimension=dim,
        num_workers=K,
        proofs_per_step=K,
        learning_rate=lr,
        use_momentum_centric=use_momentum_centric,
        momentum_warmup_steps=5 if use_momentum_centric else 0,
        use_adc=use_adc,
        adc_rank=32 if use_adc else 0,
        momentum=0.0,
        max_grad_norm=float("inf"),
    )

    coord = Coordinator(config)
    workers = [Worker(WorkerConfig(worker_id=i, dimension=dim)) for i in range(K)]

    if use_adc and coord.codebook is not None:
        for w in workers:
            w.set_codebook(coord.codebook)

    params = np.random.randn(dim).astype(np.float32)
    coord.set_parameters(params)

    losses = []

    for step in range(num_steps):
        loss = float(np.sum(coord._current_params**2))
        losses.append(loss)

        coord.set_batch(np.array([step]), seed=step)
        tasks = coord.publish_tasks(step=step)

        gradient = 2 * coord._current_params

        if use_adc and coord.codebook is not None:
            coord.codebook.update(gradient)

        proofs = [w.compute_proof(tasks[i], gradient=gradient) for i, w in enumerate(workers)]

        for proof in proofs:
            coord.collect_proof(proof)

        reconstructed, _ = coord.aggregate()
        coord.update_parameters(reconstructed)

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: loss={loss:.4f}")

    if use_momentum_centric:
        scalars = num_workers
    else:
        scalars = K

    return BenchmarkResult(name=name, losses=losses, scalars_per_step=scalars)


def main():
    print("=" * 70)
    print("HoloGrad Variant Comparison - Quadratic Loss")
    print("=" * 70)

    DIM = 10000
    NUM_STEPS = 200
    NUM_WORKERS = 8
    K = 64
    LR = 0.01

    print(f"\nSettings:")
    print(f"  Dimension: {DIM:,}")
    print(f"  Steps: {NUM_STEPS}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  K (directions): {K}")
    print(f"  Learning rate: {LR}")

    np.random.seed(42)

    baseline = run_full_sgd(DIM, NUM_STEPS, LR)

    np.random.seed(42)
    random_result = run_holograd_variant(
        "Random HoloGrad (K=64)",
        DIM,
        NUM_STEPS,
        NUM_WORKERS,
        LR,
        use_momentum_centric=False,
        use_adc=False,
        K=K,
    )

    np.random.seed(42)
    adc_result = run_holograd_variant(
        "ADC HoloGrad (r=32)",
        DIM,
        NUM_STEPS,
        NUM_WORKERS,
        LR,
        use_momentum_centric=False,
        use_adc=True,
        K=K,
    )

    np.random.seed(42)
    momentum_result = run_holograd_variant(
        "Momentum-Centric (N=8)",
        DIM,
        NUM_STEPS,
        NUM_WORKERS,
        LR,
        use_momentum_centric=True,
        use_adc=False,
        K=K,
    )

    results = [baseline, random_result, adc_result, momentum_result]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\n{'Method':<25} {'Initial':>10} {'Final':>10} {'Reduction':>12} {'Scalars':>10} {'Efficiency':>12}"
    )
    print("-" * 85)

    baseline_reduction = baseline.loss_reduction_pct

    for r in results:
        efficiency = (
            r.loss_reduction_pct / baseline_reduction * 100 if baseline_reduction > 0 else 0
        )
        print(
            f"{r.name:<25} {r.losses[0]:>10.2f} {r.final_loss:>10.4f} {r.loss_reduction_pct:>11.2f}% {r.scalars_per_step:>10} {efficiency:>11.1f}%"
        )

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    comm_random = K
    comm_momentum = NUM_WORKERS

    eff_random = (
        random_result.loss_reduction_pct / baseline_reduction * 100 if baseline_reduction > 0 else 0
    )
    eff_momentum = (
        momentum_result.loss_reduction_pct / baseline_reduction * 100
        if baseline_reduction > 0
        else 0
    )

    print(f"""
Communication:
  - Random HoloGrad: {comm_random} scalars/step
  - Momentum-Centric: {comm_momentum} scalars/step
  - Momentum uses {comm_random / comm_momentum:.0f}x fewer scalars

Efficiency (% of Full SGD convergence):
  - Random HoloGrad: {eff_random:.1f}%
  - ADC HoloGrad: {adc_result.loss_reduction_pct / baseline_reduction * 100 if baseline_reduction > 0 else 0:.1f}%
  - Momentum-Centric: {eff_momentum:.1f}%

Communication Efficiency Ratio:
  - Random: {eff_random / comm_random:.2f}% per scalar
  - Momentum: {eff_momentum / comm_momentum:.2f}% per scalar
  -> Momentum is {(eff_momentum / comm_momentum) / (eff_random / comm_random) if eff_random > 0 else 0:.1f}x more efficient per scalar
""")


if __name__ == "__main__":
    main()
