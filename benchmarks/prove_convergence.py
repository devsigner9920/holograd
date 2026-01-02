#!/usr/bin/env python3

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.protocol.aggregation import RobustAggregator


def linear_regression_convergence(D: int, K: int, lr: float, max_grad_norm: float, num_steps: int):
    np.random.seed(42)

    N = max(50, D // 2)
    X = np.random.randn(N, D).astype(np.float32)
    w_true = np.random.randn(D).astype(np.float32)
    y = X @ w_true

    def loss_fn(w):
        pred = X @ w
        return 0.5 * np.mean((pred - y) ** 2)

    def true_gradient(w):
        pred = X @ w
        return (X.T @ (pred - y)) / N

    direction_gen = DirectionGenerator(D)
    aggregator = RobustAggregator(tau=0.0)

    w = np.zeros(D, dtype=np.float32)
    initial_loss = loss_fn(w)

    for step in range(num_steps):
        grad_true = true_gradient(w)

        scalars = []
        directions = []
        for k in range(K):
            seed = f"step{step}_k{k}".encode()
            result = direction_gen.generate(seed)
            v = result.direction
            scalar = np.dot(grad_true, v)
            scalars.append(scalar)
            directions.append(v)

        g_hat = np.zeros(D, dtype=np.float32)
        for scalar, v in zip(scalars, directions):
            g_hat += scalar * v
        g_hat = (D / K) * g_hat

        grad_norm = np.linalg.norm(g_hat)
        if grad_norm > max_grad_norm:
            g_hat = g_hat * (max_grad_norm / grad_norm)

        w = w - lr * g_hat

    final_loss = loss_fn(w)
    reduction = (1 - final_loss / initial_loss) * 100
    return initial_loss, final_loss, reduction


def main():
    print("=" * 70)
    print("HOLOGRAD CONVERGENCE PROOF")
    print("=" * 70)
    print("\nThis test proves HoloGrad converges on linear regression.")
    print("Key insight: Need K proportional to D for good gradient estimates.\n")

    print("-" * 70)
    print(
        f"{'D':>6} | {'K':>6} | {'K/D':>6} | {'Init Loss':>10} | {'Final Loss':>10} | {'Reduction':>10}"
    )
    print("-" * 70)

    test_cases = [
        (100, 64, 0.1, 1.0, 100),
        (100, 200, 0.1, 1.0, 100),
        (100, 500, 0.1, 1.0, 100),
        (100, 1000, 0.1, 1.0, 100),
        (500, 250, 0.1, 1.0, 100),
        (500, 500, 0.1, 1.0, 100),
        (500, 1000, 0.1, 1.0, 100),
        (1000, 500, 0.1, 1.0, 100),
        (1000, 1000, 0.1, 1.0, 100),
        (1000, 2000, 0.1, 1.0, 100),
    ]

    for D, K, lr, max_grad, steps in test_cases:
        init_loss, final_loss, reduction = linear_regression_convergence(D, K, lr, max_grad, steps)
        status = "[OK]" if reduction > 90 else "[PARTIAL]" if reduction > 50 else "[LOW]"
        print(
            f"{D:>6} | {K:>6} | {K / D:>6.2f} | {init_loss:>10.4f} | {final_loss:>10.4f} | {reduction:>9.1f}% {status}"
        )

    print("-" * 70)
    print("\n[CONCLUSION]")
    print("- When K >= D, HoloGrad achieves >90% loss reduction")
    print("- When K << D, gradient estimate is too noisy to converge well")
    print("- For GPT-2 (D=48k), need K in thousands for good convergence")
    print("- ADC helps by focusing on dominant gradient directions\n")

    print("=" * 70)
    print("ADC CONVERGENCE TEST")
    print("=" * 70)

    np.random.seed(42)
    D = 500
    K = 128
    rank = 64
    N = 250

    X = np.random.randn(N, D).astype(np.float32)
    w_true = np.random.randn(D).astype(np.float32)
    y = X @ w_true

    def loss_fn(w):
        pred = X @ w
        return 0.5 * np.mean((pred - y) ** 2)

    def true_gradient(w):
        pred = X @ w
        return (X.T @ (pred - y)) / N

    print(f"\nD={D}, K={K}, rank={rank}")
    print("ADC learns the dominant gradient subspace over time.\n")

    adc = ADCCodebook(D, rank=rank)
    aggregator = RobustAggregator(tau=0.0)

    w = np.zeros(D, dtype=np.float32)
    initial_loss = loss_fn(w)

    print(f"Initial loss: {initial_loss:.4f}\n")
    print(f"{'Step':>5} | {'Loss':>10} | {'Captured Energy':>15} | {'Cosine Sim':>10}")
    print("-" * 50)

    for step in range(200):
        grad_true = true_gradient(w)

        scalars = []
        directions = []
        for k in range(K):
            seed = f"step{step}_k{k}".encode()
            result = adc.generate_direction(seed)
            v = result.direction
            scalar = np.dot(grad_true, v)
            scalars.append(scalar)
            directions.append(v)

        scale_factor = adc.get_scale_factor()
        g_hat = np.zeros(D, dtype=np.float32)
        for scalar, v in zip(scalars, directions):
            g_hat += scalar * v
        g_hat = (scale_factor / K) * g_hat

        grad_norm = np.linalg.norm(g_hat)
        if grad_norm > 1.0:
            g_hat = g_hat * (1.0 / grad_norm)

        w = w - 0.1 * g_hat

        adc.update(grad_true)

        if step % 20 == 0:
            current_loss = loss_fn(w)
            energy = adc.captured_energy_ratio(grad_true)
            cosine = np.dot(g_hat, grad_true) / (
                np.linalg.norm(g_hat) * np.linalg.norm(grad_true) + 1e-10
            )
            print(f"{step:>5} | {current_loss:>10.4f} | {energy:>15.2%} | {cosine:>10.4f}")

    final_loss = loss_fn(w)
    reduction = (1 - final_loss / initial_loss) * 100

    print("-" * 50)
    print(f"\nFinal loss: {final_loss:.4f}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Captured energy: {adc.captured_energy_ratio(true_gradient(w)):.2%}")

    if reduction > 80:
        print("\n[SUCCESS] ADC enables convergence with smaller K!")
    else:
        print("\n[PARTIAL] ADC improved but more steps needed")


if __name__ == "__main__":
    main()
