#!/usr/bin/env python3

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.protocol.aggregation import RobustAggregator


def debug_linear_regression():
    print("=" * 60)
    print("DEBUG: LINEAR REGRESSION STEP BY STEP")
    print("=" * 60)

    np.random.seed(42)

    D = 100
    N = 50
    K = 64

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
    scale_factor = direction_gen.scale_factor
    aggregator = RobustAggregator(tau=0.1)

    print(f"\nD={D}, K={K}, scale_factor={scale_factor}")
    print(f"sigma_squared = {direction_gen.sigma_squared}")

    w = np.zeros(D, dtype=np.float32)
    print(f"\nInitial loss: {loss_fn(w):.6f}")

    grad_true = true_gradient(w)
    print(f"True gradient norm: {np.linalg.norm(grad_true):.4f}")

    scalars = []
    directions = []

    for k in range(K):
        seed = f"step0_k{k}".encode()
        result = direction_gen.generate(seed)
        d = result.direction

        scalar = float(np.dot(grad_true, d))
        scalars.append(scalar)
        directions.append(d)

        if k < 5:
            print(f"  k={k}: scalar={scalar:.6f}, ||d||={np.linalg.norm(d):.4f}")

    print(f"\nScalar statistics:")
    print(f"  Mean: {np.mean(scalars):.6f}")
    print(f"  Std:  {np.std(scalars):.6f}")
    print(f"  Expected std (||g||/sqrt(D)): {np.linalg.norm(grad_true) / np.sqrt(D):.6f}")

    agg_result = aggregator.aggregate(scalars, directions, scale_factor)
    grad_hat = agg_result.gradient

    print(f"\nReconstructed gradient:")
    print(f"  ||grad_hat|| = {np.linalg.norm(grad_hat):.4f}")
    print(f"  ||grad_true|| = {np.linalg.norm(grad_true):.4f}")
    print(f"  Ratio: {np.linalg.norm(grad_hat) / np.linalg.norm(grad_true):.4f}")

    cosine = np.dot(grad_hat, grad_true) / (np.linalg.norm(grad_hat) * np.linalg.norm(grad_true))
    print(f"  Cosine similarity: {cosine:.4f}")

    print(f"\n--- Testing different learning rates ---")

    for lr in [1.0, 0.1, 0.01, 0.001]:
        w_test = np.zeros(D, dtype=np.float32)
        losses = [loss_fn(w_test)]

        for step in range(20):
            grad_true = true_gradient(w_test)

            scalars = []
            directions = []
            for k in range(K):
                seed = f"step{step}_k{k}".encode()
                result = direction_gen.generate(seed)
                d = result.direction
                scalar = float(np.dot(grad_true, d))
                scalars.append(scalar)
                directions.append(d)

            agg_result = aggregator.aggregate(scalars, directions, scale_factor)
            grad_hat = agg_result.gradient

            grad_norm = np.linalg.norm(grad_hat)
            if grad_norm > 1.0:
                grad_hat = grad_hat * (1.0 / grad_norm)

            w_test = w_test - lr * grad_hat
            losses.append(loss_fn(w_test))

        print(
            f"lr={lr}: {losses[0]:.2f} -> {losses[-1]:.2f} ({(1 - losses[-1] / losses[0]) * 100:.1f}%)"
        )


def debug_adc():
    print("\n" + "=" * 60)
    print("DEBUG: ADC DIRECTION PROPERTIES")
    print("=" * 60)

    np.random.seed(42)
    D = 100
    rank = 32

    adc = ADCCodebook(D, rank=rank)

    print(f"\nD={D}, rank={rank}")
    print(f"Codebook shape: {adc.codebook.shape}")
    print(f"Scale factor: {adc.get_scale_factor()}")

    norms = []
    for k in range(100):
        seed = f"test_k{k}".encode()
        result = adc.generate_direction(seed)
        d = result.direction
        norms.append(np.linalg.norm(d))

    print(f"\nDirection norms (should be ~sqrt(rank)={np.sqrt(rank):.2f}):")
    print(f"  Mean: {np.mean(norms):.4f}")
    print(f"  Std:  {np.std(norms):.4f}")
    print(f"  Range: [{np.min(norms):.4f}, {np.max(norms):.4f}]")

    print("\n[BUG FOUND] ADC directions are NOT unit-norm!")
    print("DirectionGenerator produces unit-norm, but ADCCodebook does not normalize.")
    print("This breaks the scale_factor=1.0 assumption.")


def debug_gradient_reconstruction_math():
    print("\n" + "=" * 60)
    print("DEBUG: GRADIENT RECONSTRUCTION MATH")
    print("=" * 60)

    np.random.seed(42)
    D = 100
    K = 1000

    g = np.random.randn(D).astype(np.float32)
    g_norm = np.linalg.norm(g)

    print(f"\nD={D}, K={K}")
    print(f"True gradient norm: {g_norm:.4f}")

    direction_gen = DirectionGenerator(D)

    scalars = []
    directions = []

    for k in range(K):
        seed = f"k{k}".encode()
        result = direction_gen.generate(seed)
        v = result.direction
        scalar = np.dot(g, v)
        scalars.append(scalar)
        directions.append(v)

    g_hat = np.zeros(D, dtype=np.float32)
    for scalar, v in zip(scalars, directions):
        g_hat += scalar * v
    g_hat = (direction_gen.scale_factor / K) * g_hat

    print(f"\nWithout trimming (K={K}):")
    print(f"  ||g_hat|| = {np.linalg.norm(g_hat):.4f}")
    print(f"  Cosine similarity = {np.dot(g_hat, g) / (np.linalg.norm(g_hat) * g_norm):.4f}")
    print(f"  Relative error = {np.linalg.norm(g_hat - g) / g_norm:.4f}")

    K_small = 64
    g_hat_small = np.zeros(D, dtype=np.float32)
    for scalar, v in zip(scalars[:K_small], directions[:K_small]):
        g_hat_small += scalar * v
    g_hat_small = (direction_gen.scale_factor / K_small) * g_hat_small

    print(f"\nWith K={K_small}:")
    print(f"  ||g_hat|| = {np.linalg.norm(g_hat_small):.4f}")
    print(
        f"  Cosine similarity = {np.dot(g_hat_small, g) / (np.linalg.norm(g_hat_small) * g_norm):.4f}"
    )
    print(f"  Relative error = {np.linalg.norm(g_hat_small - g) / g_norm:.4f}")

    print("\n[ANALYSIS]")
    print(f"Expected relative error ~ sqrt(D/K) = {np.sqrt(D / K_small):.2f}")
    print("The gradient estimate has high variance with small K!")


def fix_and_test():
    print("\n" + "=" * 60)
    print("FIX: PROPER LEARNING RATE SCALING")
    print("=" * 60)

    np.random.seed(42)

    D = 100
    N = 50
    K = 64

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

    lr_base = 0.1
    max_grad_norm = 1.0

    print(f"D={D}, K={K}, base_lr={lr_base}")
    print(f"Initial loss: {loss_fn(w):.4f}\n")

    losses = []
    for step in range(100):
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

        w = w - lr_base * g_hat

        current_loss = loss_fn(w)
        losses.append(current_loss)

        if step % 20 == 0:
            cosine = np.dot(g_hat, grad_true) / (
                np.linalg.norm(g_hat) * np.linalg.norm(grad_true) + 1e-10
            )
            print(
                f"Step {step:3d}: loss={current_loss:.4f}, ||g_hat||={np.linalg.norm(g_hat):.4f}, cos={cosine:.3f}"
            )

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Reduction: {(1 - losses[-1] / losses[0]) * 100:.1f}%")

    if losses[-1] < 0.01:
        print("[SUCCESS] Converged to near-zero loss!")
    elif losses[-1] < losses[0] * 0.1:
        print("[SUCCESS] Good convergence")
    else:
        print("[PARTIAL] Some convergence but not complete")


def main():
    debug_linear_regression()
    debug_adc()
    debug_gradient_reconstruction_math()
    fix_and_test()


if __name__ == "__main__":
    main()
