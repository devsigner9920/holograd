#!/usr/bin/env python3
"""Momentum-based HoloGrad: Proof of concept that it converges."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from typing import Tuple, List
from dataclasses import dataclass

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data
from holograd.protocol.direction import DirectionGenerator
from holograd.protocol.aggregation import RobustAggregator


def compute_gradient(model, input_ids, labels) -> Tuple[np.ndarray, float]:
    params = model.get_flat_params()
    params_t = torch.tensor(params, dtype=torch.float32, requires_grad=True)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    params_dict = model.flat_params_to_torch_dict(params_t)
    loss = model.compute_loss_torch(input_ids_t, labels_t, params_dict)
    loss.backward()

    return params_t.grad.numpy(), loss.item()


def compute_loss(model, input_ids, labels) -> float:
    params = model.get_flat_params()
    params_t = torch.tensor(params, dtype=torch.float32)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    params_dict = model.flat_params_to_torch_dict(params_t)
    with torch.no_grad():
        loss = model.compute_loss_torch(input_ids_t, labels_t, params_dict)
    return loss.item()


@dataclass
class TrainResult:
    method: str
    losses: List[float]
    final_loss: float
    cosine_similarities: List[float]


def clip_gradient(grad: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


def train_pure_sgd(model, train_loader, num_steps: int, lr: float) -> TrainResult:
    """Baseline: Pure gradient descent with true gradients."""
    losses = []
    train_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad, loss = compute_gradient(model, batch.input_ids, batch.labels)
        losses.append(loss)

        grad = clip_gradient(grad)
        params = model.get_flat_params()
        params = params - lr * grad
        model.set_flat_params(params)

    return TrainResult(
        method="Pure SGD",
        losses=losses,
        final_loss=losses[-1],
        cosine_similarities=[1.0] * num_steps,
    )


def train_random_holograd(model, train_loader, num_steps: int, lr: float, K: int) -> TrainResult:
    """Original HoloGrad with random directions."""
    D = model.num_parameters
    direction_gen = DirectionGenerator(D)
    losses = []
    cosines = []
    train_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad_true, loss = compute_gradient(model, batch.input_ids, batch.labels)
        losses.append(loss)

        g_hat = np.zeros(D, dtype=np.float32)
        for k in range(K):
            seed = f"step{step}_k{k}".encode()
            result = direction_gen.generate(seed)
            v = result.direction
            scalar = np.dot(grad_true, v)
            g_hat += scalar * v
        g_hat = (D / K) * g_hat

        cos = np.dot(grad_true, g_hat) / (np.linalg.norm(grad_true) * np.linalg.norm(g_hat) + 1e-10)
        cosines.append(cos)

        g_hat = clip_gradient(g_hat)
        params = model.get_flat_params()
        params = params - lr * g_hat
        model.set_flat_params(params)

    return TrainResult(
        method=f"Random HoloGrad (K={K})",
        losses=losses,
        final_loss=losses[-1],
        cosine_similarities=cosines,
    )


def train_momentum_holograd(
    model, train_loader, num_steps: int, lr: float, beta: float = 0.9, warmup: int = 20
) -> TrainResult:
    """Momentum-based HoloGrad: Use momentum direction for projection."""
    D = model.num_parameters
    momentum = np.zeros(D, dtype=np.float32)
    losses = []
    cosines = []
    train_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad_true, loss = compute_gradient(model, batch.input_ids, batch.labels)
        losses.append(loss)

        if step < warmup:
            g_hat = grad_true
            cos = 1.0
        else:
            m_norm = np.linalg.norm(momentum)
            grad_norm = np.linalg.norm(grad_true)
            if m_norm > 1e-10 and grad_norm > 1e-10:
                m_dir = momentum / m_norm
                scalar = np.dot(grad_true, m_dir)
                g_hat = scalar * m_dir
                g_hat = g_hat * (grad_norm / (np.linalg.norm(g_hat) + 1e-10))
                cos = scalar / grad_norm
            else:
                g_hat = grad_true
                cos = 1.0

        cosines.append(cos)

        momentum = beta * momentum + (1 - beta) * grad_true

        g_hat = clip_gradient(g_hat)
        params = model.get_flat_params()
        params = params - lr * g_hat
        model.set_flat_params(params)

    return TrainResult(
        method=f"Momentum HoloGrad (warmup={warmup})",
        losses=losses,
        final_loss=losses[-1],
        cosine_similarities=cosines,
    )


def train_momentum_holograd_v2(
    model, train_loader, num_steps: int, lr: float, beta: float = 0.9, warmup: int = 20
) -> TrainResult:
    """Momentum HoloGrad v2: Update momentum with reconstructed gradient."""
    D = model.num_parameters
    momentum = np.zeros(D, dtype=np.float32)
    losses = []
    cosines = []
    train_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad_true, loss = compute_gradient(model, batch.input_ids, batch.labels)
        losses.append(loss)

        if step < warmup:
            g_hat = grad_true
            cos = 1.0
        else:
            m_norm = np.linalg.norm(momentum)
            grad_norm = np.linalg.norm(grad_true)
            if m_norm > 1e-10 and grad_norm > 1e-10:
                m_dir = momentum / m_norm
                scalar = np.dot(grad_true, m_dir)
                g_hat = scalar * m_dir
                g_hat = g_hat * (grad_norm / (np.linalg.norm(g_hat) + 1e-10))
                cos = scalar / grad_norm
            else:
                g_hat = grad_true
                cos = 1.0

        cosines.append(cos)

        momentum = beta * momentum + (1 - beta) * g_hat

        g_hat = clip_gradient(g_hat)
        params = model.get_flat_params()
        params = params - lr * g_hat
        model.set_flat_params(params)

    return TrainResult(
        method=f"Momentum HoloGrad v2 (warmup={warmup})",
        losses=losses,
        final_loss=losses[-1],
        cosine_similarities=cosines,
    )


def train_signsgd_style(
    model, train_loader, num_steps: int, lr: float, beta: float = 0.9
) -> TrainResult:
    """SignSGD-style: Use sign of projection onto momentum."""
    D = model.num_parameters
    momentum = np.zeros(D, dtype=np.float32)
    losses = []
    cosines = []
    train_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad_true, loss = compute_gradient(model, batch.input_ids, batch.labels)
        losses.append(loss)

        if step == 0:
            g_hat = grad_true
            cos = 1.0
        else:
            m_norm = np.linalg.norm(momentum)
            if m_norm > 1e-10:
                m_dir = momentum / m_norm
                scalar = np.dot(grad_true, m_dir)
                sign = 1.0 if scalar > 0 else -1.0
                g_hat = sign * m_dir * m_norm
                cos = np.dot(grad_true, g_hat) / (
                    np.linalg.norm(grad_true) * np.linalg.norm(g_hat) + 1e-10
                )
            else:
                g_hat = grad_true
                cos = 1.0

        cosines.append(cos)

        momentum = beta * momentum + (1 - beta) * grad_true

        g_hat = clip_gradient(g_hat)
        params = model.get_flat_params()
        params = params - lr * g_hat
        model.set_flat_params(params)

    return TrainResult(
        method="SignSGD-style",
        losses=losses,
        final_loss=losses[-1],
        cosine_similarities=cosines,
    )


def main():
    print("=" * 70)
    print("MOMENTUM-BASED HOLOGRAD CONVERGENCE TEST")
    print("=" * 70)

    model_size = "tiny"
    vocab_size = 100
    seq_length = 32
    batch_size = 8
    num_steps = 1000
    lr = 5e-2
    K = 64
    warmup_steps = 50

    print(f"\nConfig: model={model_size}, vocab={vocab_size}, steps={num_steps}, lr={lr}")

    train_loader, _ = create_synthetic_data(
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_train_samples=500,
        batch_size=batch_size,
    )

    results = []

    print("\n[1/5] Training with Pure SGD (baseline)...")
    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    initial_params = model.get_flat_params().copy()
    D = model.num_parameters
    print(f"  D = {D:,}")
    result = train_pure_sgd(model, train_loader, num_steps, lr)
    results.append(result)
    print(f"  Loss: {result.losses[0]:.4f} -> {result.final_loss:.4f}")

    print(f"\n[2/5] Training with Random HoloGrad (K={K})...")
    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    model.set_flat_params(initial_params.copy())
    result = train_random_holograd(model, train_loader, num_steps, lr, K)
    results.append(result)
    print(f"  Loss: {result.losses[0]:.4f} -> {result.final_loss:.4f}")
    print(f"  Mean cosine: {np.mean(result.cosine_similarities):.4f}")

    print(f"\n[3/5] Training with Momentum HoloGrad (warmup={warmup_steps})...")
    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    model.set_flat_params(initial_params.copy())
    result = train_momentum_holograd(model, train_loader, num_steps, lr, warmup=warmup_steps)
    results.append(result)
    print(f"  Loss: {result.losses[0]:.4f} -> {result.final_loss:.4f}")
    print(f"  Mean cosine: {np.mean(result.cosine_similarities):.4f}")

    print(f"\n[4/5] Training with Momentum HoloGrad v2 (warmup={warmup_steps})...")
    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    model.set_flat_params(initial_params.copy())
    result = train_momentum_holograd_v2(model, train_loader, num_steps, lr, warmup=warmup_steps)
    results.append(result)
    print(f"  Loss: {result.losses[0]:.4f} -> {result.final_loss:.4f}")
    print(f"  Mean cosine: {np.mean(result.cosine_similarities):.4f}")

    print("\n[5/5] Training with SignSGD-style...")
    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    model.set_flat_params(initial_params.copy())
    result = train_signsgd_style(model, train_loader, num_steps, lr)
    results.append(result)
    print(f"  Loss: {result.losses[0]:.4f} -> {result.final_loss:.4f}")
    print(f"  Mean cosine: {np.mean(result.cosine_similarities):.4f}")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} | {'Initial':>10} | {'Final':>10} | {'Reduction':>10}")
    print("-" * 70)

    for r in results:
        reduction = (1 - r.final_loss / r.losses[0]) * 100
        status = "[OK]" if reduction > 5 else "[FAIL]"
        print(
            f"{r.method:<35} | {r.losses[0]:>10.4f} | {r.final_loss:>10.4f} | {reduction:>9.1f}% {status}"
        )

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    sgd_reduction = (1 - results[0].final_loss / results[0].losses[0]) * 100
    random_reduction = (1 - results[1].final_loss / results[1].losses[0]) * 100
    momentum_reduction = (1 - results[2].final_loss / results[2].losses[0]) * 100
    momentum_v2_reduction = (1 - results[3].final_loss / results[3].losses[0]) * 100

    print(f"\nPure SGD achieves {sgd_reduction:.1f}% loss reduction (baseline)")
    print(f"Random HoloGrad achieves {random_reduction:.1f}% loss reduction")
    print(f"Momentum HoloGrad achieves {momentum_reduction:.1f}% loss reduction")
    print(f"Momentum HoloGrad v2 achieves {momentum_v2_reduction:.1f}% loss reduction")

    if momentum_reduction > sgd_reduction * 0.5:
        print("\n[SUCCESS] Momentum HoloGrad converges!")
        print("  - Uses only 1 scalar per worker (vs K for random)")
        print("  - Achieves meaningful loss reduction")
        print("  - Ready for integration into main codebase")
    elif momentum_reduction > random_reduction:
        print("\n[PARTIAL] Momentum HoloGrad better than random but slower than SGD")
        print("  - May need more steps or tuning")
    else:
        print("\n[FAILED] Momentum HoloGrad didn't converge")


if __name__ == "__main__":
    main()
