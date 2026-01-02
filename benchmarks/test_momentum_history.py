#!/usr/bin/env python3
"""Test using history of momentum directions for HoloGrad."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from typing import Tuple, List

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data


def compute_gradient(model, input_ids, labels) -> Tuple[np.ndarray, float]:
    params = model.get_flat_params()
    params_t = torch.tensor(params, dtype=torch.float32, requires_grad=True)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    params_dict = model.flat_params_to_torch_dict(params_t)
    loss = model.compute_loss_torch(input_ids_t, labels_t, params_dict)
    loss.backward()

    return params_t.grad.numpy(), loss.item()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def orthonormalize(vectors: List[np.ndarray]) -> np.ndarray:
    """Gram-Schmidt orthonormalization."""
    D = len(vectors[0])
    K = len(vectors)
    basis = np.zeros((D, K), dtype=np.float32)

    for i, v in enumerate(vectors):
        u = v.copy()
        for j in range(i):
            u = u - np.dot(u, basis[:, j]) * basis[:, j]
        norm = np.linalg.norm(u)
        if norm > 1e-10:
            basis[:, i] = u / norm
        else:
            basis[:, i] = 0

    return basis


def reconstruct_with_basis(grad: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project grad onto orthonormal basis and reconstruct."""
    projection = basis.T @ grad
    return basis @ projection


def main():
    print("=" * 70)
    print("MOMENTUM HISTORY TEST")
    print("=" * 70)

    model_size = "tiny"
    seq_length = 32
    batch_size = 4
    history_sizes = [1, 2, 4, 8, 16, 32]
    num_test = 10

    print("\nUsing synthetic data...")
    vocab_size = 1000
    train_loader, _ = create_synthetic_data(
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_train_samples=500,
        batch_size=batch_size,
    )

    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    D = model.num_parameters
    print(f"Model: {model_size}, D={D:,}")

    max_history = max(history_sizes) + num_test
    print(f"\nCollecting {max_history} gradients...")

    train_iter = iter(train_loader)
    gradients = []

    for i in range(max_history):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad, _ = compute_gradient(model, batch.input_ids, batch.labels)
        gradients.append(grad)

    print("\nMethod 1: Raw gradient history as basis")
    print("-" * 50)

    for H in history_sizes:
        cosines = []
        for t in range(H, H + num_test):
            history_grads = gradients[t - H : t]
            test_grad = gradients[t]

            basis = orthonormalize(history_grads)
            g_hat = reconstruct_with_basis(test_grad, basis)
            cos = cosine_similarity(test_grad, g_hat)
            cosines.append(cos)

        print(f"  H={H:>2}: cosine = {np.mean(cosines):.4f} +/- {np.std(cosines):.4f}")

    print("\nMethod 2: Exponentially weighted momentum history")
    print("-" * 50)

    for H in history_sizes:
        cosines = []
        for t in range(H, H + num_test):
            history_momentums = []
            momentum = np.zeros(D, dtype=np.float32)
            beta = 0.9

            for i in range(t):
                momentum = beta * momentum + (1 - beta) * gradients[i]
                if i >= t - H:
                    history_momentums.append(momentum.copy())

            history_dirs = [m / (np.linalg.norm(m) + 1e-10) for m in history_momentums]

            test_grad = gradients[t]
            basis = orthonormalize(history_dirs)
            g_hat = reconstruct_with_basis(test_grad, basis)
            cos = cosine_similarity(test_grad, g_hat)
            cosines.append(cos)

        print(f"  H={H:>2}: cosine = {np.mean(cosines):.4f} +/- {np.std(cosines):.4f}")

    print("\nMethod 3: Power-law decayed directions")
    print("-" * 50)
    print("Using momentum + decayed versions of momentum")

    for H in history_sizes:
        cosines = []
        for t in range(max(32, H), max(32, H) + num_test):
            momentum = np.zeros(D, dtype=np.float32)
            beta = 0.9
            for i in range(t):
                momentum = beta * momentum + (1 - beta) * gradients[i]

            directions = []
            current = momentum.copy()
            for h in range(H):
                directions.append(current / (np.linalg.norm(current) + 1e-10))
                current = current * 0.8
                noise = np.random.randn(D).astype(np.float32) * 0.01
                current = current + noise

            test_grad = gradients[t]
            basis = orthonormalize(directions)
            g_hat = reconstruct_with_basis(test_grad, basis)
            cos = cosine_similarity(test_grad, g_hat)
            cosines.append(cos)

        print(f"  H={H:>2}: cosine = {np.mean(cosines):.4f} +/- {np.std(cosines):.4f}")

    print("\nMethod 4: Comparison with SVD of recent gradients")
    print("-" * 50)

    for H in history_sizes:
        cosines = []
        for t in range(H, H + num_test):
            history_grads = gradients[t - H : t]
            test_grad = gradients[t]

            G = np.stack(history_grads, axis=0)
            _, S, Vt = np.linalg.svd(G, full_matrices=False)

            K = min(H, len(S))
            basis = Vt[:K, :].T
            g_hat = reconstruct_with_basis(test_grad, basis)
            cos = cosine_similarity(test_grad, g_hat)
            cosines.append(cos)

        print(f"  H={H:>2}: cosine = {np.mean(cosines):.4f} +/- {np.std(cosines):.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key insight: The gradient subspace is DYNAMIC - it changes every step.

Options for HoloGrad:
1. Use momentum direction (K=1): ~17% capture, minimal communication
2. Use recent gradient SVD: ~25-35% capture, need to share SVD basis
3. Use gradient history: Similar to SVD but simpler

For the paper, consider:
- HoloGrad works well for CONVEX problems (stable subspace)
- For neural networks, need momentum-based approach
- Or accept that K must be very high (~D) for random projections
""")


if __name__ == "__main__":
    main()
