#!/usr/bin/env python3
"""Analyze batch-to-batch gradient variation to understand why ADC fails."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from typing import Tuple

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data


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


def main():
    print("=" * 70)
    print("BATCH-TO-BATCH GRADIENT VARIATION ANALYSIS")
    print("=" * 70)

    model_size = "tiny"
    seq_length = 32
    batch_size = 4
    num_gradients = 20

    print("\nLoading WikiText data...")
    train_loader, _, vocab_size = create_wikitext_data(
        seq_length=seq_length,
        batch_size=batch_size,
        dataset_name="wikitext-2-raw-v1",
        max_train_samples=500,
        max_val_samples=50,
    )

    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    D = model.num_parameters
    print(f"Model: {model_size}, D={D:,}")

    print(f"\nCollecting {num_gradients} gradients...")
    gradients = []
    train_iter = iter(train_loader)

    for i in range(num_gradients):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad, _ = compute_gradient(model, batch.input_ids, batch.labels)
        gradients.append(grad)

    print("\n" + "-" * 70)
    print("PAIRWISE COSINE SIMILARITY")
    print("-" * 70)

    cos_matrix = np.zeros((num_gradients, num_gradients))
    for i in range(num_gradients):
        for j in range(num_gradients):
            cos_matrix[i, j] = cosine_similarity(gradients[i], gradients[j])

    off_diag = cos_matrix[np.triu_indices(num_gradients, k=1)]
    print(f"Mean pairwise cosine: {np.mean(off_diag):.4f}")
    print(f"Std pairwise cosine:  {np.std(off_diag):.4f}")
    print(f"Min pairwise cosine:  {np.min(off_diag):.4f}")
    print(f"Max pairwise cosine:  {np.max(off_diag):.4f}")

    print("\n" + "-" * 70)
    print("SINGLE GRADIENT SVD (self-reconstruction)")
    print("-" * 70)
    print("How much of a single gradient lies in a low-rank subspace?")

    G_single = gradients[0].reshape(1, -1)
    g_norm = np.linalg.norm(gradients[0])
    print(f"Gradient norm: {g_norm:.4f}")
    print(f"Note: Single vector is rank-1 by definition")

    print("\n" + "-" * 70)
    print("LEAVE-ONE-OUT RECONSTRUCTION")
    print("-" * 70)
    print("Train SVD on N-1 gradients, test on held-out gradient")

    loo_similarities = []
    for i in range(min(10, num_gradients)):
        train_grads = [g for j, g in enumerate(gradients) if j != i]
        test_grad = gradients[i]

        G_train = np.stack(train_grads, axis=0)
        _, S, Vt = np.linalg.svd(G_train, full_matrices=False)

        for K in [32, 64]:
            basis = Vt[:K, :].T
            proj = basis @ (basis.T @ test_grad)
            cos = cosine_similarity(test_grad, proj)
            loo_similarities.append((i, K, cos))

    print(f"{'Held-out':>10} | {'K':>5} | {'Cosine':>10}")
    print("-" * 35)
    for i, K, cos in loo_similarities:
        print(f"{i:>10} | {K:>5} | {cos:>10.4f}")

    k32_sims = [c for i, k, c in loo_similarities if k == 32]
    k64_sims = [c for i, k, c in loo_similarities if k == 64]
    print(f"\nMean K=32: {np.mean(k32_sims):.4f}")
    print(f"Mean K=64: {np.mean(k64_sims):.4f}")

    print("\n" + "-" * 70)
    print("CUMULATIVE SVD ANALYSIS")
    print("-" * 70)
    print("How does reconstruction improve as we add more training gradients?")

    test_grad = gradients[-1]
    results = []

    for n_train in [2, 5, 10, 15, 19]:
        if n_train >= num_gradients:
            continue
        train_grads = gradients[:n_train]
        G_train = np.stack(train_grads, axis=0)
        _, S, Vt = np.linalg.svd(G_train, full_matrices=False)

        for K in [16, 32, 64]:
            if K > n_train:
                continue
            basis = Vt[:K, :].T
            proj = basis @ (basis.T @ test_grad)
            cos = cosine_similarity(test_grad, proj)
            results.append((n_train, K, cos))

    print(f"{'N_train':>8} | {'K':>5} | {'Cosine':>10}")
    print("-" * 35)
    for n, k, cos in results:
        print(f"{n:>8} | {k:>5} | {cos:>10.4f}")

    print("\n" + "-" * 70)
    print("GRADIENT MOMENTUM ANALYSIS")
    print("-" * 70)
    print("Does the average gradient help with reconstruction?")

    mean_grad = np.mean(gradients[:-1], axis=0)
    test_grad = gradients[-1]

    mean_grad_normalized = mean_grad / (np.linalg.norm(mean_grad) + 1e-10)
    cos_with_mean = cosine_similarity(test_grad, mean_grad_normalized)
    print(f"Cosine(test_grad, mean_grad): {cos_with_mean:.4f}")

    ema_grad = gradients[0].copy()
    ema_alpha = 0.1
    for g in gradients[1:-1]:
        ema_grad = ema_alpha * g + (1 - ema_alpha) * ema_grad

    cos_with_ema = cosine_similarity(test_grad, ema_grad)
    print(f"Cosine(test_grad, EMA_grad): {cos_with_ema:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    mean_loo_k64 = np.mean(k64_sims)
    mean_pairwise = np.mean(off_diag)

    if mean_loo_k64 < 0.3:
        print("\n[CRITICAL] Gradients are highly variable across batches")
        print(f"  - Leave-one-out reconstruction: {mean_loo_k64:.4f}")
        print(f"  - Mean pairwise cosine: {mean_pairwise:.4f}")
        print("\n  This means:")
        print("  1. No common low-rank subspace exists across batches")
        print("  2. ADC cannot learn a useful direction set")
        print("  3. Random projection variance is fundamentally irreducible")
        print("\n  Potential solutions:")
        print("  - Coordinate descent (update parameter subsets)")
        print("  - SignSGD (only communicate sign of projection)")
        print("  - Per-layer projections (different K per layer)")
        print("  - Much higher K (defeats communication efficiency)")
    else:
        print(f"\n[OK] Some gradient structure exists (LOO: {mean_loo_k64:.4f})")
        print("  ADC should be able to learn something useful")


if __name__ == "__main__":
    main()
