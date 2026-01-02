#!/usr/bin/env python3
"""Measure effective rank of gradients during training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data, create_synthetic_data


def compute_gradient(model, input_ids, labels):
    params = model.get_flat_params()
    params_t = torch.tensor(params, dtype=torch.float32, requires_grad=True)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    params_dict = model.flat_params_to_torch_dict(params_t)
    loss = model.compute_loss_torch(input_ids_t, labels_t, params_dict)
    loss.backward()

    return params_t.grad.numpy(), loss.item()


def measure_effective_rank(singular_values, threshold=0.99):
    """r such that top r singular values capture `threshold` of total energy."""
    total_energy = np.sum(singular_values**2)
    cumulative_energy = np.cumsum(singular_values**2)
    r = np.searchsorted(cumulative_energy, threshold * total_energy) + 1
    return min(r, len(singular_values))


def analyze_gradient(gradient, name="gradient"):
    D = len(gradient)
    norm = np.linalg.norm(gradient)

    print(f"\n{'=' * 60}")
    print(f"{name} (D={D:,})")
    print(f"{'=' * 60}")
    print(f"Norm: {norm:.6f}")
    print(f"Mean abs: {np.mean(np.abs(gradient)):.8f}")
    print(f"Max abs: {np.max(np.abs(gradient)):.6f}")

    # Participation ratio: (Σ|g_i|)² / Σ|g_i|² - measures intrinsic dimensionality
    participation_ratio = (np.sum(np.abs(gradient)) ** 2) / (np.sum(gradient**2) + 1e-10)
    print(
        f"Participation ratio: {participation_ratio:.1f} / {D:,} ({100 * participation_ratio / D:.2f}%)"
    )

    near_zero = np.sum(np.abs(gradient) < 1e-6)
    print(f"Near-zero elements: {near_zero:,} / {D:,} ({100 * near_zero / D:.1f}%)")

    return participation_ratio


def analyze_gradient_matrix(gradients, name="gradient matrix"):
    G = np.stack(gradients, axis=0)
    num_samples, D = G.shape

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"Matrix shape: {num_samples} samples x {D:,} parameters")

    print("Computing SVD...")
    U, S, Vt = np.linalg.svd(G, full_matrices=False)

    print(f"Number of singular values: {len(S)}")
    print(f"Top 10 singular values: {S[:10]}")

    for threshold in [0.5, 0.9, 0.95, 0.99]:
        r = measure_effective_rank(S, threshold)
        print(f"Effective rank ({threshold * 100:.0f}% energy): {r} / {min(num_samples, D)}")

    total_energy = np.sum(S**2)
    top1_energy = S[0] ** 2 / total_energy
    top10_energy = np.sum(S[:10] ** 2) / total_energy
    top50_energy = np.sum(S[:50] ** 2) / total_energy if len(S) >= 50 else 1.0

    print(f"\nEnergy concentration:")
    print(f"  Top 1: {top1_energy * 100:.1f}%")
    print(f"  Top 10: {top10_energy * 100:.1f}%")
    if len(S) >= 50:
        print(f"  Top 50: {top50_energy * 100:.1f}%")

    return S


def main():
    print("=" * 60)
    print("EFFECTIVE RANK MEASUREMENT")
    print("=" * 60)

    model_size = "tiny"
    seq_length = 32
    batch_size = 4
    num_samples = 50
    use_wikitext = True

    if use_wikitext:
        print("\nLoading WikiText data...")
        train_loader, _, vocab_size = create_wikitext_data(
            seq_length=seq_length,
            batch_size=batch_size,
            dataset_name="wikitext-2-raw-v1",
            max_train_samples=500,
            max_val_samples=50,
        )
        model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    else:
        print("\nUsing synthetic data...")
        vocab_size = 100
        train_loader, _ = create_synthetic_data(
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_train_samples=500,
            batch_size=batch_size,
        )
        model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)

    D = model.num_parameters
    print(f"Model: {model_size}, D={D:,}, vocab={vocab_size}")

    print(f"\nCollecting {num_samples} gradient samples...")
    gradients = []
    losses = []

    train_iter = iter(train_loader)
    for i in range(num_samples):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad, loss = compute_gradient(model, batch.input_ids, batch.labels)
        gradients.append(grad)
        losses.append(loss)

        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{num_samples} gradients")

    print(f"Average loss: {np.mean(losses):.4f}")

    analyze_gradient(gradients[0], "Single gradient (batch 0)")

    S = analyze_gradient_matrix(gradients, f"Gradient matrix ({num_samples} samples)")

    r_90 = measure_effective_rank(S, 0.90)
    r_99 = measure_effective_rank(S, 0.99)

    print(f"\n{'=' * 60}")
    print("IMPLICATIONS FOR HOLOGRAD")
    print(f"{'=' * 60}")
    print(f"Model dimension D = {D:,}")
    print(f"Effective rank (90% energy) = {r_90}")
    print(f"Effective rank (99% energy) = {r_99}")
    print(f"\nIf ADC learns this subspace:")
    print(f"  Need K >= {r_90} for 90% gradient capture")
    print(f"  Need K >= {r_99} for 99% gradient capture")
    print(f"\nCurrent K=64 captures: {min(64, len(S))} directions")

    if r_99 < 100:
        print(f"\n[GOOD] Effective rank is low!")
        print(f"  K=64~128 should be sufficient with proper ADC")
    elif r_99 < D // 10:
        print(f"\n[MODERATE] Effective rank is {r_99}")
        print(f"  Need K={r_99}+ or stronger ADC")
    else:
        print(f"\n[PROBLEM] Effective rank is high ({r_99})")
        print(f"  Low-rank assumption may not hold")


if __name__ == "__main__":
    main()
