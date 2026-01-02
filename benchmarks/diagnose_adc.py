#!/usr/bin/env python3
"""Diagnose ADC effectiveness: Compare random vs ADC vs SVD oracle directions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from typing import Tuple, List

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data, create_synthetic_data
from holograd.protocol.direction import DirectionGenerator, ADCCodebook


def compute_gradient(model, input_ids, labels) -> Tuple[np.ndarray, float]:
    """Compute true gradient using PyTorch autograd."""
    params = model.get_flat_params()
    params_t = torch.tensor(params, dtype=torch.float32, requires_grad=True)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    params_dict = model.flat_params_to_torch_dict(params_t)
    loss = model.compute_loss_torch(input_ids_t, labels_t, params_dict)
    loss.backward()

    return params_t.grad.numpy(), loss.item()


def reconstruct_gradient_random(grad: np.ndarray, K: int, D: int, seed_prefix: str) -> np.ndarray:
    """Reconstruct gradient using K random directions."""
    direction_gen = DirectionGenerator(D)
    g_hat = np.zeros(D, dtype=np.float32)

    for k in range(K):
        seed = f"{seed_prefix}_k{k}".encode()
        result = direction_gen.generate(seed)
        v = result.direction
        scalar = np.dot(grad, v)
        g_hat += scalar * v

    g_hat = (D / K) * g_hat
    return g_hat


def reconstruct_gradient_adc(
    grad: np.ndarray, K: int, adc: ADCCodebook, seed_prefix: str
) -> np.ndarray:
    """Reconstruct gradient using K ADC directions."""
    D = adc.dimension
    g_hat = np.zeros(D, dtype=np.float32)

    for k in range(K):
        seed = f"{seed_prefix}_k{k}".encode()
        result = adc.generate_direction(seed)
        v = result.direction
        scalar = np.dot(grad, v)
        g_hat += scalar * v

    scale_factor = adc.get_scale_factor()
    g_hat = (scale_factor / K) * g_hat
    return g_hat


def reconstruct_gradient_svd(grad: np.ndarray, top_k_basis: np.ndarray) -> np.ndarray:
    """Reconstruct gradient using top-K SVD basis. top_k_basis: (D, K) orthonormal."""
    projection = top_k_basis.T @ grad
    g_hat = top_k_basis @ projection
    return g_hat


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def relative_error(g_true: np.ndarray, g_hat: np.ndarray) -> float:
    """Compute ||g_true - g_hat|| / ||g_true||."""
    norm_true = np.linalg.norm(g_true)
    if norm_true < 1e-10:
        return 0.0
    return float(np.linalg.norm(g_true - g_hat) / norm_true)


def main():
    print("=" * 70)
    print("ADC DIAGNOSTIC: Random vs ADC vs SVD Oracle")
    print("=" * 70)

    model_size = "tiny"
    seq_length = 32
    batch_size = 4
    num_gradient_samples = 30
    num_test_samples = 10
    use_wikitext = True

    K_values = [32, 64, 128, 256]
    adc_rank = 64

    if use_wikitext:
        print("\nLoading WikiText data...")
        train_loader, _, vocab_size = create_wikitext_data(
            seq_length=seq_length,
            batch_size=batch_size,
            dataset_name="wikitext-2-raw-v1",
            max_train_samples=500,
            max_val_samples=50,
        )
    else:
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

    print(f"\n[Phase 1] Collecting {num_gradient_samples} gradients for SVD basis & ADC warmup...")

    gradients = []
    train_iter = iter(train_loader)
    adc = ADCCodebook(D, rank=adc_rank, warmup_samples=num_gradient_samples, oja_alpha=1e-2)

    for i in range(num_gradient_samples):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad, _ = compute_gradient(model, batch.input_ids, batch.labels)
        gradients.append(grad)
        adc.update(grad)

        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{num_gradient_samples}")

    print("\n[Phase 2] Computing SVD basis...")
    G = np.stack(gradients, axis=0)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)

    print(f"  Singular values (top 10): {S[:10].round(4)}")
    total_energy = np.sum(S**2)
    for k in [32, 64, 128, 256]:
        if k <= len(S):
            energy = np.sum(S[:k] ** 2) / total_energy
            print(f"  Top {k} SVD: {energy * 100:.1f}% energy")

    print(f"\n[Phase 3] Testing reconstruction on {num_test_samples} new gradients...")

    print(f"\nADC status: warmed_up={adc.is_warmed_up}, step={adc.step}")

    results = {K: {"random": [], "adc": [], "svd": []} for K in K_values}

    for i in range(num_test_samples):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad_true, _ = compute_gradient(model, batch.input_ids, batch.labels)
        seed_prefix = f"test_{i}"

        adc_energy = adc.captured_energy_ratio(grad_true)

        for K in K_values:
            g_random = reconstruct_gradient_random(grad_true, K, D, seed_prefix)
            cos_random = cosine_similarity(grad_true, g_random)

            g_adc = reconstruct_gradient_adc(grad_true, K, adc, seed_prefix)
            cos_adc = cosine_similarity(grad_true, g_adc)

            svd_basis = Vt[:K, :].T
            g_svd = reconstruct_gradient_svd(grad_true, svd_basis)
            cos_svd = cosine_similarity(grad_true, g_svd)

            results[K]["random"].append(cos_random)
            results[K]["adc"].append(cos_adc)
            results[K]["svd"].append(cos_svd)

        adc.update(grad_true)

        if (i + 1) % 5 == 0:
            print(f"  Tested {i + 1}/{num_test_samples} (ADC energy: {adc_energy:.2%})")

    print("\n" + "=" * 70)
    print("RESULTS: Cosine Similarity (mean +/- std)")
    print("=" * 70)
    print(f"{'K':>6} | {'Random':>15} | {'ADC':>15} | {'SVD Oracle':>15}")
    print("-" * 60)

    for K in K_values:
        rand_mean = np.mean(results[K]["random"])
        rand_std = np.std(results[K]["random"])
        adc_mean = np.mean(results[K]["adc"])
        adc_std = np.std(results[K]["adc"])
        svd_mean = np.mean(results[K]["svd"])
        svd_std = np.std(results[K]["svd"])

        print(
            f"{K:>6} | {rand_mean:.3f} +/- {rand_std:.3f} | {adc_mean:.3f} +/- {adc_std:.3f} | {svd_mean:.3f} +/- {svd_std:.3f}"
        )

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    K = 128
    rand_mean = np.mean(results[K]["random"])
    adc_mean = np.mean(results[K]["adc"])
    svd_mean = np.mean(results[K]["svd"])

    print(f"\nFor K={K}:")
    print(f"  Random: {rand_mean:.3f}")
    print(f"  ADC:    {adc_mean:.3f} ({(adc_mean / rand_mean - 1) * 100:+.1f}% vs random)")
    print(f"  SVD:    {svd_mean:.3f} ({(svd_mean / rand_mean - 1) * 100:+.1f}% vs random)")

    if svd_mean > 0.8:
        print("\n[GOOD] SVD oracle achieves high similarity -> gradient lies in low-rank subspace")
        if adc_mean > 0.5:
            print("[GOOD] ADC is learning the subspace effectively")
        else:
            print("[PROBLEM] ADC is NOT learning the subspace despite it existing")
            print("  -> ADC implementation needs debugging")
    else:
        print("\n[PROBLEM] Even SVD oracle has low similarity")
        print("  -> Gradients don't lie in a low-rank subspace (across batches)")
        print("  -> Need per-batch SVD or different approach")

    print("\n" + "-" * 70)
    print("ADC INTERNAL STATE")
    print("-" * 70)
    print(f"ADC rank: {adc.rank}")
    print(f"ADC steps: {adc.step}")
    print(f"ADC warmed up: {adc.is_warmed_up}")
    print(f"ADC current alpha: {adc.current_alpha:.6f}")
    print(f"ADC energy EMA: {adc.energy_ema:.4f}")

    print(f"\nADC codebook shape: {adc.codebook.shape}")
    print(
        f"ADC codebook norms: min={np.linalg.norm(adc.codebook, axis=0).min():.4f}, max={np.linalg.norm(adc.codebook, axis=0).max():.4f}"
    )

    U_adc = adc.codebook
    ortho_error = np.linalg.norm(U_adc.T @ U_adc - np.eye(adc.rank))
    print(f"ADC orthogonality error: {ortho_error:.6f}")

    svd_basis = Vt[: adc.rank, :].T
    overlap = np.abs(U_adc.T @ svd_basis)
    max_overlaps = np.max(overlap, axis=1)
    print(f"\nADC-SVD basis overlap (per ADC direction):")
    print(f"  Mean max overlap: {np.mean(max_overlaps):.4f}")
    print(f"  Min max overlap: {np.min(max_overlaps):.4f}")

    _, S_overlap, _ = np.linalg.svd(U_adc.T @ svd_basis)
    subspace_similarity = np.mean(S_overlap)
    print(f"  Subspace similarity (mean singular value): {subspace_similarity:.4f}")


if __name__ == "__main__":
    main()
