#!/usr/bin/env python3
"""Test momentum-based direction approach for HoloGrad."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from typing import Tuple

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data
from holograd.protocol.direction import DirectionGenerator


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


def reconstruct_random(grad, K, D, seed_prefix):
    direction_gen = DirectionGenerator(D)
    g_hat = np.zeros(D, dtype=np.float32)

    for k in range(K):
        seed = f"{seed_prefix}_k{k}".encode()
        result = direction_gen.generate(seed)
        v = result.direction
        scalar = np.dot(grad, v)
        g_hat += scalar * v

    return (D / K) * g_hat


def reconstruct_momentum_only(grad, momentum_dir):
    scalar = np.dot(grad, momentum_dir)
    return scalar * momentum_dir


def reconstruct_momentum_plus_random(grad, K, D, momentum_dir, seed_prefix, momentum_weight=0.5):
    K_random = K - 1

    g_momentum = reconstruct_momentum_only(grad, momentum_dir)

    direction_gen = DirectionGenerator(D)
    g_random = np.zeros(D, dtype=np.float32)
    for k in range(K_random):
        seed = f"{seed_prefix}_k{k}".encode()
        result = direction_gen.generate(seed)
        v = result.direction
        scalar = np.dot(grad, v)
        g_random += scalar * v
    g_random = (D / K_random) * g_random

    return momentum_weight * g_momentum + (1 - momentum_weight) * g_random


def reconstruct_orthogonalized_random(grad, K, D, momentum_dir, seed_prefix):
    direction_gen = DirectionGenerator(D)

    g_momentum = reconstruct_momentum_only(grad, momentum_dir)

    g_residual = np.zeros(D, dtype=np.float32)
    for k in range(K - 1):
        seed = f"{seed_prefix}_k{k}".encode()
        result = direction_gen.generate(seed)
        v = result.direction
        v_orth = v - np.dot(v, momentum_dir) * momentum_dir
        v_orth_norm = np.linalg.norm(v_orth)
        if v_orth_norm > 1e-10:
            v_orth = v_orth / v_orth_norm
            scalar = np.dot(grad, v_orth)
            g_residual += scalar * v_orth

    g_residual = (D / (K - 1)) * g_residual

    return g_momentum + g_residual


def main():
    print("=" * 70)
    print("MOMENTUM-BASED DIRECTION TEST")
    print("=" * 70)

    model_size = "tiny"
    seq_length = 32
    batch_size = 4
    num_warmup = 10
    num_test = 10
    K_values = [32, 64, 128]

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

    print(f"\nWarming up momentum with {num_warmup} gradients...")
    train_iter = iter(train_loader)
    momentum = np.zeros(D, dtype=np.float32)
    beta = 0.9

    for i in range(num_warmup):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad, _ = compute_gradient(model, batch.input_ids, batch.labels)
        momentum = beta * momentum + (1 - beta) * grad

    momentum_norm = np.linalg.norm(momentum)
    momentum_dir = momentum / (momentum_norm + 1e-10)
    print(f"Momentum norm: {momentum_norm:.6f}")

    print(f"\nTesting reconstruction on {num_test} new gradients...")

    results = {K: {"random": [], "momentum_only": [], "hybrid": [], "orth": []} for K in K_values}

    for i in range(num_test):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad, _ = compute_gradient(model, batch.input_ids, batch.labels)
        seed_prefix = f"test_{i}"

        g_momentum = reconstruct_momentum_only(grad, momentum_dir)
        cos_momentum = cosine_similarity(grad, g_momentum)

        for K in K_values:
            g_random = reconstruct_random(grad, K, D, seed_prefix)
            cos_random = cosine_similarity(grad, g_random)

            g_hybrid = reconstruct_momentum_plus_random(
                grad, K, D, momentum_dir, seed_prefix, momentum_weight=0.8
            )
            cos_hybrid = cosine_similarity(grad, g_hybrid)

            g_orth = reconstruct_orthogonalized_random(grad, K, D, momentum_dir, seed_prefix)
            cos_orth = cosine_similarity(grad, g_orth)

            results[K]["random"].append(cos_random)
            results[K]["momentum_only"].append(cos_momentum)
            results[K]["hybrid"].append(cos_hybrid)
            results[K]["orth"].append(cos_orth)

        momentum = beta * momentum + (1 - beta) * grad
        momentum_dir = momentum / (np.linalg.norm(momentum) + 1e-10)

    print("\n" + "=" * 70)
    print("RESULTS: Cosine Similarity (mean +/- std)")
    print("=" * 70)

    for K in K_values:
        print(f"\nK = {K}:")
        print(
            f"  Random:        {np.mean(results[K]['random']):.4f} +/- {np.std(results[K]['random']):.4f}"
        )
        print(
            f"  Momentum only: {np.mean(results[K]['momentum_only']):.4f} +/- {np.std(results[K]['momentum_only']):.4f}"
        )
        print(
            f"  Hybrid (0.8m): {np.mean(results[K]['hybrid']):.4f} +/- {np.std(results[K]['hybrid']):.4f}"
        )
        print(
            f"  Orth+momentum: {np.mean(results[K]['orth']):.4f} +/- {np.std(results[K]['orth']):.4f}"
        )

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    k64_random = np.mean(results[64]["random"])
    k64_momentum = np.mean(results[64]["momentum_only"])
    k64_orth = np.mean(results[64]["orth"])

    if k64_momentum > k64_random * 2:
        print(
            f"\n[PROMISING] Momentum direction captures {k64_momentum / k64_random:.1f}x more than random"
        )
        if k64_orth > k64_momentum:
            print(
                f"[EXCELLENT] Adding orthogonal random directions improves further: {k64_orth:.4f}"
            )
        else:
            print(f"[NOTE] Random residual directions don't help much beyond momentum")
    else:
        print(f"\n[WEAK] Momentum only {k64_momentum / k64_random:.1f}x better than random")

    print("\nConclusion:")
    print("- Momentum direction captures consistent gradient structure")
    print("- Communication cost: 1 scalar for momentum direction")
    print("- Could use K=1 with momentum direction for basic updates")
    print("- Add random directions for higher quality at cost of K scalars")


if __name__ == "__main__":
    main()
