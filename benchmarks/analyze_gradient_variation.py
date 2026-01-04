#!/usr/bin/env python3
"""
E1: Gradient Variability Analysis

Validates paper claim C2: "NN gradients are near-orthogonal across batches"
Expected: pairwise cosine similarity ~ 0.07

Usage:
    python benchmarks/analyze_gradient_variation.py --quick          # tiny model
    python benchmarks/analyze_gradient_variation.py --full           # GPT2-like 50M
    python benchmarks/analyze_gradient_variation.py --n-layer 6 --n-head 8 --n-embd 512
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data


def compute_gradient(
    model: SimpleGPT2, input_ids: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, float]:
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


def run_experiment(
    n_layer: int,
    n_head: int,
    n_embd: int,
    seq_length: int,
    batch_size: int,
    num_gradients: int,
    vocab_size: int = 50257,
) -> Dict[str, Any]:
    print("=" * 70)
    print("E1: GRADIENT VARIABILITY ANALYSIS")
    print("Validating C2: pairwise cosine similarity ~ 0.07")
    print("=" * 70)

    results: Dict[str, Any] = {
        "config": {
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "seq_length": seq_length,
            "batch_size": batch_size,
            "num_gradients": num_gradients,
        },
        "timestamp": datetime.now().isoformat(),
    }

    print("\nLoading WikiText data...")
    train_loader, _, actual_vocab_size = create_wikitext_data(
        seq_length=seq_length,
        batch_size=batch_size,
        dataset_name="wikitext-2-raw-v1",
        max_train_samples=num_gradients * 2,
        max_val_samples=50,
    )

    print(f"Creating model: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
    model = SimpleGPT2(
        size="tiny",
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_seq_len=seq_length,
        vocab_size=actual_vocab_size,
    )
    D = model.num_parameters
    print(f"Model parameters: D={D:,}")
    results["model_params"] = D

    print(f"\nCollecting {num_gradients} gradients...")
    gradients: List[np.ndarray] = []
    losses: List[float] = []
    train_iter = iter(train_loader)

    for i in range(num_gradients):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        grad, loss = compute_gradient(model, batch.input_ids, batch.labels)
        gradients.append(grad)
        losses.append(loss)

        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{num_gradients} gradients")

    print("\n" + "-" * 70)
    print("PAIRWISE COSINE SIMILARITY")
    print("-" * 70)

    cos_matrix = np.zeros((num_gradients, num_gradients))
    for i in range(num_gradients):
        for j in range(num_gradients):
            cos_matrix[i, j] = cosine_similarity(gradients[i], gradients[j])

    off_diag = cos_matrix[np.triu_indices(num_gradients, k=1)]

    pairwise_stats = {
        "mean": float(np.mean(off_diag)),
        "std": float(np.std(off_diag)),
        "min": float(np.min(off_diag)),
        "max": float(np.max(off_diag)),
        "median": float(np.median(off_diag)),
    }
    results["pairwise_cosine"] = pairwise_stats

    print(f"Mean pairwise cosine: {pairwise_stats['mean']:.4f}")
    print(f"Std pairwise cosine:  {pairwise_stats['std']:.4f}")
    print(f"Min pairwise cosine:  {pairwise_stats['min']:.4f}")
    print(f"Max pairwise cosine:  {pairwise_stats['max']:.4f}")
    print(f"Median:               {pairwise_stats['median']:.4f}")

    print("\n" + "-" * 70)
    print("LEAVE-ONE-OUT RECONSTRUCTION")
    print("-" * 70)
    print("Train SVD on N-1 gradients, test on held-out gradient")

    loo_results: List[Dict[str, Any]] = []
    num_test = min(10, num_gradients)

    for i in range(num_test):
        train_grads = [g for j, g in enumerate(gradients) if j != i]
        test_grad = gradients[i]

        G_train = np.stack(train_grads, axis=0)
        _, S, Vt = np.linalg.svd(G_train, full_matrices=False)

        for K in [32, 64, 128]:
            if K > len(train_grads):
                continue
            basis = Vt[:K, :].T
            proj = basis @ (basis.T @ test_grad)
            cos = cosine_similarity(test_grad, proj)
            loo_results.append({"held_out": i, "K": K, "cosine": cos})

    print(f"{'Held-out':>10} | {'K':>5} | {'Cosine':>10}")
    print("-" * 35)
    for r in loo_results[:20]:
        print(f"{r['held_out']:>10} | {r['K']:>5} | {r['cosine']:>10.4f}")

    for K in [32, 64, 128]:
        k_sims = [r["cosine"] for r in loo_results if r["K"] == K]
        if k_sims:
            print(f"\nMean LOO K={K}: {np.mean(k_sims):.4f}")
            results[f"loo_k{K}_mean"] = float(np.mean(k_sims))

    print("\n" + "-" * 70)
    print("GRADIENT MOMENTUM ANALYSIS")
    print("-" * 70)

    mean_grad = np.mean(gradients[:-1], axis=0)
    test_grad = gradients[-1]
    mean_grad_normalized = mean_grad / (np.linalg.norm(mean_grad) + 1e-10)
    cos_with_mean = cosine_similarity(test_grad, mean_grad_normalized)
    print(f"Cosine(test_grad, mean_grad): {cos_with_mean:.4f}")
    results["momentum_mean_cosine"] = float(cos_with_mean)

    ema_grad = gradients[0].copy()
    ema_alpha = 0.1
    for g in gradients[1:-1]:
        ema_grad = ema_alpha * g + (1 - ema_alpha) * ema_grad
    cos_with_ema = cosine_similarity(test_grad, ema_grad)
    print(f"Cosine(test_grad, EMA_grad):  {cos_with_ema:.4f}")
    results["momentum_ema_cosine"] = float(cos_with_ema)

    print("\n" + "=" * 70)
    print("CLAIM C2 VERIFICATION")
    print("=" * 70)

    expected_cosine = 0.07
    actual_cosine = pairwise_stats["mean"]

    if abs(actual_cosine) < 0.15:
        status = "VERIFIED"
        color = "\033[92m"
    elif abs(actual_cosine) < 0.30:
        status = "PARTIAL"
        color = "\033[93m"
    else:
        status = "FAILED"
        color = "\033[91m"

    reset = "\033[0m"

    print(f"\nExpected: cosine ~ {expected_cosine}")
    print(f"Actual:   cosine = {actual_cosine:.4f}")
    print(f"Status:   {color}[{status}]{reset}")

    if status == "VERIFIED":
        print("\nGradients are near-orthogonal across batches as predicted.")
        print("This validates the paper's core observation about gradient variability.")
    elif status == "PARTIAL":
        print("\nGradients show some correlation but less than random vectors.")
        print("ADC may still provide some benefit.")
    else:
        print("\nGradients are more correlated than expected.")
        print("This may affect ADC performance.")

    results["claim_c2_status"] = status
    results["claim_c2_expected"] = expected_cosine
    results["claim_c2_actual"] = actual_cosine

    return results


def main():
    parser = argparse.ArgumentParser(
        description="E1: Gradient Variability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--quick", action="store_true", help="Quick test with tiny model")
    parser.add_argument(
        "--full", action="store_true", help="Full E1 experiment with GPT2-like 50M model"
    )
    parser.add_argument(
        "--gpt2-small", action="store_true", help="GPT-2-small 124M (paper setting)"
    )

    parser.add_argument("--n-layer", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--seq-length", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--num-gradients", type=int, default=50, help="Number of gradients to collect"
    )

    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument(
        "--output", type=str, default="results/e1_gradient_variability.json", help="Output path"
    )

    args = parser.parse_args()

    if args.quick:
        n_layer, n_head, n_embd = 2, 2, 64
        num_gradients = 20
        seq_length = 32
        batch_size = 4
    elif args.gpt2_small:
        n_layer, n_head, n_embd = 12, 12, 768
        num_gradients = 50
        seq_length = 128
        batch_size = 2
    elif args.full:
        n_layer, n_head, n_embd = 6, 8, 512
        num_gradients = 50
        seq_length = 128
        batch_size = 4
    else:
        n_layer = args.n_layer
        n_head = args.n_head
        n_embd = args.n_embd
        num_gradients = args.num_gradients
        seq_length = args.seq_length
        batch_size = args.batch_size

    results = run_experiment(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        seq_length=seq_length,
        batch_size=batch_size,
        num_gradients=num_gradients,
    )

    if args.save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
