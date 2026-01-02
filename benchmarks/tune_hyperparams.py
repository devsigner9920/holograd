#!/usr/bin/env python3
"""
Hyperparameter Tuning for HoloGrad Convergence

Systematically searches for settings that achieve actual loss reduction.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograd.core.config import HoloGradConfig, ProtocolConfig, ADCConfig
from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data
from holograd.training.trainer import HoloGradTrainer


def run_config(
    lr: float,
    k: int,
    rank: int,
    momentum: float,
    num_steps: int,
    seed: int,
) -> Dict:
    config = HoloGradConfig(
        protocol=ProtocolConfig(
            K=k,
            learning_rate=lr,
            momentum=momentum,
            global_seed=f"tune_{seed}",
        ),
        adc=ADCConfig(enabled=True, rank=rank),
    )
    config.distributed.num_workers = max(k, 8)

    model = SimpleGPT2(size="tiny")
    train_loader, val_loader = create_synthetic_data(
        vocab_size=model.vocab_size,
        seq_length=64,
        num_train_samples=max(num_steps * 4, 500),
        batch_size=4,
    )

    trainer = HoloGradTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    losses = []
    train_iter = iter(train_loader)

    start = time.perf_counter()
    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        metrics = trainer.train_step(batch)
        losses.append(metrics.loss)

    elapsed = time.perf_counter() - start
    val_loss = trainer.evaluate()

    initial = np.mean(losses[:5])
    final = np.mean(losses[-5:])
    improvement = (initial - final) / initial * 100

    return {
        "lr": lr,
        "K": k,
        "rank": rank,
        "momentum": momentum,
        "initial_loss": float(initial),
        "final_loss": float(final),
        "improvement_pct": float(improvement),
        "val_loss": float(val_loss),
        "min_loss": float(np.min(losses)),
        "time": float(elapsed),
        "losses": [float(l) for l in losses],
        "converged": improvement > 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--steps", type=int, default=100, help="Steps per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer configs)")
    parser.add_argument("--evidence", action="store_true", help="Save evidence")
    args = parser.parse_args()

    print("=" * 70)
    print("HoloGrad Hyperparameter Tuning")
    print("=" * 70)

    if args.quick:
        lrs = [1e-2, 1e-3]
        ks = [32, 64]
        ranks = [16, 32]
        momentums = [0.0, 0.9]
    else:
        lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
        ks = [16, 32, 64, 128]
        ranks = [8, 16, 32]
        momentums = [0.0, 0.5, 0.9]

    configs = list(product(lrs, ks, ranks, momentums))
    print(f"Testing {len(configs)} configurations...")
    print(f"Steps per config: {args.steps}")
    print("-" * 70)

    evidence = None
    if args.evidence:
        from holograd.experiments.evidence import ExperimentEvidence

        evidence = ExperimentEvidence("hyperparameter_tuning")
        evidence.__enter__()
        evidence.set_config(
            {
                "steps": args.steps,
                "seed": args.seed,
                "lrs": lrs,
                "ks": ks,
                "ranks": ranks,
                "momentums": momentums,
            }
        )

    results = []
    best_result = None
    best_improvement = -float("inf")

    print(
        f"\n{'LR':<10} {'K':<6} {'Rank':<6} {'Mom':<6} {'Init':<10} {'Final':<10} {'Improv':<10} {'Status':<10}"
    )
    print("-" * 70)

    for i, (lr, k, rank, mom) in enumerate(configs):
        result = run_config(
            lr=lr,
            k=k,
            rank=rank,
            momentum=mom,
            num_steps=args.steps,
            seed=args.seed,
        )
        results.append(result)

        status = "✓ GOOD" if result["converged"] else "✗"
        print(
            f"{lr:<10.0e} {k:<6} {rank:<6} {mom:<6.1f} {result['initial_loss']:<10.4f} {result['final_loss']:<10.4f} {result['improvement_pct']:>+8.2f}% {status}"
        )

        if result["improvement_pct"] > best_improvement:
            best_improvement = result["improvement_pct"]
            best_result = result

        if evidence:
            evidence.add_table_row(
                "tuning",
                {
                    "lr": lr,
                    "K": k,
                    "rank": rank,
                    "momentum": mom,
                    "initial_loss": result["initial_loss"],
                    "final_loss": result["final_loss"],
                    "improvement_pct": result["improvement_pct"],
                    "converged": result["converged"],
                },
            )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    converged = [r for r in results if r["converged"]]
    print(f"\nConverged configs: {len(converged)}/{len(results)}")

    if best_result:
        print(f"\nBest configuration:")
        print(f"  Learning rate: {best_result['lr']}")
        print(f"  K (directions): {best_result['K']}")
        print(f"  ADC rank: {best_result['rank']}")
        print(f"  Momentum: {best_result['momentum']}")
        print(f"  Improvement: {best_result['improvement_pct']:+.2f}%")
        print(f"  Final loss: {best_result['final_loss']:.4f}")

    if converged:
        print(f"\nAll converged configs (improvement > 1%):")
        sorted_converged = sorted(converged, key=lambda x: x["improvement_pct"], reverse=True)
        for r in sorted_converged[:5]:
            print(
                f"  lr={r['lr']:.0e}, K={r['K']}, rank={r['rank']}, mom={r['momentum']:.1f} -> {r['improvement_pct']:+.2f}%"
            )

    if evidence:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for r in sorted(results, key=lambda x: x["improvement_pct"], reverse=True)[:10]:
                label = f"lr={r['lr']:.0e},K={r['K']}"
                axes[0, 0].plot(r["losses"], label=label, alpha=0.7)
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Top 10 Training Curves")
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)

            improvements = [r["improvement_pct"] for r in results]
            axes[0, 1].hist(improvements, bins=20, edgecolor="black")
            axes[0, 1].axvline(x=1.0, color="r", linestyle="--", label="Convergence threshold")
            axes[0, 1].set_xlabel("Improvement %")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title("Improvement Distribution")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            lr_improvements = {}
            for r in results:
                lr = r["lr"]
                if lr not in lr_improvements:
                    lr_improvements[lr] = []
                lr_improvements[lr].append(r["improvement_pct"])

            lr_labels = [f"{lr:.0e}" for lr in sorted(lr_improvements.keys())]
            lr_means = [np.mean(lr_improvements[lr]) for lr in sorted(lr_improvements.keys())]
            axes[1, 0].bar(range(len(lr_labels)), lr_means, tick_label=lr_labels)
            axes[1, 0].set_xlabel("Learning Rate")
            axes[1, 0].set_ylabel("Mean Improvement %")
            axes[1, 0].set_title("Improvement by Learning Rate")
            axes[1, 0].grid(True, alpha=0.3, axis="y")

            k_improvements = {}
            for r in results:
                k = r["K"]
                if k not in k_improvements:
                    k_improvements[k] = []
                k_improvements[k].append(r["improvement_pct"])

            k_labels = [str(k) for k in sorted(k_improvements.keys())]
            k_means = [np.mean(k_improvements[k]) for k in sorted(k_improvements.keys())]
            axes[1, 1].bar(range(len(k_labels)), k_means, tick_label=k_labels)
            axes[1, 1].set_xlabel("K (Directions)")
            axes[1, 1].set_ylabel("Mean Improvement %")
            axes[1, 1].set_title("Improvement by K")
            axes[1, 1].grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            evidence.save_figure("tuning_results", fig)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create figure: {e}")

        if best_result:
            evidence.add_metadata(
                "best_config",
                {
                    "lr": best_result["lr"],
                    "K": best_result["K"],
                    "rank": best_result["rank"],
                    "momentum": best_result["momentum"],
                    "improvement_pct": best_result["improvement_pct"],
                },
            )

        evidence.__exit__(None, None, None)
        print(f"\nEvidence saved to: {evidence.output_dir}")

    if best_result and best_result["converged"]:
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        print(f"""
다음 설정으로 장기 학습을 실행하세요:

python benchmarks/train_wikitext.py \\
    --steps 1000 \\
    --lr {best_result["lr"]} \\
    --K {best_result["K"]} \\
    --rank {best_result["rank"]} \\
    --momentum {best_result["momentum"]} \\
    --evidence
""")


if __name__ == "__main__":
    main()
