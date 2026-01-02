#!/usr/bin/env python3
"""
K Parameter Ablation Benchmark

Tests sensitivity of HoloGrad to the number of directions K.
K controls communication cost vs gradient approximation quality.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograd.core.config import HoloGradConfig, ProtocolConfig, ADCConfig
from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data
from holograd.training.trainer import HoloGradTrainer


def run_k_experiment(
    k_value: int,
    num_steps: int,
    seed: int,
    use_adc: bool = True,
    adc_rank: int = 32,
) -> Dict:
    config = HoloGradConfig(
        protocol=ProtocolConfig(
            K=k_value,
            learning_rate=1e-3,
            global_seed=f"k_ablation_{seed}",
        ),
        adc=ADCConfig(enabled=use_adc, rank=adc_rank),
    )
    config.distributed.num_workers = max(k_value, 8)

    model = SimpleGPT2(size="tiny")
    train_loader, val_loader = create_synthetic_data(
        vocab_size=model.vocab_size,
        seq_length=64,
        num_train_samples=num_steps * 4,
        batch_size=4,
    )

    trainer = HoloGradTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    losses = []
    energies = []
    step_times = []
    train_iter = iter(train_loader)

    start_time = time.perf_counter()

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        metrics = trainer.train_step(batch)
        losses.append(metrics.loss)
        energies.append(metrics.captured_energy_ratio)
        step_times.append(metrics.step_time)

    total_time = time.perf_counter() - start_time
    val_loss = trainer.evaluate()

    scalar_bytes = k_value * 8
    overhead_bytes = 32
    comm_bytes_per_step = scalar_bytes + overhead_bytes

    return {
        "K": k_value,
        "use_adc": use_adc,
        "adc_rank": adc_rank,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "mean_loss": float(np.mean(losses)),
        "val_loss": float(val_loss),
        "mean_energy": float(np.mean(energies)),
        "mean_step_time": float(np.mean(step_times)),
        "total_time": float(total_time),
        "comm_bytes_per_step": comm_bytes_per_step,
        "losses": [float(l) for l in losses],
        "energies": [float(e) for e in energies],
    }


def main():
    parser = argparse.ArgumentParser(description="K Parameter Ablation Benchmark")
    parser.add_argument("--num-steps", type=int, default=100, help="Training steps per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--evidence", action="store_true", help="Save evidence")
    args = parser.parse_args()

    print("=" * 60)
    print("K Parameter Ablation Benchmark")
    print("=" * 60)
    print(f"Steps per config: {args.num_steps}")
    print(f"Seed: {args.seed}")
    print("-" * 60)

    evidence = None
    if args.evidence:
        from holograd.experiments.evidence import ExperimentEvidence

        evidence = ExperimentEvidence("k_ablation")
        evidence.__enter__()
        evidence.set_config(
            {
                "num_steps": args.num_steps,
                "seed": args.seed,
            }
        )

    k_values = [4, 8, 16, 32, 64, 128]
    results = []

    print(
        f"\n{'K':<6} {'Final Loss':<12} {'Val Loss':<12} {'Energy':<10} {'Comm (KB)':<12} {'Time (s)':<10}"
    )
    print("-" * 62)

    for k in k_values:
        print(f"Running K={k}...", end=" ", flush=True)

        result = run_k_experiment(
            k_value=k,
            num_steps=args.num_steps,
            seed=args.seed,
        )
        results.append(result)

        comm_kb = result["comm_bytes_per_step"] / 1024
        print(
            f"\r{k:<6} {result['final_loss']:<12.4f} {result['val_loss']:<12.4f} {result['mean_energy']:<10.3f} {comm_kb:<12.2f} {result['total_time']:<10.1f}"
        )

        if evidence:
            evidence.add_table_row(
                "k_sweep",
                {
                    "K": k,
                    "final_loss": result["final_loss"],
                    "val_loss": result["val_loss"],
                    "mean_energy": result["mean_energy"],
                    "comm_bytes": result["comm_bytes_per_step"],
                    "total_time": result["total_time"],
                },
            )

            for step, loss in enumerate(result["losses"]):
                evidence.add_result(f"loss_K{k}", loss)

    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    baseline_k = 64
    baseline_result = next(r for r in results if r["K"] == baseline_k)

    print(f"\nRelative to K={baseline_k} baseline:")
    print(f"{'K':<6} {'Loss Ratio':<12} {'Comm Ratio':<12} {'Efficiency':<12}")
    print("-" * 42)

    for r in results:
        loss_ratio = r["final_loss"] / baseline_result["final_loss"]
        comm_ratio = r["comm_bytes_per_step"] / baseline_result["comm_bytes_per_step"]
        efficiency = loss_ratio / comm_ratio

        print(f"{r['K']:<6} {loss_ratio:<12.3f} {comm_ratio:<12.3f} {efficiency:<12.3f}")

        if evidence:
            evidence.add_table_row(
                "relative",
                {
                    "K": r["K"],
                    "loss_ratio": loss_ratio,
                    "comm_ratio": comm_ratio,
                    "efficiency": efficiency,
                },
            )

    if evidence:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            ks = [r["K"] for r in results]
            final_losses = [r["final_loss"] for r in results]
            val_losses = [r["val_loss"] for r in results]
            energies = [r["mean_energy"] for r in results]
            comm_bytes = [r["comm_bytes_per_step"] / 1024 for r in results]

            axes[0, 0].plot(ks, final_losses, "o-", label="Train Loss")
            axes[0, 0].plot(ks, val_losses, "s--", label="Val Loss")
            axes[0, 0].set_xlabel("K (directions)")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Loss vs K")
            axes[0, 0].legend()
            axes[0, 0].set_xscale("log", base=2)
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(ks, energies, "o-", color="green")
            axes[0, 1].set_xlabel("K (directions)")
            axes[0, 1].set_ylabel("Captured Energy Ratio")
            axes[0, 1].set_title("ADC Energy Capture vs K")
            axes[0, 1].set_xscale("log", base=2)
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].bar(range(len(ks)), comm_bytes, tick_label=[str(k) for k in ks])
            axes[1, 0].set_xlabel("K (directions)")
            axes[1, 0].set_ylabel("Communication (KB)")
            axes[1, 0].set_title("Communication Cost vs K")
            axes[1, 0].grid(True, alpha=0.3, axis="y")

            for r in results:
                axes[1, 1].plot(r["losses"], label=f"K={r['K']}", alpha=0.7)
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].set_title("Training Curves")
            axes[1, 1].legend(loc="upper right")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            evidence.save_figure("k_ablation", fig)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create figure: {e}")

        evidence.add_metadata(
            "summary",
            {
                "k_values_tested": k_values,
                "baseline_k": baseline_k,
                "best_k": min(results, key=lambda r: r["final_loss"])["K"],
                "most_efficient_k": max(
                    results, key=lambda r: r["final_loss"] / r["comm_bytes_per_step"]
                )["K"],
            },
        )
        evidence.__exit__(None, None, None)
        print(f"\nEvidence saved to: {evidence.output_dir}")


if __name__ == "__main__":
    main()
