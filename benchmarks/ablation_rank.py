#!/usr/bin/env python3
"""
ADC Rank Ablation Benchmark

Tests sensitivity of HoloGrad to ADC subspace rank.
Rank controls compression vs gradient approximation quality.
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


def run_rank_experiment(
    rank_value: int,
    num_steps: int,
    seed: int,
    k_value: int = 16,
) -> Dict:
    use_adc = rank_value > 0

    config = HoloGradConfig(
        protocol=ProtocolConfig(
            K=k_value,
            learning_rate=1e-3,
            global_seed=f"rank_ablation_{seed}",
        ),
        adc=ADCConfig(enabled=use_adc, rank=rank_value if use_adc else 32),
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

    adc_bytes = rank_value * 4 if use_adc else 0

    return {
        "rank": rank_value,
        "use_adc": use_adc,
        "K": k_value,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "mean_loss": float(np.mean(losses)),
        "val_loss": float(val_loss),
        "mean_energy": float(np.mean(energies)),
        "final_energy": float(energies[-1]) if energies else 0.0,
        "mean_step_time": float(np.mean(step_times)),
        "total_time": float(total_time),
        "adc_bytes_per_proof": adc_bytes,
        "losses": [float(l) for l in losses],
        "energies": [float(e) for e in energies],
    }


def main():
    parser = argparse.ArgumentParser(description="ADC Rank Ablation Benchmark")
    parser.add_argument("--num-steps", type=int, default=100, help="Training steps per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--evidence", action="store_true", help="Save evidence")
    args = parser.parse_args()

    print("=" * 60)
    print("ADC Rank Ablation Benchmark")
    print("=" * 60)
    print(f"Steps per config: {args.num_steps}")
    print(f"Seed: {args.seed}")
    print("-" * 60)

    evidence = None
    if args.evidence:
        from holograd.experiments.evidence import ExperimentEvidence

        evidence = ExperimentEvidence("rank_ablation")
        evidence.__enter__()
        evidence.set_config(
            {
                "num_steps": args.num_steps,
                "seed": args.seed,
            }
        )

    rank_values = [0, 4, 8, 16, 32, 64]
    results = []

    print(
        f"\n{'Rank':<6} {'ADC':<6} {'Final Loss':<12} {'Val Loss':<12} {'Energy':<10} {'ADC Bytes':<12}"
    )
    print("-" * 58)

    for rank in rank_values:
        adc_str = "Yes" if rank > 0 else "No"
        print(f"Running rank={rank}...", end=" ", flush=True)

        result = run_rank_experiment(
            rank_value=rank,
            num_steps=args.num_steps,
            seed=args.seed,
        )
        results.append(result)

        print(
            f"\r{rank:<6} {adc_str:<6} {result['final_loss']:<12.4f} {result['val_loss']:<12.4f} {result['mean_energy']:<10.3f} {result['adc_bytes_per_proof']:<12}"
        )

        if evidence:
            evidence.add_table_row(
                "rank_sweep",
                {
                    "rank": rank,
                    "use_adc": result["use_adc"],
                    "final_loss": result["final_loss"],
                    "val_loss": result["val_loss"],
                    "mean_energy": result["mean_energy"],
                    "adc_bytes": result["adc_bytes_per_proof"],
                    "total_time": result["total_time"],
                },
            )

            for step, (loss, energy) in enumerate(zip(result["losses"], result["energies"])):
                evidence.add_result(f"loss_rank{rank}", loss)
                evidence.add_result(f"energy_rank{rank}", energy)

    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    baseline_result = next(r for r in results if r["rank"] == 32)
    no_adc_result = next(r for r in results if r["rank"] == 0)

    print(f"\nRelative to rank=32 baseline:")
    print(f"{'Rank':<6} {'Loss Ratio':<12} {'Energy Ratio':<14} {'Bytes Ratio':<12}")
    print("-" * 44)

    for r in results:
        loss_ratio = r["final_loss"] / baseline_result["final_loss"]
        energy_ratio = r["mean_energy"] / (baseline_result["mean_energy"] + 1e-10)
        bytes_ratio = (
            r["adc_bytes_per_proof"] / (baseline_result["adc_bytes_per_proof"] + 1e-10)
            if r["rank"] > 0
            else 0
        )

        print(f"{r['rank']:<6} {loss_ratio:<12.3f} {energy_ratio:<14.3f} {bytes_ratio:<12.3f}")

        if evidence:
            evidence.add_table_row(
                "relative",
                {
                    "rank": r["rank"],
                    "loss_ratio": loss_ratio,
                    "energy_ratio": energy_ratio,
                    "bytes_ratio": bytes_ratio,
                },
            )

    print(f"\nADC vs No-ADC comparison:")
    print(f"  No ADC loss:  {no_adc_result['final_loss']:.4f}")
    print(f"  ADC-32 loss:  {baseline_result['final_loss']:.4f}")
    print(f"  ADC-32 energy: {baseline_result['mean_energy']:.3f}")

    if evidence:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            ranks = [r["rank"] for r in results if r["rank"] > 0]
            final_losses = [r["final_loss"] for r in results if r["rank"] > 0]
            val_losses = [r["val_loss"] for r in results if r["rank"] > 0]
            energies = [r["mean_energy"] for r in results if r["rank"] > 0]
            adc_bytes = [r["adc_bytes_per_proof"] for r in results if r["rank"] > 0]

            axes[0, 0].plot(ranks, final_losses, "o-", label="Train Loss")
            axes[0, 0].plot(ranks, val_losses, "s--", label="Val Loss")
            axes[0, 0].axhline(
                y=no_adc_result["final_loss"], color="r", linestyle=":", label="No ADC"
            )
            axes[0, 0].set_xlabel("ADC Rank")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Loss vs ADC Rank")
            axes[0, 0].legend()
            axes[0, 0].set_xscale("log", base=2)
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(ranks, energies, "o-", color="green")
            axes[0, 1].set_xlabel("ADC Rank")
            axes[0, 1].set_ylabel("Captured Energy Ratio")
            axes[0, 1].set_title("Energy Capture vs ADC Rank")
            axes[0, 1].set_xscale("log", base=2)
            axes[0, 1].set_ylim([0, 1.1])
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].bar(range(len(ranks)), adc_bytes, tick_label=[str(r) for r in ranks])
            axes[1, 0].set_xlabel("ADC Rank")
            axes[1, 0].set_ylabel("ADC Bytes per Proof")
            axes[1, 0].set_title("Communication Overhead vs Rank")
            axes[1, 0].grid(True, alpha=0.3, axis="y")

            for r in results:
                if r["rank"] > 0:
                    label = f"rank={r['rank']}"
                    axes[1, 1].plot(r["energies"], label=label, alpha=0.7)
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Captured Energy")
            axes[1, 1].set_title("Energy Capture Over Training")
            axes[1, 1].legend(loc="lower right")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            evidence.save_figure("rank_ablation", fig)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create figure: {e}")

        evidence.add_metadata(
            "summary",
            {
                "rank_values_tested": rank_values,
                "baseline_rank": 32,
                "best_rank": min(
                    [r for r in results if r["rank"] > 0], key=lambda r: r["final_loss"]
                )["rank"],
                "adc_vs_no_adc_improvement": (
                    no_adc_result["final_loss"] - baseline_result["final_loss"]
                )
                / no_adc_result["final_loss"],
            },
        )
        evidence.__exit__(None, None, None)
        print(f"\nEvidence saved to: {evidence.output_dir}")


if __name__ == "__main__":
    main()
