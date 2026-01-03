#!/usr/bin/env python3 -u
"""
WikiText Training Benchmark with Evidence Collection.

Usage:
    python benchmarks/train_wikitext.py --steps 100 --evidence
    python benchmarks/train_wikitext.py --dataset wikitext-2-raw-v1 --steps 50
    python benchmarks/train_wikitext.py --device cuda --model-size small --steps 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograd.core.config import (
    HoloGradConfig,
    ProtocolConfig,
    ADCConfig,
    VerificationConfig,
    AggregationConfig,
    DistributedConfig,
    TrainingConfig,
    LoggingConfig,
)
from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data
from holograd.training.trainer import HoloGradTrainer
from holograd.utils.logging import MetricsLogger
from holograd.experiments.evidence import ExperimentEvidence


def create_training_figures(
    evidence: ExperimentEvidence, losses: list[float], val_losses: list[tuple[int, float]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1, ax = plt.subplots(figsize=(10, 6))
    steps = list(range(1, len(losses) + 1))
    ax.plot(steps, losses, "b-", alpha=0.3, linewidth=0.5, label="Train Loss (raw)")

    window = min(10, len(losses) // 5 + 1)
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        smooth_steps = list(range(window, len(losses) + 1))
        ax.plot(
            smooth_steps, smoothed, "b-", linewidth=2, label=f"Train Loss (smoothed, w={window})"
        )

    if val_losses:
        val_steps, val_vals = zip(*val_losses)
        ax.plot(val_steps, val_vals, "ro-", linewidth=2, markersize=6, label="Val Loss")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("HoloGrad Training on WikiText", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    evidence.save_figure("training_curve", fig1)
    plt.close(fig1)

    if len(losses) > 10:
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(steps, losses, "b-", alpha=0.5, linewidth=1)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss (log scale)", fontsize=12)
        ax.set_title("Training Loss (Log Scale)", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        evidence.save_figure("training_curve_log", fig2)
        plt.close(fig2)


def run_training(
    model_size: str = "tiny",
    dataset_name: str = "wikitext-2-raw-v1",
    steps: int = 100,
    batch_size: int = 4,
    seq_length: int = 64,
    lr: float = 1e-3,
    K: int = 16,
    num_workers: int = 4,
    use_adc: bool = True,
    adc_rank: int = 32,
    max_train_samples: int = 5000,
    max_val_samples: int = 500,
    eval_interval: int = 25,
    log_interval: int = 5,
    save_evidence: bool = True,
    device: str = "auto",
) -> dict:
    if save_evidence:
        evidence = ExperimentEvidence("wikitext_training")
        evidence.set_config(
            {
                "model_size": model_size,
                "dataset": dataset_name,
                "steps": steps,
                "batch_size": batch_size,
                "seq_length": seq_length,
                "lr": lr,
                "K": K,
                "num_workers": num_workers,
                "use_adc": use_adc,
                "adc_rank": adc_rank if use_adc else 0,
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
            }
        )
    else:
        evidence = None

    import torch

    if device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        actual_device = device

    print("=" * 60, flush=True)
    print("HoloGrad WikiText Training", flush=True)
    print("=" * 60, flush=True)
    print(f"Model: {model_size}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Steps: {steps}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Sequence length: {seq_length}", flush=True)
    print(f"Learning rate: {lr}", flush=True)
    print(f"K (directions): {K}", flush=True)
    print(f"Workers: {num_workers}", flush=True)
    print(f"ADC: {'enabled (rank=' + str(adc_rank) + ')' if use_adc else 'disabled'}", flush=True)
    print(f"Device: {actual_device}", flush=True)
    print("-" * 60, flush=True)

    print("Loading WikiText dataset...", flush=True)
    train_loader, val_loader, vocab_size = create_wikitext_data(
        seq_length=seq_length,
        batch_size=batch_size,
        dataset_name=dataset_name,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)
    print(f"Vocab size: {vocab_size}", flush=True)

    print("Creating model...", flush=True)
    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    print(f"Model parameters: {model.num_parameters:,}", flush=True)

    config = HoloGradConfig(
        protocol=ProtocolConfig(
            K=K,
            global_seed="wikitext_holograd",
            learning_rate=lr,
        ),
        adc=ADCConfig(
            enabled=use_adc,
            rank=adc_rank,
            warmup_samples=0,
            alpha_decay=0.99,
            use_power_iteration=False,
        ),
        verification=VerificationConfig(p_verify=0.0),
        aggregation=AggregationConfig(tau=0.1),
        distributed=DistributedConfig(
            num_workers=num_workers,
            simulate_delays=False,
        ),
        training=TrainingConfig(
            max_steps=steps,
            batch_size=batch_size,
            sequence_length=seq_length,
            eval_interval=eval_interval,
        ),
        logging=LoggingConfig(log_interval=log_interval),
    )

    print("Initializing trainer...", flush=True)
    trainer = HoloGradTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=actual_device,
    )

    print("-" * 60, flush=True)
    print("Starting training...", flush=True)
    print("-" * 60, flush=True)

    val_losses = []
    step_timeout_seconds = 300
    debug_mode = steps <= 20

    train_iter = iter(train_loader)
    for step in range(steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step_start = time.time()
        metrics = trainer.train_step(batch, debug=debug_mode)
        step_elapsed = time.time() - step_start

        if step_elapsed > step_timeout_seconds:
            print(
                f"[ERROR] Step {step + 1} exceeded {step_timeout_seconds}s. Aborting.", flush=True
            )
            break

        if (step + 1) % log_interval == 0:
            print(
                f"Step {step + 1}/{steps} | Loss: {metrics.loss:.4f} | "
                f"Time: {metrics.step_time:.2f}s | Energy: {metrics.captured_energy_ratio:.3f}",
                flush=True,
            )

        if (step + 1) % eval_interval == 0:
            val_loss = trainer.evaluate()
            val_losses.append((step + 1, val_loss))
            print(f"  -> Val Loss: {val_loss:.4f}", flush=True)

    final_val_loss = trainer.evaluate()

    print("-" * 60)
    print("Training complete!")
    print("-" * 60)

    losses = trainer.state.training_losses
    initial_loss = np.mean(losses[:3]) if len(losses) >= 3 else losses[0]
    final_loss = np.mean(losses[-3:]) if len(losses) >= 3 else losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    results = {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "improvement_pct": float(improvement),
        "final_val_loss": float(final_val_loss),
        "total_steps": steps,
        "total_tokens": trainer.state.total_tokens,
        "min_loss": float(np.min(losses)),
        "mean_loss": float(np.mean(losses)),
    }

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Final val loss: {final_val_loss:.4f}")
    print(f"Total tokens: {trainer.state.total_tokens:,}")

    if evidence:
        for i, loss in enumerate(losses):
            evidence.add_result("train_loss", loss)

        for step, val_loss in val_losses:
            evidence.add_table_row("validation", {"step": step, "val_loss": val_loss})

        evidence.add_metadata("results", results)
        evidence.add_metadata("model_params", model.num_parameters)
        evidence.add_metadata("train_samples", len(train_loader) * batch_size)

        create_training_figures(evidence, losses, val_losses)

        evidence_path = evidence.save()
        print(f"\nEvidence saved to: {evidence_path}")
        results["evidence_path"] = str(evidence_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="WikiText Training Benchmark")
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use-adc", action="store_true", default=True)
    parser.add_argument("--no-adc", dest="use_adc", action="store_false")
    parser.add_argument("--adc-rank", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--evidence", action="store_true", default=True)
    parser.add_argument("--no-evidence", dest="evidence", action="store_false")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    run_training(
        model_size=args.model_size,
        dataset_name=args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        lr=args.lr,
        K=args.K,
        num_workers=args.workers,
        use_adc=args.use_adc,
        adc_rank=args.adc_rank,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        save_evidence=args.evidence,
        device=args.device,
    )


if __name__ == "__main__":
    main()
