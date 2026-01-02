#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

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
from holograd.training.data import create_synthetic_data
from holograd.training.trainer import HoloGradTrainer
from holograd.utils.logging import MetricsLogger, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 with HoloGrad protocol")

    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large"],
        help="GPT-2 model size",
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--K", type=int, default=16, help="Number of directions per step")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of simulated workers")
    parser.add_argument("--use-adc", action="store_true", help="Enable Adaptive Direction Codebook")
    parser.add_argument("--adc-rank", type=int, default=32, help="ADC subspace rank")
    parser.add_argument(
        "--p-verify", type=float, default=0.0, help="Verification probability (0 to disable)"
    )
    parser.add_argument("--tau", type=float, default=0.1, help="Trimmed mean trim rate")
    parser.add_argument(
        "--seed", type=str, default="holograd_experiment", help="Global seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval (steps)")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval (steps)")
    parser.add_argument(
        "--save-checkpoint", action="store_true", help="Save checkpoint at end of training"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("HoloGrad GPT-2 Training")
    logger.info("=" * 60)

    config = HoloGradConfig(
        protocol=ProtocolConfig(
            K=args.K,
            global_seed=args.seed,
            learning_rate=args.lr,
        ),
        adc=ADCConfig(
            enabled=args.use_adc,
            rank=args.adc_rank,
        ),
        verification=VerificationConfig(
            p_verify=args.p_verify,
            epsilon=1e-4,
        ),
        aggregation=AggregationConfig(
            tau=args.tau,
        ),
        distributed=DistributedConfig(
            num_workers=args.num_workers,
            simulate_delays=False,
        ),
        training=TrainingConfig(
            max_steps=args.steps,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            eval_interval=args.eval_interval,
        ),
        logging=LoggingConfig(
            log_interval=args.log_interval,
            output_dir=args.output_dir,
        ),
    )

    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"K (directions): {args.K}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"ADC enabled: {args.use_adc}")
    if args.use_adc:
        logger.info(f"ADC rank: {args.adc_rank}")
    logger.info(f"Verification rate: {args.p_verify}")
    logger.info(f"Trim rate (tau): {args.tau}")
    logger.info("-" * 60)

    logger.info("Creating model...")
    model = SimpleGPT2(size=args.model_size, max_seq_len=args.seq_length)
    logger.info(f"Model parameters: {model.num_parameters:,}")

    logger.info("Creating synthetic data...")
    train_loader, val_loader = create_synthetic_data(
        vocab_size=model.vocab_size,
        seq_length=args.seq_length,
        num_train_samples=max(1000, args.steps * args.batch_size * 2),
        num_val_samples=100,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_logger = MetricsLogger(
        output_dir=args.output_dir,
        experiment_name=f"gpt2_{args.model_size}_{args.seed}",
    )

    logger.info("Initializing trainer...")
    trainer = HoloGradTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=metrics_logger,
    )

    logger.info("-" * 60)
    logger.info("Starting training...")
    logger.info("-" * 60)

    result = trainer.train(num_steps=args.steps)

    logger.info("-" * 60)
    logger.info("Training complete!")
    logger.info("-" * 60)
    logger.info(f"Total steps: {result['total_steps']}")
    logger.info(f"Total tokens: {result['total_tokens']:,}")
    logger.info(f"Final train loss: {result['final_train_loss']:.4f}")
    logger.info(f"Best val loss: {result['best_val_loss']:.4f}")

    losses = trainer.state.training_losses
    if len(losses) >= 6:
        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        improvement = (early_avg - late_avg) / early_avg * 100
        logger.info(f"Loss improvement: {improvement:.1f}%")

    if args.save_checkpoint:
        ckpt_path = output_path / "checkpoint.npz"
        trainer.save_checkpoint(ckpt_path)
        logger.info(f"Checkpoint saved to: {ckpt_path}")

    metrics_logger.close()
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
