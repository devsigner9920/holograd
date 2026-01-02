#!/usr/bin/env python3

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograd.core.config import HoloGradConfig, ProtocolConfig, ADCConfig
from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data, BatchData
from holograd.training.trainer import HoloGradTrainer


def compute_true_gradient_sample(
    model: SimpleGPT2, batch: BatchData, sample_size: int = 500
) -> np.ndarray:
    params = model.get_flat_params()
    eps = 1e-5
    gradient = np.zeros_like(params)

    base_loss = model.compute_loss(batch.input_ids, batch.labels)

    indices = np.random.choice(len(params), min(sample_size, len(params)), replace=False)

    for idx in indices:
        params_plus = params.copy()
        params_plus[idx] += eps
        model.set_flat_params(params_plus)
        loss_plus = model.compute_loss(batch.input_ids, batch.labels)
        gradient[idx] = (loss_plus - base_loss) / eps

    model.set_flat_params(params)
    return gradient, indices


def diagnose_gradient_quality():
    print("=" * 60)
    print("GRADIENT QUALITY DIAGNOSTIC")
    print("=" * 60)

    model = SimpleGPT2(size="tiny")
    D = model.num_parameters
    print(f"\nModel parameters: {D:,}")
    print(f"Vocab size: {model.vocab_size}")
    print(f"Expected optimal loss (random data): {np.log(model.vocab_size):.4f}")

    train_loader, _ = create_synthetic_data(
        vocab_size=model.vocab_size,
        seq_length=32,
        num_train_samples=100,
        batch_size=4,
    )
    batch = next(iter(train_loader))

    loss = model.compute_loss(batch.input_ids, batch.labels)
    print(f"\nActual loss: {loss:.4f}")
    print(f"Gap from optimal: {loss - np.log(model.vocab_size):.6f}")

    print("\n--- True Gradient Analysis ---")
    true_grad, indices = compute_true_gradient_sample(model, batch, sample_size=500)
    sampled_grad = true_grad[indices]

    print(f"Sampled gradient stats:")
    print(f"  Mean: {np.mean(sampled_grad):.6e}")
    print(f"  Std:  {np.std(sampled_grad):.6e}")
    print(f"  Max:  {np.max(np.abs(sampled_grad)):.6e}")
    print(f"  Non-zero (>1e-8): {np.sum(np.abs(sampled_grad) > 1e-8)}/{len(sampled_grad)}")

    print("\n--- HoloGrad Reconstruction ---")

    configs = [
        (
            "Random directions (scale=D)",
            HoloGradConfig(
                protocol=ProtocolConfig(K=64, learning_rate=1e-4),
                adc=ADCConfig(enabled=False),
            ),
        ),
        (
            "ADC (scale=1.0)",
            HoloGradConfig(
                protocol=ProtocolConfig(K=64, learning_rate=1e-4),
                adc=ADCConfig(enabled=True, rank=32),
            ),
        ),
    ]

    for name, config in configs:
        print(f"\n{name}:")
        config.distributed.num_workers = 64

        model_copy = SimpleGPT2(size="tiny")
        model_copy.set_flat_params(model.get_flat_params())

        trainer = HoloGradTrainer(
            config=config,
            model=model_copy,
            train_loader=train_loader,
        )

        trainer._current_batch = batch
        params = model_copy.get_flat_params()
        trainer._coordinator.set_parameters(params)
        trainer._coordinator.set_batch(batch.indices, batch.batch_seed)

        tasks = trainer._coordinator.publish_tasks(0)
        gradient_fn = trainer._create_gradient_fn(batch)
        trainer._worker_pool.set_gradient_fn(gradient_fn)

        proofs = trainer._worker_pool.compute_proofs_parallel(tasks, first_k=config.protocol.K)

        for proof in proofs:
            trainer._coordinator.collect_proof(proof)

        gradient, agg_result = trainer._coordinator.aggregate()

        print(f"  Reconstructed gradient norm: {np.linalg.norm(gradient):.4e}")
        print(f"  Gradient mean: {np.mean(gradient):.6e}")
        print(f"  Gradient std: {np.std(gradient):.6e}")
        print(f"  Gradient max: {np.max(np.abs(gradient)):.6e}")

        if np.linalg.norm(sampled_grad) > 1e-10:
            cosine_sim = np.dot(gradient[indices], sampled_grad) / (
                np.linalg.norm(gradient[indices]) * np.linalg.norm(sampled_grad) + 1e-10
            )
            print(f"  Cosine similarity with true grad: {cosine_sim:.4f}")


def create_pattern_dataset(vocab_size: int, seq_length: int, num_samples: int, batch_size: int):
    from holograd.training.data import BatchData

    class PatternDataLoader:
        def __init__(self, vocab_size, seq_length, num_samples, batch_size):
            self.vocab_size = vocab_size
            self.seq_length = seq_length
            self.num_samples = num_samples
            self.batch_size = batch_size
            self._idx = 0

        def __iter__(self):
            self._idx = 0
            return self

        def __next__(self):
            if self._idx >= self.num_samples:
                raise StopIteration

            batch_inputs = []
            batch_labels = []

            for i in range(self.batch_size):
                start = (self._idx + i) % self.vocab_size
                tokens = np.arange(start, start + self.seq_length + 1) % self.vocab_size
                batch_inputs.append(tokens[:-1])
                batch_labels.append(tokens[1:])

            self._idx += self.batch_size

            return BatchData(
                input_ids=np.stack(batch_inputs),
                labels=np.stack(batch_labels),
                indices=np.arange(self._idx - self.batch_size, self._idx),
                batch_seed=self._idx,
            )

    return PatternDataLoader(vocab_size, seq_length, num_samples, batch_size)


def diagnose_pattern_data():
    print("\n" + "=" * 60)
    print("PATTERN DATA TEST")
    print("=" * 60)
    print("Pattern: next_token = (current_token + 1) % vocab_size")
    print("This is a deterministic pattern the model should learn.")

    vocab_size = 100
    seq_length = 32

    model = SimpleGPT2(size="tiny")
    D = model.num_parameters
    print(f"\nModel parameters: {D:,}")
    print(f"Using vocab_size: {vocab_size}")
    print(f"Optimal loss if perfect: 0.0 (deterministic pattern)")
    print(f"Random guess loss: {np.log(vocab_size):.4f}")

    train_loader = create_pattern_dataset(vocab_size, seq_length, num_samples=500, batch_size=4)

    config = HoloGradConfig(
        protocol=ProtocolConfig(K=128, learning_rate=1e-3, momentum=0.9),
        adc=ADCConfig(enabled=True, rank=64),
    )
    config.distributed.num_workers = 128
    config.training.max_grad_norm = 1.0

    trainer = HoloGradTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
    )

    print("\nTraining with ADC (K=128, lr=1e-3):")
    print("-" * 40)

    losses = []
    for step in range(50):
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            train_loader = create_pattern_dataset(
                vocab_size, seq_length, num_samples=500, batch_size=4
            )
            batch = next(iter(train_loader))

        metrics = trainer.train_step(batch)
        losses.append(metrics.loss)

        if step % 10 == 0:
            print(f"Step {step:3d}: loss = {metrics.loss:.4f}")

    print("-" * 40)
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Improvement:  {losses[0] - losses[-1]:.4f} ({(1 - losses[-1] / losses[0]) * 100:.1f}%)")

    if losses[-1] < losses[0] * 0.95:
        print("\n[SUCCESS] Loss decreased significantly!")
    else:
        print("\n[WARNING] Loss did not decrease much - further investigation needed")


def diagnose_scale_factor():
    print("\n" + "=" * 60)
    print("SCALE FACTOR ANALYSIS")
    print("=" * 60)

    from holograd.protocol.direction import DirectionGenerator, ADCCodebook

    dimensions = [100, 1000, 10000, 48448]

    print("\nScale factors for different dimensions:")
    print("-" * 50)
    print(f"{'Dimension':>10} | {'Random Dir':>15} | {'ADC':>10}")
    print("-" * 50)

    for D in dimensions:
        dir_gen = DirectionGenerator(D)
        adc = ADCCodebook(D, rank=32)

        print(f"{D:>10,} | {dir_gen.scale_factor:>15,.0f} | {adc.get_scale_factor():>10.1f}")

    print("-" * 50)
    print("\nRandom directions: scale_factor = D (amplifies noise by D)")
    print("ADC: scale_factor = 1.0 (no amplification)")


def main():
    np.random.seed(42)

    diagnose_gradient_quality()
    diagnose_scale_factor()
    diagnose_pattern_data()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. For synthetic random data, loss is already optimal (~10.83)
   - True gradient ≈ 0, so HoloGrad reconstructs pure noise
   - This is expected behavior, not a bug!

2. For real/learnable data:
   - Use ADC (scale_factor=1.0) for more stable training
   - OR use random directions with smaller learning rate (lr / D)

3. To verify HoloGrad works:
   - Use pattern data or real text data
   - The model should be able to learn and reduce loss

4. Scale factor math is correct:
   - E[D * <g,v> * v] = g (in expectation)
   - But variance ~ D^2 / K, so K should be proportional to D
   - For D=48k, need K >> 64 for stable estimates
""")


if __name__ == "__main__":
    main()
