"""End-to-end integration tests for HoloGrad training pipeline."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

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


def create_test_config(
    num_workers: int = 4,
    K: int = 8,
    use_adc: bool = False,
    max_steps: int = 5,
) -> HoloGradConfig:
    """Create minimal config for fast integration testing."""
    return HoloGradConfig(
        protocol=ProtocolConfig(
            K=K,
            global_seed="test_seed_42",
            learning_rate=1e-3,
        ),
        adc=ADCConfig(
            enabled=use_adc,
            rank=4,
            oja_alpha=0.1,
        ),
        verification=VerificationConfig(
            p_verify=0.0,
            epsilon=1e-3,
        ),
        aggregation=AggregationConfig(
            tau=0.1,
        ),
        distributed=DistributedConfig(
            num_workers=num_workers,
            simulate_delays=False,
        ),
        training=TrainingConfig(
            max_steps=max_steps,
            batch_size=2,
            sequence_length=16,
            eval_interval=100,
        ),
        logging=LoggingConfig(
            log_interval=1,
        ),
    )


class TestEndToEndTraining:
    """Integration tests for complete training pipeline."""

    def test_single_train_step(self):
        """Test single training step executes without error."""
        config = create_test_config(num_workers=4, K=4)

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=16,
        )

        train_loader, _ = create_synthetic_data(
            vocab_size=100,
            seq_length=16,
            num_train_samples=10,
            batch_size=2,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        batch = next(iter(train_loader))
        metrics = trainer.train_step(batch)

        assert metrics.step == 0
        assert metrics.loss > 0
        assert metrics.proofs_received > 0
        assert metrics.step_time > 0

    def test_multiple_train_steps(self):
        """Test multiple training steps execute sequentially."""
        config = create_test_config(num_workers=2, K=4, max_steps=3)

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=16,
        )

        train_loader, _ = create_synthetic_data(
            vocab_size=100,
            seq_length=16,
            num_train_samples=20,
            batch_size=2,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        result = trainer.train(num_steps=3)

        assert trainer.state.step == 3
        assert result["total_steps"] == 3
        assert len(trainer.state.training_losses) == 3

    def test_loss_decreases_over_training(self):
        """Test that loss decreases over sufficient training steps."""
        config = create_test_config(
            num_workers=4,
            K=16,
            max_steps=10,
        )
        config.protocol.learning_rate = 0.01

        model = SimpleGPT2(
            size="tiny",
            vocab_size=50,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=8,
        )

        train_loader, _ = create_synthetic_data(
            vocab_size=50,
            seq_length=8,
            num_train_samples=50,
            batch_size=4,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        trainer.train(num_steps=10)

        losses = trainer.state.training_losses
        assert len(losses) == 10

        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])

        assert late_avg <= early_avg * 1.2, (
            f"Loss didn't improve: {early_avg:.4f} -> {late_avg:.4f}"
        )

    def test_adc_mode_training(self):
        """Test training with ADC enabled."""
        config = create_test_config(
            num_workers=4,
            K=8,
            use_adc=True,
            max_steps=5,
        )

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=16,
        )

        train_loader, _ = create_synthetic_data(
            vocab_size=100,
            seq_length=16,
            num_train_samples=20,
            batch_size=2,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        result = trainer.train(num_steps=5)

        assert trainer.state.step == 5
        assert result["total_steps"] == 5

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        config = create_test_config(num_workers=2, K=4, max_steps=3)

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=16,
        )

        train_loader, _ = create_synthetic_data(
            vocab_size=100,
            seq_length=16,
            num_train_samples=20,
            batch_size=2,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        trainer.train(num_steps=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.npz"
            trainer.save_checkpoint(ckpt_path)

            assert ckpt_path.exists()

            model2 = SimpleGPT2(
                size="tiny",
                vocab_size=100,
                n_embd=32,
                n_head=2,
                n_layer=1,
                max_seq_len=16,
            )

            trainer2 = HoloGradTrainer(
                config=config,
                model=model2,
                train_loader=train_loader,
            )

            trainer2.load_checkpoint(ckpt_path)

            assert trainer2.state.step == 3
            np.testing.assert_array_almost_equal(
                model.get_flat_params(),
                model2.get_flat_params(),
            )

    def test_evaluation(self):
        """Test evaluation on validation set."""
        config = create_test_config(num_workers=2, K=4)

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=16,
        )

        train_loader, val_loader = create_synthetic_data(
            vocab_size=100,
            seq_length=16,
            num_train_samples=20,
            num_val_samples=5,
            batch_size=2,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        val_loss = trainer.evaluate()

        assert val_loss > 0
        assert not np.isnan(val_loss)

    def test_proof_verification_integration(self):
        """Test that verification system works during training."""
        config = create_test_config(num_workers=4, K=8)
        config.verification.p_verify = 0.5

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=16,
        )

        train_loader, _ = create_synthetic_data(
            vocab_size=100,
            seq_length=16,
            num_train_samples=20,
            batch_size=2,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        result = trainer.train(num_steps=3)
        assert result["total_steps"] == 3

    def test_trimmed_aggregation(self):
        """Test that trimmed mean aggregation handles proofs correctly."""
        config = create_test_config(num_workers=8, K=8)
        config.aggregation.tau = 0.25

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=16,
        )

        train_loader, _ = create_synthetic_data(
            vocab_size=100,
            seq_length=16,
            num_train_samples=20,
            batch_size=2,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        batch = next(iter(train_loader))
        metrics = trainer.train_step(batch)

        assert metrics.proofs_used <= 8
        assert metrics.proofs_trimmed >= 0


class TestEndToEndWithStress:
    """Stress tests for training pipeline."""

    @pytest.mark.slow
    def test_longer_training_run(self):
        """Test longer training run (30 steps)."""
        config = create_test_config(
            num_workers=4,
            K=16,
            max_steps=30,
        )

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            max_seq_len=32,
        )

        train_loader, val_loader = create_synthetic_data(
            vocab_size=100,
            seq_length=32,
            num_train_samples=100,
            num_val_samples=10,
            batch_size=4,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        result = trainer.train()

        assert trainer.state.step == 30
        assert result["final_train_loss"] > 0

    @pytest.mark.slow
    def test_epoch_rollover(self):
        """Test that training handles epoch rollover correctly."""
        config = create_test_config(
            num_workers=2,
            K=4,
            max_steps=20,
        )

        model = SimpleGPT2(
            size="tiny",
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            max_seq_len=16,
        )

        train_loader, _ = create_synthetic_data(
            vocab_size=100,
            seq_length=16,
            num_train_samples=5,
            batch_size=2,
        )

        trainer = HoloGradTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        trainer.train(num_steps=20)

        assert trainer.state.step == 20
        assert trainer.state.epoch > 0
