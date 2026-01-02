"""
Configuration dataclasses for HoloGrad protocol.

All hyperparameters are defined here with paper-recommended defaults:
- K=64: Number of direction proofs per step
- r=32: ADC rank (low-rank subspace dimension)
- p_verify=0.05: Probability of verifying a proof
- epsilon=1e-4: Verification tolerance
- tau=0.1: Trim rate for robust aggregation
- alpha=1e-3: Oja step size for codebook update
- T_qr=100: Steps between QR decomposition
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import yaml


@dataclass
class ProtocolConfig:
    """Core protocol hyperparameters."""

    K: int = 64
    learning_rate: float = 3e-4
    global_seed: str = "holograd_v1_seed_2025"
    gradient_method: Literal["backprop", "jvp"] = "jvp"
    momentum: float = 0.9

    # Direction mode: "random" (original), "adc" (subspace), "momentum" (momentum-centric)
    direction_mode: Literal["random", "adc", "momentum"] = "random"


@dataclass
class MomentumCentricConfig:
    """Momentum-Centric HoloGrad configuration.

    This variant projects gradients onto the coordinator's momentum direction,
    reducing communication to a single scalar per worker.

    Paper reference: Section 8 (Momentum-Centric HoloGrad)
    """

    # Momentum coefficient (beta in the paper)
    beta: float = 0.9

    # Gradient norm EMA coefficient for magnitude estimation
    grad_norm_ema_alpha: float = 0.1

    # Number of warmup steps using random directions before switching to momentum
    warmup_steps: int = 10

    # Whether to use the gradient norm estimate for scaling
    use_grad_norm_scaling: bool = True

    # Initial gradient norm estimate
    initial_grad_norm: float = 1.0


@dataclass
class ADCConfig:
    rank: int = 32
    oja_alpha: float = 1e-3
    qr_period: int = 100
    enabled: bool = True
    normalize_columns: bool = True

    warmup_samples: int = 0
    alpha_decay: float = 1.0
    alpha_min: float = 1e-4
    use_power_iteration: bool = False
    power_iteration_steps: int = 3


@dataclass
class VerificationConfig:
    """PoGP verification system configuration."""

    # Probability of verifying each proof
    p_verify: float = 0.05

    # Acceptance tolerance for |a - a*|
    epsilon: float = 1e-4

    # Slashing penalty (simulation only)
    slash_penalty: float = 1.0

    # Track false positive/negative rates
    track_rates: bool = True


@dataclass
class AggregationConfig:
    """Robust aggregation configuration."""

    # Trim rate for trimmed-mean
    tau: float = 0.1

    # Aggregation method
    method: Literal["trimmed_mean", "median", "mean"] = "trimmed_mean"

    # Track rejection counts
    track_rejections: bool = True


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    # Model configuration
    model_name: str = "gpt2"
    model_size: Literal["small", "medium", "large"] = "small"

    # Dataset configuration
    dataset_name: Literal["wikitext-103", "openwebtext"] = "wikitext-103"
    sequence_length: int = 256
    batch_size: int = 8

    # Training parameters
    max_steps: int = 10000
    warmup_steps: int = 100
    eval_interval: int = 500
    checkpoint_interval: int = 1000

    # Optimizer
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class DistributedConfig:
    """Distributed worker simulation configuration."""

    # Number of simulated workers
    num_workers: int = 8

    # Worker collection mode
    collection_mode: Literal["first_k", "synchronous", "time_windowed"] = "first_k"

    # Time window for time_windowed mode (seconds)
    time_window: float = 10.0

    # Delay simulation
    simulate_delays: bool = True
    delay_distribution: Literal["normal", "exponential", "uniform"] = "normal"
    delay_mean: float = 0.1  # seconds
    delay_std: float = 0.05  # seconds

    # Byzantine worker simulation
    byzantine_fraction: float = 0.0
    byzantine_strategy: Literal["random", "sign_flip", "scale"] = "random"


@dataclass
class LoggingConfig:
    """Metrics logging configuration."""

    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Experiment name (auto-generated if None)
    experiment_name: Optional[str] = None

    # Output formats
    log_to_csv: bool = True
    log_to_json: bool = True
    log_to_tensorboard: bool = True

    # Logging frequency
    log_interval: int = 10

    # What to log
    log_training_metrics: bool = True
    log_communication_metrics: bool = True
    log_verification_metrics: bool = True
    log_adc_metrics: bool = True
    log_system_metrics: bool = True


@dataclass
class HoloGradConfig:
    """
    Master configuration for HoloGrad protocol.

    Aggregates all sub-configurations and provides methods for
    loading from files and CLI overrides.
    """

    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    adc: ADCConfig = field(default_factory=ADCConfig)
    momentum_centric: MomentumCentricConfig = field(default_factory=MomentumCentricConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Random seed for reproducibility
    seed: int = 42

    # Device configuration
    device: Literal["cpu", "cuda", "mps", "tpu"] = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HoloGradConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "HoloGradConfig":
        """Create config from dictionary."""
        config = cls()

        if "protocol" in data:
            config.protocol = ProtocolConfig(**data["protocol"])
        if "adc" in data:
            config.adc = ADCConfig(**data["adc"])
        if "momentum_centric" in data:
            config.momentum_centric = MomentumCentricConfig(**data["momentum_centric"])
        if "verification" in data:
            config.verification = VerificationConfig(**data["verification"])
        if "aggregation" in data:
            config.aggregation = AggregationConfig(**data["aggregation"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "distributed" in data:
            config.distributed = DistributedConfig(**data["distributed"])
        if "logging" in data:
            logging_data = data["logging"].copy()
            if "output_dir" in logging_data:
                logging_data["output_dir"] = Path(logging_data["output_dir"])
            config.logging = LoggingConfig(**logging_data)

        if "seed" in data:
            config.seed = data["seed"]
        if "device" in data:
            config.device = data["device"]

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict

        def convert(obj: object) -> object:
            if isinstance(obj, Path):
                return str(obj)
            return obj

        data = {}
        for field_name in [
            "protocol",
            "adc",
            "momentum_centric",
            "verification",
            "aggregation",
            "training",
            "distributed",
            "logging",
        ]:
            field_value = getattr(self, field_name)
            field_dict = asdict(field_value)
            # Convert Path objects to strings
            for k, v in field_dict.items():
                field_dict[k] = convert(v)
            data[field_name] = field_dict

        data["seed"] = self.seed
        data["device"] = self.device

        return data

    def override(self, **kwargs: object) -> "HoloGradConfig":
        """
        Create a new config with overridden values.

        Supports dotted notation: config.override(protocol.K=128)
        """
        import copy

        new_config = copy.deepcopy(self)

        for key, value in kwargs.items():
            parts = key.split(".")
            obj = new_config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        return new_config


def get_default_config() -> HoloGradConfig:
    """Get default HoloGrad configuration with paper-recommended values."""
    return HoloGradConfig()
