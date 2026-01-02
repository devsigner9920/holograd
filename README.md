# HoloGrad

Verifiable distributed gradient computation protocol for decentralized ML training.

## Overview

HoloGrad enables communication-efficient gradient aggregation with:

- **Scalar proofs** instead of full gradient tensors (D-dimensional -> 1 scalar per direction)
- **SHA-256 hash commitment chain** for deterministic reproducibility
- **Adaptive Direction Codebook (ADC)** with Oja-QR for improved convergence
- **Byzantine-fault tolerant** trimmed-mean aggregation
- **PoGP verification** with sampling-based proof checking

## Installation

```bash
# Clone and setup
git clone <repository-url>
cd holograd

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (for testing)
pip install -r requirements-dev.txt
```

## Quick Start

```python
from holograd.core.config import HoloGradConfig
from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data
from holograd.training.trainer import HoloGradTrainer

# Create config
config = HoloGradConfig()

# Create model (tiny for demo, use "small" for GPT-2 124M)
model = SimpleGPT2(size="tiny")

# Create data loaders
train_loader, val_loader = create_synthetic_data(
    vocab_size=model.vocab_size,
    seq_length=256,
    num_train_samples=1000,
    batch_size=8,
)

# Train
trainer = HoloGradTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
)
result = trainer.train(num_steps=100)
print(f"Final loss: {result['final_train_loss']:.4f}")
```

## Architecture

```
Coordinator                    Worker j                    Verifier
    |                             |                            |
    |-- publish_tasks(t) -------->|                            |
    |   (seeds, commitments)      |                            |
    |                             |                            |
    |                        v = Dir(s)                        |
    |                        a = <g, v>                        |
    |                             |                            |
    |<---- submit_proof(s,a) -----|                            |
    |                             |                            |
    |-- aggregate (trimmed-mean)                               |
    |-- g_hat = reconstruct                                    |
    |-- theta = theta - lr*g_hat                               |
    |                             |                            |
    |-- sample_verify(p=0.05) ---------------------------->    |
    |                                                     verify|
    |<--------------------------------- accept/slash ----------|
```

## Core Components

### Protocol Layer
- `CommitmentChain`: SHA-256 hash chain for deterministic seed derivation
- `DirectionGenerator`: Unit-norm isotropic direction generation
- `ADCCodebook`: Adaptive low-rank subspace with Oja-QR updates
- `RobustAggregator`: Trimmed-mean aggregation for Byzantine tolerance

### Distributed Layer
- `Worker`: Computes scalar projections a = <g, v>
- `Coordinator`: Publishes tasks, collects proofs, updates parameters
- `WorkerPool`: Single-machine multiprocessing simulation

### Verification Layer
- `Verifier`: Sampling-based PoGP verification with configurable p_verify

### Training Layer
- `SimpleGPT2`: GPT-2 model (tiny/small/medium/large)
- `HoloGradTrainer`: Training loop with HoloGrad protocol

## Configuration

```python
from holograd.core.config import (
    HoloGradConfig,
    ProtocolConfig,
    ADCConfig,
    VerificationConfig,
)

config = HoloGradConfig(
    protocol=ProtocolConfig(
        K=64,                          # Directions per step
        learning_rate=3e-4,            # SGD learning rate
        global_seed="my_experiment",   # Reproducibility seed
    ),
    adc=ADCConfig(
        enabled=True,                  # Use adaptive codebook
        rank=32,                       # Subspace dimension
        oja_alpha=1e-3,                # Oja step size
        qr_period=100,                 # QR decomposition frequency
    ),
    verification=VerificationConfig(
        p_verify=0.05,                 # Verification sampling rate
        epsilon=1e-4,                  # Tolerance threshold
    ),
)
```

## Key Formulas

**Scale-corrected gradient synthesis:**
```
g_hat = (1/K) * sum_j (1/sigma^2) * a_j * v_j

where sigma^2 = 1/D (full-space) or sigma^2 = 1 (ADC)
```

**Seed derivation:**
```
s_{t,j} = SHA256(S_0 || H(theta_t) || H(B_t) || t || j)
```

**ADC Oja update:**
```
U_tilde = U_t + alpha * g_hat * (g_hat^T * U_t)
U_{t+1} = QR(U_tilde)  // every T_qr steps
```

## Testing

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run unit tests only
PYTHONPATH=src pytest tests/unit/ -v

# Run integration tests
PYTHONPATH=src pytest tests/integration/ -v

# Skip slow tests
PYTHONPATH=src pytest tests/ -v -m "not slow"
```

## Project Structure

```
holograd/
├── src/holograd/
│   ├── core/           # Config, types, seeding
│   ├── protocol/       # Commitment, direction, aggregation
│   ├── distributed/    # Worker, coordinator, simulation
│   ├── gradient/       # Backprop, JVP implementations
│   ├── verification/   # PoGP verifier
│   ├── training/       # Model, data, trainer
│   ├── experiments/    # Experiment runner, ablations
│   └── utils/          # Logging, seeding utilities
├── tests/
│   ├── unit/           # Component tests
│   └── integration/    # End-to-end tests
├── configs/            # YAML configurations
└── requirements.txt
```

## Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| K | 64 | Directions per step |
| r | 32 | ADC subspace rank |
| p_verify | 0.05 | Verification probability |
| epsilon | 1e-4 | Verification tolerance |
| tau | 0.1 | Trim rate for aggregation |
| lr | 3e-4 | Learning rate |

## Limitations

- Single-machine simulation only (no multi-node deployment)
- Slashing is simulated (no actual blockchain integration)
- GPT-2 small max (no larger models)
- Numerical tolerance ~1-2% for finite difference gradients

## References

- HoloGrad paper: Verifiable distributed gradient computation
- Oja's algorithm for streaming PCA
- Byzantine-tolerant distributed learning
