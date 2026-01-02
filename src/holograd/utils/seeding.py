"""
Seeding utilities for reproducibility.

Provides deterministic seed generation using SHA-256 hash chain
as specified in the HoloGrad protocol.
"""

import hashlib
import random
from typing import Optional

import numpy as np

# Check for available frameworks
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def set_seed(seed: int) -> None:
    """
    Set random seed for all frameworks for reproducibility.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if JAX_AVAILABLE:
        # JAX uses explicit PRNG keys, but we can set numpy seed
        # which affects some initialization routines
        pass


def get_deterministic_seed(
    global_seed: str | bytes,
    param_hash: bytes,
    batch_hash: bytes,
    step: int,
    worker_id: int,
    codebook_hash: Optional[bytes] = None,
) -> bytes:
    """
    Generate deterministic seed using hash chain.

    Implements: s_{t,j} = SHA256(S_0 || H(sigma_t) || H(theta_t) || t || j)

    Args:
        global_seed: Protocol initialization seed S_0
        param_hash: H(theta_t) - hash of current parameters
        batch_hash: H(B_t) - hash of current minibatch
        step: Current training step t
        worker_id: Worker index j
        codebook_hash: H(U_t) - hash of ADC codebook (optional)

    Returns:
        32-byte deterministic seed
    """
    # Convert global seed to bytes if string
    if isinstance(global_seed, str):
        global_seed = global_seed.encode("utf-8")

    # Build hash input
    # Format: S_0 || H(sigma_t) || H(theta_t) || t || j
    # where sigma_t includes batch and optionally codebook
    hasher = hashlib.sha256()

    # Global seed
    hasher.update(global_seed)

    # Batch commitment (sigma_t component)
    hasher.update(batch_hash)

    # Codebook commitment if ADC enabled
    if codebook_hash is not None:
        hasher.update(codebook_hash)

    # Parameter commitment
    hasher.update(param_hash)

    # Step number (8 bytes, big-endian)
    hasher.update(step.to_bytes(8, byteorder="big"))

    # Worker ID (4 bytes, big-endian)
    hasher.update(worker_id.to_bytes(4, byteorder="big"))

    return hasher.digest()


def hash_parameters(params: np.ndarray | dict) -> bytes:
    """
    Compute deterministic hash of model parameters.

    H(theta_t) = SHA256(Serialize(theta_t))

    Args:
        params: Model parameters as numpy array or dict of arrays

    Returns:
        32-byte SHA-256 hash
    """
    hasher = hashlib.sha256()

    if isinstance(params, dict):
        # Sort keys for determinism
        for key in sorted(params.keys()):
            hasher.update(key.encode("utf-8"))
            arr = np.asarray(params[key])
            hasher.update(arr.tobytes())
    else:
        arr = np.asarray(params)
        hasher.update(arr.tobytes())

    return hasher.digest()


def hash_batch(
    batch_indices: np.ndarray,
    batch_seed: int,
) -> bytes:
    """
    Compute deterministic hash of minibatch.

    H(B_t) includes batch indices and random seed used for sampling.

    Args:
        batch_indices: Indices of samples in the batch
        batch_seed: Random seed used for batch sampling

    Returns:
        32-byte SHA-256 hash
    """
    hasher = hashlib.sha256()
    hasher.update(np.asarray(batch_indices).tobytes())
    hasher.update(batch_seed.to_bytes(8, byteorder="big"))
    return hasher.digest()


def hash_codebook(codebook: np.ndarray, step: int) -> bytes:
    """
    Compute deterministic hash of ADC codebook.

    H(U_t) includes the orthonormal matrix and metadata.

    Args:
        codebook: Orthonormal matrix U_t in R^(D x r)
        step: Current step for metadata

    Returns:
        32-byte SHA-256 hash
    """
    hasher = hashlib.sha256()
    hasher.update(codebook.tobytes())
    hasher.update(step.to_bytes(8, byteorder="big"))
    # Include shape for validation
    for dim in codebook.shape:
        hasher.update(dim.to_bytes(4, byteorder="big"))
    return hasher.digest()


def seed_to_rng_key(seed: bytes) -> np.random.Generator:
    """
    Convert seed bytes to numpy random generator.

    Args:
        seed: 32-byte seed from hash chain

    Returns:
        Seeded numpy random generator
    """
    # Use first 8 bytes as integer seed
    int_seed = int.from_bytes(seed[:8], byteorder="big")
    return np.random.default_rng(int_seed)


if JAX_AVAILABLE:
    import jax.numpy as jnp
    from jax import random as jax_random

    def seed_to_jax_key(seed: bytes) -> jax_random.PRNGKey:
        """
        Convert seed bytes to JAX PRNG key.

        Args:
            seed: 32-byte seed from hash chain

        Returns:
            JAX PRNG key
        """
        # Use first 4 bytes as integer seed (JAX uses 32-bit seeds)
        int_seed = int.from_bytes(seed[:4], byteorder="big")
        return jax_random.PRNGKey(int_seed)
