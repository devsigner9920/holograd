"""
HoloGrad: Verifiable Distributed Gradient Computation Protocol

A protocol for communication-efficient gradient aggregation with cryptographic
verification guarantees, Byzantine-fault tolerance, and adaptive direction
codebooks for improved convergence in distributed machine learning.
"""

__version__ = "0.1.0"
__author__ = "HoloGrad Team"

from holograd.core.config import HoloGradConfig, ProtocolConfig, ADCConfig, VerificationConfig
from holograd.core.types import Proof, Task, VerificationResult

__all__ = [
    "__version__",
    "HoloGradConfig",
    "ProtocolConfig",
    "ADCConfig",
    "VerificationConfig",
    "Proof",
    "Task",
    "VerificationResult",
]
