import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class CommitmentChain:
    global_seed: bytes

    @classmethod
    def from_string(cls, seed_str: str) -> "CommitmentChain":
        return cls(global_seed=seed_str.encode("utf-8"))

    def get_worker_seed(
        self,
        param_commitment: bytes,
        batch_commitment: bytes,
        step: int,
        worker_id: int,
        codebook_commitment: Optional[bytes] = None,
    ) -> bytes:
        hasher = hashlib.sha256()
        hasher.update(self.global_seed)
        hasher.update(batch_commitment)
        if codebook_commitment is not None:
            hasher.update(codebook_commitment)
        hasher.update(param_commitment)
        hasher.update(step.to_bytes(8, byteorder="big"))
        hasher.update(worker_id.to_bytes(4, byteorder="big"))
        return hasher.digest()

    @staticmethod
    def hash_parameters(params: NDArray[np.float32] | dict) -> bytes:
        hasher = hashlib.sha256()
        if isinstance(params, dict):
            for key in sorted(params.keys()):
                hasher.update(key.encode("utf-8"))
                hasher.update(np.asarray(params[key]).tobytes())
        else:
            hasher.update(np.asarray(params).tobytes())
        return hasher.digest()

    @staticmethod
    def hash_batch(batch_indices: NDArray[np.int64], batch_seed: int) -> bytes:
        hasher = hashlib.sha256()
        hasher.update(np.asarray(batch_indices).tobytes())
        hasher.update(batch_seed.to_bytes(8, byteorder="big"))
        return hasher.digest()

    @staticmethod
    def hash_codebook(codebook: NDArray[np.float32], step: int) -> bytes:
        hasher = hashlib.sha256()
        hasher.update(step.to_bytes(8, byteorder="big"))
        for dim in codebook.shape:
            hasher.update(dim.to_bytes(4, byteorder="big"))
        sample_size = min(10000, codebook.size)
        indices = np.linspace(0, codebook.size - 1, sample_size, dtype=np.int64)
        hasher.update(codebook.flat[indices].tobytes())
        return hasher.digest()
