from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def seed_to_rng(seed: bytes) -> np.random.Generator:
    int_seed = int.from_bytes(seed[:8], byteorder="big")
    return np.random.default_rng(int_seed)


@dataclass
class DirectionResult:
    direction: NDArray[np.float32]
    z_projection: Optional[NDArray[np.float32]] = None


class DirectionGenerator:
    def __init__(self, dimension: int, dtype: np.dtype = np.float32):
        self.dimension = dimension
        self.dtype = dtype
        self._sigma_squared = 1.0 / dimension

    @property
    def sigma_squared(self) -> float:
        return self._sigma_squared

    @property
    def scale_factor(self) -> float:
        return 1.0 / self._sigma_squared

    def generate(self, seed: bytes) -> DirectionResult:
        rng = seed_to_rng(seed)
        z = rng.standard_normal(self.dimension).astype(self.dtype)
        norm = np.linalg.norm(z)
        direction = z / norm
        return DirectionResult(direction=direction)

    def generate_chunked(self, seed: bytes, chunk_size: int = 1_000_000) -> DirectionResult:
        rng = seed_to_rng(seed)
        chunks = []
        norm_squared = 0.0

        remaining = self.dimension
        while remaining > 0:
            size = min(chunk_size, remaining)
            chunk = rng.standard_normal(size).astype(self.dtype)
            norm_squared += np.sum(chunk**2)
            chunks.append(chunk)
            remaining -= size

        norm = np.sqrt(norm_squared)
        direction = np.concatenate(chunks) / norm
        return DirectionResult(direction=direction)


class ADCCodebook:
    def __init__(
        self,
        dimension: int,
        rank: int = 32,
        oja_alpha: float = 1e-3,
        qr_period: int = 100,
        dtype: np.dtype = np.float16,
        warmup_samples: int = 0,
        alpha_decay: float = 1.0,
        alpha_min: float = 1e-4,
        use_power_iteration: bool = False,
        power_iteration_steps: int = 3,
        device: str = "cpu",
    ):
        self.dimension = dimension
        self.rank = rank
        self.oja_alpha = oja_alpha
        self.oja_alpha_initial = oja_alpha
        self.qr_period = qr_period
        self.dtype = dtype
        self.device = device

        self.warmup_samples = warmup_samples
        self._warmup_buffer: list[NDArray[np.float32]] = []
        self._is_warmed_up = warmup_samples == 0

        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self._current_alpha = oja_alpha

        self.use_power_iteration = use_power_iteration
        self.power_iteration_steps = power_iteration_steps

        self._U: NDArray = self._initialize_codebook()
        self._step = 0

        self._energy_history: list[float] = []
        self._energy_ema = 0.0
        self._energy_ema_alpha = 0.1

    @property
    def codebook(self) -> NDArray[np.float32]:
        return self._U

    @property
    def step(self) -> int:
        return self._step

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    @property
    def current_alpha(self) -> float:
        return self._current_alpha

    @property
    def energy_ema(self) -> float:
        return self._energy_ema

    def _initialize_codebook(self, seed: Optional[int] = None) -> NDArray:
        import time

        print(
            f"[ADC] Initializing codebook: dim={self.dimension:,}, rank={self.rank}, device={self.device}",
            flush=True,
        )
        start_time = time.time()

        if TORCH_AVAILABLE and self.device != "cpu":
            U = self._initialize_codebook_torch(seed)
        else:
            U = self._initialize_codebook_numpy(seed)

        total_time = time.time() - start_time
        print(f"[ADC] Codebook initialized in {total_time:.1f}s", flush=True)
        return U

    def _initialize_codebook_torch(self, seed: Optional[int] = None) -> NDArray:
        import time

        device = self.device if self.device else "cuda"
        if seed is not None:
            torch.manual_seed(seed)

        print(f"[ADC] Using torch on {device} (Gram-Schmidt)", flush=True)

        U = np.zeros((self.dimension, self.rank), dtype=np.float32)

        for col in range(self.rank):
            col_start = time.time()

            v = torch.randn(self.dimension, device=device, dtype=torch.float32)

            if col > 0:
                U_prev = torch.from_numpy(U[:, :col]).to(device)
                projections = U_prev.T @ v
                v = v - U_prev @ projections
                del U_prev

            norm = torch.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm

            U[:, col] = v.cpu().numpy()
            del v
            torch.cuda.empty_cache()

            col_time = time.time() - col_start
            if (col + 1) % 4 == 0 or col == 0:
                print(f"[ADC] Column {col + 1}/{self.rank} done ({col_time:.2f}s)", flush=True)

        return U.astype(self.dtype)

    def _initialize_codebook_numpy(self, seed: Optional[int] = None) -> NDArray:
        import time

        rng = np.random.default_rng(seed)
        U = np.zeros((self.dimension, self.rank), dtype=self.dtype)

        chunk_size = 1_000_000
        for col in range(self.rank):
            col_start = time.time()
            column = np.empty(self.dimension, dtype=self.dtype)
            remaining = self.dimension
            offset = 0
            while remaining > 0:
                size = min(chunk_size, remaining)
                chunk = rng.standard_normal(size).astype(self.dtype)
                column[offset : offset + size] = chunk
                offset += size
                remaining -= size

            for prev_col in range(col):
                dot_product = np.dot(U[:, prev_col].astype(np.float32), column.astype(np.float32))
                column = column - dot_product * U[:, prev_col]

            norm = np.sqrt(np.sum(column.astype(np.float32) ** 2))
            if norm > 1e-10:
                column = column / norm
            U[:, col] = column

            col_time = time.time() - col_start
            if (col + 1) % 4 == 0 or col == 0:
                elapsed = time.time() - col_start
                print(
                    f"[ADC] Column {col + 1}/{self.rank} done ({col_time:.1f}s)",
                    flush=True,
                )

        return U

    def _initialize_from_gradients(
        self, gradients: list[NDArray[np.float32]]
    ) -> NDArray[np.float32]:
        if len(gradients) == 0:
            return self._initialize_codebook()

        G = np.stack(gradients, axis=0)

        try:
            if G.shape[0] >= self.rank:
                _, _, Vh = np.linalg.svd(G, full_matrices=False)
                U_init = Vh[: self.rank, :].T
            else:
                _, _, Vh = np.linalg.svd(G, full_matrices=False)
                num_from_svd = min(G.shape[0], self.rank)
                U_init = np.zeros((self.dimension, self.rank), dtype=self.dtype)
                U_init[:, :num_from_svd] = Vh[:num_from_svd, :].T

                if num_from_svd < self.rank:
                    rng = np.random.default_rng()
                    remaining = rng.standard_normal((self.dimension, self.rank - num_from_svd))
                    for i in range(self.rank - num_from_svd):
                        v = remaining[:, i]
                        for j in range(num_from_svd + i):
                            v = v - np.dot(U_init[:, j], v) * U_init[:, j]
                        v = v / np.linalg.norm(v)
                        U_init[:, num_from_svd + i] = v

            Q, _ = np.linalg.qr(U_init.astype(np.float32))
            return Q.astype(self.dtype)

        except np.linalg.LinAlgError:
            return self._initialize_codebook()

    def generate_direction(self, seed: bytes) -> DirectionResult:
        rng = seed_to_rng(seed)
        z = rng.standard_normal(self.rank).astype(self.dtype)
        direction = self._U @ z
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        return DirectionResult(direction=direction, z_projection=z)

    def reconstruct_direction(self, z: NDArray[np.float32]) -> NDArray[np.float32]:
        direction = self._U @ z
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        return direction

    def _update_alpha(self) -> None:
        if self.alpha_decay < 1.0:
            self._current_alpha = max(
                self.alpha_min, self.oja_alpha_initial * (self.alpha_decay**self._step)
            )

    def _power_iteration_update(self, gradient: NDArray[np.float32]) -> None:
        for _ in range(self.power_iteration_steps):
            projection = self._U.T @ gradient
            reconstructed = self._U @ projection
            residual = gradient - reconstructed

            if np.linalg.norm(residual) > 1e-10:
                residual_normalized = residual / np.linalg.norm(residual)
                projections = np.abs(self._U.T @ gradient)
                min_idx = np.argmin(projections)

                self._U[:, min_idx] = (1 - self._current_alpha) * self._U[
                    :, min_idx
                ] + self._current_alpha * residual_normalized

        self._U, _ = np.linalg.qr(self._U.astype(np.float32))
        self._U = self._U.astype(self.dtype)

    def _oja_update(self, gradient: NDArray[np.float32]) -> None:
        projection = self._U.T @ gradient
        outer = np.outer(gradient, projection)
        self._U = self._U + self._current_alpha * outer

        if (self._step + 1) % self.qr_period == 0:
            self._U, _ = np.linalg.qr(self._U.astype(np.float32))
            self._U = self._U.astype(self.dtype)
        else:
            norms = np.linalg.norm(self._U, axis=0, keepdims=True)
            self._U = self._U / norms

    def update(self, gradient: NDArray[np.float32]) -> None:
        energy = self.captured_energy_ratio(gradient)
        self._energy_history.append(energy)
        self._energy_ema = (
            self._energy_ema_alpha * energy + (1 - self._energy_ema_alpha) * self._energy_ema
        )

        if not self._is_warmed_up:
            self._warmup_buffer.append(gradient.copy())

            if len(self._warmup_buffer) >= self.warmup_samples:
                self._U = self._initialize_from_gradients(self._warmup_buffer)
                self._warmup_buffer = []
                self._is_warmed_up = True
            return

        self._update_alpha()

        if self.use_power_iteration and self._step < self.qr_period:
            self._power_iteration_update(gradient)
        else:
            self._oja_update(gradient)

        self._step += 1

    def captured_energy_ratio(self, gradient: NDArray[np.float32]) -> float:
        projection = self._U.T @ gradient
        projected_norm_sq = np.sum(projection**2)
        gradient_norm_sq = np.sum(gradient**2)

        if gradient_norm_sq < 1e-12:
            return 0.0

        return float(projected_norm_sq / gradient_norm_sq)

    def get_scale_factor(self) -> float:
        # ADC directions live in r-dimensional subspace
        # Scale factor is r (rank), not D (dimension)
        return float(self.rank)

    def reset(self, seed: Optional[int] = None) -> None:
        self._U = self._initialize_codebook(seed)
        self._step = 0
        self._warmup_buffer = []
        self._is_warmed_up = self.warmup_samples == 0
        self._current_alpha = self.oja_alpha_initial
        self._energy_history = []
        self._energy_ema = 0.0
