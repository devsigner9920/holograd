from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GradientResult:
    scalar: float
    gradient: Optional[NDArray[np.float32]] = None
    compute_time: float = 0.0
    memory_bytes: int = 0


class GradientComputer(ABC):
    @abstractmethod
    def compute_projection(
        self,
        params: NDArray[np.float32],
        direction: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
    ) -> GradientResult:
        pass


class BackpropGradient(GradientComputer):
    def __init__(self, framework: str = "numpy"):
        self.framework = framework

    def compute_full_gradient(
        self,
        params: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
    ) -> NDArray[np.float32]:
        if self.framework == "jax":
            return self._compute_jax(params, loss_fn, batch)
        elif self.framework == "torch":
            return self._compute_torch(params, loss_fn, batch)
        else:
            return self._compute_numerical(params, loss_fn, batch)

    def compute_projection(
        self,
        params: NDArray[np.float32],
        direction: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
    ) -> GradientResult:
        import time

        start = time.perf_counter()

        gradient = self.compute_full_gradient(params, loss_fn, batch)
        scalar = float(np.dot(gradient.flatten(), direction.flatten()))

        elapsed = time.perf_counter() - start
        memory = gradient.nbytes

        return GradientResult(
            scalar=scalar,
            gradient=gradient,
            compute_time=elapsed,
            memory_bytes=memory,
        )

    def _compute_numerical(
        self,
        params: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
        eps: float = 1e-5,
    ) -> NDArray[np.float32]:
        gradient = np.zeros_like(params, dtype=np.float32)
        params_flat = params.flatten()

        for i in range(len(params_flat)):
            params_plus = params_flat.copy()
            params_minus = params_flat.copy()
            params_plus[i] += eps
            params_minus[i] -= eps

            loss_plus = loss_fn(params_plus.reshape(params.shape), batch)
            loss_minus = loss_fn(params_minus.reshape(params.shape), batch)

            gradient.flat[i] = (loss_plus - loss_minus) / (2 * eps)

        return gradient

    def _compute_jax(
        self,
        params: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
    ) -> NDArray[np.float32]:
        try:
            import jax
            import jax.numpy as jnp

            grad_fn = jax.grad(loss_fn)
            gradient = grad_fn(jnp.array(params), batch)
            return np.array(gradient, dtype=np.float32)
        except ImportError:
            return self._compute_numerical(params, loss_fn, batch)

    def _compute_torch(
        self,
        params: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
    ) -> NDArray[np.float32]:
        try:
            import torch

            params_tensor = torch.tensor(params, requires_grad=True, dtype=torch.float32)
            loss = loss_fn(params_tensor, batch)
            loss.backward()

            return params_tensor.grad.numpy().astype(np.float32)
        except ImportError:
            return self._compute_numerical(params, loss_fn, batch)
