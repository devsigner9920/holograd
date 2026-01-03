from typing import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from holograd.gradient.backprop import GradientComputer, GradientResult


class JVPGradient(GradientComputer):
    def __init__(self, framework: str = "numpy"):
        self.framework = framework

    def compute_projection(
        self,
        params: NDArray[np.float32],
        direction: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
    ) -> GradientResult:
        import time

        start = time.perf_counter()

        if self.framework == "jax":
            scalar = self._compute_jax(params, direction, loss_fn, batch)
        elif self.framework == "torch":
            scalar = self._compute_torch(params, direction, loss_fn, batch)
        else:
            scalar = self._compute_numerical(params, direction, loss_fn, batch)

        elapsed = time.perf_counter() - start

        return GradientResult(
            scalar=scalar,
            gradient=None,
            compute_time=elapsed,
            memory_bytes=direction.nbytes,
        )

    def _compute_numerical(
        self,
        params: NDArray[np.float32],
        direction: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
        eps: float = 1e-5,
    ) -> float:
        params_plus = params + eps * direction
        params_minus = params - eps * direction

        loss_plus = loss_fn(params_plus, batch)
        loss_minus = loss_fn(params_minus, batch)

        return float((loss_plus - loss_minus) / (2 * eps))

    def _compute_jax(
        self,
        params: NDArray[np.float32],
        direction: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
    ) -> float:
        try:
            import jax
            import jax.numpy as jnp

            def loss_wrapper(p):
                return loss_fn(p, batch)

            primals = (jnp.array(params),)
            tangents = (jnp.array(direction),)

            _, jvp_value = jax.jvp(loss_wrapper, primals, tangents)
            return float(jvp_value)
        except ImportError:
            return self._compute_numerical(params, direction, loss_fn, batch)

    def _compute_torch(
        self,
        params: NDArray[np.float32],
        direction: NDArray[np.float32],
        loss_fn: Callable,
        batch: any,
    ) -> float:
        try:
            import torch
            from torch.autograd.functional import jvp

            params_tensor = torch.tensor(params, dtype=torch.float32)
            direction_tensor = torch.tensor(direction, dtype=torch.float32)

            def loss_wrapper(p):
                return loss_fn(p, batch)

            _, jvp_value = jvp(loss_wrapper, (params_tensor,), (direction_tensor,))
            return float(jvp_value)
        except ImportError:
            return self._compute_numerical(params, direction, loss_fn, batch)


def compute_jvp_gradient_projection(
    model,
    input_ids,
    labels,
    direction: NDArray[np.float32],
    device: str = "cuda",
) -> float:
    import torch
    from holograd.training.model import SimpleGPT2

    if isinstance(model, SimpleGPT2):
        return _compute_jvp_simplegpt2(model, input_ids, labels, direction, device)

    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

    model.zero_grad()

    tangent_params = []
    offset = 0
    for p in model.model.parameters():
        numel = p.numel()
        tangent = direction_t[offset : offset + numel].view_as(p)
        tangent_params.append(tangent)
        offset += numel

    with torch.enable_grad():
        for p in model.model.parameters():
            p.requires_grad_(True)

        output = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)),
            labels.view(-1),
        )

        grads = torch.autograd.grad(loss, model.model.parameters(), create_graph=False)

        jvp_value = 0.0
        for g, t in zip(grads, tangent_params):
            jvp_value += torch.sum(g * t).item()

    return jvp_value


def _compute_jvp_simplegpt2(
    model,
    input_ids,
    labels,
    direction: NDArray[np.float32],
    device: str = "cuda",
) -> float:
    import torch

    flat_params = model.get_flat_params()
    flat_params_t = torch.tensor(
        flat_params, dtype=torch.float32, device=device, requires_grad=True
    )
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

    params_dict = model.flat_params_to_torch_dict(flat_params_t)

    logits = model.forward_torch(input_ids, params_dict)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

    grad = torch.autograd.grad(loss, flat_params_t)[0]
    jvp_value = torch.sum(grad * direction_t).item()

    return jvp_value
