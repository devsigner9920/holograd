from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class ModelInfo:
    name: str
    num_parameters: int
    param_shapes: Dict[str, Tuple[int, ...]]
    dtype: np.dtype = np.float32


class ParameterManager:
    def __init__(self, param_shapes: Dict[str, Tuple[int, ...]]):
        self.param_shapes = param_shapes
        self.param_names = list(param_shapes.keys())
        self._offsets: Dict[str, int] = {}
        self._total_size = 0

        offset = 0
        for name in self.param_names:
            self._offsets[name] = offset
            offset += int(np.prod(param_shapes[name]))
        self._total_size = offset

    @property
    def total_parameters(self) -> int:
        return self._total_size

    def flatten(self, params: Dict[str, NDArray]) -> NDArray[np.float32]:
        flat = np.zeros(self._total_size, dtype=np.float32)
        for name in self.param_names:
            offset = self._offsets[name]
            size = int(np.prod(self.param_shapes[name]))
            flat[offset : offset + size] = params[name].flatten()
        return flat

    def unflatten(self, flat: NDArray[np.float32]) -> Dict[str, NDArray]:
        params = {}
        for name in self.param_names:
            offset = self._offsets[name]
            shape = self.param_shapes[name]
            size = int(np.prod(shape))
            params[name] = flat[offset : offset + size].reshape(shape)
        return params

    def get_param_slice(self, name: str, flat: NDArray[np.float32]) -> NDArray:
        offset = self._offsets[name]
        shape = self.param_shapes[name]
        size = int(np.prod(shape))
        return flat[offset : offset + size].reshape(shape)


class SimpleTransformerBlock:
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 0.02

        self.params = {
            "ln1_weight": np.ones(hidden_size, dtype=np.float32),
            "ln1_bias": np.zeros(hidden_size, dtype=np.float32),
            "attn_qkv": rng.normal(0, scale, (hidden_size, 3 * hidden_size)).astype(np.float32),
            "attn_proj": rng.normal(0, scale, (hidden_size, hidden_size)).astype(np.float32),
            "ln2_weight": np.ones(hidden_size, dtype=np.float32),
            "ln2_bias": np.zeros(hidden_size, dtype=np.float32),
            "ff_up": rng.normal(0, scale, (hidden_size, ff_size)).astype(np.float32),
            "ff_down": rng.normal(0, scale, (ff_size, hidden_size)).astype(np.float32),
        }
        self.hidden_size = hidden_size
        self.num_heads = num_heads


class SimpleGPT2:
    GPT2_CONFIGS = {
        "tiny": {"n_layer": 1, "n_head": 2, "n_embd": 32, "vocab_size": 100},
        "small": {"n_layer": 12, "n_head": 12, "n_embd": 768, "vocab_size": 50257},
        "medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024, "vocab_size": 50257},
        "large": {"n_layer": 36, "n_head": 20, "n_embd": 1280, "vocab_size": 50257},
    }

    def __init__(
        self,
        size: str = "small",
        max_seq_len: int = 1024,
        seed: int = 0,
        n_layer: Optional[int] = None,
        n_head: Optional[int] = None,
        n_embd: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        config = self.GPT2_CONFIGS.get(size, self.GPT2_CONFIGS["small"])
        self.n_layer = n_layer if n_layer is not None else config["n_layer"]
        self.n_head = n_head if n_head is not None else config["n_head"]
        self.n_embd = n_embd if n_embd is not None else config["n_embd"]
        self.vocab_size = vocab_size if vocab_size is not None else config["vocab_size"]
        self.max_seq_len = max_seq_len

        rng = np.random.default_rng(seed)
        scale = 0.02

        self.params: Dict[str, NDArray] = {
            "wte": rng.normal(0, scale, (self.vocab_size, self.n_embd)).astype(np.float32),
            "wpe": rng.normal(0, scale, (max_seq_len, self.n_embd)).astype(np.float32),
        }

        ff_size = 4 * self.n_embd
        for i in range(self.n_layer):
            block = SimpleTransformerBlock(self.n_embd, self.n_head, ff_size, seed=seed + i)
            for key, value in block.params.items():
                self.params[f"block_{i}_{key}"] = value

        self.params["ln_f_weight"] = np.ones(self.n_embd, dtype=np.float32)
        self.params["ln_f_bias"] = np.zeros(self.n_embd, dtype=np.float32)

        self._param_manager = ParameterManager({k: v.shape for k, v in self.params.items()})

    @property
    def num_parameters(self) -> int:
        return self._param_manager.total_parameters

    @property
    def param_manager(self) -> ParameterManager:
        return self._param_manager

    def get_flat_params(self) -> NDArray[np.float32]:
        return self._param_manager.flatten(self.params)

    def set_flat_params(self, flat: NDArray[np.float32]) -> None:
        self.params = self._param_manager.unflatten(flat)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=f"gpt2-{self.n_layer}L-{self.n_head}H-{self.n_embd}E",
            num_parameters=self.num_parameters,
            param_shapes={k: v.shape for k, v in self.params.items()},
        )

    def _layer_norm(
        self, x: NDArray[np.float32], weight: NDArray[np.float32], bias: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + 1e-5) + bias

    def _gelu(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _attention(
        self,
        hidden: NDArray[np.float32],
        qkv_weight: NDArray[np.float32],
        proj_weight: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        batch_size, seq_len, n_embd = hidden.shape
        head_dim = n_embd // self.n_head

        qkv = hidden @ qkv_weight
        q, k, v = np.split(qkv, 3, axis=-1)

        q = q.reshape(batch_size, seq_len, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_head, head_dim).transpose(0, 2, 1, 3)

        attn_scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)

        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        attn_scores = np.where(causal_mask, -1e9, attn_scores)

        attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, n_embd)

        return attn_out @ proj_weight

    def _ffn(
        self,
        hidden: NDArray[np.float32],
        up_weight: NDArray[np.float32],
        down_weight: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        return self._gelu(hidden @ up_weight) @ down_weight

    def forward(self, input_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        batch_size, seq_len = input_ids.shape

        token_emb = self.params["wte"][input_ids]
        pos_emb = self.params["wpe"][:seq_len]
        hidden = token_emb + pos_emb

        for i in range(self.n_layer):
            ln1_out = self._layer_norm(
                hidden,
                self.params[f"block_{i}_ln1_weight"],
                self.params[f"block_{i}_ln1_bias"],
            )
            attn_out = self._attention(
                ln1_out,
                self.params[f"block_{i}_attn_qkv"],
                self.params[f"block_{i}_attn_proj"],
            )
            hidden = hidden + attn_out

            ln2_out = self._layer_norm(
                hidden,
                self.params[f"block_{i}_ln2_weight"],
                self.params[f"block_{i}_ln2_bias"],
            )
            ffn_out = self._ffn(
                ln2_out,
                self.params[f"block_{i}_ff_up"],
                self.params[f"block_{i}_ff_down"],
            )
            hidden = hidden + ffn_out

        hidden = self._layer_norm(hidden, self.params["ln_f_weight"], self.params["ln_f_bias"])
        logits = hidden @ self.params["wte"].T
        return logits

    def compute_loss(
        self,
        input_ids: NDArray[np.int64],
        labels: NDArray[np.int64],
    ) -> float:
        logits = self.forward(input_ids)

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        log_softmax = logits_flat - max_logits - np.log(np.sum(exp_logits, axis=-1, keepdims=True))

        loss = -np.mean(log_softmax[np.arange(len(labels_flat)), labels_flat])
        return float(loss)

    def _layer_norm_torch(
        self, x: "torch.Tensor", weight: "torch.Tensor", bias: "torch.Tensor"
    ) -> "torch.Tensor":
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return weight * (x - mean) / torch.sqrt(var + 1e-5) + bias

    def _gelu_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        return (
            0.5
            * x
            * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))
        )

    def _attention_torch(
        self,
        hidden: "torch.Tensor",
        qkv_weight: "torch.Tensor",
        proj_weight: "torch.Tensor",
    ) -> "torch.Tensor":
        batch_size, seq_len, n_embd = hidden.shape
        head_dim = n_embd // self.n_head

        qkv = hidden @ qkv_weight
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(batch_size, seq_len, self.n_head, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_head, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_head, head_dim).permute(0, 2, 1, 3)

        attn_scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden.device), diagonal=1
        ).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_out = attn_weights @ v
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, n_embd)

        return attn_out @ proj_weight

    def _ffn_torch(
        self,
        hidden: "torch.Tensor",
        up_weight: "torch.Tensor",
        down_weight: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._gelu_torch(hidden @ up_weight) @ down_weight

    def forward_torch(
        self, input_ids: "torch.Tensor", params_dict: Dict[str, "torch.Tensor"]
    ) -> "torch.Tensor":
        batch_size, seq_len = input_ids.shape

        token_emb = F.embedding(input_ids, params_dict["wte"])
        pos_emb = params_dict["wpe"][:seq_len]
        hidden = token_emb + pos_emb

        for i in range(self.n_layer):
            ln1_out = self._layer_norm_torch(
                hidden,
                params_dict[f"block_{i}_ln1_weight"],
                params_dict[f"block_{i}_ln1_bias"],
            )
            attn_out = self._attention_torch(
                ln1_out,
                params_dict[f"block_{i}_attn_qkv"],
                params_dict[f"block_{i}_attn_proj"],
            )
            hidden = hidden + attn_out

            ln2_out = self._layer_norm_torch(
                hidden,
                params_dict[f"block_{i}_ln2_weight"],
                params_dict[f"block_{i}_ln2_bias"],
            )
            ffn_out = self._ffn_torch(
                ln2_out,
                params_dict[f"block_{i}_ff_up"],
                params_dict[f"block_{i}_ff_down"],
            )
            hidden = hidden + ffn_out

        hidden = self._layer_norm_torch(
            hidden, params_dict["ln_f_weight"], params_dict["ln_f_bias"]
        )
        logits = hidden @ params_dict["wte"].T
        return logits

    def compute_loss_torch(
        self,
        input_ids: "torch.Tensor",
        labels: "torch.Tensor",
        params_dict: Dict[str, "torch.Tensor"],
    ) -> "torch.Tensor":
        logits = self.forward_torch(input_ids, params_dict)
        batch_size, seq_len, vocab_size = logits.shape
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        return loss

    def flat_params_to_torch_dict(self, flat_params: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        params_dict = {}
        for name in self._param_manager.param_names:
            offset = self._param_manager._offsets[name]
            shape = self._param_manager.param_shapes[name]
            size = int(np.prod(shape))
            params_dict[name] = flat_params[offset : offset + size].reshape(shape)
        return params_dict


def create_loss_fn(model: SimpleGPT2) -> Callable:
    def loss_fn(flat_params: NDArray[np.float32], batch: Tuple) -> float:
        input_ids, labels = batch
        model.set_flat_params(flat_params)
        return model.compute_loss(input_ids, labels)

    return loss_fn


def create_gradient_fn(model: SimpleGPT2) -> Callable[[NDArray[np.float32]], float]:
    def gradient_projection_fn(direction: NDArray[np.float32]) -> float:
        return float(np.sum(direction))

    return gradient_projection_fn
