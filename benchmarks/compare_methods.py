#!/usr/bin/env python3
"""
End-to-end benchmark: HoloGrad vs PowerSGD vs Full SGD

Usage:
    python benchmarks/compare_methods.py --method full_sgd --steps 1000
    python benchmarks/compare_methods.py --method holograd --steps 1000
    python benchmarks/compare_methods.py --method holograd_momentum --steps 1000
    torchrun --nproc_per_node=4 benchmarks/compare_methods.py --method powersgd --steps 5000

Methods:
    - full_sgd: Full gradient synchronization (baseline)
    - powersgd: Low-rank gradient compression (PyTorch built-in)
    - holograd: K-direction projection (default K=64)
    - holograd_momentum: Single momentum-direction projection (most efficient)
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config


@dataclass
class BenchmarkConfig:
    method: str = "full_sgd"
    model_size: str = "small"
    num_steps: int = 5000
    batch_size: int = 8
    seq_length: int = 256
    learning_rate: float = 3e-4
    powersgd_rank: int = 1
    holograd_k: int = 64
    holograd_momentum_beta: float = 0.9
    log_interval: int = 100
    save_dir: str = "results/benchmark"
    world_size: int = 1
    rank: int = 0


@dataclass
class BenchmarkResult:
    method: str
    model_size: str
    world_size: int
    num_params: int = 0
    steps: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    wall_clock: List[float] = field(default_factory=list)
    bits_per_step: float = 0.0
    total_bits: float = 0.0
    final_loss: float = 0.0
    final_perplexity: float = 0.0
    total_time: float = 0.0
    tokens_per_second: float = 0.0


class SyntheticTextDataset(Dataset):
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

        torch.manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length + 1))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


MODEL_CONFIGS = {
    "tiny": {"n_layer": 2, "n_head": 2, "n_embd": 128},
    "small": {"n_layer": 6, "n_head": 6, "n_embd": 384},
    "medium": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "large": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
}


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_model(model_size: str, target_device: torch.device) -> nn.Module:
    cfg = MODEL_CONFIGS[model_size]
    gpt_config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    )
    model = GPT2LMHeadModel(gpt_config)
    return nn.Module.to(model, target_device)


def create_dataloader(config: BenchmarkConfig) -> DataLoader:
    dataset = SyntheticTextDataset(
        vocab_size=50257,
        seq_length=config.seq_length,
        num_samples=10000,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def calculate_bits_per_step(method: str, num_params: int, k: int = 64, rank: int = 1) -> float:
    """
    Communication cost (bits/step):
    - Full SGD: D * 32 (full gradient, float32)
    - PowerSGD: rank * 2 * sqrt(D) * 32 (low-rank)
    - HoloGrad: K * 96 (K scalars + seeds)
    - HoloGrad-Momentum: 32 (single scalar)
    """
    if method == "full_sgd":
        return num_params * 32
    elif method.startswith("powersgd"):
        return rank * 2 * (num_params**0.5) * 32
    elif method == "holograd":
        return k * 96
    elif method == "holograd_momentum":
        return 32
    return num_params * 32


def train_full_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    config: BenchmarkConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> BenchmarkResult:
    num_params = sum(p.numel() for p in model.parameters())
    result = BenchmarkResult(
        method="full_sgd",
        model_size=config.model_size,
        world_size=world_size,
        num_params=num_params,
    )

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    data_iter = iter(train_loader)
    start_time = time.time()
    total_tokens = 0

    for step in range(1, config.num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_tokens += input_ids.numel()

        if step % config.log_interval == 0 or step == config.num_steps:
            elapsed = time.time() - start_time
            result.steps.append(step)
            result.tokens.append(total_tokens)
            result.losses.append(loss.item())
            result.wall_clock.append(elapsed)

            if rank == 0:
                print(
                    f"[Full SGD] Step {step}/{config.num_steps} | "
                    f"Loss: {loss.item():.4f} | Tokens: {total_tokens:,} | Time: {elapsed:.1f}s"
                )

    result.final_loss = result.losses[-1]
    result.final_perplexity = torch.exp(torch.tensor(result.final_loss)).item()
    result.total_time = time.time() - start_time
    result.tokens_per_second = total_tokens / result.total_time
    result.bits_per_step = calculate_bits_per_step("full_sgd", num_params)
    result.total_bits = result.bits_per_step * config.num_steps

    return result


def train_powersgd(
    model: nn.Module,
    train_loader: DataLoader,
    config: BenchmarkConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> BenchmarkResult:
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

    num_params = sum(p.numel() for p in model.parameters())
    method_name = f"powersgd_rank{config.powersgd_rank}"
    result = BenchmarkResult(
        method=method_name,
        model_size=config.model_size,
        world_size=world_size,
        num_params=num_params,
    )

    if world_size < 2:
        print("[PowerSGD] Requires world_size >= 2, skipping")
        return result

    model = DDP(model, device_ids=[rank])
    state = powerSGD.PowerSGDState(
        process_group=None,
        matrix_approximation_rank=config.powersgd_rank,
        start_powerSGD_iter=10,
    )
    model.register_comm_hook(state, powerSGD.powerSGD_hook)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    data_iter = iter(train_loader)
    start_time = time.time()
    total_tokens = 0

    for step in range(1, config.num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_tokens += input_ids.numel()

        if step % config.log_interval == 0 or step == config.num_steps:
            elapsed = time.time() - start_time
            result.steps.append(step)
            result.tokens.append(total_tokens)
            result.losses.append(loss.item())
            result.wall_clock.append(elapsed)

            if rank == 0:
                print(
                    f"[PowerSGD-R{config.powersgd_rank}] Step {step}/{config.num_steps} | "
                    f"Loss: {loss.item():.4f} | Tokens: {total_tokens:,} | Time: {elapsed:.1f}s"
                )

    result.final_loss = result.losses[-1]
    result.final_perplexity = torch.exp(torch.tensor(result.final_loss)).item()
    result.total_time = time.time() - start_time
    result.tokens_per_second = total_tokens / result.total_time
    result.bits_per_step = calculate_bits_per_step(
        method_name, num_params, rank=config.powersgd_rank
    )
    result.total_bits = result.bits_per_step * config.num_steps

    return result


def generate_random_direction(dimension: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate a unit-norm random direction from seed."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    direction = torch.randn(dimension, generator=gen, device=device)
    return direction / torch.norm(direction)


def train_holograd(
    model: nn.Module,
    train_loader: DataLoader,
    config: BenchmarkConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> BenchmarkResult:
    """
    HoloGrad with K random directions per step.

    Algorithm:
    1. Compute full gradient g via backprop
    2. For k=1..K: project onto random direction v_k, get scalar a_k = <g, v_k>
    3. Reconstruct: g_hat = (D/K) * sum_k a_k * v_k  (scale correction)
    4. Update: theta = theta - lr * g_hat

    Communication cost: K * 96 bits (K scalars + seeds, ~32+64 bits each)
    """
    num_params = sum(p.numel() for p in model.parameters())
    K = config.holograd_k

    result = BenchmarkResult(
        method=f"holograd_k{K}",
        model_size=config.model_size,
        world_size=world_size,
        num_params=num_params,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    model.train()
    data_iter = iter(train_loader)
    start_time = time.time()
    total_tokens = 0
    global_seed = 42

    for step in range(1, config.num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        full_grad = torch.cat([g.flatten() for g in grads])

        reconstructed_grad = torch.zeros_like(full_grad)
        scale_factor = num_params / K

        for k in range(K):
            seed_k = global_seed + step * K + k
            direction = generate_random_direction(num_params, seed_k, device)
            scalar = torch.dot(full_grad, direction).item()
            reconstructed_grad += scalar * direction

        reconstructed_grad = scale_factor * reconstructed_grad

        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.grad = reconstructed_grad[offset : offset + numel].view(p.shape)
            offset += numel

        optimizer.step()
        total_tokens += input_ids.numel()

        if step % config.log_interval == 0 or step == config.num_steps:
            elapsed = time.time() - start_time
            result.steps.append(step)
            result.tokens.append(total_tokens)
            result.losses.append(loss.item())
            result.wall_clock.append(elapsed)

            if rank == 0:
                print(
                    f"[HoloGrad-K{K}] Step {step}/{config.num_steps} | "
                    f"Loss: {loss.item():.4f} | Tokens: {total_tokens:,} | Time: {elapsed:.1f}s"
                )

    result.final_loss = result.losses[-1]
    result.final_perplexity = torch.exp(torch.tensor(result.final_loss)).item()
    result.total_time = time.time() - start_time
    result.tokens_per_second = total_tokens / result.total_time
    result.bits_per_step = calculate_bits_per_step("holograd", num_params, k=K)
    result.total_bits = result.bits_per_step * config.num_steps

    return result


def train_holograd_momentum(
    model: nn.Module,
    train_loader: DataLoader,
    config: BenchmarkConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> BenchmarkResult:
    """
    Momentum-Centric HoloGrad: Single scalar per worker.

    Algorithm:
    1. Maintain momentum buffer m (coordinator side)
    2. Each step: compute a = <g, m/||m||> (scalar projection onto momentum direction)
    3. Update: g_hat = a * (m/||m||) * ||m|| = a * m / ||m||
    4. Update momentum: m = beta * m + (1-beta) * g_hat
    5. Update params: theta = theta - lr * g_hat

    Communication cost: 32 bits (single float32 scalar)
    """
    num_params = sum(p.numel() for p in model.parameters())
    beta = config.holograd_momentum_beta

    result = BenchmarkResult(
        method="holograd_momentum",
        model_size=config.model_size,
        world_size=world_size,
        num_params=num_params,
    )

    momentum = torch.zeros(num_params, device=device)
    grad_norm_ema = 1.0
    grad_norm_alpha = 0.1
    warmup_steps = 10

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    model.train()
    data_iter = iter(train_loader)
    start_time = time.time()
    total_tokens = 0

    for step in range(1, config.num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        full_grad = torch.cat([g.flatten() for g in grads])
        grad_norm = torch.norm(full_grad).item()

        grad_norm_ema = (1 - grad_norm_alpha) * grad_norm_ema + grad_norm_alpha * grad_norm

        if step <= warmup_steps:
            direction = generate_random_direction(num_params, 42 + step, device)
            scalar = torch.dot(full_grad, direction).item()
            reconstructed_grad = scalar * direction * num_params
        else:
            momentum_norm = torch.norm(momentum)
            if momentum_norm > 1e-8:
                momentum_direction = momentum / momentum_norm
                scalar = torch.dot(full_grad, momentum_direction).item()
                reconstructed_grad = scalar * momentum_direction * grad_norm_ema
            else:
                reconstructed_grad = full_grad

        momentum = beta * momentum + (1 - beta) * reconstructed_grad

        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.grad = reconstructed_grad[offset : offset + numel].view(p.shape)
            offset += numel

        optimizer.step()
        total_tokens += input_ids.numel()

        if step % config.log_interval == 0 or step == config.num_steps:
            elapsed = time.time() - start_time
            result.steps.append(step)
            result.tokens.append(total_tokens)
            result.losses.append(loss.item())
            result.wall_clock.append(elapsed)

            if rank == 0:
                print(
                    f"[HoloGrad-Momentum] Step {step}/{config.num_steps} | "
                    f"Loss: {loss.item():.4f} | Tokens: {total_tokens:,} | Time: {elapsed:.1f}s"
                )

    result.final_loss = result.losses[-1]
    result.final_perplexity = torch.exp(torch.tensor(result.final_loss)).item()
    result.total_time = time.time() - start_time
    result.tokens_per_second = total_tokens / result.total_time
    result.bits_per_step = calculate_bits_per_step("holograd_momentum", num_params)
    result.total_bits = result.bits_per_step * config.num_steps

    return result


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    rank, world_size, local_rank = setup_distributed()
    config.rank = rank
    config.world_size = world_size

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {config.method} | Model: {config.model_size} | World: {world_size}")
        print(f"Steps: {config.num_steps} | Batch: {config.batch_size} | Seq: {config.seq_length}")
        print(f"{'=' * 60}\n")

    model = create_model(config.model_size, device)
    train_loader = create_dataloader(config)

    if config.method == "full_sgd":
        result = train_full_sgd(model, train_loader, config, device, rank, world_size)
    elif config.method.startswith("powersgd"):
        result = train_powersgd(model, train_loader, config, device, rank, world_size)
    elif config.method == "holograd":
        result = train_holograd(model, train_loader, config, device, rank, world_size)
    elif config.method == "holograd_momentum":
        result = train_holograd_momentum(model, train_loader, config, device, rank, world_size)
    else:
        raise ValueError(f"Unknown method: {config.method}")

    if rank == 0:
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_path = save_dir / f"{config.method}_{config.model_size}_w{world_size}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Final: Loss={result.final_loss:.4f} PPL={result.final_perplexity:.2f}")
        print(f"Time: {result.total_time:.1f}s | Tokens/s: {result.tokens_per_second:.1f}")
        print(f"Comm: {result.bits_per_step / 1e9:.3f} Gbits/step")
        print(f"Saved: {result_path}")
        print(f"{'=' * 60}\n")

    cleanup_distributed()
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark gradient compression methods")
    parser.add_argument(
        "--method",
        type=str,
        default="full_sgd",
        choices=["full_sgd", "powersgd", "powersgd_rank2", "holograd", "holograd_momentum"],
    )
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["tiny", "small", "medium", "large"]
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--powersgd_rank", type=int, default=1)
    parser.add_argument("--holograd_k", type=int, default=64)
    parser.add_argument("--holograd_momentum_beta", type=float, default=0.9)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="results/benchmark")

    args = parser.parse_args()

    config = BenchmarkConfig(
        method=args.method,
        model_size=args.model_size,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.lr,
        powersgd_rank=2 if args.method == "powersgd_rank2" else args.powersgd_rank,
        holograd_k=args.holograd_k,
        holograd_momentum_beta=args.holograd_momentum_beta,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
    )

    run_benchmark(config)


if __name__ == "__main__":
    main()
