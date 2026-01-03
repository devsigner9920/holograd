#!/usr/bin/env python3
"""
Distributed GPT-2 training on WikiText-2 using HoloGrad protocol.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import requests

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data
from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
from holograd.core.types import Proof


WORKERS = {
    "29463757": {"host": "ssh2.vast.ai", "port": 23756, "local_port": 8001},
    "29463761": {"host": "ssh5.vast.ai", "port": 23760, "local_port": 8002},
    "29463762": {"host": "ssh2.vast.ai", "port": 23762, "local_port": 8003},
    "29463764": {"host": "ssh2.vast.ai", "port": 23764, "local_port": 8004},
    "29463766": {"host": "ssh6.vast.ai", "port": 23766, "local_port": 8005},
    "29463768": {"host": "ssh9.vast.ai", "port": 23768, "local_port": 8006},
    "29463778": {"host": "ssh3.vast.ai", "port": 23778, "local_port": 8007},
    "29463780": {"host": "ssh8.vast.ai", "port": 23780, "local_port": 8008},
    "29464653": {"host": "ssh9.vast.ai", "port": 24652, "local_port": 8009},
    "29464654": {"host": "ssh6.vast.ai", "port": 24654, "local_port": 8010},
    "29464657": {"host": "ssh4.vast.ai", "port": 24656, "local_port": 8011},
}


class WorkerManager:
    def __init__(self):
        self.tunnels: Dict[str, subprocess.Popen] = {}

    def start_tunnel(self, worker_id: str, info: dict) -> bool:
        cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-N",
            "-L",
            f"{info['local_port']}:localhost:8000",
            "-p",
            str(info["port"]),
            f"root@{info['host']}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.tunnels[worker_id] = proc
        time.sleep(2)
        return proc.poll() is None

    def check_worker_health(self, local_port: int) -> bool:
        try:
            resp = requests.get(f"http://localhost:{local_port}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def init_worker(
        self,
        local_port: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        vocab_size: int,
        seq_length: int,
        params: List[float],
    ) -> bool:
        try:
            resp = requests.post(
                f"http://localhost:{local_port}/init",
                json={
                    "model_size": "custom",
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "n_embd": n_embd,
                    "vocab_size": vocab_size,
                    "seq_length": seq_length,
                    "params": params,
                },
                timeout=60,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"    Init error: {e}")
            return False

    def update_worker_params(self, local_port: int, params: List[float]) -> bool:
        try:
            resp = requests.post(
                f"http://localhost:{local_port}/update_params",
                json={"params": params},
                timeout=30,
            )
            return resp.status_code == 200
        except:
            return False

    def update_worker_codebook(
        self, local_port: int, U: List[List[float]], rank: int, dimension: int
    ) -> bool:
        try:
            resp = requests.post(
                f"http://localhost:{local_port}/update_codebook",
                json={"U": U, "rank": rank, "dimension": dimension},
                timeout=30,
            )
            return resp.status_code == 200
        except:
            return False

    def compute_task(
        self,
        local_port: int,
        step: int,
        seed: bytes,
        use_adc: bool,
        input_ids: List[List[int]],
        labels: List[List[int]],
    ) -> Optional[dict]:
        try:
            seed_hex = seed.hex() if isinstance(seed, bytes) else str(seed)
            resp = requests.post(
                f"http://localhost:{local_port}/compute",
                json={
                    "step": step,
                    "seed": seed_hex,
                    "use_adc": use_adc,
                    "input_ids": input_ids,
                    "labels": labels,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"  HTTP {resp.status_code} on port {local_port}")
        except Exception as e:
            print(f"  Task error on port {local_port}: {e}")
        return None

    def cleanup(self):
        for proc in self.tunnels.values():
            proc.terminate()


def run_training(
    num_steps: int = 100,
    K: int = 8,
    n_layer: int = 4,
    n_head: int = 6,
    n_embd: int = 384,
    seq_length: int = 128,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    max_train_samples: int = 5000,
):
    manager = WorkerManager()

    print("=" * 70)
    print("HoloGrad Distributed Training - GPT-2 on WikiText-2")
    print("=" * 70)

    print(f"\nModel: {n_layer}L-{n_head}H-{n_embd}E, seq_length={seq_length}")
    print(f"Training: {num_steps} steps, K={K}, lr={learning_rate}")

    print("\n[1] Creating SSH tunnels...")
    active_workers = []
    for wid, info in WORKERS.items():
        if manager.start_tunnel(wid, info):
            active_workers.append((wid, info))
            print(f"  {wid}: tunnel on localhost:{info['local_port']}")

    time.sleep(3)

    print("\n[2] Checking worker health...")
    healthy_workers = []
    for wid, info in active_workers:
        if manager.check_worker_health(info["local_port"]):
            healthy_workers.append((wid, info))
            print(f"  {wid}: healthy")
        else:
            print(f"  {wid}: not responding")

    if len(healthy_workers) < K // 2:
        print(f"Not enough healthy workers ({len(healthy_workers)} < {K // 2})")
        manager.cleanup()
        return

    print(f"\n[3] Loading WikiText-2 data...")
    train_loader, val_loader, vocab_size = create_wikitext_data(
        seq_length=seq_length,
        batch_size=batch_size,
        dataset_name="wikitext-2-raw-v1",
        max_train_samples=max_train_samples,
        max_val_samples=500,
    )
    print(f"  vocab_size={vocab_size}, train_batches={len(train_loader)}")

    print(f"\n[4] Creating model...")
    model = SimpleGPT2(
        size="custom",
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=vocab_size,
        max_seq_len=seq_length,
    )
    print(f"  Parameters: {model.num_parameters:,}")

    params = model.get_flat_params().tolist()

    print(f"\n[5] Initializing {len(healthy_workers)} workers...")
    initialized_workers = []
    for wid, info in healthy_workers:
        ok = manager.init_worker(
            info["local_port"], n_layer, n_head, n_embd, vocab_size, seq_length, params
        )
        if ok:
            initialized_workers.append((wid, info))
            print(f"  {wid}: initialized")
        else:
            print(f"  {wid}: failed")

    if len(initialized_workers) < K // 2:
        print(f"Not enough initialized workers")
        manager.cleanup()
        return

    healthy_workers = initialized_workers

    print(f"\n[6] Creating coordinator...")
    coord_config = CoordinatorConfig(
        dimension=model.num_parameters,
        num_workers=len(healthy_workers),
        proofs_per_step=K,
        use_adc=True,
        adc_rank=16,
        learning_rate=learning_rate,
        device="cpu",
    )
    coordinator = Coordinator(coord_config)
    coordinator.set_parameters(model.get_flat_params())

    if coordinator.codebook is not None:
        U = coordinator.codebook.codebook.tolist()
        for wid, info in healthy_workers:
            manager.update_worker_codebook(info["local_port"], U, 16, model.num_parameters)

    print(f"\n[7] Starting training ({num_steps} steps)...")
    print("-" * 70)

    train_iter = iter(train_loader)
    losses = []
    start_time = time.time()

    for step in range(num_steps):
        step_start = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch.input_ids.tolist()
        labels = batch.labels.tolist()

        coordinator.set_batch(batch.input_ids.flatten(), step)
        tasks = coordinator.publish_tasks(step)

        proofs = []
        with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
            futures = {}
            for i, task in enumerate(tasks[: len(healthy_workers)]):
                wid, info = healthy_workers[i % len(healthy_workers)]
                future = executor.submit(
                    manager.compute_task,
                    info["local_port"],
                    step,
                    task.seed,
                    task.use_adc,
                    input_ids,
                    labels,
                )
                futures[future] = wid

            for future in as_completed(futures):
                result = future.result()
                if result:
                    proof = Proof(
                        step=result["step"],
                        worker_id=int(futures[future]),
                        seed=bytes.fromhex(result["seed"]),
                        scalar=result["scalar"],
                        timestamp=time.time(),
                        adc_projection=np.array(result["adc_projection"])
                        if result.get("adc_projection")
                        else None,
                    )
                    proofs.append(proof)
                    coordinator.collect_proof(proof)

        if len(proofs) < K // 2:
            print(f"Step {step}: Only {len(proofs)} proofs, skipping")
            continue

        gradient, _ = coordinator.aggregate()
        new_params = coordinator.update_parameters(gradient)
        model.set_flat_params(new_params)

        for wid, info in healthy_workers:
            manager.update_worker_params(info["local_port"], new_params.tolist())

        loss_val = model.compute_loss(batch.input_ids, batch.labels)
        losses.append(loss_val)

        if (step + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step + 1:3d}/{num_steps} | Loss: {loss_val:.4f} | "
                f"Time: {time.time() - step_start:.1f}s | Proofs: {len(proofs)} | "
                f"Total: {elapsed:.0f}s"
            )

    print("-" * 70)
    if losses:
        print(f"Training complete!")
        print(f"  Final loss: {np.mean(losses[-5:]):.4f}")
        print(f"  Total time: {time.time() - start_time:.1f}s")

    manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 WikiText-2 Distributed Training")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--K", type=int, default=6, help="Proofs per step")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max training samples")
    args = parser.parse_args()

    run_training(
        num_steps=args.steps,
        K=args.K,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_train_samples=args.max_samples,
    )
