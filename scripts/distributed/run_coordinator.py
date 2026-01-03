#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import requests

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_synthetic_data
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
        self.servers: Dict[str, subprocess.Popen] = {}

    def start_worker_server(self, worker_id: str, info: dict) -> bool:
        # Kill any existing server first, then start new one with proper backgrounding
        cmd = (
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p {info['port']} root@{info['host']} "
            f"\"bash -c 'pkill -f worker_server.py 2>/dev/null; sleep 1; "
            f"cd /root/holograd && nohup python3 scripts/distributed/worker_server.py --port 8000 > /tmp/worker.log 2>&1 & sleep 1'\""
        )
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=45)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

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
        model_size: str,
        vocab_size: int,
        seq_length: int,
        params: List[float],
    ) -> bool:
        try:
            resp = requests.post(
                f"http://localhost:{local_port}/init",
                json={
                    "model_size": model_size,
                    "vocab_size": vocab_size,
                    "seq_length": seq_length,
                    "params": params,
                },
                timeout=30,
            )
            return resp.status_code == 200
        except:
            return False

    def update_worker_params(self, local_port: int, params: List[float]) -> bool:
        try:
            resp = requests.post(
                f"http://localhost:{local_port}/update_params", json={"params": params}, timeout=30
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
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"  Task error on port {local_port}: {e}")
        return None

    def cleanup(self):
        for proc in self.tunnels.values():
            proc.terminate()
        for proc in self.servers.values():
            proc.terminate()


def run_training(num_steps: int = 100, K: int = 8, skip_server_start: bool = False):
    manager = WorkerManager()

    print("=" * 60)
    print("HoloGrad FastAPI-Based Distributed Training")
    print("=" * 60)

    if not skip_server_start:
        print("\n[1] Starting worker servers...")
        for wid, info in WORKERS.items():
            ok = manager.start_worker_server(wid, info)
            print(f"  {wid}: {'started' if ok else 'failed'}")
        time.sleep(5)
    else:
        print("\n[1] Skipping server start (--skip-server-start)")
        time.sleep(2)

    print("\n[2] Creating SSH tunnels...")
    active_workers = []
    for wid, info in WORKERS.items():
        ok = manager.start_tunnel(wid, info)
        if ok:
            active_workers.append((wid, info))
            print(f"  {wid}: tunnel on localhost:{info['local_port']}")

    time.sleep(3)

    print("\n[3] Checking worker health...")
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

    print(f"\n[4] Initializing model on {len(healthy_workers)} workers...")

    vocab_size = 100
    seq_length = 64
    model = SimpleGPT2(size="tiny", max_seq_len=seq_length, vocab_size=vocab_size)
    params = model.get_flat_params().tolist()

    for wid, info in healthy_workers:
        ok = manager.init_worker(info["local_port"], "tiny", vocab_size, seq_length, params)
        print(f"  {wid}: {'initialized' if ok else 'failed'}")

    print(f"\n[5] Creating coordinator...")
    coord_config = CoordinatorConfig(
        dimension=model.num_parameters,
        num_workers=len(healthy_workers),
        proofs_per_step=K,
        use_adc=True,
        adc_rank=8,
        learning_rate=1e-3,
        device="cpu",
    )
    coordinator = Coordinator(coord_config)
    coordinator.set_parameters(model.get_flat_params())

    if coordinator.codebook is not None:
        U = coordinator.codebook.codebook.tolist()
        for wid, info in healthy_workers:
            manager.update_worker_codebook(info["local_port"], U, 8, model.num_parameters)

    print(f"\n[6] Loading data...")
    train_loader, _ = create_synthetic_data(
        vocab_size=vocab_size, seq_length=seq_length, num_train_samples=1000, batch_size=4
    )

    print(f"\n[7] Starting training ({num_steps} steps, K={K})...")
    print("-" * 60)

    train_iter = iter(train_loader)
    losses = []

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
                        seed=result["seed"],
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
            print(
                f"Step {step + 1}/{num_steps} | Loss: {loss_val:.4f} | Time: {time.time() - step_start:.1f}s | Proofs: {len(proofs)}"
            )

    print("-" * 60)
    if losses:
        print(f"Training complete! Final loss: {np.mean(losses[-5:]):.4f}")

    manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--skip-server-start", action="store_true")
    args = parser.parse_args()

    run_training(num_steps=args.steps, K=args.K, skip_server_start=args.skip_server_start)
