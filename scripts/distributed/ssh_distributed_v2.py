#!/usr/bin/env python3
"""
SSH-based distributed training for Vast.ai - Version 2
Uses file-based data transfer instead of inline scripts.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any
import base64

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np


WORKER_INSTANCES = {
    "29463757": {"host": "ssh2.vast.ai", "port": 23756, "gpu": "GTX_1080"},
    "29463761": {"host": "ssh5.vast.ai", "port": 23760, "gpu": "RTX_2060"},
    "29463762": {"host": "ssh2.vast.ai", "port": 23762, "gpu": "GTX_1660_S"},
    "29463764": {"host": "ssh2.vast.ai", "port": 23764, "gpu": "RTX_3070"},
    "29463766": {"host": "ssh6.vast.ai", "port": 23766, "gpu": "Quadro_P4000"},
    "29463768": {"host": "ssh9.vast.ai", "port": 23768, "gpu": "GTX_1660_Ti"},
    "29463778": {"host": "ssh3.vast.ai", "port": 23778, "gpu": "RTX_4060_Ti"},
    "29463780": {"host": "ssh8.vast.ai", "port": 23780, "gpu": "RTX_4070S"},
    "29464653": {"host": "ssh9.vast.ai", "port": 24652, "gpu": "GTX_1080"},
    "29464654": {"host": "ssh6.vast.ai", "port": 24654, "gpu": "GTX_1080"},
    "29464657": {"host": "ssh4.vast.ai", "port": 24656, "gpu": "GTX_1080"},
}


def scp_file(local_path: str, host: str, port: int, remote_path: str) -> bool:
    cmd = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=30",
        "-P",
        str(port),
        local_path,
        f"root@{host}:{remote_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0


def run_ssh(host: str, port: int, command: str, timeout: int = 120) -> Tuple[str, str, int]:
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=30",
        "-p",
        str(port),
        f"root@{host}",
        command,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", -1
    except Exception as e:
        return "", str(e), -1


def compute_jvp_remote(
    instance_id: str,
    host: str,
    port: int,
    step: int,
    seed: int,
    use_adc: bool,
) -> Dict[str, Any]:
    """Execute pre-deployed JVP computation on worker."""
    cmd = f'cd /root/holograd && python3 -c "from scripts.distributed.worker_task import run_task; run_task({step}, {seed}, {use_adc})"'

    stdout, stderr, rc = run_ssh(host, port, cmd, timeout=120)

    if rc != 0:
        return {"success": False, "error": stderr[:200], "instance_id": instance_id}

    for line in stdout.split("\n"):
        if line.startswith("RESULT:"):
            result = json.loads(line[7:])
            result["success"] = True
            result["instance_id"] = instance_id
            return result

    return {"success": False, "error": "No result", "instance_id": instance_id}


def deploy_task_data(workers: List[Tuple[str, Dict]], params, input_ids, labels, codebook_data):
    """Deploy task data to all workers via SCP."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        data = {
            "params": params,
            "input_ids": input_ids,
            "labels": labels,
            "codebook": codebook_data,
        }
        pickle.dump(data, f)
        local_path = f.name

    def deploy_one(instance_id, info):
        ok = scp_file(local_path, info["host"], info["port"], "/tmp/task_data.pkl")
        return instance_id, ok

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(deploy_one, iid, info) for iid, info in workers]
        results = [f.result() for f in as_completed(futures)]

    os.unlink(local_path)
    return sum(1 for _, ok in results if ok)


def run_distributed_training(
    num_steps: int = 100,
    batch_size: int = 4,
    seq_length: int = 64,
    lr: float = 1e-3,
    K: int = 8,
    adc_rank: int = 8,
):
    from holograd.training.model import SimpleGPT2
    from holograd.training.data import create_synthetic_data
    from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
    from holograd.core.types import Proof

    print("=" * 60)
    print("HoloGrad SSH-Based Distributed Training v2")
    print("=" * 60)
    print(f"Workers: {len(WORKER_INSTANCES)}")
    print(f"Steps: {num_steps}")
    print(f"K: {K}")
    print("-" * 60)

    vocab_size = 100
    train_loader, val_loader = create_synthetic_data(
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_train_samples=1000,
        batch_size=batch_size,
    )

    model = SimpleGPT2(size="tiny", max_seq_len=seq_length, vocab_size=vocab_size)
    print(f"Model parameters: {model.num_parameters:,}")

    coord_config = CoordinatorConfig(
        dimension=model.num_parameters,
        num_workers=len(WORKER_INSTANCES),
        proofs_per_step=K,
        use_adc=True,
        adc_rank=adc_rank,
        learning_rate=lr,
        device="cpu",
    )
    coordinator = Coordinator(coord_config)
    params = model.get_flat_params()
    coordinator.set_parameters(params)

    workers = list(WORKER_INSTANCES.items())
    train_iter = iter(train_loader)
    losses = []

    print("Starting training...")

    for step in range(num_steps):
        step_start = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch.input_ids
        labels = batch.labels

        coordinator.set_batch(input_ids.flatten(), step)
        tasks = coordinator.publish_tasks(step)

        codebook_data = None
        if coordinator.codebook is not None:
            codebook_data = {
                "U": coordinator.codebook.codebook,
                "rank": adc_rank,
                "dimension": model.num_parameters,
            }

        deployed = deploy_task_data(workers[:K], params, input_ids, labels, codebook_data)
        print(f"Step {step}: Deployed to {deployed} workers")

        proofs = []
        with ThreadPoolExecutor(max_workers=K) as executor:
            futures = {}
            for i, task in enumerate(tasks[:K]):
                instance_id, info = workers[i % len(workers)]
                future = executor.submit(
                    compute_jvp_remote,
                    instance_id,
                    info["host"],
                    info["port"],
                    step,
                    task.seed,
                    task.use_adc,
                )
                futures[future] = instance_id

            for future in as_completed(futures):
                result = future.result()
                if result.get("success"):
                    proof = Proof(
                        step=result["step"],
                        worker_id=int(result["instance_id"]),
                        seed=result["seed"],
                        scalar=result["scalar"],
                        timestamp=time.time(),
                        adc_projection=np.array(result["adc_projection"])
                        if result.get("adc_projection")
                        else None,
                    )
                    proofs.append(proof)
                    coordinator.collect_proof(proof)
                else:
                    print(
                        f"  Worker {result['instance_id']} failed: {result.get('error', '')[:50]}"
                    )

        if len(proofs) < K // 2:
            print(f"Step {step}: Only {len(proofs)} proofs, skipping")
            continue

        gradient, _ = coordinator.aggregate()
        new_params = coordinator.update_parameters(gradient)
        model.set_flat_params(new_params)
        params = new_params

        loss_val = model.compute_loss(input_ids, labels)
        losses.append(loss_val)

        print(
            f"Step {step + 1}/{num_steps} | Loss: {loss_val:.4f} | Time: {time.time() - step_start:.1f}s | Proofs: {len(proofs)}"
        )

    print("-" * 60)
    print(f"Training complete! Final loss: {np.mean(losses[-5:]) if losses else float('nan'):.4f}")


def test_workers():
    print("Testing workers...")
    for iid, info in WORKER_INSTANCES.items():
        stdout, stderr, rc = run_ssh(
            info["host"],
            info["port"],
            'python3 -c "import torch; print(torch.cuda.is_available())"',
            30,
        )
        status = "OK" if rc == 0 else "FAIL"
        print(f"  [{iid}] {info['gpu']:15} - {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--K", type=int, default=6)
    args = parser.parse_args()

    if args.test:
        test_workers()
    else:
        run_distributed_training(num_steps=args.steps, K=args.K)
