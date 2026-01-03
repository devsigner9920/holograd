#!/usr/bin/env python3
"""
SSH-based distributed training for Vast.ai
Bypasses network port restrictions by using SSH for all communication.
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
    "29463765": {"host": "ssh7.vast.ai", "port": 23764, "gpu": "RTX_2060S"},
    "29463766": {"host": "ssh6.vast.ai", "port": 23766, "gpu": "Quadro_P4000"},
    "29463768": {"host": "ssh9.vast.ai", "port": 23768, "gpu": "GTX_1660_Ti"},
    "29463777": {"host": "ssh1.vast.ai", "port": 23776, "gpu": "RTX_4070"},
    "29463778": {"host": "ssh3.vast.ai", "port": 23778, "gpu": "RTX_4060_Ti"},
    "29463780": {"host": "ssh8.vast.ai", "port": 23780, "gpu": "RTX_4070S"},
    "29464653": {"host": "ssh9.vast.ai", "port": 24652, "gpu": "GTX_1080"},
    "29464654": {"host": "ssh6.vast.ai", "port": 24654, "gpu": "GTX_1080"},
    "29464657": {"host": "ssh4.vast.ai", "port": 24656, "gpu": "GTX_1080"},
}


def run_ssh_command(host: str, port: int, command: str, timeout: int = 300) -> Tuple[str, str, int]:
    """Execute command on remote instance via SSH."""
    ssh_cmd = [
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
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", -1
    except Exception as e:
        return "", str(e), -1


def compute_jvp_on_worker(
    instance_id: str,
    host: str,
    port: int,
    params_b64: str,
    input_ids_b64: str,
    labels_b64: str,
    seed: int,
    step: int,
    use_adc: bool,
    codebook_b64: str = None,
) -> Dict[str, Any]:
    """Send JVP computation task to a worker and get result."""

    # Create remote Python script
    remote_script = f'''
import sys
sys.path.insert(0, "/root/holograd/src")
import base64
import pickle
import time
import numpy as np
import torch

from holograd.training.model import SimpleGPT2
from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.gradient.jvp import compute_jvp_gradient_projection

# Decode data
params = pickle.loads(base64.b64decode("{params_b64}"))
input_ids = pickle.loads(base64.b64decode("{input_ids_b64}"))
labels = pickle.loads(base64.b64decode("{labels_b64}"))
seed = {seed}
step = {step}
use_adc = {use_adc}
codebook_data = pickle.loads(base64.b64decode("{codebook_b64}")) if "{codebook_b64}" else None

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = SimpleGPT2(size="small", max_seq_len=64, vocab_size=50257)
model.to(device)
model.set_flat_params(params)

# Generate direction
direction_gen = DirectionGenerator(model.num_parameters)
adc_projection = None

if use_adc and codebook_data is not None:
    adc = ADCCodebook(
        dimension=codebook_data["dimension"],
        rank=codebook_data["rank"],
        device=device,
    )
    adc._U = codebook_data["U"]
    result = adc.generate_direction(seed)
    direction = result.direction
    adc_projection = result.z_projection.tolist() if result.z_projection is not None else None
else:
    result = direction_gen.generate(seed)
    direction = result.direction

# Compute JVP
input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
labels_t = torch.tensor(labels, dtype=torch.long, device=device)

start = time.time()
scalar = compute_jvp_gradient_projection(
    model=model,
    input_ids=input_ids_t,
    labels=labels_t,
    direction=direction,
    device=device,
)
elapsed = time.time() - start

# Output result
result = {{
    "step": step,
    "seed": seed,
    "scalar": float(scalar),
    "adc_projection": adc_projection,
    "time": elapsed,
    "device": device,
}}
import json
print("RESULT:" + json.dumps(result))
'''

    # Execute on remote
    stdout, stderr, returncode = run_ssh_command(
        host,
        port,
        f"python3 -c '{remote_script}'",
        timeout=180,
    )

    if returncode != 0:
        return {
            "success": False,
            "error": stderr,
            "instance_id": instance_id,
            "step": step,
        }

    # Parse result
    for line in stdout.split("\n"):
        if line.startswith("RESULT:"):
            result = json.loads(line[7:])
            result["success"] = True
            result["instance_id"] = instance_id
            return result

    return {
        "success": False,
        "error": "No result found in output",
        "stdout": stdout,
        "instance_id": instance_id,
        "step": step,
    }


def run_distributed_training(
    num_steps: int = 100,
    batch_size: int = 4,
    seq_length: int = 64,
    lr: float = 1e-3,
    K: int = 16,
    adc_rank: int = 8,
    max_workers: int = 9,
):
    """Run distributed training using SSH-based communication."""
    import torch
    from holograd.training.model import SimpleGPT2
    from holograd.training.data import create_wikitext_data
    from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
    from holograd.core.types import Proof

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("HoloGrad SSH-Based Distributed Training")
    print("=" * 60)
    print(f"Workers: {len(WORKER_INSTANCES)}")
    print(f"Steps: {num_steps}")
    print(f"K (directions): {K}")
    print(f"ADC rank: {adc_rank}")
    print(f"Local device: {device}")
    print("-" * 60)

    # Load dataset locally
    print("[Local] Loading dataset...")
    train_loader, val_loader, vocab_size = create_wikitext_data(
        seq_length=seq_length,
        batch_size=batch_size,
        dataset_name="wikitext-2-raw-v1",
        max_train_samples=10000,
        max_val_samples=500,
    )

    # Create model locally
    print("[Local] Creating model...")
    model = SimpleGPT2(size="small", max_seq_len=seq_length, vocab_size=vocab_size)
    model.to(device)
    print(f"[Local] Model parameters: {model.num_parameters:,}")

    # Create coordinator
    coord_config = CoordinatorConfig(
        dimension=model.num_parameters,
        num_workers=len(WORKER_INSTANCES),
        proofs_per_step=K,
        use_adc=True,
        adc_rank=adc_rank,
        learning_rate=lr,
        device=device,
    )
    coordinator = Coordinator(coord_config)

    params = model.get_flat_params()
    coordinator.set_parameters(params)

    # Prepare workers list
    workers = list(WORKER_INSTANCES.items())

    print("-" * 60)
    print("[Local] Starting training...")
    print("-" * 60)

    train_iter = iter(train_loader)
    losses = []

    for step in range(num_steps):
        step_start = time.time()

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].numpy()
        labels = batch["labels"].numpy()

        # Prepare tasks
        coordinator.set_batch(input_ids.flatten(), step)
        tasks = coordinator.publish_tasks(step)

        # Encode data for transmission
        params_b64 = base64.b64encode(pickle.dumps(params)).decode()
        input_ids_b64 = base64.b64encode(pickle.dumps(input_ids)).decode()
        labels_b64 = base64.b64encode(pickle.dumps(labels)).decode()

        codebook_b64 = ""
        if coordinator.codebook is not None:
            codebook_data = {
                "U": coordinator.codebook.codebook,
                "rank": adc_rank,
                "dimension": model.num_parameters,
            }
            codebook_b64 = base64.b64encode(pickle.dumps(codebook_data)).decode()

        # Distribute tasks to workers
        proofs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for i, task in enumerate(tasks[: len(workers)]):
                instance_id, info = workers[i % len(workers)]

                future = executor.submit(
                    compute_jvp_on_worker,
                    instance_id,
                    info["host"],
                    info["port"],
                    params_b64,
                    input_ids_b64,
                    labels_b64,
                    task.seed,
                    step,
                    task.use_adc,
                    codebook_b64,
                )
                futures[future] = (instance_id, task)

            for future in as_completed(futures):
                instance_id, task = futures[future]
                try:
                    result = future.result()
                    if result.get("success"):
                        proof = Proof(
                            step=result["step"],
                            worker_id=int(instance_id),
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
                            f"  [!] Worker {instance_id} failed: {result.get('error', 'unknown')}"
                        )
                except Exception as e:
                    print(f"  [!] Worker {instance_id} exception: {e}")

        if len(proofs) < K // 2:
            print(f"Step {step}: Only {len(proofs)} proofs collected, skipping update")
            continue

        # Aggregate and update
        gradient, agg_result = coordinator.aggregate()
        new_params = coordinator.update_parameters(gradient)

        model.set_flat_params(new_params)
        params = new_params

        # Compute loss
        with torch.no_grad():
            input_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            output = model(input_t)
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.size(-1)),
                labels_t.view(-1),
            )

        losses.append(loss.item())
        step_time = time.time() - step_start

        if (step + 1) % 5 == 0:
            print(
                f"Step {step + 1}/{num_steps} | Loss: {loss.item():.4f} | "
                f"Time: {step_time:.2f}s | Proofs: {len(proofs)}"
            )

    print("-" * 60)
    print("Training complete!")
    print(f"Initial loss: {np.mean(losses[:10]) if len(losses) >= 10 else np.mean(losses):.4f}")
    print(f"Final loss: {np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses):.4f}")
    print("-" * 60)

    return losses


def test_workers():
    """Test connectivity to all workers."""
    print("Testing worker connectivity...")

    for instance_id, info in WORKER_INSTANCES.items():
        stdout, stderr, rc = run_ssh_command(
            info["host"],
            info["port"],
            'python3 -c "import torch; print(f\\"GPU: {torch.cuda.is_available()}\\")"',
            timeout=30,
        )
        status = "OK" if rc == 0 else "FAILED"
        gpu_status = stdout.strip() if rc == 0 else stderr[:50]
        print(
            f"  [{instance_id}] {info['gpu']:15s} @ {info['host']}:{info['port']} - {status} - {gpu_status}"
        )


def main():
    parser = argparse.ArgumentParser(description="SSH-based distributed training")
    parser.add_argument("--test", action="store_true", help="Test worker connectivity")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=9)
    parser.add_argument("--adc-rank", type=int, default=8)

    args = parser.parse_args()

    if args.test:
        test_workers()
        return

    run_distributed_training(
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        K=args.K,
        adc_rank=args.adc_rank,
        max_workers=len(WORKER_INSTANCES),
    )


if __name__ == "__main__":
    main()
