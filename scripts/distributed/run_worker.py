#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np

from holograd.training.model import SimpleGPT2
from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.gradient.jvp import compute_jvp_gradient_projection
from scripts.distributed.network import WorkerClient


def run_worker(
    coordinator_host: str,
    coordinator_port: int = 5555,
    worker_id: int = 0,
    model_size: str = "small",
    seq_length: int = 64,
    device: str = "cuda",
):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Worker {worker_id}] Starting...")
    print(f"[Worker {worker_id}] Coordinator: {coordinator_host}:{coordinator_port}")
    print(f"[Worker {worker_id}] Device: {device}")

    client = WorkerClient(coordinator_host, coordinator_port, worker_id)

    print(f"[Worker {worker_id}] Registering with coordinator...")
    if not client.register():
        print(f"[Worker {worker_id}] Failed to register!")
        return
    print(f"[Worker {worker_id}] Registered successfully!")

    model: SimpleGPT2 = None
    direction_gen: DirectionGenerator = None
    adc_codebook: ADCCodebook = None
    params: np.ndarray = None

    print(f"[Worker {worker_id}] Waiting for initial data...")

    while True:
        msg = client.receive(timeout=600)

        if msg is None:
            print(f"[Worker {worker_id}] Timeout waiting for message")
            continue

        if msg.msg_type == b"SHUTDOWN":
            print(f"[Worker {worker_id}] Received shutdown signal")
            break

        elif msg.msg_type == b"PARAMS":
            params = msg.payload
            print(f"[Worker {worker_id}] Received params: {params.shape}")

            if model is None:
                vocab_size = 50257
                model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
                model.to(device)
                direction_gen = DirectionGenerator(model.num_parameters)
                print(f"[Worker {worker_id}] Model initialized: {model.num_parameters:,} params")

            model.set_flat_params(params)

        elif msg.msg_type == b"CODEBOOK":
            codebook_data = msg.payload
            print(f"[Worker {worker_id}] Received codebook update")

            if adc_codebook is None:
                adc_codebook = ADCCodebook(
                    dimension=codebook_data["dimension"],
                    rank=codebook_data["rank"],
                    device=device,
                )

            adc_codebook._U = codebook_data["U"]
            adc_codebook._invalidate_gpu_cache()

        elif msg.msg_type == b"TASK":
            task = msg.payload
            step = task["step"]
            seed = task["seed"]
            use_adc = task["use_adc"]
            input_ids = task["input_ids"]
            labels = task["labels"]

            import torch

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            adc_projection = None
            if use_adc and adc_codebook is not None:
                result = adc_codebook.generate_direction(seed)
                direction = result.direction
                adc_projection = result.z_projection
            else:
                result = direction_gen.generate(seed)
                direction = result.direction

            start_time = time.time()
            scalar = compute_jvp_gradient_projection(
                model=model,
                input_ids=input_ids_t,
                labels=labels_t,
                direction=direction,
                device=device,
            )
            jvp_time = time.time() - start_time

            proof_data = {
                "step": step,
                "worker_id": worker_id,
                "seed": seed,
                "scalar": scalar,
                "timestamp": time.time(),
                "adc_projection": adc_projection,
            }

            client.send_proof(proof_data)

            if step % 50 == 0:
                print(
                    f"[Worker {worker_id}] Step {step} | JVP: {jvp_time:.2f}s | Scalar: {scalar:.6f}"
                )

    client.close()
    print(f"[Worker {worker_id}] Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="HoloGrad Distributed Worker")
    parser.add_argument("--coordinator", type=str, required=True, help="Coordinator host IP")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_worker(
        coordinator_host=args.coordinator,
        coordinator_port=args.port,
        worker_id=args.worker_id,
        model_size=args.model_size,
        seq_length=args.seq_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
