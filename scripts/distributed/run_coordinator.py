#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np

from holograd.core.config import (
    HoloGradConfig,
    ProtocolConfig,
    ADCConfig,
    VerificationConfig,
    AggregationConfig,
    DistributedConfig,
    TrainingConfig,
    LoggingConfig,
)
from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data
from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
from holograd.protocol.direction import ADCCodebook
from scripts.distributed.network import (
    CoordinatorServer,
    MSG_PARAMS,
    MSG_CODEBOOK,
    MSG_TASK,
    MSG_SHUTDOWN,
)


def run_coordinator(
    num_workers: int = 15,
    port: int = 5555,
    steps: int = 1000,
    batch_size: int = 4,
    seq_length: int = 64,
    lr: float = 1e-3,
    K: int = 16,
    adc_rank: int = 8,
    model_size: str = "small",
    dataset: str = "wikitext-2-raw-v1",
    device: str = "cuda",
):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("HoloGrad Distributed Coordinator")
    print("=" * 60)
    print(f"Workers expected: {num_workers}")
    print(f"Port: {port}")
    print(f"Steps: {steps}")
    print(f"K (directions): {K}")
    print(f"ADC rank: {adc_rank}")
    print(f"Device: {device}")
    print("-" * 60)

    print("[Coordinator] Loading dataset...")
    train_loader, val_loader, vocab_size = create_wikitext_data(
        seq_length=seq_length,
        batch_size=batch_size,
        dataset_name=dataset,
        max_train_samples=10000,
        max_val_samples=500,
    )

    print("[Coordinator] Creating model...")
    model = SimpleGPT2(size=model_size, max_seq_len=seq_length, vocab_size=vocab_size)
    model.to(device)
    print(f"[Coordinator] Model parameters: {model.num_parameters:,}")

    coord_config = CoordinatorConfig(
        dimension=model.num_parameters,
        num_workers=num_workers,
        proofs_per_step=K,
        use_adc=True,
        adc_rank=adc_rank,
        learning_rate=lr,
        device=device,
    )
    coordinator = Coordinator(coord_config)

    params = model.get_flat_params()
    coordinator.set_parameters(params)

    print("[Coordinator] Starting server...")
    server = CoordinatorServer(port=port)

    print(f"[Coordinator] Waiting for {num_workers} workers...")
    worker_ids = server.wait_for_workers(num_workers, timeout=600)
    print(f"[Coordinator] All {len(worker_ids)} workers connected!")

    print("[Coordinator] Broadcasting initial parameters...")
    server.broadcast_params(worker_ids, params)

    if coordinator.codebook is not None:
        codebook_data = {
            "U": coordinator.codebook.codebook,
            "rank": adc_rank,
            "dimension": model.num_parameters,
        }
        server.broadcast_codebook(worker_ids, codebook_data)

    print("-" * 60)
    print("[Coordinator] Starting training...")
    print("-" * 60)

    train_iter = iter(train_loader)
    losses = []

    for step in range(steps):
        step_start = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        coordinator.set_batch(input_ids.cpu().numpy().flatten(), step)

        tasks = coordinator.publish_tasks(step)

        worker_tasks = []
        for i, task in enumerate(tasks[:num_workers]):
            wid = worker_ids[i % len(worker_ids)]
            task_data = {
                "step": task.step,
                "seed": task.seed,
                "use_adc": task.use_adc,
                "input_ids": input_ids.cpu().numpy(),
                "labels": labels.cpu().numpy(),
            }
            worker_tasks.append((wid, task_data))

        server.send_tasks(worker_tasks)

        proofs_data = server.collect_proofs(len(worker_tasks), timeout=120)

        from holograd.core.types import Proof

        proofs = []
        for pd in proofs_data:
            proof = Proof(
                step=pd["step"],
                worker_id=pd["worker_id"],
                seed=pd["seed"],
                scalar=pd["scalar"],
                timestamp=pd["timestamp"],
                adc_projection=pd.get("adc_projection"),
            )
            proofs.append(proof)
            coordinator.collect_proof(proof)

        gradient, agg_result = coordinator.aggregate()
        new_params = coordinator.update_parameters(gradient)

        model.set_flat_params(new_params)

        with torch.no_grad():
            output = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.size(-1)),
                labels.view(-1),
            )

        losses.append(loss.item())
        step_time = time.time() - step_start

        if (step + 1) % 10 == 0:
            print(
                f"Step {step + 1}/{steps} | Loss: {loss.item():.4f} | "
                f"Time: {step_time:.2f}s | Proofs: {len(proofs)}"
            )

        if coordinator.codebook is not None and (step + 1) % 50 == 0:
            codebook_data = {
                "U": coordinator.codebook.codebook,
                "rank": adc_rank,
                "dimension": model.num_parameters,
            }
            server.broadcast_codebook(worker_ids, codebook_data)

    print("-" * 60)
    print("Training complete!")
    print(f"Initial loss: {np.mean(losses[:10]):.4f}")
    print(f"Final loss: {np.mean(losses[-10:]):.4f}")
    print("-" * 60)

    server.shutdown_workers(worker_ids)
    server.close()

    return losses


def main():
    parser = argparse.ArgumentParser(description="HoloGrad Distributed Coordinator")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--adc-rank", type=int, default=8)
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_coordinator(
        num_workers=args.workers,
        port=args.port,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        lr=args.lr,
        K=args.K,
        adc_rank=args.adc_rank,
        model_size=args.model_size,
        dataset=args.dataset,
        device=args.device,
    )


if __name__ == "__main__":
    main()
