#!/usr/bin/env python3
import argparse
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
    "29473291": "http://113.201.14.131:43698",
    "29473292": "http://1.208.108.242:30473",
    "29473293": "http://1.208.108.242:63837",
    "29473294": "http://1.208.108.242:61803",
    "29473295": "http://70.68.84.2:46439",
    "29473296": "http://14.187.66.74:10181",
    "29473308": "http://142.170.89.112:23501",
    "29473314": "http://142.170.89.112:32222",
    "29473316": "http://171.101.231.208:50563",
}


class WorkerManager:
    def __init__(self):
        self.session = requests.Session()

    def check_health(self, url: str) -> bool:
        try:
            resp = self.session.get(f"{url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def init_worker(
        self,
        url: str,
        n_layer: int,
        n_head: int,
        n_embd: int,
        vocab_size: int,
        seq_length: int,
        seed: int,
        adc_rank: int,
        codebook_seed: Optional[int],
    ) -> bool:
        try:
            resp = self.session.post(
                f"{url}/init",
                json={
                    "model_size": "custom",
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "n_embd": n_embd,
                    "vocab_size": vocab_size,
                    "seq_length": seq_length,
                    "seed": seed,
                    "adc_rank": adc_rank,
                    "codebook_seed": codebook_seed,
                },
                timeout=60,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"    Init error: {e}")
            return False

    def update_codebook(self, url: str, U: List[List[float]], rank: int, dimension: int) -> bool:
        try:
            resp = self.session.post(
                f"{url}/update_codebook",
                json={"U": U, "rank": rank, "dimension": dimension},
                timeout=30,
            )
            return resp.status_code == 200
        except:
            return False

    def apply_seed_gradient(
        self, url: str, seeds: List[str], scalars: List[float], lr: float, use_adc: bool
    ) -> bool:
        try:
            resp = self.session.post(
                f"{url}/apply_seed_gradient",
                json={"seeds": seeds, "scalars": scalars, "learning_rate": lr, "use_adc": use_adc},
                timeout=30,
            )
            return resp.status_code == 200
        except:
            return False

    def compute_task(
        self, url: str, step: int, seed: bytes, use_adc: bool, input_ids: List, labels: List
    ) -> Optional[dict]:
        try:
            seed_hex = seed.hex() if isinstance(seed, bytes) else str(seed)
            resp = self.session.post(
                f"{url}/compute",
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
        except Exception as e:
            print(f"  Task error: {e}")
        return None


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
    print("HoloGrad Distributed Training - Direct IP Mode")
    print("=" * 70)
    print(f"\nModel: {n_layer}L-{n_head}H-{n_embd}E, seq_length={seq_length}")
    print(f"Training: {num_steps} steps, K={K}, lr={learning_rate}")

    print("\n[1] Checking worker health...")
    healthy_workers = []
    for wid, url in WORKERS.items():
        if manager.check_health(url):
            healthy_workers.append((wid, url))
            print(f"  ✓ {wid}")
        else:
            print(f"  ✗ {wid}")

    min_workers = 3
    if len(healthy_workers) < min_workers:
        print(f"Not enough healthy workers ({len(healthy_workers)} < {min_workers})")
        return

    print(f"\n[2] Loading WikiText-2 data...")
    train_loader, val_loader, vocab_size = create_wikitext_data(
        seq_length=seq_length,
        batch_size=batch_size,
        dataset_name="wikitext-2-raw-v1",
        max_train_samples=max_train_samples,
        max_val_samples=500,
    )
    print(f"  vocab_size={vocab_size}, train_batches={len(train_loader)}")

    print(f"\n[3] Creating model...")
    model_seed = 42
    codebook_seed = 12345
    adc_rank = 16
    model = SimpleGPT2(
        size="custom",
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=vocab_size,
        max_seq_len=seq_length,
        seed=model_seed,
    )
    print(f"  Parameters: {model.num_parameters:,}")

    print(f"\n[4] Initializing {len(healthy_workers)} workers...")
    initialized_workers = []
    for wid, url in healthy_workers:
        ok = manager.init_worker(
            url,
            n_layer,
            n_head,
            n_embd,
            vocab_size,
            seq_length,
            model_seed,
            adc_rank,
            codebook_seed,
        )
        if ok:
            initialized_workers.append((wid, url))
            print(f"  {wid}: OK")
        else:
            print(f"  {wid}: FAILED")

    if len(initialized_workers) < min_workers:
        print(f"Not enough initialized workers")
        return

    healthy_workers = initialized_workers

    print(f"\n[5] Creating coordinator...")
    coord_config = CoordinatorConfig(
        dimension=model.num_parameters,
        num_workers=len(healthy_workers),
        proofs_per_step=K,
        use_adc=True,
        adc_rank=adc_rank,
        adc_warmup_samples=20,
        learning_rate=learning_rate,
        device="cpu",
    )
    coordinator = Coordinator(coord_config)
    coordinator.set_parameters(model.get_flat_params())

    print(f"\n[6] Starting training ({num_steps} steps)...")
    print("-" * 70)

    train_iter = iter(train_loader)
    losses = []
    start_time = time.time()
    codebook_synced = False

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
        with ThreadPoolExecutor(max_workers=len(healthy_workers) * 2) as executor:
            futures = {}
            for i, task in enumerate(tasks):
                wid, url = healthy_workers[i % len(healthy_workers)]
                future = executor.submit(
                    manager.compute_task, url, step, task.seed, task.use_adc, input_ids, labels
                )
                futures[future] = (wid, task.seed)

            for future in as_completed(futures):
                result = future.result()
                if result:
                    wid, _ = futures[future]
                    proof = Proof(
                        step=result["step"],
                        worker_id=int(wid),
                        seed=bytes.fromhex(result["seed"]),
                        scalar=result["scalar"],
                        timestamp=time.time(),
                        adc_projection=np.array(result["adc_projection"])
                        if result.get("adc_projection")
                        else None,
                    )
                    proofs.append(proof)
                    coordinator.collect_proof(proof)

        if len(proofs) < min_workers:
            print(f"Step {step}: Only {len(proofs)} proofs, skipping")
            continue

        gradient, agg_result = coordinator.aggregate()
        new_params = coordinator.update_parameters(gradient)
        model.set_flat_params(new_params)

        if (
            coordinator.codebook is not None
            and coordinator.codebook.is_warmed_up
            and not codebook_synced
        ):
            print(f"\n  [ADC Warmup Complete] Syncing codebook...")
            U = coordinator.codebook.codebook.tolist()
            cb_ok = sum(
                1
                for wid, url in healthy_workers
                if manager.update_codebook(url, U, adc_rank, model.num_parameters)
            )
            print(f"  Synced to {cb_ok}/{len(healthy_workers)} workers\n")
            codebook_synced = True

        use_adc_for_sync = codebook_synced and coordinator.codebook is not None
        seeds_hex = [p.seed.hex() for p in proofs]
        scalars_list = [p.scalar for p in proofs]

        with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
            list(
                executor.map(
                    lambda w: manager.apply_seed_gradient(
                        w[1], seeds_hex, scalars_list, learning_rate, use_adc_for_sync
                    ),
                    healthy_workers,
                )
            )

        loss_val = model.compute_loss(batch.input_ids, batch.labels)
        losses.append(loss_val)

        if (step + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step + 1:3d}/{num_steps} | Loss: {loss_val:.4f} | "
                f"Time: {time.time() - step_start:.1f}s | Proofs: {len(proofs)} | Total: {elapsed:.0f}s"
            )

    print("-" * 70)
    if losses:
        print(f"Training complete! Final loss: {np.mean(losses[-5:]):.4f}")
        print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=5000)
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
