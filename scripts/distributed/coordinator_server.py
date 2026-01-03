#!/usr/bin/env python3
import asyncio
import sys
import threading
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

import numpy as np
import requests
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data
from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
from holograd.core.types import Proof

app = FastAPI(title="HoloGrad Coordinator")

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


@dataclass
class WorkerStatus:
    id: str
    url: str
    healthy: bool = False
    initialized: bool = False
    tasks_completed: int = 0
    last_task_time: float = 0.0
    last_scalar: float = 0.0
    errors: int = 0
    busy: bool = False


@dataclass
class StepDetail:
    step: int
    loss: float
    proofs_collected: int
    proofs_needed: int
    gradient_norm: float
    scalars: List[float]
    worker_times: Dict[str, float]
    step_time: float
    phase: str


@dataclass
class TrainingState:
    running: bool = False
    phase: str = "idle"
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    losses: List[float] = field(default_factory=list)
    start_time: float = 0.0
    elapsed: float = 0.0
    step_time: float = 0.0
    proofs_count: int = 0

    workers: Dict[str, WorkerStatus] = field(default_factory=dict)

    adc_warmup_current: int = 0
    adc_warmup_needed: int = 20
    adc_warmed_up: bool = False
    codebook_synced: bool = False

    gradient_norms: List[float] = field(default_factory=list)
    scalar_stats: Dict[str, float] = field(default_factory=dict)

    recent_steps: deque = field(default_factory=lambda: deque(maxlen=10))

    error: Optional[str] = None
    stop_requested: bool = False

    model_params: int = 0
    K: int = 0
    learning_rate: float = 0.0


state = TrainingState()
state_lock = threading.Lock()


class TrainRequest(BaseModel):
    steps: int = 100
    K: int = 8
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 64
    seq_length: int = 32
    batch_size: int = 4
    lr: float = 0.1
    max_samples: int = 5000
    use_adc: bool = True  # Set to False to disable ADC and use random directions only


class WorkerClient:
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
        codebook_seed: int,
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
        self,
        url: str,
        worker_id: str,
        step: int,
        seed: bytes,
        use_adc: bool,
        input_ids: List,
        labels: List,
    ) -> Optional[dict]:
        try:
            seed_hex = seed.hex() if isinstance(seed, bytes) else str(seed)
            start = time.time()
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
            elapsed = time.time() - start
            if resp.status_code == 200:
                result = resp.json()
                result["worker_time"] = elapsed
                result["worker_id"] = worker_id
                return result
        except:
            pass
        return None


def run_training(config: TrainRequest):
    global state
    client = WorkerClient()

    with state_lock:
        state = TrainingState()
        state.running = True
        state.phase = "initializing"
        state.total_steps = config.steps
        state.start_time = time.time()
        state.K = config.K
        state.learning_rate = config.lr
        state.adc_warmup_needed = 20

    try:
        with state_lock:
            state.phase = "checking_workers"

        healthy_workers = []
        for wid, url in WORKERS.items():
            ws = WorkerStatus(id=wid, url=url)
            ws.healthy = client.check_health(url)
            if ws.healthy:
                healthy_workers.append((wid, url))
            with state_lock:
                state.workers[wid] = ws

        if len(healthy_workers) < 3:
            with state_lock:
                state.error = f"Not enough workers: {len(healthy_workers)}/3 minimum"
                state.running = False
            return

        with state_lock:
            state.phase = "loading_data"

        train_loader, _, vocab_size = create_wikitext_data(
            seq_length=config.seq_length,
            batch_size=config.batch_size,
            dataset_name="wikitext-2-raw-v1",
            max_train_samples=config.max_samples,
            max_val_samples=100,
        )

        with state_lock:
            state.phase = "creating_model"

        model_seed, codebook_seed, adc_rank = 42, 12345, 16
        model = SimpleGPT2(
            size="custom",
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            vocab_size=vocab_size,
            max_seq_len=config.seq_length,
            seed=model_seed,
        )

        with state_lock:
            state.model_params = model.num_parameters

        with state_lock:
            state.phase = "initializing_workers"

        for wid, url in healthy_workers:
            ok = client.init_worker(
                url,
                config.n_layer,
                config.n_head,
                config.n_embd,
                vocab_size,
                config.seq_length,
                model_seed,
                adc_rank,
                codebook_seed,
            )
            with state_lock:
                state.workers[wid].initialized = ok

        with state_lock:
            state.phase = "creating_coordinator"

        coord_config = CoordinatorConfig(
            dimension=model.num_parameters,
            num_workers=len(healthy_workers),
            proofs_per_step=config.K,
            use_adc=config.use_adc,
            adc_rank=adc_rank,
            adc_warmup_samples=20 if config.use_adc else 0,
            learning_rate=config.lr,
            device="cpu",
        )
        coordinator = Coordinator(coord_config)
        coordinator.set_parameters(model.get_flat_params())

        train_iter = iter(train_loader)
        codebook_synced = False

        for step in range(config.steps):
            with state_lock:
                if state.stop_requested:
                    state.phase = "stopped"
                    state.running = False
                    return
                state.phase = "adc_warmup" if not codebook_synced else "training"

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

            with state_lock:
                state.phase = f"step_{step}_computing"
                for wid in state.workers:
                    state.workers[wid].busy = False

            proofs = []
            worker_times = {}
            scalars_this_step = []

            with ThreadPoolExecutor(max_workers=len(healthy_workers) * 2) as executor:
                futures = {}
                for i, task in enumerate(tasks):
                    wid, url = healthy_workers[i % len(healthy_workers)]
                    with state_lock:
                        state.workers[wid].busy = True
                    future = executor.submit(
                        client.compute_task,
                        url,
                        wid,
                        step,
                        task.seed,
                        task.use_adc,
                        input_ids,
                        labels,
                    )
                    futures[future] = (wid, task.seed)

                for future in as_completed(futures):
                    result = future.result()
                    wid, _ = futures[future]

                    with state_lock:
                        state.workers[wid].busy = False

                    if result:
                        with state_lock:
                            state.workers[wid].tasks_completed += 1
                            state.workers[wid].last_task_time = result.get("worker_time", 0)
                            state.workers[wid].last_scalar = result["scalar"]

                        worker_times[wid] = result.get("worker_time", 0)
                        scalars_this_step.append(result["scalar"])

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
                    else:
                        with state_lock:
                            state.workers[wid].errors += 1

            if len(proofs) < 3:
                with state_lock:
                    state.phase = f"step_{step}_insufficient_proofs"
                continue

            with state_lock:
                state.phase = f"step_{step}_aggregating"

            gradient, agg_result = coordinator.aggregate()
            gradient_norm = float(np.linalg.norm(gradient))

            # Log aggregation details
            logger.info(f"[Step {step}] Aggregated {len(proofs)} proofs")
            logger.info(f"[Step {step}] Gradient norm: {gradient_norm:.6f}")
            logger.info(
                f"[Step {step}] Scalars: mean={np.mean(scalars_this_step):.6f}, std={np.std(scalars_this_step):.6f}"
            )

            with state_lock:
                state.phase = f"step_{step}_updating"

            new_params = coordinator.update_parameters(gradient)
            model.set_flat_params(new_params)

            if coordinator.codebook and coordinator.codebook.is_warmed_up and not codebook_synced:
                # Sync learned codebook to all workers
                with state_lock:
                    state.phase = f"step_{step}_syncing_codebook"

                logger.info(f"[Step {step}] ADC warmup complete! Syncing codebook to workers...")
                codebook_U = coordinator.codebook.codebook  # rank x dimension matrix
                logger.info(
                    f"[Step {step}] Codebook shape: {codebook_U.shape}, size: {codebook_U.nbytes / 1024 / 1024:.1f}MB"
                )

                # Send codebook to each worker using base64-encoded binary for efficiency
                import base64

                codebook_bytes = codebook_U.astype(np.float16).tobytes()
                codebook_b64 = base64.b64encode(codebook_bytes).decode("ascii")
                logger.info(
                    f"[Step {step}] Codebook encoded size: {len(codebook_b64) / 1024 / 1024:.1f}MB"
                )

                with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:

                    def sync_codebook(worker_info):
                        wid, url = worker_info
                        try:
                            resp = client.session.post(
                                f"{url}/update_codebook_b64",
                                json={
                                    "U_b64": codebook_b64,
                                    "rank": coordinator.codebook.rank,
                                    "dimension": coordinator.codebook.dimension,
                                    "dtype": "float16",
                                },
                                timeout=300,
                            )
                            return resp.status_code == 200
                        except Exception as e:
                            logger.error(f"[Step {step}] Failed to sync codebook to {wid}: {e}")
                            return False

                    results = list(executor.map(sync_codebook, healthy_workers))
                    sync_count = sum(results)
                    logger.info(
                        f"[Step {step}] Codebook synced to {sync_count}/{len(healthy_workers)} workers"
                    )

                codebook_synced = True
                with state_lock:
                    state.codebook_synced = True
                    state.adc_warmed_up = True
                    state.adc_warmup_current = state.adc_warmup_needed

            with state_lock:
                if not codebook_synced and coordinator.codebook:
                    state.adc_warmup_current = min(step + 1, state.adc_warmup_needed)

            use_adc = codebook_synced and coordinator.codebook is not None
            seeds_hex = [p.seed.hex() for p in proofs]
            scalars_list = [p.scalar for p in proofs]

            with state_lock:
                state.phase = f"step_{step}_syncing"

            with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
                list(
                    executor.map(
                        lambda w: client.apply_seed_gradient(
                            w[1], seeds_hex, scalars_list, config.lr, use_adc
                        ),
                        healthy_workers,
                    )
                )

            loss_val = model.compute_loss(batch.input_ids, batch.labels)
            step_time = time.time() - step_start

            logger.info(f"[Step {step}] Loss: {loss_val:.4f}, Step time: {step_time:.2f}s")
            logger.info(
                f"[Step {step}] ADC warmed up: {codebook_synced}, use_adc: {config.use_adc}"
            )

            step_detail = StepDetail(
                step=step,
                loss=loss_val,
                proofs_collected=len(proofs),
                proofs_needed=config.K,
                gradient_norm=gradient_norm,
                scalars=scalars_this_step,
                worker_times=worker_times,
                step_time=step_time,
                phase="warmup" if not codebook_synced else "training",
            )

            with state_lock:
                state.step = step + 1
                state.loss = loss_val
                state.losses.append(loss_val)
                state.step_time = step_time
                state.elapsed = time.time() - state.start_time
                state.proofs_count = len(proofs)
                state.gradient_norms.append(gradient_norm)
                state.recent_steps.append(
                    {
                        "step": step,
                        "loss": round(loss_val, 4),
                        "gradient_norm": round(gradient_norm, 6),
                        "proofs": len(proofs),
                        "step_time": round(step_time, 2),
                        "scalars_mean": round(np.mean(scalars_this_step), 6)
                        if scalars_this_step
                        else 0,
                        "scalars_std": round(np.std(scalars_this_step), 6)
                        if scalars_this_step
                        else 0,
                    }
                )

                if scalars_this_step:
                    state.scalar_stats = {
                        "mean": round(float(np.mean(scalars_this_step)), 6),
                        "std": round(float(np.std(scalars_this_step)), 6),
                        "min": round(float(np.min(scalars_this_step)), 6),
                        "max": round(float(np.max(scalars_this_step)), 6),
                    }

        with state_lock:
            state.phase = "completed"
            state.running = False

    except Exception as e:
        import traceback

        with state_lock:
            state.error = f"{str(e)}\n{traceback.format_exc()}"
            state.phase = "error"
            state.running = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def start_training(config: TrainRequest, background_tasks: BackgroundTasks):
    global state
    with state_lock:
        if state.running:
            return {"error": "Training already running", "phase": state.phase}
    background_tasks.add_task(run_training, config)
    return {"status": "started", "config": config.model_dump()}


@app.get("/status")
def get_status():
    with state_lock:
        avg_loss = np.mean(state.losses[-10:]) if state.losses else 0.0
        avg_grad_norm = np.mean(state.gradient_norms[-10:]) if state.gradient_norms else 0.0

        workers_summary = {}
        for wid, ws in state.workers.items():
            workers_summary[wid] = {
                "healthy": ws.healthy,
                "initialized": ws.initialized,
                "tasks": ws.tasks_completed,
                "errors": ws.errors,
                "busy": ws.busy,
                "last_time": round(ws.last_task_time, 2),
                "last_scalar": round(ws.last_scalar, 6),
            }

        return {
            "running": state.running,
            "phase": state.phase,
            "step": state.step,
            "total_steps": state.total_steps,
            "progress_pct": round(state.step / state.total_steps * 100, 1)
            if state.total_steps
            else 0,
            "loss": round(state.loss, 4),
            "avg_loss_10": round(float(avg_loss), 4),
            "loss_trend": "↓"
            if len(state.losses) > 5 and np.mean(state.losses[-5:]) < np.mean(state.losses[-10:-5])
            else "→",
            "step_time": round(state.step_time, 2),
            "elapsed": round(state.elapsed, 1),
            "eta": round((state.total_steps - state.step) * state.step_time, 0)
            if state.step_time > 0
            else 0,
            "proofs": state.proofs_count,
            "K": state.K,
            "adc": {
                "warmed_up": state.adc_warmed_up,
                "warmup_progress": f"{state.adc_warmup_current}/{state.adc_warmup_needed}",
                "codebook_synced": state.codebook_synced,
            },
            "gradient": {
                "norm": round(state.gradient_norms[-1], 6) if state.gradient_norms else 0,
                "avg_norm": round(float(avg_grad_norm), 6),
            },
            "scalars": state.scalar_stats,
            "model_params": state.model_params,
            "learning_rate": state.learning_rate,
            "error": state.error,
        }


@app.get("/workers")
def get_workers():
    with state_lock:
        result = {}
        for wid, ws in state.workers.items():
            result[wid] = {
                "url": ws.url,
                "healthy": ws.healthy,
                "initialized": ws.initialized,
                "tasks_completed": ws.tasks_completed,
                "errors": ws.errors,
                "busy": ws.busy,
                "last_task_time": round(ws.last_task_time, 3),
                "last_scalar": round(ws.last_scalar, 6),
            }
        return result


@app.get("/steps")
def get_recent_steps():
    with state_lock:
        return {"recent_steps": list(state.recent_steps)}


@app.get("/losses")
def get_losses():
    with state_lock:
        return {
            "losses": state.losses,
            "gradient_norms": state.gradient_norms,
        }


@app.get("/analysis")
def get_analysis():
    with state_lock:
        if len(state.losses) < 10:
            return {"message": "Not enough data yet"}

        losses = np.array(state.losses)
        grad_norms = np.array(state.gradient_norms) if state.gradient_norms else np.array([])

        return {
            "loss_analysis": {
                "initial": round(float(losses[0]), 4) if len(losses) > 0 else 0,
                "current": round(float(losses[-1]), 4) if len(losses) > 0 else 0,
                "min": round(float(np.min(losses)), 4),
                "max": round(float(np.max(losses)), 4),
                "mean": round(float(np.mean(losses)), 4),
                "std": round(float(np.std(losses)), 4),
                "improvement": round(float(losses[0] - losses[-1]), 4) if len(losses) > 0 else 0,
                "improvement_pct": round(float((losses[0] - losses[-1]) / losses[0] * 100), 2)
                if len(losses) > 0 and losses[0] != 0
                else 0,
            },
            "gradient_analysis": {
                "mean_norm": round(float(np.mean(grad_norms)), 6) if len(grad_norms) > 0 else 0,
                "std_norm": round(float(np.std(grad_norms)), 6) if len(grad_norms) > 0 else 0,
                "min_norm": round(float(np.min(grad_norms)), 6) if len(grad_norms) > 0 else 0,
                "max_norm": round(float(np.max(grad_norms)), 6) if len(grad_norms) > 0 else 0,
            },
            "convergence": {
                "is_converging": bool(
                    len(losses) > 20 and np.mean(losses[-10:]) < np.mean(losses[:10])
                ),
                "loss_variance_recent": round(float(np.var(losses[-10:])), 6)
                if len(losses) >= 10
                else 0,
            },
            "explanation": {
                "why_loss_high": "Random projections capture only ~0.0002% of gradient energy in 3M dimensions. ADC warmup helps focus on important subspace.",
                "why_slow_convergence": f"With K={state.K} proofs and D={state.model_params} params, effective SNR is low. Need more K or smaller model.",
                "adc_status": "ADC is warmed up - using learned subspace"
                if state.adc_warmed_up
                else f"ADC warming up ({state.adc_warmup_current}/{state.adc_warmup_needed}). Using random directions.",
            },
        }


@app.post("/stop")
def stop_training():
    global state
    with state_lock:
        if not state.running:
            return {"status": "not running"}
        state.stop_requested = True
    return {"status": "stop requested"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
