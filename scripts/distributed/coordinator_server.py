#!/usr/bin/env python3
"""
HoloGrad Coordinator Server - Production-ready with stability improvements.

Features:
- Dynamic worker discovery from Vast.ai
- Automatic retry with exponential backoff
- Checkpoint saving and resumption
- Graceful degradation on worker failures
- Detailed progress monitoring
"""

import asyncio
import sys
import threading
import time
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from collections import deque
import subprocess

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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from holograd.training.model import SimpleGPT2
from holograd.training.data import create_wikitext_data
from holograd.distributed.coordinator import Coordinator, CoordinatorConfig
from holograd.core.types import Proof

app = FastAPI(title="HoloGrad Coordinator")

# Will be populated dynamically
WORKERS: Dict[str, str] = {}


def discover_workers() -> Dict[str, str]:
    """Discover workers from Vast.ai instances."""
    workers = {}
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            instances = json.loads(result.stdout)
            for inst in instances:
                if inst.get("actual_status") == "running":
                    instance_id = str(inst["id"])
                    # Get direct HTTP access info
                    public_ip = inst.get("public_ipaddr")
                    ports = inst.get("ports", {})
                    # Find the mapped port for 8000
                    for port_mapping in ports.values():
                        if port_mapping.get("PrivatePort") == 8000:
                            host_port = port_mapping.get("PublicPort")
                            if public_ip and host_port:
                                workers[instance_id] = f"http://{public_ip}:{host_port}"
                                break
                    # Fallback: try direct_port_end if available
                    if instance_id not in workers:
                        direct_port = inst.get("direct_port_end")
                        if public_ip and direct_port:
                            workers[instance_id] = f"http://{public_ip}:{direct_port}"
            logger.info(f"Discovered {len(workers)} workers from Vast.ai")
    except Exception as e:
        logger.warning(f"Failed to discover workers: {e}")
    return workers


def load_workers_from_env() -> Dict[str, str]:
    """Load workers from environment variable or file."""
    workers = {}

    # Try environment variable first
    workers_json = os.environ.get("HOLOGRAD_WORKERS")
    if workers_json:
        try:
            workers = json.loads(workers_json)
            logger.info(f"Loaded {len(workers)} workers from environment")
            return workers
        except:
            pass

    # Try workers.json file
    workers_file = Path(__file__).parent / "workers.json"
    if workers_file.exists():
        try:
            with open(workers_file) as f:
                workers = json.load(f)
            logger.info(f"Loaded {len(workers)} workers from {workers_file}")
            return workers
        except:
            pass

    return workers


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
    consecutive_errors: int = 0
    busy: bool = False
    disabled: bool = False


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

    # Checkpoint info
    checkpoint_dir: str = ""
    last_checkpoint_step: int = 0


state = TrainingState()
state_lock = threading.Lock()

# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


class TrainRequest(BaseModel):
    steps: int = 100
    K: int = 64
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    seq_length: int = 64
    batch_size: int = 8
    lr: float = 0.01
    max_samples: int = 10000
    use_adc: bool = True
    adc_rank: int = 32
    checkpoint_every: int = 50
    resume_from: Optional[str] = None


class WorkerClient:
    def __init__(self, timeout: int = 30, retries: int = 3):
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.timeout = timeout

    def check_health(self, url: str) -> Tuple[bool, Optional[dict]]:
        try:
            resp = self.session.get(f"{url}/health", timeout=10)
            if resp.status_code == 200:
                return True, resp.json()
            return False, None
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False, None

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
    ) -> Tuple[bool, Optional[dict]]:
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
                timeout=600,
            )
            if resp.status_code == 200:
                return True, resp.json()
            return False, None
        except Exception as e:
            logger.error(f"Init failed for {url}: {e}")
            return False, None

    def apply_seed_gradient(
        self, url: str, seeds: List[str], scalars: List[float], lr: float, use_adc: bool
    ) -> bool:
        try:
            resp = self.session.post(
                f"{url}/apply_seed_gradient",
                json={"seeds": seeds, "scalars": scalars, "learning_rate": lr, "use_adc": use_adc},
                timeout=60,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Apply gradient failed for {url}: {e}")
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
        timeout: int = 180,
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
                timeout=timeout,
            )
            elapsed = time.time() - start
            if resp.status_code == 200:
                result = resp.json()
                result["worker_time"] = elapsed
                result["worker_id"] = worker_id
                return result
            else:
                logger.warning(f"Compute failed for {worker_id}: status={resp.status_code}")
        except Exception as e:
            logger.warning(f"Compute failed for {worker_id}: {e}")
        return None

    def sync_codebook(
        self,
        url: str,
        codebook_b64: str,
        rank: int,
        dimension: int,
        scale: float,
        timeout: int = 120,
    ) -> bool:
        try:
            resp = self.session.post(
                f"{url}/update_codebook_b64",
                json={
                    "U_b64": codebook_b64,
                    "rank": rank,
                    "dimension": dimension,
                    "dtype": "int8",
                    "scale": scale,
                },
                timeout=timeout,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Codebook sync failed for {url}: {e}")
            return False


def save_checkpoint(
    step: int,
    model: SimpleGPT2,
    coordinator: Coordinator,
    losses: List[float],
    config: TrainRequest,
) -> str:
    """Save training checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_step_{step}.npz"

    np.savez(
        checkpoint_path,
        step=step,
        params=model.get_flat_params(),
        losses=np.array(losses),
        config=json.dumps(config.model_dump()),
        codebook=coordinator.codebook.codebook if coordinator.codebook else None,
        codebook_warmed_up=coordinator.codebook.is_warmed_up if coordinator.codebook else False,
    )

    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return str(checkpoint_path)


def load_checkpoint(path: str) -> dict:
    """Load training checkpoint."""
    data = np.load(path, allow_pickle=True)
    return {
        "step": int(data["step"]),
        "params": data["params"],
        "losses": data["losses"].tolist(),
        "config": json.loads(str(data["config"])),
        "codebook": data["codebook"] if data["codebook"] is not None else None,
        "codebook_warmed_up": bool(data["codebook_warmed_up"]),
    }


def get_healthy_workers(workers: Dict[str, WorkerStatus]) -> List[Tuple[str, str]]:
    """Get list of healthy, non-disabled workers."""
    return [(wid, ws.url) for wid, ws in workers.items() if ws.healthy and not ws.disabled]


def run_training(config: TrainRequest):
    global state, WORKERS
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
        state.checkpoint_dir = str(CHECKPOINT_DIR)

    try:
        # Discover workers
        with state_lock:
            state.phase = "discovering_workers"

        WORKERS = load_workers_from_env()
        if not WORKERS:
            WORKERS = discover_workers()

        if not WORKERS:
            with state_lock:
                state.error = "No workers found. Set HOLOGRAD_WORKERS env or create workers.json"
                state.running = False
            return

        # Health check workers
        with state_lock:
            state.phase = "checking_workers"

        for wid, url in WORKERS.items():
            ws = WorkerStatus(id=wid, url=url)
            healthy, info = client.check_health(url)
            ws.healthy = healthy
            if info:
                logger.info(
                    f"Worker {wid}: healthy, device={info.get('device')}, memory={info.get('memory')}"
                )
            with state_lock:
                state.workers[wid] = ws

        healthy_workers = get_healthy_workers(state.workers)

        if len(healthy_workers) < 2:
            with state_lock:
                state.error = f"Not enough workers: {len(healthy_workers)}/2 minimum"
                state.running = False
            return

        logger.info(f"Found {len(healthy_workers)} healthy workers")

        # Load data
        with state_lock:
            state.phase = "loading_data"

        train_loader, _, vocab_size = create_wikitext_data(
            seq_length=config.seq_length,
            batch_size=config.batch_size,
            dataset_name="wikitext-2-raw-v1",
            max_train_samples=config.max_samples,
            max_val_samples=100,
        )

        # Create model
        with state_lock:
            state.phase = "creating_model"

        model_seed, codebook_seed = 42, 12345
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

        logger.info(f"Model created: {model.num_parameters:,} parameters")

        # Resume from checkpoint if specified
        start_step = 0
        codebook_synced = False

        if config.resume_from:
            try:
                checkpoint = load_checkpoint(config.resume_from)
                model.set_flat_params(checkpoint["params"])
                start_step = checkpoint["step"]
                state.losses = checkpoint["losses"]
                codebook_synced = checkpoint["codebook_warmed_up"]
                logger.info(f"Resumed from checkpoint: step={start_step}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

        # Initialize workers
        with state_lock:
            state.phase = "initializing_workers"

        init_results = []
        with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
            futures = {}
            for wid, url in healthy_workers:
                future = executor.submit(
                    client.init_worker,
                    url,
                    config.n_layer,
                    config.n_head,
                    config.n_embd,
                    vocab_size,
                    config.seq_length,
                    model_seed,
                    config.adc_rank,
                    codebook_seed,
                )
                futures[future] = wid

            for future in as_completed(futures, timeout=180):
                wid = futures[future]
                try:
                    ok, info = future.result()
                    with state_lock:
                        state.workers[wid].initialized = ok
                    if ok:
                        init_results.append(wid)
                        logger.info(f"Worker {wid} initialized: {info}")
                    else:
                        logger.warning(f"Worker {wid} failed to initialize")
                except Exception as e:
                    logger.error(f"Worker {wid} init error: {e}")

        # Update healthy workers list
        healthy_workers = [(wid, state.workers[wid].url) for wid in init_results]

        if len(healthy_workers) < 2:
            with state_lock:
                state.error = f"Not enough initialized workers: {len(healthy_workers)}"
                state.running = False
            return

        # Create coordinator
        with state_lock:
            state.phase = "creating_coordinator"

        coord_config = CoordinatorConfig(
            dimension=model.num_parameters,
            num_workers=len(healthy_workers),
            proofs_per_step=config.K,
            use_adc=config.use_adc,
            adc_rank=config.adc_rank,
            adc_warmup_samples=20 if config.use_adc else 0,
            learning_rate=config.lr,
            device="cpu",
        )
        coordinator = Coordinator(coord_config)
        coordinator.set_parameters(model.get_flat_params())

        train_iter = iter(train_loader)

        # Training loop
        for step in range(start_step, config.steps):
            with state_lock:
                if state.stop_requested:
                    save_checkpoint(step, model, coordinator, state.losses, config)
                    state.phase = "stopped"
                    state.running = False
                    return
                state.phase = "adc_warmup" if not codebook_synced else "training"

            step_start = time.time()

            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch.input_ids.tolist()
            labels = batch.labels.tolist()
            coordinator.set_batch(batch.input_ids.flatten(), step)
            tasks = coordinator.publish_tasks(step)

            # Refresh healthy workers list
            healthy_workers = get_healthy_workers(state.workers)

            if len(healthy_workers) < 2:
                logger.error("No healthy workers available!")
                time.sleep(5)
                continue

            with state_lock:
                state.phase = f"step_{step}_computing"
                for wid in state.workers:
                    state.workers[wid].busy = False

            proofs = []
            worker_times = {}
            scalars_this_step = []

            # Compute with retry logic
            max_retries = 2
            tasks_remaining = list(enumerate(tasks))

            for retry in range(max_retries + 1):
                if not tasks_remaining:
                    break

                with ThreadPoolExecutor(max_workers=len(healthy_workers) * 2) as executor:
                    futures = {}
                    for i, task in tasks_remaining:
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
                            timeout=300,  # Longer timeout for large K
                        )
                        futures[future] = (i, wid, task)

                    completed_indices = set()
                    try:
                        for future in as_completed(futures, timeout=360):
                            i, wid, task = futures[future]

                            with state_lock:
                                state.workers[wid].busy = False

                            try:
                                result = future.result(timeout=5)
                            except Exception as e:
                                result = None
                                logger.warning(f"Task {i} result error: {e}")

                            if result:
                                completed_indices.add(i)

                                with state_lock:
                                    state.workers[wid].tasks_completed += 1
                                    state.workers[wid].consecutive_errors = 0
                                    state.workers[wid].last_task_time = result.get("worker_time", 0)
                                    state.workers[wid].last_scalar = result["scalar"]

                                worker_times[wid] = result.get("worker_time", 0)
                                scalars_this_step.append(result["scalar"])

                                proof = Proof(
                                    step=result["step"],
                                    worker_id=hash(wid) % 10000,
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
                                    state.workers[wid].consecutive_errors += 1
                                    # Disable worker after 5 consecutive errors
                                    if state.workers[wid].consecutive_errors >= 5:
                                        state.workers[wid].disabled = True
                                        logger.warning(f"Worker {wid} disabled due to errors")

                    except FuturesTimeoutError:
                        logger.warning(f"Step {step} timeout on retry {retry}")

                    # Update tasks remaining
                    tasks_remaining = [
                        (i, task) for i, task in tasks_remaining if i not in completed_indices
                    ]

                if tasks_remaining and retry < max_retries:
                    logger.info(f"Retrying {len(tasks_remaining)} failed tasks")
                    time.sleep(1)

            # Check if we have enough proofs
            if len(proofs) < max(2, config.K // 4):
                logger.warning(f"Step {step}: only {len(proofs)} proofs, skipping")
                with state_lock:
                    state.phase = f"step_{step}_insufficient_proofs"
                continue

            with state_lock:
                state.phase = f"step_{step}_aggregating"

            gradient, agg_result = coordinator.aggregate()
            gradient_norm = float(np.linalg.norm(gradient))

            logger.info(
                f"[Step {step}] Aggregated {len(proofs)}/{config.K} proofs, grad_norm={gradient_norm:.6f}"
            )

            with state_lock:
                state.phase = f"step_{step}_updating"

            new_params = coordinator.update_parameters(gradient)
            model.set_flat_params(new_params)

            # Codebook sync after warmup
            if coordinator.codebook and coordinator.codebook.is_warmed_up and not codebook_synced:
                with state_lock:
                    state.phase = f"step_{step}_syncing_codebook"

                logger.info(f"[Step {step}] Syncing codebook to workers...")

                import base64

                codebook_U = coordinator.codebook.codebook
                codebook_max = np.max(np.abs(codebook_U))
                codebook_scale = codebook_max / 127.0 if codebook_max > 0 else 1.0
                codebook_int8 = (codebook_U / codebook_scale).astype(np.int8)
                codebook_b64 = base64.b64encode(codebook_int8.tobytes()).decode("ascii")

                sync_count = 0
                with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
                    futures = {}
                    for wid, url in healthy_workers:
                        future = executor.submit(
                            client.sync_codebook,
                            url,
                            codebook_b64,
                            coordinator.codebook.rank,
                            coordinator.codebook.dimension,
                            float(codebook_scale),
                        )
                        futures[future] = wid

                    for future in as_completed(futures, timeout=180):
                        wid = futures[future]
                        try:
                            if future.result():
                                sync_count += 1
                                logger.info(f"Codebook synced to {wid}")
                        except Exception as e:
                            logger.warning(f"Codebook sync failed for {wid}: {e}")

                if sync_count >= max(2, len(healthy_workers) // 2):
                    codebook_synced = True
                    with state_lock:
                        state.codebook_synced = True
                        state.adc_warmed_up = True
                    logger.info(f"ADC enabled! Synced to {sync_count} workers")

            # Sync gradient to workers
            use_adc = codebook_synced and coordinator.codebook is not None
            seeds_hex = [p.seed.hex() for p in proofs]
            scalars_list = [p.scalar for p in proofs]

            with state_lock:
                state.phase = f"step_{step}_syncing_params"

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

            logger.info(
                f"[Step {step}] Loss: {loss_val:.4f}, Time: {step_time:.1f}s, Proofs: {len(proofs)}/{config.K}"
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
                    }
                )

                if not codebook_synced and coordinator.codebook:
                    state.adc_warmup_current = min(step + 1, state.adc_warmup_needed)

                if scalars_this_step:
                    state.scalar_stats = {
                        "mean": round(float(np.mean(scalars_this_step)), 6),
                        "std": round(float(np.std(scalars_this_step)), 6),
                    }

            # Save checkpoint periodically
            if config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
                save_checkpoint(step + 1, model, coordinator, state.losses, config)
                with state_lock:
                    state.last_checkpoint_step = step + 1

        # Final checkpoint
        save_checkpoint(config.steps, model, coordinator, state.losses, config)

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
                "disabled": ws.disabled,
                "tasks": ws.tasks_completed,
                "errors": ws.errors,
                "busy": ws.busy,
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
            "step_time": round(state.step_time, 2),
            "elapsed": round(state.elapsed, 1),
            "eta_seconds": round((state.total_steps - state.step) * state.step_time, 0)
            if state.step_time > 0
            else 0,
            "proofs": state.proofs_count,
            "K": state.K,
            "model_params": state.model_params,
            "adc": {
                "warmed_up": state.adc_warmed_up,
                "codebook_synced": state.codebook_synced,
            },
            "workers": workers_summary,
            "last_checkpoint": state.last_checkpoint_step,
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
                "disabled": ws.disabled,
                "tasks_completed": ws.tasks_completed,
                "errors": ws.errors,
                "consecutive_errors": ws.consecutive_errors,
                "busy": ws.busy,
            }
        return result


@app.get("/losses")
def get_losses():
    with state_lock:
        return {
            "losses": state.losses,
            "gradient_norms": state.gradient_norms,
        }


@app.post("/stop")
def stop_training():
    global state
    with state_lock:
        if not state.running:
            return {"status": "not running"}
        state.stop_requested = True
    return {"status": "stop requested, will checkpoint and stop"}


@app.get("/checkpoints")
def list_checkpoints():
    checkpoints = []
    if CHECKPOINT_DIR.exists():
        for f in CHECKPOINT_DIR.glob("checkpoint_*.npz"):
            checkpoints.append(
                {
                    "path": str(f),
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                }
            )
    return {"checkpoints": sorted(checkpoints, key=lambda x: x["name"])}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
