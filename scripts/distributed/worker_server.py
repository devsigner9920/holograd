#!/usr/bin/env python3
"""
HoloGrad Worker Server - Production-ready with stability improvements.

Features:
- OOM prevention with aggressive memory cleanup
- Graceful error handling with detailed logging
- Health monitoring with memory stats
- Automatic CUDA cache management
"""

import sys
import gc
import logging
import signal
import time
import traceback
from contextlib import contextmanager
from typing import Optional, List

sys.path.insert(0, "/root/holograd/src")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from holograd.training.model import SimpleGPT2
from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.gradient.jvp import compute_jvp_gradient_projection

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("worker")

app = FastAPI(title="HoloGrad Worker")

# Global state
model: Optional[SimpleGPT2] = None
direction_gen: Optional[DirectionGenerator] = None
adc_codebook: Optional[ADCCodebook] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Stats tracking
stats = {
    "tasks_completed": 0,
    "tasks_failed": 0,
    "oom_recoveries": 0,
    "start_time": time.time(),
    "last_task_time": 0.0,
}


def get_memory_stats() -> dict:
    """Get current memory usage stats."""
    result = {"device": device}

    if device == "cuda" and torch.cuda.is_available():
        result["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
        result["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
        result["cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)

        # Get total GPU memory
        total = torch.cuda.get_device_properties(0).total_memory
        result["cuda_total_mb"] = round(total / 1024 / 1024, 2)
        result["cuda_free_mb"] = round((total - torch.cuda.memory_reserved()) / 1024 / 1024, 2)

    return result


def cleanup_memory(aggressive: bool = False):
    """Clean up memory to prevent OOM."""
    gc.collect()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()


@contextmanager
def safe_compute():
    """Context manager for safe computation with OOM recovery."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM detected: {e}")
        stats["oom_recoveries"] += 1
        cleanup_memory(aggressive=True)
        raise HTTPException(status_code=503, detail="GPU OOM - cleaned up, please retry")
    except Exception as e:
        logger.error(f"Compute error: {e}\n{traceback.format_exc()}")
        cleanup_memory()
        raise


class InitRequest(BaseModel):
    model_size: str = "tiny"
    vocab_size: int = 100
    seq_length: int = 64
    params: Optional[List[float]] = None
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    seed: int = 0
    adc_rank: int = 16
    codebook_seed: Optional[int] = None


class UpdateParamsRequest(BaseModel):
    params: List[float]


class CodebookB64Request(BaseModel):
    U_b64: str
    rank: int
    dimension: int
    dtype: str = "float16"
    scale: float = 1.0


class SeedGradientSyncRequest(BaseModel):
    seeds: List[str]
    scalars: List[float]
    learning_rate: float
    use_adc: bool = False


class TaskRequest(BaseModel):
    step: int
    seed: str
    use_adc: bool
    input_ids: List[List[int]]
    labels: List[List[int]]


class TaskResponse(BaseModel):
    step: int
    seed: str
    scalar: float
    adc_projection: Optional[List[float]] = None
    time: float
    device: str


@app.get("/health")
def health():
    """Health check with detailed stats."""
    mem = get_memory_stats()
    return {
        "status": "ok",
        "device": device,
        "model_loaded": model is not None,
        "adc_loaded": adc_codebook is not None,
        "uptime_seconds": round(time.time() - stats["start_time"], 1),
        "tasks_completed": stats["tasks_completed"],
        "tasks_failed": stats["tasks_failed"],
        "oom_recoveries": stats["oom_recoveries"],
        "memory": mem,
    }


@app.post("/init")
def init_model(req: InitRequest):
    """Initialize model with memory-efficient loading."""
    global model, direction_gen, adc_codebook

    # Clean up existing model first
    cleanup_memory(aggressive=True)
    model = None
    direction_gen = None
    adc_codebook = None
    cleanup_memory(aggressive=True)

    try:
        logger.info(
            f"Initializing model: size={req.model_size}, n_layer={req.n_layer}, n_head={req.n_head}, n_embd={req.n_embd}"
        )

        if req.model_size == "custom" and req.n_layer and req.n_head and req.n_embd:
            model = SimpleGPT2(
                size="custom",
                n_layer=req.n_layer,
                n_head=req.n_head,
                n_embd=req.n_embd,
                vocab_size=req.vocab_size,
                max_seq_len=req.seq_length,
                seed=req.seed,
            )
        else:
            model = SimpleGPT2(
                size=req.model_size,
                max_seq_len=req.seq_length,
                vocab_size=req.vocab_size,
                seed=req.seed,
            )

        direction_gen = DirectionGenerator(model.num_parameters)

        if req.codebook_seed is not None:
            adc_codebook = ADCCodebook(
                dimension=model.num_parameters,
                rank=req.adc_rank,
                device=device,
            )
            adc_codebook.reset(seed=req.codebook_seed)
            logger.info(f"ADC codebook initialized: seed={req.codebook_seed}, rank={req.adc_rank}")

        if req.params:
            params = np.array(req.params, dtype=np.float32)
            model.set_flat_params(params)

        mem = get_memory_stats()
        logger.info(f"Model initialized: {model.num_parameters:,} params, memory: {mem}")

        return {
            "status": "initialized",
            "num_parameters": model.num_parameters,
            "adc_initialized": req.codebook_seed is not None,
            "memory": mem,
        }

    except Exception as e:
        logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
        cleanup_memory(aggressive=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_params")
def update_params(req: UpdateParamsRequest):
    """Update model parameters."""
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    try:
        params = np.array(req.params, dtype=np.float32)
        model.set_flat_params(params)
        return {"status": "updated"}
    except Exception as e:
        logger.error(f"Update params failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_codebook_b64")
def update_codebook_b64(req: CodebookB64Request):
    """Update codebook using base64-encoded binary data."""
    global adc_codebook
    import base64

    try:
        logger.info(f"Receiving codebook: dim={req.dimension}, rank={req.rank}, dtype={req.dtype}")

        # Decode base64 to bytes
        codebook_bytes = base64.b64decode(req.U_b64)

        # Convert to numpy array based on dtype
        if req.dtype == "int8":
            U_int8 = np.frombuffer(codebook_bytes, dtype=np.int8).reshape(req.dimension, req.rank)
            U_array = U_int8.astype(np.float32) * req.scale
        elif req.dtype == "float16":
            U_array = np.frombuffer(codebook_bytes, dtype=np.float16).reshape(
                req.dimension, req.rank
            )
            U_array = U_array.astype(np.float32)
        else:
            U_array = np.frombuffer(codebook_bytes, dtype=np.float32).reshape(
                req.dimension, req.rank
            )

        logger.info(
            f"Decoded codebook: shape={U_array.shape}, norm={np.linalg.norm(U_array[:, 0]):.4f}"
        )

        # Clean up old codebook
        if adc_codebook is not None:
            del adc_codebook
            cleanup_memory()

        adc_codebook = ADCCodebook(
            dimension=req.dimension,
            rank=req.rank,
            device=device,
        )
        adc_codebook._U = U_array
        adc_codebook._is_warmed_up = True

        logger.info("Codebook updated successfully")
        return {"status": "codebook_updated", "shape": list(U_array.shape)}

    except Exception as e:
        logger.error(f"Codebook update failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply_seed_gradient")
def apply_seed_gradient(req: SeedGradientSyncRequest):
    """Apply gradient update using seeds for deterministic direction reconstruction."""
    global model, direction_gen, adc_codebook

    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    try:
        K = len(req.seeds)
        if K == 0:
            raise HTTPException(status_code=400, detail="No seeds provided")

        dimension = model.num_parameters

        if req.use_adc and adc_codebook is not None:
            scale_factor = float(adc_codebook.rank)
            effective_dim = float(adc_codebook.rank)
            logger.debug(f"Using ADC: scale_factor={scale_factor}, K={K}")
        else:
            scale_factor = float(dimension)
            effective_dim = float(dimension)
            logger.debug(f"Using random: scale_factor={scale_factor}, K={K}")

        gradient = np.zeros(dimension, dtype=np.float32)
        for seed_hex, scalar in zip(req.seeds, req.scalars):
            seed_bytes = bytes.fromhex(seed_hex)
            if req.use_adc and adc_codebook is not None:
                result = adc_codebook.generate_direction(seed_bytes)
            else:
                result = direction_gen.generate(seed_bytes)
            gradient += scalar * result.direction

        # Match coordinator's formula
        variance_correction = float(np.sqrt(K / effective_dim))
        gradient = gradient * (scale_factor / K) * variance_correction

        gradient_norm = float(np.linalg.norm(gradient))

        current_params = model.get_flat_params()
        new_params = (current_params - req.learning_rate * gradient).astype(np.float32)
        model.set_flat_params(new_params)

        param_change = float(np.linalg.norm(new_params - current_params))
        logger.debug(f"Gradient applied: norm={gradient_norm:.6f}, param_change={param_change:.6f}")

        # Periodic memory cleanup
        if stats["tasks_completed"] % 50 == 0:
            cleanup_memory()

        return {"status": "seed_gradient_applied", "gradient_norm": gradient_norm}

    except Exception as e:
        logger.error(f"Apply gradient failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compute", response_model=TaskResponse)
def compute_jvp(req: TaskRequest):
    """Compute JVP with OOM protection and memory management."""
    global model, direction_gen, adc_codebook

    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    start = time.time()

    with safe_compute():
        try:
            input_ids = np.array(req.input_ids, dtype=np.int64)
            labels = np.array(req.labels, dtype=np.int64)

            seed_bytes = bytes.fromhex(req.seed)

            adc_projection = None
            if req.use_adc and adc_codebook is not None:
                result = adc_codebook.generate_direction(seed_bytes)
                direction = result.direction
                if result.z_projection is not None:
                    adc_projection = result.z_projection.tolist()
            else:
                result = direction_gen.generate(seed_bytes)
                direction = result.direction

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            scalar = compute_jvp_gradient_projection(
                model=model,
                input_ids=input_ids_t,
                labels=labels_t,
                direction=direction,
                device=device,
            )

            elapsed = time.time() - start

            # Update stats
            stats["tasks_completed"] += 1
            stats["last_task_time"] = elapsed

            # Cleanup tensors
            del input_ids_t, labels_t

            # Periodic aggressive cleanup
            if stats["tasks_completed"] % 100 == 0:
                cleanup_memory(aggressive=True)
                logger.info(
                    f"Periodic cleanup at task {stats['tasks_completed']}, memory: {get_memory_stats()}"
                )

            return TaskResponse(
                step=req.step,
                seed=req.seed,
                scalar=float(scalar),
                adc_projection=adc_projection,
                time=elapsed,
                device=device,
            )

        except Exception as e:
            stats["tasks_failed"] += 1
            logger.error(f"Task failed: step={req.step}, error={e}")
            raise


class DiagnosticRequest(BaseModel):
    input_ids: List[List[int]]
    labels: List[List[int]]


@app.post("/diagnostic")
def run_diagnostic(req: DiagnosticRequest):
    """Compute full gradient for diagnostics."""
    global model

    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    with safe_compute():
        input_ids = np.array(req.input_ids, dtype=np.int64)
        labels = np.array(req.labels, dtype=np.int64)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        flat_params = model.get_flat_params()
        flat_params_t = torch.tensor(
            flat_params, dtype=torch.float32, device=device, requires_grad=True
        )

        params_dict = model.flat_params_to_torch_dict(flat_params_t)
        logits = model.forward_torch(input_ids_t, params_dict)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels_t.view(-1),
        )

        grad = torch.autograd.grad(loss, flat_params_t)[0]
        grad_np = grad.cpu().numpy()

        # Cleanup
        del input_ids_t, labels_t, flat_params_t, logits, grad
        cleanup_memory()

        return {
            "loss": float(loss.item()),
            "gradient_norm": float(np.linalg.norm(grad_np)),
            "gradient_mean": float(np.mean(grad_np)),
            "gradient_std": float(np.std(grad_np)),
            "gradient_max": float(np.max(np.abs(grad_np))),
            "num_parameters": model.num_parameters,
            "memory": get_memory_stats(),
        }


@app.post("/cleanup")
def force_cleanup():
    """Force memory cleanup."""
    cleanup_memory(aggressive=True)
    return {"status": "cleaned", "memory": get_memory_stats()}


@app.get("/stats")
def get_stats():
    """Get worker statistics."""
    return {
        **stats,
        "uptime_seconds": round(time.time() - stats["start_time"], 1),
        "memory": get_memory_stats(),
    }


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup_memory(aggressive=True)
    sys.exit(0)


if __name__ == "__main__":
    import argparse

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logger.info(f"Starting worker on {args.host}:{args.port}, device={device}")
    logger.info(f"Initial memory: {get_memory_stats()}")

    uvicorn.run(app, host=args.host, port=args.port)
