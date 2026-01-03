#!/usr/bin/env python3
import sys

sys.path.insert(0, "/root/holograd/src")

import json
import time
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from holograd.training.model import SimpleGPT2
from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.gradient.jvp import compute_jvp_gradient_projection


app = FastAPI()

model: Optional[SimpleGPT2] = None
direction_gen: Optional[DirectionGenerator] = None
adc_codebook: Optional[ADCCodebook] = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class InitRequest(BaseModel):
    model_size: str = "tiny"
    vocab_size: int = 100
    seq_length: int = 64
    params: Optional[List[float]] = None
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    seed: int = 0


class UpdateParamsRequest(BaseModel):
    params: List[float]


class CodebookRequest(BaseModel):
    U: List[List[float]]
    rank: int
    dimension: int


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
    return {"status": "ok", "device": device, "model_loaded": model is not None}


@app.post("/init")
def init_model(req: InitRequest):
    global model, direction_gen

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

    if req.params:
        params = np.array(req.params, dtype=np.float32)
        model.set_flat_params(params)

    return {"status": "initialized", "num_parameters": model.num_parameters}


@app.post("/update_params")
def update_params(req: UpdateParamsRequest):
    global model
    if model is None:
        return {"error": "Model not initialized"}

    params = np.array(req.params, dtype=np.float32)
    model.set_flat_params(params)
    return {"status": "updated"}


@app.post("/update_codebook")
def update_codebook(req: CodebookRequest):
    global adc_codebook

    adc_codebook = ADCCodebook(
        dimension=req.dimension,
        rank=req.rank,
        device=device,
    )
    adc_codebook._U = np.array(req.U, dtype=np.float32)

    return {"status": "codebook_updated"}


@app.post("/compute", response_model=TaskResponse)
def compute_jvp(req: TaskRequest):
    global model, direction_gen, adc_codebook

    if model is None:
        raise ValueError("Model not initialized")

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

    start = time.time()
    scalar = compute_jvp_gradient_projection(
        model=model,
        input_ids=input_ids_t,
        labels=labels_t,
        direction=direction,
        device=device,
    )
    elapsed = time.time() - start

    return TaskResponse(
        step=req.step,
        seed=req.seed,
        scalar=float(scalar),
        adc_projection=adc_projection,
        time=elapsed,
        device=device,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
