#!/usr/bin/env python3
import sys

sys.path.insert(0, "/root/holograd/src")
import json
import pickle
import time
import numpy as np
import torch

from holograd.training.model import SimpleGPT2
from holograd.protocol.direction import DirectionGenerator, ADCCodebook
from holograd.gradient.jvp import compute_jvp_gradient_projection


def run_task(step: int, seed: int, use_adc: bool):
    with open("/tmp/task_data.pkl", "rb") as f:
        data = pickle.load(f)

    params = data["params"]
    input_ids = data["input_ids"]
    labels = data["labels"]
    codebook_data = data.get("codebook")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleGPT2(size="tiny", max_seq_len=64, vocab_size=100)
    model.set_flat_params(params)

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

    result = {
        "step": step,
        "seed": seed,
        "scalar": float(scalar),
        "adc_projection": adc_projection,
        "time": elapsed,
        "device": device,
    }
    print("RESULT:" + json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        run_task(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].lower() == "true")
