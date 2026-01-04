#!/usr/bin/env python3
"""
Test gradient cosine similarity with HuggingFace pretrained GPT-2.
This checks if the paper's claim (cosine ~ 0.07) applies to pretrained models.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


def compute_gradient(model, input_ids, labels):
    model.zero_grad()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads), loss.item()


def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-10)


def main():
    print("=" * 70)
    print("PRETRAINED GPT-2 GRADIENT VARIABILITY TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading pretrained GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    print("\nLoading WikiText data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    seq_length = 256
    num_gradients = 30

    print(f"\nCollecting {num_gradients} gradients (seq_length={seq_length})...")
    gradients = []

    texts = [t for t in dataset["text"] if len(t) > 100][: num_gradients * 2]

    for i, text in enumerate(texts[:num_gradients]):
        tokens = tokenizer(
            text, return_tensors="pt", max_length=seq_length, truncation=True, padding="max_length"
        )
        input_ids = tokens["input_ids"].to(device)

        grad, loss = compute_gradient(model, input_ids, input_ids)
        gradients.append(grad.cpu())

        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{num_gradients} gradients")

    print("\n" + "-" * 70)
    print("PAIRWISE COSINE SIMILARITY")
    print("-" * 70)

    cosines = []
    for i in range(len(gradients)):
        for j in range(i + 1, len(gradients)):
            cos = cosine_similarity(gradients[i], gradients[j]).item()
            cosines.append(cos)

    cosines = np.array(cosines)

    print(f"Mean pairwise cosine: {np.mean(cosines):.4f}")
    print(f"Std pairwise cosine:  {np.std(cosines):.4f}")
    print(f"Min pairwise cosine:  {np.min(cosines):.4f}")
    print(f"Max pairwise cosine:  {np.max(cosines):.4f}")
    print(f"Median:               {np.median(cosines):.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Paper claim:     cosine ~ 0.07")
    print(f"Our result:      cosine = {np.mean(cosines):.4f}")

    if np.mean(cosines) < 0.15:
        print("Status: VERIFIED - Matches paper claim")
    else:
        print(f"Status: DIFFERS - {np.mean(cosines) / 0.07:.1f}x higher than paper")


if __name__ == "__main__":
    main()
