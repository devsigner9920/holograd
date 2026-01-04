#!/usr/bin/env python3
"""
Test gradient cosine after training for a few steps.
Hypothesis: Paper measured gradients during training, not at initialization.
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
    print("GRADIENT VARIABILITY DURING TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading pretrained GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.train()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print("\nLoading data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = list(set([t for t in dataset["text"] if len(t) > 300]))
    np.random.seed(42)
    np.random.shuffle(texts)

    seq_length = 256
    warmup_steps = 100
    num_gradients = 50

    print(f"\nWarming up with {warmup_steps} training steps...")
    for i in range(warmup_steps):
        text = texts[i % len(texts)]
        tokens = tokenizer(
            text, return_tensors="pt", max_length=seq_length, truncation=True, padding="max_length"
        )
        input_ids = tokens["input_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"  Step {i + 1}/{warmup_steps}, loss: {loss.item():.4f}")

    print(f"\nCollecting {num_gradients} gradients after training...")
    gradients = []

    test_texts = texts[warmup_steps : warmup_steps + num_gradients]

    for i, text in enumerate(test_texts):
        tokens = tokenizer(
            text, return_tensors="pt", max_length=seq_length, truncation=True, padding="max_length"
        )
        input_ids = tokens["input_ids"].to(device)

        grad, loss = compute_gradient(model, input_ids, input_ids)
        gradients.append(grad.cpu())

        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{num_gradients} gradients")

    print("\n" + "-" * 70)
    print("PAIRWISE COSINE SIMILARITY (after training)")
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
        print("Status: VERIFIED")
    else:
        print(f"Status: {np.mean(cosines) / 0.07:.1f}x higher than paper")


if __name__ == "__main__":
    main()
