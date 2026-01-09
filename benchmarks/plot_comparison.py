#!/usr/bin/env python3
"""
Generate comparison plots for gradient compression methods.

Produces:
- Figure 1: Loss vs Tokens (training efficiency)
- Figure 2: Loss vs Cumulative Bits (communication efficiency)
- Figure 3: Final Loss vs Bits/Step (Pareto frontier)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

METHOD_COLORS = {
    "full_sgd": "#2ecc71",
    "powersgd_rank1": "#3498db",
    "powersgd_rank2": "#9b59b6",
    "holograd_k64": "#e74c3c",
    "holograd_k32": "#e67e22",
    "holograd_momentum": "#1abc9c",
}

METHOD_LABELS = {
    "full_sgd": "Full SGD",
    "powersgd_rank1": "PowerSGD (r=1)",
    "powersgd_rank2": "PowerSGD (r=2)",
    "holograd_k64": "HoloGrad (K=64)",
    "holograd_k32": "HoloGrad (K=32)",
    "holograd_momentum": "HoloGrad-Momentum",
}

METHOD_MARKERS = {
    "full_sgd": "o",
    "powersgd_rank1": "s",
    "powersgd_rank2": "^",
    "holograd_k64": "D",
    "holograd_k32": "v",
    "holograd_momentum": "*",
}


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    results = {}
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            method = data.get("method", json_file.stem)
            results[method] = data
    return results


def get_color(method: str) -> str:
    for key, color in METHOD_COLORS.items():
        if key in method.lower():
            return color
    return "#7f8c8d"


def get_label(method: str) -> str:
    for key, label in METHOD_LABELS.items():
        if key in method.lower():
            return label
    return method


def get_marker(method: str) -> str:
    for key, marker in METHOD_MARKERS.items():
        if key in method.lower():
            return marker
    return "o"


def plot_loss_vs_tokens(results: Dict[str, Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in sorted(results.items()):
        if not data.get("tokens") or not data.get("losses"):
            continue

        tokens = np.array(data["tokens"]) / 1e6
        losses = np.array(data["losses"])

        ax.plot(
            tokens,
            losses,
            color=get_color(method),
            label=get_label(method),
            marker=get_marker(method),
            markevery=max(1, len(tokens) // 10),
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Tokens (millions)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Tokens Processed")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_loss_vs_bits(results: Dict[str, Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in sorted(results.items()):
        if not data.get("steps") or not data.get("losses"):
            continue

        bits_per_step = data.get("bits_per_step", 0)
        steps = np.array(data["steps"])
        cumulative_bits = steps * bits_per_step / 1e12
        losses = np.array(data["losses"])

        ax.plot(
            cumulative_bits,
            losses,
            color=get_color(method),
            label=get_label(method),
            marker=get_marker(method),
            markevery=max(1, len(steps) // 10),
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Cumulative Communication (Tbits)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Communication Cost")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pareto_frontier(results: Dict[str, Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    points = []
    for method, data in results.items():
        bits_per_step = data.get("bits_per_step", 0)
        final_loss = data.get("final_loss", 0)
        if bits_per_step > 0 and final_loss > 0:
            points.append((method, bits_per_step, final_loss))

    for method, bits, loss in points:
        ax.scatter(
            bits / 1e6,
            loss,
            color=get_color(method),
            label=get_label(method),
            marker=get_marker(method),
            s=150,
            edgecolors="black",
            linewidths=1,
        )

    ax.set_xlabel("Bits per Step (Mbits)")
    ax.set_ylabel("Final Loss")
    ax.set_title("Communication Efficiency: Final Loss vs Bits/Step")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison_summary(results: Dict[str, Dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax1, ax2, ax3 = axes

    for method, data in sorted(results.items()):
        if not data.get("tokens") or not data.get("losses"):
            continue

        tokens = np.array(data["tokens"]) / 1e6
        losses = np.array(data["losses"])
        color = get_color(method)
        label = get_label(method)
        marker = get_marker(method)

        ax1.plot(
            tokens,
            losses,
            color=color,
            label=label,
            marker=marker,
            markevery=max(1, len(tokens) // 8),
            linewidth=2,
            markersize=5,
        )

    ax1.set_xlabel("Tokens (millions)")
    ax1.set_ylabel("Loss")
    ax1.set_title("(a) Loss vs Tokens")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    for method, data in sorted(results.items()):
        if not data.get("steps") or not data.get("losses"):
            continue

        bits_per_step = data.get("bits_per_step", 0)
        steps = np.array(data["steps"])
        cumulative_bits = steps * bits_per_step / 1e9
        losses = np.array(data["losses"])

        ax2.plot(
            cumulative_bits,
            losses,
            color=get_color(method),
            label=get_label(method),
            marker=get_marker(method),
            markevery=max(1, len(steps) // 8),
            linewidth=2,
            markersize=5,
        )

    ax2.set_xlabel("Cumulative Bits (Gbits)")
    ax2.set_ylabel("Loss")
    ax2.set_title("(b) Loss vs Communication")
    ax2.grid(True, alpha=0.3)

    methods = []
    bits_list = []
    loss_list = []
    colors = []

    for method, data in results.items():
        bits_per_step = data.get("bits_per_step", 0)
        final_loss = data.get("final_loss", 0)
        if bits_per_step > 0 and final_loss > 0:
            methods.append(get_label(method))
            bits_list.append(bits_per_step / 1e6)
            loss_list.append(final_loss)
            colors.append(get_color(method))

    ax3.scatter(bits_list, loss_list, c=colors, s=150, edgecolors="black", linewidths=1)
    for i, method in enumerate(methods):
        ax3.annotate(
            method,
            (bits_list[i], loss_list[i]),
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax3.set_xlabel("Bits/Step (Mbits)")
    ax3.set_ylabel("Final Loss")
    ax3.set_title("(c) Efficiency Frontier")
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument("--results_dir", type=str, default="results/benchmark")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded {len(results)} result files:")
    for method, data in results.items():
        bits = data.get("bits_per_step", 0) / 1e6
        loss = data.get("final_loss", 0)
        print(f"  - {method}: {bits:.3f} Mbits/step, final_loss={loss:.4f}")

    plot_loss_vs_tokens(results, output_dir / "comparison_loss_tokens.png")
    plot_loss_vs_bits(results, output_dir / "comparison_loss_bits.png")
    plot_pareto_frontier(results, output_dir / "comparison_pareto.png")
    plot_comparison_summary(results, output_dir / "comparison_summary.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
