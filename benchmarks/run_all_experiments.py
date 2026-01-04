#!/usr/bin/env python3
"""
HoloGrad Experiment Runner

Runs all experiments (E1-E7) and collects results.
See EXPERIMENT_PLAN.md for details.

Usage:
    python benchmarks/run_all_experiments.py --local      # Run local experiments only
    python benchmarks/run_all_experiments.py --all        # Run all experiments
    python benchmarks/run_all_experiments.py --experiment E1  # Run specific experiment
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

EXPERIMENTS = {
    "E1": {
        "name": "Gradient Variability",
        "script": "analyze_gradient_variation.py",
        "args": ["--full", "--save"],
        "claim": "C2",
        "local": True,
        "description": "Validates pairwise cosine similarity ~ 0.07",
    },
    "E2": {
        "name": "K/D Ratio Sweep",
        "script": "ablation_k.py",
        "args": ["--evidence"],
        "claim": "C1",
        "local": True,
        "description": "Validates K/D >= 0.01 for convergence",
    },
    "E3": {
        "name": "Momentum vs Random",
        "script": "momentum_holograd.py",
        "args": [],
        "claim": "C3, C4",
        "local": True,
        "description": "Validates momentum 180x more efficient",
    },
    "E4": {
        "name": "ADC Captured Energy",
        "script": "ablation_rank.py",
        "args": ["--evidence"],
        "claim": "C5",
        "local": True,
        "description": "Validates ADC learns gradient subspace",
    },
    "E5": {
        "name": "Unbiased Estimator",
        "script": "verify_convergence.py",
        "args": [],
        "claim": "C6",
        "local": True,
        "description": "Validates E[g_hat] = g",
    },
    "E6": {
        "name": "PoGP Verification",
        "script": "prove_convergence.py",
        "args": [],
        "claim": "C8",
        "local": True,
        "description": "Validates scalar reproducibility",
    },
    "E7": {
        "name": "Byzantine Tolerance",
        "script": "byzantine.py",
        "args": ["--evidence"],
        "claim": "C7",
        "local": True,
        "description": "Validates trimmed mean tolerates 20% Byzantine",
    },
}


def run_experiment(exp_id: str, benchmarks_dir: Path) -> Dict[str, Any]:
    exp = EXPERIMENTS[exp_id]
    script_path = benchmarks_dir / exp["script"]

    if not script_path.exists():
        return {
            "experiment": exp_id,
            "name": exp["name"],
            "status": "SKIPPED",
            "reason": f"Script not found: {exp['script']}",
        }

    print(f"\n{'=' * 60}")
    print(f"Running {exp_id}: {exp['name']}")
    print(f"Claim: {exp['claim']}")
    print(f"Script: {exp['script']}")
    print(f"{'=' * 60}\n")

    cmd = [sys.executable, str(script_path)] + exp["args"]
    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(benchmarks_dir.parent),
            capture_output=True,
            text=True,
            timeout=3600,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")

        return {
            "experiment": exp_id,
            "name": exp["name"],
            "claim": exp["claim"],
            "status": "SUCCESS" if result.returncode == 0 else "FAILED",
            "returncode": result.returncode,
            "duration_seconds": duration,
            "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        }

    except subprocess.TimeoutExpired:
        return {
            "experiment": exp_id,
            "name": exp["name"],
            "status": "TIMEOUT",
            "reason": "Exceeded 1 hour timeout",
        }
    except Exception as e:
        return {
            "experiment": exp_id,
            "name": exp["name"],
            "status": "ERROR",
            "reason": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="HoloGrad Experiment Runner")
    parser.add_argument("--local", action="store_true", help="Run local experiments only")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--experiment", type=str, help="Run specific experiment (E1-E7)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--output", type=str, default="results/experiment_results.json")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable Experiments:")
        print("-" * 60)
        for exp_id, exp in EXPERIMENTS.items():
            local_tag = "[LOCAL]" if exp["local"] else "[DISTRIBUTED]"
            print(f"  {exp_id}: {exp['name']} {local_tag}")
            print(f"       Claim: {exp['claim']}")
            print(f"       {exp['description']}")
            print()
        return

    benchmarks_dir = Path(__file__).parent

    if args.experiment:
        exp_ids = [args.experiment.upper()]
        if exp_ids[0] not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
    elif args.local:
        exp_ids = [k for k, v in EXPERIMENTS.items() if v["local"]]
    elif args.all:
        exp_ids = list(EXPERIMENTS.keys())
    else:
        print("Specify --local, --all, or --experiment <ID>")
        print("Use --list to see available experiments")
        return

    print("=" * 60)
    print("HoloGrad Experiment Runner")
    print("=" * 60)
    print(f"Experiments to run: {', '.join(exp_ids)}")
    print(f"Start time: {datetime.now().isoformat()}")

    results: List[Dict[str, Any]] = []

    for exp_id in exp_ids:
        result = run_experiment(exp_id, benchmarks_dir)
        results.append(result)

        status = result["status"]
        if status == "SUCCESS":
            print(f"\n[OK] {exp_id} completed successfully")
        else:
            print(f"\n[{status}] {exp_id}: {result.get('reason', '')}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        status_icon = {
            "SUCCESS": "[OK]",
            "FAILED": "[X]",
            "SKIPPED": "[-]",
            "TIMEOUT": "[T]",
            "ERROR": "[!]",
        }
        icon = status_icon.get(r["status"], "[?]")
        duration = r.get("duration_seconds", 0)
        print(f"  {icon} {r['experiment']}: {r['name']} ({duration:.1f}s)")

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"\nTotal: {success_count}/{len(results)} succeeded")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments_run": len(results),
        "success_count": success_count,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
