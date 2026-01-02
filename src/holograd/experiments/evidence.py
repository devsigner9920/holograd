"""
Experiment evidence collection for reproducible paper results.

Captures:
- Environment: Python version, package versions, hardware specs
- Reproducibility: git commit, branch, config snapshot
- Results: raw data, statistics, confidence intervals
- Visualization: auto-generated figures
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json
import platform
import subprocess
import sys


@dataclass
class EnvironmentInfo:
    python_version: str = ""
    platform_system: str = ""
    platform_release: str = ""
    platform_machine: str = ""
    cpu_count: int = 0
    cpu_brand: str = ""
    memory_gb: float = 0.0
    gpu_info: list[str] = field(default_factory=list)
    package_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def collect(cls) -> "EnvironmentInfo":
        info = cls()
        info.python_version = sys.version
        info.platform_system = platform.system()
        info.platform_release = platform.release()
        info.platform_machine = platform.machine()

        import os

        info.cpu_count = os.cpu_count() or 0

        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                info.cpu_brand = result.stdout.strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            info.cpu_brand = line.split(":")[1].strip()
                            break
        except Exception:
            info.cpu_brand = "unknown"

        try:
            import psutil

            info.memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    info.gpu_info.append(torch.cuda.get_device_name(i))
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info.gpu_info.append("Apple Silicon MPS")
        except ImportError:
            pass

        packages = ["numpy", "torch", "jax", "datasets", "transformers"]
        for pkg in packages:
            try:
                mod = __import__(pkg)
                info.package_versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                pass

        return info


@dataclass
class GitInfo:
    commit_hash: str = ""
    commit_short: str = ""
    branch: str = ""
    is_dirty: bool = False
    commit_date: str = ""
    commit_message: str = ""
    remote_url: str = ""

    @classmethod
    def collect(cls, repo_path: Optional[Path] = None) -> "GitInfo":
        info = cls()
        cwd = str(repo_path) if repo_path else None

        def run_git(args: list[str]) -> str:
            try:
                result = subprocess.run(
                    ["git"] + args, capture_output=True, text=True, timeout=5, cwd=cwd
                )
                return result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                return ""

        info.commit_hash = run_git(["rev-parse", "HEAD"])
        info.commit_short = run_git(["rev-parse", "--short", "HEAD"])
        info.branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        info.is_dirty = run_git(["status", "--porcelain"]) != ""
        info.commit_date = run_git(["log", "-1", "--format=%ci"])
        info.commit_message = run_git(["log", "-1", "--format=%s"])
        info.remote_url = run_git(["remote", "get-url", "origin"])

        return info


@dataclass
class StatsSummary:
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    n_samples: int = 0

    @classmethod
    def from_values(cls, values: list[float]) -> "StatsSummary":
        import numpy as np

        if not values:
            return cls()

        arr = np.array(values)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        # 95% CI using t-distribution
        if n > 1:
            from scipy import stats

            ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        else:
            ci_lower, ci_upper = mean, mean

        return cls(
            mean=mean,
            std=std,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=float(np.median(arr)),
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper,
            n_samples=n,
        )


class ExperimentEvidence:
    """
    Unified evidence collection for reproducible experiments.

    Usage:
        with ExperimentEvidence("bandwidth_benchmark") as evidence:
            evidence.set_config({"K": 64, "adc_rank": 32})
            evidence.add_result("compression_ratio", 40000.0)
            evidence.add_result("compression_ratio", 41000.0)
            evidence.create_figure("compression", fig)
        # Auto-saves on exit
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: Path | str = "outputs/experiments",
        auto_collect_env: bool = True,
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.experiment_id = f"{timestamp}_{experiment_name}"
        self.experiment_path = self.output_dir / self.experiment_id

        self.environment: Optional[EnvironmentInfo] = None
        self.git: Optional[GitInfo] = None
        self.config: dict[str, Any] = {}
        self.results: dict[str, list[float]] = {}
        self.metadata: dict[str, Any] = {}
        self.figures_saved: list[str] = []

        self._start_time = datetime.now()

        if auto_collect_env:
            self.collect_environment()

    def collect_environment(self) -> None:
        self.environment = EnvironmentInfo.collect()
        self.git = GitInfo.collect()

    def set_config(self, config: dict[str, Any]) -> None:
        self.config = config

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def add_result(self, metric_name: str, value: float) -> None:
        if metric_name not in self.results:
            self.results[metric_name] = []
        self.results[metric_name].append(value)

    def add_results_batch(self, metric_name: str, values: list[float]) -> None:
        if metric_name not in self.results:
            self.results[metric_name] = []
        self.results[metric_name].extend(values)

    def add_table_row(self, table_name: str, row: dict[str, Any]) -> None:
        key = f"_table_{table_name}"
        if key not in self.metadata:
            self.metadata[key] = []
        self.metadata[key].append(row)

    def get_stats(self, metric_name: str) -> StatsSummary:
        values = self.results.get(metric_name, [])
        return StatsSummary.from_values(values)

    def get_all_stats(self) -> dict[str, StatsSummary]:
        return {name: self.get_stats(name) for name in self.results}

    def save_figure(self, name: str, fig: Any, formats: list[str] = ["png", "pdf"]) -> list[Path]:
        figures_dir = self.experiment_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for fmt in formats:
            path = figures_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            saved.append(path)
            self.figures_saved.append(str(path.relative_to(self.experiment_path)))

        return saved

    def save(self) -> Path:
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        end_time = datetime.now()
        duration_seconds = (end_time - self._start_time).total_seconds()

        if self.environment:
            with open(self.experiment_path / "environment.json", "w") as f:
                json.dump(asdict(self.environment), f, indent=2)

        if self.git:
            with open(self.experiment_path / "git_info.json", "w") as f:
                json.dump(asdict(self.git), f, indent=2)

        if self.config:
            with open(self.experiment_path / "config.json", "w") as f:
                json.dump(self.config, f, indent=2)

        with open(self.experiment_path / "results_raw.json", "w") as f:
            json.dump(self.results, f, indent=2)

        stats = self.get_all_stats()
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "start_time": self._start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "statistics": {name: asdict(s) for name, s in stats.items()},
            "figures": self.figures_saved,
            "metadata": self.metadata,
        }
        with open(self.experiment_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self._save_csv()
        self._save_tables()

        return self.experiment_path

    def _save_csv(self) -> None:
        if not self.results:
            return

        import csv

        csv_path = self.experiment_path / "results.csv"

        max_len = max(len(v) for v in self.results.values())

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index"] + list(self.results.keys()))

            for i in range(max_len):
                row = [i]
                for metric in self.results:
                    values = self.results[metric]
                    row.append(values[i] if i < len(values) else "")
                writer.writerow(row)

    def _save_tables(self) -> None:
        import csv

        for key, rows in self.metadata.items():
            if not key.startswith("_table_") or not rows:
                continue

            table_name = key.replace("_table_", "")
            csv_path = self.experiment_path / f"table_{table_name}.csv"

            headers = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

    def __enter__(self) -> "ExperimentEvidence":
        return self

    def __exit__(self, *args: Any) -> None:
        self.save()


def create_comparison_figure(
    x_values: list[float],
    y_values_dict: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
) -> Any:
    """Create a comparison line plot with multiple series."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors = plt.cm.tab10.colors

    for i, (label, y_values) in enumerate(y_values_dict.items()):
        ax.plot(
            x_values,
            y_values,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=label,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_bar_chart(
    categories: list[str],
    values_dict: dict[str, list[float]],
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> Any:
    """Create a grouped bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    n_groups = len(values_dict)
    width = 0.8 / n_groups

    colors = plt.cm.tab10.colors

    for i, (label, values) in enumerate(values_dict.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")

    if log_scale:
        ax.set_yscale("log")

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig
