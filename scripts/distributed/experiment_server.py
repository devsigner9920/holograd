#!/usr/bin/env python3
"""
Experiment API Server for HoloGrad

Runs experiments via HTTP API with polling support.

Usage:
    python scripts/distributed/experiment_server.py --port 8080

Endpoints:
    POST /start   - Start experiment {"experiment": "E1", "config": {...}}
    GET  /status  - Get current status
    GET  /result  - Get experiment result
    POST /stop    - Stop current experiment
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

EXPERIMENTS = {
    "E1": {
        "name": "Gradient Variability",
        "script": "benchmarks/analyze_gradient_variation.py",
        "default_args": ["--full", "--save"],
    },
    "E3": {
        "name": "Momentum vs Random",
        "script": "benchmarks/momentum_holograd.py",
        "default_args": [],
    },
    "E7": {
        "name": "Byzantine Tolerance",
        "script": "benchmarks/byzantine.py",
        "default_args": ["--skip-training"],
    },
}


@dataclass
class ExperimentState:
    status: str = "idle"
    experiment: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExperimentRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.state = ExperimentState()
        self.process: Optional[subprocess.Popen] = None
        self.output_lines: list = []
        self._lock = threading.Lock()

    def start(self, experiment_id: str, config: Optional[Dict] = None) -> bool:
        with self._lock:
            if self.state.status == "running":
                return False

            if experiment_id not in EXPERIMENTS:
                self.state.error = f"Unknown experiment: {experiment_id}"
                return False

            exp = EXPERIMENTS[experiment_id]
            self.state = ExperimentState(
                status="running",
                experiment=experiment_id,
                started_at=datetime.now().isoformat(),
            )
            self.output_lines = []

        thread = threading.Thread(target=self._run, args=(exp, config))
        thread.daemon = True
        thread.start()
        return True

    def _run(self, exp: Dict, config: Optional[Dict]):
        script_path = self.project_root / exp["script"]
        args = exp["default_args"].copy()

        if config:
            for k, v in config.items():
                args.extend([f"--{k}", str(v)])

        cmd = [sys.executable, "-u", str(script_path)] + args
        env = {"PYTHONPATH": str(self.project_root / "src"), "PYTHONUNBUFFERED": "1"}

        try:
            import os

            full_env = os.environ.copy()
            full_env.update(env)

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.project_root),
                env=full_env,
            )

            for line in self.process.stdout:
                with self._lock:
                    self.output_lines.append(line.rstrip())
                    if len(self.output_lines) > 500:
                        self.output_lines = self.output_lines[-300:]
                    self.state.progress = line.rstrip()[:100]

            self.process.wait()

            with self._lock:
                if self.process.returncode == 0:
                    self.state.status = "completed"
                    self._load_result()
                else:
                    self.state.status = "failed"
                    self.state.error = f"Exit code: {self.process.returncode}"
                self.state.finished_at = datetime.now().isoformat()

        except Exception as e:
            with self._lock:
                self.state.status = "failed"
                self.state.error = str(e)
                self.state.finished_at = datetime.now().isoformat()

    def _load_result(self):
        result_path = self.project_root / "results" / "e1_gradient_variability.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    self.state.result = json.load(f)
            except Exception:
                pass

    def stop(self) -> bool:
        with self._lock:
            if self.process and self.state.status == "running":
                self.process.terminate()
                self.state.status = "stopped"
                self.state.finished_at = datetime.now().isoformat()
                return True
            return False

    def get_status(self) -> Dict:
        with self._lock:
            return {
                **asdict(self.state),
                "output_tail": self.output_lines[-30:] if self.output_lines else [],
            }


runner: Optional[ExperimentRunner] = None


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: Dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def do_GET(self):
        if self.path == "/status":
            self._send_json(runner.get_status())
        elif self.path == "/result":
            status = runner.get_status()
            if status["result"]:
                self._send_json({"success": True, "result": status["result"]})
            else:
                self._send_json({"success": False, "status": status["status"]})
        elif self.path == "/health":
            self._send_json({"status": "ok", "gpu": self._check_gpu()})
        elif self.path == "/experiments":
            self._send_json({"experiments": list(EXPERIMENTS.keys())})
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode() if content_length else "{}"

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        if self.path == "/start":
            experiment = data.get("experiment", "E1")
            config = data.get("config", {})

            if runner.start(experiment, config):
                self._send_json({"success": True, "message": f"Started {experiment}"})
            else:
                self._send_json({"success": False, "error": "Already running or invalid"}, 400)

        elif self.path == "/stop":
            if runner.stop():
                self._send_json({"success": True, "message": "Stopped"})
            else:
                self._send_json({"success": False, "error": "Not running"}, 400)
        else:
            self._send_json({"error": "Not found"}, 404)

    def _check_gpu(self) -> str:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "not available"
        except Exception:
            return "not available"


def main():
    global runner

    parser = argparse.ArgumentParser(description="Experiment API Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    runner = ExperimentRunner(project_root)

    server = HTTPServer((args.host, args.port), RequestHandler)
    print(f"Experiment server running on http://{args.host}:{args.port}")
    print(f"Project root: {project_root}")
    print("\nEndpoints:")
    print("  GET  /health      - Check server health")
    print("  GET  /experiments - List available experiments")
    print("  POST /start       - Start experiment")
    print("  GET  /status      - Get current status")
    print("  GET  /result      - Get experiment result")
    print("  POST /stop        - Stop current experiment")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
