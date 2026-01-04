#!/usr/bin/env python3
"""
HoloGrad Distributed Deployment Script

One-click deployment and training for Vast.ai instances.

Usage:
    # Deploy to all running instances and start training
    python deploy.py --deploy --train

    # Just deploy (setup workers)
    python deploy.py --deploy

    # Just start training (workers already deployed)
    python deploy.py --train

    # Check status
    python deploy.py --status

    # Custom training config
    python deploy.py --train --steps 1000 --K 512 --lr 0.01
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class VastInstance:
    id: str
    ssh_host: str
    ssh_port: int
    public_ip: str
    direct_port: int
    gpu_name: str
    status: str


def run_cmd(cmd: List[str], timeout: int = 30, capture: bool = True) -> Tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def get_vast_instances() -> List[VastInstance]:
    """Get all running Vast.ai instances."""
    code, stdout, stderr = run_cmd(["vastai", "show", "instances", "--raw"], timeout=30)

    if code != 0:
        print(f"Error getting instances: {stderr}")
        return []

    try:
        instances = json.loads(stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse instances JSON")
        return []

    result = []
    for inst in instances:
        if inst.get("actual_status") != "running":
            continue

        # Parse SSH info
        ssh_host = inst.get("ssh_host", "")
        ssh_port = inst.get("ssh_port", 22)
        public_ip = inst.get("public_ipaddr", "")

        # Find direct port for 8000
        direct_port = None
        ports = inst.get("ports", {})
        if isinstance(ports, dict):
            for port_key, port_info in ports.items():
                if isinstance(port_info, dict) and port_info.get("PrivatePort") == 8000:
                    direct_port = port_info.get("PublicPort")
                    break

        # Fallback to direct_port_end
        if not direct_port:
            direct_port = inst.get("direct_port_end")

        if not ssh_host or not direct_port:
            continue

        result.append(
            VastInstance(
                id=str(inst["id"]),
                ssh_host=ssh_host,
                ssh_port=ssh_port,
                public_ip=public_ip,
                direct_port=direct_port,
                gpu_name=inst.get("gpu_name", "Unknown"),
                status=inst.get("actual_status", "unknown"),
            )
        )

    return result


def ssh_exec(instance: VastInstance, command: str, timeout: int = 300) -> Tuple[bool, str]:
    """Execute command on instance via SSH."""
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-p",
        str(instance.ssh_port),
        f"root@{instance.ssh_host}",
        command,
    ]

    code, stdout, stderr = run_cmd(ssh_cmd, timeout=timeout)

    if code == 0:
        return True, stdout
    else:
        return False, stderr or stdout


def deploy_worker(
    instance: VastInstance, git_repo: str = "https://github.com/code-yeongyu/holograd.git"
) -> Tuple[bool, str]:
    """Deploy worker to a single instance."""
    print(f"[{instance.id}] Deploying to {instance.gpu_name}...")

    # Setup script that runs on the instance
    setup_script = f"""
set -e

# Install system dependencies
apt-get update -qq
apt-get install -y -qq git curl > /dev/null 2>&1

# Clone or update repo
if [ -d "/root/holograd" ]; then
    cd /root/holograd
    git fetch origin
    git reset --hard origin/main
    echo "Repository updated"
else
    git clone {git_repo} /root/holograd
    echo "Repository cloned"
fi

cd /root/holograd

# Install Python dependencies
pip install --quiet --upgrade pip
pip install --quiet torch numpy scipy requests fastapi uvicorn pydantic datasets transformers tokenizers

# Verify installation
python -c "import torch; print(f'PyTorch: {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')"

echo "DEPLOY_SUCCESS"
"""

    success, output = ssh_exec(instance, setup_script, timeout=600)

    if success and "DEPLOY_SUCCESS" in output:
        print(f"[{instance.id}] Deploy successful")
        return True, output
    else:
        print(f"[{instance.id}] Deploy failed: {output[-500:]}")
        return False, output


def start_worker(instance: VastInstance) -> Tuple[bool, str]:
    """Start worker server on instance."""
    print(f"[{instance.id}] Starting worker...")

    start_script = """
set -e

# Kill any existing worker
pkill -f "worker_server.py" 2>/dev/null || true
sleep 2

# Start worker in background
cd /root/holograd
nohup python scripts/distributed/worker_server.py --port 8000 > /tmp/worker.log 2>&1 &

# Wait for startup
sleep 5

# Check if running
if pgrep -f "worker_server.py" > /dev/null; then
    echo "WORKER_STARTED"
else
    cat /tmp/worker.log
    echo "WORKER_FAILED"
fi
"""

    success, output = ssh_exec(instance, start_script, timeout=60)

    if success and "WORKER_STARTED" in output:
        print(f"[{instance.id}] Worker started")
        return True, output
    else:
        print(f"[{instance.id}] Worker start failed: {output[-500:]}")
        return False, output


def check_worker_health(instance: VastInstance) -> Tuple[bool, Optional[dict]]:
    """Check if worker is healthy via HTTP."""
    import requests

    url = f"http://{instance.public_ip}:{instance.direct_port}/health"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception as e:
        pass

    return False, None


def deploy_all(instances: List[VastInstance], parallel: int = 4) -> Dict[str, bool]:
    """Deploy to all instances in parallel."""
    results = {}

    print(f"\n{'=' * 60}")
    print(f"Deploying to {len(instances)} instances...")
    print(f"{'=' * 60}\n")

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(deploy_worker, inst): inst for inst in instances}

        for future in as_completed(futures):
            inst = futures[future]
            try:
                success, _ = future.result()
                results[inst.id] = success
            except Exception as e:
                print(f"[{inst.id}] Exception: {e}")
                results[inst.id] = False

    # Start workers
    print(f"\n{'=' * 60}")
    print("Starting workers...")
    print(f"{'=' * 60}\n")

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(start_worker, inst): inst
            for inst in instances
            if results.get(inst.id, False)
        }

        for future in as_completed(futures):
            inst = futures[future]
            try:
                success, _ = future.result()
                if not success:
                    results[inst.id] = False
            except Exception as e:
                print(f"[{inst.id}] Exception: {e}")
                results[inst.id] = False

    # Verify health
    print(f"\n{'=' * 60}")
    print("Verifying worker health...")
    print(f"{'=' * 60}\n")

    time.sleep(5)  # Wait for workers to fully start

    for inst in instances:
        if results.get(inst.id, False):
            healthy, info = check_worker_health(inst)
            if healthy:
                print(f"[{inst.id}] ✓ Healthy: {info.get('device', 'unknown')}")
            else:
                print(f"[{inst.id}] ✗ Not responding")
                results[inst.id] = False

    return results


def save_workers_config(instances: List[VastInstance], results: Dict[str, bool]):
    """Save workers configuration for coordinator."""
    workers = {}
    for inst in instances:
        if results.get(inst.id, False):
            workers[inst.id] = f"http://{inst.public_ip}:{inst.direct_port}"

    config_path = Path(__file__).parent / "workers.json"
    with open(config_path, "w") as f:
        json.dump(workers, f, indent=2)

    print(f"\nSaved {len(workers)} workers to {config_path}")
    return workers


def start_training(
    steps: int = 500,
    K: int = 512,
    lr: float = 0.01,
    n_layer: int = 6,
    n_head: int = 8,
    n_embd: int = 256,
    adc_rank: int = 32,
    batch_size: int = 8,
    checkpoint_every: int = 50,
):
    """Start training via coordinator API."""
    import requests

    # Load workers config
    config_path = Path(__file__).parent / "workers.json"
    if not config_path.exists():
        print("Error: workers.json not found. Run --deploy first.")
        return False

    with open(config_path) as f:
        workers = json.load(f)

    if not workers:
        print("Error: No workers configured.")
        return False

    print(f"\n{'=' * 60}")
    print(f"Starting training with {len(workers)} workers")
    print(f"Config: steps={steps}, K={K}, lr={lr}")
    print(f"Model: layers={n_layer}, heads={n_head}, embed={n_embd}")
    print(f"{'=' * 60}\n")

    # Set workers in environment for coordinator
    os.environ["HOLOGRAD_WORKERS"] = json.dumps(workers)

    # Start coordinator server in background
    print("Starting coordinator server...")

    coordinator_path = Path(__file__).parent / "coordinator_server.py"
    coordinator_proc = subprocess.Popen(
        [sys.executable, str(coordinator_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )

    # Wait for coordinator to start
    time.sleep(5)

    # Check coordinator health
    try:
        resp = requests.get("http://localhost:8080/health", timeout=10)
        if resp.status_code != 200:
            print("Coordinator failed to start")
            coordinator_proc.kill()
            return False
    except Exception as e:
        print(f"Coordinator not responding: {e}")
        coordinator_proc.kill()
        return False

    print("Coordinator ready. Starting training...")

    # Start training
    train_config = {
        "steps": steps,
        "K": K,
        "lr": lr,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "adc_rank": adc_rank,
        "batch_size": batch_size,
        "checkpoint_every": checkpoint_every,
        "use_adc": True,
    }

    try:
        resp = requests.post(
            "http://localhost:8080/train",
            json=train_config,
            timeout=30,
        )
        if resp.status_code == 200:
            print("Training started!")
            print(f"Monitor: curl http://localhost:8080/status")
        else:
            print(f"Failed to start training: {resp.text}")
            coordinator_proc.kill()
            return False
    except Exception as e:
        print(f"Error starting training: {e}")
        coordinator_proc.kill()
        return False

    # Monitor progress
    print(f"\n{'=' * 60}")
    print("Monitoring training progress (Ctrl+C to stop)...")
    print(f"{'=' * 60}\n")

    try:
        while True:
            time.sleep(10)
            try:
                resp = requests.get("http://localhost:8080/status", timeout=10)
                status = resp.json()

                if not status.get("running"):
                    if status.get("error"):
                        print(f"\nError: {status['error']}")
                    else:
                        print(f"\nTraining completed!")
                        print(f"Final loss: {status.get('loss', 'N/A')}")
                    break

                print(
                    f"Step {status['step']}/{status['total_steps']} | "
                    f"Loss: {status['loss']:.4f} | "
                    f"Time: {status['step_time']:.1f}s | "
                    f"ETA: {status.get('eta_seconds', 0) / 60:.1f}min"
                )

            except Exception as e:
                print(f"Status check failed: {e}")

    except KeyboardInterrupt:
        print("\n\nStopping training...")
        try:
            requests.post("http://localhost:8080/stop", timeout=10)
            time.sleep(5)  # Wait for checkpoint
        except:
            pass

    finally:
        coordinator_proc.terminate()
        coordinator_proc.wait()

    return True


def show_status():
    """Show current status of instances and workers."""
    instances = get_vast_instances()

    print(f"\n{'=' * 60}")
    print(f"Vast.ai Instances ({len(instances)} running)")
    print(f"{'=' * 60}\n")

    for inst in instances:
        healthy, info = check_worker_health(inst)
        status_icon = "✓" if healthy else "✗"
        device = info.get("device", "N/A") if info else "N/A"
        tasks = info.get("tasks_completed", 0) if info else 0

        print(f"[{inst.id}] {status_icon} {inst.gpu_name}")
        print(f"    SSH: ssh -p {inst.ssh_port} root@{inst.ssh_host}")
        print(f"    HTTP: http://{inst.public_ip}:{inst.direct_port}")
        print(f"    Device: {device}, Tasks: {tasks}")
        print()

    # Check coordinator
    import requests

    try:
        resp = requests.get("http://localhost:8080/status", timeout=5)
        if resp.status_code == 200:
            status = resp.json()
            print(f"{'=' * 60}")
            print("Coordinator Status")
            print(f"{'=' * 60}")
            print(f"Running: {status.get('running')}")
            print(f"Phase: {status.get('phase')}")
            print(f"Step: {status.get('step')}/{status.get('total_steps')}")
            print(f"Loss: {status.get('loss')}")
    except:
        print("Coordinator not running")


def main():
    parser = argparse.ArgumentParser(description="HoloGrad Distributed Deployment")

    parser.add_argument("--deploy", action="store_true", help="Deploy workers to all instances")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--status", action="store_true", help="Show status")

    # Training config
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--K", type=int, default=512, help="Directions per step")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--n-layer", type=int, default=6, help="Model layers")
    parser.add_argument("--n-head", type=int, default=8, help="Attention heads")
    parser.add_argument("--n-embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--adc-rank", type=int, default=32, help="ADC rank")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if not args.deploy and not args.train:
        parser.print_help()
        return

    instances = get_vast_instances()

    if not instances:
        print("No running Vast.ai instances found!")
        print("Create instances first: vastai create instance ...")
        return

    print(f"Found {len(instances)} running instances")

    if args.deploy:
        results = deploy_all(instances)
        workers = save_workers_config(instances, results)

        success_count = sum(1 for v in results.values() if v)
        print(f"\n{'=' * 60}")
        print(f"Deployment complete: {success_count}/{len(instances)} successful")
        print(f"{'=' * 60}")

        if success_count < 2:
            print("Not enough workers. Aborting.")
            return

    if args.train:
        start_training(
            steps=args.steps,
            K=args.K,
            lr=args.lr,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            adc_rank=args.adc_rank,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
