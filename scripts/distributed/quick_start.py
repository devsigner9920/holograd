#!/usr/bin/env python3
import subprocess
import sys
import time
import json


def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_start.py <VASTAI_API_KEY>")
        print("\nGet your API key from: https://cloud.vast.ai/account/")
        sys.exit(1)

    api_key = sys.argv[1]
    num_instances = 16

    print("=" * 60)
    print("HoloGrad Vast.ai Quick Start")
    print("=" * 60)

    print("\n[1/5] Setting API key...")
    run(f"vastai set api-key {api_key}")

    print("\n[2/5] Searching for GPU instances...")
    search_cmd = """vastai search offers --type on-demand --gpu-name "T4" --num-gpus 1 --disk 20 --inet-down 50 --order dph --limit 20 -o"""
    output, code = run(search_cmd)

    if code != 0:
        print(f"Search failed. Make sure vastai CLI is installed: pip install vastai")
        sys.exit(1)

    lines = [l for l in output.split("\n") if l.strip() and not l.startswith("ID")]
    if len(lines) < num_instances:
        print(f"Not enough instances available. Found {len(lines)}, need {num_instances}")
        sys.exit(1)

    offer_ids = []
    for line in lines[:num_instances]:
        parts = line.split()
        if parts:
            offer_ids.append(parts[0])

    print(f"Found {len(offer_ids)} suitable offers")

    print(f"\n[3/5] Renting {num_instances} instances...")
    instance_ids = []
    for i, offer_id in enumerate(offer_ids):
        cmd = f'vastai create instance {offer_id} --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime --disk 20 --onstart-cmd "pip install pyzmq datasets transformers"'
        output, code = run(cmd)
        if code == 0 and "new contract" in output.lower():
            instance_id = output.split()[-1]
            instance_ids.append(instance_id)
            print(f"  Instance {i + 1}/{num_instances}: {instance_id}")
        else:
            print(f"  Failed to create instance {i + 1}: {output}")

    print(f"\n[4/5] Waiting for instances to start...")
    time.sleep(30)

    output, _ = run("vastai show instances --raw")
    try:
        instances = json.loads(output)
        running = [i for i in instances if i.get("actual_status") == "running"]
        print(f"  {len(running)}/{len(instance_ids)} instances running")
    except:
        print("  Could not parse instance status")

    print("\n[5/5] Setup complete!")
    print("\nNext steps:")
    print("1. Wait for all instances to be 'running' (check: vastai show instances)")
    print("2. Get coordinator IP: vastai show instances | head -2")
    print("3. SSH to coordinator and run:")
    print(
        "   vastai ssh <COORD_ID> 'cd ~ && git clone <repo> && cd holograd && python scripts/distributed/run_coordinator.py --workers 15'"
    )
    print("4. SSH to each worker and run:")
    print(
        "   vastai ssh <WORKER_ID> 'cd ~/holograd && python scripts/distributed/run_worker.py --coordinator <COORD_IP> --worker-id <N>'"
    )
    print("\nTo destroy all instances when done:")
    print("   vastai destroy instance <ID>  # for each instance")

    print("\n" + "=" * 60)
    print(f"Created {len(instance_ids)} instances")
    print("=" * 60)


if __name__ == "__main__":
    main()
