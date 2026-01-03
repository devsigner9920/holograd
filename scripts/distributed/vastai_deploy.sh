#!/bin/bash
set -e

# Vast.ai Deployment Script for HoloGrad Distributed Training
# Usage: ./vastai_deploy.sh <API_KEY> <NUM_WORKERS> [COORDINATOR_INSTANCE_ID]

API_KEY="${1:?API Key required}"
NUM_WORKERS="${2:-15}"
COORDINATOR_ID="${3:-}"

echo "=========================================="
echo "HoloGrad Vast.ai Deployment"
echo "=========================================="
echo "Workers: $NUM_WORKERS"
echo ""

# Install vastai CLI if not present
if ! command -v vastai &> /dev/null; then
    echo "Installing vastai CLI..."
    pip install vastai
fi

# Set API key
vastai set api-key "$API_KEY"

# Docker image with PyTorch + CUDA
DOCKER_IMAGE="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"

# Search for T4 GPUs with good price
echo "Searching for T4 instances..."
SEARCH_RESULTS=$(vastai search offers \
    --type on-demand \
    --gpu-name "RTX 3090|RTX 4090|T4|A10|L4" \
    --num-gpus 1 \
    --disk 30 \
    --inet-down 100 \
    --reliability 0.95 \
    --order dph \
    --limit 20 \
    2>/dev/null || true)

echo "$SEARCH_RESULTS" | head -20

echo ""
echo "To rent instances, run:"
echo ""
echo "# Rent coordinator (1 instance):"
echo "vastai create instance <OFFER_ID> --image $DOCKER_IMAGE --disk 30"
echo ""
echo "# Rent workers ($NUM_WORKERS instances):"
echo "for i in \$(seq 1 $NUM_WORKERS); do"
echo "    vastai create instance <OFFER_ID> --image $DOCKER_IMAGE --disk 20"
echo "done"
echo ""
echo "# After instances are running, get their IPs:"
echo "vastai show instances"
echo ""
echo "# Then run setup on each instance:"
echo "# On coordinator:"
echo "vastai ssh <COORD_ID> 'git clone https://github.com/YOUR_REPO/holograd && cd holograd && pip install -r requirements.txt pyzmq && python scripts/distributed/run_coordinator.py --workers $NUM_WORKERS --port 5555'"
echo ""
echo "# On each worker:"
echo "vastai ssh <WORKER_ID> 'git clone https://github.com/YOUR_REPO/holograd && cd holograd && pip install -r requirements.txt pyzmq && python scripts/distributed/run_worker.py --coordinator <COORD_IP> --worker-id <ID>'"
