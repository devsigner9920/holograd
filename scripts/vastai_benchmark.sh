#!/bin/bash
set -e

echo "=============================================="
echo "HoloGrad NeurIPS Benchmark - Vast.ai Setup"
echo "=============================================="

cd /workspace

if [ ! -d "holograd" ]; then
    echo "[1/4] Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/holograd.git || {
        echo "Git clone failed. Uploading code manually..."
        mkdir -p holograd
    }
fi

cd holograd

echo "[2/4] Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers numpy matplotlib

echo "[3/4] Verifying GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

echo "[4/4] Running benchmark experiments..."
echo ""

STEPS=2000
MODEL_SIZE="medium"
BATCH_SIZE=16
SEQ_LENGTH=512
LR=3e-4
LOG_INTERVAL=100
SAVE_DIR="results/benchmark_vastai"

mkdir -p ${SAVE_DIR}

echo "Config: ${MODEL_SIZE} model, ${STEPS} steps, batch=${BATCH_SIZE}, seq=${SEQ_LENGTH}"
echo ""

echo "=== [1/3] Full SGD baseline ==="
python benchmarks/compare_methods.py \
    --method full_sgd \
    --model_size ${MODEL_SIZE} \
    --steps ${STEPS} \
    --batch_size ${BATCH_SIZE} \
    --seq_length ${SEQ_LENGTH} \
    --lr ${LR} \
    --log_interval ${LOG_INTERVAL} \
    --save_dir ${SAVE_DIR}

echo ""
echo "=== [2/3] HoloGrad (K=64) ==="
python benchmarks/compare_methods.py \
    --method holograd \
    --model_size ${MODEL_SIZE} \
    --steps ${STEPS} \
    --batch_size ${BATCH_SIZE} \
    --seq_length ${SEQ_LENGTH} \
    --lr ${LR} \
    --holograd_k 64 \
    --log_interval ${LOG_INTERVAL} \
    --save_dir ${SAVE_DIR}

echo ""
echo "=== [3/3] HoloGrad-Momentum ==="
python benchmarks/compare_methods.py \
    --method holograd_momentum \
    --model_size ${MODEL_SIZE} \
    --steps ${STEPS} \
    --batch_size ${BATCH_SIZE} \
    --seq_length ${SEQ_LENGTH} \
    --lr ${LR} \
    --log_interval ${LOG_INTERVAL} \
    --save_dir ${SAVE_DIR}

echo ""
echo "=== Generating plots ==="
python benchmarks/plot_comparison.py \
    --results_dir ${SAVE_DIR} \
    --output_dir figures

echo ""
echo "=============================================="
echo "Benchmark completed!"
echo "Results: ${SAVE_DIR}/"
echo "Figures: figures/"
echo "=============================================="

cat ${SAVE_DIR}/*.json | python -c "
import sys, json, os
print('\n=== RESULTS SUMMARY ===')
print(f'{\"Method\":25s} | {\"Loss\":>8s} | {\"PPL\":>10s} | {\"Bits/step\":>12s} | {\"Time\":>8s}')
print('-' * 75)
for f in sorted(os.listdir('${SAVE_DIR}')):
    if f.endswith('.json'):
        with open(f'${SAVE_DIR}/{f}') as fp:
            d = json.load(fp)
            print(f'{d[\"method\"]:25s} | {d[\"final_loss\"]:8.4f} | {d[\"final_perplexity\"]:10,.0f} | {d[\"bits_per_step\"]/1e6:10.4f} M | {d[\"total_time\"]:6.1f}s')
"
