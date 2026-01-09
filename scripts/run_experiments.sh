#!/bin/bash
set -e

STEPS=5000
MODEL_SIZE="medium"
BATCH_SIZE=8
SEQ_LENGTH=256
LR=3e-4
LOG_INTERVAL=100
SAVE_DIR="results/benchmark"

echo "=============================================="
echo "HoloGrad NeurIPS Benchmark Experiments"
echo "=============================================="
echo "Model: GPT-2 ${MODEL_SIZE}"
echo "Steps: ${STEPS}"
echo "Batch: ${BATCH_SIZE}"
echo "Save: ${SAVE_DIR}"
echo "=============================================="

mkdir -p ${SAVE_DIR}

echo ""
echo "[1/4] Running Full SGD baseline..."
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
echo "[2/4] Running HoloGrad (K=64)..."
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
echo "[3/4] Running HoloGrad (K=32)..."
python benchmarks/compare_methods.py \
    --method holograd \
    --model_size ${MODEL_SIZE} \
    --steps ${STEPS} \
    --batch_size ${BATCH_SIZE} \
    --seq_length ${SEQ_LENGTH} \
    --lr ${LR} \
    --holograd_k 32 \
    --log_interval ${LOG_INTERVAL} \
    --save_dir ${SAVE_DIR}

echo ""
echo "[4/4] Running HoloGrad-Momentum..."
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
echo "=============================================="
echo "Generating comparison plots..."
echo "=============================================="
python benchmarks/plot_comparison.py \
    --results_dir ${SAVE_DIR} \
    --output_dir figures

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results: ${SAVE_DIR}/"
echo "Figures: figures/"
echo "=============================================="
