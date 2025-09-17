#!/usr/bin/env bash
set -euo pipefail

# --- Environment for ALL worker processes ---
# CUDA and device configuration
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Accelerate and distributed training configuration
export ACCELERATE_CONFIG_FILE=/workspace/.cache/huggingface/accelerate/default_config.yml
export ACCELERATE_USE_FSDP=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Tokenizer and dataset settings
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1

# NCCL communication settings
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo

# Torch distributed and error handling
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# TorchDynamo settings
export TORCHDYNAMO_DISABLE=1

# MLflow settings
export MLFLOW_TRACKING_URI=file:/workspace/outputs/mlflow_runs
export MLFLOW_EXPERIMENT_NAME=Default
export HF_MLFLOW_LOG_ARTIFACTS=true

LOG=logs/training_pref.log

# Launch in background and survive disconnects
nohup stdbuf -oL -eL accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" 2-pref.py \
  > "$LOG" 2>&1 &

PID=$!
echo "🚀 Training started (PID: $PID)"
echo "📝 Logs: tail -f $LOG"
echo "🛑 Stop: kill $PID"
echo "$PID" > training.pid
