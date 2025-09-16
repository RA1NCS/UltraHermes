#!/usr/bin/env bash
set -euo pipefail

# --- Environment for ALL worker processes ---
export CUDA_VISIBLE_DEVICES=0,1
export ACCELERATE_CONFIG_FILE=/workspace/.cache/huggingface/accelerate/default_config.yml

# Tokenizers & dataloaders: avoid fork deadlocks
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1

# NCCL: single-node, no Infiniband
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Early Failure Visibility
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NCCL_SOCKET_IFNAME=eth0



LOG=logs/training_pref.log

# Launch in background and survive disconnects
nohup stdbuf -oL -eL accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" 2-pref.py \
  > "$LOG" --report_to mlflow 2>&1 &

PID=$!
echo "🚀 Training started (PID: $PID)"
echo "📝 Logs: tail -f $LOG"
echo "🛑 Stop: kill $PID"
echo "$PID" > training.pid
