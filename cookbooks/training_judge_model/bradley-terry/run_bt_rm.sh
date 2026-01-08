#!/bin/bash
set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# ============================================================================
# Distributed Training Configuration
# ============================================================================
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

echo "=== Distributed Training Configuration ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "N_NODES: $NNODES"
echo "N_GPUS_PER_NODE: $N_GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"
echo "=========================================="

# ============================================================================
# Training Configuration
# ============================================================================
SAVE_PATH=./checkpoints/bt
MODEL_PATH=./models/Qwen3-8B
TRAIN_FILE=./data/train.parquet
VAL_FILE=./data/test.parquet

PROJECT_NAME=open_judge-bt
EXPERIMENT_NAME=qwen3-8b-bt-${TIMESTAMP}

# ============================================================================
# Hyperparameters (aligned with VERL best practices)
# ============================================================================
# Data settings
MAX_LENGTH=4096
MICRO_BATCH_SIZE=1  # Per GPU, for BT pairs
TRAIN_BATCH_SIZE=256  # Global batch size

# Optimizer settings (VERL standard)
LR=2e-6
WEIGHT_DECAY=0.001
LR_WARMUP_RATIO=0.03
CLIP_GRAD=2.0
LR_SCHEDULER=cosine  # cosine, wsd, or constant

# Training settings
TOTAL_EPOCHS=3
SAVE_FREQ=500
TEST_FREQ=500

# Model settings
ENABLE_GRAD_CKPT=False  # Enable gradient checkpointing to save memory
USE_LIGER=False  # Use Liger kernel optimizations
STRATEGY=fsdp2  # fsdp or fsdp2

# ============================================================================
# Run Training with torchrun
# ============================================================================
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./trainer.py \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.max_length=$MAX_LENGTH \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=$ENABLE_GRAD_CKPT \
    model.use_liger=$USE_LIGER \
    model.strategy=$STRATEGY \
    optim.lr=$LR \
    optim.weight_decay=$WEIGHT_DECAY \
    optim.warmup_steps_ratio=$LR_WARMUP_RATIO \
    optim.clip_grad=$CLIP_GRAD \
    optim.lr_scheduler=$LR_SCHEDULER \
    trainer.device=cuda \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.logger=['console','swanlab'] \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.project_name=$PROJECT_NAME \
    trainer.default_local_dir=$SAVE_PATH/$EXPERIMENT_NAME

echo "Training completed! Checkpoints saved to: $SAVE_PATH/$EXPERIMENT_NAME"

