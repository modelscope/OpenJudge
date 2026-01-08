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
SAVE_PATH=./checkpoints/sft
MODEL_PATH=./models/Qwen3-14B   #Qwen/Qwen3-8B
TRAIN_FILE=./data/train.parquet
VAL_FILE=./data/test.parquet

PROJECT_NAME=open_judge-sft
EXPERIMENT_NAME=qwen3-14b-sft-${TIMESTAMP}

# ============================================================================
# Hyperparameters
# ============================================================================
# Data settings
MAX_LENGTH=4096
MICRO_BATCH_SIZE=12  # Per GPU
TRAIN_BATCH_SIZE=96  # Global batch size

# Sequence parallel settings
# For single node with 8 GPUs, use SP=8
# For fewer GPUs, reduce SP accordingly
if [ "$N_GPUS_PER_NODE" -lt 8 ]; then
    SP_SIZE=$N_GPUS_PER_NODE
else
    SP_SIZE=8
fi

# Training settings
TOTAL_EPOCHS=1

# Model settings
ENABLE_GRAD_CKPT=true  # Enable gradient checkpointing to save memory
CPU_OFFLOAD=false  # Enable CPU offload for memory-constrained setups

# ============================================================================
# Run Training with torchrun
# ============================================================================
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.max_length=$MAX_LENGTH \
    data.micro_batch_size=$MICRO_BATCH_SIZE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=null \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=$ENABLE_GRAD_CKPT \
    model.fsdp_config.cpu_offload=$CPU_OFFLOAD \
    model.fsdp_config.model_dtype=bf16 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$SAVE_PATH/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','swanlab'] \
    ulysses_sequence_parallel_size=$SP_SIZE \
    use_remove_padding=true

echo "Training completed! Checkpoints saved to: $SAVE_PATH/$EXPERIMENT_NAME"
