# Training Examples for RM-Gallery

This directory contains training examples for reward models using the VERL framework. The training code is independent from the core evaluation framework.

## Structure

```
examples/train/
├── base.py                 # Base dataset class for training
├── pairwise/              # Pairwise comparison training
│   ├── dataset.py         # Pairwise dataset
│   ├── template.py        # Pairwise template
│   ├── reward_fn.py       # Pairwise reward function
│   └── run_pairwise.sh    # Training script
├── pointwise/             # Pointwise scoring training
│   ├── dataset.py         # Pointwise dataset
│   ├── template.py        # Pointwise template
│   ├── reward_fn.py       # Pointwise reward function
│   └── run_pointwise.sh   # Training script
├── bradley-terry/         # Bradley-Terry preference training
│   ├── dataset.py         # BT dataset
│   ├── trainer.py         # BT trainer
│   └── run_bt.sh          # Training script
└── sft/                   # Supervised fine-tuning
    └── sft_rm.sh          # SFT training script
```

## Training Types

### 1. Pairwise Comparison Training

Train models to compare and rank two responses:

```bash
cd pairwise
bash run_pairwise.sh
```

**Data Format**: Parquet files with columns:
- `input`: List of messages (query)
- `output`: List of responses (response_1, response_2)
- `metadata`: Contains `preferred` field ("A", "B", or "tie")

### 2. Pointwise Scoring Training

Train models to score individual responses:

```bash
cd pointwise
bash run_pointwise.sh
```

**Data Format**: Parquet files with columns:
- `input`: List of messages (query)
- `output`: Response with label (helpfulness score 0-4)
- `data_source`: Source identifier

### 3. Bradley-Terry Training

Train reward models using Bradley-Terry loss:

```bash
cd bradley-terry
bash run_bt.sh
```

**Data Format**: Parquet files with columns:
- `chosen`: Text of chosen response
- `rejected`: Text of rejected response

### 4. Supervised Fine-Tuning

Fine-tune models on multi-turn conversations:

```bash
cd sft
bash sft_rm.sh
```

**Data Format**: Parquet files with `messages` field containing conversation history.

## Data Preparation

The training examples expect data in parquet format. You can convert your data using the data export utilities:

```python
# Example: Export data to parquet
from rm_gallery.gallery.data import export_data

export_data(
    data_samples=your_data,
    output_dir="./exports",
    formats=["parquet"],
    split_ratio={"train": 0.8, "test": 0.2}
)
```

## Configuration

Each training type has configuration files:
- `data_config.yaml`: Data loading configuration
- `runtime_env.yaml`: Python environment for Ray
- Shell scripts contain hyperparameters

Key parameters to adjust:
- `MODEL_PATH`: Path to base model
- `TRAIN_FILE`/`VAL_FILE`: Data file paths
- `data.train_batch_size`: Batch size
- `data.max_prompt_length`: Max prompt length
- `actor_rollout_ref.actor.optim.lr`: Learning rate

## Requirements

Training requires:
- VERL framework (`pip install verl`)
- Ray for distributed training
- Transformers, torch, datasets

See `runtime_env.yaml` files for specific dependencies.

## Notes

- Training code is independent from `rm_gallery.core` evaluation framework
- Uses VERL for distributed training with FSDP/PPO
- Supports multi-node multi-GPU training
- Compatible with Qwen models with thinking capability

