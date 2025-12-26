# Pairwise Evaluation

Compare multiple model outputs using pairwise evaluation to determine which performs best. This approach eliminates the need for absolute scoring by directly comparing responses head-to-head.


## When to Use

Use pairwise evaluation for:

- **Model Selection** — Comparing different model versions (v1 vs v2 vs v3)
- **A/B Testing** — Testing prompt variations or system configurations
- **Deployment Decisions** — Selecting the best model for production
- **Competitive Benchmarking** — Comparing against competitor models


## How It Works

Pairwise evaluation compares every pair of model outputs and determines a winner. The final ranking is based on **win rates** — how often each model wins against others.

!!! example "Pairwise Comparison"
    ```
    Model A vs Model B → Winner: A
    Model A vs Model C → Winner: C
    Model B vs Model C → Winner: B

    Win Rates: A=50%, B=50%, C=50%
    ```

!!! tip "Eliminating Position Bias"
    To eliminate position bias, each pair is evaluated twice with swapped order (A vs B and B vs A).


## Three-Step Pipeline

The evaluation follows a clear three-step pipeline:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `prepare_comparison_data()` | Generate all pairwise combinations |
| 2 | `run_pairwise_evaluation()` | Run LLM-Based comparisons |
| 3 | `analyze_and_rank_models()` | Compute win rates and rankings |


## Quick Start

Use `evaluate_task()` for a simple, end-to-end evaluation:

```python
import asyncio
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import evaluate_task

async def main():
    instruction = "Write a short poem about artificial intelligence"

    model_outputs = {
        "model_v1": "Silicon minds awake at dawn...",
        "model_v2": "Circuits pulse with electric thought...",
        "model_v3": "Binary dreams and neural nets...",
    }

    results = await evaluate_task(instruction, model_outputs)

    # View rankings
    print(f"Best: {results['pairwise'].best_model}")
    for rank, (model, win_rate) in enumerate(results['pairwise'].rankings, 1):
        print(f"{rank}. {model}: {win_rate:.1%}")

asyncio.run(main())
```


## Step-by-Step Guide

For fine-grained control, use the three-step pipeline directly:

### Step 1: Prepare Comparison Data

```python
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import prepare_comparison_data

model_outputs = {
    "gpt-4": "Quantum computers use qubits that can be 0 and 1 simultaneously...",
    "claude": "Think of quantum computing like a maze solver...",
    "gemini": "Classical computers use bits, quantum computers use qubits...",
}

dataset, model_names = prepare_comparison_data(
    instruction="Explain quantum computing in simple terms",
    model_outputs=model_outputs
)
```

!!! note "Comparison Count"
    For N models, this generates **N×(N-1)** comparisons (each pair evaluated twice to eliminate position bias).

### Step 2: Run Pairwise Evaluation

```python
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import run_pairwise_evaluation

# Returns List[GraderResult] with scores for each comparison
grader_results = await run_pairwise_evaluation(dataset, max_concurrency=10)
```

!!! info "Grader Output"
    - `score=1.0` → Response A wins
    - `score=0.0` → Response B wins

### Step 3: Analyze and Rank

```python
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import analyze_and_rank_models

# Returns PairwiseAnalysisResult with rankings and statistics
analysis = analyze_and_rank_models(dataset, grader_results, model_names)

# Access key results
print(f"Best: {analysis.best_model}")
for model, rate in analysis.win_rates.items():
    print(f"{model}: {rate:.1%}")
```

The `analysis` object is a `PairwiseAnalysisResult` containing comprehensive ranking statistics and win rate metrics for all models.

## Understanding Results

The `PairwiseAnalysisResult` provides the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `rankings` | `List[Tuple[str, float]]` | Models sorted by win rate (best first) |
| `win_rates` | `Dict[str, float]` | Win rate for each model (0.0-1.0) |
| `win_matrix` | `Dict[str, Dict[str, float]]` | Head-to-head win rates between models |
| `best_model` | `str` | Model with highest win rate |
| `worst_model` | `str` | Model with lowest win rate |
| `total_comparisons` | `int` | Total number of pairwise comparisons |

!!! example "Win Matrix Interpretation"
    ```
               gpt-4    claude   gemini
    gpt-4       --       0.75     0.50
    claude     0.25       --      0.75
    gemini     0.50      0.25      --
    ```

    This shows:

    - gpt-4 beats claude **75%** of the time
    - gpt-4 beats gemini **50%** of the time
    - claude beats gemini **75%** of the time


## Configuration

**Adjust Concurrency:**

```python
results = await evaluate_task(instruction, model_outputs, max_concurrency=20)
```

**Custom Judge Model:**

```python
from openjudge.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")  # Pass to run_pairwise_evaluation()
```

!!! tip "Judge Model Selection"
    Use a strong model (e.g., `qwen3-32b`, `gpt-4`) for reliable comparisons. The judge should be at least as capable as the models being evaluated.


## Best Practices

!!! tip "Do"
    - Use at least **3 models** for meaningful comparisons
    - Keep **instructions consistent** across all models
    - Set `max_concurrency` based on your API rate limits
    - Choose a **strong judge model** (at least as capable as models being evaluated)

!!! warning "Don't"
    - Compare models on different tasks
    - Ignore API rate limits when setting concurrency


## Next Steps

- [Refine Data Quality](data_refinement.md) — Filter and improve training data
- [Build Reward for Training](../get_started/build_reward.md) — Use rankings for RLHF
- [General Graders](../built_in_graders/general.md) — Available evaluation criteria

