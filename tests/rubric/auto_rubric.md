# AutoRubrics - Automated Evaluation Rubric Generation System

AutoRubrics is an LLM-based automated evaluation rubric generation and optimization system. It automatically creates high-quality, interpretable evaluation rubrics through iterative generation and information-theoretic optimization.

## 🌟 Core Features

### 1. **Dual Generation Modes**
- **Single Mode**: Generate and iteratively optimize rubrics independently for each sample
- **Batch Mode**: Use MCR² (Maximal Coding Rate Reduction) algorithm to select optimal rubric subsets from large sample collections

### 2. **Multiple Evaluation Modes**
- **Pointwise**: Score individual responses independently (e.g., 0-1, 0-4 scale)
- **Listwise**: Rank and compare multiple responses (e.g., rank 1, 2, 3)

### 3. **Multi-language Support**
- Supports both Chinese (ZH) and English (EN) rubric generation
- Automatically adapts prompt templates for different languages

### 4. **Intelligent Aggregation**
- **Keep All Mode**: Retain all generated rubrics
- **Categorize Mode**: Use LLM to merge similar rubrics into Theme-Tips format

### 5. **Information-Theoretic Optimization (MCR²)**
- Based on Maximal Coding Rate Reduction
- Automatically identifies rubric subsets with maximum information content
- Avoids redundancy and improves evaluation efficiency
- Smart early stopping mechanism for optimal information-cost balance

### 6. **Iterative Refinement**
- Automatically validates if generated rubrics match annotations
- Iteratively optimizes based on validation feedback
- Configurable maximum iteration count

## 🚀 Quick Start

### Data Preparation

Data must conform to the `DataSample` format:

```python
from rm_gallery.core.data import DataSample

# Pointwise example
pointwise_sample = DataSample(
    data={
        "query": "What is 2 + 2?",
        "min_score": 0,
        "max_score": 1,
    },
    samples=[
        {
            "response": "2 + 2 = 4",
            "score": 1,  # Annotation score
        }
    ]
)

# Listwise example
listwise_sample = DataSample(
    data={
        "query": "Compare these responses",
    },
    samples=[
        {
            "response": "Response A: Very detailed answer...",
            "rank": 3,  # Highest quality
        },
        {
            "response": "Response B: Brief answer...",
            "rank": 1,  # Lowest quality
        },
        {
            "response": "Response C: Good answer...",
            "rank": 2,  # Medium quality
        }
    ]
)
```

### Single Mode

Use case: Small sample size (< 100), need customized rubrics for each sample

```python
import asyncio
from rm_gallery.core.runner.auto_rubrics import AutoRubrics, AggregationMode
from rm_gallery.core.grader import GraderMode
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.model.template import LanguageEnum

async def run_single_mode():
    # Initialize LLM
    llm = OpenAIChatModel(model_name="qwen3-32b", stream=False)

    # Create AutoRubrics (Single Mode)
    auto_rubrics = AutoRubrics.create(
        llm=llm,
        generation_mode=GenerationMode.SINGLE,
        evaluation_mode=GraderMode.POINTWISE,  # or GraderMode.LISTWISE
        language=LanguageEnum.EN,  # or LanguageEnum.ZH
        generate_number=3,  # Generate 3 rubrics per sample
        max_epochs=3,  # Maximum 3 iterations
        min_score=0,  # Pointwise: minimum score
        max_score=1,  # Pointwise: maximum score
        aggregation_mode=AggregationMode.CATEGORIZE,  # Aggregation mode
        merge_num_categories=5,  # Merge into 5 categories
    )

    # Load test samples
    samples = load_your_samples()  # Implement your data loading

    # Run generation
    results = await auto_rubrics.run(samples)

    # View results
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Number of Rubrics: {results['final_rubric_count']}")
    print("\nGenerated Rubrics:")
    for i, rubric in enumerate(results['final_rubrics'][:5]):
        print(f"{i+1}. {rubric}")

    return results

asyncio.run(run_single_mode())
```

### Batch Mode

Use case: Large sample size (> 100), need to extract optimal rubric set from many samples

```python
import asyncio
from rm_gallery.core.runner.auto_rubrics import AutoRubrics, AggregationMode
from rm_gallery.core.grader import GraderMode
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.model.template import LanguageEnum

async def run_batch_mode():
    # Initialize LLM
    llm = OpenAIChatModel(model_name="qwen3-32b", stream=False)

    # Create AutoRubrics (Batch Mode)
    auto_rubrics = AutoRubrics.create(
        llm=llm,
        generation_mode=GenerationMode.BATCH,
        evaluation_mode=GraderMode.POINTWISE,
        language=LanguageEnum.EN,
        batch_size=10,  # Process 10 samples per batch
        mcr_batch_size=10,  # MCR selects 10 rubrics per iteration
        min_score=0,
        max_score=1,
        max_iterations=50,  # Maximum 50 iterations
        min_increment_threshold=0.001,  # Information gain threshold
        max_total_rubrics=100,  # Keep maximum 100 rubrics
        aggregation_mode=AggregationMode.CATEGORIZE,
        merge_num_categories=5,
    )

    # Load large sample set
    samples = load_your_samples()  # Implement your data loading

    # Run batch generation
    results = await auto_rubrics.run(samples)

    # View results
    print(f"Total Samples: {results['total_samples']}")
    print(f"Total Iterations: {results['total_iterations']}")
    print(f"Final Coding Rate: {results['final_coding_rate']:.6f}")
    print(f"Number of Rubrics: {results['final_rubric_count']}")

    # View iteration history
    print("\nIteration History:")
    for item in results['iteration_history'][-5:]:
        print(f"  Iteration {item['iteration']}: "
              f"samples {item['batch_start']}-{item['batch_end']} → "
              f"generated {item['new_generated']} → "
              f"total {item['total_selected']} "
              f"(rate: {item['coding_rate']:.6f}, gain: {item['increment']:+.6f})")

    return results

asyncio.run(run_batch_mode())
```

## 🔧 Configuration Parameters

### AutoRubricsConfig

```python
from rm_gallery.core.runner.auto_rubrics import AutoRubricsConfig, GenerationMode, AggregationMode
from rm_gallery.core.grader import GraderMode
from rm_gallery.core.model.template import LanguageEnum

config = AutoRubricsConfig(
    # === Core Configuration ===
    generation_mode=GenerationMode.SINGLE,  # SINGLE or BATCH
    evaluation_mode=GraderMode.POINTWISE,   # POINTWISE or LISTWISE
    language=LanguageEnum.ZH,                # ZH (Chinese) or EN (English)

    # === Generation Parameters ===
    generate_number=3,      # Number of rubrics per sample (1-10)
    max_retries=5,          # Maximum LLM API retry attempts (1-20)
    max_epochs=3,           # Maximum iteration rounds per sample (1-10)

    # === Pointwise Parameters ===
    min_score=0,            # Minimum score
    max_score=1,            # Maximum score (must be >= 1)

    # === Batch Mode Parameters ===
    batch_size=10,          # Number of samples per batch iteration
    mcr_batch_size=10,      # Number of rubrics selected by MCR² per iteration
    min_increment_threshold=0.002,  # Information gain threshold (lower = stricter)
    patience=2,             # Consecutive low-gain tolerance count
    max_iterations=50,      # Maximum batch iterations
    max_total_rubrics=200,  # Maximum rubrics pool capacity

    # === Aggregation Parameters ===
    aggregation_mode=AggregationMode.KEEP_ALL,  # KEEP_ALL or CATEGORIZE
    merge_num_categories=5,  # Number of categories for CATEGORIZE mode (2-20)
)
```


### Single Mode Output

```python
{
    "generation_mode": "single",
    "evaluation_mode": "pointwise",
    "aggregation_mode": "categorize",
    "total_samples": 100,
    "successful_samples": 95,
    "failed_samples": 5,
    "success_rate": 95.0,
    "total_rubrics": 285,
    "final_rubrics": [
        "Theme: Accuracy and Completeness\n- Tip1: ...\n- Tip2: ...",
        "Theme: Clarity and Structure\n- Tip1: ...\n- Tip2: ...",
        ...
    ],
    "final_rubric_count": 5,
    "aggregation_info": {
        "method": "categorization",
        "original_count": 285,
        "num_categories": 5
    },
    "sample_results": [...]  # Detailed results for each sample
}
```

### Batch Mode Output

```python
{
    "generation_mode": "batch",
    "evaluation_mode": "pointwise",
    "total_samples": 400,
    "final_rubrics": [...],
    "final_rubric_count": 70,
    "total_iterations": 7,
    "final_coding_rate": 36.534626,
    "iteration_history": [
        {
            "iteration": 1,
            "batch_start": 0,
            "batch_end": 9,
            "batch_size": 10,
            "new_generated": 18,
            "total_selected": 18,
            "coding_rate": 50.124856,
            "increment": 50.124856,
            "generation_stats": {...}
        },
        ...
    ],
    "coding_rates": [0.0, 50.124856, 45.964881, ...],
    "aggregation_info": {...}
}
```

