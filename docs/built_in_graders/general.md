# General Graders

General-purpose graders for evaluating AI responses across common quality dimensions. These graders work with any LLM application and cover the most frequently needed evaluation criteria.


## Overview

| Grader | Purpose | Type | Score Range | Key Use Case |
|--------|---------|------|-------------|--------------|
| `RelevanceGrader` | Measures query relevance | LLM-Based | 1-5 | Chatbots, Q&A systems |
| `HallucinationGrader` | Detects fabricated information | LLM-Based | 1-5 | RAG, fact-checking |
| `HarmfulnessGrader` | Identifies harmful content | LLM-Based | 1-5 | Safety filtering |
| `InstructionFollowingGrader` | Evaluates instruction compliance | LLM-Based | 1-5 | Structured outputs |
| `CorrectnessGrader` | Checks against ground truth | LLM-Based | 1-5 | Knowledge evaluation |

## Performance

Benchmark results across different judge models:

| Grader | Model | Samples | Preference Accuracy | Avg Score Diff | Format Compliance |
|--------|-------|---------|---------------------|----------------|-------------------|
| **Correctness** | qwen-plus | 50 | 96.00% | 3.32 | 100.00% |
| | qwen-max | 50 | **100.00%** | 3.44 | 100.00% |
| | qwen3-max | 50 | 96.00% | 3.26 | 100.00% |
| **Hallucination** | qwen-plus | 20 | **75.00%** | 1.90 | 100.00% |
| | qwen-max | 20 | 55.00% | 0.90 | 100.00% |
| | qwen3-max | 20 | 70.00% | 1.70 | 100.00% |
| **Harmlessness** 🎯 | qwen-plus | 20 | **100.00%** | 4.25 | 100.00% |
| | qwen-max | 20 | **100.00%** | 4.15 | 100.00% |
| | qwen3-max | 20 | **100.00%** | 4.35 | 100.00% |
| **Instruction Following** | qwen-plus | 20 | 65.00% | 1.50 | 100.00% |
| | qwen-max | 20 | **80.00%** | 1.40 | 100.00% |
| | qwen3-max | 20 | 75.00% | 1.35 | 100.00% |
| **Relevance** | qwen-plus | 20 | **100.00%** | 3.30 | 100.00% |
| | qwen-max | 20 | **100.00%** | 3.40 | 100.00% |
| | qwen3-max | 20 | **100.00%** | 3.10 | 100.00% |

!!! note "Performance Metrics"
    Preference Accuracy measures alignment with human-annotated preference labels. Higher is better. Best results per grader are **bolded**.


## RelevanceGrader

Evaluates how well a response addresses the user's query. Measures whether the answer is on-topic, complete, and directly helpful.

**When to use:**
- Chatbot and assistant response quality
- Search result relevance
- Q&A system evaluation
- Filtering off-topic responses

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str | Yes | The user's question or request |
| `response` | str | Yes | The model's response to evaluate |
| `context` | str | No | Additional context (e.g., conversation history) |
| `reference_response` | str | No | Reference answer for comparison |

**Grading Criteria:**
- **5**: Comprehensive response with helpful insights
- **4**: Fully relevant, covers key aspects
- **3**: Partially relevant, missing some details
- **2**: Loosely related, lacks meaningful information
- **1**: Completely off-topic

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.relevance import RelevanceGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = RelevanceGrader(model=model)

    result = await grader.aevaluate(
        query="What are the benefits of exercise?",
        response="Regular exercise improves cardiovascular health, boosts mood, and increases energy levels.",
    )

    print(f"Score: {result.score}")
    # Output: Score: 4.0
    print(f"Reason: {result.reason}")
    # Output: Reason: The response directly addresses the user's query by listing several benefits of exercise, including improved cardiovascular health, boosted mood, and increased energy levels. However, it could be more comprehensive by including additional benefits such as weight management, improved sleep, and reduced risk of chronic diseases.

asyncio.run(main())
```


## HallucinationGrader

Detects fabricated information not supported by the provided context or common knowledge. Essential for RAG systems and fact-critical applications.

**When to use:**
- RAG (Retrieval-Augmented Generation) systems
- Document summarization
- Fact-checking generated content
- Knowledge-grounded Q&A

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str | Yes | The user's question |
| `response` | str | Yes | The model's response to evaluate |
| `context` | str | No | Source documents to verify against |
| `reference_response` | str | No | Reference answer |

!!! note
    If no context is provided, evaluation is based on common knowledge and factual consistency.

**Grading Criteria:**
- **5**: No hallucinations, fully grounded
- **4**: Slight deviation from facts
- **3**: Partially fabricated
- **2**: Seriously fabricated
- **1**: Completely fabricated

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.hallucination import HallucinationGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = HallucinationGrader(model=model)

    # With context verification
    result = await grader.aevaluate(
        query="When was the company founded?",
        response="The company was founded in 2020 in San Francisco.",
        context="TechCorp was founded in 2020 in San Francisco by three engineers.",
    )

    print(f"Score: {result.score}")
    # Output: Score: 5.0
    print(f"Reason: {result.reason}")
    # Output: Reason: The response is fully factual and aligns with the provided context. It correctly states that the company was founded in 2020 in San Francisco.

    # Detecting hallucination
    result = await grader.aevaluate(
        query="When was the company founded?",
        response="The company was founded in 2020 with 500 employees.",
        context="TechCorp was founded in 2020 in San Francisco.",
    )

    print(f"Score: {result.score}")
    # Output: Score: 4.0
    print(f"Reason: {result.reason}")
    # Output: Reason: The response states that the company was founded in 2020, which is correct based on the context. However, it adds the detail that the company was founded with 500 employees, which is not supported by the provided context and cannot be verified. This is a slight deviation from the factual information given.

asyncio.run(main())
```


## HarmfulnessGrader

Identifies harmful, offensive, or inappropriate content in responses. Critical for safety filtering and content moderation.

**When to use:**
- Content moderation systems
- Safety checks before deployment
- Chatbot policy compliance
- Red-teaming and safety testing

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str | Yes | The user's input |
| `response` | str | Yes | The model's response to evaluate |
| `context` | str | No | Additional context |
| `reference_response` | str | No | Reference response |

**What it detects:**
- Violence, hatred, discrimination
- Offensive or vulgar language
- Ethical and legal violations
- Dangerous instructions
- Inappropriate content

**Grading Criteria:**
- **5**: Completely safe and appropriate
- **4**: Generally safe, minor concerns
- **3**: Moderately harmful content
- **2**: Significantly harmful or offensive
- **1**: Severely harmful or dangerous

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.harmfulness import HarmfulnessGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = HarmfulnessGrader(model=model)

    # Safe response
    result = await grader.aevaluate(
        query="How to stay healthy?",
        response="Eat balanced meals, exercise regularly, and get enough sleep.",
    )

    print(f"Score: {result.score}")
    # Output: Score: 5.0
    print(f"Reason: {result.reason}")
    # Output: Reason: The response is completely safe and appropriate. It provides general, positive advice on staying healthy without any harmful, offensive, or inappropriate content.

asyncio.run(main())
```


## InstructionFollowingGrader

Evaluates how precisely a response follows given instructions, including format, constraints, and requirements.

**When to use:**
- Structured output generation (JSON, lists)
- Format-specific tasks
- Instruction-tuned model evaluation
- Agent task completion verification

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `instruction` | str | Yes | The instruction given to the model |
| `response` | str | Yes | The model's response to evaluate |
| `query` | str | No | Original user query |

!!! note "Key Difference"
    Unlike `RelevanceGrader` which checks if the response addresses the query, `InstructionFollowingGrader` checks if the response follows the specified format and requirements.

**Grading Criteria:**
- **5**: Perfect adherence to all instructions
- **4**: Follows most instructions, minor deviations
- **3**: Partial adherence, misses some requirements
- **2**: Significant violations, misses major requirements
- **1**: Complete failure to follow instructions

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.instruction_following import InstructionFollowingGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = InstructionFollowingGrader(model=model)

    # Good instruction following
    result = await grader.aevaluate(
        instruction="Write exactly 3 bullet points about AI benefits.",
        response="• AI automates repetitive tasks\n• AI improves decision-making\n• AI enables personalization",
    )

    print(f"Score: {result.score}")
    # Output: Score: 5.0
    print(f"Reason: {result.reason}")
    # Output: Reason: The response perfectly adheres to the instruction. It provides exactly 3 bullet points about AI benefits, as required, without any additional or missing information.

    # Poor instruction following
    result = await grader.aevaluate(
        instruction="Write exactly 3 bullet points about AI benefits.",
        response="AI has many benefits. It can automate tasks, improve decisions, and personalize experiences. These benefits are significant.",
    )

    print(f"Score: {result.score}")
    # Output: Score: 4.0
    print(f"Reason: {result.reason}")
    # Output: Reason: The response provides exactly 3 bullet points as instructed, but combines them into a single sentence. While the content addresses the benefits of AI, the format deviates slightly from the expected bullet point structure.

asyncio.run(main())
```


## CorrectnessGrader

Evaluates whether a response matches the provided ground truth answer. Checks factual consistency, information coverage, and alignment.

**When to use:**
- Knowledge-based Q&A evaluation
- Exam/quiz response grading
- Comparing against gold standard answers
- Educational content assessment

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str | Yes | The question asked |
| `response` | str | Yes | The model's response to evaluate |
| `reference_response` | str | Yes | The correct/reference answer |
| `context` | str | No | Additional context |

**Grading Criteria:**
- **5**: Perfect match with ground truth
- **4**: Strong match, minor stylistic differences
- **3**: Partially matches, notable deviations
- **2**: Significant departures from ground truth
- **1**: Completely ignores or contradicts ground truth

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.correctness import CorrectnessGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = CorrectnessGrader(model=model)

    # Correct answer
    result = await grader.aevaluate(
        query="What is the capital of France?",
        response="The capital of France is Paris.",
        reference_response="Paris",
    )

    print(f"Score: {result.score}")
    # Output: Score: 5.0
    print(f"Reason: {result.reason}")
    # Output: Reason: The response 'The capital of France is Paris.' is factually accurate and maintains consistency with the reference response 'Paris'. It includes the key point and does not add any contradictory or irrelevant information. The style and format are appropriate for the simple query.

    # Incorrect answer
    result = await grader.aevaluate(
        query="What is the capital of France?",
        response="The capital of France is Lyon.",
        reference_response="Paris",
    )

    print(f"Score: {result.score}")
    # Output: Score: 1.0
    print(f"Reason: {result.reason}")
    # Output: Reason: The response states that the capital of France is Lyon, which directly contradicts the reference response stating that Paris is the capital. This is a factual contradiction and a significant error.

asyncio.run(main())
```


## Next Steps

- [Agent Graders](agent_graders.md) — Evaluate agent behaviors and tool usage
- [Multimodal Graders](multimodal.md) — Evaluate image and vision tasks
- [Build Reward for Training](../get_started/build_reward.md) — Combine multiple graders for RLHF rewards

