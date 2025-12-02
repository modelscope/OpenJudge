# -*- coding: utf-8 -*-
"""Iterative Rubric Generator test module.

This module contains unit tests for the Iterative Rubric Generator functionality
with different configurations and evaluation modes.

Demonstrates workflow:
1. Create generator with configuration
2. Generate rubrics from training data (with labels)
3. Evaluate test data using the generated grader (without labels)

Supports pointwise and listwise evaluation modes.
"""

import asyncio
from copy import deepcopy

import pytest
from loguru import logger

from rm_gallery.core.generator.iterative_rubric.generator import (
    IterativeListwiseRubricsGeneratorConfig,
    IterativePointwiseRubricsGeneratorConfig,
    IterativeRubricsGenerator,
)
from rm_gallery.core.generator.iterative_rubric.query_rubric_generator import (
    LISTWISE_EVALUATION_TEMPLATE,
    POINTWISE_EVALUATION_TEMPLATE,
)
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum

# pylint: disable=line-too-long

# =============================================================================
# Test Data
# =============================================================================

# Pointwise training data (with labels for rubric generation)
POINTWISE_TRAINING_SAMPLE = [
    {
        "query": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "label_score": 1,
    },
]

# Pointwise test data (without labels for evaluation)
POINTWISE_TEST_SAMPLE = [
    {
        "query": "What is the capital of China?",
        "response": "Beijing is the capital of China.",
    },
]

# Listwise training data (with labels for rubric generation)
LISTWISE_TRAINING_SAMPLE = [
    {
        "query": "Write a short story about a robot learning to paint.",
        "responses": [
            "BEEP-7 was a maintenance robot who discovered paint cans in an abandoned art studio. Curious, it dipped its mechanical fingers in blue paint and made its first mark on canvas. Days passed as BEEP-7 experimented with colors, learning that art wasn't about precision but expression. Its circuits hummed with joy as it created its first masterpiece - a sunset that somehow captured the warmth it has never felt.",
            "There was a robot. It painted. The end.",
            "A robot painted a sunset.",
        ],
        "label_rank": [1, 2, 3],
    },
]

# Listwise test data (without labels for evaluation)
LISTWISE_TEST_SAMPLE = [
    {
        "query": "Explain what a for loop does in programming.",
        "responses": [
            "A for loop is a control structure that repeats a block of code a specific number of times. It consists of three parts: initialization (setting a counter), condition (when to stop), and increment (updating the counter). For example, in Python: 'for i in range(5):' will execute the loop body 5 times, with i taking values 0 through 4.",
            "A for loop repeats code multiple times.",
        ],
    },
]


# =============================================================================
# Helper Functions
# =============================================================================


def get_test_model() -> OpenAIChatModel:
    """Get test model instance.

    Returns:
        OpenAIChatModel: Configured OpenAI chat model for testing.
    """
    return OpenAIChatModel(model="qwen3-32b", stream=False)


# =============================================================================
# Pointwise Tests
# =============================================================================


@pytest.mark.asyncio
async def test_iterative_grader_pointwise_without_categorization() -> None:
    """Test pointwise grader generation without categorization.

    This test verifies that a pointwise grader can be generated and used
    for evaluation without enabling rubric categorization.
    """
    model = get_test_model()

    config = IterativePointwiseRubricsGeneratorConfig(
        model=model,
        grader_name="Iterative_Pointwise_Grader",
        custom_evaluation_prompt=POINTWISE_EVALUATION_TEMPLATE,
        min_score=0,
        max_score=1,
        query_specific_generate_number=1,
        enable_categorization=False,
        language=LanguageEnum.EN,
    )

    generator = IterativeRubricsGenerator(config)
    grader = await generator.generate(dataset=POINTWISE_TRAINING_SAMPLE)

    test_query = POINTWISE_TEST_SAMPLE[0]["query"]
    test_response = POINTWISE_TEST_SAMPLE[0]["response"]
    result = await grader.aevaluate(query=test_query, answer=test_response)

    logger.info(f"Pointwise mode without categorization result: {result}")


@pytest.mark.asyncio
async def test_iterative_grader_pointwise_with_categorization() -> None:
    """Test pointwise grader generation with categorization.

    This test verifies that a pointwise grader can be generated with
    LLM-based categorization enabled. Uses larger training dataset to
    trigger smart sampling mode.
    """
    model = get_test_model()

    config = IterativePointwiseRubricsGeneratorConfig(
        model=model,
        grader_name="Iterative_Pointwise_Grader",
        custom_evaluation_prompt=POINTWISE_EVALUATION_TEMPLATE,
        min_score=0,
        max_score=1,
        query_specific_generate_number=1,
        enable_categorization=True,
        categories_number=5,
        language=LanguageEnum.EN,
    )

    generator = IterativeRubricsGenerator(config)
    # Use larger dataset to trigger smart_sampling mode (>100 samples)
    training_data = [deepcopy(POINTWISE_TRAINING_SAMPLE[0]) for _ in range(200)]
    grader = await generator.generate(dataset=training_data)

    test_query = POINTWISE_TEST_SAMPLE[0]["query"]
    test_response = POINTWISE_TEST_SAMPLE[0]["response"]
    result = await grader.aevaluate(query=test_query, answer=test_response)

    logger.info(f"Pointwise mode with categorization result: {result}")


# =============================================================================
# Listwise Tests
# =============================================================================


@pytest.mark.asyncio
async def test_iterative_grader_listwise() -> None:
    """Test listwise grader generation.

    This test verifies that a listwise grader can be generated and used
    for ranking multiple responses.
    """
    model = get_test_model()

    config = IterativeListwiseRubricsGeneratorConfig(
        model=model,
        grader_name="Iterative_Listwise_Grader",
        custom_evaluation_prompt=LISTWISE_EVALUATION_TEMPLATE,
        enable_categorization=False,
        language=LanguageEnum.EN,
        categories_number=5,
        query_specific_generate_number=2,
    )

    generator = IterativeRubricsGenerator(config)
    grader = await generator.generate(dataset=LISTWISE_TRAINING_SAMPLE)

    test_query = LISTWISE_TEST_SAMPLE[0]["query"]
    test_responses = LISTWISE_TEST_SAMPLE[0]["responses"]
    answer = "\n".join([f"Answer {i+1}: {ans}" for i, ans in enumerate(test_responses)])
    num_responses = len(test_responses)

    result = await grader.aevaluate(
        query=test_query,
        answer=answer,
        num_responses=num_responses,
    )

    logger.info(f"Listwise mode result: {result}")


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run all test functions."""
    await test_iterative_grader_pointwise_without_categorization()
    await test_iterative_grader_pointwise_with_categorization()
    await test_iterative_grader_listwise()


if __name__ == "__main__":
    asyncio.run(main())
