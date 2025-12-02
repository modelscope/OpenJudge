# -*- coding: utf-8 -*-
"""AutoRubrics test module.

This module tests the iterative rubric generation workflow:
1. Generate rubrics based on annotation information
2. Evaluate samples using generated rubrics
3. Verify results match annotations
4. If not matched, revise rubrics and retry
5. Repeat until validation passes or maximum iterations reached

Supports pointwise and listwise evaluation modes.
Note: Pairwise is a special case of listwise (2 responses).
"""
import asyncio
from typing import List

from loguru import logger

from rm_gallery.core.generator.auto_rubric_generator import (
    ListwiseRubricsGeneratorConfig,
    PointwiseRubricsGeneratorConfig,
    RubricsGenerator,
)
from rm_gallery.core.graders.base_grader import GraderMode
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


# pylint: disable=line-too-long


def create_test_samples() -> List[List[dict]]:
    """Create test samples for different evaluation modes.

    Returns:
        List[List[dict]]: List containing three test sample sets:
            - Sample 1: Pointwise mode - Score single response
            - Sample 2: Pairwise mode (listwise with 2 responses) - Compare two responses
            - Sample 3: Listwise mode - Rank multiple responses
    """
    samples = []

    # Sample 1: Pointwise Mode Test
    # Goal: Verify if response score matches expectation
    sample_pointwise = [
        {
            "query": "What is 2 + 2? Explain your reasoning.",
            "response": "2 + 2 = 4. This is basic arithmetic where we add two identical numbers.",
            "label_score": 1,
        },
    ]
    samples.append(sample_pointwise)

    # Sample 2: Pairwise Mode Test (listwise with 2 responses)
    sample_pairwise = [
        {
            "query": "Write a short story about a robot learning to paint.",
            "responses": [
                "BEEP-7 was a maintenance robot who discovered paint cans in an abandoned art studio. Curious, it dipped its mechanical fingers in blue paint and made its first mark on canvas. Days passed as BEEP-7 experimented with colors, learning that art wasn't about precision but expression. Its circuits hummed with joy as it created its first masterpiece - a sunset that somehow captured the warmth it has never felt.",
                "There was a robot. It painted. The end.",
            ],
            "label_rank": [1, 2],
        },
    ]
    samples.append(sample_pairwise)

    # Sample 3: Listwise Mode Test
    # Goal: Verify ranking capability with multiple responses
    sample_listwise = [
        {
            "query": "Explain what a for loop does in programming.",
            "responses": [
                "A for loop is a control structure that repeats a block of code a specific number of times. It consists of three parts: initialization (setting a counter), condition (when to stop), and increment (updating the counter). For example, in Python: 'for i in range(5):' will execute the loop body 5 times, with i taking values 0 through 4.",
                "A for loop is used to iterate over a sequence (like a list, tuple, or string) or other iterable objects. You use it when you know how many times you want to repeat something. For example, if you want to print 'Hello' 5 times, you can use a for loop to do it.",
                "It does stuff.",
                "A for loop repeats code multiple times. ",
            ],
            "label_rank": [1, 3, 4, 2],
        },
    ]
    samples.append(sample_listwise)

    return samples


async def test_sampling_mode(mode: GraderMode, samples: List[dict]) -> str:
    """Test rubric generation for a specific grader mode.

    This function demonstrates the rubric generation workflow:
    1. Each sample is processed independently
    2. Generate targeted rubrics for the given samples
    3. Parallel processing for efficiency
    4. Return generated rubrics

    Args:
        mode: Grader mode (POINTWISE or LISTWISE).
        samples: List of training samples for rubric generation.

    Returns:
        str: Generated rubrics as a formatted string.
    """
    model = OpenAIChatModel(
        model="qwen3-32b",
        stream=False,
    )

    # Create configuration based on grader mode
    if mode == GraderMode.POINTWISE:
        config = PointwiseRubricsGeneratorConfig(
            model=model,
            grader_name="Rubric-based_Pointwise_Grader",
            query_specific_generate_number=1,
            enable_categorization=False,
            language=LanguageEnum.EN,
            min_score=0,
            max_score=1,
        )
    else:
        config = ListwiseRubricsGeneratorConfig(
            model=model,
            grader_name="Rubric-based_Listwise_Grader",
            query_specific_generate_number=1,
            enable_categorization=False,
            language=LanguageEnum.EN,
        )

    generator = RubricsGenerator(config)
    rubrics = await generator._generate_rubrics(samples)

    logger.info(f"Generated rubrics for {mode.value} mode:\n{rubrics}")

    return rubrics


async def main() -> None:
    """Run all test cases for different evaluation modes."""
    samples = create_test_samples()

    # Test pointwise mode
    await test_sampling_mode(GraderMode.POINTWISE, samples[0])

    # Test pairwise mode (listwise with 2 responses)
    await test_sampling_mode(GraderMode.LISTWISE, samples[1])

    # Test listwise mode (multiple responses)
    await test_sampling_mode(GraderMode.LISTWISE, samples[2])


if __name__ == "__main__":
    asyncio.run(main())
