# -*- coding: utf-8 -*-
"""
Multimodal G-Eval Utility Functions

Helper functions for multimodal G-Eval evaluation.
"""

from typing import List, Optional, Tuple

from rm_gallery.gallery.grader.multimodal._internal.helpers import MLLMTestCaseParams
from rm_gallery.gallery.grader.multimodal._internal.schema import Rubric

# Mapping for multimodal test case parameters
MULTIMODAL_G_EVAL_PARAMS = {
    MLLMTestCaseParams.INPUT: "Input",
    MLLMTestCaseParams.ACTUAL_OUTPUT: "Actual Output",
    MLLMTestCaseParams.EXPECTED_OUTPUT: "Expected Output",
    MLLMTestCaseParams.CONTEXT: "Context",
    MLLMTestCaseParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    MLLMTestCaseParams.TOOLS: "Tools",
    MLLMTestCaseParams.EXPECTED_TOOLS: "Expected Tools",
}


def validate_criteria_and_evaluation_steps(
    criteria: Optional[str] = None,
    evaluation_steps: Optional[List[str]] = None,
) -> None:
    """
    Validate that either criteria or evaluation steps are provided

    Args:
        criteria: Evaluation criteria description
        evaluation_steps: List of evaluation steps

    Raises:
        ValueError: If neither or both are invalid

    Example:
        >>> validate_criteria_and_evaluation_steps(
        ...     criteria="Evaluate image quality"
        ... )  # OK
        >>> validate_criteria_and_evaluation_steps()  # Raises ValueError
    """
    # Check if both criteria and evaluation_steps are not None at the same time
    if criteria is None and evaluation_steps is None:
        raise ValueError("Either 'criteria' or 'evaluation_steps' must be provided.")

    # Check if criteria is provided, it cannot be an empty string
    if criteria is not None and not criteria.strip():
        raise ValueError("Criteria provided cannot be an empty string.")

    # Check if evaluation_steps is provided, it cannot be an empty list
    if evaluation_steps is not None and len(evaluation_steps) == 0:
        raise ValueError(
            "'evaluation_steps' must not be an empty list. Either omit evaluation steps or include a non-empty list of steps.",
        )


def validate_and_sort_rubrics(
    rubrics: Optional[List[Rubric]] = None,
    score_range: Tuple[int, int] = (0, 10),
) -> Optional[List[Rubric]]:
    """
    Validate and sort rubrics by score range

    Args:
        rubrics: List of rubric definitions
        score_range: Valid score range (min, max)

    Returns:
        Sorted list of rubrics or None

    Raises:
        ValueError: If rubrics have overlapping ranges

    Example:
        >>> rubrics = [
        ...     Rubric(score_range=(7, 10), expected_outcome="Excellent"),
        ...     Rubric(score_range=(0, 6), expected_outcome="Poor")
        ... ]
        >>> sorted_rubrics = validate_and_sort_rubrics(rubrics)
    """
    if rubrics is None:
        return None

    # Sort rubrics by start of range
    sorted_rubrics = sorted(rubrics, key=lambda r: r.score_range[0])

    # Full overlap check
    for i in range(len(sorted_rubrics)):
        a_start, a_end = sorted_rubrics[i].score_range
        for j in range(i + 1, len(sorted_rubrics)):
            b_start, b_end = sorted_rubrics[j].score_range
            # Check if ranges overlap
            if a_end >= b_start:
                raise ValueError(
                    f"Overlapping score ranges: {sorted_rubrics[i].score_range} and {sorted_rubrics[j].score_range}",
                )

    return sorted_rubrics


def format_rubrics(rubrics: Optional[List[Rubric]]) -> Optional[str]:
    """
    Format rubrics into a readable string

    Args:
        rubrics: List of rubric definitions

    Returns:
        Formatted rubric string or None

    Example:
        >>> rubrics = [
        ...     Rubric(score_range=(0, 3), expected_outcome="Poor quality"),
        ...     Rubric(score_range=(7, 10), expected_outcome="High quality")
        ... ]
        >>> print(format_rubrics(rubrics))
        0-3: Poor quality
        7-10: High quality
    """
    if rubrics is None:
        return None

    return "\n".join(
        (
            f"{start}: {rubric.expected_outcome}"
            if start == end
            else f"{start}-{end}: {rubric.expected_outcome}"
        )
        for rubric in rubrics
        for start, end in [rubric.score_range]
    )


def construct_g_eval_params_string(
    evaluation_params: List[MLLMTestCaseParams],
) -> str:
    """
    Construct a readable string from evaluation parameters

    Args:
        evaluation_params: List of evaluation parameters

    Returns:
        Formatted parameter string

    Example:
        >>> params = [
        ...     MLLMTestCaseParams.INPUT,
        ...     MLLMTestCaseParams.ACTUAL_OUTPUT
        ... ]
        >>> construct_g_eval_params_string(params)
        'Input and Actual Output'
    """
    g_eval_params = [MULTIMODAL_G_EVAL_PARAMS[param] for param in evaluation_params]

    if len(g_eval_params) == 1:
        g_eval_params_str = g_eval_params[0]
    elif len(g_eval_params) == 2:
        g_eval_params_str = " and ".join(g_eval_params)
    else:
        g_eval_params_str = ", ".join(g_eval_params[:-1]) + ", and " + g_eval_params[-1]

    return g_eval_params_str


def get_score_range(rubric: Optional[List[Rubric]]) -> Tuple[int, int]:
    """
    Get the overall score range from rubrics

    Args:
        rubric: List of rubric definitions

    Returns:
        Tuple of (min_score, max_score)

    Example:
        >>> rubrics = [
        ...     Rubric(score_range=(0, 3), expected_outcome="Poor"),
        ...     Rubric(score_range=(7, 10), expected_outcome="Excellent")
        ... ]
        >>> get_score_range(rubrics)
        (0, 10)
    """
    if rubric is None:
        return (0, 10)

    return rubric[0].score_range[0], rubric[-1].score_range[1]


__all__ = [
    "MULTIMODAL_G_EVAL_PARAMS",
    "validate_criteria_and_evaluation_steps",
    "validate_and_sort_rubrics",
    "format_rubrics",
    "construct_g_eval_params_string",
    "get_score_range",
]
