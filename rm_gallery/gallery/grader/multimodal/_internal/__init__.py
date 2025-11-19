# -*- coding: utf-8 -*-
"""
Internal utilities and supporting modules for multimodal graders

This module contains all non-grader supporting code including:
- Data schemas (Pydantic models)
- Templates for various evaluation types
- Helper functions for image and text processing
- Context extraction utilities
- G-Eval specific utilities
"""

from rm_gallery.gallery.grader.multimodal._internal.context_templates import (
    ImageCoherenceTemplate,
    ImageHelpfulnessTemplate,
    ImageReferenceTemplate,
)

# Context utilities
from rm_gallery.gallery.grader.multimodal._internal.context_utils import (
    get_image_context,
    get_image_indices,
)
from rm_gallery.gallery.grader.multimodal._internal.generation_templates import (
    ImageEditingTemplate,
    TextToImageTemplate,
)

# G-Eval utilities
from rm_gallery.gallery.grader.multimodal._internal.geval_utils import (
    MULTIMODAL_G_EVAL_PARAMS,
    construct_g_eval_params_string,
    format_rubrics,
    get_score_range,
    validate_and_sort_rubrics,
    validate_criteria_and_evaluation_steps,
)

# Helper functions and data types
from rm_gallery.gallery.grader.multimodal._internal.helpers import (
    MLLMImage,
    MLLMTestCaseParams,
    format_image_content,
)

# Schema definitions
from rm_gallery.gallery.grader.multimodal._internal.schema import (
    EvaluationSteps,
    Rubric,
)

# Template classes
from rm_gallery.gallery.grader.multimodal._internal.templates import (
    MultimodalGEvalTemplate,
)

__all__ = [
    # Schema
    "EvaluationSteps",
    "Rubric",
    # Templates
    "MultimodalGEvalTemplate",
    "ImageCoherenceTemplate",
    "ImageHelpfulnessTemplate",
    "ImageReferenceTemplate",
    "ImageEditingTemplate",
    "TextToImageTemplate",
    # Data types
    "MLLMImage",
    "MLLMTestCaseParams",
    # Helper functions
    "format_image_content",
    # Context functions
    "get_image_indices",
    "get_image_context",
    # G-Eval utilities
    "MULTIMODAL_G_EVAL_PARAMS",
    "validate_criteria_and_evaluation_steps",
    "validate_and_sort_rubrics",
    "format_rubrics",
    "construct_g_eval_params_string",
    "get_score_range",
]
