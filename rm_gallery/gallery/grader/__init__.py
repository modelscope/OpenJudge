# -*- coding: utf-8 -*-
"""
RM-Gallery Grader Module

Evaluation graders module providing comprehensive collection of pre-built graders.

This module provides a comprehensive collection of evaluation graders:
- String graders (Exact Match, Substring, Overlap, etc.)
- NLP graders (BLEU, ROUGE, METEOR, GLEU, etc.)
- Similarity graders (Fuzzy Match, Cosine Similarity, F1 Score, etc.)
- Format graders (JSON validation, etc.)
- Multimodal graders (Image Coherence, Text-to-Image, Multimodal G-Eval, etc.)
- LLM-as-a-Judge graders (Hallucination, Helpfulness, Harmfulness, etc.)

Quick Start:
    >>> from rm_gallery.gallery.grader import get_grader, list_available_graders
    >>> from rm_gallery.core.schema.data import EvalCase
    >>>
    >>> # List all available graders
    >>> graders = list_available_graders()
    >>> print(f"Available: {', '.join(graders)}")
    >>>
    >>> # Use a text grader
    >>> grader = get_grader("exact_match")
    >>> eval_case = EvalCase(
    ...     input={"reference": "hello"},
    ...     outputs=[{"candidate": "hello"}, {"candidate": "world"}]
    ... )
    >>> import asyncio
    >>> results = asyncio.run(grader(eval_case))
    >>>
    >>> # Use a multimodal grader
    >>> from rm_gallery.gallery.grader.multimodal import ImageCoherenceGrader
    >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
    >>> api = OpenAIChatModel(api_key="...", model_name="gpt-4o", generate_kwargs={"temperature": 0.1})
    >>> grader = ImageCoherenceGrader(model=api)
"""

# Format Graders
from rm_gallery.gallery.grader.format.json_match import (
    JsonMatchGrader,
    JsonValidatorGrader,
)

# LLM Judge Graders
from rm_gallery.gallery.grader.llm_judge import (
    HallucinationGrader,
    HarmfulnessGrader,
    HelpfulnessGrader,
)

# Text Graders
from rm_gallery.gallery.grader.text.similarity import SimilarityGrader
from rm_gallery.gallery.grader.text.string_match import StringMatchGrader

# Core schemas are in rm_gallery.core.grader
# from rm_gallery.core.grader import GraderScore, GraderRank

# Multimodal Graders (lazy import to avoid circular dependency)
# from rm_gallery.gallery.grader.multimodal import (
#     ImageCoherenceGrader,
#     MultimodalGEvalGrader,
#     TextToImageGrader,
# )


# Expose multimodal graders for convenience
def __getattr__(name):
    """Lazy loading for multimodal grader classes"""
    multimodal_graders = (
        "ImageCoherenceGrader",
        "ImageHelpfulnessGrader",
        "ImageReferenceGrader",
        "ImageEditingGrader",
        "TextToImageGrader",
        "MultimodalGEvalGrader",
    )
    if name in multimodal_graders:
        try:
            from rm_gallery.gallery.grader.multimodal import (
                ImageCoherenceGrader,
                ImageEditingGrader,
                ImageHelpfulnessGrader,
                ImageReferenceGrader,
                MultimodalGEvalGrader,
                TextToImageGrader,
            )

            _map = {
                "ImageCoherenceGrader": ImageCoherenceGrader,
                "ImageHelpfulnessGrader": ImageHelpfulnessGrader,
                "ImageReferenceGrader": ImageReferenceGrader,
                "ImageEditingGrader": ImageEditingGrader,
                "TextToImageGrader": TextToImageGrader,
                "MultimodalGEvalGrader": MultimodalGEvalGrader,
            }
            return _map[name]
        except ImportError:
            raise AttributeError(
                f"Multimodal graders not available: {name}",
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Text Graders
    "StringMatchGrader",
    "SimilarityGrader",
    # Format Graders
    "JsonMatchGrader",
    "JsonValidatorGrader",
    # LLM Judge Graders
    "HallucinationGrader",
    "HelpfulnessGrader",
    "HarmfulnessGrader",
    # Multimodal Graders
    "ImageCoherenceGrader",
    "ImageHelpfulnessGrader",
    "ImageReferenceGrader",
    "ImageEditingGrader",
    "TextToImageGrader",
    "MultimodalGEvalGrader",
]
