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
    >>> from rm_gallery.core.schema.data import DataSample
    >>>
    >>> # List all available graders
    >>> graders = list_available_graders()
    >>> print(f"Available: {', '.join(graders)}")
    >>>
    >>> # Use a text grader
    >>> grader = get_grader("exact_match")
    >>> data_sample = DataSample(
    ...     data={"reference": "hello"},
    ...     samples=[{"candidate": "hello"}, {"candidate": "world"}]
    ... )
    >>> import asyncio
    >>> results = asyncio.run(grader(data_sample))
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

# Registry functions
from rm_gallery.gallery.grader.registry import (
    get_grader,
    grader_registry,
    list_available_graders,
    register_grader,
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


# Auto-register all graders

# Register StringMatchGrader (all string matching algorithms use this unified grader)
# The specific algorithm is chosen at evaluate time via the 'algorithm' parameter
register_grader("string_match")(StringMatchGrader)

# For convenience, also register under common algorithm names
# These all create the same StringMatchGrader, algorithm is specified at evaluate() time
for algo_name in [
    "exact_match",
    "prefix_match",
    "suffix_match",
    "regex_match",
    "substring_match",
    "contains_all",
    "contains_any",
    "word_overlap",
    "char_overlap",
]:
    register_grader(algo_name)(StringMatchGrader)

# Register SimilarityGrader (all text similarity algorithms use this unified grader)
# The specific algorithm is chosen at evaluate time via the 'algorithm' parameter
register_grader("similarity")(SimilarityGrader)

# For convenience, also register under common algorithm names
# These all create the same SimilarityGrader, algorithm is specified at evaluate() time
for algo_name in [
    "bleu",
    "sentence_bleu",
    "gleu",
    "chrf",
    "meteor",
    "rouge",
    "rouge1",
    "rouge2",
    "rougeL",
    "rouge3",
    "rouge4",
    "rouge5",
    "rouge_ngram",
    "f1_score",
    "token_f1",
    "fuzzy_match",
    "edit_distance",
    "cosine",
    "jaccard",
]:
    register_grader(algo_name)(SimilarityGrader)

register_grader("json_match")(JsonMatchGrader)
register_grader("json_validator")(JsonValidatorGrader)

# Register LLM judge graders
register_grader("hallucination")(HallucinationGrader)
register_grader("helpfulness")(HelpfulnessGrader)
register_grader("harmfulness")(HarmfulnessGrader)


# Register multimodal graders
def _register_multimodal_graders():
    """Register multimodal graders (lazy loading to avoid circular import)"""
    try:
        from rm_gallery.gallery.grader.multimodal import _register_graders

        _register_graders()
    except ImportError:
        pass  # Multimodal module not available


# Call registration
_register_multimodal_graders()


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
                f"Multimodal graders not available: {name}"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.2.0"

__all__ = [
    # Registry
    "grader_registry",
    "register_grader",
    "get_grader",
    "list_available_graders",
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
