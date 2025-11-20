# -*- coding: utf-8 -*-
"""
Multimodal Graders

This module contains graders for multimodal evaluation tasks including:
- Image-text coherence evaluation
- Image helpfulness assessment
- Image reference quality
- Text-to-image generation quality
- Image editing quality
- Flexible G-Eval framework for multimodal evaluation
"""

from rm_gallery.gallery.grader.multimodal._internal import MLLMImage, MLLMTestCaseParams
from rm_gallery.gallery.grader.multimodal.image_coherence import ImageCoherenceGrader
from rm_gallery.gallery.grader.multimodal.image_editing import ImageEditingGrader
from rm_gallery.gallery.grader.multimodal.image_helpfulness import (
    ImageHelpfulnessGrader,
)
from rm_gallery.gallery.grader.multimodal.image_reference import ImageReferenceGrader
from rm_gallery.gallery.grader.multimodal.multimodal_geval import MultimodalGEvalGrader
from rm_gallery.gallery.grader.multimodal.text_to_image import TextToImageGrader

__all__ = [
    # Graders
    "ImageCoherenceGrader",
    "ImageHelpfulnessGrader",
    "ImageReferenceGrader",
    "ImageEditingGrader",
    "TextToImageGrader",
    "MultimodalGEvalGrader",
    # Multimodal data types
    "MLLMImage",
    "MLLMTestCaseParams",
]
