# -*- coding: utf-8 -*-
"""
Multimodal Graders

This module contains graders for multimodal evaluation tasks including:
- Image-text coherence evaluation
- Image helpfulness assessment
- Image reference quality
- Text-to-image generation quality
- Image editing quality
- Flexible custom criteria framework for multimodal evaluation
"""

from rm_gallery.core.graders.predefined.multimodal._internal import MLLMImage
from rm_gallery.core.graders.predefined.multimodal.custom_criteria import (
    CustomCriteriaGrader,
)
from rm_gallery.core.graders.predefined.multimodal.image_coherence import (
    ImageCoherenceGrader,
)
from rm_gallery.core.graders.predefined.multimodal.image_editing import ImageEditingGrader
from rm_gallery.core.graders.predefined.multimodal.image_helpfulness import (
    ImageHelpfulnessGrader,
)
from rm_gallery.core.graders.predefined.multimodal.image_reference import (
    ImageReferenceGrader,
)
from rm_gallery.core.graders.predefined.multimodal.text_to_image import TextToImageGrader

__all__ = [
    # Graders
    "ImageCoherenceGrader",
    "ImageHelpfulnessGrader",
    "ImageReferenceGrader",
    "ImageEditingGrader",
    "TextToImageGrader",
    "CustomCriteriaGrader",
    # Multimodal data types
    "MLLMImage",
]
