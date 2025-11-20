# -*- coding: utf-8 -*-
"""
Text-to-Image Quality Grader

Evaluates the quality of AI-generated images based on text prompts.
Restructured to work with Grader framework.
"""

import asyncio
import math
from typing import Any, List, Tuple, Union

from loguru import logger

from rm_gallery.core.grader.base import Grader
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.grader import GraderMode, GraderScore
from rm_gallery.gallery.grader.multimodal._internal import (
    MLLMImage,
    TextToImageTemplate,
    format_image_content,
)


class TextToImageGrader(Grader):
    """
    Text-to-Image Quality Grader

    Evaluates AI-generated images based on:
    1. Semantic Consistency (SC): How well the image matches the text prompt (0-10)
    2. Perceptual Quality (PQ): Visual quality of the image
       - Naturalness (0-10)
       - Artifacts (0-10)

    Final score = sqrt(semantic_consistency * min(perceptual_quality)) / 10

    Attributes:
        name: Grader name
        model: OpenAIChatModel instance for evaluation
        threshold: Success threshold [0, 1] (default: 0.5)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.gallery.grader.multimodal import MLLMImage
        >>>
        >>> vlm_api = OpenAIChatModel(
        ...     api_key="...",
        ...     model_name="gpt-4o",
        ...     generate_kwargs={"temperature": 0.1},
        ... )
        >>> grader = TextToImageGrader(model=vlm_api, threshold=0.7)
        >>>
        >>> result = await grader.aevaluate(
        ...     text_prompt="A cat sitting on a blue sofa",
        ...     generated_image=MLLMImage(url="https://example.com/cat.jpg")
        ... )
        >>> print(f"Score: {result.score:.4f}")
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        name: str = "text_to_image",
        threshold: float = 0.5,
        description: str = "Evaluate text-to-image generation quality",
    ):
        super().__init__(
            name=name,
            grader_mode=GraderMode.POINTWISE,
            description=description,
        )
        self.model = model
        self.threshold = threshold
        self.evaluation_cost = 0.0

    async def _aevaluate_semantic_consistency(
        self,
        text_prompt: str,
        generated_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency asynchronously"""
        template = TextToImageTemplate.generate_semantic_consistency_prompt()
        messages = template.get()
        prompt = messages[0].content.format(text_prompt=text_prompt)

        try:
            content = format_image_content(prompt, [generated_image])
            response = await self.model(
                messages=[{"role": "user", "content": content}],
            )

            # Parse response from text content
            import json

            text_content = "".join(
                [block.text for block in response.content if hasattr(block, "text")],
            )

            # Parse JSON response
            result_data = json.loads(text_content.strip())
            score_data = result_data.get("score", 0)
            scores = score_data if isinstance(score_data, list) else [score_data]
            reasoning = result_data.get("reasoning", "No reasoning provided")

            return scores, reasoning

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    async def _aevaluate_perceptual_quality(
        self,
        generated_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality asynchronously"""
        template = TextToImageTemplate.generate_perceptual_quality_prompt()
        messages = template.get()
        prompt = messages[0].content

        try:
            content = format_image_content(prompt, [generated_image])
            response = await self.model(
                messages=[{"role": "user", "content": content}],
            )

            # Parse response from text content
            import json

            text_content = "".join(
                [block.text for block in response.content if hasattr(block, "text")],
            )

            # Parse JSON response
            result_data = json.loads(text_content.strip())
            score_data = result_data.get("score", 0)
            reasoning = result_data.get("reasoning", "No reasoning provided")

            # Ensure score is a list with 2 elements
            scores = (
                score_data if isinstance(score_data, list) else [score_data, score_data]
            )
            if len(scores) < 2:
                scores = [scores[0], scores[0]]

            return scores[:2], reasoning

        except Exception as e:
            logger.error(f"Error evaluating perceptual quality: {e}")
            return [5.0, 5.0], f"Error during evaluation: {str(e)}"

    async def _a_compute(
        self,
        text_prompt: str,
        generated_image: MLLMImage,
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """
        Compute text-to-image quality score (asynchronous)

        Args:
            text_prompt: Original text prompt
            generated_image: Generated image to evaluate

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """
        self.evaluation_cost = 0.0

        # Evaluate semantic consistency and perceptual quality in parallel
        (sc_scores, sc_reasoning), (
            pq_scores,
            pq_reasoning,
        ) = await asyncio.gather(
            self._aevaluate_semantic_consistency(
                text_prompt,
                generated_image,
            ),
            self._aevaluate_perceptual_quality(generated_image),
        )

        # Calculate final score using geometric mean
        if not sc_scores or not pq_scores:
            final_score = 0.0
        else:
            min_sc = min(sc_scores)
            min_pq = min(pq_scores)
            final_score = math.sqrt(min_sc * min_pq) / 10.0
            final_score = min(1.0, max(0.0, final_score))

        details = {
            "semantic_consistency_scores": sc_scores,
            "semantic_consistency_reasoning": sc_reasoning,
            "perceptual_quality_scores": pq_scores,
            "perceptual_quality_reasoning": pq_reasoning,
            "min_sc": min(sc_scores) if sc_scores else 0.0,
            "min_pq": min(pq_scores) if pq_scores else 0.0,
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
        }

        return final_score, details

    async def aevaluate(
        self,
        text_prompt: str,
        generated_image: Union[MLLMImage, List[MLLMImage]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate text-to-image generation quality

        Args:
            text_prompt: Original text prompt (string)
            generated_image: Generated image (MLLMImage or list with single MLLMImage)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized quality value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     text_prompt="A cat sitting on a blue sofa",
            ...     generated_image=MLLMImage(url="cat.jpg")
            ... )
        """
        # Handle if generated_image is a list
        if isinstance(generated_image, list):
            if not generated_image:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="No generated image provided",
                    metadata={"error": "Empty image list"},
                )
            generated_image = generated_image[0]

        if not isinstance(generated_image, MLLMImage):
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="Invalid image type",
                metadata={"error": "generated_image must be MLLMImage"},
            )

        score, details = await self._a_compute(text_prompt, generated_image, **kwargs)

        # Generate comprehensive reason
        reason = f"""Text-to-Image Quality Score: {score:.4f}

Semantic Consistency: {details['min_sc']:.2f}/10
{details['semantic_consistency_reasoning']}

Perceptual Quality: {details['min_pq']:.2f}/10
- Naturalness: {details['perceptual_quality_scores'][0]:.2f}/10
- Artifacts: {details['perceptual_quality_scores'][1]:.2f}/10
{details['perceptual_quality_reasoning']}

The score combines semantic consistency and perceptual quality using geometric mean.
"""

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason.strip(),
            metadata=details,
        )


__all__ = ["TextToImageGrader"]
