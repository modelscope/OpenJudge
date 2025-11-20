# -*- coding: utf-8 -*-
"""
Image Editing Quality Grader

Evaluates the quality of AI-based image editing.
Restructured to work with Grader framework.
"""

import asyncio
import math
from typing import Any, List, Tuple, Union

from loguru import logger

from rm_gallery.core.grader.base import Grader
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.grader import GraderMode, GraderScore, _GraderScore
from rm_gallery.gallery.grader.multimodal._internal import (
    ImageEditingTemplate,
    MLLMImage,
    format_image_content,
)


class ImageEditingGrader(Grader):
    """
    Image Editing Quality Grader

    Evaluates AI-edited images based on:
    1. Semantic Consistency (SC): How well the edits follow the instruction (0-10)
    2. Perceptual Quality (PQ): Visual quality of the edited image
       - Naturalness (0-10): How natural the edited areas look
       - Artifacts (0-10): Whether there are editing artifacts

    Final score = sqrt(semantic_consistency * min(perceptual_quality)) / 10

    This metric checks:
    - Whether the specified edits were correctly applied
    - Whether unmentioned areas remained unchanged
    - Whether edits blend naturally with the original image

    Attributes:
        name: Grader name
        model: OpenAIChatModel instance for evaluation
        threshold: Success threshold [0, 1] (default: 0.5)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.gallery.grader.multimodal import MLLMImage
        >>>
        >>> vlm_api = VisionModelAdapter.from_qwen(api_key="...", model_name="qwen-vl-plus")
        >>> grader = ImageEditingGrader(model=vlm_api, threshold=0.7)
        >>>
        >>> result = await grader.aevaluate(
        ...     original_image=MLLMImage(url="https://example.com/original.jpg"),
        ...     edit_instruction="Change the sofa color to blue",
        ...     edited_image=MLLMImage(url="https://example.com/edited.jpg")
        ... )
        >>> print(f"Score: {result.score:.4f}")
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        name: str = "image_editing",
        threshold: float = 0.5,
        description: str = "Evaluate image editing quality",
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
        original_image: MLLMImage,
        edit_instruction: str,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency asynchronously"""
        template = ImageEditingTemplate.generate_semantic_consistency_prompt()
        messages = template.to_messages()
        prompt = messages[0].format(edit_instruction=edit_instruction).content

        try:
            content = format_image_content(prompt, [original_image, edited_image])
            response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=_GraderScore,
            )
            score = response.metadata["score"]
            score = score if isinstance(score, list) else [score]
            reason = response.metadata["reason"]
            return score, reason

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    async def _aevaluate_perceptual_quality(
        self,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality asynchronously"""
        template = ImageEditingTemplate.generate_perceptual_quality_prompt()
        messages = template.to_messages()
        prompt = messages[0].content

        try:
            content = format_image_content(prompt, [edited_image])
            response = await self.model.achat(
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
            reason = result_data.get("reason", "No reason provided")

            # Ensure score is a list with 2 elements
            scores = (
                score_data if isinstance(score_data, list) else [score_data, score_data]
            )
            if len(scores) < 2:
                scores = [scores[0], scores[0]]

            return scores[:2], reason

        except Exception as e:
            logger.error(f"Error evaluating perceptual quality: {e}")
            return [5.0, 5.0], f"Error during evaluation: {str(e)}"

    async def _acompute(
        self,
        original_image: MLLMImage,
        edit_instruction: str,
        edited_image: MLLMImage,
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """
        Compute image editing quality score (asynchronous)

        Args:
            original_image: Original image before editing
            edit_instruction: Editing instruction
            edited_image: Edited image to evaluate

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """
        self.evaluation_cost = 0.0

        # Evaluate semantic consistency and perceptual quality in parallel
        (sc_scores, sc_reason), (
            pq_scores,
            pq_reason,
        ) = await asyncio.gather(
            self._aevaluate_semantic_consistency(
                original_image,
                edit_instruction,
                edited_image,
            ),
            self._aevaluate_perceptual_quality(edited_image),
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
            "semantic_consistency_reason": sc_reason,
            "perceptual_quality_scores": pq_scores,
            "perceptual_quality_reason": pq_reason,
            "min_sc": min(sc_scores) if sc_scores else 0.0,
            "min_pq": min(pq_scores) if pq_scores else 0.0,
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
        }

        return final_score, details

    async def aevaluate(
        self,
        original_image: Union[MLLMImage, List[MLLMImage]],
        edit_instruction: str,
        edited_image: Union[MLLMImage, List[MLLMImage]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate image editing quality

        Args:
            original_image: Original image before editing (MLLMImage or list)
            edit_instruction: Editing instruction (string)
            edited_image: Edited image to evaluate (MLLMImage or list)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized quality value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     original_image=MLLMImage(url="original.jpg"),
            ...     edit_instruction="Change the sofa color to blue",
            ...     edited_image=MLLMImage(url="edited.jpg")
            ... )
        """
        # Handle if images are lists
        if isinstance(original_image, list):
            if not original_image:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="No original image provided",
                    metadata={"error": "Empty original image list"},
                )
            original_image = original_image[0]

        if isinstance(edited_image, list):
            if not edited_image:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="No edited image provided",
                    metadata={"error": "Empty edited image list"},
                )
            edited_image = edited_image[0]

        if not isinstance(original_image, MLLMImage) or not isinstance(
            edited_image,
            MLLMImage,
        ):
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="Invalid image type",
                metadata={"error": "Images must be MLLMImage"},
            )

        score, details = await self._acompute(
            original_image,
            edit_instruction,
            edited_image,
            **kwargs,
        )

        # Generate comprehensive reason
        reason = f"""Image Editing Quality Score: {score:.4f}

Semantic Consistency: {details['min_sc']:.2f}/10
{details['semantic_consistency_reason']}

Perceptual Quality: {details['min_pq']:.2f}/10
- Naturalness: {details['perceptual_quality_scores'][0]:.2f}/10
- Artifacts: {details['perceptual_quality_scores'][1]:.2f}/10
{details['perceptual_quality_reason']}

The score combines semantic consistency and perceptual quality using geometric mean.
"""

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason.strip(),
            metadata=details,
        )


__all__ = ["ImageEditingGrader"]
