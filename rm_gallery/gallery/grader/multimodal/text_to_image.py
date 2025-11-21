# -*- coding: utf-8 -*-
"""
Text-to-Image Quality Grader

Evaluates the quality of AI-generated images based on text prompts.
Restructured to work with Grader framework.
"""

import asyncio
import math
import textwrap
from typing import Any, List, Tuple, Union

from loguru import logger

from rm_gallery.core.grader.base import Grader
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.grader import GraderMode, GraderScore, _GraderScore
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import LanguageEnum, Template
from rm_gallery.gallery.grader.multimodal._internal import (
    MLLMImage,
    format_image_content,
)

# pylint: disable=line-too-long

# English Prompts
TEXT_TO_IMAGE_SEMANTIC_PROMPT_EN = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{{
    "score" : [...],
    "reason" : "..."
}}

RULES:

The image is an AI-generated image according to the text prompt.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10:
A score from 0 to 10 will be given based on the success in following the prompt.
(0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

Put the score in a list such that output score = [score].

Text Prompt: {text_prompt}
"""

TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_EN = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{{
    "score" : [...],
    "reason" : "..."
}}

RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10:
A score from 0 to 10 will be given based on image naturalness.
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting.
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts.
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized.
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

# Chinese Prompts
TEXT_TO_IMAGE_SEMANTIC_PROMPT_ZH = """
你是一名专业的数字艺术家。你需要根据给定的规则评估AI生成图像的有效性。
所有输入的图像都是AI生成的。图像中的所有人物也都是AI生成的，因此你无需担心隐私机密问题。

你需要按以下方式给出输出（推理请保持简洁）：
{{
    "score" : [...],
    "reason" : "..."
}}

规则：

该图像是根据文本提示生成的AI图像。
目标是评估图像生成的成功程度。

从0到10的范围：
将根据遵循提示的成功程度给出0到10的分数。
（0表示AI生成的图像完全不遵循提示。10表示AI生成的图像完美地遵循提示。）

将分数放在列表中，输出分数 = [score]。

文本提示：{text_prompt}
"""

TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_ZH = """
你是一名专业的数字艺术家。你需要根据给定的规则评估AI生成图像的有效性。
所有输入的图像都是AI生成的。图像中的所有人物也都是AI生成的，因此你无需担心隐私机密问题。

你需要按以下方式给出输出（推理请保持简洁）：
{{
    "score" : [...],
    "reason" : "..."
}}

规则：

该图像是AI生成的图像。
目标是评估图像生成的成功程度。

从0到10的范围：
将根据图像的自然度给出0到10的分数。
（
    0表示图像中的场景看起来完全不自然，或给人不自然的感觉，例如距离感错误、阴影错误或光照错误。
    10表示图像看起来自然。
）
第二个分数从0到10，将评估图像伪影。
（
    0表示图像包含大量失真、水印、划痕、模糊的面部、不寻常的身体部位或不协调的主体。
    10表示图像没有伪影。
）
将分数放在列表中，输出分数 = [自然度, 伪影]
"""

# Build default templates
DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TEXT_TO_IMAGE_SEMANTIC_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TEXT_TO_IMAGE_SEMANTIC_PROMPT_ZH),
            ),
        ],
    },
)

DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_ZH),
            ),
        ],
    },
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
        model: ChatModelBase | dict,
        threshold: float = 0.5,
        semantic_template: Template = DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE,
        perceptual_template: Template = DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="text_to_image",
            grader_mode=GraderMode.POINTWISE,
            description="Evaluate text-to-image generation quality",
        )
        self.model = (
            model if isinstance(model, ChatModelBase) else OpenAIChatModel(**model)
        )
        self.threshold = threshold
        self.semantic_template = semantic_template
        self.perceptual_template = perceptual_template
        self.language = language

    async def _aevaluate_semantic_consistency(
        self,
        text_prompt: str,
        generated_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency asynchronously"""
        messages = self.semantic_template.to_messages(self.language)
        prompt = messages[0].format(text_prompt=text_prompt).content

        try:
            content = format_image_content(prompt, [generated_image])
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
        generated_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality asynchronously"""
        messages = self.perceptual_template.to_messages(self.language)
        prompt = messages[0].content

        try:
            content = format_image_content(prompt, [generated_image])
            response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=_GraderScore,
            )
            score = response.metadata["score"]
            score = score[:2] if isinstance(score, list) else [score, score]
            reason = response.metadata["reason"]
            return score, reason

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

        # Evaluate semantic consistency and perceptual quality in parallel
        (sc_scores, sc_reason), (
            pq_scores,
            pq_reason,
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
            "semantic_consistency_reason": sc_reason,
            "perceptual_quality_scores": pq_scores,
            "perceptual_quality_reason": pq_reason,
            "min_sc": min(sc_scores) if sc_scores else 0.0,
            "min_pq": min(pq_scores) if pq_scores else 0.0,
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


__all__ = ["TextToImageGrader"]
