# -*- coding: utf-8 -*-
"""
Image Reference Grader

Evaluates the accuracy and clarity of image references in surrounding text.
Restructured to work with Grader framework.
"""

import asyncio
import textwrap
from typing import Any, List, Optional, Tuple, Union

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.multimodal._internal import (
    MLLMImage,
    format_image_content,
    get_image_context,
    get_image_indices,
)
from rm_gallery.core.graders.schema import GraderScoreCallback
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
IMAGE_REFERENCE_PROMPT_EN = """
# Task Description
You are a multi-modal document quality assessment assistant. You will receive an image and its accompanying textual context.
Your task is to determine whether the image is explicitly referenced or explained within the surrounding text (both above and below the image).

# Context Above
{context_above}

# Context Below
{context_below}

# Image
[The image is provided below this section.]

# Scoring Criteria
Evaluate the extent to which the image is referenced or explained in the text, assigning a score from 0 to 10:
- 0: The image is not mentioned or referenced in the context.
- 1-3: The image is referenced implicitly, and the reference is improper or incorrect.
- 4-6: The image is referenced explicitly but in an improper manner, or it is referenced implicitly.
- 7-9: The image is referenced explicitly, with the reference being generally proper and correct.
- 10: The image is referenced explicitly, with the placement and explanation being completely proper and correct.

Be rigorous and discerning when assigning your score.

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 0 and 10>,
    "reason": "<brief explanation for the assigned score>"
}}

# Image
[Insert Image Here]
"""

# Chinese Prompt
IMAGE_REFERENCE_PROMPT_ZH = """
# 任务描述
你是一名多模态文档质量评估助手。你将收到一张图片及其伴随的文本背景。
你的任务是判断图片是否在其周围文本（图片上方和下方）中被明确引用或解释。

# 上文
{context_above}

# 下文
{context_below}

# 图片
[图片将在本节下方提供。]

# 评分标准
评估图片在文本中被引用或解释的程度，给出0到10的分数：
- 0：图片在上下文中未被提及或引用。
- 1-3：图片被隐式引用，且引用不当或不正确。
- 4-6：图片被明确引用但方式不当，或仅被隐式引用。
- 7-9：图片被明确引用，引用总体上恰当且正确。
- 10：图片被明确引用，位置和解释完全恰当且正确。

请严格审慎地评分。

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0到10之间的整数>,
    "reason": "<对所给分数的简要解释>"
}}

# 图片
[在此插入图片]
"""

# Build default template from prompts
DEFAULT_IMAGE_REFERENCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_REFERENCE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_REFERENCE_PROMPT_ZH),
            ),
        ],
    },
)


class ImageReferenceGrader(LLMGrader):
    """
    Image Reference Grader

    Purpose:
        Evaluates how well images are referenced in surrounding text. Assesses whether
        text properly introduces, cites, and discusses images, ensuring readers know
        what to look at and why.

    What it evaluates:
        - Reference Clarity: Explicit and clear image references in text
        - Reference Accuracy: Description matches actual image content
        - Reference Placement: References at appropriate text locations
        - Reference Types: Explicit ("Figure 1") vs. Implicit ("shown here")
        - Contextual Integration: Image well-integrated into narrative flow

    When to use:
        - Academic and research paper evaluation
        - Technical documentation quality check
        - Report and presentation assessment
        - Editorial content review
        - Style guide compliance verification

    Scoring:
        - 10: Perfect explicit references with clear descriptions
        - 7-9: Good explicit or appropriate implicit references
        - 4-6: Some references but unclear or improper
        - 0-3: No references or completely inappropriate
        Note: For multiple images, returns average score

    Args:
        model: Vision-language model instance or dict config
        max_context_size: Max characters from text context (default: 500)
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom template (default: DEFAULT_IMAGE_REFERENCE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore with normalized reference quality score [0, 1]

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.multimodal import ImageReferenceGrader, MLLMImage
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = ImageReferenceGrader(model=model)
        >>>
        >>> result = await grader.aevaluate(
        ...     response=[
        ...         "Figure 1 shows the sales trend.",
        ...         MLLMImage(url="https://example.com/chart.jpg"),
        ...         "As seen in Figure 1, Q3 peaked."
        ...     ]
        ... )
        >>> print(result.score)  # 1.0 - explicit references before and after
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        max_context_size: int = 500,
        threshold: float = 0.7,
        template: PromptTemplate = DEFAULT_IMAGE_REFERENCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize ImageReferenceGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            max_context_size: Maximum characters to extract from context (default: 500)
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_IMAGE_REFERENCE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="image_reference",
            grader_mode=GraderMode.POINTWISE,
            description="Evaluate image reference quality in text",
            model=model,
            template=template,
            language=language,
        )
        self.max_context_size = max_context_size
        self.threshold = threshold

    async def _aevaluate_single_image(
        self,
        image: MLLMImage,
        context_above: Optional[str],
        context_below: Optional[str],
    ) -> Tuple[float, str]:
        """Async evaluation of single image reference"""
        messages = self.template.to_messages(self.language)
        prompt = (
            messages[0]
            .format(
                context_above=context_above or "",
                context_below=context_below or "",
            )
            .content
        )

        try:
            content = format_image_content(prompt, [image])
            chat_response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=GraderScoreCallback,
            )

            # Handle both streaming and non-streaming responses
            if hasattr(chat_response, "__aiter__"):
                # This is a streaming response, we need to collect it first
                collected_content = []
                metadata = {}
                async for chunk in chat_response:
                    if chunk.content:
                        collected_content.extend(chunk.content)
                    if chunk.metadata:
                        metadata.update(chunk.metadata)

                # Extract score and reason from metadata
                score = metadata.get("score", 0.0)
                reason = metadata.get("reason", "")
            else:
                # Non-streaming response
                score = chat_response.metadata["score"]
                reason = chat_response.metadata["reason"]
            return score, reason
        except Exception as e:
            logger.error(f"Error evaluating image reference: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _acompute(
        self,
        response: List[Union[str, MLLMImage]],
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """Compute image reference score (asynchronous)"""

        image_indices = get_image_indices(response)

        if not image_indices:
            return 0.0, {
                "error": "No images found in response",
                "num_images": 0,
            }

        tasks = []
        for image_index in image_indices:
            context_above, context_below = get_image_context(
                image_index,
                response,
                self.max_context_size,
            )
            image = response[image_index]
            tasks.append(
                self._aevaluate_single_image(
                    image,
                    context_above,
                    context_below,
                ),
            )

        results = await asyncio.gather(*tasks)

        scores = []
        reasons = []
        for raw_score, reason in results:
            normalized_score = raw_score / 10.0
            scores.append(normalized_score)
            reasons.append(reason)

        final_score = sum(scores) / len(scores) if scores else 0.0

        details = {
            "num_images": len(image_indices),
            "individual_scores": scores,
            "individual_reasons": reasons,
            "threshold": self.threshold,
        }

        return final_score, details

    async def aevaluate(
        self,
        response: List[Union[str, MLLMImage]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate image reference quality

        Args:
            response: List containing text and images (mixed)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized reference quality value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     response=[
            ...         "See the chart below.",
            ...         MLLMImage(url="chart.jpg"),
            ...         "The chart shows growth trends."
            ...     ]
            ... )
        """
        score, details = await self._acompute(response, **kwargs)

        if "error" in details:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=details["error"],
                metadata=details,
            )

        # Generate combined reason
        if len(details["individual_reasons"]) == 1:
            reason = details["individual_reasons"][0]
        else:
            reason_parts = []
            for i, (s, r) in enumerate(
                zip(
                    details["individual_scores"],
                    details["individual_reasons"],
                ),
                1,
            ):
                reason_parts.append(f"Image {i} (score: {s:.2f}): {r}")
            reason = "\n".join(reason_parts)

        return GraderScore(
            name=self.name,
            score=score,
            reason=f"Image reference quality score: {score:.4f}\n{reason}",
            metadata=details,
        )


__all__ = ["ImageReferenceGrader"]
