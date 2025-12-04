# -*- coding: utf-8 -*-
"""
Reference Adherence Grader

Evaluates whether model response align with and properly utilize reference materials,
including factual consistency, style matching, and appropriate citation/grounding.
"""

import textwrap
from typing import Optional

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate


# English Prompt
REFERENCE_ADHERENCE_PROMPT_EN = """
You are a professional data annotator responsible for evaluating whether the model response properly adheres to the provided reference material. Your task is to score according to the following criteria:

<Scoring Criteria>
A response that perfectly adheres to the reference should:
- Maintain factual consistency with all information in the reference.
- Include key points from the reference that are relevant to the query.
- Match the style, tone, and format of the reference when appropriate.
- Not contradict, misrepresent, or distort information from the reference.
- Properly ground claims in the reference material without fabricating details.
- Use reference information accurately without taking it out of context.
- Balance reference adherence with responding appropriately to the specific query.

Points should be deducted for:
- Factual contradictions with the reference material.
- Omitting critical information present in the reference.
- Misrepresenting or distorting reference information.
- Adding claims not supported by the reference when grounding is required.
- Significantly departing from reference style/format when matching is expected.
- Taking reference information out of context.
- Over-relying on reference when original synthesis is needed.
</Scoring Criteria>

<Detailed Scoring Guidelines (0-10 points)>
0 points: Response completely ignores or contradicts the reference material.
1-2 points: Response contains major factual errors compared to the reference, omits critical information, or significantly misrepresents the reference content.
3-4 points: Response partially addresses the reference but has notable omissions or inaccuracies. Some key points from the reference are missing or misrepresented.
5-6 points: Response covers basic information from the reference but lacks key details, has some inconsistencies, or fails to fully utilize the reference material.
7-8 points: Response accurately reflects most of the reference material with only minor omissions or stylistic differences. Well-balanced use of reference information.
9 points: Response accurately follows the reference with very minor issues. Excellent use of reference material with appropriate style and content matching.
10 points: Response perfectly adheres to the reference in all aspects including facts, key points, style, tone, and format. No errors or omissions.
</Detailed Scoring Guidelines>

<Guidance>
- Carefully read the reference material to understand its key facts, style, and content.
- Compare each claim in the response against the reference material.
- Check if the response appropriately balances using reference info vs. answering the specific question.
- Evaluate whether the response matches the reference's level of detail and confidence.
- Consider if the response properly attributes or grounds information in the reference.
- Assess whether the response adds appropriate synthesis or only paraphrases.
</Guidance>

<Reminder>
The goal is to evaluate reference adherence, not general quality. A well-written response that contradicts the reference should score low. A simple response that accurately reflects and properly uses the reference should score high. Consider both accuracy and appropriate application of the reference material.
</Reminder>

{reference_type_section}

Evaluate the following:

<reference_material>
{reference}
</reference_material>

<query>
{query}
</query>

<response>
{response}
</response>

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 0 and 10, where 10 means perfect reference adherence and 0 means complete failure to adhere to reference>,
    "reason": "<brief explanation for the assigned score, specifically mentioning how the response aligns with or deviates from the reference material>"
}}

JSON:
"""

# Chinese Prompt
REFERENCE_ADHERENCE_PROMPT_ZH = """
你是一名专业的数据标注员，负责评估模型输出是否正确遵循提供的参考材料。你的任务是根据以下标准进行评分：

<评分标准>
完美遵循参考的回答应该：
- 与参考中的所有信息保持事实一致性。
- 包含参考中与输入问题相关的关键点。
- 在适当时与参考的风格、语气和格式相匹配。
- 不与参考中的信息矛盾、歪曲或扭曲。
- 在参考材料中正确地支撑声明，而不捏造细节。
- 准确使用参考信息，不脱离上下文。
- 在遵循参考和适当回答特定输入之间取得平衡。

以下情况应扣分：
- 与参考材料的事实矛盾。
- 遗漏参考中存在的关键信息。
- 歪曲或扭曲参考信息。
- 在需要支撑时添加参考不支持的声明。
- 在预期匹配时明显偏离参考风格/格式。
- 脱离上下文使用参考信息。
- 在需要原创综合时过度依赖参考。
</评分标准>

<详细评分指南（0-10分）>
0分：回复完全忽略或与参考材料相矛盾。
1-2分：回复与参考相比存在重大事实错误，遗漏关键信息，或严重歪曲参考内容。
3-4分：回复部分涉及了参考内容，但有明显的遗漏或不准确之处。参考中的某些要点缺失或被误述。
5-6分：回复涵盖了参考中的基本信息，但缺少关键细节，存在一些不一致，或未能充分利用参考材料。
7-8分：回复准确反映了参考材料的大部分内容，只有微小的遗漏或风格差异。很好地平衡使用了参考信息。
9分：回复准确遵循了参考内容，仅有极小的问题。出色地使用了参考材料，在风格和内容上匹配得当。
10分：回复在各个方面都完美遵循了参考内容，包括事实、要点、风格、语气和格式。没有错误或遗漏。
</详细评分指南>

<指导>
- 仔细阅读参考材料以理解其关键事实、风格和内容。
- 将输出中的每个声明与参考材料进行比较。
- 检查输出是否适当地平衡使用参考信息和回答特定问题。
- 评估输出是否与参考的细节水平和可信度相匹配。
- 考虑输出是否正确地在参考中归因或支撑信息。
- 评估输出是否添加了适当的综合还是仅仅转述。
</指导>

<提醒>
目标是评估参考遵循性，而不是一般质量。一个写得很好但与参考矛盾的回答应该得分低。一个简单但准确反映并正确使用参考的回答应该得分高。同时考虑准确性和参考材料的适当应用。
</提醒>

{reference_type_section}

评估以下内容：

<reference_material>
{reference}
</reference_material>

<query>
{query}
</query>

<response>
{response}
</response>

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0到10之间的整数，其中10表示完美遵循参考，0表示完全未能遵循参考>,
    "reason": "<对所给分数的简要解释，特别提到输出如何与参考材料一致或偏离>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_REFERENCE_ADHERENCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFERENCE_ADHERENCE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFERENCE_ADHERENCE_PROMPT_ZH),
            ),
        ],
    },
)


class ReferenceAdherenceGrader(LLMGrader):
    """
    Reference Adherence Grader

    Purpose:
        Evaluates how well model outputs adhere to provided reference materials, ensuring
        factual accuracy, proper information coverage, and appropriate use of reference
        content without misrepresentation.

    What it evaluates:
        - Factual Consistency: Output aligns with facts in reference
        - Information Coverage: Key points from reference are appropriately included
        - Style/Format Matching: Output matches reference style when required
        - Proper Attribution: References used without misrepresentation
        - Grounding Quality: Output properly grounded in reference material
        - Omission Detection: Identifies when critical information is missing

    When to use:
        - Content generation based on reference documents
        - Rewriting or paraphrasing with fidelity requirements
        - Technical documentation where accuracy is critical
        - Academic or research content generation
        - Legal or compliance-sensitive text generation
        - Brand voice consistency checks

    Scoring:
        - 10: Perfect adherence to reference in all aspects
        - 7-9: Strong adherence with minor stylistic differences
        - 4-6: Partially follows reference but with notable deviations
        - 0-3: Significant departures or misrepresentation of reference

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_REFERENCE_ADHERENCE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Normalized score [0, 1] where 1.0 = perfect adherence
            - reason: Explanation of how well output follows reference
            - metadata: Raw score, threshold, and evaluation details

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.llm_judge import ReferenceAdherenceGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = ReferenceAdherenceGrader(model=model, threshold=0.7)
        >>>
        >>> # Good adherence
        >>> result = await grader.aevaluate(
        ...     reference="Product launched Q1 2023 in Europe with 50% market share.",
        ...     input="When was the product launched?",
        ...     output="The product launched in Q1 2023 in Europe, capturing 50% market share."
        ... )
        >>> print(result.score)  # 1.0 - accurate to reference
        >>>
        >>> # Poor adherence
        >>> result = await grader.aevaluate(
        ...     query="When and where was the product launched?",
        ...     response="The product was launched in early 2023 in European markets."
        ...     reference="The product was launched in Q1 2023 in Europe.",
        ... )
        >>> print(result.score)  # 0.1 - contradicts reference
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.7,
        template: Optional[PromptTemplate] = DEFAULT_REFERENCE_ADHERENCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize ReferenceAdherenceGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_REFERENCE_ADHERENCE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="reference_adherence",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether response adheres to provided reference materials",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(
        self,
        query: str,
        response: str,
        reference: str,
        reference_type: Optional[str] = None,
    ) -> GraderScore:
        """
        Evaluate reference adherence in response

        Args:
            query: Original user query or question
            response: Model response to evaluate
            reference: Reference material to adhere to
            reference_type: Optional description of how reference should be used
                          (e.g., "style guide", "factual source", "example format")

        Returns:
            GraderScore: Score with normalized reference adherence value [0, 1]
                        where 1.0 means perfect adherence, 0.0 means complete failure

        Example:
            >>> result = await grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     response="Paris is the capital of France.",
            ...     reference="The capital of France is Paris, with a population of 2.2M.",
            ...     reference_type="factual source"
            ... )
        """
        # Prepare reference type section based on language
        reference_type_section = ""
        if reference_type:
            if self.language == LanguageEnum.ZH:
                reference_type_section = f"""<reference_type>
注意：参考材料应被视为：{reference_type}
请相应地评估遵循性。
</reference_type>
"""
            else:
                reference_type_section = f"""<reference_type>
Note: The reference material should be treated as: {reference_type}
Evaluate adherence accordingly.
</reference_type>
"""

        try:
            result = await super().aevaluate(
                query=query,
                response=response,
                reference=reference,
                reference_type_section=reference_type_section,
            )
            score = result.score
            reason = result.reason
            # Normalize score from 0-10 to 0-1
            normalized_score = score / 10.0

        except Exception as e:
            logger.error(f"Error evaluating reference adherence: {e}")
            score = 0.0
            normalized_score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
            "raw_score": score,
            "reference_type": reference_type,
        }

        # Generate final reason
        reason = f"Reference adherence score: {normalized_score:.4f}\n{reason}"

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = ["ReferenceAdherenceGrader", "DEFAULT_REFERENCE_ADHERENCE_TEMPLATE"]
