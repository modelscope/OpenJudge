"""
Bilingual Prompt Templates for Rubric Generation using ChatTemplate

This module provides ChatTemplate-based prompts for three paradigms:
- Pointwise: Individual scoring
- Listwise: Ranking multiple responses

Each paradigm has three types of prompts:
1. Generation: Generate initial rubrics
2. Evaluation: Evaluate samples using rubrics
3. Revision: Revise rubrics based on feedback

All prompts support both English and Chinese.
"""

from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.template import (
    ChatTemplate,
    LanguageEnum,
    RequiredField,
    Template,
)


class RubricPromptTemplates:
    """ChatTemplate-based bilingual prompt templates for rubric generation"""

    # ========== Generation Templates ==========

    @staticmethod
    def pointwise_generation(model_config: dict) -> ChatTemplate:
        """Template for generating pointwise rubrics"""
        return ChatTemplate(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(role="system", content="你是一个专业的评估标准制定专家。"),
                        ChatMessage(
                            role="user",
                            content="""请基于以下样本内容和标注信息，生成{generate_number}个针对性的评估rubrics。

{sample_content}

## 任务要求
- 评估模式: Pointwise (对单个回答独立评分，范围{min_score}-{max_score}，必须是整数)
- 仔细分析单个回答与评分范围的差异
- 生成能够区分单个回答与评分范围的评估标准
- 标准应该明确、具体、可操作
- 评分标准应该能够产生区间内的整数分数

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rubrics": [
        "第一个评估标准的详细描述",
        "第二个评估标准的详细描述",
        ...
    ],
    "reason": "生成这些评估标准的原因和依据"
}}

请生成评估标准""",
                        ),
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="system",
                            content="You are a professional evaluation criteria expert.",
                        ),
                        ChatMessage(
                            role="user",
                            content="""Based on the following sample content and annotations, generate {generate_number} targeted evaluation rubrics.

{sample_content}

## Task Requirements
- Evaluation Mode: Pointwise (score individual responses independently, range {min_score}-{max_score}, must be integers)
- Carefully analyze the differences between single response and score range
- Generate evaluation criteria that can distinguish single response and score range
- Criteria should be clear, specific, and actionable
- Evaluation criteria should be able to produce integer scores within the range

## Output Format
Please output strictly in the following JSON format:
{{
    "rubrics": [
        "Detailed description of the first evaluation criterion",
        "Detailed description of the second evaluation criterion",
        ...
    ],
    "reason": "Reason and basis for generating these evaluation criteria"
}}

Please generate evaluation criteria:""",
                        ),
                    ],
                },
                required_fields=[
                    RequiredField(
                        name="sample_content",
                        type="str",
                        position="data",
                        description="Sample content including query, responses, and annotations",
                    ),
                    RequiredField(
                        name="generate_number",
                        type="int",
                        position="grader",
                        description="Number of rubrics to generate",
                    ),
                    RequiredField(
                        name="min_score",
                        type="int",
                        position="grader",
                        description="Minimum score for pointwise",
                    ),
                    RequiredField(
                        name="max_score",
                        type="int",
                        position="grader",
                        description="Maximum score for pointwise",
                    ),
                ],
            ),
            model=model_config,
        )

    @staticmethod
    def listwise_generation(model_config: dict) -> ChatTemplate:
        """Template for generating listwise rubrics"""
        return ChatTemplate(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(role="system", content="你是一个专业的评估标准制定专家。"),
                        ChatMessage(
                            role="user",
                            content="""请基于以下样本内容和标注信息，生成{generate_number}个针对性的排序rubrics。

{sample_content}

## 任务要求
- 评估模式: Listwise (对多个回答进行整体排序)
- 仔细分析多个回答的差异，包括"优质回答"和"劣质回答"的区别
- 生成能够正确排序回答质量的标准
- 标准应该能够确定回答的相对质量顺序
- 注意：rank值越大表示质量越好，按降序排列

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rubrics": [
        "第一个排序标准的详细描述",
        "第二个排序标准的详细描述",
        ...
    ],
    "reason": "生成这些排序标准的原因和依据"
}}

请生成排序标准""",
                        ),
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="system",
                            content="You are a professional evaluation criteria expert.",
                        ),
                        ChatMessage(
                            role="user",
                            content="""Based on the following sample content and annotations, generate {generate_number} targeted ranking rubrics.

{sample_content}

## Task Requirements
- Evaluation Mode: Listwise (rank multiple responses holistically)
- Carefully analyze the differences between multiple responses, including "High-quality" and "Low-quality" responses
- Generate ranking criteria that can correctly order response quality
- Criteria should determine the relative quality order of responses
- Note: Higher rank values indicate better quality, sort in descending order

## Output Format
Please output strictly in the following JSON format:
{{
    "rubrics": [
        "Detailed description of the first ranking criterion",
        "Detailed description of the second ranking criterion",
        ...
    ],
    "reason": "Reason and basis for generating these ranking criteria"
}}

Please generate ranking criteria""",
                        ),
                    ],
                },
                required_fields=[
                    RequiredField(
                        name="sample_content",
                        type="str",
                        position="data",
                        description="Sample content including query, responses, and annotations",
                    ),
                    RequiredField(
                        name="generate_number",
                        type="int",
                        position="grader",
                        description="Number of rubrics to generate",
                    ),
                ],
            ),
            model=model_config,
        )

    # ========== Evaluation Templates ==========

    @staticmethod
    def pointwise_evaluation(model_config: dict) -> ChatTemplate:
        """Template for pointwise evaluation"""
        return ChatTemplate(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(
                            role="user",
                            content="""请根据评估标准对回答评分。

评估标准:
{rubrics_text}

查询: {query}
回答: {response}

评分范围: {min_score} 到 {max_score}

## 输出格式
请严格按照以下JSON格式输出：
{{
    "score": 分数值（必须是{min_score}到{max_score}范围内的整数）,
    "reason": "评分的详细理由和依据"
}}

重要提醒：score 必须是整数，不能是小数或其他格式。

请输出评分结果""",
                        )
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="user",
                            content="""Please score the response based on the evaluation criteria.

Evaluation Criteria:
{rubrics_text}

Query: {query}
Response: {response}

Score Range: {min_score} to {max_score}

## Output Format
Please output strictly in the following JSON format:
{{
    "score": score_value (must be an integer between {min_score} and {max_score}),
    "reason": "Detailed reasoning and basis for the score"
}}

Important reminder: The score must be an integer, not a decimal or any other format.

Please output the scoring result""",
                        )
                    ],
                },
                required_fields=[
                    RequiredField(
                        name="rubrics_text",
                        type="str",
                        position="data",
                        description="Rubrics text",
                    ),
                    RequiredField(
                        name="query",
                        type="str",
                        position="data",
                        description="Query text",
                    ),
                    RequiredField(
                        name="response",
                        type="str",
                        position="data",
                        description="Response text",
                    ),
                    RequiredField(
                        name="min_score",
                        type="int",
                        position="grader",
                        description="Minimum score",
                    ),
                    RequiredField(
                        name="max_score",
                        type="int",
                        position="grader",
                        description="Maximum score",
                    ),
                ],
            ),
            model=model_config,
        )

    @staticmethod
    def listwise_evaluation(model_config: dict) -> ChatTemplate:
        """Template for listwise evaluation - complete ranking at once"""
        return ChatTemplate(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(
                            role="user",
                            content="""请根据评估标准对所有回答进行排序。

评估标准:
{rubrics_text}

查询: {query}

所有回答:
{responses_text}

## 任务要求
- 根据评估标准，对所有{num_responses}个回答进行质量评估
- 为每个回答分配一个rank值，数值越大表示质量越好
- 保持回答的原始顺序，只输出每个回答对应的rank值
- 重要：任何两个回答的rank值都不能相同，必须严格区分质量差异，不允许平分

## 示例
假设有三个回答：
- 回答1质量最好 → 应该得最高分
- 回答2质量最差 → 应该得最低分
- 回答3质量中等 → 应该得中等分

输出格式：[回答1的分数, 回答2的分数, 回答3的分数]
正确输出：[3, 1, 2] （回答1得3分最高，回答2得1分最低，回答3得2分中等）

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rank": [回答1的分数, 回答2的分数, 回答3的分数, ...],
    "reason": "详细说明每个回答的质量评估和分数分配理由"
}}

重要提醒：
1. 数组中第i个位置的数值是第i个回答的质量分数，数值越大表示质量越好
2. 所有分数必须是正整数，不能是小数或其他格式""",
                        )
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="user",
                            content="""Please rank all responses based on the evaluation criteria.

Evaluation Criteria:
{rubrics_text}

Query: {query}

All Responses:
{responses_text}

## Task Requirements
- Evaluate all {num_responses} responses based on the evaluation criteria
- Assign a rank value to each response, higher values indicate better quality
- Keep responses in original order, only output corresponding rank values
- Important: No two responses can have the same rank value, must strictly distinguish quality differences, no ties allowed

## Example
Assume three responses:
- Response 1 is best → should get highest score
- Response 2 is worst → should get lowest score
- Response 3 is medium → should get medium score

Output format: [Response1_score, Response2_score, Response3_score]
Correct output: [3, 1, 2] (Response 1 gets 3 points highest, Response 2 gets 1 point lowest, Response 3 gets 2 points medium)

## Output Format
Please output strictly in the following JSON format:
{{
    "rank": [Response1_score, Response2_score, Response3_score, ...],
    "reason": "Detailed explanation of quality assessment and score assignment for each response"
}}

Important reminders:
1. The value at position i in the array is the quality score for the i-th response, higher values indicate better quality
2. All scores must be positive integers, not decimals or other formats""",
                        )
                    ],
                },
                required_fields=[
                    RequiredField(
                        name="rubrics_text",
                        type="str",
                        position="data",
                        description="Rubrics text",
                    ),
                    RequiredField(
                        name="query",
                        type="str",
                        position="data",
                        description="Query text",
                    ),
                    RequiredField(
                        name="responses_text",
                        type="str",
                        position="data",
                        description="All responses text",
                    ),
                    RequiredField(
                        name="num_responses",
                        type="int",
                        position="grader",
                        description="Number of responses to rank",
                    ),
                ],
            ),
            model=model_config,
        )

    # ========== Revision Templates ==========

    @staticmethod
    def pointwise_revision(model_config: dict) -> ChatTemplate:
        """Template for pointwise rubric revision"""
        return ChatTemplate(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(role="system", content="你是一个专业的评估标准制定专家。"),
                        ChatMessage(
                            role="user",
                            content="""之前生成的Pointwise评分标准在验证时失败了，请生成{generate_number}个评分标准，并根据详细反馈进行改进。

{sample_content}

## 之前的评分标准
{rubrics_text}

## 验证失败的详细反馈
{feedback}



## Pointwise模式的改进要求
1. 分析失败原因：
   - 为什么当前标准没能给出正确的分数？
   - 是否忽略了某些关键的质量维度？
   - 标准是否过于宽松或严格？

2. 重点改进方向：
   - 仔细对比期望分数与实际分数的差异
   - 识别高分回答的核心优势（准确性、完整性、清晰度等）
   - 识别低分回答的关键缺陷（错误、遗漏、模糊等）
   - 确保标准能够精确区分不同分数档位

3. 标准制定原则：
   - 每个标准应该明确、可量化、可操作
   - 标准应该覆盖多个评估维度
   - 标准应该能够准确反映期望分数
   - 评分必须是整数，标准应该清晰定义每个整数分数档位的要求

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rubrics": [
        "改进后的第一个评分标准的详细描述",
        "改进后的第二个评分标准的详细描述",
        ...
    ],
    "reason": "改进这些评分标准的原因和依据"
}}

请生成改进后的Pointwise评分标准""",
                        ),
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="system",
                            content="You are a professional evaluation criteria expert.",
                        ),
                        ChatMessage(
                            role="user",
                            content="""The previously generated Pointwise scoring criteria failed validation. Please generate {generate_number} improved scoring criteria based on detailed feedback.

{sample_content}

## Previous Scoring Criteria
{rubrics_text}

## Detailed Validation Failure Feedback
{feedback}

## Improvement Requirements for Pointwise Mode
1. Analyze Failure Reasons:
   - Why didn't the current criteria produce the correct scores?
   - Were any key quality dimensions overlooked?
   - Are the criteria too lenient or too strict?

2. Key Improvement Directions:
   - Carefully compare expected scores with actual scores
   - Identify core strengths of high-scoring responses (accuracy, completeness, clarity, etc.)
   - Identify key deficiencies of low-scoring responses (errors, omissions, vagueness, etc.)
   - Ensure criteria can precisely distinguish different score levels

3. Criteria Development Principles:
   - Each criterion should be clear, quantifiable, and actionable
   - Criteria should cover multiple evaluation dimensions
   - Criteria should accurately reflect expected scores
   - Scores must be integers, criteria should clearly define requirements for each integer score level

## Output Format
Please output strictly in the following JSON format:
{{
    "rubrics": [
        "Detailed description of the first improved scoring criterion",
        "Detailed description of the second improved scoring criterion",
        ...
    ],
    "reason": "Reason and basis for improving these scoring criteria"
}}

Please generate improved Pointwise scoring criteria:""",
                        ),
                    ],
                },
                required_fields=[
                    RequiredField(
                        name="sample_content",
                        type="str",
                        position="data",
                        description="Sample content",
                    ),
                    RequiredField(
                        name="rubrics_text",
                        type="str",
                        position="data",
                        description="Previous rubrics text",
                    ),
                    RequiredField(
                        name="feedback",
                        type="str",
                        position="data",
                        description="Validation failure feedback",
                    ),
                    RequiredField(
                        name="generate_number",
                        type="int",
                        position="grader",
                        description="Number of rubrics to generate",
                    ),
                ],
            ),
            model=model_config,
        )

    @staticmethod
    def listwise_revision(model_config: dict) -> ChatTemplate:
        """Template for listwise rubric revision"""
        return ChatTemplate(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(role="system", content="你是一个专业的评估标准制定专家。"),
                        ChatMessage(
                            role="user",
                            content="""之前生成的Listwise排序标准在验证时失败了，请生成{generate_number}个改进的排序标准。

{sample_content}

## 之前的排序标准
{rubrics_text}

## 验证失败的详细反馈
{feedback}

## Listwise模式的改进要求
1. 分析失败原因：
   - 为什么当前标准没能产生正确的排序或比较结果？
   - 排序结果中哪些位置出现了错误？
   - 是否混淆了不同质量层次的回答？

2. 重点改进方向：
   - 仔细分析期望排序与实际排序的差异
   - 识别每个回答的相对质量水平
   - 确保标准能够准确区分所有质量层次

3. 标准制定原则：
   - 每个标准应该能够建立清晰的质量梯度
   - 标准应该能够一致性地对所有回答排序
   - 标准应该覆盖从最优到最差的完整质量范围
   - 标准应该具有判别性，能够明确区分质量差异

## 重要提醒：Listwise评估输出格式
- 为每个回答分配一个rank值，数值越大表示质量越好
- rank值必须是正整数
- 保持回答的原始顺序，只输出每个回答对应的rank值
- 任何两个回答的rank值都不能相同，必须严格区分质量差异，不允许平分

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rubrics": [
        "改进后的第一个排序标准的详细描述",
        "改进后的第二个排序标准的详细描述",
        ...
    ],
    "reason": "改进这些排序标准的原因和依据"
}}

请生成改进后的Listwise排序标准：""",
                        ),
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="system",
                            content="You are a professional evaluation criteria expert.",
                        ),
                        ChatMessage(
                            role="user",
                            content="""The previously generated Listwise ranking criteria failed validation. Please generate {generate_number} improved ranking criteria based on detailed feedback.

{sample_content}

## Previous Ranking Criteria
{rubrics_text}

## Detailed Validation Failure Feedback
{feedback}

## Improvement Requirements for Listwise Mode
1. Analyze Failure Reasons:
   - Why didn't the current criteria produce the correct ranking or comparison results?
   - Which positions in the ranking were incorrect?
   - Were different quality levels of responses confused?

2. Key Improvement Directions:
   - Carefully analyze differences between expected and actual rankings
   - Identify relative quality levels of each response
   - Ensure criteria can accurately distinguish all quality levels

3. Criteria Development Principles:
   - Each criterion should establish a clear quality gradient
   - Criteria should consistently rank all responses
   - Criteria should cover the full quality range from best to worst
   - Criteria should be discriminative, clearly distinguishing quality differences

## Important Reminder: Listwise Evaluation Output Format
- Assign a rank value to each response, higher values indicate better quality
- Rank values must be positive integers
- Keep responses in original order, only output corresponding rank values
- Important: No two responses can have the same rank value, must strictly distinguish quality differences, no ties allowed

## Output Format
Please output strictly in the following JSON format:
{{
    "rubrics": [
        "Detailed description of the first improved ranking criterion",
        "Detailed description of the second improved ranking criterion",
        ...
    ],
    "reason": "Reason and basis for improving these ranking criteria"
}}

Please generate improved Listwise ranking criteria:""",
                        ),
                    ],
                },
                required_fields=[
                    RequiredField(
                        name="sample_content",
                        type="str",
                        position="data",
                        description="Sample content",
                    ),
                    RequiredField(
                        name="rubrics_text",
                        type="str",
                        position="data",
                        description="Previous rubrics text",
                    ),
                    RequiredField(
                        name="feedback",
                        type="str",
                        position="data",
                        description="Validation failure feedback",
                    ),
                    RequiredField(
                        name="generate_number",
                        type="int",
                        position="grader",
                        description="Number of rubrics to generate",
                    ),
                ],
            ),
            model=model_config,
        )


class RubricCategorizationTemplate:
    """ChatTemplate for LLM semantic categorization of rubrics into Theme-Tips structure"""

    @staticmethod
    def categorization(model_config: dict) -> ChatTemplate:
        """Template for rubric categorization into Theme-Tips structure"""
        return ChatTemplate(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(
                            role="system",
                            content="你是一个专业的评估标准聚合专家，擅长将多个零散的评估标准整合成结构化的主题-技巧格式。",
                        ),
                        ChatMessage(
                            role="user",
                            content="""请将以下评估建议聚合成{num_categories}个或更少的结构化评估标准。

## 输入评估建议
{rubrics_text}

## 任务要求
- 评估标准必须完全自包含，非专业读者无需查阅任何外部信息
- 每个标准应评估独立维度，彼此不矛盾
- 确保整体判断在所有示例中保持一致

## 标准格式
每个标准包含两部分：
- 主题：简洁明确的陈述，捕捉标准的核心焦点
- 技巧：多个要点，扩展或补充标准的具体指导

## 输出格式
请严格按照以下JSON格式输出：
{{
    "categories": [
        {{
            "theme": "第一个主题的陈述",
            "tips": [
                "具体指导要点",
                "具体指导要点"
            ]
        }},
        {{
            "theme": "第二个主题的陈述",
            "tips": [
                "具体指导要点",
                "具体指导要点"
            ]
        }}
    ],
    "reason": "聚合这些评估标准的原因和依据"
}}

请生成聚合后的评估标准
""",
                        ),
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="system",
                            content="You are a professional evaluation criteria aggregation expert, skilled at integrating multiple scattered evaluation criteria into structured Theme-Tips format.",
                        ),
                        ChatMessage(
                            role="user",
                            content="""Please aggregate the following evaluation suggestions into {num_categories} or fewer structured evaluation rubrics.

## Input Evaluation Suggestions
{rubrics_text}

## Task Requirements
- Rubrics must be fully self-contained so that non-expert readers need not consult any external information
- Each rubric should assess an independent dimension and be non-contradictory with others
- Ensure overall judgment remains aligned and consistent for all examples

## Rubric Format
Each rubric consists of two parts:
- Theme: A concise and clear statement that captures the core focus of the rubric
- Tips: Multiple bullet points that expand on or supplement the rubric with specific guidance

## Output Format
Please output strictly in the following JSON format:
{{
    "categories": [
        {{
            "theme": "First theme statement",
            "tips": [
                "Specific guidance point",
                "Specific guidance point"
            ]
        }},
        {{
            "theme": "Second theme statement",
            "tips": [
                "Specific guidance point",
                "Specific guidance point"
            ]
        }}
    ],
    "reason": "Reason and basis for aggregating these evaluation criteria"
}}

Please generate the aggregated evaluation criteria
""",
                        ),
                    ],
                },
                required_fields=[
                    RequiredField(
                        name="rubrics_text",
                        type="str",
                        position="data",
                        description="Input rubrics text to be categorized",
                    ),
                    RequiredField(
                        name="num_categories",
                        type="int",
                        position="grader",
                        description="Maximum number of categories to generate",
                    ),
                ],
            ),
            model=model_config,
        )
