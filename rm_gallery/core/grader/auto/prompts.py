# -*- coding: utf-8 -*-
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Chat, LanguageEnum, Template


class EvaluationPromptTemplates:
    @staticmethod
    def pointwise_evaluation(model: ChatModelBase) -> Chat:
        """Template for pointwise evaluation"""
        return Chat(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(
                            role="user",
                            content="""您是一位专业评估员，负责评估回复的质量，请根据提供的评分标准评估以下回复。

评估标准:
{rubrics}

查询: {query}
回答: {answer}

评分范围: {min_score} 到 {max_score}

## 输出格式
请严格按照以下JSON格式输出：
{{
    "score": 分数值（必须是{min_score}到{max_score}范围内的整数）,
    "reason": "评分的详细理由和依据"
}}

重要提醒：score 必须是整数，不能是小数或其他格式。

请输出评分结果""",
                        ),
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="user",
                            content="""You are a professional evaluator that evaluates the quality of \
responses. Please evaluate the following response based on the provided rubrics.

Evaluation Criteria:
{rubrics}

Query: {query}
Response: {answer}

Score Range: {min_score} to {max_score}

## Output Format
Please output strictly in the following JSON format:
{{
    "score": score_value (must be an integer between {min_score} and {max_score}),
    "reason": "Detailed reasoning and basis for the score"
}}

Important reminder: The score must be an integer, not a decimal or any other format.

Please output the scoring result""",
                        ),
                    ],
                },
            ),
            model=model,
        )

    @staticmethod
    def listwise_evaluation(model: ChatModelBase) -> Chat:
        """Template for listwise evaluation - complete ranking at once"""
        return Chat(
            template=Template(
                messages={
                    LanguageEnum.ZH: [
                        ChatMessage(
                            role="user",
                            content="""您是一位专业评估员，负责评估回复的质量，请根据提供的评分标准评估以下回复，并根据评估标准对所有回答进行排序。

评估标准:
{rubrics}

查询: {query}

所有回答:
{answer}

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
                        ),
                    ],
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="user",
                            content="""You are a professional evaluator that evaluates the quality of \
responses. Please evaluate the following response based on the provided rubrics, \
and rank all responses based on the evaluation criteria.

Evaluation Criteria:
{rubrics}

Query: {query}

All Responses:
{answer}

## Task Requirements
- Evaluate all {num_responses} responses based on the evaluation criteria
- Assign a rank value to each response, higher values indicate better quality
- Keep responses in original order, only output corresponding rank values
- Important: No two responses can have the same rank value, must strictly \
distinguish quality differences, no ties allowed

## Example
Assume three responses:
- Response 1 is best → should get highest score
- Response 2 is worst → should get lowest score
- Response 3 is medium → should get medium score

Output format: [Response1_score, Response2_score, Response3_score]
Correct output: [3, 1, 2] (Response 1 gets 3 points highest, Response 2 gets \
1 point lowest, Response 3 gets 2 points medium)

## Output Format
Please output strictly in the following JSON format:
{{
    "rank": [Response1_score, Response2_score, Response3_score, ...],
    "reason": "Detailed explanation of quality assessment and score assignment for each response"
}}

Important reminders:
1. The value at position i in the array is the quality score for the i-th response, \
higher values indicate better quality
2. All scores must be positive integers, not decimals or other formats""",
                        ),
                    ],
                },
            ),
            model=model,
        )
