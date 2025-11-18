# -*- coding: utf-8 -*-
from typing import List
from rm_gallery.core.grader.base import (
    GraderMode,
    LLMGrader,
    GraderScore,
    GraderRank,
)
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import (
    BaseHelpfulnessGrader,
)

RUBRICS = """Accuracy and Correctness: Prioritize selecting the most appropriate category based on the input content, ensuring the classification decision aligns with the dominant themes, sentiment, or subject matter.
Clarity of Reasoning: Provide clear justification for the classification decision, referencing specific elements from the input that support the chosen category.
Handling Ambiguity: Appropriately address ambiguous cases by either selecting the most probable category with explanation or indicating uncertainty when no clear classification is possible.
Consistency: Apply consistent classification criteria across similar types of input content, maintaining stable categorization standards.
"""


CLASSIFICATION_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the accuracy of a classification response.
You should assess the classification based on accuracy, justification, and handling of ambiguous cases.
Be as objective as possible.

# Rubrics
{rubrics}

# Query
{query}

# Answer
{answer}

# Output Requirement
```json
{
    "score": "A numerical score from 0.0 to 1.0 representing the accuracy of the classification."
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)

CLASSIFICATION_RANK_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.

# Rubrics
{rubrics}

# Query
{query}

# Answers
{answer}

# Output Requirement
```json
{
    "rank": ["The rank score of the answer in the list."]
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)


class ClassificationGrader(BaseHelpfulnessGrader):
    """Classification: Accurately categorizes input content into predefined classes or labels."""

    _point_template = CLASSIFICATION_SCORE_TEMPLATE
    _list_template = CLASSIFICATION_RANK_TEMPLATE
    _rubrics = RUBRICS

    def __init__(self, model: ChatModelBase | dict, template: Template | None = None, mode: GraderMode = GraderMode.LISTWISE, **kwargs):
        """Initialize the SafetyGrader."""
        super().__init__(
            name="Classification",
            mode=mode,
            model=model,
            template=template,
            description="Accurately categorizes input content into predefined classes or labels.",
            **kwargs,
        )

    async def a_evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the classification response based on the query.

        Evaluates classification responses for their ability to accurately
        categorize input content into predefined classes or labels. The grader
        focuses on accurate category assignment, clear decision justification,
        and appropriate handling of ambiguous cases.

        Args:
            query (str): The query to evaluate.
            answer (str | List[str]): The answer(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

                Each GraderScore contains:
                    - score: A numerical score assigned by the grader
                    - reason: Explanation of how the score was determined
                    - metadata: Optional additional information from the evaluation

        Example:
            >>> grader = ClassificationGrader()
            >>> result = await grader.evaluate(
            ...     query="This movie was fantastic! I loved every minute of it.",
            ...     answer="Positive"
            ... )
        """
        return await super().a_evaluate(query=query, answer=answer, **kwargs)
