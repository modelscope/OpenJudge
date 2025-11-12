# 自定义LLMGrader
import asyncio

from loguru import logger

from rm_gallery.core.data import DataSample
from rm_gallery.core.grader import GraderMode, LLMGrader, evaluate
from rm_gallery.core.model.template import Template

DEFAULT_TEMPLATE = {
    "messages": [
        dict(
            role="system",
            content=(
                "You are a helpful assistant that evaluates the quality of a "
                "response. Your job is to evaluate the quality of the response "
                "and give a score between 0 and 1. The score should be based on "
                "the quality of the response. The higher the score, the better "
                "the response. The score should be a number between 0 and 1"
            ),
        ),
        dict(
            role="user",
            content=(
                "Please evaluate the quality of the response provided by the "
                "assistant.\nThe user question is: {query}\nThe assistant "
                "response is: {answer}\n\nPlease output as the following json "
                "object:\n{{\n    score: <score>,\n    reason: <reason>\n}}"
            ),
        ),
    ],
    "required_fields": [
        {
            "name": "query",
            "type": "string",
            "position": "data",
            "description": "The user question in data",
        },
        {
            "name": "answer",
            "type": "string",
            "position": "sample",
            "description": "The assistant response in sample",
        },
    ],
}

DEFAULT_MODEL = {
    "model_name": "qwen-plus",
    "stream": False,
    "client_args": {
        "timeout": 60,
    },
}


# 示例1：对于简单的LLM as a judge，可以直接使用LLMGrader
grader = LLMGrader(
    name="factual_grader",
    mode=GraderMode.POINTWISE,
    description="factual grader",
    template=Template(**DEFAULT_TEMPLATE),
    model=DEFAULT_MODEL,
    rubrics="",
)


# 示例2：对于负责LLM场景，继承LLMGrader/Grader自定义
class FactualGrader(LLMGrader):
    """Factual grader."""

    def __init__(self, **kwargs):
        super().__init__(
            name="factual_grader",
            mode=GraderMode.POINTWISE,
            description="factual grader",
            template=DEFAULT_TEMPLATE,
            model=DEFAULT_MODEL,
            rubrics="",
            **kwargs,
        )


grader = FactualGrader()


def test_factual_grader():
    """Test the factual grader."""
    data_sample = DataSample(
        data={"query": "What is the capital of France?"},
        samples=[{"answer": "Paris"}, {"answer": "London"}],
    )

    result = asyncio.run(
        evaluate(
            grader,
            mapping=None,
            data_sample=data_sample,
        )
    )
    logger.info(result)


if __name__ == "__main__":
    test_factual_grader()
