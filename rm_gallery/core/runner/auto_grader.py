import asyncio
from typing import Callable, List

from rm_gallery.core.data import DataSample, DataSampleParser
from rm_gallery.core.grader import LLMGrader, evaluate
from rm_gallery.core.model import OpenAIChatModel
from rm_gallery.core.runner.auto_rubrics import AutoRubrics, AutoRubricsConfig
from rm_gallery.core.runner.base import BaseRunner


class AutoGraderConfig(AutoRubricsConfig):
    ...


class AutoGrader(BaseRunner):
    def __init__(
        self,
        model: OpenAIChatModel,
        parser: DataSampleParser | Callable | None = None,
        config: AutoGraderConfig | None = None,
    ):
        """AutoGrader"""
        self.config = config
        self.auto_rubrics = AutoRubrics(model=model, parser=parser, config=config)

    async def __call__(
        self, data_samples: List[DataSample], *args, **kwargs
    ) -> LLMGrader:
        # Generate rubrics using AutoRubrics
        rubrics_result = await self.auto_rubrics(data_samples)

        # Extract the final rubrics from the result
        if isinstance(rubrics_result, dict) and "final_rubrics" in rubrics_result:
            rubrics = "\n".join(rubrics_result["final_rubrics"])
        else:
            # Fallback if the structure is different
            rubrics = str(rubrics_result)

        # Create a default template for the LLMGrader
        default_template = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that evaluates the quality of responses. Please evaluate the following response based on the provided rubrics.",
                },
                {
                    "role": "user",
                    "content": 'Question: {query}\nAnswer: {answer}\n\nRubrics for evaluation:\n{rubrics}\n\nPlease provide a score from 0 to 1 and a reason for your score in the following JSON format:\n{"score": <score>, "reason": "<reason>"}',
                },
            ],
            "required_fields": [
                {
                    "name": "query",
                    "type": "string",
                    "position": "data",
                    "description": "The user question",
                },
                {
                    "name": "answer",
                    "type": "string",
                    "position": "sample",
                    "description": "The assistant response",
                },
                {
                    "name": "rubrics",
                    "type": "string",
                    "position": "grader",
                    "description": "The rubrics for evaluation",
                },
            ],
        }

        # Create and return an LLMGrader with the generated rubrics
        return LLMGrader(
            name="Auto Grader",
            template=default_template,
            model=getattr(self.auto_rubrics.model, "__dict__", {}),
            rubrics=rubrics,
        )


if __name__ == "__main__":
    from rm_gallery.core.model import OpenAIChatModel

    model = OpenAIChatModel(model_name="qwen-plus")
    auto_grader = AutoGrader(model, AutoGraderConfig())
    data_samples = [
        DataSample(
            query="What is the capital of France?",
            answer="The capital of France is Paris.",
        )
    ]
    grader = auto_grader(data_samples)
    result = asyncio.run(
        evaluate(
            grader,
            parser=None,
            data_sample=data_samples[0],
        )
    )
