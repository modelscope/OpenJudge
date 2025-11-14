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
                    "content": 'Question: {query}\nAnswer: {answer}\n\nRubrics for evaluation:\n{rubrics}\n\nPlease provide a score from 0 to 1 and a reason for your score in the following JSON format:{{"score": <score>, "reason": "<reason>"}}',
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

        model_dict = {
            "model_name": self.auto_rubrics.model.model_name,
        }

        return LLMGrader(
            name="Auto Grader",
            template=default_template,
            model=model_dict,
            rubrics=rubrics,
        )


if __name__ == "__main__":
    from rm_gallery.core.model import OpenAIChatModel

    model = OpenAIChatModel(model_name="qwen3-32b")
    auto_grader = AutoGrader(model)
    data_samples_label = [
        DataSample(
            data={
                "query": "What is the capital of France?",
            },
            samples=[
                {
                    "answer": "The capital of France is Paris.",
                    "score": 1,
                }
            ],
        )
    ]
    data_samples_unlabel = [
        DataSample(
            data={
                "query": "What is the capital of Germany?",
            },
            samples=[{"answer": "The capital of Germany is not Berlin."}],
        )
    ]

    async def main():
        grader = await auto_grader(data_samples_label)
        return await evaluate(
            grader,
            parser=None,
            data_sample=data_samples_unlabel[0],
        )

    result = asyncio.run(main())

    print(result)
