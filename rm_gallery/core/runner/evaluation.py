import asyncio
from typing import Any, Callable, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.data import DataSample, DataSampleParser, validate_data_samples
from rm_gallery.core.grader import Grader, GraderScore, evaluate
from rm_gallery.core.registry import GR
from rm_gallery.core.runner.base import BaseRunner
from rm_gallery.core.utils.concurrency import ConcurrencyManager


class GraderConfig(BaseModel):
    """Configuration for a grader."""

    grader: str | dict | Any = Field(
        ...,
        description="Grader to use for the experiment",
    )
    parser: dict | Any = Field(
        None,
        description="Parser to use for the experiment",
    )

    def to_instance(
        self,
    ) -> tuple[Grader | Callable, DataSampleParser | Callable | None]:
        grader = self.grader
        parser = self.parser
        if isinstance(self.grader, str):
            grader = GR.get(self.grader)

        if isinstance(self.parser, dict):
            parser = DataSampleParser(**parser)
        return grader, parser


class EvaluationRunner(BaseRunner):
    """Runner for evaluating graders."""

    def __init__(self, grader_configs: dict, max_concurrent: int = 32):
        """Initialize the EvaluationRunner.

        Args:
            graders: dict of graders to use for the experiment
            parsers: Parsers for the graders
            max_concurrent: Maximum number of concurrent evaluations
        """
        self.grader_configs = grader_configs
        # Set global concurrency limit using the manager class
        concurrency_manager = ConcurrencyManager()
        concurrency_manager.set_max_concurrent(max_concurrent)

    async def evaluate(self, data_sample: DataSample) -> Dict[str, List[GraderScore]]:
        """Run experiment for a single sample.

        Args:
            data_sample: The data sample to evaluate

        Returns:
            List of evaluation results
        """
        results = []
        coroutines = []
        for key, config in self.grader_configs.items():
            grader, parser = GraderConfig(**config).to_instance()
            coro = evaluate(grader=grader, parser=parser, data_sample=data_sample)
            coroutines.append(coro)
        results = await asyncio.gather(*coroutines)
        return {
            key: results for key, results in zip(self.grader_configs.keys(), results)
        }

    async def __call__(self, data_samples: List[DataSample], *args, **kwargs) -> dict:
        """Run experiment.

        Args:
            dataset: The evaluation dataset

        Returns:
            Evaluation result
        """
        results = []
        coroutines = []

        # Create async tasks for each data sample
        for data_sample in data_samples:
            coroutines.append(self.evaluate(data_sample))

        # Execute all tasks in parallel
        results = await asyncio.gather(*coroutines)
        logger.info(f"Results: {results}")

        # TODO: summary of results
        return {
            "results": results,
        }


if __name__ == "__main__":
    data_sample_schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            "samples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
        "required": ["data", "samples"],
    }
    data_samples = [
        {
            "data": {
                "query": "What is the capital of France?",
            },
            "samples": [{"answer": "Paris"}, {"answer": "Marseille"}],
        },
        {
            "data": {
                "query": "What is the capital of Germany?",
            },
            "samples": [{"answer": "Berlin"}, {"answer": "Munich"}],
        },
    ]
    data_samples = validate_data_samples(data_samples, data_sample_schema)
    from rm_gallery.gallery.example.llm import FactualGrader

    runner = EvaluationRunner(
        grader_configs={
            "factual_grader": {
                "grader": FactualGrader(),
            }
        }
    )
    # Run using async method
    result = asyncio.run(runner(data_samples=data_samples))
