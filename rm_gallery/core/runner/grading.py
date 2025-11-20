# -*- coding: utf-8 -*-
"""
Grading runner for evaluating data samples using graders.
"""
import asyncio
from typing import Any, Callable, Dict, List, Tuple, TypedDict

from loguru import logger

from rm_gallery.core.grader.base import Grader, GraderScore
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.registry import GR
from rm_gallery.core.runner.base import BaseRunner
from rm_gallery.core.schema.data import (
    DataSample,
    DataSampleParser,
    validate_data_samples,
)
from rm_gallery.core.utils.concurrency import ConcurrencyManager
from rm_gallery.core.utils.instance import init_instance_by_config


class GradingConfig(TypedDict, total=False):
    """Configuration for a grader."""

    grader: str | dict | Callable | Grader
    parser: dict | Callable
    weight: float


class GradingResult(TypedDict):
    """Result of a grading experiment."""

    total_score: float
    dimensions: Dict[str, GraderScore]


def parse_grading_config(
    config: GradingConfig,
) -> Tuple[Grader | Callable | None, DataSampleParser | Callable | None]:
    """Parse config into grader and parser."""
    grader_config = config.get("grader")  # type: ignore
    grader = None
    if isinstance(grader_config, str):
        grader = GR.get(grader_config)
    elif grader_config is not None:
        grader = init_instance_by_config(grader_config, accept_type=Grader)
    else:
        raise ValueError("Grader config must be a string or a dict")

    parser = config.get("parser", None)
    return grader, parser


class GradingRunner(BaseRunner):
    """Runner for grading by graders."""

    def __init__(
        self,
        grading_configs: Dict[str, GradingConfig],
        max_concurrent: int = 32,
    ):
        """Initialize the GradingRunner.

        Args:
            graders: dict of graders to use for the experiment
            parsers: Parsers for the graders
            max_concurrent: Maximum number of concurrent evaluations
        """
        self.grading_configs = grading_configs
        # Set global concurrency limit using the manager class
        concurrency_manager = ConcurrencyManager()
        concurrency_manager.set_max_concurrent(max_concurrent)

    async def aevaluate(self, data_sample: DataSample) -> GradingResult:
        """Run experiment for a single sample.

        Args:
            data_sample: The data sample to evaluate

        Returns:
            Grading result with scores for each dimension
        """
        results: Dict[str, GraderScore] = {}
        coroutines = []
        keys = []

        for key, config in self.grading_configs.items():
            grader, parser = parse_grading_config(config)
            if grader is not None:
                coro = grader.aevaluate_data_samples(
                    parser=parser,
                    data_sample=data_sample,
                )
                coroutines.append(coro)
                keys.append(key)

        scores = await asyncio.gather(*coroutines)

        total_score = 0.0
        for key, score in zip(keys, scores):
            results[key] = score
            config = self.grading_configs[key]
            weight = config.get("weight", 1.0) if "weight" in config else 1.0
            total_score += score.score * weight

        return {"total_score": total_score, "dimensions": results}

    async def __call__(
        self,
        data_samples: List[DataSample],
        *args: Any,
        **kwargs: Any,
    ) -> dict:
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
            coroutines.append(self.aevaluate(data_sample))

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
    test_samples = [
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
    test_samples = validate_data_samples(test_samples, data_sample_schema)
    from rm_gallery.gallery.grader.alignment.honesty.factuality import FactualityGrader

    model = OpenAIChatModel(model_name="qwen-plus")

    runner = GradingRunner(
        grading_configs={
            "factual_grader": {
                "grader": FactualityGrader(model=model),
                "weight": 1.0,
            },
        },
    )
    # Run using async method
    result = asyncio.run(runner(data_samples=test_samples))
