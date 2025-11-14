import asyncio
from typing import Callable, Dict, List, Tuple, TypedDict

from loguru import logger

from rm_gallery.core.data import DataSample, DataSampleParser, validate_data_samples
from rm_gallery.core.grader import Grader, GraderScore, evaluate
from rm_gallery.core.registry import GR
from rm_gallery.core.runner.base import BaseRunner
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
    if isinstance(parser, dict):
        parser = DataSampleParser(**parser)
    return grader, parser


class GradingRunner(BaseRunner):
    """Runner for grading by graders."""

    def __init__(
        self, grading_configs: Dict[str, GradingConfig], max_concurrent: int = 32
    ):
        """Initialize the EvaluationRunner.

        Args:
            graders: dict of graders to use for the experiment
            parsers: Parsers for the graders
            max_concurrent: Maximum number of concurrent evaluations
        """
        self.grading_configs = grading_configs
        # Set global concurrency limit using the manager class
        concurrency_manager = ConcurrencyManager()
        concurrency_manager.set_max_concurrent(max_concurrent)

    async def evaluate(self, data_sample: DataSample) -> GradingResult:
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
                coro = evaluate(grader=grader, parser=parser, data_sample=data_sample)
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

    runner = GradingRunner(
        grading_configs={"factual_grader": {"grader": FactualGrader(), "weight": 1.0}}
    )
    # Run using async method
    result = asyncio.run(runner(data_samples=data_samples))
