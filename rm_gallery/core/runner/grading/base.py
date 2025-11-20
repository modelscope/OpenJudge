# -*- coding: utf-8 -*-
"""
Grading runner for evaluating data samples using graders.
"""
import asyncio
from typing import Any, Callable, Dict, List, Tuple, TypedDict

from loguru import logger

from rm_gallery.core.grader.base import Grader
from rm_gallery.core.runner.base import BaseRunner
from rm_gallery.core.runner.grading.strategy.base import GraderStrategy
from rm_gallery.core.schema.data import EvalCase, EvalCaseParser
from rm_gallery.core.schema.grader import GraderScore
from rm_gallery.core.utils.concurrency import ConcurrencyManager
from rm_gallery.core.utils.instance import init_instance_by_config


class GradingConfig(TypedDict, total=False):
    """Configuration for a grader."""

    grader: Grader
    parser: EvalCaseParser | None
    strategy: GraderStrategy | None
    weight: float | None


class GradingResult(TypedDict):
    """Result of a grading experiment."""

    total_score: float
    dimensions: Dict[str, GraderScore]


def parse_grading_config(
    config: GradingConfig,
) -> Tuple[
    Grader | Callable | None,
    EvalCaseParser | Callable | None,
    GraderStrategy | None,
]:
    """Parse config into grader and parser."""
    grader_config = config.get("grader")  # type: ignore
    grader = None
    if grader_config is not None:
        grader = init_instance_by_config(grader_config, accept_type=Grader)
    else:
        raise ValueError("Grader config must be a string or a dict")

    strategy = config.get("strategy", None)
    parser = config.get("parser", None)
    return grader, parser, strategy


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

    async def aevaluate(self, eval_case: EvalCase) -> GradingResult:
        """Run experiment for a single sample.

        Args:
            eval_case: The eval case to evaluate

        Returns:
            Grading result with scores for each dimension
        """
        results: Dict[str, GraderScore] = {}
        coroutines = []
        keys = []

        for key, config in self.grading_configs.items():
            grader, parser, strategy = parse_grading_config(config)
            if grader is not None:
                if strategy is not None:
                    coroutine = strategy.aevaluate_batch(grader, [eval_case])
                else:
                    coroutine = grader.aevaluate_batch(
                        parser=parser,
                        eval_cases=[eval_case],
                    )
                coroutines.append(coroutine)
                keys.append(key)

        scores = await asyncio.gather(*coroutines)

        total_score = 0.0
        for key, score in zip(keys, scores):
            results[key] = score[0]
            config = self.grading_configs[key]
            weight = config.get("weight", 1.0) if "weight" in config else 1.0
            total_score += score.score * weight

        return {"total_score": total_score, "dimensions": results}

    async def aevaluate_batch(
        self,
        eval_cases: List[EvalCase],
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

        # Create async tasks for each eval case
        for eval_case in eval_cases:
            coroutines.append(self.aevaluate(eval_case))

        # Execute all tasks in parallel
        results = await asyncio.gather(*coroutines)
        logger.info(f"Results: {results}")

        # TODO: summary of results
        return {
            "results": results,
        }
