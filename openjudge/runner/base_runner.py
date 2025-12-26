# -*- coding: utf-8 -*-
"""
Abstract base class for runners that execute evaluations.

This module defines the abstract base class for runners that execute evaluators
on datasets and collect their results. Runners are responsible for managing the
execution flow of evaluations, including concurrency control and result organization.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from openjudge.graders.schema import GraderResult

RunnerResult = Dict[str, List[GraderResult]]


class BaseRunner(ABC):
    """Abstract base class for runners that execute evaluations.

    This class defines the interface for runners that execute evaluators on datasets
    and collect their results. Concrete implementations should handle various aspects
    such as concurrency control, data mapping, and result organization.

    Attributes:
        max_concurrency (int): Maximum number of concurrent operations.

    Example:
        >>> class MyRunner(BaseRunner):
        ...     async def arun(self, dataset, *args, **kwargs):
        ...         # Implementation here
        ...         pass
        ...     def run(self, dataset, *args, **kwargs):
        ...         # Implementation here
        ...         pass
        >>> runner = MyRunner(max_concurrency=10)
        >>> print(runner.max_concurrency)
        10
    """

    def __init__(self, max_concurrency: int = 32) -> None:
        """Initialize the base runner.

        Args:
            max_concurrency: Maximum number of concurrent operations. Defaults to 32.

        Example:
            >>> runner = BaseRunner(max_concurrency=16)
            >>> print(runner.max_concurrency)
            16
        """
        self.max_concurrency = max_concurrency

    @abstractmethod
    async def arun(
        self,
        dataset: List[dict],
        *args: Any,
        **kwargs: Any,
    ) -> RunnerResult:
        """Run evaluators on the provided data asynchronously.

        This method should execute evaluators on the provided data samples
        and return the results organized by evaluator.

        Args:
            dataset: List of data samples to evaluate. Each dictionary represents
                a sample with input parameters and expected outputs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnerResult: Results of the evaluation run organized by evaluator.
                The keys are evaluator names and values are lists of GraderResult objects.

        Example:
            >>> # This is an abstract method that must be implemented by subclasses
            >>> # Implementation would execute evaluators on data and return results
        """

    def run(
        self,
        dataset: List[dict],
        *args: Any,
        **kwargs: Any,
    ) -> RunnerResult:
        """Run evaluators on the provided data synchronously.

        This method should execute evaluators on the provided data samples
        and return the results organized by evaluator.

        Args:
            dataset: List of data samples to evaluate. Each dictionary represents
                a sample with input parameters and expected outputs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnerResult: Results of the evaluation run organized by evaluator.
                The keys are evaluator names and values are lists of GraderResult objects.
        """
        return asyncio.run(self.arun(dataset, *args, **kwargs))
