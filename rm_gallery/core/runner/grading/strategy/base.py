# -*- coding: utf-8 -*-
import asyncio
from abc import ABC, abstractmethod
from typing import Any, List

from rm_gallery.core.grader.base import Grader, GraderScore
from rm_gallery.core.schema.data import EvalCase


class GraderStrategy(ABC):
    """Base grader strategy class for optimizing input grader functions.

    This class serves as an abstract base class that defines the basic interface
    for grader strategies. Subclasses should implement the specific optimization logic.
    """

    @abstractmethod
    async def aevaluate(
        self,
        grader: Grader,
        eval_case: EvalCase,
        *args,
        **kwargs: Any,
    ) -> List[GraderScore]:
        """Core method for optimizing grader functions.

        Args:
            eval_case: EvalCase containing data and samples
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List of optimized grader results
        """
        ...

    async def aevaluate_batch(
        self,
        grader: Grader,
        eval_cases: List[EvalCase],
        *args,
        **kwargs: Any,
    ) -> List[List[GraderScore]]:
        """Evaluate a batch of eval cases using the optimized grader function.

        Args:
            grader: Grader instance
            eval_cases: List of EvalCase instances
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        Returns:
            List of lists of optimized grader results
        """
        coroutines = [
            self.aevaluate(grader, eval_case, *args, **kwargs)
            for eval_case in eval_cases
        ]
        return await asyncio.gather(*coroutines)
