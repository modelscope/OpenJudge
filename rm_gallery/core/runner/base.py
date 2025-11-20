# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, List

from rm_gallery.core.schema.data import EvalCase


class BaseRunner(ABC):
    """
    Base class for auto-runners.
    """

    @abstractmethod
    async def aevaluate_batch(
        self,
        eval_cases: List[EvalCase],
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """
        Auto-Runner on the data.
        Args:
            eval_cases: The training data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing the results.
        """
        ...
