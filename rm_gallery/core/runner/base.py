# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List

from rm_gallery.core.schema.data import DataSample


class BaseRunner(ABC):
    """
    Base class for auto-runners.
    """

    @abstractmethod
    async def __call__(
        self,
        data_samples: List[DataSample],
        *args,
        **kwargs,
    ) -> dict:
        """
        Auto-Runner on the data.
        Args:
            data_samples: The training data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing the results.
        """
        ...
