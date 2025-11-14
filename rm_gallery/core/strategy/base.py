from abc import ABC, abstractmethod
from typing import List

from rm_gallery.core.data import DataSample
from rm_gallery.core.grader import GraderScore


class GraderStrategy(ABC):
    """Base grader strategy class for optimizing input grader functions.

    This class serves as an abstract base class that defines the basic interface
    for grader strategies. Subclasses should implement the specific optimization logic.
    """

    @abstractmethod
    async def __call__(
        self, data_sample: DataSample, *args, **kwargs
    ) -> List[GraderScore]:
        """Core method for optimizing grader functions.

        Args:
            data_sample: Data sample containing data and samples
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List of optimized grader results
        """
        ...
