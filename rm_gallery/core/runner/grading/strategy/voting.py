# -*- coding: utf-8 -*-
import asyncio
from typing import Any, List

from rm_gallery.core.grader.base import Grader, GraderScore
from rm_gallery.core.runner.strategy.base import GraderStrategy
from rm_gallery.core.schema.data import EvalCase


class VotingStrategy(GraderStrategy):
    """Voting grader strategy that optimizes results by executing the grader
    multiple times and averaging the results.
    """

    def __init__(
        self,
        num_repeats: int = 5,
        **kwargs: Any,
    ):
        """Initialize VotingStrategy.

        Args:
            num_repeats: Number of repetitions, defaults to 5
            **kwargs: Other parameters
        """
        super().__init__(**kwargs)
        self.num_repeats = num_repeats

    async def aevaluate(
        self,
        grader: Grader,
        eval_case: EvalCase,
        *args,
        **kwargs: Any,
    ) -> List[GraderScore]:
        """Optimize reward results by voting (repeating execution and averaging).

        Args:
            eval_case: EvalCase containing data and samples
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List of optimized reward results with scores averaged over multiple runs
        """
        # Collect all repeated execution tasks
        tasks = [
            grader.aevaluate_case(eval_case, *args, **kwargs)
            for _ in range(self.num_repeats)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Calculate average scores
        if not results:
            return []

        # Initialize averaged results list
        averaged_results = []
        num_samples = len(
            results[0],
        )  # Assume all results have the same length

        for i in range(num_samples):
            # Get scores for the i-th sample from all repetitions
            scores = [result[i].score for result in results]
            reasons = [result[i].reason for result in results]

            # Calculate average score
            avg_score = sum(scores) / len(scores)

            # Create new GraderScore with detailed voting information
            averaged_results.append(
                GraderScore(
                    name=grader.name,
                    score=avg_score,
                    reason=f"Voting optimization over {self.num_repeats} runs. "
                    f"Individual scores: {scores}, reasons: {reasons}",
                    metadata={
                        f"attempt_{j+1}": result[i] for j, result in enumerate(results)
                    },
                ),
            )

        return averaged_results
