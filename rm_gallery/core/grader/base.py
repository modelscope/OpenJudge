# -*- coding: utf-8 -*-
import asyncio

# import inspect
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List

from loguru import logger

from rm_gallery.core.schema.data import (
    DataSample,
    DataSampleParser,
    parse_data_sample,
)
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.grader import (
    GraderError,
    GraderInfo,
    GraderMode,
    GraderRank,
    GraderScore,
)
from rm_gallery.core.schema.template import Chat, RequiredField, Template
from rm_gallery.core.utils.concurrency import ConcurrencyManager


class Grader(ABC):
    """Base class for graders.

    This abstract base class defines the interface for all graders.
    Subclasses must implement the evaluate method.

    Attributes:
        name (str): The name of the grader.
        mode (GraderMode): The grader mode (pointwise or listwise).
        description: The description of the grader.
        required_fields (List[RequiredField]): The required fields for the grader.
    """

    def __init__(
        self,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
        required_fields: List[RequiredField | dict] = [],
        **kwargs,
    ):
        """Initialize a Grader.

        Args:
            name: The name of the grader.
            mode: The grader mode. Defaults to POINTWISE.
            description: The description of the grader.
            required_fields: The required fields for the grader.
        """
        self.name = name
        self.mode = mode
        self.description = description
        self.required_fields = [
            RequiredField(**field) if isinstance(field, dict) else field
            for field in required_fields
        ]
        self.kwargs = {}
        self.reset(**kwargs)

    def reset(self, **kwargs):
        """Reset the grader.

        Args:
            **kwargs: Additional keyword arguments.
        """
        self.kwargs.update(kwargs)

    @property
    def meta(self) -> GraderInfo:
        """Get the metadata of the grader.

        Returns:
            GraderInfo: The metadata of the grader.
        """
        return GraderInfo(
            name=self.name,
            mode=self.mode,
            description=self.description,
            required_fields=self.required_fields,
        )

    @abstractmethod
    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate method to be implemented by subclasses.

        This abstract method must be implemented by all Grader subclasses. It performs
        the actual evaluation logic and returns either a score or a ranking based on
        the grader's mode (pointwise or listwise).

        In pointwise mode, each sample is evaluated independently, returning a
        GraderScore with a numerical value and explanation. In listwise mode, all
        samples are evaluated together, returning a GraderRank with a ranked list and
        explanation.

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     The specific arguments depend on the grader implementation but
                     typically include fields like 'query', 'answer', 'context', etc.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical score (typically 0.0-1.0 or 1-5 scale)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of items (e.g., [1, 3, 2] means first
                      item is best, third item is second best, second item is worst)
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise grader
            >>> class AccuracyGrader(Grader):
            ...     def __init__(self):
            ...         super().__init__(
            ...             name="accuracy",
            ...             mode=GraderMode.POINTWISE,
            ...             description="Evaluates factual accuracy of answers"
            ...         )
            ...
            ...     async def evaluate(self, query: str, answer: str, **kwargs) -> GraderScore:
            ...         # Implementation would evaluate accuracy
            ...         return GraderScore(
            ...             score=0.8,
            ...             reason="Answer is mostly accurate but missing some details"
            ...         )
            ...
            >>> # Example for listwise grader
            >>> class RelevanceRanker(Grader):
            ...     def __init__(self):
            ...         super().__init__(
            ...             name="relevance_ranking",
            ...             mode=GraderMode.LISTWISE,
            ...             description="Ranks answers by relevance"
            ...         )
            ...
            ...     async def evaluate(self, query: str, answer_1: str, answer_2: str, **kwargs) -> GraderRank:
            ...         # Implementation would rank answers by relevance
            ...         return GraderRank(
            ...             rank=[1, 2],
            ...             reason="First answer is more relevant to the query than the second"
            ...         )
        """
        ...

    async def _safe_evaluate(
        self,
        **kwargs,
    ) -> GraderScore | GraderRank | GraderError:
        """Safely evaluate, handling exceptions gracefully and control concurrency.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The grader result or error result.
        """
        concurrency_manager = ConcurrencyManager()

        async def _evaluate_task():
            try:
                # Check if self.evaluate is a coroutine function
                if asyncio.iscoroutinefunction(self.evaluate):
                    result = await self.evaluate(**kwargs)
                else:
                    # If it's a synchronous function, run it in a thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.evaluate(**kwargs),
                    )
                return result
            except Exception as e:
                error_msg = f"Error in {self.name} during evaluation: {str(e)}"
                logger.error(error_msg)
                return GraderError(reason=error_msg)

        # Use the concurrency manager to control execution
        return await concurrency_manager.run_with_concurrency_control(
            _evaluate_task(),
        )

    async def evaluate_data_sample(
        self,
        data_sample: DataSample,
        parser: DataSampleParser | None = None,
        *args,
        **kwargs,
    ) -> List[GraderScore]:
        """Main entry point to evaluate data sample.

        Evaluates one data samples using the  grader.

        Args:
            data_sample (DataSample):
                The data sample to evaluate.
                DataSample consists of:
                    - data: A dictionary containing shared data for all samples
                    - samples: A list of dictionaries, each representing an individual
                    sample to evaluate

            parser (DataSampleParser | Callable | None, optional):
                Optional parser to transform the data sample before evaluation. This
                allows for mapping field names between the data structure and what the
                grader expects. Can be:
                1. A dictionary with direct field mappings where paths start with "data" or "sample"
                2. A callable function that takes a DataSample and returns a DataSample
                3. None, in which case the data sample is used as is
                Defaults to None.
            *args:
                Additional positional arguments to pass to the grader.
            **kwargs:
                Additional keyword arguments to pass to the grader.

        Returns:
            List[GraderScore] | List[List[GraderScore]]:
                For a single DataSample: a list of GraderScore objects, one for each
                sample within the DataSample.

                For a list of DataSamples: a list of lists of GraderScore objects,
                where each inner list contains the scores for the corresponding
                DataSample in the input list.

                Each GraderScore contains:
                    - score: A numerical score assigned by the grader
                    - reason: Explanation of how the score was determined
                    - metadata: Optional additional information from the evaluation

        Raises:
            ValueError: If grader function signature is invalid.

        Example:
            >>> from rm_gallery.core.schema.data import DataSample
            >>> from rm_gallery.core.grader.base import LLMGrader, evaluate
            >>>
            >>> # Create data sample
            >>> data_sample = DataSample(
            ...     data={"query": "What is the capital of France?"},
            ...     samples=[
            ...         {"answer": "Paris"},
            ...         {"answer": "London"}
            ...     ]
            ... )
            >>>
            >>> # Create grader
            >>> grader = LLMGrader(
            ...     name="factual_accuracy",
            ...     template=[
            ...         {
            ...             "role": "system",
            ...             "content": "You are evaluating factual accuracy."
            ...         },
            ...         {
            ...             "role": "user",
            ...             "content": "Question: {query}\\nAnswer: {answer}\\nRate accuracy (0-1):"
            ...         }
            ...     ],
            ...     model={"model_name": "qwen-plus"}
            ... )
            >>>
            >>> # Evaluate
            >>> results = await grader.evaluate(data_sample=data_sample)
            >>> print(results)
        """
        if parser is not None:
            data_sample = parse_data_sample(data_sample, parser)

        if self.mode == GraderMode.POINTWISE:
            # Pointwise: Evaluate each sample individually
            coroutines = [
                self._safe_evaluate(**data_sample.data, **sample)
                for sample in data_sample.samples
            ]
            results: List[GraderScore] = await asyncio.gather(*coroutines)  # type: ignore
            _results = []
            for result in results:
                if isinstance(result, GraderScore):
                    _results.append(result)
                elif isinstance(result, GraderError):
                    _results.append(
                        GraderScore(score=0.0, reason=result.reason),
                    )
                else:
                    raise ValueError(f"Invalid result type: {type(result)}")
            return results

        elif self.mode == GraderMode.LISTWISE:
            # Listwise: Evaluate all samples together in one call
            params = {key: value for key, value in kwargs.items()}
            params.update(data_sample.data)
            if len(data_sample.samples) > 1:
                if data_sample.samples:
                    for key in data_sample.samples[0].keys():
                        params[key] = [
                            sample[key] for sample in data_sample.samples
                        ]

            result = await self._safe_evaluate(**params)
            if isinstance(result, GraderRank):
                results = [
                    GraderScore(
                        score=score,
                        reason=result.reason,
                    )
                    for score in result.rank
                ]
            elif isinstance(result, GraderError):
                results = [GraderScore(score=0.0, reason=result.reason)]
            else:
                raise ValueError(f"Invalid result type: {type(result)}")

            return results
        else:
            raise ValueError(f"Invalid grader mode: {self.mode}")


class LLMGrader(Grader):
    """LLM-based evaluation grader.

    A grader that uses a large language model to perform evaluations.

    Attributes:
        name (str): The name of the grader.
        mode (GraderMode): The grader mode.
        chat (Chat): The chat template for the LLM.
        rubrics (str): The rubrics for the evaluation.
        kwargs (dict): The kwargs for the grader.
    """

    def __init__(
        self,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        template: Template | dict | None = None,
        model: ChatModelBase | None = None,
        rubrics: str = "",
        description: str = "",
        required_fields: List[RequiredField] | List[dict] = [],
        **kwargs,
    ):
        """Initialize an LLMGrader.

        Args:
            name: The name of the grader.
            mode: The grader mode.
            template: The chat template for the LLM.
            model: The model parameters for the LLM.
            rubrics: The rubrics for the evaluation.
            description: The description of the grader.
            required_fields: The required fields for the grader.
            kwargs: The kwargs for the grader.
        """
        if template is None:
            raise ValueError("Template is not set")
        self.template = (
            template
            if isinstance(template, Template)
            else Template(**template)
        )
        super().__init__(
            name=name,
            mode=mode,
            description=description,
            required_fields=required_fields,
            model=model,
            rubrics=rubrics,
            **kwargs,
        )

    def reset(
        self,
        model: ChatModelBase | None = None,
        rubrics: str = "",
        **kwargs,
    ):
        """
        Reset the grader with new model and rubrics.
        """
        super().reset(**kwargs)
        self.model = model if model is not None else self.model
        self.rubrics = rubrics
        if self.template is not None and self.model is not None:
            self.chat = Chat(template=self.template, model=self.model)
        else:
            raise ValueError("Chat template or model is not set")

    def to_dict(self) -> dict:
        """Convert the grader to a dictionary.

        Returns:
            A dictionary representation of the LLM grader.
        """
        return {
            "name": self.name,
            "mode": self.mode,
            "description": self.description,
            "required_fields": [
                field.model_dump() for field in self.required_fields
            ],
            "template": self.template.model_dump(),
            "rubrics": self.rubrics,
            **self.kwargs,
        }

    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using LLM.

        Performs evaluation using a large language model according to the configured
        template and rubrics. The method formats the input parameters according to the
        template, sends the request to the LLM, and parses the structured response into
        either a GraderScore or GraderRank depending on the grader mode.

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     These are passed to the LLM template and typically include fields
                     like 'query', 'answer', 'context', etc. The specific fields depend
                     on the template definition.

        Returns:
            GraderScore | GraderRank: The evaluation result from the LLM.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical score assigned by the LLM
                    - reason (str): LLM's explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional information from the LLM

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of items as determined by the LLM
                    - reason (str): LLM's explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional information from the LLM

        Raises:
            ValueError: If the grader mode is unsupported or chat template is not set.

        Example:
            >>> # Example for pointwise LLM grader
            >>> from rm_gallery.core.model.dashscope_llm import DashScopeLLM
            >>> grader = LLMGrader(
            ...     name="helpfulness",
            ...     mode=GraderMode.POINTWISE,
            ...     template=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "{query}\\n{answer}\\n\\nRate helpfulness:"}
            ...     ],
            ...     model=DashScopeLLM(model_name="qwen-plus"),
            ...     rubrics="Rate the helpfulness of the answer (0.0-1.0)"
            ... )
            >>> result = await grader.evaluate(
            ...     query="How do I make a cake?",
            ...     answer="Preheat oven to 350F, mix ingredients, bake for 30 minutes."
            ... )
            >>> print(result.score, result.reason)
            0.9 Well-structured answer providing essential steps

            >>> # Example for listwise LLM grader
            >>> ranking_grader = LLMGrader(
            ...     name="relevance_ranking",
            ...     mode=GraderMode.LISTWISE,
            ...     template=[
            ...         {"role": "system", "content": "Rank the following answers by relevance."},
            ...         {"role": "user", "content": "Query: {query}\\nAnswers:\\n1. {answer_1}\\n2. {answer_2}"}
            ...     ],
            ...     model=DashScopeLLM(model_name="qwen-plus"),
            ...     rubrics="Rank answers by relevance (1 is most relevant)"
            ... )
            >>> result = await ranking_grader.evaluate(
            ...     query="What is the capital of France?",
            ...     answer_1="Paris is the capital of France.",
            ...     answer_2="France is a country in Europe."
            ... )
            >>> print(result.rank, result.reason)
            [1, 2] First answer directly addresses the query while second is tangential
        """
        if self.mode == GraderMode.LISTWISE:
            structured_model = GraderRank
        else:
            structured_model = GraderScore

        # Check if chat is not None before calling it
        if self.chat is None:
            raise ValueError("Chat template is not set")
        params = {"rubrics": self.rubrics, **self.kwargs}
        params.update(kwargs)

        response = await self.chat(structured_model=structured_model, **params)
        if self.mode == GraderMode.LISTWISE:
            result = GraderRank(
                rank=response.metadata["rank"],  # type: ignore
                reason=response.metadata["reason"],  # type: ignore
                metadata=response.metadata.get("meta_data", {}),  # type: ignore
            )
        elif self.mode == GraderMode.POINTWISE:
            result = GraderScore(
                score=response.metadata["score"],  # type: ignore
                reason=response.metadata["reason"],  # type: ignore
                metadata=response.metadata.get("meta_data", {}),  # type: ignore
            )
        else:
            raise ValueError(f"Unsupported grader mode: {self.mode}")
        return result


class FunctionGrader(Grader):
    """Function-based grader.

    A grader that uses a provided function to perform evaluations.

    Attributes:
        func (Callable): The function to use for evaluation.
        name (str): The name of the grader.
        mode (GraderMode): The grader mode.
    """

    def __init__(
        self,
        func: Callable,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
        required_fields: List[RequiredField] = [],
        **kwargs,
    ):
        """Initialize a FunctionGrader.

        Args:
            func: The function to use for evaluation.
            name: The name of the grader.
            mode: The grader mode.
            description: The description of the grader.
        """
        super().__init__(
            name,
            mode,
            description,
            required_fields=required_fields,
            **kwargs,
        )
        self.func = func

    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using a function.

        Performs evaluation by calling the wrapped function with the provided arguments.
        The function must return either a GraderScore (for pointwise mode) or a
        GraderRank (for listwise mode) object.

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     These are passed directly to the wrapped function and typically
                     include fields like 'query', 'answer', 'context', etc. The specific
                     fields depend on the function's requirements.

        Returns:
            GraderScore | GraderRank: The evaluation result from the wrapped function.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical score computed by the function
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of items computed by the function
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Raises:
            TypeError: If result type doesn't match grader mode (e.g., function returns
                      GraderScore in listwise mode).

        Example:
            >>> # Example for pointwise function grader
            >>> def accuracy_function(query: str, answer: str) -> GraderScore:
            ...     # Simple accuracy function - checks if answer contains key facts
            ...     if "Paris" in answer and "capital" in answer.lower():
            ...         return GraderScore(score=1.0, reason="Correctly identifies Paris as capital")
            ...     else:
            ...         return GraderScore(score=0.0, reason="Missing key information")
            ...
            >>> grader = FunctionGrader(
            ...     func=accuracy_function,
            ...     name="accuracy_checker",
            ...     mode=GraderMode.POINTWISE
            ... )
            >>> result = await grader.evaluate(
            ...     query="What is the capital of France?",
            ...     answer="Paris is the capital of France."
            ... )
            >>> print(result.score, result.reason)
            1.0 Correctly identifies Paris as capital

            >>> # Example for listwise function grader
            >>> def relevance_ranker(query: str, answer_1: str, answer_2: str) -> GraderRank:
            ...     # Simple ranking function - longer answer assumed more relevant
            ...     if len(answer_1) > len(answer_2):
            ...         return GraderRank(rank=[1, 2], reason="First answer is more detailed")
            ...     else:
            ...         return GraderRank(rank=[2, 1], reason="Second answer is more detailed")
            ...
            >>> ranking_grader = FunctionGrader(
            ...     func=relevance_ranker,
            ...     name="length_ranker",
            ...     mode=GraderMode.LISTWISE
            ... )
            >>> result = await ranking_grader.evaluate(
            ...     query="Explain photosynthesis",
            ...     answer_1="Photosynthesis converts light to energy.",
            ...     answer_2="Photosynthesis is the process by which plants convert light energy into chemical energy."
            ... )
            >>> print(result.rank, result.reason)
            [2, 1] Second answer is more detailed
        """
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.func, **kwargs)

        # Check return type based on grader mode
        if self.mode == GraderMode.POINTWISE:
            if not isinstance(result, GraderScore):
                raise TypeError(
                    f"Expected GraderScore for pointwise mode, got {type(result)}",
                )
        elif self.mode == GraderMode.LISTWISE:
            if not isinstance(result, GraderRank):
                raise TypeError(
                    f"Expected GraderRank for listwise mode, got {type(result)}",
                )
        else:
            raise ValueError(f"Unsupported grader mode: {self.mode}")

        return result

    @classmethod
    def wrap(cls, func: Callable) -> Callable:
        """Wrap a function as a grader.

        Returns:
            The Callable grader.
        """

        return partial(FunctionGrader, func=func)
