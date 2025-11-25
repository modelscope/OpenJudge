# -*- coding: utf-8 -*-
"""Base class for graders."""
import asyncio
import os

# import inspect
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, List, Type

from loguru import logger
from pydantic import BaseModel

from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.data import EvalCase, EvalCaseParser, parse_eval_case
from rm_gallery.core.schema.grader import (
    GraderError,
    GraderInfo,
    GraderMode,
    GraderRank,
    GraderScore,
    GraderRankCallback,
    GraderScoreCallback,
)
from rm_gallery.core.schema.template import Chat, LanguageEnum, Template
from rm_gallery.core.utils.concurrency import ConcurrencyManager


class Grader(ABC):
    """Base class for graders.

    This abstract base class defines the interface for all graders.
    Subclasses must implement the _aevaluate method.

    Attributes:
        name (str): The name of the grader.
        mode (GraderMode): The grader mode (pointwise or listwise).
        description (str): Description of what this grader evaluates.
    """

    def __init__(
        self,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
        **kwargs: Any,
    ):
        """Initialize a Grader.

        Args:
            name: The name of the grader. Used for identification and logging.
            mode: The grader mode. Either POINTWISE (individual sample evaluation) 
                  or LISTWISE (joint evaluation of multiple samples). 
                  Defaults to POINTWISE.
            description: Human-readable description of what this grader evaluates.
            **kwargs: Additional keyword arguments that will be stored and 
                     accessible to subclasses.
        """
        self._name = name
        self.mode = mode
        self.description = description
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        """Get the name of the grader.

        Returns:
            str: The name of the grader.
        """
        return self._name

    @abstractmethod
    async def _aevaluate(self, **kwargs: Any) -> GraderScore | GraderRank:
        """Abstract method for performing the actual evaluation logic.

        This method must be implemented by all Grader subclasses. It performs
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
                    - name (str): Name of the grader
                    - score (float): Numerical score (typically 0.0-1.0 or 1-5 scale)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - name (str): Name of the grader
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
            ...     async def _aevaluate(self, query: str, answer: str, **kwargs) -> GraderScore:
            ...         # Implementation would evaluate accuracy
            ...         return GraderScore(
            ...             name=self.name,
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
            ...     async def _aevaluate(self,
            ...                         query: str,
            ...                         answer_1: str,
            ...                         answer_2: str,
            ...                        **kwargs) -> GraderRank:
            ...         # Implementation would rank answers by relevance
            ...         return GraderRank(
            ...             name=self.name,
            ...             rank=[1, 2],
            ...             reason="First answer is more relevant to the query than the second"
            ...         )
        """

    async def aevaluate(
        self,
        **kwargs: Any,
    ) -> GraderScore | GraderRank | GraderError:
        """Public method to safely evaluate with exception handling and concurrency control.

        This method serves as a wrapper around _aevaluate that adds error handling and concurrency control.
        It catches any exceptions that occur during evaluation and wraps them in a 
        GraderError object. It also controls concurrency using the ConcurrencyManager.

        Args:
            **kwargs: Arguments for the evaluation, passed directly to _aevaluate.

        Returns:
            GraderScore | GraderRank | GraderError: The grader result or error result.
            
            On successful evaluation:
                - GraderScore in POINTWISE mode
                - GraderRank in LISTWISE mode
                
            On failure:
                - GraderError containing the error information
        """
        concurrency_manager = ConcurrencyManager()

        async def _evaluate_task() -> GraderScore | GraderRank | GraderError:
            try:
                return await self._aevaluate(**kwargs)
            except Exception as e:
                error_msg = f"Error in {self.name} during evaluation: {str(e)}"
                logger.error(error_msg)
                return GraderError(name=self.name, error=error_msg)

        # Use the concurrency manager to control execution
        return await concurrency_manager.run_with_concurrency_control(
            _evaluate_task(),
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "Grader":
        """Create a grader from a configuration dictionary.

        This class method creates a new grader instance using the provided configuration.
        It extracts standard grader properties (name, mode, description) from the config
        and passes any remaining items as additional keyword arguments.

        Args:
            config: A dictionary containing the grader configuration.
                   Expected keys include 'name', 'mode', 'description', and any
                   additional parameters required by specific grader implementations.

        Returns:
            A new instance of the grader subclass.
        """
        # Extract standard grader properties
        name = config.pop("name", "")
        mode = config.pop("mode", GraderMode.POINTWISE)
        description = config.pop("description", "")

        # Create and return new instance with remaining config items as kwargs
        return cls(
            name=name,
            mode=mode,
            description=description,
            **config,
        )

    def to_dict(self) -> dict:
        """Convert the grader to a dictionary representation.

        This method serializes the grader's essential properties (name, mode, description)
        and any additional keyword arguments into a dictionary. The mode is converted to
        its string value for serialization purposes.

        Returns:
            A dictionary containing the serialized grader information.
        """
        return {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "kwargs": self.kwargs,
        }


class LLMGrader(Grader):
    """LLM-based grader that uses large language models for evaluation.

    This class extends the base Grader class to provide LLM-based evaluation capabilities.
    It uses a language model to perform evaluations according to specified rubrics and templates.

    The LLMGrader constructs prompts using a template, sends them to an LLM, and parses
    the structured response into either a GraderScore or GraderRank depending on the mode.

    Attributes:
        template (Template): The template for generating prompts.
        model (ChatModelBase): The language model used for evaluation.
        rubrics (str): The rubrics used for evaluation.
        language (LanguageEnum): The language for the evaluation.
        callback (Callable | Type[BaseModel]): Function or Pydantic model to process model response 
                                             into GraderScore or GraderRank.
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        language: LanguageEnum | str | None = None,
        description: str = "",
        template: dict | Template | None = None,
        callback: Callable | Type[BaseModel] | None = None,
        rubrics: str = "",
        **kwargs: Any,
    ):
        """Initialize an LLMGrader.

        Args:
            model: The language model used for evaluation. Can be either a ChatModelBase
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            name: The name of the grader. Used for identification and logging.
            mode: The grader mode. Either POINTWISE (individual sample evaluation) 
                  or LISTWISE (joint evaluation of multiple samples). 
                  Defaults to POINTWISE.
            language: The language of the grader. Can be LanguageEnum, string, or None.
                     If None, defaults to environment variable LANGUAGE or "en".
            description: Human-readable description of what this grader evaluates.
            template: The template for generating prompts. Defines how inputs are formatted
                     for the LLM. Can be a dict or Template object.
            callback: The callback function or Pydantic model for processing model response 
                      to GraderScore or GraderRank.
                      
                      Can be one of the following:
                      1. A Callable that processes the response and populates metadata
                      2. A Pydantic BaseModel subclass for structured output parsing
                      3. None, in which case uses GraderScoreCallback for POINTWISE mode
                         or GraderRankCallback for LISTWISE mode
            rubrics: The rubrics used for evaluation. These guide the LLM on how to perform
                    the evaluation.
            **kwargs: Additional keyword arguments passed to the parent Grader class and
                     used in the template rendering.
        """
        super().__init__(
            name=name,
            mode=mode,
            description=description,
            **kwargs,
        )

        # Handle language parameter
        if language is None:
            language = os.environ.get("LANGUAGE", "en")

        if isinstance(language, str):
            # Convert string to LanguageEnum
            self.language = (
                LanguageEnum(language)
                if language in [item.value for item in LanguageEnum]
                else LanguageEnum.EN
            )
        else:
            self.language = language

        template = template or {}
        self.template = (
            template if isinstance(template, Template) else Template(**template)
        )

        if callback:
            self.callback = callback
        else:
            self.callback = (
                GraderScoreCallback if self.mode == GraderMode.POINTWISE else GraderRankCallback
            )

        if isinstance(model, dict):
            model = OpenAIChatModel(**model)

        self.model = model
        self.rubrics = rubrics
        if self.template is not None and self.model is not None:
            self.chat = Chat(template=self.template, model=self.model)
        else:
            raise ValueError("Chat template or model is not set")

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
        )

    def to_dict(self) -> dict:
        """Convert the LLMGrader to a dictionary representation.

        This method serializes the LLMGrader's properties (name, mode, description, template,
        model, and rubrics) into a dictionary. The mode is converted to its string value,
        and the template and model are converted to dictionaries if they are not already.

        Returns:
            A dictionary containing the serialized LLMGrader information.
        """
        return {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "template": (
                self.template.model_dump()
                if isinstance(self.template, Template)
                else self.template
            ),
            "rubrics": self.rubrics,
        }

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "LLMGrader":
        """Create an LLMGrader from a configuration dictionary.

        This class method creates a new LLMGrader instance using the provided configuration.
        It extracts standard grader properties (name, mode, description) and LLM-specific
        properties (template, model, rubrics) from the config.

        Args:
            config: A dictionary containing the LLMGrader configuration.
                   Expected keys include 'name', 'mode', 'description', 'template',
                   'model', 'rubrics', and any additional parameters.

        Returns:
            A new LLMGrader instance.
        """
        # Extract standard grader properties
        name = config.pop("name", "")
        mode = config.pop("mode", GraderMode.POINTWISE)
        description = config.pop("description", "")

        # Extract LLMGrader-specific properties
        template = config.pop("template", {})
        model = config.pop("model", {})
        rubrics = config.pop("rubrics", "")

        # Create and return new instance with remaining config items as kwargs
        return cls(
            name=name,
            mode=mode,
            description=description,
            template=template,
            model=model,
            rubrics=rubrics,
            **config,
        )

    async def _aevaluate(self, **kwargs: Any) -> GraderScore | GraderRank:
        """Evaluate using LLM.

        Performs evaluation using a large language model according to the configured
        template and rubrics. The method formats the input parameters according to the
        template, sends the request to the LLM, and parses the structured response into
        either a GraderScore or GraderRank depending on the grader mode.

        The callback mechanism supports two modes:
        1. Callable functions that process the response and populate metadata
        2. Pydantic BaseModel subclasses for structured output parsing

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     These are passed to the LLM template and typically include fields
                     like 'query', 'answer', 'context', etc. The specific fields depend
                     on the template definition.

        Returns:
            GraderScore | GraderRank: The evaluation result from the LLM.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - name (str): Name of the grader
                    - score (float): Numerical score assigned by the LLM
                    - reason (str): LLM's explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional information from the LLM

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - name (str): Name of the grader
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
            ...     model=DashScopeLLM(model="qwen-plus"),
            ...     rubrics="Rate the helpfulness of the answer (0.0-1.0)"
            ... )
            >>> result = await grader.aevaluate(
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
            ...         {"role": "user",
            ...          "content": "Query: {query}\\nAnswers:\\n1. {answer_1}\\n2. {answer_2}"}
            ...     ],
            ...     model=DashScopeLLM(model="qwen-plus"),
            ...     rubrics="Rank answers by relevance (1 is most relevant)"
            ... )
            >>> result = await ranking_grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     answer_1="Paris is the capital of France.",
            ...     answer_2="France is a country in Europe."
            ... )
            >>> print(result.rank, result.reason)
            [1, 2] First answer directly addresses the query while second is tangential
        """
        # Check if chat is not None before calling it
        if self.chat is None:
            raise ValueError("Chat template is not set")
        params = {"rubrics": self.rubrics, **self.kwargs}
        params.update(kwargs)

        response = await self.chat(
            callback=self.callback,
            language=self.language,
            **params,
        )
        metadata = response.metadata if response.metadata else {}
        if self.mode == GraderMode.LISTWISE:
            rank = metadata.pop("rank")
            reason = metadata.pop("reason")
            result = GraderRank(
                name=self.name,
                rank=rank,  # type: ignore
                reason=reason,  # type: ignore
                metadata=metadata,  # type: ignore
            )
        elif self.mode == GraderMode.POINTWISE:
            score = metadata.pop("score")
            reason = metadata.pop("reason")
            result = GraderScore(
                name=self.name,
                score=score,  # type: ignore
                reason=reason,  # type: ignore
                metadata=metadata,  # type: ignore
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
        **kwargs: Any,
    ):
        """Initialize a FunctionGrader.

        Args:
            func: The function to use for evaluation. This function will be called
                  with the evaluation data and must return either a GraderScore (for
                  pointwise mode) or a GraderRank (for listwise mode).
                  
                  For pointwise mode, typical signature:
                  ```async def my_func(query: str, answer: str, **kwargs) -> GraderScore:```
                  
                  For listwise mode, typical signature:
                  ```async def my_func(query: str, answer_1: str, answer_2: str, **kwargs) -> GraderRank:```
            name: The name of the grader. Used for identification and logging.
            mode: The grader mode. Either POINTWISE (individual sample evaluation) 
                  or LISTWISE (joint evaluation of multiple samples). 
                  Defaults to POINTWISE.
            description: Human-readable description of what this grader evaluates.
            **kwargs: Additional keyword arguments passed to the parent Grader class.
        """
        super().__init__(
            name,
            mode,
            description,
            **kwargs,
        )
        self.func = func

    async def _aevaluate(self, **kwargs: Any) -> GraderScore | GraderRank:
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
            ...         return GraderScore(name=self.name,
            ...                            score=1.0,
            ...                            reason="Correctly identifies Paris as capital")
            ...     else:
            ...         return GraderScore(name=self.name,
            ...                            score=0.0,
            ...                            reason="Missing key information")
            ...
            >>> grader = FunctionGrader(
            ...     func=accuracy_function,
            ...     name="accuracy_checker",
            ...     mode=GraderMode.POINTWISE
            ... )
            >>> result = await grader.aevaluate(
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
            >>> result = await ranking_grader.aevaluate(
            ...     query="Explain photosynthesis",
            ...     answer_1="Photosynthesis converts light to energy.",
            ...     answer_2="Photosynthesis is the process by which plants convert light "
                             "energy into chemical energy."
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
        """Decorator to wrap a function as a FunctionGrader.

        This class method allows you to easily convert a regular Python function
        into a FunctionGrader instance. The wrapped function must follow the
        FunctionGrader requirements and return either a GraderScore or GraderRank.

        Args:
            func: The function to wrap as a grader. Must return GraderScore or GraderRank.

        Returns:
            A partially applied FunctionGrader constructor that can be instantiated
            with additional parameters like mode, name, and description.

        Example:
            >>> @FunctionGrader.wrap
            >>> def my_accuracy_function(query: str, answer: str) -> GraderScore:
            >>>     # Custom accuracy evaluation logic
            >>>     score = calculate_accuracy(query, answer)
            >>>     return GraderScore(name="accuracy", score=score, reason="Custom calculation")
            >>>
            >>> # Create the grader instance
            >>> accuracy_grader = my_accuracy_function(mode=GraderMode.POINTWISE, 
            ...                                       name="my_accuracy",
            ...                                       description="My custom accuracy evaluator")
        """

        return partial(FunctionGrader, func=func)


async def aevaluate_with_case(
    grader: Grader,
    eval_case: EvalCase,
    parser: EvalCaseParser | None = None,
    *args: Any,
    **kwargs: Any,
) -> List[GraderScore]:
    """Evaluate a single evaluation case using the specified grader.

    This function evaluates an EvalCase based on the grader's mode (pointwise or listwise).
    In pointwise mode, each sample is evaluated individually. In listwise mode, all samples
    are evaluated together in one call. The function handles both GraderScore and GraderError
    results, converting errors to zero-score GraderScore instances.

    Args:
        grader: The grader instance to use for evaluation.
        eval_case: The evaluation case to evaluate, containing input data and output samples.
                  An EvalCase consists of:
                  - input: A dictionary containing shared data for all samples
                  - outputs: A list of dictionaries, each representing an individual sample to evaluate
        parser: Optional parser to transform the eval case before evaluation. Defaults to None.
               Allows for mapping field names between the data structure and what the grader expects.
        *args: Additional positional arguments passed to the grader (currently unused).
        **kwargs: Additional keyword arguments passed to the grader.

    Returns:
        A list of GraderScore objects, with one score for each sample in the eval case.
        If a GraderError occurs during evaluation, it is converted to a GraderScore with
        a score of 0.0 and the error reason as the explanation.

    Raises:
        ValueError: If the grader mode is neither POINTWISE nor LISTWISE.

    Example:
        >>> from rm_gallery.core.schema.data import EvalCase
        >>> from rm_gallery.core.grader.base import LLMGrader
        >>>
        >>> # Create eval case
        >>> eval_case = EvalCase(
        ...     input={"query": "What is the capital of France?"},
        ...     outputs=[
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
        ...     model={"model": "qwen-plus"}
        ... )
        >>>
        >>> # Evaluate
        >>> results = await aevaluate_with_case(grader, eval_case)
        >>> print(results)
    """
    del args
    if parser is not None:
        eval_case = parse_eval_case(eval_case, parser)

    if grader.mode == GraderMode.POINTWISE:
        # Pointwise: Evaluate each sample individually
        coroutines = [
            grader.aevaluate(**eval_case.input, **output)
            for output in eval_case.outputs
        ]
        results: List[GraderScore] = await asyncio.gather(*coroutines)  # type: ignore
        _results = []
        for result in results:
            if isinstance(result, GraderScore):
                _results.append(result)
            elif isinstance(result, GraderError):
                _results.append(
                    GraderScore(
                        name=result.name,
                        score=0.0,
                        reason=result.error,
                    ),
                )
            else:
                raise ValueError(f"Invalid result type: {type(result)}")
        return results

    elif grader.mode == GraderMode.LISTWISE:
        # Listwise: Evaluate all samples together in one call
        params = kwargs
        params.update(eval_case.input)
        if len(eval_case.outputs) > 1:
            if eval_case.outputs:
                for key in eval_case.outputs[0].keys():
                    params[key] = [output[key] for output in eval_case.outputs]

        result = await grader.aevaluate(**params)
        if isinstance(result, GraderRank):
            results = [
                GraderScore(
                    name=result.name,
                    score=score,
                    reason=result.reason,
                    metadata=result.metadata,
                )
                for score in result.rank
            ]
        elif isinstance(result, GraderError):
            results = [
                GraderScore(
                    name=result.name,
                    score=0.0,
                    reason=result.error,
                ),
            ]
        else:
            raise ValueError(f"Invalid result type: {type(result)}")

        return results
    else:
        raise ValueError(f"Invalid grader mode: {grader.mode}")


async def aevaluate_with_cases(
    grader: Grader,
    eval_cases: List[EvalCase],
    parser: EvalCaseParser | Callable | None = None,
    **kwargs: Any,
) -> List[List[GraderScore]]:
    """Evaluate multiple evaluation cases using the specified grader concurrently.

    This is the main entry point to evaluate multiple eval cases using a grader. It
    processes all eval cases concurrently using asyncio.gather for improved performance.
    Each eval case is processed independently and the results are collected into a list
    of lists, where each inner list corresponds to the scores for one eval case.

    Args:
        grader: The grader instance to use for evaluation.
        eval_cases: A list of evaluation cases to evaluate. Each EvalCase consists of:
                   - input: A dictionary containing shared data for all samples
                   - outputs: A list of dictionaries, each representing an individual sample to evaluate
        parser: Optional parser to transform eval cases before evaluation. This allows for
                mapping field names between the data structure and what the grader expects.
                Can be:
                   1. An EvalCaseParser instance with field mappings
                   2. A callable function that takes an EvalCase and returns a transformed EvalCase
                   3. None, in which case the eval cases are used as-is
                Defaults to None.
        **kwargs: Additional keyword arguments passed to each grader evaluation.

    Returns:
        A list of lists of GraderScore objects. The outer list corresponds to each input
        eval case, and each inner list contains the scores for the corresponding eval case's
        output samples.

    Example:
        >>> from rm_gallery.core.schema.data import EvalCase
        >>> from rm_gallery.core.grader.base import LLMGrader
        >>>
        >>> # Create eval cases
        >>> eval_cases = [
        ...     EvalCase(
        ...         input={"query": "What is the capital of France?"},
        ...         outputs=[
        ...             {"answer": "Paris"},
        ...             {"answer": "London"}
        ...         ]
        ...     ),
        ...     EvalCase(
        ...         input={"query": "What is 2+2?"},
        ...         outputs=[
        ...             {"answer": "4"},
        ...             {"answer": "5"}
        ...         ]
        ...     )
        ... ]
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
        ...     model={"model": "qwen-plus"}
        ... )
        >>>
        >>> # Evaluate
        >>> results = await aevaluate_with_cases(grader, eval_cases)
        >>> print(results)
    """
    coroutines: List = [
        aevaluate_with_case(
            grader=grader,
            eval_case=_eval_case,
            parser=parser,
            **kwargs,
        )
        for _eval_case in eval_cases
    ]
    results = await asyncio.gather(*coroutines)
    return list(results)  # type: ignore
