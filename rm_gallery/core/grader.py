import asyncio

# import inspect
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.data import DataSample, DataSampleParser
from rm_gallery.core.model.template import ChatTemplate, RequiredField, Template
from rm_gallery.core.utils.concurrency import ConcurrencyManager


class GraderMode(str, Enum):
    """Grader modes for grader functions.

    Attributes:
        POINTWISE: Pointwise grader mode.
        LISTWISE: Listwise grader mode.
    """

    POINTWISE = "pointwise"
    LISTWISE = "listwise"


class GraderResult(BaseModel):
    """Base class for grader results.

    This Pydantic model defines the structure for grader results,
    which include a reason and optional metadata.
    """

    reason: str = Field(default=..., description="The reason for the result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="The metadata of the grader result"
    )


class GraderScore(GraderResult):
    """Grader score result.

    Represents a numerical score assigned by a grader along with a reason.
    """

    score: float = Field(default=..., description="score")


class GraderRank(GraderResult):
    """Grader rank result.

    Represents a ranking of items assigned by a grader along with a reason.
    """

    rank: List[int] = Field(default=..., description="rank")


class GraderError(GraderResult):
    """Grader error result.

    Represents an error encountered during evaluation.
    """


class GraderInfo(BaseModel):
    """Grader info.

    Represents meta information about a grader.
    """

    name: str = Field(default=..., description="The name of the grader")
    mode: GraderMode = Field(default=..., description="The grader mode")
    description: str = Field(default=..., description="The description of the grader")
    required_fields: List[RequiredField] = Field(
        default_factory=list, description="The required fields for the grader"
    )


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
        return GraderInfo(
            name=self.name,
            mode=self.mode,
            description=self.description,
            required_fields=self.required_fields,
        )

    @abstractmethod
    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate method to be implemented by subclasses.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.
        """
        ...

    async def _safe_evaluate(self, **kwargs) -> GraderScore | GraderRank | GraderError:
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
                        None, lambda: self.evaluate(**kwargs)
                    )
                return result
            except Exception as e:
                error_msg = f"Error in {self.name} during evaluation: {str(e)}"
                logger.error(error_msg)
                return GraderError(reason=error_msg)

        # Use the concurrency manager to control execution
        return await concurrency_manager.run_with_concurrency_control(_evaluate_task())

    async def __call__(
        self, data_sample: DataSample, *args, **kwargs
    ) -> List[GraderScore]:
        """Evaluate based on the specified grader mode.

        Args:
            data_sample: The data sample to evaluate.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List of grader scores.

        Raises:
            ValueError: If the grader mode is invalid.
        """
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
                    _results.append(GraderScore(score=0.0, reason=result.reason))
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
                        params[key] = "\n".join(
                            [
                                f"Sample {i+1}: {sample[key]}"
                                for i, sample in enumerate(data_sample.samples)
                            ]
                        )

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
        chat (ChatTemplate): The chat template for the LLM.
        rubrics (str): The rubrics for the evaluation.
        kwargs (dict): The kwargs for the grader.
    """

    def __init__(
        self,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        template: Template | dict | None = None,
        model: Dict[str, Any] | None = None,
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
            template if isinstance(template, Template) else Template(**template)
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

    def reset(self, model: Dict[str, Any] | None = None, rubrics: str = "", **kwargs):
        """
        Reset the grader with new model and rubrics.
        """
        super().reset(**kwargs)
        self.model = model
        self.rubrics = rubrics
        if self.template is not None and self.model is not None:
            self.chat = ChatTemplate(template=self.template, model=self.model)
        else:
            raise ValueError("Chat template or model is not set")

    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using LLM.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            Evaluation result (score or rank).

        Raises:
            ValueError: If the grader mode is unsupported.
        """
        if self.mode == GraderMode.LISTWISE:
            chat_output = GraderRank
        else:
            chat_output = GraderScore

        # Check if chat is not None before calling it
        if self.chat is None:
            raise ValueError("Chat template is not set")
        params = {"rubrics": self.rubrics, **self.kwargs}
        params.update(kwargs)

        response = await self.chat(chat_output=chat_output, **params)
        if self.mode == GraderMode.LISTWISE:
            result = GraderRank(
                rank=response.metadata["rank"],  # type: ignore
                reason=response.metadata["reason"],  # type: ignore
            )
        elif self.mode == GraderMode.POINTWISE:
            result = GraderScore(
                score=response.metadata["score"],  # type: ignore
                reason=response.metadata["reason"],  # type: ignore
            )
        else:
            raise ValueError(f"Unsupported grader mode: {self.mode}")
        return result

    def to_dict(self) -> dict:
        """Convert the grader to a dictionary.

        Returns:
            A dictionary representation of the LLM grader.
        """
        return {
            "name": self.name,
            "mode": self.mode,
            "template": self.template.model_dump(),
            "model": {},
            "rubrics": self.rubrics,
            "description": self.description,
            "required_fields": [field.model_dump() for field in self.required_fields],
        }


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
            name, mode, description, required_fields=required_fields, **kwargs
        )
        self.func = func

    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using a function.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            Evaluation result (score or rank).

        Raises:
            TypeError: If result type doesn't match grader mode.
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
                    f"Expected GraderScore for pointwise mode, got {type(result)}"
                )
        elif self.mode == GraderMode.LISTWISE:
            if not isinstance(result, GraderRank):
                raise TypeError(
                    f"Expected GraderRank for listwise mode, got {type(result)}"
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


async def evaluate(
    grader: Callable,
    data_sample: DataSample | List[DataSample],
    parser: DataSampleParser | Callable | None = None,
    *args,
    **kwargs,
) -> List[GraderScore] | List[List[GraderScore]]:
    """Evaluate a data sample using a grader.

    Args:
        grader: The grader function to use.
        data_sample: The data sample to evaluate.
        parser: Data sample parser function.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        List of grader scores.

    Raises:
        ValueError: If grader function signature is invalid.
    """
    if isinstance(data_sample, list):
        corutines = [
            evaluate(
                grader=grader, data_sample=_data_sample, parser=parser, *args, **kwargs
            )
            for _data_sample in data_sample
        ]
        return await asyncio.gather(*corutines)

    # Check that function has at least one parameter and first parameter is data_sample
    # sig = inspect.signature(grader)
    # params = list(sig.parameters.keys())

    # if not params:
    #     raise ValueError(f"Function {grader.name} must have at least one parameter")

    # if "data_sample" not in params:
    #     raise ValueError(
    #         f"Function {grader.name} must have 'data_sample' as its first parameter"
    #     )

    return await grader(
        data_sample=parser(data_sample) if parser is not None else data_sample,
        *args,
        **kwargs,
    )
