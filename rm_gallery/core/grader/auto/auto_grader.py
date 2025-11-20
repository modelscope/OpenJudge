# -*- coding: utf-8 -*-
from typing import Any, Callable, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.grader.auto.auto_rubrics import AutoRubrics, AutoRubricsConfig
from rm_gallery.core.grader.auto.prompts import EvaluationPromptTemplates
from rm_gallery.core.grader.base import GraderMode, LLMGrader
from rm_gallery.core.model import OpenAIChatModel
from rm_gallery.core.runner.base import BaseRunner
from rm_gallery.core.schema.data import EvalCase, EvalCaseParser
from rm_gallery.core.schema.template import ChatMessage, Template


class AutoGraderConfig(BaseModel):
    """Configuration for AutoGrader."""

    # Method selection
    method: str = Field(
        default="auto_rubrics",
        description="Method to generate rubrics",
    )

    # Method configuration
    method_config: Optional[Any] = Field(
        default=AutoRubricsConfig(),
        description="Configuration for the selected method",
    )

    # Grader configuration
    grader_name: str = Field(
        default="Auto Grader",
        description="Name for the generated grader",
    )

    # Optional custom prompts (if provided, will override template selection)
    custom_evaluation_prompt: Optional[str] = Field(
        default=None,
        description="Custom evaluation prompt (overrides template selection for evaluation)",
    )


class AutoGrader(BaseRunner):
    """AutoGrader with flexible rubric generation methods and prompt templates."""

    def __init__(
        self,
        model: OpenAIChatModel,
        parser: EvalCaseParser | Callable | None = None,
        config: AutoGraderConfig | None = None,
    ):
        """AutoGrader initialization.

        Args:
            model: OpenAI chat model for LLM operations
            parser: EvalCase parser
            config: AutoGrader configuration
        """
        self.model = model
        self.parser = parser
        self.config = config or AutoGraderConfig()

        # Initialize rubric generator based on method
        self.rubric_generator = self._create_rubric_generator()

        logger.info(
            f"AutoGrader initialized with method='{self.config.method}', "
            f"grader_mode={self.config.method_config.grader_mode.value}",
        )

    def _create_rubric_generator(self):
        """Create rubric generator based on method configuration."""
        if self.config.method == "auto_rubrics":
            return AutoRubrics(
                model=self.model,
                parser=self.parser,
                config=self.config.method_config,
            )
        else:
            raise ValueError(
                f"Unsupported method: {self.config.method}. Supported methods: 'auto_rubrics'.",
            )

    async def aevaluate_batch(
        self,
        eval_cases: List[EvalCase],
        *args: Any,
        **kwargs: Any,
    ) -> LLMGrader:
        """Generate rubrics and create LLMGrader.

        Args:
            eval_cases: List of eval cases to generate rubrics from

        Returns:
            LLMGrader instance with generated rubrics
        """
        # Generate rubrics using the selected method
        rubrics_result = await self.rubric_generator(eval_cases)

        # Extract the final rubrics from the result
        if isinstance(rubrics_result, dict) and "final_rubrics" in rubrics_result:
            rubrics = "\n".join(rubrics_result["final_rubrics"])
        else:
            rubrics = str(rubrics_result)

        logger.info(f"Rubrics: {rubrics}")

        template = self._create_evaluation_template()

        # Prepare kwargs for evaluation - these will be passed to template formatting
        eval_kwargs = {}
        if self.config.method_config.grader_mode == GraderMode.POINTWISE:
            eval_kwargs["min_score"] = self.config.method_config.min_score
            eval_kwargs["max_score"] = self.config.method_config.max_score
        elif self.config.method_config.grader_mode == GraderMode.LISTWISE:
            eval_kwargs["num_responses"] = len(eval_cases[0].outputs)

        return LLMGrader(
            name=self.config.grader_name,
            mode=self.config.method_config.grader_mode,
            language=self.config.method_config.language,
            template=template,
            model=self.model,
            rubrics=rubrics,
            **eval_kwargs,
        )

    def _create_evaluation_template(self) -> Template:
        """Create evaluation template based on language and grader mode.

        Returns:
            Template object for LLMGrader
        """
        # Use custom prompts if provided
        if self.config.custom_evaluation_prompt:
            return Template(
                messages={
                    self.config.method_config.language: [
                        ChatMessage(
                            role="user",
                            content=self.config.custom_evaluation_prompt,
                        ),
                    ],
                },
            )

        # Select template based on grader mode and return the multi-language template directly
        if self.config.method_config.grader_mode == GraderMode.POINTWISE:
            chat_template = EvaluationPromptTemplates.pointwise_evaluation(
                self.model,
            )
        elif self.config.method_config.grader_mode == GraderMode.LISTWISE:
            chat_template = EvaluationPromptTemplates.listwise_evaluation(
                self.model,
            )
        else:
            raise ValueError(
                f"Unsupported grader mode: {self.config.method_config.grader_mode}",
            )

        return chat_template.template

    @classmethod
    def create(
        cls,
        model: OpenAIChatModel,
        parser: EvalCaseParser | Callable | None = None,
        # Method configuration
        method: str = "auto_rubrics",
        method_config: Optional[Any] = None,
        # Grader configuration
        grader_name: str = "Auto Grader",
        # Custom prompts (optional)
        custom_evaluation_prompt: Optional[str] = None,
        # Method-specific parameters (when method_config is None)
        **method_kwargs,
    ) -> "AutoGrader":
        """Create AutoGrader instance with flexible configuration.

        Args:
            model: OpenAI chat model
            parser: EvalCase parser
            method: Method to generate rubrics ('auto_rubrics' or 'checklist')
            method_config: Configuration object for the selected method
            grader_name: Name for the generated grader
            custom_evaluation_prompt: Custom evaluation prompt (overrides template selection for evaluation)
            **method_kwargs: Additional parameters for the method (when method_config is None)

        Returns:
            AutoGrader instance

        """
        # Create method config if needed and not provided
        if method_config is None and method_kwargs:
            if method == "auto_rubrics":
                method_config = AutoRubricsConfig(**method_kwargs)
            else:
                logger.warning(
                    f"Unknown method '{method}', using method_kwargs as config",
                )
                method_config = method_kwargs

        # Create AutoGrader config
        config = AutoGraderConfig(
            method=method,
            method_config=method_config,
            grader_name=grader_name,
            custom_evaluation_prompt=custom_evaluation_prompt,
        )

        return cls(
            model=model,
            parser=parser,
            config=config,
        )
