"""
RM Gallery Grader Judge Integration

This module integrates RM Gallery's Graders into astuner's judge system.
It provides a way to evaluate workflow outputs using pre-defined RM Gallery graders.

Key Features:
- Support for both pointwise and listwise evaluation modes
- Flexible scoring based on RM Gallery graders
- Seamless integration with astuner's workflow system
- Support for mapper to handle data format conversion using parse_data_with_mapper

Example Configuration:
    task_judge:
      judge_type: customized_protocol
      judge_protocol: astuner.task_judge.rm_grader_judge->RMGraderJudge
      grader:
        class_name: "LLMGrader"
        module_path: "rm_gallery.core.graders.llm_grader"
        kwargs:
          name: "helpfulness_grader"
          mode: "pointwise"
          description: "Evaluates helpfulness of answers using LLM as a judge"
          model:
            model: "qwen-plus"
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
          template:
            messages:
              - role: "system"
                content: "You are an expert evaluator of AI assistant responses..."
              - role: "user"
                content: "Question: {query}\\nAnswer: {answer}\\n\\nPlease rate the helpfulness..."
          rubrics: "Consider factors like accuracy, completeness, and clarity..."
      # Optional mapper to convert data fields
      mapper:
        # Maps task.main_query to 'query' parameter
        query: "task.main_query"
        # Maps workflow_output.metadata.final_answer to 'answer' parameter
        answer: "workflow_output.metadata.final_answer"
"""

import asyncio
from typing import Any, Dict, List, Union

from loguru import logger

from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderError, GraderMode

# pylint: disable=ungrouped-imports, wrong-import-order
# from astuner.schema.task import Task, WorkflowOutput
# from astuner.task_judge.judge_base import JudgeBase
from rm_gallery.core.utils.instance import init_instance_by_config
from rm_gallery.core.utils.mapping import parse_data_with_mapper

Task = dict
WorkflowOutput = dict


# to inherit from JudgeBase
class RMGraderJudge:
    """
    A judge that uses RM Gallery's Graders to evaluate workflow outputs.

    This judge allows using pre-defined RM Gallery graders to score workflow outputs.
    It supports data mapping through the mapper configuration to transform task and
    workflow output data into the format expected by the grader.

    Workflow:
    1. Initialize with configuration for a specific grader
    2. Evaluate each workflow output using the grader

    Attributes:
        grader (BaseGrader): The RM Gallery grader instance used for evaluation
        mapper (Dict[str, str]): Mapping configuration to transform input data fields

    Example Config (in YAML):
        task_judge:
          judge_type: customized_protocol
          judge_protocol: astuner.task_judge.rm_grader_judge->RMGraderJudge
          grader:
            class_name: "LLMGrader"
            module_path: "rm_gallery.core.graders.llm_grader"
            kwargs:
              name: "helpfulness_grader"
              mode: "pointwise"
              description: "Evaluates helpfulness of answers using LLM as a judge"
              model:
                model: "qwen-plus"
                base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
              template:
                messages:
                  - role: "system"
                    content: "You are an expert evaluator of AI assistant responses..."
                  - role: "user"
                    content: "Question: {query}\\nAnswer: {answer}\\n\\nPlease rate the helpfulness..."
              rubrics: "Consider factors like accuracy, completeness, and clarity..."
          # Optional mapper to convert data fields
          mapper:
            # Maps task.main_query to 'query' parameter
            query: "task.main_query"
            # Maps workflow_output.metadata.final_answer to 'answer' parameter
            answer: "workflow_output.metadata.final_answer"
    """

    def __init__(self, config):
        """Initialize the RMGraderJudge.

        Args:
            config: Configuration object containing grader configuration and optional mapper.
                   The config should contain:
                   - config.astuner.task_judge.grader: Configuration for the RM Gallery grader
                   - config.astuner.task_judge.mapper (optional): Data mapping configuration
        """

        self.config = config

        # Initialize the grader from config
        self.grader: BaseGrader = init_instance_by_config(
            config["astuner"]["task_judge"]["grader"],  # config.astuner.task_judge.grader
            accept_type=BaseGrader,
        )

        # Get mapper configuration if available
        self.mapper = config["astuner"]["task_judge"].get("mapper", {})  # config.astuner.task_judge

        logger.info(
            f"RMGraderJudge initialized with grader name={self.grader.name}, " f"mode={self.grader.mode.value}",
        )

    def _prepare_eval_params(
        self,
        task: Task,
        workflow_output: Union[WorkflowOutput, List[WorkflowOutput]],
    ) -> Dict[str, Any]:
        """
        Prepare evaluation parameters by applying mapper configuration.

        Creates a data dictionary with 'task' and 'workflow_output' keys and applies
        the mapper configuration using RM-Gallery's parse_data_with_mapper utility.

        Args:
            task: The task being evaluated
            workflow_output: Single output or list of outputs

        Returns:
            Dictionary of evaluation parameters transformed by the mapper
        """
        # Create a combined data dictionary with task and workflow output informationinformation
        data = {"task": task, "workflow_output": workflow_output}

        # Apply mapper using parse_data_with_mapper
        mapped_data = parse_data_with_mapper(data, self.mapper)
        return mapped_data

    async def _async_compute_reward(
        self,
        task: Task,
        workflow_output: Union[WorkflowOutput, List[WorkflowOutput]],
    ):
        """
        Asynchronously compute reward using the RM Gallery grader.

        Args:
            task: The task being evaluated
            workflow_output: Single output for pointwise, or list of outputs for listwise

        Returns:
            For pointwise: tuple (raw_reward, is_success)
            For listwise: tuple (rank_list, is_success)

        Note:
            - For pointwise evaluation, raw_reward is a float score and is_success indicates
              if the score exceeds the threshold (0.5 by default)
            - For listwise evaluation, raw_reward is a list of ranks and is_success is always True
        """
        try:
            # Prepare evaluation parameters using mapper
            eval_params = self._prepare_eval_params(task, workflow_output)

            # Perform evaluation using the grader
            result = await self.grader.aevaluate(**eval_params)

            if isinstance(result, GraderError):
                if self.grader.mode == GraderMode.POINTWISE:
                    return (0.0, False)  # Default for pointwise
                else:
                    # For listwise, return zeros for each output
                    count = len(workflow_output) if isinstance(workflow_output, list) else 1
                    return ([0.0] * count, False)

            # Process result based on grader mode
            if self.grader.mode == GraderMode.POINTWISE:
                # For pointwise, return score and success flag
                raw_reward = getattr(result, "score", 0.0)
                is_success = raw_reward > 0.5  # Simple threshold for success
                return (raw_reward, is_success)
            else:
                # For listwise, return rank
                raw_reward = getattr(result, "rank", [])
                return (raw_reward, True)

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            if self.grader.mode == GraderMode.POINTWISE:
                return (0.0, False)  # Default for pointwise
            else:
                # For listwise, return zeros for each output
                count = len(workflow_output) if isinstance(workflow_output, list) else 1
                return ([0.0] * count, False)

    def compute_reward(self, task: Task, workflow_output: WorkflowOutput) -> tuple:
        """
        Compute reward for a workflow output (synchronous wrapper).

        This is the main interface called by astuner's workflow system.

        Args:
            task: The task being evaluated
            workflow_output: The output to evaluate

        Returns:
            tuple: (raw_reward, is_success)
                - For pointwise: (score: float, is_success: bool)
                - For listwise: (rank: list, is_success: bool)

        Raises:
            RuntimeError: If called from an async context without nest_asyncio installed
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context
            # We need to use nest_asyncio or raise an error
            try:
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(self._async_compute_reward(task, workflow_output))
            except ImportError as e:
                raise RuntimeError(
                    "compute_reward() was called from an async context. "
                    "Please use 'await judge._async_compute_reward(task, output)' instead, "
                    "or install nest_asyncio: pip install nest_asyncio",
                ) from e
        except RuntimeError:
            # No event loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_compute_reward(task, workflow_output))
            finally:
                loop.close()


if __name__ == "__main__":
    # Example usage of RMGraderJudge
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    # os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

    def example_usage():
        """
        Example demonstrating how to use RMGraderJudge

        This example shows:
        1. How to create a configuration for RMGraderJudge
        2. How to initialize the judge
        3. How to prepare mock task and workflow output data
        4. How to perform evaluation
        """

        # Example configuration for a pointwise LLM-based grader
        config = {
            "astuner": {
                "task_judge": {
                    "grader": {
                        "class_name": "HelpfulnessGrader",
                        "module_path": "rm_gallery.core.graders.alignment.helpfulness.helpfulness",
                        "kwargs": {
                            "model": {
                                "model": "qwen-plus",
                            },
                        },
                    },
                    "mapper": {
                        "query": "task.main_query",
                        "response": "workflow_output.metadata.final_answer",
                    },
                },
            },
        }

        try:
            # Initialize the judge
            judge = RMGraderJudge(config)
            print(f"Initialized RMGraderJudge with grader: {judge.grader.name}")

            # Create mock task and workflow output data
            task = {
                "main_query": "What is the capital of France?",
                "task_id": "example_task_001",
                "metadata": {},
            }
            workflow_output = {"metadata": {"final_answer": "The capital of France is Paris."}}

            # Perform evaluation
            print("\nPerforming evaluation...")
            result = judge.compute_reward(task, workflow_output)
            print(f"Evaluation result: {result}")

        except Exception as e:
            print(f"Error in example usage: {e}")

    # Run the example
    print("Running RMGraderJudge example...")
    example_usage()
    print("Example completed.")
