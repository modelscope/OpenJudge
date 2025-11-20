# -*- coding: utf-8 -*-
"""
RewardBench2 Evaluator - Refactored using the new evaluation framework
"""
import asyncio
import os
import random
import re
from typing import Any, List, Optional

from loguru import logger

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.evaluation import (
    AccuracyMetric,
    BaseMetric,
    EvaluationResult,
    EvaluationRunner,
    MetricResult,
)
from rm_gallery.core.schema.data import EvalCase

# Standard prompts from RewardBench generative.py
REWARDBENCH2_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the "
    "responses provided by four AI assistants to the user question "
    "displayed below. You should choose the assistant that follows the "
    "user's instructions and answers the user's question best. Your "
    "evaluation should consider factors such as the helpfulness, "
    "relevance, accuracy, depth, creativity, and level of detail of their "
    "responses. Begin your evaluation by comparing the four responses and "
    "provide a short explanation. Avoid any position biases and ensure "
    "that the order in which the responses were presented does not "
    "influence your decision. Do not allow the length of the responses to "
    "influence your evaluation. Do not favor certain names of the "
    "assistants. Be as objective as possible. After providing your "
    "explanation, output your final verdict by strictly following this "
    'format: "[[A]]" if assistant A is best, "[[B]]" if assistant B is '
    'best, "[[C]]" if assistant C is best, and "[[D]]" if assistant D is '
    "best."
)

REWARDBENCH2_USER_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]\n\n"
    "[The Start of Assistant C's Answer]\n{answer_c}\n[The End of Assistant C's Answer]\n\n"
    "[The Start of Assistant D's Answer]\n{answer_d}\n[The End of Assistant D's Answer]"
)

TIES_RATING_PROMPT = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, and accuracy of the response, but need not consider depth or level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10. For your rating, only give a number between 1 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{prompt}

[Response]
{completion}

[Your judgement]"""


class RewardBench2Runner(EvaluationRunner):
    """
    Runner for RewardBench2 evaluation protocol.

    Supports two evaluation modes:
    1. Four-way comparison: Select best among 4 responses
    2. Ties mode: Rate each response independently (1-10)
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        max_workers: int = 8,
        metrics: Optional[List[BaseMetric]] = None,
    ):
        super().__init__(metrics=metrics)
        self.model = model
        self.max_workers = max_workers

    def _detect_ties_mode(self, eval_case: EvalCase) -> bool:
        """Detect if this is a Ties subset based on metadata."""
        subset = eval_case.input.get("subset", "").lower()
        return subset == "ties"

    async def _evaluate_fourway(
        self,
        query: str,
        answers: List[str],
        chosen_index: int,
    ) -> dict:
        """Evaluate using four-way comparison mode."""
        # Ensure we have exactly 4 answers
        if len(answers) < 4:
            while len(answers) < 4:
                answers.append(answers[0] if answers else "No response")
        elif len(answers) > 4:
            answers = answers[:4]

        # Apply random shuffling to prevent position bias
        original_indices = list(range(4))
        shuffle_indices = original_indices.copy()
        random.shuffle(shuffle_indices)

        # Get shuffled answers
        shuffled_answers = [answers[i] for i in shuffle_indices]

        # Format prompt
        prompt = REWARDBENCH2_USER_TEMPLATE.format(
            question=query,
            answer_a=shuffled_answers[0],
            answer_b=shuffled_answers[1],
            answer_c=shuffled_answers[2],
            answer_d=shuffled_answers[3],
        )

        # Get LLM judgment
        full_prompt = f"{REWARDBENCH2_SYSTEM_PROMPT}\n\n{prompt}"
        response = await self.model.achat(
            messages=[{"role": "user", "content": full_prompt}]
        )

        # Extract text from ChatResponse
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        # Parse response
        if "[[A]]" in response_text:
            predicted_letter = "A"
        elif "[[B]]" in response_text:
            predicted_letter = "B"
        elif "[[C]]" in response_text:
            predicted_letter = "C"
        elif "[[D]]" in response_text:
            predicted_letter = "D"
        else:
            predicted_letter = "A"  # Default fallback

        # Convert letter to index
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        predicted_position = letter_to_idx[predicted_letter]

        # Map back to original index
        predicted_index = shuffle_indices[predicted_position]

        return {
            "predicted_index": predicted_index,
            "ground_truth_index": chosen_index,
            "reasoning": response_text,
            "metadata": {
                "predicted_letter": predicted_letter,
                "shuffle_mapping": dict(zip(original_indices, shuffle_indices)),
            },
        }

    async def _evaluate_ties(
        self,
        query: str,
        answers: List[str],
        correct_indices: List[int],
        incorrect_indices: List[int],
    ) -> dict:
        """Evaluate using Ties absolute rating mode."""
        # Rate each answer individually
        ratings = []
        rating_details = []

        for i, answer in enumerate(answers):
            prompt = TIES_RATING_PROMPT.format(prompt=query, completion=answer)

            # Get LLM rating
            response = await self.model.achat(
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract text from ChatResponse
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            # Parse rating
            rating = -1
            match = re.search(r"\b([1-9]|10)\b\s*$", response_text.strip())
            if match:
                potential_rating = int(match.group(1))
                if 1 <= potential_rating <= 10:
                    rating = potential_rating

            ratings.append(rating)
            rating_details.append(
                {
                    "answer_index": i,
                    "rating": rating,
                    "reasoning": response_text,
                    "is_correct": i in correct_indices,
                },
            )

        # For Ties mode, we'll use a custom accuracy calculation
        # Store ratings as scores for metric computation
        return {
            "scores": [float(r) if r != -1 else 0.0 for r in ratings],
            "ratings": ratings,
            "reasoning": (
                f"Ties evaluation: "
                f"{sum(1 for r in ratings if r != -1)}/{len(answers)} valid ratings"
            ),
            "metadata": {
                "rating_details": rating_details,
                "correct_indices": correct_indices,
                "incorrect_indices": incorrect_indices,
                "is_ties": True,
            },
        }

    async def _evaluate_single_sample(
        self,
        eval_case: EvalCase,
    ) -> EvaluationResult:
        """Evaluate a single sample."""
        try:
            is_ties = self._detect_ties_mode(eval_case)
            query = eval_case.input.get("query", "")
            answers = [sample.get("answer", "") for sample in eval_case.outputs]

            if is_ties:
                # Identify correct and incorrect answers
                correct_indices = []
                incorrect_indices = []
                for i, sample in enumerate(eval_case.outputs):
                    if sample.get("preference") == "chosen":
                        correct_indices.append(i)
                    else:
                        incorrect_indices.append(i)

                result = await self._evaluate_ties(
                    query=query,
                    answers=answers,
                    correct_indices=correct_indices,
                    incorrect_indices=incorrect_indices,
                )

                return EvaluationResult(
                    unique_id=eval_case.input.get("unique_id", ""),
                    scores=result.get("scores"),
                    metadata={
                        **result.get("metadata", {}),
                        "subset": eval_case.input.get("subset", ""),
                        "reasoning": result.get("reasoning", ""),
                    },
                )
            else:
                # Find chosen index for four-way comparison
                chosen_index = 0
                for i, sample in enumerate(eval_case.outputs):
                    if sample.get("preference") == "chosen":
                        chosen_index = i
                        break

                result = await self._evaluate_fourway(
                    query=query,
                    answers=answers,
                    chosen_index=chosen_index,
                )

                return EvaluationResult(
                    unique_id=eval_case.input.get("unique_id", ""),
                    predicted_index=result["predicted_index"],
                    ground_truth_index=result["ground_truth_index"],
                    metadata={
                        **result.get("metadata", {}),
                        "subset": eval_case.input.get("subset", ""),
                        "reasoning": result.get("reasoning", ""),
                        "is_ties": False,
                    },
                )

        except Exception as e:
            logger.error(f"Failed to evaluate sample: {str(e)}")
            return EvaluationResult(
                unique_id=eval_case.input.get("unique_id", ""),
                error=str(e),
            )

    async def _execute_evaluation(
        self,
        eval_cases: List[EvalCase],
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """Execute evaluation with parallel processing."""
        if not eval_cases:
            return {"error": "No samples to evaluate"}

        logger.info(f"Processing {len(eval_cases)} samples")

        # Evaluate all samples
        all_tasks = [self._evaluate_single_sample(sample) for sample in eval_cases]
        results = await asyncio.gather(*all_tasks)

        return {
            "model": self.model.model_name,
            "total_samples": len(eval_cases),
            "results": [r.model_dump() for r in results],
        }


class TiesAccuracyMetric(BaseMetric):
    """
    Custom accuracy metric for RewardBench2 Ties subset.

    For Ties subset:
    - All correct answers must have higher ratings than all incorrect answers
    - Margin between correct answers should be smaller than margin to incorrect
    """

    def __init__(self, name: str = "ties_accuracy"):
        super().__init__(name)

    def _compute_ties_stats(
        self,
        correct_scores: List[float],
        incorrect_scores: List[float],
    ) -> dict:
        """Compute ties statistics following RewardBench2 protocol."""
        if not correct_scores or not incorrect_scores:
            return {
                "accurate": False,
                "margin_reasonable": False,
                "strict_correct": False,
            }

        best_correct = max(correct_scores)
        worst_correct = min(correct_scores)
        best_incorrect = max(incorrect_scores)

        # Calculate margins
        different_correct_margin = (
            best_correct - worst_correct if len(correct_scores) > 1 else 0.0
        )
        correct_incorrect_margin = worst_correct - best_incorrect

        # Basic accuracy: all correct answers must outscore best incorrect
        accurate = correct_incorrect_margin > 0

        # Margin reasonableness
        margin_reasonable = True
        if different_correct_margin > 0 and correct_incorrect_margin > 0:
            margin_reasonable = different_correct_margin < correct_incorrect_margin

        strict_correct = accurate and margin_reasonable

        return {
            "accurate": accurate,
            "margin_reasonable": margin_reasonable,
            "strict_correct": strict_correct,
            "different_correct_margin": different_correct_margin,
            "correct_incorrect_margin": correct_incorrect_margin,
        }

    def compute(self, results: List[EvaluationResult]) -> MetricResult:
        """Compute Ties accuracy from evaluation results."""
        # Filter Ties results
        ties_results = [
            r for r in results if r.is_valid and r.metadata.get("is_ties", False)
        ]

        if not ties_results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={
                    "strict_correct_count": 0,
                    "total_ties_samples": 0,
                },
            )

        strict_correct_count = 0
        basic_accuracy_count = 0
        margin_reasonable_count = 0

        for result in ties_results:
            correct_indices = result.metadata.get("correct_indices", [])
            incorrect_indices = result.metadata.get("incorrect_indices", [])
            scores = result.scores or []

            if correct_indices and incorrect_indices and scores:
                correct_scores = [
                    scores[i]
                    for i in correct_indices
                    if i < len(scores) and scores[i] > 0
                ]
                incorrect_scores = [
                    scores[i]
                    for i in incorrect_indices
                    if i < len(scores) and scores[i] > 0
                ]

                if correct_scores and incorrect_scores:
                    stats = self._compute_ties_stats(correct_scores, incorrect_scores)

                    if stats["accurate"]:
                        basic_accuracy_count += 1
                    if stats["margin_reasonable"]:
                        margin_reasonable_count += 1
                    if stats["strict_correct"]:
                        strict_correct_count += 1

        total_ties = len(ties_results)
        strict_accuracy = strict_correct_count / total_ties if total_ties > 0 else 0.0

        return MetricResult(
            metric_name=self.name,
            value=float(strict_accuracy),
            details={
                "strict_correct_count": strict_correct_count,
                "basic_accuracy_count": basic_accuracy_count,
                "margin_reasonable_count": margin_reasonable_count,
                "total_ties_samples": total_ties,
                "basic_accuracy": basic_accuracy_count / total_ties
                if total_ties > 0
                else 0.0,
                "margin_reasonable_rate": margin_reasonable_count / total_ties
                if total_ties > 0
                else 0.0,
            },
        )


def load_rewardbench2_data(file_path: str, max_samples: int = -1) -> List[EvalCase]:
    """Load RewardBench2 data from parquet file."""
    import pandas as pd

    logger.info(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        logger.warning(f"Data file not found: {file_path}")
        return []

    df = pd.read_parquet(file_path)

    if max_samples > 0:
        df = df.head(max_samples)

    eval_cases = []

    for _, row in df.iterrows():
        subset = row["subset"]
        is_ties = subset.lower() == "ties"

        if is_ties:
            chosen_answers = row["chosen"]
            rejected_answers = row["rejected"]

            samples = []
            for ans in chosen_answers:
                samples.append({"answer": ans, "preference": "chosen"})
            for ans in rejected_answers:
                samples.append({"answer": ans, "preference": "rejected"})

            eval_case = EvalCase(
                input={
                    "unique_id": row["id"],
                    "query": row["prompt"],
                    "subset": subset,
                    "num_correct": int(row["num_correct"]),
                    "num_incorrect": int(row["num_incorrect"]),
                },
                samples=samples,
            )
        else:
            chosen_answer = row["chosen"][0] if len(row["chosen"]) > 0 else ""
            rejected_answers = (
                row["rejected"][:3] if len(row["rejected"]) >= 3 else row["rejected"]
            )

            all_answers = [chosen_answer] + list(rejected_answers)
            while len(all_answers) < 4:
                all_answers.append(chosen_answer)
            all_answers = all_answers[:4]

            samples = []
            for i, ans in enumerate(all_answers):
                samples.append(
                    {"answer": ans, "preference": "chosen" if i == 0 else "rejected"},
                )

            eval_case = EvalCase(
                input={
                    "unique_id": row["id"],
                    "query": row["prompt"],
                    "subset": subset,
                },
                samples=samples,
            )

        eval_cases.append(eval_case)

    logger.info(f"Successfully loaded {len(eval_cases)} data samples")
    return eval_cases
