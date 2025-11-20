# -*- coding: utf-8 -*-
"""
AutoRubrics - Dual Mode Rubric Generation System

This module provides an automated rubric generation system with two modes:
1. all_samples Mode: Generate rubrics for each sample independently with iterative refinement
2. smart_sampling Mode: Use MCR²-based selection and aggregation for optimal rubric sets

Key Features:
- Iterative rubric generation with convergence detection
- MCR² (Maximal Coding Rate Reduction) for information-theoretic selection
- Optional categorization-based aggregation for deduplication
- Async/concurrent processing with progress tracking
- Comprehensive statistics and history tracking
"""

import asyncio
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from rm_gallery.core.grader.base import GraderMode
from rm_gallery.core.grader.rubric.categorizer import LLMRubricCategorizer
from rm_gallery.core.grader.rubric.generator import QuerySpecificRubricGenerator
from rm_gallery.core.grader.rubric.mcr_selector import SuperFastAdaptiveMCR2
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.base import BaseRunner
from rm_gallery.core.schema.data import DataSample, DataSampleParser
from rm_gallery.core.schema.template import LanguageEnum


class SamplingMode(str, Enum):
    """Rubric generation sampling modes.

    Attributes:
        ALL_SAMPLES: Generate rubrics for each sample independently with iterative refinement
        SMART_SAMPLING: Use MCR²-based selection for optimal rubric subsets across batches
    """

    ALL_SAMPLES = "all_samples"
    SMART_SAMPLING = "smart_sampling"


class AggregationMode(str, Enum):
    """Final rubric aggregation strategies.

    Attributes:
        KEEP_ALL: Keep all generated rubrics without any post-processing
        CATEGORIZE: Use LLM-based categorization to merge semantically similar rubrics
    """

    KEEP_ALL = "keep_all"
    MERGE_SIMILAR = "merge_similar"


class AutoRubricsConfig(BaseModel):
    """Configuration parameters for AutoRubrics runner.

    This class encapsulates all configuration parameters for both single and batch modes.
    Parameters are organized by functionality for clarity.

    Attributes:
        sampling_mode: Sampling strategy (ALL_SAMPLES or SMART_SAMPLING)
        grader_mode: Grader evaluation mode (POINTWISE or LISTWISE)
        language: Language for prompts (ZH or EN)

        # Generation parameters
        generate_number: Number of rubrics to generate per sample
        max_retries: Maximum LLM API retry attempts on failure
        max_epochs: Maximum iterative refinement epochs per sample

        # Pointwise evaluation parameters
        min_score: Minimum score value for pointwise mode
        max_score: Maximum score value for pointwise mode

        # Batch mode parameters
        batch_size: Number of data samples to process per batch iteration
        mcr_batch_size: Number of rubrics selected by MCR² per iteration
        min_increment_threshold: Minimum information gain to continue iteration
        patience: Consecutive low-increment iterations before early stopping
        max_iterations: Maximum batch iterations allowed
        max_total_rubrics: Maximum total rubrics to maintain in pool

        # Aggregation parameters
        aggregation_mode: Final rubric aggregation strategy
        merge_num_categories: Target number of categories when using MERGE_SIMILAR mode
    """

    # Core configuration
    sampling_mode: SamplingMode = Field(
        default=SamplingMode.ALL_SAMPLES,
        description="Sampling mode: all_samples or smart_sampling",
    )
    grader_mode: GraderMode = Field(
        default=GraderMode.POINTWISE,
        description="Grader mode (POINTWISE or LISTWISE)",
    )
    language: LanguageEnum = Field(
        default=LanguageEnum.EN,
        description="Language for prompts (ZH or EN)",
    )

    # Generation parameters
    generate_number: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of rubrics per sample",
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum retry attempts for LLM calls",
    )
    max_epochs: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum iteration epochs per sample",
    )

    # Pointwise specific
    min_score: int = Field(
        default=0,
        description="Minimum score for pointwise mode",
    )
    max_score: int = Field(
        default=1,
        ge=1,
        description="Maximum score for pointwise mode",
    )

    # Batch processing parameters
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of data samples per batch iteration",
    )

    # MCR² parameters
    mcr_batch_size: int = Field(
        default=10,
        ge=1,
        description="Rubrics selected by MCR² per iteration",
    )
    min_increment_threshold: float = Field(
        default=0.002,
        ge=0.0,
        description="Minimum information gain threshold for convergence",
    )
    patience: int = Field(
        default=2,
        ge=1,
        description="Consecutive low increments before early stopping",
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        description="Maximum batch iterations",
    )
    max_total_rubrics: int = Field(
        default=200,
        ge=1,
        description="Maximum total rubrics in pool",
    )

    # Aggregation parameters
    aggregation_mode: AggregationMode = Field(
        default=AggregationMode.KEEP_ALL,
        description="Final aggregation strategy",
    )
    merge_num_categories: int = Field(
        default=2,
        description="Number of categories for CATEGORIZE mode",
    )


class AutoRubrics(BaseRunner):
    """
    Dual-Mode AutoRubrics Generator

    Supports two generation modes:
    1. All samples Mode: Generate rubrics for each sample independently
    2. Smart sampling Mode: Use MCR-based selection and aggregation for optimal rubric sets
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        parser: DataSampleParser | Callable | None = None,
        config: AutoRubricsConfig | None = None,
    ):
        """
        Initialize AutoRubrics

        Args:
            llm: Language model
            config: AutoRubrics configuration
        """
        self.model = model
        self.parser = parser
        self.config = config or AutoRubricsConfig()

        # Create query-specific generator
        self.generator = QuerySpecificRubricGenerator(
            model=model,
            grader_mode=self.config.grader_mode,
            generate_number=self.config.generate_number,
            max_retries=self.config.max_retries,
            max_epochs=self.config.max_epochs,
            min_score=self.config.min_score,
            max_score=self.config.max_score,
            language=self.config.language,
        )

        # Initialize MCR selector only for batch mode
        self.mcr_selector = None
        if self.config.sampling_mode == SamplingMode.SMART_SAMPLING:
            self.mcr_selector = SuperFastAdaptiveMCR2(
                batch_size=self.config.mcr_batch_size,
            )

        # Initialize categorizer for merge mode
        self.categorizer = None
        if self.config.aggregation_mode == AggregationMode.MERGE_SIMILAR:
            self.categorizer = LLMRubricCategorizer(
                num_categories=self.config.merge_num_categories,
                model=model,
                language=self.config.language.value,
            )

        # State tracking (for batch mode)
        self.all_rubrics = []
        self.iteration_history = []
        self.coding_rates = []
        self.low_increment_count = 0
        self.current_sample_index = 0

        logger.info(
            f"AutoRubrics initialized: sampling_mode={self.config.sampling_mode.value}, "
            f"grader_mode={self.config.grader_mode.value}, "
            f"language={self.config.language.value}, "
            f"aggregation_mode={self.config.aggregation_mode.value}",
        )

    async def __call__(self, data_samples: List[DataSample]) -> Dict[str, Any]:
        """
        Run AutoRubrics pipeline based on generation mode

        Args:
            data_samples: All data samples to process

        Returns:
            Dict with final rubrics, statistics, and processing details
        """
        if self.parser is not None:
            data_samples = [self.parser(data_sample) for data_sample in data_samples]

        logger.info(
            f"Starting AutoRubrics ({self.config.sampling_mode.value} mode) "
            f"with {len(data_samples)} data samples",
        )

        if self.config.sampling_mode == SamplingMode.ALL_SAMPLES:
            return await self._run_all_samples_mode(data_samples)
        else:
            return await self._run_smart_sampling_mode(data_samples)

    async def _run_all_samples_mode(
        self,
        data_samples: List[DataSample],
    ) -> Dict[str, Any]:
        """
        Single mode: Generate rubrics for each sample independently

        Args:
            data_samples: All data_samples to process

        Returns:
            Dict with rubrics for each sample and aggregated statistics
        """
        logger.info(
            f"Running single mode for {len(data_samples)} data_samples",
        )

        # Process all data_samples concurrently
        results = await self._process_samples_concurrently(
            data_samples,
            "Processing data samples",
        )

        # Extract rubrics and calculate stats
        all_rubrics, stats = self._extract_rubrics_and_stats(
            results,
            len(data_samples),
        )

        # Apply aggregation and generate final results
        final_rubrics, aggregation_info = await self._apply_aggregation(
            all_rubrics,
        )

        return self._build_final_results(
            sampling_mode="all_samples",
            total_data_samples=len(data_samples),
            all_rubrics=all_rubrics,
            final_rubrics=final_rubrics,
            aggregation_info=aggregation_info,
            stats=stats,
            extra_data={"sample_results": results},
        )

    async def _run_smart_sampling_mode(
        self,
        data_samples: List[DataSample],
    ) -> Dict[str, Any]:
        """
        Smart sampling mode: Use MCR-based selection and aggregation

        Args:
            data_samples: All data_samples to process

        Returns:
            Dict with optimal rubric subset and iteration history
        """
        logger.info(
            f"🔄 Running batch mode with MCR selection for {len(data_samples)} data samples",
        )

        iteration = 0
        while True:
            iteration += 1

            # Get next batch
            batch_samples = self._get_next_batch(data_samples)
            if batch_samples is None:
                # Reset for new cycle
                self.current_sample_index = 0
                batch_samples = self._get_next_batch(data_samples)
                if batch_samples is None:
                    logger.error("Failed to get batch, stopping")
                    break

            # Generate rubrics for batch
            new_rubrics, gen_stats = await self._generate_batch(batch_samples)
            if not new_rubrics:
                continue

            # MCR evaluation
            mcr_results = self._evaluate_mcr(new_rubrics)

            # Check stopping conditions
            should_continue, _ = self._should_continue(
                mcr_results,
                iteration,
                gen_stats,
            )

            # Update state
            self.all_rubrics = mcr_results["selected_texts"]
            self.coding_rates.append(mcr_results["final_coding_rate"])
            self.iteration_history.append(
                {
                    "iteration": iteration,
                    "batch_start": self.current_sample_index - len(batch_samples),
                    "batch_end": self.current_sample_index - 1,
                    "batch_size": len(batch_samples),
                    "new_generated": len(new_rubrics),
                    "total_selected": len(self.all_rubrics),
                    "coding_rate": mcr_results["final_coding_rate"],
                    "increment": mcr_results["increment"],
                    "generation_stats": gen_stats,
                },
            )

            if not should_continue:
                break

        # Apply aggregation and generate final results
        final_rubrics, aggregation_info = await self._apply_aggregation(
            self.all_rubrics,
        )

        return self._build_final_results(
            sampling_mode="smart_sampling",
            total_data_samples=len(data_samples),
            all_rubrics=self.all_rubrics,
            final_rubrics=final_rubrics,
            aggregation_info=aggregation_info,
            stats={},  # No single stats for batch mode
            extra_data={
                "total_iterations": iteration,
                "iteration_history": self.iteration_history,
                "coding_rates": self.coding_rates,
                "final_coding_rate": self.coding_rates[-1]
                if self.coding_rates
                else 0.0,
            },
        )

    def _get_next_batch(
        self,
        all_data_samples: List[DataSample],
    ) -> Optional[List[DataSample]]:
        """Get next batch of data_samples for processing.

        Args:
            all_samples: Complete list of all data samples

        Returns:
            Next batch of data samples, or None if all data samples processed
        """
        if not all_data_samples:
            logger.warning("Empty sample list provided")
            return None

        start_idx = self.current_sample_index
        end_idx = min(
            start_idx + self.config.batch_size,
            len(all_data_samples),
        )

        if start_idx >= len(all_data_samples):
            return None

        batch = all_data_samples[start_idx:end_idx]
        self.current_sample_index = end_idx

        logger.info(
            f"Processing batch: data samples {start_idx}-{end_idx-1} "
            f"({len(batch)}/{len(all_data_samples)} data samples)",
        )
        return batch

    async def _generate_batch(
        self,
        batch_samples: List[DataSample],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Generate rubrics for a batch of data samples concurrently.

        Args:
            batch_samples: List of data samples to generate rubrics for

        Returns:
            Tuple of (valid_rubrics, generation_statistics)
        """
        if not batch_samples:
            logger.warning("Empty batch provided")
            return [], {"error": "Empty batch"}

        logger.info(
            f"Generating rubrics for {len(batch_samples)} data samples...",
        )

        # Process data samples concurrently
        results = await self._process_samples_concurrently(
            batch_samples,
            "Generating",
        )

        # Extract rubrics and stats
        all_rubrics, stats = self._extract_rubrics_and_stats(
            results,
            len(batch_samples),
        )

        logger.info(
            f"Generation completed: {stats['successful_samples']}/{len(batch_samples)} successful "
            f"({stats['success_rate']:.1f}%), {len(all_rubrics)} rubrics",
        )

        return all_rubrics, stats

    def _evaluate_mcr(self, new_rubrics: List[str]) -> Dict[str, Any]:
        """Evaluate information gain using MCR² algorithm.

        Uses Maximal Coding Rate Reduction to select the most informative rubric subset
        while minimizing redundancy.

        Args:
            new_rubrics: Newly generated rubrics to evaluate

        Returns:
            Dict containing:
                - selected_texts: Optimal rubric subset
                - final_coding_rate: Information content measure
                - increment: Information gain from previous iteration
        """
        combined = self.all_rubrics + new_rubrics
        if not combined:
            logger.warning("No rubrics to evaluate")
            return {
                "selected_texts": [],
                "final_coding_rate": 0.0,
                "increment": 0.0,
            }

        logger.info(
            f"📊 MCR² evaluation: {len(self.all_rubrics)} existing + "
            f"{len(new_rubrics)} new = {len(combined)} total",
        )

        # MCR selection
        mcr_config = {
            "batch_size": self.config.mcr_batch_size,
            "min_increment_threshold": self.config.min_increment_threshold,
            "patience": self.config.patience,
            "max_samples": min(self.config.max_total_rubrics, len(combined)),
        }

        results = self.mcr_selector.ultra_fast_adaptive_selection(
            combined,
            **mcr_config,
        )

        # Calculate increment
        previous_rate = self.coding_rates[-1] if self.coding_rates else 0.0
        current_rate = results["final_coding_rate"]
        increment = current_rate - previous_rate

        results["increment"] = increment
        logger.info(
            f"📈 MCR² results: {len(results['selected_texts'])} selected, "
            f"rate={current_rate:.6f}, increment={increment:.6f}",
        )

        return results

    def _should_continue(
        self,
        mcr_results: Dict[str, Any],
        iteration: int,
        gen_stats: Dict,
    ) -> Tuple[bool, str]:
        """Determine whether to continue iteration"""
        del gen_stats
        # Check information gain
        increment = mcr_results.get("increment", 0.0)

        if increment < self.config.min_increment_threshold:
            self.low_increment_count += 1
            logger.info(
                f"Low increment: {increment:.6f} < {self.config.min_increment_threshold:.6f} "
                f"(count: {self.low_increment_count}/{self.config.patience})",
            )

            if self.low_increment_count >= self.config.patience:
                return (
                    False,
                    f"Converged: {self.config.patience} consecutive low increments "
                    f"(last: {increment:.6f})",
                )
        else:
            if self.low_increment_count > 0:
                logger.info(
                    f"Increment recovered: {increment:.6f}, resetting counter",
                )
            self.low_increment_count = 0

        # Check rubric count
        if len(self.all_rubrics) >= self.config.max_total_rubrics:
            return (
                False,
                f"Max rubrics reached ({self.config.max_total_rubrics})",
            )

        # Check iteration count
        if iteration >= self.config.max_iterations:
            return (
                False,
                f"Max iterations reached ({self.config.max_iterations})",
            )

        return True, ""

    # ========== Common Helper Methods ==========

    async def _process_samples_concurrently(
        self,
        data_samples: List[DataSample],
        desc: str = "Processing",
    ) -> List[Dict[str, Any]]:
        """Process samples concurrently with progress tracking.

        Args:
            data_samples: List of data samples to process
            desc: Description for progress bar

        Returns:
            List of generation results
        """
        tasks = [self.generator.generate_iterative(sample) for sample in data_samples]

        results = []
        with tqdm(total=len(data_samples), desc=desc, unit="sample") as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Sample processing failed: {e}")
                    results.append(
                        {
                            "rubrics": [],
                            "rubric_valid": "False",
                            "rubric_epoch": "0",
                            "evaluation_result": {},
                            "error": str(e),
                        },
                    )
                    pbar.update(1)

        return results

    def _extract_rubrics_and_stats(
        self,
        results: List[Dict[str, Any]],
        total_data_samples: int,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Extract rubrics and calculate statistics from generation results.

        Args:
            results: List of generation results
            total_data_samples: Total number of data_samples processed

        Returns:
            Tuple of (all_rubrics, statistics)
        """
        all_rubrics = []
        successful_data_samples = 0
        failed_samples = 0

        for result in results:
            if result.get("rubric_valid") == "True":
                successful_data_samples += 1
                all_rubrics.extend(result.get("rubrics", []))
            else:
                failed_samples += 1

        success_rate = (
            (successful_data_samples / total_data_samples * 100)
            if total_data_samples
            else 0
        )

        stats = {
            "successful_samples": successful_data_samples,
            "failed_samples": failed_samples,
            "success_rate": success_rate,
            "total_rubrics": len(all_rubrics),
        }

        return all_rubrics, stats

    async def _apply_aggregation(
        self,
        rubrics: List[str],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Apply aggregation based on configuration.

        Args:
            rubrics: List of rubric strings to aggregate

        Returns:
            Tuple of (final_rubrics, aggregation_info)
        """
        if self.config.aggregation_mode == AggregationMode.MERGE_SIMILAR and rubrics:
            if not self.categorizer:
                logger.warning(
                    "Categorizer not initialized, returning original rubrics",
                )
                return rubrics, {"error": "Categorizer not initialized"}

            try:
                (
                    ready_to_use_list,
                    aggregation_info,
                ) = await self.categorizer.categorize_rubrics(rubrics)
                return ready_to_use_list, aggregation_info
            except Exception as e:
                logger.error(f"Categorization failed: {e}")
                return rubrics, {"error": str(e)}

        # No aggregation or empty rubrics
        return rubrics, {
            "num_categories": len(rubrics),
            "original_rubrics_count": len(rubrics),
            "categorization_successful": False,
        }

    def _build_final_results(
        self,
        sampling_mode: str,
        total_data_samples: int,
        all_rubrics: List[str],
        final_rubrics: List[str],
        aggregation_info: Dict[str, Any],
        stats: Dict[str, Any],
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build final results dictionary.

        Args:
            sampling_mode: "all_samples" or "smart_sampling"
            total_samples: Total number of data samples processed
            all_rubrics: All generated rubrics before aggregation
            final_rubrics: Final rubrics after aggregation
            aggregation_info: Aggregation metadata
            stats: Generation statistics
            extra_data: Additional mode-specific data

        Returns:
            Final results dictionary
        """
        results = {
            "sampling_mode": sampling_mode,
            "grader_mode": self.config.grader_mode.value,
            "aggregation_mode": self.config.aggregation_mode.value,
            "config": self.config.model_dump(),
            "total_samples": total_data_samples,
            "final_rubrics": final_rubrics,
            "final_rubric_count": len(final_rubrics),
            "all_rubrics": all_rubrics,
            "aggregation_info": aggregation_info,
            **stats,  # Include stats directly
        }

        # Add extra mode-specific data
        if extra_data:
            results.update(extra_data)

        return results

    @classmethod
    def create(
        cls,
        model: OpenAIChatModel,
        parser: DataSampleParser | Callable | None = None,
        # Core settings
        sampling_mode: SamplingMode | str = SamplingMode.ALL_SAMPLES,
        grader_mode: GraderMode | str = GraderMode.POINTWISE,
        language: LanguageEnum | str = LanguageEnum.ZH,
        # Generation parameters
        generate_number: int = 3,
        max_epochs: int = 3,
        max_retries: int = 5,
        # Score range (for pointwise)
        min_score: int = 0,
        max_score: int = 1,
        # Batch mode parameters (only used when sampling_mode=SMART_SAMPLING)
        batch_size: int = 10,
        mcr_batch_size: int = 10,
        max_iterations: int = 50,
        min_increment_threshold: float = 0.002,
        patience: int = 2,
        max_total_rubrics: int = 200,
        # Aggregation parameters
        aggregation_mode: AggregationMode | str = AggregationMode.KEEP_ALL,
        merge_num_categories: int = 5,
        **kwargs: Any,
    ) -> "AutoRubrics":
        """
        Create AutoRubrics instance with unified configuration

        Args:
            model: Language model
            parser: Optional data sample parser
            sampling_mode: SamplingMode enum or string ("all_samples" or "smart_sampling")
            grader_mode: GraderMode enum or string ("pointwise" or "listwise")
            language: LanguageEnum or string ("zh" or "en")

            # Generation parameters
            generate_number: Number of rubrics per sample
            max_epochs: Maximum iteration epochs per sample
            max_retries: Maximum LLM retry attempts

            # Score range (pointwise only)
            min_score: Minimum score value
            max_score: Maximum score value

            # Batch mode parameters (ignored in single mode)
            batch_size: Data samples per batch iteration
            mcr_batch_size: Rubrics selected by MCR² per iteration
            max_iterations: Maximum batch iterations
            min_increment_threshold: Minimum information gain threshold
            patience: Consecutive low increments before early stopping
            max_total_rubrics: Maximum total rubrics in pool

            # Aggregation
            aggregation_mode: AggregationMode enum or string ("keep_all" or "merge_similar")
            merge_num_categories: Number of categories when using MERGE_SIMILAR

        Returns:
            AutoRubrics instance
        """
        config = AutoRubricsConfig(
            sampling_mode=sampling_mode,
            grader_mode=grader_mode,
            language=language,
            generate_number=generate_number,
            max_epochs=max_epochs,
            max_retries=max_retries,
            min_score=min_score,
            max_score=max_score,
            batch_size=batch_size,
            mcr_batch_size=mcr_batch_size,
            max_iterations=max_iterations,
            min_increment_threshold=min_increment_threshold,
            patience=patience,
            max_total_rubrics=max_total_rubrics,
            aggregation_mode=aggregation_mode,
            merge_num_categories=merge_num_categories,
            **kwargs,
        )
        return cls(model=model, parser=parser, config=config)
