"""
AutoRubrics Example

Test AutoRubrics batch mode with processed data.

Features:
1. Load processed data
2. Test batch mode (batch mode) - use MCR selection and aggregation
3. Save results to file
"""

import asyncio
import json
from pathlib import Path
from typing import List

from loguru import logger

from rm_gallery.core.data import DataSample
from rm_gallery.core.grader import GraderMode
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.auto_rubrics import AggregationMode, AutoRubrics


def load_samples(file_path: str, max_samples: int = None) -> List[DataSample]:
    """Load samples from file"""
    logger.info(f"Loading samples from {file_path}")

    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                sample = DataSample(**data)
                samples.append(sample)
            except Exception as e:
                logger.error(f"Failed to parse line {i+1}: {e}")
                continue

    logger.info(f"Successfully loaded {len(samples)} samples")
    return samples


async def test_batch_data(
    grader_mode: GraderMode,
    aggregation_mode: AggregationMode,
    llm,
    samples: List[DataSample],
    max_samples: int = 400,
):
    """Test batch mode"""

    test_samples = samples[:max_samples]
    logger.info(f"Use {len(test_samples)} samples to test batch mode")

    # auto_rubrics = AutoRubrics.create(
    #     language="en",
    #     llm=llm,
    #     evaluation_mode=grader_mode,
    #     aggregation_mode=aggregation_mode,
    #     merge_num_categories=5,
    #     batch_size=5,
    #     mcr_batch_size=5,
    #     min_increment_threshold=0.001,
    #     max_total_rubrics=100,
    #     generate_number=2,
    #     max_epochs=3
    # )

    auto_rubrics = AutoRubrics.create(
        llm=llm,
        evaluation_mode=grader_mode,
        aggregation_mode=aggregation_mode,
        language="zh",
        merge_num_categories=5,
        generate_number=2,
        max_epochs=3,
    )

    results = await auto_rubrics.run(test_samples)

    logger.info(f"Batch mode results: {results}")

    return results


async def main():
    train_file = "./data/processed_mxc/train_samples_结论实用.jsonl"

    llm = OpenAIChatModel(
        model_name="qwen3-32b",
        stream=False,
    )

    samples = load_samples(train_file)
    if not samples:
        logger.error("No valid samples loaded")
        return

    try:
        batch_results = await test_batch_data(
            GraderMode.LISTWISE,
            AggregationMode.CATEGORIZE,
            llm,
            samples,
            max_samples=100,
        )
        output_dir = Path("results/auto_rubrics_结论实用_ZH")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "batch_mode_results.json", "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to: {output_dir}")
        logger.info("AutoRubrics test completed")

    except Exception as e:
        logger.error(f"Error during test: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
