# -*- coding: utf-8 -*-
"""
Test script for SearchCorrectnessGrader.

This file demonstrates quality tests for SearchCorrectnessGrader, comparing:
- LLM-only (CorrectnessGrader): No tools, relies on training data
- Search-based (SearchCorrectnessGrader): Uses web search for verification

Example:
    Run all tests:
    ```bash
    export TAVILY_API_KEY="tvly-..."
    pytest tests/graders/common/test_search_correctness.py -v
    ```

    Or run directly with Python:
    ```bash
    export TAVILY_API_KEY="tvly-..."
    python tests/graders/common/test_search_correctness.py
    ```
"""

import asyncio
import os

from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.search_correctness import SearchCorrectnessGrader
from openjudge.models import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum


def get_model() -> OpenAIChatModel:
    return OpenAIChatModel(model="qwen3-32b")


async def test_accurate_info():
    """Test: Accurate information about recent news."""
    print("\n" + "=" * 60)
    print("Test 1: Accurate Information")
    print("=" * 60)

    query = "介绍一下紫金矿业2024年的经营情况和主要业绩"
    response = """紫金矿业2024年经营情况如下：
1. 营业收入：全年实现营业收入约3200亿元，同比增长约15%
2. 净利润：归属于上市公司股东的净利润约320亿元，创历史新高
3. 主要产品产量：
   - 矿产金产量约75吨，同比增长约10%
   - 矿产铜产量约115万吨，同比增长约8%
   - 矿产锌产量约48万吨
4. 重大项目进展：
   - 刚果(金)卡莫阿-卡库拉铜矿三期建设顺利推进
   - 西藏巨龙铜矿一期投产
   - 塞尔维亚丘卡卢-佩吉铜金矿产能持续提升
5. 市值表现：公司市值突破4000亿元，位居全球矿业公司前列"""

    print(f"\nQuery: {query}")
    print(f"Response: {response}")

    model = get_model()
    tavily_key = os.getenv("TAVILY_API_KEY")

    # LLM-only (baseline)
    print("\n[LLM-only] CorrectnessGrader:")
    llm_grader = CorrectnessGrader(model=model, language=LanguageEnum.ZH)
    llm_result = await llm_grader.aevaluate(
        query=query,
        response=response,
        reference_response="",  # No reference, LLM judges based on its knowledge
    )
    print(f"  Score: {llm_result.score}")
    print(f"  Reason: {llm_result.reason}")

    # Search-based
    if tavily_key:
        print("\n[Search-based] SearchCorrectnessGrader:")
        search_grader = SearchCorrectnessGrader(
            model=model,
            tavily_api_key=tavily_key,
            language=LanguageEnum.ZH,
        )
        search_result = await search_grader.aevaluate(query=query, response=response)
        print(f"  Score: {search_result.score}")
        print(f"  Reason: {search_result.reason}")
        print(f"  Tool Calls: {search_result.metadata.get('tool_calls', 0)}")
    else:
        print("\n[Search-based] Skipped (TAVILY_API_KEY not set)")


async def main():
    print("SearchCorrectnessGrader Test")
    print("Comparing LLM-only vs Search-based judgment\n")

    if not os.getenv("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY not set. Only LLM-only tests will run.\n")

    await test_accurate_info()


if __name__ == "__main__":
    asyncio.run(main())
