# -*- coding: utf-8 -*-
"""Tool accuracy e2e test."""
import asyncio
import json
import os

from rm_gallery.core.graders.agent.tool.tool_call_accuracy import ToolCallAccuracyGrader
from rm_gallery.core.models.schema.prompt_template import LanguageEnum

# pylint: disable=line-too-long

def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

async def main():
    print(f"start test: {__file__}")
    model_config = {
        "model": "qwen-max",
        "base_url": os.getenv("BASE_URL"),  # read from environment if not provided
        "api_key": os.getenv("API_KEY"),  # read from environment if not provided
        "stream": False,
    }

    pass_data_file_path = 'data/agent_eval_data_pass.jsonl'
    fail_data_file_path = 'data/agent_eval_data_fail.jsonl'

    # test EN version
    evaluator = ToolCallAccuracyGrader(model=model_config, language=LanguageEnum.EN)
    for record in read_jsonl_file(pass_data_file_path):
        query = record.get('query', [])
        tool_calls = record.get('tool_calls', [])
        tools = record.get('tools', [])
        eval_result = await evaluator.aevaluate(query=query, tool_definitions=tools, tool_calls=tool_calls)
        print(eval_result)
        assert  eval_result.score > 3

    for record in read_jsonl_file(fail_data_file_path):
        query = record.get('query', [])
        tool_calls = record.get('tool_calls', [])
        tools = record.get('tools', [])
        eval_result = await evaluator.aevaluate(query=query, tool_definitions=tools, tool_calls=tool_calls)
        print(eval_result)
        assert  eval_result.score <= 3

    # test ZH version
    evaluator = ToolCallAccuracyGrader(model=model_config, language=LanguageEnum.ZH)
    for record in read_jsonl_file(pass_data_file_path):
        query = record.get('query', [])
        tool_calls = record.get('tool_calls', [])
        tools = record.get('tools', [])
        eval_result = await evaluator.aevaluate(query=query, tool_definitions=tools, tool_calls=tool_calls)
        print(eval_result)
        assert  eval_result.score > 3

    for record in read_jsonl_file(fail_data_file_path):
        query = record.get('query', [])
        tool_calls = record.get('tool_calls', [])
        tools = record.get('tools', [])
        eval_result = await evaluator.aevaluate(query=query, tool_definitions=tools, tool_calls=tool_calls)
        print(eval_result)
        assert  eval_result.score <= 3

if __name__ == "__main__":
    asyncio.run(main())
