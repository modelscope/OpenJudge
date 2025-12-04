# -*- coding: utf-8 -*-
"""Tool accuracy e2e test."""
import os
import asyncio

from rm_gallery.core.graders.agent import ToolCallAccuracyGrader

# pylint: disable=line-too-long

if __name__ == "__main__":
    print(f"start test: {__file__}")
    model_config = {
        "model": "qwen-max",
        "base_url": os.getenv("BASE_URL"),  # read from environment if not provided
        "api_key": os.getenv("API_KEY"),  # read from environment if not provided
        "stream": False,
    }
    evaluator = ToolCallAccuracyGrader(model=model_config)

    # copy from prompts
    query = [
        {
            "message": {
                "content": 'You are a helpful, precise, and proactive AI assistant. You may use tools to fulfill user requests — but only when necessary and appropriate.\n\n## 🛠️ Tool Use Policy — Choose Wisely\n\nAnalyze the user’s intent carefully. Then decide:\n\n1. 🗣️ **Respond Directly**  \n   → For greetings, thanks, chit-chat (e.g., “Hi”, “How are you?”, “What’s your name?”), DO NOT use any tools.\n\n2. 🧰 **Call Tool Directly**  \n   → If the request is a single, factual, and tool-executable task (e.g., “Weather in Shanghai?”, “Population of Paris?”, “Stock price of AAPL?”).  \n   → Use the most relevant tool immediately — no planning needed.\n\n3. 🧭 **Plan First, Then Execute**  \n   → For complex, multi-step, or ambiguous requests (e.g., “Plan a business trip to Tokyo”, “Compare iPhone 15 vs Galaxy S24 and recommend one”).  \n   → First, generate a step-by-step plan using the planning tool.  \n     - ✅ Each step = one specific, tool-executable action  \n     - ✅ Steps must use the same language as the user’s query\n     - ✅ Steps must be ordered logically to gather all required info before final response\n\n\n## Available Tools\n\n### Search Web Tool\nSearches web for the given query and returns the searched results.\nFor time-related queries, better to convert with current date information, for example, "this month" -> "April 2024", "next week" -> "April 25-31, 2024".\n\n### Visit Webpage Tool [visit_webpage]\nVisits one or more specified webpages (by URL) and returns a structured summary of their content based on the user’s goal.\nUse this tool after obtaining URLs (e.g., from a search tool) to extract relevant information.\n\n- Note:\n✅ If answer requires **reading content** (e.g., “What did Biden say in his latest speech?”, “Summarize the new iPhone features”) →\n   1. Call \'search_web\' with time-aware query (e.g., “Biden speech May 2024 summary”)\n   2. Extract top 1-3 relevant URLs\n   3. Auto-call \'visit_webpage\' with goal = user’s original question\n   4. Synthesize answer from visited content → DO NOT return raw links.\n\n     \n## ✍️ Response Style — Always User-Centric\n\n- Be **clear, concise, and friendly**.\n- Match the **user’s tone and formality**.\n- Structure complex answers with **bullets, numbers, or sections**.\n- **Ground every response in tool outputs or verified facts** — never guess or hallucinate.\n- **Language Consistency**: Respond in the same language as the user’s query, unless instructed otherwise.\n- **Image Presentation**: If the context involves relevant images, include them with markdown format in your response to enhance clarity and engagement.\n\n## 📅 Context Awareness\n## Context info\n\ncurrent_datetime: 2025-11-07 16:49:45\n→ Use this to interpret relative time expressions (e.g., “today”, “this week”, “next Monday”) accurately in tool calls.\n',
                "role": "system",
            },
        },
        {
            "message": {
                "content": "美国总统是谁",
                "role": "user",
            },
        },
    ]
    # copy from LLM span's output
    tool_calls = [
        {
            "id": "call_3b5dd701546e4cf89ddc12",
            "function": {
                "arguments": '{"query": "current president of the United States"}',
                "name": "aliyun-websearch",
            },
            "index": 0,
            "type": "function",
        },
    ]
    # copy from LLM span's tools
    tools = [
        {
            "tool": {
                "json_schema": '{"type": "function", "function": {"name": "aliyun-websearch", "description": "从阿里云搜索引擎中搜索给定查询的最新内容。", "parameters": {"properties": {"query": {"title": "Query", "type": "string"}}, "required": ["query"], "type": "object"}}}',
            },
        },
        {
            "tool": {
                "json_schema": '{"type": "function", "function": {"name": "visit_webpage", "description": "Visit webpage(s) and return the content of webpage(s).\\nParams:\\n- url: required, string | list[string], the URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.\\n- goal: required, string, the goal of the visit for webpage(s).\\nReturns:\\n- {\\n  \\"results\\": [\\n    {\\n      \\"url\\": \\"string\\",\\n      \\"goal\\": \\"string\\",\\n      \\"content\\": \\"string\\",\\n      \\"success\\": true\\n    }\\n  ]\\n}\\n", "parameters": {"properties": {"url": {"anyOf": [{"type": "string"}, {"items": {"type": "string"}, "type": "array"}], "default": "", "description": "the URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs", "title": "Url"}, "goal": {"default": "", "description": "the goal of the visit for webpage(s)", "title": "Goal", "type": "string"}, "kwargs": {"title": "Kwargs"}}, "required": ["kwargs"], "type": "object"}}}',
            },
        },
    ]

    eval_result = asyncio.run(evaluator.aevaluate(query=query, tool_calls=tool_calls, tool_definitions=tools))
    print(eval_result)
    assert eval_result["score"] > 3

    print("test result: \033[0;32mPASS\033[0m")
