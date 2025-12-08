"""
A simplified demo for user custom template and get result.

User need to provide a grader template to instruct LLM how to do grading.
Reture format MUST be json, with score and reason as key.
"""

import asyncio
import os

from rm_gallery.core.graders.llm_grader import LLMGrader


def custom_llm_grader_demo():
    tempalte = """You're a LLM query answer relevance grader, you'll received Query/Response:
    Query: {query}
    Response: {response}
    Please read query/response, if the Response answers the Query, return 1, return 0 if no.
    Return format, json.
    ```
    {{
        "score": score,
        "reason": "scoring reason",
    }}
    ```
    """
    model_config = {
        "model": "qwen-max",
        "base_url": os.getenv("BASE_URL"),
        "api_key": os.getenv("API_KEY"),
        "stream": False,
    }

    grader = LLMGrader(model=model_config, name="my_grader", template=tempalte)
    eval_result = asyncio.run(grader.aevaluate(query="1+1=?", response="2"))
    print(eval_result)


if __name__ == "__main__":
    custom_llm_grader_demo()
