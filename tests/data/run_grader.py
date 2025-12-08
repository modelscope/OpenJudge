import argparse
import asyncio
import json
import os

from rm_gallery.core.graders.agent import *
from rm_gallery.core.graders.common import *
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def run_cases(case_file: str, skip: int):
    model_config = {
        "model": "qwen-max",
        "base_url": os.getenv("BASE_URL"),
        "api_key": os.getenv("API_KEY"),
        "stream": False,
    }

    with open(case_file) as f:
        content = f.read()
        cases = json.loads(content)

        for case in cases[skip:]:
            cls_name = case.get("grader")
            kwargs = case.get("parameters")
            index = case["index"]

            try:
                grader = eval(cls_name)(model=model_config, language=LanguageEnum.ZH)
                result = asyncio.run(grader.aevaluate(**kwargs))

                if "min_expect_score" in case:
                    if result.score < case["min_expect_score"]:
                        print(f"\033[91mFAILED\033[0m, index: {index}, result: {result}")
                        continue
                elif "max_expect_score" in case:
                    if result.score > case["max_expect_score"]:
                        print(f"\033[91mFAILED\033[0m, index: {index}, result: {result}")
                        continue
                else:
                    print(f"\033[91mFAILED\033[0m: index: {index}, missing min_expect_score or max_expect_score")
                    continue

                print(f"PASSED: index: {index}, score: {result.score}")

            except Exception as e:
                print(f"failed case index: {index}")
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grader test cases.")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of test cases to skip from the beginning (default: 0)",
    )
    args = parser.parse_args()

    run_cases(case_file="grader_cases.json", skip=args.skip)
