"""
Module for computing code correctness scores using test cases.

This module provides functionality to evaluate code solutions against test cases,
checking correctness and providing scoring metrics. It uses the APPS-style
correctness checking approach with timeout handling.
"""

import json
import traceback

from loguru import logger

from .utils import check_correctness as apps_check_correctness


def compute_score(completion, test_cases, continuous=False):
    """
    Compute the correctness score of a code completion against test cases.

    Args:
        completion (str): The code completion to evaluate, potentially including markdown formatting
        test_cases (dict or str): Test cases in dictionary form or JSON string format with 'inputs' and 'outputs' keys
        continuous (bool): If True, compute a partial score based on individual test case results

    Returns:
        tuple: A tuple containing:
            - bool or float: Success flag (bool) if continuous=False, otherwise ratio of passed tests (float)
            - dict or list: Metadata about the test execution, or list of metadata if continuous=True
    """
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = completion.split("```python")[-1].split("```")[0]
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            logger.error(f"Error:{e}")

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5)
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            if success:
                return success, metadata
        except Exception:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        # pylint: disable=consider-using-enumerate
        for i in range(len(inputs)):
            test_cases_list.append(
                {"inputs": [inputs[i]], "outputs": [outputs[i]]},
            )

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=10)
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list
