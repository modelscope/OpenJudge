# -*- coding: utf-8 -*-
from typing import Callable, Dict, List, Union

from jsonschema import ValidationError, validate
from pydantic import BaseModel, Field

from rm_gallery.core.utils import get_value_by_path


class EvalCase(BaseModel):
    """EvalCase containing shared input and individual outputs.

    EvalCase is the basic data structure for evaluation tasks. It consists of
    shared context data and independent outputs to be evaluated.

    For pointwise evaluation: Each EvalCase contains one output in the outputs list.
    For listwise evaluation: Each EvalCase contains multiple outputs in the outputs list.

    Attributes:
        input (dict): A dictionary containing shared input for all outputs
                    (e.g., query, reference answer).
        outputs (List[dict]): A list of dictionaries, each representing an
                             individual output to evaluate.
    """

    input: dict = Field(
        default_factory=dict,
        description="Shared input for all outputs",
    )
    outputs: List[dict] = Field(
        default_factory=list,
        description="List of individual outputs to evaluate",
    )


# EvalCaseParser is a type alias for either a dictionary mapping or a callable function
# that takes a EvalCase and returns a EvalCase
EvalCaseParser = Union[
    Dict[str, str],
    Callable[["EvalCase"], "EvalCase"],
]


def parse_eval_case(
    eval_case: EvalCase,
    parser: EvalCaseParser | None,
) -> EvalCase:
    """Parse an eval case using a parser to transform its structure.

    This function transforms an EvalCase according to the provided parser. It supports
    two types of parsers: dictionary-based field mappings and callable functions.
    When using dictionary-based parsing, fields from the input or outputs can be
    remapped to new field names. When using callable parsing, a custom function
    can be applied to transform the entire EvalCase.

    Args:
        eval_case (EvalCase): An eval case to parse and transform.
        parser (EvalCaseParser | None): A parser to use for transforming the eval case.
            Can be either:

            1. A dictionary with direct field mappings where paths start with "input" or "output":
               - Paths starting with "input" map fields from the input section
               - Paths starting with "output" map fields from each output in the outputs list

            2. A callable function that takes an EvalCase and returns a transformed EvalCase

            3. None, in which case the eval case is returned unchanged

    Returns:
        EvalCase: The parsed/transformed eval case. If parser is None, returns the original
                 eval case. Otherwise, returns a new EvalCase with transformed structure.

    Raises:
        ValueError: If parser is of an invalid type or if a path has an invalid prefix.

    Examples:
        >>> from rm_gallery.core.schema.input import EvalCase
        >>> # Create an example eval case
        >>> eval_case = EvalCase(
        ...     input={"question": "What is 2+2?"},
        ...     outputs=[{"response": "4"}, {"response": "four"}]
        ... )
        >>>
        >>> # Dictionary-based parsing
        >>> parser_dict = {
        ...     "query": "input.question",
        ...     "answer": "output.response"
        ... }
        >>> parsed_case = parse_eval_case(eval_case, parser_dict)
        >>> print(parsed_case.input)
        {'query': 'What is 2+2?'}
        >>> print(parsed_case.outputs)
        [{'answer': '4'}, {'answer': 'four'}]
        >>>
        >>> # Callable-based parsing
        >>> def custom_parser(case):
        ...     # Custom transformation logic
        ...     new_input = case.input.copy()
        ...     new_input["processed"] = True
        ...     return EvalCase(input=new_input, outputs=case.outputs)
        >>> parsed_case = parse_eval_case(eval_case, custom_parser)
        >>> print(parsed_case.input)
        {'question': 'What is 2+2?', 'processed': True}
        >>>
        >>> # No parsing (parser is None)
        >>> parsed_case = parse_eval_case(eval_case, None)
        >>> print(parsed_case.input)
        {'question': 'What is 2+2?'}
    """
    if parser is None:
        return eval_case

    # Handle dictionary configuration
    if isinstance(parser, dict):
        # Apply mappings to the main input dictionary
        input = eval_case.input.copy()
        outputs = [output.copy() for output in eval_case.outputs]

        # Process mappings based on path prefix
        for target_field, source_path in parser.items():
            path_parts = source_path.split(".")
            if not path_parts:
                continue

            # Check the first part of the path to determine input source
            if path_parts[0] == "input":
                # Get value from input
                if len(path_parts) > 1:
                    value = get_value_by_path(
                        eval_case.input,
                        ".".join(path_parts[1:]),
                    )
                else:
                    value = eval_case.input
                input[target_field] = value

            elif path_parts[0] == "output":
                # Apply to each output in outputs list
                for i, output in enumerate(eval_case.outputs):
                    if len(path_parts) > 1:
                        value = get_value_by_path(
                            output,
                            ".".join(path_parts[1:]),
                        )
                    else:
                        value = output
                    if len(outputs) <= i:
                        outputs.append({})
                    outputs[i][target_field] = value
            else:
                raise ValueError(f"Invalid path prefix: {path_parts[0]}")

        return EvalCase(input=input, outputs=outputs)

    # Handle callable directly
    if callable(parser):
        return parser(eval_case)

    raise ValueError(f"Invalid parser type: {type(parser)}")


def validate_eval_cases(
    eval_cases: List[dict | EvalCase],
    schema: dict | None = None,
) -> List[EvalCase]:
    """Validate a list of eval cases against a JSON schema.

    This function validates that all eval cases conform to the provided schema.
    If an eval case is already an EvalCase object, it validates the input part
    against the schema. If it's a dict, it validates the dict directly.

    Args:
        eval_cases (List[dict | EvalCase]): A list of eval cases to validate.
            Can contain either dictionaries or EvalCase objects.
        schema (dict | None): The JSON schema to validate against. Required.

    Returns:
        List[EvalCase]: A list of validated EvalCase objects.

    Raises:
        ValueError: If schema is not provided or if any eval case doesn't conform
                   to the schema.

    Examples:
        >>> from rm_gallery.core.schema.data import EvalCase, validate_eval_cases
        >>> # Define a simple schema
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "question": {"type": "string"}
        ...     },
        ...     "required": ["question"]
        ... }
        >>> # Create eval cases
        >>> eval_cases = [
        ...     {"question": "What is 2+2?"},
        ...     EvalCase(input={"question": "What is 3*3?"}, outputs=[{"answer": "9"}])
        ... ]
        >>> # Validate the eval cases
        >>> validated_cases = validate_eval_cases(eval_cases, schema)
        >>> len(validated_cases)
        2
        >>> # Invalid case - missing required field
        >>> invalid_cases = [{"answer": "4"}]
        >>> try:
        ...     validate_eval_cases(invalid_cases, schema)
        ... except ValueError as e:
        ...     print(f"Validation error: {e}")
        Validation error: EvalCase at index 0 does not conform to schema: 'question' is a required property
    """
    # Validate that eval_case_schema is provided
    if not schema:
        raise ValueError("the schema of eval case is required")

    # Validate that all eval cases conform to eval_case_schema
    for i, case in enumerate(eval_cases):
        try:
            # If it's already a EvalCase, convert to dict for validation
            if isinstance(case, EvalCase):
                # For EvalCase objects, we validate the 'data' part against the schema
                case_dict = case.model_dump()
            else:
                # For dict objects, validate directly
                case_dict = case

            validate(instance=case_dict, schema=schema)
        except ValidationError as e:
            raise ValueError(
                f"EvalCase at index {i} does not conform to schema: {str(e)}",
            )
        except Exception as e:
            raise ValueError(
                f"Error validating eval case at index {i}: {str(e)}",
            )

    return [
        EvalCase(**eval_case) if isinstance(eval_case, dict) else eval_case
        for eval_case in eval_cases
    ]
