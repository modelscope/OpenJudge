# -*- coding: utf-8 -*-
import json
from typing import Any, Dict, Type

from json_repair import repair_json
from pydantic import BaseModel


def _json_loads_with_repair(
    json_str: str,
) -> dict | list | str | float | int | bool | None:
    """The given json_str maybe incomplete, e.g. '{"key', so we need to
    repair and load it into a Python object.
    """
    repaired = json_str
    try:
        repaired = repair_json(json_str)
    except Exception:
        pass

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to decode JSON string `{json_str}` after repairing it "
            f"into `{repaired}`. Error: {e}",
        ) from e


def _remove_title_field(schema: dict) -> None:
    """Remove the title field from the JSON schema to avoid
    misleading the LLM."""
    # The top level title field
    if "title" in schema:
        schema.pop("title")

    # properties
    if "properties" in schema:
        for prop in schema["properties"].values():
            if isinstance(prop, dict):
                _remove_title_field(prop)

    # items
    if "items" in schema and isinstance(schema["items"], dict):
        _remove_title_field(schema["items"])

    # additionalProperties
    if "additionalProperties" in schema and isinstance(
        schema["additionalProperties"],
        dict,
    ):
        _remove_title_field(
            schema["additionalProperties"],
        )


def _create_tool_from_base_model(
    structured_model: Type[BaseModel],
    tool_name: str = "generate_structured_output",
) -> Dict[str, Any]:
    """Create a function tool definition from a Pydantic BaseModel.
    This function converts a Pydantic BaseModel class into a tool definition
    that can be used with function calling API. The resulting tool
    definition includes the model's JSON schema as parameters, enabling
    structured output generation by forcing the model to call this function
    with properly formatted data.

    Args:
        structured_model (`Type[BaseModel]`):
            A Pydantic BaseModel class that defines the expected structure
            for the tool's output.
        tool_name (`str`, default `"generate_structured_output"`):
            The tool name that used to force the LLM to generate structured
            output by calling this function.

    Returns:
        `Dict[str, Any]`: A tool definition dictionary compatible with
            function calling API, containing type ("function") and
            function dictionary with name, description, and parameters
            (JSON schema).

    .. code-block:: python
        :caption: Example usage

        from pydantic import BaseModel

        class PersonInfo(BaseModel):
            name: str
            age: int
            email: str

        tool = _create_tool_from_base_model(PersonInfo, "extract_person")
        print(tool["function"]["name"])  # extract_person
        print(tool["type"])              # function

    .. note:: The function automatically removes the 'title' field from
        the JSON schema to ensure compatibility with function calling
        format. This is handled by the internal ``_remove_title_field()``
        function.
    """
    schema = structured_model.model_json_schema()

    _remove_title_field(schema)
    tool_definition = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Generate the required structured output with "
            "this function",
            "parameters": schema,
        },
    }
    return tool_definition
