"""
JSON Validator Grader Module

This module provides functionality to validate if a given text is valid JSON.
It contains the JsonValidatorGrader class which attempts to parse text as JSON
and returns a positive score if parsing succeeds.
"""

import json
from typing import Any
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore


class JsonValidatorGrader(BaseGrader):
    """
    JSON Format Validator Grader

    Validates if the response text is valid JSON.

    Attributes:
        name: Grader name

    Example:
        >>> grader = JsonValidatorGrader()
        >>> result = await grader.aevaluate(
        ...     reference="",  # reference not needed
        ...     response='{"valid": "json"}'
        ... )
        >>> print(f"Valid: {result.score}")
    """

    def __init__(
        self,
        name: str = "json_validator",
        description: str = "JSON format validation metric",
    ):
        super().__init__(
            name=name,
            grader_mode=GraderMode.POINTWISE,
            description=description,
        )

    def _compute(self, response: str) -> tuple[bool, dict]:
        """
        Validate JSON format

        Returns:
            tuple[bool, dict]: (is_valid, details)
        """
        try:
            json.loads(response)
            return True, {"is_valid": True, "response_length": len(response)}
        except json.JSONDecodeError as e:
            return False, {
                "is_valid": False,
                "error_message": f"JSON decode error: {str(e)}",
                "response_length": len(response),
            }
        except TypeError as e:
            return False, {
                "is_valid": False,
                "error_message": f"Type error: {str(e)}",
                "response_length": len(response),
            }
        except Exception as e:
            return False, {
                "is_valid": False,
                "error_message": f"Unexpected error: {str(e)}",
                "response_length": len(response),
            }

    # pylint: disable=unused-argument
    async def aevaluate(
        self,
        reference: str = "",
        response: str = "",
        **kwargs: Any,
    ) -> GraderScore:
        """Validate JSON format"""
        is_valid, details = self._compute(response)

        if not is_valid:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=details.get("error_message", "Invalid JSON"),
                metadata=details,
            )

        return GraderScore(
            name=self.name,
            score=1.0,
            reason="Valid JSON",
            metadata=details,
        )
