
import json
from typing import Any
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore


class JsonValidatorGrader(BaseGrader):
    """
    JSON Format Validator Grader

    Validates if the candidate text is valid JSON.

    Attributes:
        name: Grader name

    Example:
        >>> grader = JsonValidatorGrader()
        >>> result = await grader.aevaluate(
        ...     reference="",  # reference not needed
        ...     candidate='{"valid": "json"}'
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

    def _compute(self, candidate: str) -> tuple[bool, dict]:
        """
        Validate JSON format

        Returns:
            tuple[bool, dict]: (is_valid, details)
        """
        try:
            json.loads(candidate)
            return True, {"is_valid": True, "candidate_length": len(candidate)}
        except json.JSONDecodeError as e:
            return False, {
                "is_valid": False,
                "error_message": f"JSON decode error: {str(e)}",
                "candidate_length": len(candidate),
            }
        except TypeError as e:
            return False, {
                "is_valid": False,
                "error_message": f"Type error: {str(e)}",
                "candidate_length": len(candidate),
            }
        except Exception as e:
            return False, {
                "is_valid": False,
                "error_message": f"Unexpected error: {str(e)}",
                "candidate_length": len(candidate),
            }

    async def aevaluate(
        self,
        reference: str = "",
        candidate: str = "",
        **kwargs: Any,
    ) -> GraderScore:
        """Validate JSON format"""
        is_valid, details = self._compute(candidate)

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



