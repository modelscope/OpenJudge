# -*- coding: utf-8 -*-
"""The formatter module."""

import base64
import tempfile
from abc import abstractmethod
from typing import Any, List

from rm_gallery.core.schema.block import AudioBlock, ImageBlock, TextBlock
from rm_gallery.core.schema.message import ChatMessage


def _save_base64_data(
    media_type: str,
    base64_data: str,
) -> str:
    """Save the base64 data to a temp file and return the file path. The
    extension is guessed from the MIME type.

    Args:
        media_type (`str`):
            The MIME type of the data, e.g. "image/png", "audio/mpeg".
        base64_data (`str):
            The base64 data to be saved.
    """
    extension = "." + media_type.split("/")[-1]

    with tempfile.NamedTemporaryFile(
        suffix=f".{extension}",
        delete=False,
    ) as temp_file:
        decoded_data = base64.b64decode(base64_data)
        temp_file.write(decoded_data)
        temp_file.close()
        return temp_file.name


class FormatterBase:
    """The base class for formatters."""

    @abstractmethod
    async def format(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Format the ChatMessage objects to a list of dictionaries that satisfy the
        API requirements."""

    @staticmethod
    def assert_list_of_msgs(msgs: list[ChatMessage]) -> None:
        """Assert that the input is a list of ChatMessage objects.

        Args:
            msgs (`list[ChatMessage]`):
                A list of ChatMessage objects to be validated.
        """
        if not isinstance(msgs, list):
            raise TypeError("Input must be a list of ChatMessage objects.")

        for msg in msgs:
            if not isinstance(msg, ChatMessage):
                raise TypeError(
                    f"Expected ChatMessage object, got {type(msg)} instead.",
                )

    @staticmethod
    def convert_tool_result_to_string(
        output: str | List[TextBlock | ImageBlock | AudioBlock],
    ) -> str:
        """Turn the tool result list into a textual output to be compatible
        with the LLM API that doesn't support multimodal data.

        Args:
            output (`str | List[TextBlock | ImageBlock | AudioBlock]`):
                The output of the tool response, including text and multimodal
                data like images and audio.

        Returns:
            `str`:
                A string representation of the tool result, with text blocks
                concatenated and multimodal data represented by file paths
                or URLs.
        """

        if isinstance(output, str):
            return output

        textual_output = []
        for block in output:
            assert isinstance(block, dict) and "type" in block, (
                f"Invalid block: {block}, a TextBlock, ImageBlock, or "
                f"AudioBlock is expected."
            )
            if block["type"] == "text":
                textual_output.append(block["text"])

            elif block["type"] in ["image", "audio", "video"]:
                assert "source" in block, (
                    f"Invalid {block['type']} block: {block}, 'source' key "
                    "is required."
                )
                source = block["source"]
                # Save the image locally and return the file path
                if source["type"] == "url":
                    textual_output.append(
                        f"The returned {block['type']} can be found "
                        f"at: {source['url']}",
                    )

                elif source["type"] == "base64":
                    path_temp_file = _save_base64_data(
                        source["media_type"],
                        source["data"],
                    )
                    textual_output.append(
                        f"The returned {block['type']} can be found "
                        f"at: {path_temp_file}",
                    )

                else:
                    raise ValueError(
                        f"Invalid image source: {block['source']}, "
                        "expected 'url' or 'base64'.",
                    )

            else:
                raise ValueError(
                    f"Unsupported block type: {block['type']}, "
                    "expected 'text', 'image', 'audio', or 'video'.",
                )

        if len(textual_output) == 1:
            return textual_output[0]

        else:
            return "\n".join("- " + _ for _ in textual_output)
