# -*- coding: utf-8 -*-
"""The DashScope chat model class, which unifies the Generation and MultimodalConversation APIs into one method."""
import collections
from datetime import datetime
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    List,
    Literal,
    Type,
    Union,
)

from aioitertools import iter as giter
from loguru import logger
from pydantic import BaseModel

from rm_gallery.core.models.schema.message import ChatMessage

from ..utils.utils import create_tool_from_base_model, repair_and_load_json
from .base_chat_model import BaseChatModel
from .schema.block import TextBlock, ThinkingBlock, ToolUseBlock
from .schema.response import ChatResponse
from .schema.usage import ChatUsage

if TYPE_CHECKING:
    from dashscope.api_entities.dashscope_response import (
        GenerationResponse,
        MultiModalConversationResponse,
    )
else:
    GenerationResponse = "dashscope.api_entities.dashscope_response.GenerationResponse"
    MultiModalConversationResponse = "dashscope.api_entities.dashscope_response.MultiModalConversationResponse"


class DashScopeChatModel(BaseChatModel):
    """The DashScope chat model class, which unifies the Generation and
    MultimodalConversation APIs into one method."""

    def __init__(
        self,
        model: str,
        api_key: str,
        stream: bool = True,
        enable_thinking: bool | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DashScope chat model.

        Args:
            model (`str`):
                The model names.
            api_key (`str`):
                The dashscope API key.
            stream (`bool`):
                The streaming output or not
            enable_thinking (`bool | None`, optional):
                Enable thinking or not, only support Qwen3, QwQ, DeepSeek-R1.
                Refer to `DashScope documentation
                <https://help.aliyun.com/zh/model-studio/deep-thinking>`_
                for more details.
            kwargs (optional):
               The extra keyword arguments used in DashScope API generation,
               e.g. `temperature`, `seed`.
            base_url (`str | None`, optional):
                The base URL for DashScope API requests. If not provided,
                the default base URL from the DashScope SDK will be used.
        """
        if enable_thinking and not stream:
            logger.info(
                "In DashScope API, `stream` must be True when `enable_thinking` is True. ",
            )
            stream = True

        super().__init__(model, stream)

        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.kwargs = kwargs or {}

        if base_url is not None:
            import dashscope

            dashscope.base_url = base_url

    # pylint: disable=too-many-branches
    async def achat(
        self,
        messages: list[dict[str, Any] | ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        callback: Callable | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Get the response from the DashScope Generation/MultimodalConversation API.

        .. note:: This method unifies the DashScope Generation and MultimodalConversation
          APIs into one call, as they support similar arguments and share the same core
          functionality.

        Args:
            messages (list[dict[str, Any] | ChatMessage]):
                A list of message dictionaries or ChatMessage objects. Each dictionary must have
                'role' and 'content' fields.
            tools (list[dict] | None, optional): Defaults to None.
                The tools (functions) JSON schemas that the model can call.
            tool_choice (Literal["auto", "none", "any", "required"] | str | None, optional):
                Defaults to None. Controls which (if any) tool is called by the model.
                Can be "auto", "none", or the name of a specific tool. For more details,
                please refer to the `DashScope Function Calling documentation
                <https://help.aliyun.com/zh/model-studio/qwen-function-calling>`_.
            structured_model (Type[BaseModel] | None, optional): Defaults to None.
                A Pydantic BaseModel class defining the desired structure for the model's output.
                If provided, the model is forced to return data conforming to this schema by
                automatically creating and using a corresponding tool function, and `tool_choice`
                is set to require it.

                .. note:: When `structured_model` is specified, the `tools` and `tool_choice`
                  parameters are overridden and ignored. The model will only perform structured
                  output generation.
            callback (Callable | None, optional): Defaults to None.
                A callback function to process the final response metadata. Only applicable for
                non-streaming responses.
            **kwargs (Any): Additional keyword arguments for the DashScope API, such as
                `temperature`, `max_tokens`, or `top_p`. Refer to the `DashScope API documentation
                <https://help.aliyun.com/zh/dashscope/developer-reference/api-details>`_
                for a complete list.

        Returns:
            ChatResponse | AsyncGenerator[ChatResponse, None]: The parsed chat response(s).
              Returns a single `ChatResponse` object if streaming is disabled, or an asynchronous
              generator yielding `ChatResponse` objects if streaming is enabled.
        """
        import dashscope

        assert isinstance(messages, list), "messages must be a list"
        messages = [message if isinstance(message, dict) else message.to_dict() for message in messages]

        # For qvq and qwen-vl models, the content field cannot be `None` or
        # `[{"text": None}]`, so we need to convert it to an empty list.
        if self.model.startswith("qvq") or "-vl" in self.model:
            for msg in messages:
                if msg["content"] is None or msg["content"] == [
                    {"text": None},
                ]:
                    msg["content"] = []

        kwargs = {
            "messages": messages,
            "model": self.model,
            "stream": self.stream,
            **self.kwargs,
            **kwargs,
            "result_format": "message",
            # In agentscope, the `incremental_output` must be `True` when
            # `self.stream` is True
            "incremental_output": self.stream,
        }

        if tools:
            kwargs["tools"] = self._format_tools_json_schemas(tools)

        if tool_choice:
            self._validate_tool_choice(tool_choice, tools)
            kwargs["tool_choice"] = self._format_tool_choice(tool_choice)

        if self.enable_thinking is not None and "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = self.enable_thinking

        # Check if structured_model is a Pydantic BaseModel class (for structured output)
        if structured_model and isinstance(structured_model, type) and issubclass(structured_model, BaseModel):
            if tools or tool_choice:
                logger.warning(
                    "structured_model is provided. Both 'tools' and "
                    "'tool_choice' parameters will be overridden and "
                    "ignored. The model will only perform structured output "
                    "generation without calling any other tools.",
                )
            format_tool = create_tool_from_base_model(structured_model)
            kwargs["tools"] = self._format_tools_json_schemas(
                [format_tool],
            )
            kwargs["tool_choice"] = self._format_tool_choice(
                format_tool["function"]["name"],
            )
        else:
            # If structured_model is not a Pydantic class or is None
            structured_model = None

        start_datetime = datetime.now()
        if self.model.startswith("qvq") or "-vl" in self.model:
            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                **kwargs,
            )

        else:
            response = await dashscope.aigc.generation.AioGeneration.call(
                api_key=self.api_key,
                **kwargs,
            )

        if self.stream:
            return self._parse_dashscope_stream_response(
                start_datetime,
                response,
                structured_model,
            )

        parsed_response = await self._parse_dashscope_generation_response(
            start_datetime,
            response,
            structured_model,
        )

        # If callback is a function, call it with the response
        if callback and callable(callback):
            try:
                callback_result = callback(parsed_response)
                if isinstance(callback_result, dict):
                    parsed_response.metadata = parsed_response.metadata or {}
                    parsed_response.metadata.update(callback_result)
            except Exception:
                # Log the exception but don't fail the entire operation
                logger.warning("Callback function raised an exception", exc_info=True)

        return parsed_response

    # pylint: disable=too-many-branches
    async def _parse_dashscope_stream_response(
        self,
        start_datetime: datetime,
        response: Union[
            AsyncGenerator[GenerationResponse, None],
            Generator[MultiModalConversationResponse, None, None],
        ],
        structured_model: Type[BaseModel] | None = None,
    ) -> AsyncGenerator[ChatResponse, Any]:
        """Given a DashScope streaming response generator, extract the content
            blocks and usages from it and yield ChatResponse objects.

        Args:
            start_datetime (`datetime`):
                The start datetime of the response generation.
            response (
                `Union[AsyncGenerator[GenerationResponse, None], Generator[ \
                MultiModalConversationResponse, None, None]]`
            ):
                DashScope streaming response generator (GenerationResponse or
                MultiModalConversationResponse) to parse.
            structured_model (`Type[BaseModel] | None`, default `None`):
                A Pydantic BaseModel class that defines the expected structure
                for the model's output.

        Returns:
            AsyncGenerator[ChatResponse, Any]:
                An async generator that yields ChatResponse objects containing
                the content blocks and usage information for each chunk in the
                streaming response.

        .. note::
            If `structured_model` is not `None`, the expected structured output
            will be stored in the metadata of the `ChatResponse`.
        """
        acc_content, acc_thinking_content = "", ""
        acc_tool_calls = collections.defaultdict(dict)
        metadata = None

        async for chunk in giter(response):
            if chunk.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"Failed to get response from _ API: {chunk}",
                )

            message = chunk.output.choices[0].message

            # Update reasoning content
            if isinstance(message.get("reasoning_content"), str):
                acc_thinking_content += message["reasoning_content"]

            # Update text content
            if isinstance(message.content, str):
                acc_content += message.content
            elif isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and "text" in item:
                        acc_content += item["text"]

            # Update tool calls
            for tool_call in message.get("tool_calls", []):
                index = tool_call.get("index", 0)

                if "id" in tool_call and tool_call["id"] != acc_tool_calls[index].get(
                    "id",
                ):
                    acc_tool_calls[index]["id"] = acc_tool_calls[index].get("id", "") + tool_call["id"]

                if "function" in tool_call:
                    func = tool_call["function"]
                    if "name" in func:
                        acc_tool_calls[index]["name"] = acc_tool_calls[index].get("name", "") + func["name"]

                    if "arguments" in func:
                        acc_tool_calls[index]["arguments"] = (
                            acc_tool_calls[index].get("arguments", "") + func["arguments"]
                        )

            # to content blocks
            content_blocks: list[TextBlock | ToolUseBlock | ThinkingBlock] = []
            if acc_thinking_content:
                content_blocks.append(
                    ThinkingBlock(
                        type="thinking",
                        thinking=acc_thinking_content,
                    ),
                )

            if acc_content:
                content_blocks.append(
                    TextBlock(
                        type="text",
                        text=acc_content,
                    ),
                )

            for tool_call in acc_tool_calls.values():
                repaired_input = repair_and_load_json(
                    tool_call.get("arguments", "{}") or "{}",
                )

                if not isinstance(repaired_input, dict):
                    repaired_input = {}

                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tool_call.get("id", ""),
                        name=tool_call.get("name", ""),
                        input=repaired_input,
                    ),
                )

                if structured_model:
                    metadata = repaired_input

            usage = None
            if chunk.usage:
                usage = ChatUsage(
                    input_tokens=chunk.usage.input_tokens,
                    output_tokens=chunk.usage.output_tokens,
                    time=(datetime.now() - start_datetime).total_seconds(),
                )

            parsed_chunk = ChatResponse(
                content=content_blocks,
                usage=usage,
                metadata=metadata,
            )
            yield parsed_chunk

    async def _parse_dashscope_generation_response(
        self,
        start_datetime: datetime,
        response: Union[
            GenerationResponse,
            MultiModalConversationResponse,
        ],
        structured_model: Type[BaseModel] | None = None,
    ) -> ChatResponse:
        """Given a DashScope GenerationResponse object, extract the content
        blocks and usages from it.

        Args:
            start_datetime (`datetime`):
                The start datetime of the response generation.
            response (
                `Union[GenerationResponse, MultiModalConversationResponse]`
            ):
                Dashscope GenerationResponse | MultiModalConversationResponse
                object to parse.
            structured_model (`Type[BaseModel] | None`, default `None`):
                A Pydantic BaseModel class that defines the expected structure
                for the model's output.

        Returns:
            ChatResponse (`ChatResponse`):
                A ChatResponse object containing the content blocks and usage.

        .. note::
            If `structured_model` is not `None`, the expected structured output
            will be stored in the metadata of the `ChatResponse`.
        """
        # Collect the content blocks from the response.
        if response.status_code != 200:
            raise RuntimeError(response)

        content_blocks: List[TextBlock | ToolUseBlock] = []
        metadata: dict | None = None

        message = response.output.choices[0].message
        content = message.get("content")

        if response.output.choices[0].message.get("content") not in [
            None,
            "",
            [],
        ]:
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        content_blocks.append(
                            TextBlock(
                                type="text",
                                text=item["text"],
                            ),
                        )
            else:
                content_blocks.append(
                    TextBlock(
                        type="text",
                        text=content,
                    ),
                )

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                input_ = repair_and_load_json(
                    tool_call["function"].get(
                        "arguments",
                        "{}",
                    )
                    or "{}",
                )
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        name=tool_call["function"]["name"],
                        input=input_,
                        id=tool_call["id"],
                    ),
                )

                if structured_model:
                    metadata = input_

        # Usage information
        usage = None
        if response.usage:
            usage = ChatUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                time=(datetime.now() - start_datetime).total_seconds(),
            )

        parsed_response = ChatResponse(
            content=content_blocks,
            usage=usage,
            metadata=metadata,
        )

        return parsed_response

    def _format_tools_json_schemas(
        self,
        schemas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format the tools JSON schema into required format for DashScope API.

        Args:
            schemas (`dict[str, dict[str, Any]]`):
                The tools JSON schemas.
        """
        # Check schemas format
        for value in schemas:
            if (
                not isinstance(value, dict)
                or "type" not in value
                or value["type"] != "function"
                or "function" not in value
            ):
                raise ValueError(
                    f"Each schema must be a dict with 'type' as 'function' " f"and 'function' key, got {value}",
                )

        return schemas

    def _format_tool_choice(
        self,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None,
    ) -> str | dict | None:
        """Format tool_choice parameter for API compatibility.

        Args:
            tool_choice (`Literal["auto", "none",  "any", "required"] | str \
            | None`, default  `None`):
                Controls which (if any) tool is called by the model.
                 Can be "auto", "none", or specific tool name.
                 For more details, please refer to
                 https://help.aliyun.com/zh/model-studio/qwen-function-calling
        Returns:
            `dict | None`:
                The formatted tool choice configuration dict, or None if
                    tool_choice is None.
        """
        if tool_choice is None:
            return None
        if tool_choice in ["auto", "none"]:
            return tool_choice
        if tool_choice in ["any", "required"]:
            logger.warning(
                "tool_choice '%s' is not supported by DashScope API. "
                "Supported options are 'auto', 'none', or specific function "
                "name. Automatically using 'auto' instead.",
                tool_choice,
            )
            return "auto"
        return {"type": "function", "function": {"name": tool_choice}}
