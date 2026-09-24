# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's OpenAIServingResponses
"""Handler for /v1/responses requests"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from contextlib import AsyncExitStack
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Optional, Union

import jinja2
import openai.types.responses as openai_responses_types
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)
from openai_harmony import Message as OpenAIMessage

from sglang.srt.entrypoints.context import (
    ConversationContext,
    HarmonyContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from sglang.srt.entrypoints.harmony_utils import (
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_system_message,
    get_user_message,
    parse_output_message,
    parse_remaining_state,
    parse_response_input,
    render_for_completion,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    Function,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
    Tool,
    ToolChoice,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.tool_server import MCPToolServer, ToolServer
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import random_uuid
from sglang.utils import convert_json_schema_to_str

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _serialize_tool_call_constraint(constraint_type: str, constraint_value: Any) -> Any:
    if constraint_type == "structural_tag":
        return convert_json_schema_to_str(constraint_value.model_dump(by_alias=True))
    if constraint_type == "json_schema":
        return convert_json_schema_to_str(constraint_value)
    return constraint_value


def _response_event(event_type: str, sequence_number: int, **payload: Any) -> str:
    data = {"type": event_type, "sequence_number": sequence_number, **payload}
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class OpenAIServingResponses(OpenAIServingChat):
    """Handler for /v1/responses requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
        *,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        tool_server: Optional[ToolServer] = None,
    ) -> None:
        super().__init__(tokenizer_manager, template_manager)

        # template_manager is already set by parent class
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage

        # Get default sampling params from model config if available
        self.default_sampling_params = {}

        self.supports_browsing = (
            tool_server.has_tool("browser") if tool_server else False
        )
        self.supports_code_interpreter = (
            tool_server.has_tool("python") if tool_server else False
        )
        self.tool_server = tool_server
        # Get from model config
        self.use_harmony = (
            self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        if self.use_harmony:
            # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
            # We need to add them to the stop token ids.
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        # Response storage for background and retrieval operations
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # Message storage for conversation continuity
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.msg_store: dict[
            str, Union[list[ChatCompletionMessageParam], list["OpenAIMessage"]]
        ] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

    # error helpers dedicated for v1/responses
    def create_error_response(
        self,
        message: str,
        err_type: str = "invalid_request_error",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        nested_error = {
            "message": message,
            "type": err_type,
            "param": param,
            "code": status_code,
        }
        return ORJSONResponse(content={"error": nested_error}, status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        return json.dumps(
            {
                "error": {
                    "message": message,
                    "type": err_type,
                    "param": None,
                    "code": status_code,
                }
            }
        )

    def _request_id_prefix(self) -> str:
        return "resp_"

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ResponsesResponse, ORJSONResponse]:
        # Validate model
        if not self.tokenizer_manager:
            return self.create_error_response("Model not loaded")

        # FIXME: If the engine is dead, raise an error
        # This is required for the streaming case

        # Handle the previous response ID
        prev_response_id = request.previous_response_id
        if prev_response_id is not None:
            if not prev_response_id.startswith("resp_"):
                return self._make_invalid_id_error(prev_response_id)
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None

        try:
            model_name = request.model or "default"
            tokenizer = self.tokenizer_manager.tokenizer
            processed_messages = None
            require_reasoning = False

            if self.use_harmony:
                messages, request_prompts, engine_prompts = (
                    self._make_request_with_harmony(request, prev_response)
                )
            else:
                (
                    messages,
                    request_prompts,
                    engine_prompts,
                    processed_messages,
                    require_reasoning,
                ) = await self._make_request(request, prev_response, tokenizer)

        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        if not self.use_harmony:
            ignored_tool_types = sorted(
                {tool.type for tool in request.tools if tool.type != "function"}
            )
            if ignored_tool_types:
                logger.warning(
                    "Responses API accepted but did not expose unsupported DSV4 "
                    "tool types to the model: %s",
                    ", ".join(ignored_tool_types),
                )

        if (
            self.tool_server is not None
            and isinstance(self.tool_server, MCPToolServer)
            and (request.background or request.stream)
            and request.tools
            and any(
                tool.type in ["web_search_preview", "code_interpreter"]
                for tool in request.tools
            )
        ):
            return self.create_error_response(
                "MCP tool server is not supported in background mode and "
                "streaming mode"
            )

        # Schedule the request and get the result generator
        generators: list[AsyncGenerator[Any, None]] = []
        tool_list = []
        if self.use_harmony:
            if self.supports_browsing:
                tool_list.append("browser")
            if self.supports_code_interpreter:
                tool_list.append("python")
        async with AsyncExitStack() as exit_stack:
            try:
                if self.tool_server is not None:
                    tool_session_ctxs: dict[str, Any] = {
                        tool_name: exit_stack.enter_async_context(
                            self.tool_server.get_tool_session(tool_name)
                        )
                        for tool_name in tool_list
                    }
                    tool_sessions = {}
                    for tool_name in tool_list:
                        tool_sessions[tool_name] = await tool_session_ctxs[tool_name]
                else:
                    assert len(tool_list) == 0
                    tool_sessions = {}
                for i, engine_prompt in enumerate(engine_prompts):
                    # Calculate default max tokens from context length minus prompt length
                    if hasattr(engine_prompt, "__len__"):
                        prompt_length = len(engine_prompt)
                    elif isinstance(engine_prompt, list):
                        prompt_length = len(engine_prompt)
                    else:
                        prompt_length = 0

                    context_len = (
                        self.tokenizer_manager.model_config.context_len
                        if hasattr(self.tokenizer_manager.model_config, "context_len")
                        else 4096
                    )
                    default_max_tokens = max(
                        context_len - prompt_length, 512
                    )  # Ensure minimum 512 tokens
                    sampling_params = request.to_sampling_params(
                        default_max_tokens, self.default_sampling_params
                    )
                    if processed_messages is not None:
                        sampling_params["stop"] = processed_messages.stop
                        if processed_messages.tool_call_constraint is not None:
                            constraint_type, constraint_value = (
                                processed_messages.tool_call_constraint
                            )
                            constraint_value = _serialize_tool_call_constraint(
                                constraint_type, constraint_value
                            )
                            sampling_params[constraint_type] = constraint_value
                        if not getattr(processed_messages, "skip_special_tokens", True):
                            sampling_params["skip_special_tokens"] = False

                    context: ConversationContext
                    if self.use_harmony:
                        if request.stream:
                            context = StreamingHarmonyContext(messages, tool_sessions)
                        else:
                            context = HarmonyContext(messages, tool_sessions)
                    else:
                        context = SimpleContext()

                    # Create GenerateReqInput for SGLang
                    prompt_kwargs = (
                        {"text": engine_prompt}
                        if isinstance(engine_prompt, str)
                        else {"input_ids": engine_prompt}
                    )
                    adapted_request = GenerateReqInput(
                        **prompt_kwargs,
                        image_data=(
                            processed_messages.image_data
                            if processed_messages is not None
                            else None
                        ),
                        video_data=(
                            processed_messages.video_data
                            if processed_messages is not None
                            else None
                        ),
                        audio_data=(
                            processed_messages.audio_data
                            if processed_messages is not None
                            else None
                        ),
                        modalities=(
                            processed_messages.modalities
                            if processed_messages is not None
                            else None
                        ),
                        sampling_params=sampling_params,
                        stream=request.stream,
                        rid=request.request_id,
                        extra_key=self._compute_extra_key(request),
                        background=request.background,
                        require_reasoning=require_reasoning,
                    )

                    generator = self._generate_with_builtin_tools(
                        request.request_id,
                        request_prompts[i],
                        adapted_request,
                        sampling_params,
                        context,
                        raw_request=raw_request,
                        priority=request.priority,
                    )
                    generators.append(generator)
            except ValueError as e:
                return self.create_error_response(str(e))

            assert len(generators) == 1
            (result_generator,) = generators

            # Store the input messages
            if request.store:
                self.msg_store[request.request_id] = messages

            if request.background:
                created_time = int(time.time())
                response = ResponsesResponse.from_request(
                    request,
                    sampling_params,
                    model_name=model_name,
                    created_time=created_time,
                    output=[],
                    status="queued",
                    usage=None,
                )
                async with self.response_store_lock:
                    self.response_store[response.id] = response

                # Run the request in the background
                task = asyncio.create_task(
                    self._run_background_request(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                        require_reasoning=require_reasoning,
                    ),
                    name=f"create_{response.id}",
                )

                # For cleanup
                self.background_tasks[response.id] = task
                task.add_done_callback(
                    lambda _: self.background_tasks.pop(response.id, None)
                )
                return response

            if request.stream:
                if not self.use_harmony:
                    return self.responses_stream_generator_non_harmony(
                        request,
                        sampling_params,
                        result_generator,
                        model_name,
                        tokenizer,
                        request_metadata,
                        require_reasoning=require_reasoning,
                    )
                return self.responses_stream_generator(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                )
            try:
                result: Union[ORJSONResponse, ResponsesResponse] = (
                    await self.responses_full_generator(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        require_reasoning=require_reasoning,
                    )
                )
                return result
            except Exception as e:
                return self.create_error_response(str(e))
        return self.create_error_response("Unknown error")

    @staticmethod
    def _chat_tool_choice(tool_choice: Any) -> Any:
        if not isinstance(tool_choice, dict):
            return tool_choice
        if tool_choice.get("type") != "function" or not tool_choice.get("name"):
            raise ValueError(
                "Only named top-level function tool_choice is supported for "
                "non-Harmony Responses requests."
            )
        return ToolChoice(function={"name": tool_choice["name"]})

    @staticmethod
    def _response_tools_to_chat_tools(request: ResponsesRequest) -> list[Tool]:
        chat_tools = []
        for response_tool in request.tools:
            if response_tool.type != "function":
                continue
            if not response_tool.name:
                raise ValueError("Responses function tools require a non-empty name.")
            chat_tools.append(
                Tool(
                    type="function",
                    function=Function(
                        name=response_tool.name,
                        description=response_tool.description,
                        parameters=response_tool.parameters,
                        strict=bool(response_tool.strict),
                    ),
                )
            )
        return chat_tools

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
        tokenizer: Any,
    ):
        messages = self._construct_input_messages(request, prev_response)
        chat_tools = self._response_tools_to_chat_tools(request)
        chat_request = ChatCompletionRequest(
            model=request.model or "default",
            messages=messages,
            stream=bool(request.stream),
            tools=chat_tools or None,
            tool_choice=(
                self._chat_tool_choice(request.tool_choice) if chat_tools else "none"
            ),
            parallel_tool_calls=(
                True
                if request.parallel_tool_calls is None
                else request.parallel_tool_calls
            ),
            stop=request.stop,
            max_completion_tokens=request.max_output_tokens,
            reasoning_effort=(
                request.reasoning.effort if request.reasoning is not None else None
            ),
            chat_template_kwargs=(
                dict(request.chat_template_kwargs)
                if request.chat_template_kwargs is not None
                else None
            ),
        )
        validation_error = self._validate_request(chat_request)
        if validation_error is not None:
            raise ValueError(validation_error)

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal
        processed_messages = self._process_messages(chat_request, is_multimodal)
        processed_messages.skip_special_tokens = chat_request.skip_special_tokens
        request.chat_template_kwargs = chat_request.chat_template_kwargs
        require_reasoning = self._get_reasoning_from_request(chat_request)

        if is_multimodal:
            request_prompts = [processed_messages.prompt]
            engine_prompts = [processed_messages.prompt]
        else:
            request_prompts = [processed_messages.prompt_ids]
            engine_prompts = [processed_messages.prompt_ids]

        return (
            messages,
            request_prompts,
            engine_prompts,
            processed_messages,
            require_reasoning,
        )

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ):
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in " "response API"
            )
        messages = self._construct_input_messages_with_harmony(request, prev_response)
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = prompt_token_ids
        return messages, [prompt_token_ids], [engine_prompt]

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *,
        require_reasoning: bool = False,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if created_time is None:
            created_time = int(time.time())

        try:
            async for _ in result_generator:
                pass
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        if self.use_harmony:
            assert isinstance(context, HarmonyContext)
            output = self._make_response_output_items_with_harmony(context)
            # TODO: these are all 0 for now!
            num_prompt_tokens = context.num_prompt_tokens
            num_generated_tokens = context.num_output_tokens
            num_cached_tokens = context.num_cached_tokens
            num_reasoning_tokens = context.num_reasoning_tokens
        else:
            assert isinstance(context, SimpleContext)
            final_res = context.last_output
            assert final_res is not None

            output = self._make_response_output_items(
                request,
                final_res["text"],
                tokenizer,
                require_reasoning=require_reasoning,
            )

            # Calculate usage from actual output
            meta_info = (
                final_res.get("meta_info") if isinstance(final_res, dict) else None
            )
            if isinstance(meta_info, dict):
                num_prompt_tokens = meta_info.get("prompt_tokens", 0)
                num_generated_tokens = meta_info.get("completion_tokens", 0)
                num_cached_tokens = meta_info.get("cached_tokens", 0)
                num_reasoning_tokens = meta_info.get("reasoning_tokens", 0)
            elif hasattr(final_res, "prompt_token_ids") and hasattr(
                final_res, "outputs"
            ):
                # Fallback calculation if meta_info not available
                num_prompt_tokens = (
                    len(final_res.prompt_token_ids) if final_res.prompt_token_ids else 0
                )
                num_generated_tokens = (
                    len(final_res.outputs[0].token_ids)
                    if final_res.outputs and final_res.outputs[0].token_ids
                    else 0
                )
                num_cached_tokens = getattr(final_res, "num_cached_tokens", 0)
                num_reasoning_tokens = 0
            else:
                # Final fallback
                num_prompt_tokens = 0
                num_generated_tokens = 0
                num_cached_tokens = 0
                num_reasoning_tokens = 0

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            reasoning_tokens=num_reasoning_tokens,
        )
        if self.enable_prompt_tokens_details and num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=num_cached_tokens
            )
        request_metadata.final_usage_info = usage

        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status="completed",
            usage=usage,
        )

        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response

        return response

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: Any,
        tokenizer: Any,
        *,
        require_reasoning: bool = False,
    ):
        if self.reasoning_parser:
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=require_reasoning,
                request=request,
            )
            reasoning_content, content = reasoning_parser.parse_non_stream(final_output)
        else:
            reasoning_content = None
            content = final_output

        output_items = []
        if reasoning_content:
            wants_summary = (
                request.reasoning is not None and request.reasoning.summary is not None
            )
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                type="reasoning",
                summary=(
                    [
                        ResponseReasoningSummary(
                            type="summary_text", text=reasoning_content
                        )
                    ]
                    if wants_summary
                    else []
                ),
                content=[
                    ResponseReasoningTextContent(
                        type="reasoning_text", text=reasoning_content
                    ),
                ],
                status=None,
            )
            output_items.append(reasoning_item)

        tool_call_items = []
        chat_tools = self._response_tools_to_chat_tools(request)
        if (
            content
            and chat_tools
            and self.tool_call_parser
            and request.tool_choice != "none"
        ):
            tool_choice = self._chat_tool_choice(request.tool_choice)
            try:
                if tool_choice == "required" or isinstance(tool_choice, ToolChoice):
                    tool_call_data = orjson.loads(content)
                    calls = [
                        ToolCallItem(
                            tool_index=index,
                            name=item["name"],
                            parameters=json.dumps(
                                item["parameters"], ensure_ascii=False
                            ),
                        )
                        for index, item in enumerate(tool_call_data)
                    ]
                    content = ""
                else:
                    parser = FunctionCallParser(
                        chat_tools,
                        self.tool_call_parser,
                        tool_choice=tool_choice,
                    )
                    if not parser.has_tool_call(content):
                        calls = []
                    else:
                        content, calls = parser.parse_non_stream(content)
                for call in calls:
                    tool_call_items.append(
                        ResponseFunctionToolCall(
                            arguments=call.parameters or "",
                            call_id=f"call_{random_uuid()[:24]}",
                            type="function_call",
                            name=call.name,
                            id=f"fc_{random_uuid()[:8]}",
                            status="completed",
                        )
                    )
            except (KeyError, TypeError, ValueError, orjson.JSONDecodeError) as error:
                logger.warning("Responses tool-call parsing failed: %s", error)

        if content and content.strip():
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            output_items.append(message)
        output_items.extend(tool_call_items)
        return output_items

    def _make_response_output_items_with_harmony(
        self,
        context: HarmonyContext,
    ):
        output_items = []
        num_init_messages = context.num_init_messages
        for msg in context.messages[num_init_messages:]:
            output_items.extend(parse_output_message(msg))
        # Handle the generation stopped in the middle (if any).
        last_items = parse_remaining_state(context.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items

    @staticmethod
    def _normalize_response_content_part_for_chat(content_part: Any) -> Any:
        if hasattr(content_part, "model_dump"):
            content_part = content_part.model_dump(exclude_none=True)
        if not isinstance(content_part, dict):
            return content_part
        if content_part.get("type") in ("input_text", "output_text"):
            return {"type": "text", "text": content_part.get("text", "")}
        if content_part.get("type") == "input_image":
            image_url = content_part.get("image_url")
            if isinstance(image_url, str):
                image_url = {
                    "url": image_url,
                    "detail": content_part.get("detail", "auto"),
                }
            return {"type": "image_url", "image_url": image_url}
        return content_part

    @classmethod
    def _normalize_response_message_for_chat(cls, message: Any) -> Any:
        if hasattr(message, "model_dump"):
            message = message.model_dump(exclude_none=True)
        if not isinstance(message, dict):
            raise ValueError(f"Unsupported Responses input item: {message!r}")

        message_type = message.get("type")
        if message_type == "function_call":
            arguments = message.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = "{}"
            else:
                try:
                    parsed_arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    arguments = "{}"
                    parsed_arguments = {}
                if not isinstance(parsed_arguments, dict):
                    arguments = "{}"
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": message.get("call_id") or message.get("id"),
                        "type": "function",
                        "function": {
                            "name": message.get("name"),
                            "arguments": arguments,
                        },
                    }
                ],
            }

        if message_type == "function_call_output":
            output = message.get("output", "")
            if isinstance(output, list):
                output = "".join(
                    part.get("text", "") for part in output if isinstance(part, dict)
                )
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            return {
                "role": "tool",
                "tool_call_id": message.get("call_id"),
                "content": output,
            }

        if message_type == "reasoning":
            text_parts = []
            for part in message.get("summary") or message.get("content") or []:
                if hasattr(part, "model_dump"):
                    part = part.model_dump(exclude_none=True)
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(part["text"])
            if not text_parts:
                return None
            return {
                "role": "assistant",
                "reasoning_content": "\n".join(text_parts),
                "content": "",
            }

        if message_type not in (None, "message"):
            raise ValueError(
                f"Unsupported Responses API input item type: {message_type!r}"
            )

        content = message.get("content")
        normalized = {
            key: value
            for key, value in message.items()
            if value is not None and key not in ("id", "status", "type")
        }
        if isinstance(content, list):
            normalized_parts = [
                cls._normalize_response_content_part_for_chat(part) for part in content
            ]
            if normalized.get("role") in ("assistant", "developer", "system"):
                normalized["content"] = "\n\n".join(
                    part.get("text", "")
                    for part in normalized_parts
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                normalized["content"] = normalized_parts
        return normalized

    @staticmethod
    def _merge_consecutive_assistant_messages(messages: list) -> list:
        merged = []
        for message in messages:
            if (
                message.get("role") == "assistant"
                and merged
                and merged[-1].get("role") == "assistant"
            ):
                previous = merged[-1]
                if message.get("reasoning_content"):
                    previous_reasoning = previous.get("reasoning_content", "")
                    previous["reasoning_content"] = "\n".join(
                        part
                        for part in (
                            previous_reasoning,
                            message["reasoning_content"],
                        )
                        if part
                    )
                if message.get("content"):
                    previous_content = previous.get("content", "")
                    current_content = message["content"]
                    if isinstance(previous_content, str) and isinstance(
                        current_content, str
                    ):
                        previous["content"] = "\n\n".join(
                            part for part in (previous_content, current_content) if part
                        )
                    else:
                        previous_parts = (
                            previous_content
                            if isinstance(previous_content, list)
                            else (
                                [{"type": "text", "text": previous_content}]
                                if previous_content
                                else []
                            )
                        )
                        current_parts = (
                            current_content
                            if isinstance(current_content, list)
                            else [{"type": "text", "text": current_content}]
                        )
                        previous["content"] = previous_parts + current_parts
                if message.get("tool_calls"):
                    previous.setdefault("tool_calls", []).extend(message["tool_calls"])
                continue
            merged.append(message)
        return merged

    def _construct_input_messages(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse] = None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if request.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": request.instructions,
                }
            )

        # Prepend the conversation history
        if prev_response is not None:
            # Add the previous messages
            prev_msg = self.msg_store[prev_response.id]
            messages.extend(prev_msg)

            for output_item in prev_response.output:
                normalized = self._normalize_response_message_for_chat(output_item)
                if normalized is not None:
                    messages.append(normalized)

        # Append the new input
        # Responses API supports simple text inputs without chat format
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            for input_item in request.input:
                normalized = self._normalize_response_message_for_chat(input_item)
                if normalized is not None:
                    messages.append(normalized)
        return self._merge_consecutive_assistant_messages(messages)

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list["OpenAIMessage"]:
        messages: list["OpenAIMessage"] = []
        if prev_response is None:
            # New conversation.
            reasoning_effort = request.reasoning.effort if request.reasoning else None
            tool_types = [tool.type for tool in request.tools]
            enable_browser = (
                "web_search_preview" in tool_types and self.tool_server is not None
            )
            enable_code_interpreter = (
                "code_interpreter" in tool_types and self.tool_server is not None
            )
            sys_msg = get_system_message(
                reasoning_effort=reasoning_effort,
                browser_description=(
                    self.tool_server.get_tool_description("browser")
                    if self.tool_server and enable_browser
                    else None
                ),
                python_description=(
                    self.tool_server.get_tool_description("python")
                    if self.tool_server and enable_code_interpreter
                    else None
                ),
            )
            messages.append(sys_msg)
            dev_msg = get_developer_message(request.instructions, request.tools)
            messages.append(dev_msg)
        else:
            # Continue the previous conversation.
            # FIXME: Currently, request params like reasoning and
            # instructions are ignored.
            prev_msgs = self.msg_store[prev_response.id]
            # Remove the previous chain-of-thoughts if there is a new "final"
            # message.
            if (
                len(prev_msgs) > 0
                and hasattr(prev_msgs[-1], "channel")
                and prev_msgs[-1].channel == "final"
            ):  # type: ignore[union-attr]
                prev_final_msg_idx = -1
                for i in range(len(prev_msgs) - 2, -1, -1):
                    if (
                        hasattr(prev_msgs[i], "channel")
                        and prev_msgs[i].channel == "final"
                    ):  # type: ignore[union-attr]
                        prev_final_msg_idx = i
                        break
                recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                del prev_msgs[prev_final_msg_idx + 1 :]
                for msg in recent_turn_msgs:
                    if (
                        hasattr(msg, "channel") and msg.channel != "analysis"
                    ):  # type: ignore[union-attr]
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            messages.append(get_user_message(request.input))
        else:
            if prev_response is not None:
                prev_outputs = copy(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request.input:
                messages.append(parse_response_input(response_msg, prev_outputs))
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    async def _run_background_request(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *args,
        **kwargs,
    ):
        try:
            # Update the status to "in_progress"
            async with self.response_store_lock:
                stored_response = self.response_store.get(request.request_id)
                assert stored_response is not None
                stored_response.status = "in_progress"

            response = await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
                created_time,
                *args,
                **kwargs,
            )
        except Exception as e:
            logger.exception("Background request failed for %s", request.request_id)
            response = self.create_error_response(str(e))

        if isinstance(response, ORJSONResponse):
            # If the request has failed, update the status to "failed"
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in ("completed", "cancelled"):
                    stored_response.status = "failed"

    async def retrieve_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)
        return response

    async def cancel_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._make_not_found_error(response_id)

            prev_status = response.status
            if prev_status not in ("queued", "in_progress"):
                return self.create_error_response(
                    err_type="invalid_request_error",
                    message="Cannot cancel a synchronous response.",
                )

            # Update the status to "cancelled"
            response.status = "cancelled"

        # The response_id is the same as the rid used when submitting the request
        self.tokenizer_manager.abort_request(rid=response_id)

        if task := self.background_tasks.get(response_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled", response_id)
        return response

    def _make_invalid_id_error(self, response_id: str):
        return self.create_error_response(
            message=(
                f"Invalid 'response_id': '{response_id}'. "
                "Expected an ID that begins with 'resp'."
            ),
            err_type="invalid_request_error",
            param="response_id",
        )

    def _make_not_found_error(self, response_id: str):
        return self.create_error_response(
            message=f"Response with id '{response_id}' not found.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.NOT_FOUND,
            param="response_id",
        )

    async def responses_stream_generator_non_harmony(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *,
        require_reasoning: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream typed Responses API events for non-Harmony models."""

        created_time = created_time or int(time.time())
        sequence_number = 0

        def emit(event_type: str, **payload: Any) -> str:
            nonlocal sequence_number
            event = _response_event(event_type, sequence_number, **payload)
            sequence_number += 1
            return event

        initial_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump(mode="json")
        # The pinned OpenAI SDK has a narrower response-side Tool union than
        # the request schema. Tool echoing is optional, so avoid rejecting
        # namespace/web_search objects after accepting them on input.
        initial_response["tools"] = []
        yield emit("response.created", response=initial_response)
        yield emit("response.in_progress", response=initial_response)

        reasoning_parser = (
            ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=True,
                force_reasoning=require_reasoning,
                request=request,
            )
            if self.reasoning_parser
            else None
        )
        chat_tools = self._response_tools_to_chat_tools(request)
        chat_tool_choice = self._chat_tool_choice(request.tool_choice)
        tool_parser = None
        if chat_tools and self.tool_call_parser and request.tool_choice != "none":
            if chat_tool_choice == "required" or isinstance(
                chat_tool_choice, ToolChoice
            ):
                tool_parser = JsonArrayParser()
            else:
                tool_parser = FunctionCallParser(
                    chat_tools,
                    self.tool_call_parser,
                    tool_choice=chat_tool_choice,
                )

        output_index = -1
        emitted_items = []
        reasoning_state = {
            "open": False,
            "id": "",
            "index": -1,
            "text": "",
        }
        message_state = {
            "open": False,
            "id": "",
            "index": -1,
            "text": "",
        }
        tool_states: dict[int, dict[str, Any]] = {}
        wants_summary = (
            request.reasoning is not None and request.reasoning.summary is not None
        )

        def open_reasoning() -> list[str]:
            nonlocal output_index
            if reasoning_state["open"]:
                return []
            output_index += 1
            reasoning_state.update(
                open=True,
                id=f"rs_{random_uuid()}",
                index=output_index,
                text="",
            )
            item = {
                "id": reasoning_state["id"],
                "type": "reasoning",
                "summary": [],
                "content": [],
                "status": "in_progress",
            }
            events = [
                emit(
                    "response.output_item.added",
                    output_index=reasoning_state["index"],
                    item=item,
                )
            ]
            if wants_summary:
                events.append(
                    emit(
                        "response.reasoning_summary_part.added",
                        item_id=reasoning_state["id"],
                        output_index=reasoning_state["index"],
                        summary_index=0,
                        part={"type": "summary_text", "text": ""},
                    )
                )
            return events

        def close_reasoning() -> list[str]:
            if not reasoning_state["open"]:
                return []
            text = reasoning_state["text"]
            item = {
                "id": reasoning_state["id"],
                "type": "reasoning",
                "summary": (
                    [{"type": "summary_text", "text": text}] if wants_summary else []
                ),
                "content": [{"type": "reasoning_text", "text": text}],
                "status": "completed",
            }
            if wants_summary:
                events = [
                    emit(
                        "response.reasoning_summary_text.done",
                        item_id=reasoning_state["id"],
                        output_index=reasoning_state["index"],
                        summary_index=0,
                        text=text,
                    ),
                    emit(
                        "response.reasoning_summary_part.done",
                        item_id=reasoning_state["id"],
                        output_index=reasoning_state["index"],
                        summary_index=0,
                        part={"type": "summary_text", "text": text},
                    ),
                ]
            else:
                events = [
                    emit(
                        "response.reasoning_text.done",
                        item_id=reasoning_state["id"],
                        output_index=reasoning_state["index"],
                        content_index=0,
                        text=text,
                    )
                ]
            events.append(
                emit(
                    "response.output_item.done",
                    output_index=reasoning_state["index"],
                    item=item,
                )
            )
            emitted_items.append(item)
            reasoning_state["open"] = False
            return events

        def open_message() -> list[str]:
            nonlocal output_index
            if message_state["open"]:
                return []
            output_index += 1
            message_state.update(
                open=True,
                id=f"msg_{random_uuid()}",
                index=output_index,
                text="",
            )
            return [
                emit(
                    "response.output_item.added",
                    output_index=message_state["index"],
                    item={
                        "id": message_state["id"],
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress",
                    },
                ),
                emit(
                    "response.content_part.added",
                    item_id=message_state["id"],
                    output_index=message_state["index"],
                    content_index=0,
                    part={
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                        "logprobs": None,
                    },
                ),
            ]

        def close_message() -> list[str]:
            if not message_state["open"]:
                return []
            text = message_state["text"]
            part = {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": None,
            }
            item = {
                "id": message_state["id"],
                "type": "message",
                "role": "assistant",
                "content": [part],
                "status": "completed",
            }
            events = [
                emit(
                    "response.output_text.done",
                    item_id=message_state["id"],
                    output_index=message_state["index"],
                    content_index=0,
                    text=text,
                    logprobs=[],
                ),
                emit(
                    "response.content_part.done",
                    item_id=message_state["id"],
                    output_index=message_state["index"],
                    content_index=0,
                    part=part,
                ),
                emit(
                    "response.output_item.done",
                    output_index=message_state["index"],
                    item=item,
                ),
            ]
            emitted_items.append(item)
            message_state["open"] = False
            return events

        def open_tool(tool_index: int, name: str) -> list[str]:
            nonlocal output_index
            if tool_index in tool_states:
                if name:
                    tool_states[tool_index]["name"] = name
                return []
            output_index += 1
            state = {
                "id": f"fc_{random_uuid()[:8]}",
                "call_id": f"call_{random_uuid()[:24]}",
                "name": name,
                "arguments": "",
                "index": output_index,
            }
            tool_states[tool_index] = state
            return [
                emit(
                    "response.output_item.added",
                    output_index=state["index"],
                    item={
                        "id": state["id"],
                        "call_id": state["call_id"],
                        "type": "function_call",
                        "name": state["name"],
                        "arguments": "",
                        "status": "in_progress",
                    },
                )
            ]

        last_output = None
        raw_text_buffer = ""
        try:
            async for context in result_generator:
                last_output = context.last_output
                if not isinstance(last_output, dict):
                    continue
                raw_text = last_output.get("text", "")
                # SGLang's generation stream contains the cumulative decoded
                # text. Feed only the new suffix to the reasoning/tool parsers.
                delta = raw_text[len(raw_text_buffer) :]
                raw_text_buffer = raw_text
                if not delta:
                    continue

                if reasoning_parser is not None:
                    reasoning_delta, normal_delta = reasoning_parser.parse_stream_chunk(
                        delta
                    )
                else:
                    reasoning_delta, normal_delta = None, delta

                if reasoning_delta:
                    for event in open_reasoning():
                        yield event
                    reasoning_state["text"] += reasoning_delta
                    event_type = (
                        "response.reasoning_summary_text.delta"
                        if wants_summary
                        else "response.reasoning_text.delta"
                    )
                    payload = {
                        "item_id": reasoning_state["id"],
                        "output_index": reasoning_state["index"],
                        "delta": reasoning_delta,
                    }
                    if wants_summary:
                        payload["summary_index"] = 0
                    else:
                        payload["content_index"] = 0
                    yield emit(event_type, **payload)

                if normal_delta:
                    for event in close_reasoning():
                        yield event
                    if isinstance(tool_parser, JsonArrayParser):
                        parse_result = tool_parser.parse_streaming_increment(
                            normal_delta, chat_tools
                        )
                        visible_text, calls = (
                            parse_result.normal_text,
                            parse_result.calls,
                        )
                    elif tool_parser is not None:
                        visible_text, calls = tool_parser.parse_stream_chunk(
                            normal_delta
                        )
                    else:
                        visible_text, calls = normal_delta, []

                    if visible_text and visible_text.strip():
                        for event in open_message():
                            yield event
                        message_state["text"] += visible_text
                        yield emit(
                            "response.output_text.delta",
                            item_id=message_state["id"],
                            output_index=message_state["index"],
                            content_index=0,
                            delta=visible_text,
                            logprobs=[],
                        )

                    if calls:
                        for event in close_message():
                            yield event
                        for call in calls:
                            for event in open_tool(call.tool_index, call.name or ""):
                                yield event
                            state = tool_states[call.tool_index]
                            if call.parameters:
                                state["arguments"] += call.parameters
                                yield emit(
                                    "response.function_call_arguments.delta",
                                    item_id=state["id"],
                                    output_index=state["index"],
                                    delta=call.parameters,
                                )
        except asyncio.CancelledError:
            return
        except ValueError as error:
            yield emit(
                "error",
                error={
                    "message": str(error),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": 400,
                },
            )
            return

        for event in close_reasoning():
            yield event
        for event in close_message():
            yield event
        for tool_index in sorted(tool_states):
            state = tool_states[tool_index]
            item = {
                "id": state["id"],
                "call_id": state["call_id"],
                "type": "function_call",
                "name": state["name"],
                "arguments": state["arguments"],
                "status": "completed",
            }
            yield emit(
                "response.function_call_arguments.done",
                item_id=state["id"],
                output_index=state["index"],
                arguments=state["arguments"],
            )
            yield emit(
                "response.output_item.done",
                output_index=state["index"],
                item=item,
            )
            emitted_items.append(item)

        meta_info = (
            last_output.get("meta_info", {}) if isinstance(last_output, dict) else {}
        )
        usage = UsageInfo(
            prompt_tokens=meta_info.get("prompt_tokens", 0),
            completion_tokens=meta_info.get("completion_tokens", 0),
            total_tokens=(
                meta_info.get("prompt_tokens", 0)
                + meta_info.get("completion_tokens", 0)
            ),
            reasoning_tokens=meta_info.get("reasoning_tokens", 0),
        )
        request_metadata.final_usage_info = usage
        completed_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=emitted_items,
            status="completed",
            usage=usage,
        ).model_dump(mode="json")
        completed_response["tools"] = []
        yield emit("response.completed", response=completed_response)

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[StreamingHarmonyContext],
        context: StreamingHarmonyContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        # TODO:
        # 1. Handle disconnect

        created_time = created_time or int(time.time())

        sequence_number = 0

        def _send_event(event):
            nonlocal sequence_number
            # Set sequence_number if the event has this attribute
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            # Get event type from the event's type field if it exists
            event_type = getattr(event, "type", "unknown")
            return (
                f"event: {event_type}\n"
                f"data: {event.model_dump_json(indent=None)}\n\n"
            )

        current_content_index = 0
        current_output_index = 0
        current_item_id = f"item_{random_uuid()}"
        sent_output_item_added = False

        initial_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump()
        yield _send_event(
            openai_responses_types.ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            )
        )
        yield _send_event(
            openai_responses_types.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            )
        )

        async for ctx in result_generator:

            # Only process context objects that implement the `is_expecting_start()` method,
            # which indicates they support per-turn streaming (e.g., StreamingHarmonyContext).
            # Contexts without this method are skipped, as they do not represent a new turn
            # or are not compatible with per-turn handling in the /v1/responses endpoint.
            if not hasattr(ctx, "is_expecting_start"):
                continue

            if ctx.is_expecting_start():
                current_output_index += 1
                sent_output_item_added = False

                if len(ctx.parser.messages) > 0:
                    previous_item = ctx.parser.messages[-1]
                    if previous_item.recipient is not None:
                        # Deal with tool call here
                        pass
                    elif previous_item.channel == "analysis":
                        reasoning_item = ResponseReasoningItem(
                            id=f"rs_{random_uuid()}",
                            type="reasoning",
                            summary=[],
                            content=[
                                ResponseReasoningTextContent(
                                    text=previous_item.content[0].text,
                                    type="reasoning_text",
                                ),
                            ],
                            status="completed",
                        )
                        yield _send_event(
                            openai_responses_types.ResponseReasoningTextDoneEvent(
                                type="response.reasoning_text.done",
                                item_id=current_item_id,
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=reasoning_item,
                            )
                        )
                    elif previous_item.channel == "final":
                        text_content = openai_responses_types.ResponseOutputText(
                            type="output_text",
                            text=previous_item.content[0].text,
                            annotations=[],
                        )
                        yield _send_event(
                            openai_responses_types.ResponseTextDoneEvent(
                                type="response.output_text.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                                logprobs=[],
                                item_id=current_item_id,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartDoneEvent(
                                type="response.content_part.done",
                                sequence_number=-1,
                                item_id=current_item_id,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=text_content,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[text_content],
                                    status="completed",
                                ),
                            )
                        )

            if ctx.parser.last_content_delta:
                if (
                    ctx.parser.current_channel == "final"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    yield _send_event(
                        openai_responses_types.ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=ctx.parser.last_content_delta,
                            # TODO, use logprobs from ctx.last_request_output
                            logprobs=[],
                        )
                    )
                elif (
                    ctx.parser.current_channel == "analysis"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseReasoningItem(
                                    type="reasoning",
                                    id=current_item_id,
                                    summary=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                # TODO: migrate this to
                                # ResponseReasoningTextContent for now
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    # TODO: migrate to OpenAI types once updated.
                    yield _send_event(
                        openai_responses_types.ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            item_id=current_item_id,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            delta=ctx.parser.last_content_delta,
                            sequence_number=-1,
                        )
                    )

            if ctx.is_assistant_action_turn() and len(ctx.parser.messages) > 0:
                previous_item = ctx.parser.messages[-1]
                if (
                    self.supports_browsing
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("browser.")
                ):
                    function_name = previous_item.recipient[len("browser.") :]
                    action = None
                    parsed_args = orjson.loads(previous_item.content[0].text)
                    if function_name == "search":
                        action = openai_responses_types.response_function_web_search.ActionSearch(
                            type="search",
                            query=parsed_args["query"],
                        )
                    elif function_name == "open":
                        action = openai_responses_types.response_function_web_search.ActionOpenPage(
                            type="open_page",
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    elif function_name == "find":
                        action = openai_responses_types.response_function_web_search.ActionFind(
                            type="find",
                            pattern=parsed_args["pattern"],
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    else:
                        raise ValueError(f"Unknown function name: {function_name}")

                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.response_function_web_search.ResponseFunctionWebSearch(
                                # TODO: generate a unique id for web search call
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallInProgressEvent(
                            type="response.web_search_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallSearchingEvent(
                            type="response.web_search_call.searching",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )

                    # enqueue
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallCompletedEvent(
                            type="response.web_search_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseFunctionWebSearch(
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="completed",
                            ),
                        )
                    )

                if (
                    self.supports_code_interpreter
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("python")
                ):
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code="",
                                container_id="auto",
                                outputs=[],
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInProgressEvent(
                            type="response.code_interpreter_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    # TODO: do we need to add delta event here?
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCodeDoneEvent(
                            type="response.code_interpreter_call_code.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            code=previous_item.content[0].text,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInterpretingEvent(
                            type="response.code_interpreter_call.interpreting",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCompletedEvent(
                            type="response.code_interpreter_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code=previous_item.content[0].text,
                                container_id="auto",
                                # TODO: add outputs here
                                outputs=[],
                                status="completed",
                            ),
                        )
                    )

        async def empty_async_generator():
            if False:
                yield

        final_response = await self.responses_full_generator(
            request,
            sampling_params,
            empty_async_generator(),
            context,
            model_name,
            tokenizer,
            request_metadata,
            created_time=created_time,
        )
        # Convert final_response to the format expected by ResponseCompletedEvent
        response_dict = final_response.model_dump()

        yield _send_event(
            openai_responses_types.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=response_dict,
            )
        )

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        request_prompt: Any,
        adapted_request: GenerateReqInput,
        sampling_params: Any,
        context: ConversationContext,
        raw_request: Optional[Request] = None,
        priority: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Generate with builtin tool support for harmony-based models."""
        orig_priority = priority or 0

        while True:
            # Generate using SGLang's tokenizer manager
            generator = self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            )

            async for res in generator:
                context.append_output(res)
                # NOTE(woosuk): The stop condition is handled by the engine.
                yield context

            if not context.need_builtin_tool_call():
                # The model did not ask for a tool call, so we're done.
                break

            # Call the tool and update the context with the result.
            tool_output = await context.call_tool()
            context.append_output(tool_output)

            # Prepare for the next generation turn
            # Render the updated conversation for the next completion
            prompt_token_ids = context.render_for_completion()

            # Update the adapted request with new prompt
            adapted_request = GenerateReqInput(
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
                stream=adapted_request.stream,
                rid=request_id,
                extra_key=adapted_request.extra_key,
                return_logprob=adapted_request.return_logprob,
                logprob_start_len=adapted_request.logprob_start_len,
                top_logprobs_num=adapted_request.top_logprobs_num,
                return_text_in_logprobs=adapted_request.return_text_in_logprobs,
                return_hidden_states=adapted_request.return_hidden_states,
                background=adapted_request.background,
            )

            # Update sampling params with reduced max_tokens
            if hasattr(sampling_params, "max_new_tokens") or isinstance(
                sampling_params, dict
            ):
                context_len = getattr(
                    self.tokenizer_manager.model_config, "context_len", 4096
                )
                remaining_tokens = context_len - len(prompt_token_ids) - 1

                if isinstance(sampling_params, dict):
                    sampling_params["max_new_tokens"] = max(remaining_tokens, 1)
                else:
                    sampling_params.max_new_tokens = max(remaining_tokens, 1)

            # Slightly reduce priority for subsequent tool calls
            priority = orig_priority - 1
