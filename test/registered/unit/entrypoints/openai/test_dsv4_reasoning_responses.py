import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from openai.types.responses import Response as OpenAIResponse

from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv4
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import (
    resolve_dsv4_reasoning_controls,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses


class TestDSV4ReasoningAndResponses(unittest.TestCase):
    def test_dsv4_reasoning_profile_detection_and_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            encoding_dir = Path(temp_dir) / "encoding"
            encoding_dir.mkdir()
            (encoding_dir / "encoding_dsv4.py").write_text(
                "DEFAULT_REASONING_EFFORT = 'low'\n"
                "REASONING_EFFORT_PROMPTS = "
                "{'low': '', 'high': 'high', 'max': 'max'}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                chat_encoding.resolve_dsv4_reasoning_effort_profile(
                    model_path=temp_dir
                ),
                "official",
            )
        self.assertEqual(
            chat_encoding.resolve_dsv4_reasoning_effort_profile(
                model_path="unused", override="preview"
            ),
            "preview",
        )
        with self.assertRaises(ValueError):
            chat_encoding.resolve_dsv4_reasoning_effort_profile(
                model_path="unused", override="invalid"
            )

    def test_dsv4_reasoning_resolution(self):
        cases = [
            (None, "thinking", "max"),
            ("low", "thinking", "low"),
            ("high", "thinking", "high"),
            ("max", "thinking", "max"),
            ("none", "chat", None),
            ("medium", "thinking", "max"),
            ("xhigh", "thinking", "max"),
            ("invalid", "thinking", "max"),
        ]
        for effort, thinking_mode, resolved in cases:
            with self.subTest(effort=effort):
                if effort in {"medium", "xhigh", "invalid"}:
                    with self.assertLogs(
                        "sglang.srt.entrypoints.openai.serving_chat", level="WARNING"
                    ) as logs:
                        result = resolve_dsv4_reasoning_controls(
                            request_effort=effort,
                            chat_template_kwargs=None,
                            reasoning_effort_profile="official",
                        )
                    self.assertIn(
                        "falling back to thinking + max", "\n".join(logs.output)
                    )
                else:
                    result = resolve_dsv4_reasoning_controls(
                        request_effort=effort,
                        chat_template_kwargs=None,
                        reasoning_effort_profile="official",
                    )
                self.assertEqual(result, (thinking_mode, resolved))

    def test_standard_effort_wins_and_legacy_thinking_can_disable(self):
        self.assertEqual(
            resolve_dsv4_reasoning_controls(
                request_effort="high",
                chat_template_kwargs={"thinking": False, "reasoning_effort": "low"},
                reasoning_effort_profile="official",
            ),
            ("thinking", "high"),
        )
        self.assertEqual(
            resolve_dsv4_reasoning_controls(
                request_effort=None,
                chat_template_kwargs={"thinking": False, "reasoning_effort": "low"},
                reasoning_effort_profile="official",
            ),
            ("chat", "low"),
        )

    def test_official_prompt_mapping_is_distinct(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
        prompts = {
            effort: encoding_dsv4.encode_messages(
                messages,
                thinking_mode="thinking",
                reasoning_effort=effort,
                reasoning_effort_profile="official",
            )
            for effort in ("low", "high", "max")
        }
        self.assertNotIn(encoding_dsv4.REASONING_EFFORT_PREVIEW_MAX, prompts["low"])
        self.assertNotIn(encoding_dsv4.REASONING_EFFORT_OFFICIAL_MAX, prompts["low"])
        self.assertIn(encoding_dsv4.REASONING_EFFORT_PREVIEW_MAX, prompts["high"])
        self.assertNotIn(encoding_dsv4.REASONING_EFFORT_OFFICIAL_MAX, prompts["high"])
        self.assertIn(encoding_dsv4.REASONING_EFFORT_OFFICIAL_MAX, prompts["max"])

    def test_protocol_accepts_standard_max_and_codex_tool_shapes(self):
        chat = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            reasoning_effort="max",
        )
        self.assertEqual(chat.reasoning_effort, "max")

        response = ResponsesRequest(
            input=[{"role": "user", "content": "hello"}],
            reasoning={"effort": "max", "summary": "auto"},
            include=["reasoning.encrypted_content"],
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
                {
                    "type": "namespace",
                    "name": "functions",
                    "description": "Namespaced tools",
                    "tools": [{"type": "function", "name": "nested"}],
                },
                {"type": "web_search"},
            ],
            store=False,
            stream=True,
        )
        self.assertEqual(response.reasoning.effort, "max")
        self.assertEqual(response.reasoning.summary, "auto")
        self.assertEqual(
            [tool.type for tool in response.tools],
            ["function", "namespace", "web_search"],
        )

        response_payload = ResponsesResponse.from_request(
            response,
            {"max_new_tokens": 16},
            model_name="dsv4",
            created_time=1,
            output=[],
            status="completed",
            usage=UsageInfo(
                prompt_tokens=3,
                completion_tokens=4,
                reasoning_tokens=2,
                total_tokens=7,
            ),
        ).model_dump(mode="json")
        self.assertEqual(response_payload["usage"]["input_tokens"], 3)
        self.assertEqual(response_payload["usage"]["output_tokens"], 4)
        self.assertEqual(
            response_payload["usage"]["output_tokens_details"]["reasoning_tokens"],
            2,
        )

    def test_responses_normalizes_function_round_trip(self):
        request = ResponsesRequest(
            input=[
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "/tmp",
                },
            ]
        )
        serving = object.__new__(OpenAIServingResponses)
        messages = serving._construct_input_messages(request)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["name"], "exec_command"
        )
        self.assertEqual(
            messages[1],
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "/tmp",
            },
        )

        retry_request = ResponsesRequest(
            input=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "first"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "second"}],
                },
            ]
        )
        retry_messages = serving._construct_input_messages(retry_request)
        self.assertEqual(retry_messages[0]["content"], "first\n\nsecond")

    def test_non_stream_output_parses_reasoning_and_function_call(self):
        request = ResponsesRequest(
            input="use the tool",
            reasoning={"effort": "max", "summary": "auto"},
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
        )
        serving = object.__new__(OpenAIServingResponses)
        serving.reasoning_parser = "deepseek-v4"
        serving.tool_call_parser = "deepseekv4"
        output = serving._make_response_output_items(
            request,
            "reason first</think>"
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="exec_command">'
            '<｜DSML｜parameter name="cmd" string="true">pwd</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
            "</｜DSML｜tool_calls>",
            tokenizer=None,
            require_reasoning=True,
        )
        self.assertEqual([item.type for item in output], ["reasoning", "function_call"])
        self.assertEqual(output[0].summary[0].text, "reason first")
        self.assertEqual(output[1].name, "exec_command")
        self.assertEqual(json.loads(output[1].arguments), {"cmd": "pwd"})

    def test_non_stream_output_parses_named_function_choice(self):
        request = ResponsesRequest(
            input="use the tool",
            tool_choice={"type": "function", "name": "exec_command"},
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
        )
        serving = object.__new__(OpenAIServingResponses)
        serving.reasoning_parser = None
        serving.tool_call_parser = "deepseekv4"
        output = serving._make_response_output_items(
            request,
            '[{"name":"exec_command","parameters":{"cmd":"pwd"}}]',
            tokenizer=None,
        )
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].type, "function_call")
        self.assertEqual(output[0].name, "exec_command")
        self.assertEqual(json.loads(output[0].arguments), {"cmd": "pwd"})


class TestDSV4ResponsesStream(unittest.IsolatedAsyncioTestCase):
    async def test_emits_reasoning_text_and_completion(self):
        request = ResponsesRequest(
            input="answer",
            reasoning={"effort": "max", "summary": "auto"},
            stream=True,
            store=False,
        )
        serving = object.__new__(OpenAIServingResponses)
        serving.reasoning_parser = "deepseek-v4"
        serving.tool_call_parser = None
        serving.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace())

        async def results():
            for text in ("reason", "reason</think>", "reason</think>answer"):
                yield SimpleNamespace(
                    last_output={
                        "text": text,
                        "meta_info": {"prompt_tokens": 3, "completion_tokens": 4},
                    }
                )

        events = []
        async for chunk in serving.responses_stream_generator_non_harmony(
            request,
            {"max_new_tokens": 16},
            results(),
            "dsv4",
            None,
            RequestResponseMetadata(request_id=request.request_id),
            require_reasoning=True,
        ):
            if chunk.startswith("event: "):
                events.append(chunk.splitlines()[0].removeprefix("event: "))

        self.assertEqual(events[0:2], ["response.created", "response.in_progress"])
        self.assertIn("response.reasoning_summary_text.delta", events)
        self.assertIn("response.output_text.delta", events)
        self.assertEqual(events.count("response.reasoning_summary_text.delta"), 1)
        self.assertEqual(events.count("response.output_text.delta"), 1)
        self.assertEqual(events[-1], "response.completed")

    async def test_emits_function_call_events(self):
        request = ResponsesRequest(
            input="use the tool",
            stream=True,
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
        )
        serving = object.__new__(OpenAIServingResponses)
        serving.reasoning_parser = None
        serving.tool_call_parser = "deepseekv4"
        serving.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace())
        context = SimpleNamespace(
            last_output={
                "text": "<｜DSML｜tool_calls>"
                '<｜DSML｜invoke name="exec_command">'
                '<｜DSML｜parameter name="cmd" string="true">pwd</｜DSML｜parameter>'
                "</｜DSML｜invoke>"
                "</｜DSML｜tool_calls>",
                "meta_info": {"prompt_tokens": 4, "completion_tokens": 5},
            }
        )

        async def results():
            yield context

        chunks = []
        async for chunk in serving.responses_stream_generator_non_harmony(
            request,
            {"max_new_tokens": 16},
            results(),
            "dsv4",
            None,
            RequestResponseMetadata(request_id=request.request_id),
        ):
            chunks.append(chunk)

        event_names = [chunk.splitlines()[0][7:] for chunk in chunks]
        self.assertIn("response.function_call_arguments.delta", event_names)
        self.assertIn("response.function_call_arguments.done", event_names)
        completed = json.loads(chunks[-1].split("data: ", 1)[1])
        OpenAIResponse.model_validate(completed["response"])
        call = completed["response"]["output"][0]
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(call["name"], "exec_command")
        self.assertEqual(json.loads(call["arguments"]), {"cmd": "pwd"})


if __name__ == "__main__":
    unittest.main()
