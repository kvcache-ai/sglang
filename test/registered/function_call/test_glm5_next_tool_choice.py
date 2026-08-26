import json
from pathlib import Path

import pytest

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(0.2, "default")


@pytest.fixture
def tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="read_file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="list_dir",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
        ),
    ]


def test_required_uses_native_glm_ebnf(tools):
    parser = FunctionCallParser(tools, "glm47", tool_choice="required")
    constraint_type, grammar = parser.get_structure_constraint("required")

    assert constraint_type == "ebnf"
    assert '"<tool_call>read_file"' in grammar
    assert '"<tool_call>list_dir"' in grammar
    assert '"<arg_key>path</arg_key>"' in grammar
    assert "root ::= tool_call+" in grammar
    assert '"name"' not in grammar


def test_named_choice_allows_exactly_the_selected_native_call(tools):
    choice = ToolChoice(function=ToolChoiceFuncName(name="read_file"))
    parser = FunctionCallParser(tools, "glm47", tool_choice=choice)
    constraint_type, grammar = parser.get_structure_constraint(choice)

    assert constraint_type == "ebnf"
    assert "root ::= tool_0_call" in grammar
    assert '"<tool_call>read_file"' in grammar
    assert "list_dir" not in grammar
    assert "tool_call+" not in grammar


def test_forced_non_stream_parser_keeps_glm_xml_and_validates_schema(tools):
    parser = FunctionCallParser(tools, "glm47", tool_choice="required")
    text, calls = parser.parse_non_stream(
        "<tool_call>read_file"
        "<arg_key>path</arg_key><arg_value>/tmp/a.py</arg_value>"
        "<arg_key>line</arg_key><arg_value>42</arg_value>"
        "</tool_call>"
    )

    assert text == ""
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert json.loads(calls[0].parameters) == {"path": "/tmp/a.py", "line": 42}

    with pytest.raises(ValueError, match="required property"):
        parser.parse_non_stream(
            "<tool_call>read_file"
            "<arg_key>line</arg_key><arg_value>42</arg_value>"
            "</tool_call>"
        )


def test_auto_streaming_still_uses_glm_detector(tools):
    parser = FunctionCallParser(tools, "glm47", tool_choice="auto")
    chunks = [
        "<tool_",
        "call>read_file<arg_key>path</arg_key>",
        "<arg_value>/tmp/a.py</arg_value></tool_call>",
    ]
    calls = []
    normal_text = ""
    for chunk in chunks:
        normal, parsed = parser.parse_stream_chunk(chunk)
        normal_text += normal
        calls.extend(parsed)

    assert normal_text == ""
    assert any(call.name == "read_file" for call in calls)
    argument_deltas = "".join(call.parameters for call in calls if call.name is None)
    assert "/tmp/a.py" in argument_deltas


def test_serving_keeps_glm_forced_calls_on_model_parser():
    serving_source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    ).read_text(encoding="utf-8")
    assert 'self.tool_call_parser != "glm47"' in serving_source
    assert "if tool_call_constraint is None" in serving_source
