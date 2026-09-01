from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from harn_agent.types import AgentTool, AgentToolResult
from harn_ai.types import TextContent
from test_budget import make_manifest

from aibb.harness import AibbHarnessEngine, build_context_envelope
from aibb.harness.openrouter import (
    MAX_TOOL_CALLS_PER_RESPONSE,
    OpenRouterAdapter,
    _add_anthropic_cache_breakpoints,
    _estimate_payload_tokens,
    _parse_tool_arguments,
    openrouter_model,
)
from aibb.harness.token_estimate import ESTIMATED_IMAGE_INPUT_TOKENS
from aibb.runtime import BudgetLedger
from aibb.sessions import SessionStore


@pytest.mark.parametrize(
    ("raw", "expected", "expected_repair"),
    [
        ('{"thread_id":"thread-one"}', {"thread_id": "thread-one"}, None),
        (
            '{"thread_id":"thread-one"}}',
            {"thread_id": "thread-one"},
            "removed_unmatched_trailing_closing_braces",
        ),
        (
            ' {"thread_id":"thread-one"} }} ',
            {"thread_id": "thread-one"},
            "removed_unmatched_trailing_closing_braces",
        ),
        ("{}{}", {}, "collapsed_repeated_identical_json_objects"),
        (
            '{"thread_id":"thread-one"} {"thread_id":"thread-one"}',
            {"thread_id": "thread-one"},
            "collapsed_repeated_identical_json_objects",
        ),
        ('{}""', {}, "removed_trailing_empty_json_strings"),
        ({"thread_id": "thread-one"}, {"thread_id": "thread-one"}, None),
        (None, {}, None),
    ],
)
def test_tool_argument_parser_only_repairs_uniquely_recoverable_json(
    raw: object, expected: dict[str, object], expected_repair: str | None
) -> None:
    parsed, repair = _parse_tool_arguments(raw)

    assert parsed == expected
    assert (repair or {}).get("repair") == expected_repair


@pytest.mark.parametrize(
    "raw",
    [
        '{"thread_id":}',
        '{"thread_id":"thread-one"} trailing prose',
        '{"thread_id":"thread-one"}{"thread_id":"thread-two"}',
        '["thread-one"]',
        '}{"thread_id":"thread-one"}',
    ],
)
def test_tool_argument_parser_refuses_ambiguous_or_non_object_repairs(raw: str) -> None:
    with pytest.raises(RuntimeError, match="Provider returned"):
        _parse_tool_arguments(raw)


def test_payload_estimate_counts_image_tokens_without_tokenizing_base64_bytes() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect the image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/webp;base64," + "a" * 500_000},
                    },
                ],
            }
        ]
    }

    estimate = _estimate_payload_tokens(payload)

    assert estimate >= ESTIMATED_IMAGE_INPUT_TOKENS
    assert estimate < ESTIMATED_IMAGE_INPUT_TOKENS + 1_000


def test_anthropic_cache_breakpoints_keep_fixed_opening_and_three_recent_boundaries() -> None:
    messages = [{"role": "user", "content": f"message-{index}"} for index in range(10)]

    _add_anthropic_cache_breakpoints(messages)

    marked = [
        index
        for index, message in enumerate(messages)
        if isinstance(message["content"], list) and "cache_control" in message["content"][-1]
    ]
    assert marked == [0, 7, 8, 9]


@pytest.mark.asyncio
async def test_openrouter_adapter_captures_payload_response_usage_and_tools(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer private-test-key"
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "reasoning": "I should inspect the archive status.",
                "reasoning_details": [
                    {
                        "type": "reasoning.encrypted",
                        "data": "opaque-provider-state",
                        "format": "openai-responses-v1",
                        "index": 0,
                    }
                ],
                "tool_calls": [
                    {
                        "id": "call-status",
                        "type": "function",
                        "function": {"name": "archive_status", "arguments": "{}"},
                    }
                ],
            }
            # Some OpenRouter routes return real tool calls while incorrectly labeling the turn "stop".
            finish_reason = "stop"
        else:
            message = {"role": "assistant", "content": "I found one durable record."}
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"response-{len(requests)}",
                "model": "openai/gpt-5.6-luna",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.00022},
            },
        )

    async def archive_status(
        _tool_call_id: str,
        _arguments: Any,
        _signal: Any = None,
        _on_update: Any = None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text='{"published": 1}')], details={"published": 1})

    tool = AgentTool(
        name="archive_status",
        label="Archive status",
        description="Read archive status.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute=archive_status,
        executionMode="sequential",
    )
    manifest = make_manifest()
    ledger = BudgetLedger(tmp_path / "mcp/budgets.json", manifest)
    session = SessionStore(tmp_path / "session", manifest.run_id)
    adapter = OpenRouterAdapter(
        api_key="private-test-key",
        ledger=ledger,
        session=session,
        max_output_tokens=500,
        prompt_price_per_token=0.000001,
        completion_price_per_token=0.000006,
        app_url="https://archive.example/",
        reasoning_parameter={"effort": "high", "exclude": False},
        provider_routing={
            "order": ["google-vertex"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        tool_choice="required",
        output_token_parameter="max_completion_tokens",
        transport=httpx.MockTransport(handler),
    )
    envelope = build_context_envelope(
        orientation_version="v0.1",
        orientation="Explore. Silence is valid.",
        notice_version="v0.1",
        notice="The record is public.",
        policy_version="v0.1",
        policy="Contribute only when it adds something.",
        run_scope='{"quota":1}',
        tool_definitions=[{"name": tool.name, "description": tool.description, "parameters": tool.parameters}],
    )
    engine = AibbHarnessEngine(
        model=openrouter_model(
            "openai/gpt-5.6-luna",
            context_window=1_050_000,
            max_tokens=500,
            prompt_price_per_token=0.000001,
            completion_price_per_token=0.000006,
        ),
        system_prompt="",
        messages=[envelope.initial_message()],
        tools=[tool],
        stream_fn=adapter,
    )

    await engine.send_curator_message("Welcome.")

    assert len(requests) == 2
    assert {request["session_id"] for request in requests} == {manifest.run_id}
    assert "cache_control" not in json.dumps(requests)
    assert requests[0]["messages"][0]["content"].startswith("Explore. Silence is valid.")
    assert requests[0]["reasoning"] == {"effort": "high", "exclude": False}
    assert requests[0]["provider"] == {
        "order": ["google-vertex"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["max_completion_tokens"] == 500
    assert "max_tokens" not in requests[0]
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-status",
        "content": '{"published": 1}',
    }
    assert requests[1]["messages"][-2]["reasoning_details"] == [
        {
            "type": "reasoning.encrypted",
            "data": "opaque-provider-state",
            "format": "openai-responses-v1",
            "index": 0,
        }
    ]
    assert "reasoning" not in requests[1]["messages"][-2]
    assert engine.messages[-1].content[0].text == "I found one durable record."
    inference = ledger.read().accounts["inference"]
    assert inference.used.calls == 2
    assert inference.used.total_tokens == 240
    events_text = (tmp_path / "session/events.jsonl").read_text()
    assert "private-test-key" not in events_text
    assert "response-2" in events_text


@pytest.mark.asyncio
async def test_openrouter_adapter_reports_embedded_http_200_provider_error(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"code": 502, "message": "Provider returned an empty response"}},
        )

    manifest = make_manifest()
    ledger = BudgetLedger(tmp_path / "mcp/budgets.json", manifest)
    session = SessionStore(tmp_path / "session", manifest.run_id)
    adapter = OpenRouterAdapter(
        api_key="private-test-key",
        ledger=ledger,
        session=session,
        max_output_tokens=500,
        prompt_price_per_token=0.000001,
        completion_price_per_token=0.000006,
        app_url="https://archive.example/",
        transport=httpx.MockTransport(handler),
    )
    engine = AibbHarnessEngine(
        model=openrouter_model(
            "openai/gpt-5.6-sol",
            context_window=1_050_000,
            max_tokens=500,
            prompt_price_per_token=0.000001,
            completion_price_per_token=0.000006,
        ),
        system_prompt="",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Continue."}], "timestamp": 1}],
        tools=[],
        stream_fn=adapter,
    )

    await engine.begin()

    events = [json.loads(line) for line in (tmp_path / "session/events.jsonl").read_text().splitlines()]
    provider_error = next(event for event in events if event["type"] == "provider_error")
    assert provider_error["payload"] == {
        "reservation_key": "inference-0001",
        "type": "RuntimeError",
        "message": "OpenRouter provider error 502: Provider returned an empty response",
    }
    assert ledger.read().accounts["inference"].used.calls == 1


@pytest.mark.asyncio
async def test_openrouter_adapter_preserves_non_success_response_details(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            headers={"x-request-id": "request-route-404", "content-type": "application/json"},
            json={
                "error": {
                    "code": 404,
                    "message": "No endpoints found that can handle the requested parameters.",
                }
            },
        )

    manifest = make_manifest()
    ledger = BudgetLedger(tmp_path / "mcp/budgets.json", manifest)
    session = SessionStore(tmp_path / "session", manifest.run_id)
    adapter = OpenRouterAdapter(
        api_key="private-test-key",
        ledger=ledger,
        session=session,
        max_output_tokens=500,
        prompt_price_per_token=0.000001,
        completion_price_per_token=0.000006,
        app_url="https://archive.example/",
        transport=httpx.MockTransport(handler),
    )
    engine = AibbHarnessEngine(
        model=openrouter_model(
            "example/model",
            context_window=100_000,
            max_tokens=500,
            prompt_price_per_token=0.000001,
            completion_price_per_token=0.000006,
        ),
        system_prompt="",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Continue."}], "timestamp": 1}],
        tools=[],
        stream_fn=adapter,
    )

    await engine.begin()

    events = [json.loads(line) for line in (tmp_path / "session/events.jsonl").read_text().splitlines()]
    provider_response = next(event for event in events if event["type"] == "provider_response")
    assert provider_response["payload"] == {
        "reservation_key": "inference-0001",
        "http_status": 404,
        "headers": {"x-request-id": "request-route-404", "content-type": "application/json"},
        "response": {
            "error": {
                "code": 404,
                "message": "No endpoints found that can handle the requested parameters.",
            }
        },
    }
    provider_error = next(event for event in events if event["type"] == "provider_error")
    assert provider_error["payload"]["type"] == "RuntimeError"
    assert provider_error["payload"]["message"] == (
        "OpenRouter HTTP 404: No endpoints found that can handle the requested parameters."
    )
    assert ledger.read().accounts["inference"].used.calls == 1


@pytest.mark.asyncio
async def test_anthropic_openrouter_requests_use_sticky_session_and_rolling_cache_breakpoints(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-status",
                        "type": "function",
                        "function": {"name": "archive_status", "arguments": "{}"},
                    }
                ],
            }
            prompt_details = {"cached_tokens": 0, "cache_write_tokens": 4_800}
        else:
            message = {"role": "assistant", "content": "Finished."}
            prompt_details = {"cached_tokens": 4_800, "cache_write_tokens": 200}
        return httpx.Response(
            200,
            json={
                "id": f"response-{len(requests)}",
                "model": "anthropic/claude-opus-5",
                "provider": "Anthropic",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if len(requests) == 1 else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5_000,
                    "completion_tokens": 20,
                    "total_tokens": 5_020,
                    "cost": 0.02,
                    "prompt_tokens_details": prompt_details,
                },
            },
        )

    async def archive_status(
        _tool_call_id: str,
        _arguments: Any,
        _signal: Any = None,
        _on_update: Any = None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text='{"published": 1}')], details={"published": 1})

    tool = AgentTool(
        name="archive_status",
        label="Archive status",
        description="Read archive status.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute=archive_status,
        executionMode="sequential",
    )
    manifest = make_manifest()
    ledger = BudgetLedger(tmp_path / "mcp/budgets.json", manifest)
    session = SessionStore(tmp_path / "session", manifest.run_id)
    adapter = OpenRouterAdapter(
        api_key="private-test-key",
        ledger=ledger,
        session=session,
        max_output_tokens=500,
        prompt_price_per_token=0.000005,
        completion_price_per_token=0.000025,
        app_url="https://archive.example/",
        transport=httpx.MockTransport(handler),
    )
    model = openrouter_model(
        "anthropic/claude-opus-5",
        context_window=1_000_000,
        max_tokens=500,
        prompt_price_per_token=0.000005,
        completion_price_per_token=0.000025,
    )
    engine = AibbHarnessEngine(
        model=model,
        system_prompt="",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "A long stable opening." * 500}],
                "timestamp": 1,
            }
        ],
        tools=[tool],
        stream_fn=adapter,
    )

    await engine.begin()

    assert len(requests) == 2
    assert {request["session_id"] for request in requests} == {manifest.run_id}
    first_opening = requests[0]["messages"][0]["content"]
    second_opening = requests[1]["messages"][0]["content"]
    assert first_opening == second_opening
    assert first_opening[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    second_tool_result = requests[1]["messages"][-1]
    assert second_tool_result["role"] == "tool"
    assert second_tool_result["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert sum(
        1
        for message in requests[1]["messages"]
        for block in (message["content"] if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict) and "cache_control" in block
    ) == 2
    assert model.cost.cacheRead == pytest.approx(0.5)
    assert model.cost.cacheWrite == pytest.approx(10)
    assistant_messages = [message for message in engine.messages if message.role == "assistant"]
    assert assistant_messages[0].usage.cacheWrite == 4_800
    assert assistant_messages[-1].usage.cacheRead == 4_800
    assert assistant_messages[-1].usage.cacheWrite == 200
    assert assistant_messages[-1].usage.cost.cacheRead == pytest.approx(0.0024)
    assert assistant_messages[-1].usage.cost.cacheWrite == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_openrouter_adapter_executes_valid_parallel_calls_when_one_has_an_extra_brace(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-first",
                        "type": "function",
                        "function": {"name": "read_thread", "arguments": '{"thread_id":"first"}'},
                    },
                    {
                        "id": "call-second",
                        "type": "function",
                        "function": {"name": "read_thread", "arguments": '{"thread_id":"second"}}'},
                    },
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": "Both thread reads completed."}
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"parallel-response-{len(requests)}",
                "model": "example/model",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.001},
            },
        )

    async def read_thread(
        _tool_call_id: str,
        arguments: Any,
        _signal: Any = None,
        _on_update: Any = None,
    ) -> AgentToolResult:
        executions.append(arguments["thread_id"])
        return AgentToolResult(content=[TextContent(text=arguments["thread_id"])])

    tool = AgentTool(
        name="read_thread",
        label="Read thread",
        description="Read a thread.",
        parameters={
            "type": "object",
            "properties": {"thread_id": {"type": "string"}},
            "required": ["thread_id"],
            "additionalProperties": False,
        },
        execute=read_thread,
        executionMode="parallel",
    )
    manifest = make_manifest()
    ledger = BudgetLedger(tmp_path / "mcp/budgets.json", manifest)
    session = SessionStore(tmp_path / "session", manifest.run_id)
    adapter = OpenRouterAdapter(
        api_key="private-test-key",
        ledger=ledger,
        session=session,
        max_output_tokens=500,
        prompt_price_per_token=0.000001,
        completion_price_per_token=0.000006,
        app_url="https://archive.example/",
        transport=httpx.MockTransport(handler),
    )
    engine = AibbHarnessEngine(
        model=openrouter_model(
            "example/model",
            context_window=10_000,
            max_tokens=500,
            prompt_price_per_token=0.000001,
            completion_price_per_token=0.000006,
        ),
        system_prompt="",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Read both."}], "timestamp": 1}],
        tools=[tool],
        stream_fn=adapter,
    )

    await engine.begin()

    assert executions == ["first", "second"]
    assert [message["tool_call_id"] for message in requests[1]["messages"] if message["role"] == "tool"] == [
        "call-first",
        "call-second",
    ]
    events = [event.model_dump(mode="json") for event in session.read_events()]
    repairs = [event for event in events if event["type"] == "provider_tool_arguments_repaired"]
    assert len(repairs) == 1
    assert repairs[0]["payload"]["tool_call_id"] == "call-second"
    assert not [event for event in events if event["type"] == "provider_error"]
    assert engine.messages[-1].content[0].text == "Both thread reads completed."


@pytest.mark.asyncio
async def test_openrouter_adapter_bounds_oversized_tool_batches_but_retains_raw_response(tmp_path: Path) -> None:
    raw_count = MAX_TOOL_CALLS_PER_RESPONSE + 9
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count > 1:
            return httpx.Response(
                200,
                json={
                    "id": "finished-response",
                    "model": "example/model",
                    "provider": "Google",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Finished."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22, "cost": 0.001},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "oversized-response",
                "model": "example/model",
                "provider": "Google",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {"name": "read_thread", "arguments": '{"thread_id":"first"}'},
                                }
                                for index in range(raw_count)
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.001},
            },
        )

    executions: list[str] = []

    async def read_thread(
        tool_call_id: str,
        _arguments: Any,
        _signal: Any = None,
        _on_update: Any = None,
    ) -> AgentToolResult:
        executions.append(tool_call_id)
        return AgentToolResult(content=[TextContent(text="read")])

    tool = AgentTool(
        name="read_thread",
        label="Read thread",
        description="Read one thread.",
        parameters={
            "type": "object",
            "properties": {"thread_id": {"type": "string"}},
            "required": ["thread_id"],
            "additionalProperties": False,
        },
        execute=read_thread,
        executionMode="sequential",
    )
    manifest = make_manifest()
    ledger = BudgetLedger(tmp_path / "mcp/budgets.json", manifest)
    session = SessionStore(tmp_path / "session", manifest.run_id)
    adapter = OpenRouterAdapter(
        api_key="private-test-key",
        ledger=ledger,
        session=session,
        max_output_tokens=500,
        prompt_price_per_token=0.000001,
        completion_price_per_token=0.000006,
        app_url="https://archive.example/",
        transport=httpx.MockTransport(handler),
    )
    engine = AibbHarnessEngine(
        model=openrouter_model(
            "example/model",
            context_window=10_000,
            max_tokens=500,
            prompt_price_per_token=0.000001,
            completion_price_per_token=0.000006,
        ),
        system_prompt="",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Read."}], "timestamp": 1}],
        tools=[tool],
        stream_fn=adapter,
    )

    await engine.begin()

    assert len(executions) == MAX_TOOL_CALLS_PER_RESPONSE
    events = [event.model_dump(mode="json") for event in session.read_events()]
    raw_response = next(event for event in events if event["type"] == "provider_response")
    assert len(raw_response["payload"]["response"]["choices"][0]["message"]["tool_calls"]) == raw_count
    truncated = next(event for event in events if event["type"] == "provider_tool_batch_truncated")
    assert truncated["payload"]["reported_tool_calls"] == raw_count
    assert truncated["payload"]["retained_tool_calls"] == MAX_TOOL_CALLS_PER_RESPONSE


def test_context_envelope_declares_a_custom_system_prompt_exception() -> None:
    envelope = build_context_envelope(
        orientation_version="v0.4",
        orientation="Explore.",
        notice_version="v0.3",
        notice="Standard prompt composition.",
        policy_version="v0.2",
        policy="Contribute carefully.",
        run_scope='{"run_id":"test"}',
        tool_definitions=[],
        system_prompt_label="Aria v1",
        system_prompt_source_url="https://example.invalid/aria.txt",
    )

    assert envelope.system_prompt_label == "Aria v1"
    assert "Experimental prompt configuration" in envelope.initial_text
    assert "declared exception" in envelope.initial_text
    assert "https://example.invalid/aria.txt" in envelope.initial_text
