"""Lossless OpenRouter Chat Completions adapter for the controlled Harn loop."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Literal

import httpx
from harn_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    StartEvent,
    TextContent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    UsageCost,
)
from harn_ai.utils.event_stream import AssistantMessageEventStream

from aibb.harness.token_estimate import estimate_json_tokens
from aibb.runtime import BudgetLedger
from aibb.runtime.budget import Usage as LedgerUsage
from aibb.sessions.store import SessionStore

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
REASONING_DETAILS_SIGNATURE_PREFIX = "openrouter-reasoning-details:"
MAX_TOOL_CALLS_PER_RESPONSE = 16
ANTHROPIC_CACHE_TTL = "1h"
ANTHROPIC_CACHE_READ_MULTIPLIER = 0.1
ANTHROPIC_CACHE_WRITE_MULTIPLIER = 2.0
MAX_ANTHROPIC_CACHE_BREAKPOINTS = 4
MAX_PROVIDER_ERROR_TEXT_CHARS = 4_000


def _encode_reasoning_details(value: list[dict[str, Any]]) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return REASONING_DETAILS_SIGNATURE_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_reasoning_details(value: str | None) -> list[dict[str, Any]] | None:
    if not value or not value.startswith(REASONING_DETAILS_SIGNATURE_PREFIX):
        return None
    encoded = value.removeprefix(REASONING_DETAILS_SIGNATURE_PREFIX)
    decoded = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
    return decoded if isinstance(decoded, list) else None


def _parse_tool_arguments(value: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Decode one provider tool call, repairing only uniquely recoverable JSON.

    Some OpenRouter routes occasionally return a valid JSON object followed by one or
    more unmatched ``}`` characters, or repeat the same complete object twice.  The
    intended object is uniquely recoverable in those cases.  Everything else remains
    an error rather than inviting a general JSON repair heuristic that could change
    model intent.
    """

    if isinstance(value, dict):
        return value, None
    raw = value if isinstance(value, str) else "{}" if value is None else str(value)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        candidate = raw.strip()
        try:
            parsed, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            raise RuntimeError(f"Provider returned invalid tool arguments: {error}") from error
        trailing = candidate[end:].strip()
        if trailing and set(trailing) == {"}"}:
            if not isinstance(parsed, dict):
                raise RuntimeError("Provider returned tool arguments that are not a JSON object") from error
            return parsed, {
                "repair": "removed_unmatched_trailing_closing_braces",
                "raw_arguments": raw,
                "normalized_arguments": parsed,
                "removed_suffix": trailing,
            }
        if not isinstance(parsed, dict):
            raise RuntimeError("Provider returned tool arguments that are not a JSON object") from error
        trailing_values: list[Any] = []
        while trailing:
            try:
                next_value, end = json.JSONDecoder().raw_decode(trailing)
            except json.JSONDecodeError:
                raise RuntimeError(f"Provider returned invalid tool arguments: {error}") from error
            trailing_values.append(next_value)
            trailing = trailing[end:].strip()
        if trailing_values and all(item == "" for item in trailing_values):
            return parsed, {
                "repair": "removed_trailing_empty_json_strings",
                "raw_arguments": raw,
                "normalized_arguments": parsed,
                "removed_value_count": len(trailing_values),
            }
        if trailing_values and all(isinstance(item, dict) and item == parsed for item in trailing_values):
            return parsed, {
                "repair": "collapsed_repeated_identical_json_objects",
                "raw_arguments": raw,
                "normalized_arguments": parsed,
                "repetition_count": len(trailing_values) + 1,
            }
        raise RuntimeError(f"Provider returned invalid tool arguments: {error}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("Provider returned tool arguments that are not a JSON object")
    return parsed, None


def openrouter_model(
    model_id: str,
    *,
    context_window: int,
    max_tokens: int,
    prompt_price_per_token: float,
    completion_price_per_token: float,
    image_input_supported: bool = False,
    reasoning_enabled: bool = False,
) -> Model:
    anthropic_cache = _is_anthropic_model(model_id)
    return Model(
        id=model_id,
        name=model_id,
        api="aibb-openrouter-chat-completions",
        provider="openrouter",
        baseUrl=OPENROUTER_ENDPOINT,
        reasoning=reasoning_enabled,
        input=["text", "image"] if image_input_supported else ["text"],
        cost=ModelCost(
            input=prompt_price_per_token * 1_000_000,
            output=completion_price_per_token * 1_000_000,
            cacheRead=(
                prompt_price_per_token * ANTHROPIC_CACHE_READ_MULTIPLIER * 1_000_000
                if anthropic_cache
                else 0
            ),
            cacheWrite=(
                prompt_price_per_token * ANTHROPIC_CACHE_WRITE_MULTIPLIER * 1_000_000
                if anthropic_cache
                else 0
            ),
        ),
        contextWindow=context_window,
        maxTokens=max_tokens,
    )


def _is_anthropic_model(model_id: str) -> bool:
    return model_id.removeprefix("~").startswith("anthropic/")


def _cacheable_text_block(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if isinstance(content, str) and content:
        block = {"type": "text", "text": content}
        message["content"] = [block]
        return block
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                return block
    return None


def _add_anthropic_cache_breakpoints(messages: list[dict[str, Any]]) -> None:
    """Keep one stable and up to three rolling cache boundaries.

    Explicit content-block breakpoints work across Anthropic-compatible
    OpenRouter routes, including Anthropic, Bedrock, and Vertex. Keeping recent
    prior boundaries in the next append-only request gives the selected backend
    an exact prefix to reuse while the first boundary protects the fixed opening
    context.
    """

    cacheable = [
        index
        for index, message in enumerate(messages)
        if message.get("role") in {"system", "user", "tool"} and message.get("content")
    ]
    if not cacheable:
        return
    selected = [cacheable[0], *cacheable[-(MAX_ANTHROPIC_CACHE_BREAKPOINTS - 1) :]]
    for index in dict.fromkeys(selected):
        block = _cacheable_text_block(messages[index])
        if block is not None:
            block["cache_control"] = {"type": "ephemeral", "ttl": ANTHROPIC_CACHE_TTL}


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(block.text for block in value if getattr(block, "type", None) == "text")


def _image_content(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return []
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
        }
        for block in value
        if getattr(block, "type", None) == "image"
    ]


def _messages(context: Context, *, image_input_supported: bool = False) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if context.systemPrompt:
        messages.append({"role": "system", "content": context.systemPrompt})
    pending_images: list[dict[str, Any]] = []
    for index, message in enumerate(context.messages):
        if message.role == "user":
            messages.append({"role": "user", "content": _text_content(message.content)})
        elif message.role == "assistant":
            tool_calls = []
            text = []
            reasoning = []
            reasoning_details: list[dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    text.append(block.text)
                elif block.type == "thinking":
                    reasoning.append(block.thinking)
                    details = _decode_reasoning_details(block.thinkingSignature)
                    if details:
                        reasoning_details.extend(details)
                elif block.type == "toolCall":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments, ensure_ascii=False, separators=(",", ":")),
                            },
                        }
                    )
            payload: dict[str, Any] = {"role": "assistant", "content": "\n".join(text) or None}
            if tool_calls:
                payload["tool_calls"] = tool_calls
            if reasoning_details:
                payload["reasoning_details"] = reasoning_details
            elif reasoning:
                payload["reasoning"] = "\n".join(reasoning)
            messages.append(payload)
        elif message.role == "toolResult":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.toolCallId,
                    "content": _text_content(message.content),
                }
            )
            if image_input_supported:
                pending_images.extend(_image_content(message.content))
                next_role = (
                    getattr(context.messages[index + 1], "role", None) if index + 1 < len(context.messages) else None
                )
                if next_role != "toolResult" and pending_images:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "The controlled harness is presenting image output from the preceding "
                                        "tool result(s) for visual inspection."
                                    ),
                                },
                                *pending_images,
                            ],
                        }
                    )
                    pending_images = []
    return messages


def _empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cacheRead=0,
        cacheWrite=0,
        totalTokens=0,
        cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
    )


def _estimate_payload_tokens(payload: dict[str, Any]) -> int:
    """Estimate context use without misclassifying encoded image bytes as text.

    OpenAI-compatible transports carry image bytes in JSON data URLs, but providers
    meter those inputs as vision tokens rather than tokenizing the base64 string.
    Reserve a deliberately conservative fixed allowance for each image and retain
    the full encoded payload separately for request-byte accounting.
    """

    return estimate_json_tokens(payload)


def _http_error_record(response: httpx.Response) -> tuple[dict[str, Any], str]:
    """Return bounded provider diagnostics and a concise operator message."""

    try:
        decoded = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        text = response.text
        record: dict[str, Any] = {
            "body_text": text[:MAX_PROVIDER_ERROR_TEXT_CHARS],
            "body_truncated": len(text) > MAX_PROVIDER_ERROR_TEXT_CHARS,
        }
        return record, f"OpenRouter HTTP {response.status_code}: {record['body_text'] or response.reason_phrase}"

    error = decoded.get("error") if isinstance(decoded, dict) else None
    if isinstance(error, dict):
        code = error.get("code")
        message = str(error.get("message") or response.reason_phrase)[:MAX_PROVIDER_ERROR_TEXT_CHARS]
        record = {"error": {"code": code, "message": message}}
        prefix = f"OpenRouter HTTP {response.status_code}"
        if code is not None and str(code) != str(response.status_code):
            prefix += f" (provider code {code})"
        return record, f"{prefix}: {message}"

    serialized = json.dumps(decoded, ensure_ascii=False, separators=(",", ":"))
    record = {
        "body_json": serialized[:MAX_PROVIDER_ERROR_TEXT_CHARS],
        "body_truncated": len(serialized) > MAX_PROVIDER_ERROR_TEXT_CHARS,
    }
    return record, f"OpenRouter HTTP {response.status_code}: {record['body_json']}"


class OpenRouterAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        ledger: BudgetLedger,
        session: SessionStore,
        max_output_tokens: int,
        prompt_price_per_token: float,
        completion_price_per_token: float,
        app_url: str,
        app_title: str = "AIBB controlled harness",
        reasoning_parameter: dict[str, object] | None = None,
        provider_routing: dict[str, object] | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
        output_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        endpoint: str = OPENROUTER_ENDPOINT,
        request_headers: dict[str, str] | None = None,
        timeout_seconds: float = 180,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self.ledger = ledger
        self.session = session
        self.max_output_tokens = max_output_tokens
        self.prompt_price_per_token = prompt_price_per_token
        self.completion_price_per_token = completion_price_per_token
        self.app_url = app_url
        self.app_title = app_title
        self.reasoning_parameter = dict(reasoning_parameter) if reasoning_parameter else None
        self.provider_routing = dict(provider_routing) if provider_routing else None
        self.tool_choice = tool_choice
        self.output_token_parameter = output_token_parameter
        self.endpoint = endpoint
        self.request_headers = dict(request_headers) if request_headers else {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.app_url,
            "X-Title": self.app_title,
        }
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.last_payload: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None

    def __call__(self, model: Model, context: Context, _options: Any) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()
        asyncio.create_task(self._run(stream, model, context))
        return stream

    def _next_key(self) -> str:
        account = self.ledger.read().accounts["inference"]
        return f"inference-{len(account.settled) + len(account.reservations) + 1:04d}"

    async def _run(self, stream: AssistantMessageEventStream, model: Model, context: Context) -> None:
        output = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=_empty_usage(),
            stopReason="stop",
            timestamp=int(time.time() * 1000),
        )
        reservation_key = self._next_key()
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": _messages(context, image_input_supported="image" in model.input),
            "session_id": self.session.run_id,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in (context.tools or [])
            ],
            "tool_choice": self.tool_choice,
            self.output_token_parameter: min(self.max_output_tokens, model.maxTokens),
            "stream": False,
        }
        anthropic_cache = _is_anthropic_model(model.id)
        if anthropic_cache:
            _add_anthropic_cache_breakpoints(payload["messages"])
        if not payload["tools"]:
            payload.pop("tools")
            payload.pop("tool_choice")
        if self.reasoning_parameter:
            payload["reasoning"] = self.reasoning_parameter
        if self.provider_routing:
            payload["provider"] = self.provider_routing
        estimated_input = _estimate_payload_tokens(payload)
        available_output = model.contextWindow - estimated_input
        if available_output < 1:
            message = f"Estimated input ({estimated_input}) exceeds the model context window ({model.contextWindow})"
            self.session.append(
                "provider_error",
                {"reservation_key": reservation_key, "type": "ContextWindowError", "message": message},
                "private_provider",
            )
            output.stopReason = "error"
            output.errorMessage = message
            stream.push(ErrorEvent(reason="error", error=output))
            stream.end()
            return
        effective_output = min(int(payload[self.output_token_parameter]), available_output)
        payload[self.output_token_parameter] = effective_output
        self.last_payload = payload
        estimated_input = _estimate_payload_tokens(payload)
        input_reservation_multiplier = ANTHROPIC_CACHE_WRITE_MULTIPLIER if anthropic_cache else 1.0
        reserved_cost = (
            estimated_input * self.prompt_price_per_token * input_reservation_multiplier
            + effective_output * self.completion_price_per_token
        )
        requested = LedgerUsage(
            calls=1,
            input_tokens=estimated_input,
            output_tokens=effective_output,
            total_tokens=estimated_input + effective_output,
            cost_usd=reserved_cost,
            request_bytes=len(json.dumps(payload, ensure_ascii=False).encode()),
        )
        try:
            self.ledger.reserve("inference", reservation_key, requested)
            self.session.append(
                "provider_request",
                {"reservation_key": reservation_key, "endpoint": self.endpoint, "payload": payload},
                "private_provider",
            )
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = await client.post(
                    self.endpoint,
                    headers=self.request_headers,
                    json=payload,
                )
            if response.is_error:
                error_record, error_message = _http_error_record(response)
                self.last_response = error_record
                self.session.append(
                    "provider_response",
                    {
                        "reservation_key": reservation_key,
                        "http_status": response.status_code,
                        "headers": {
                            name: value
                            for name, value in response.headers.items()
                            if name.lower() in {"x-request-id", "openrouter-processing-time", "content-type"}
                        },
                        "response": error_record,
                    },
                    "private_provider",
                )
                raise RuntimeError(error_message)
            response.raise_for_status()
            raw = response.json()
            self.last_response = raw
            self.session.append(
                "provider_response",
                {
                    "reservation_key": reservation_key,
                    "http_status": response.status_code,
                    "headers": {
                        name: value
                        for name, value in response.headers.items()
                        if name.lower() in {"x-request-id", "openrouter-processing-time", "content-type"}
                    },
                    "response": raw,
                },
                "private_provider",
            )
            embedded_error = raw.get("error")
            if isinstance(embedded_error, dict):
                error_code = embedded_error.get("code")
                error_message = embedded_error.get("message") or "unknown provider error"
                prefix = (
                    f"OpenRouter provider error {error_code}"
                    if error_code is not None
                    else "OpenRouter provider error"
                )
                raise RuntimeError(f"{prefix}: {error_message}")
            choices = raw.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("OpenRouter response did not include any choices")
            usage_payload = raw.get("usage") or {}
            input_tokens = int(usage_payload.get("prompt_tokens") or 0)
            output_tokens = int(usage_payload.get("completion_tokens") or 0)
            total_tokens = int(usage_payload.get("total_tokens") or input_tokens + output_tokens)
            prompt_details = usage_payload.get("prompt_tokens_details") or {}
            cache_read_tokens = int(prompt_details.get("cached_tokens") or 0)
            cache_write_tokens = int(prompt_details.get("cache_write_tokens") or 0)
            uncached_input_tokens = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
            actual_cost = float(
                usage_payload.get("cost")
                or input_tokens * self.prompt_price_per_token + output_tokens * self.completion_price_per_token
            )
            self.ledger.reconcile(
                "inference",
                reservation_key,
                LedgerUsage(
                    calls=1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=actual_cost,
                    request_bytes=requested.request_bytes,
                    result_bytes=len(response.content),
                ),
            )
            output.usage = Usage(
                input=input_tokens,
                output=output_tokens,
                cacheRead=cache_read_tokens,
                cacheWrite=cache_write_tokens,
                totalTokens=total_tokens,
                cost=UsageCost(
                    input=uncached_input_tokens * self.prompt_price_per_token,
                    output=output_tokens * self.completion_price_per_token,
                    cacheRead=(
                        cache_read_tokens * self.prompt_price_per_token * ANTHROPIC_CACHE_READ_MULTIPLIER
                    ),
                    cacheWrite=(
                        cache_write_tokens * self.prompt_price_per_token * ANTHROPIC_CACHE_WRITE_MULTIPLIER
                    ),
                    total=actual_cost,
                ),
            )
            output.responseId = raw.get("id")
            output.responseModel = raw.get("model")
            choice = choices[0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
            raw_tool_calls = message.get("tool_calls") or []
            if len(raw_tool_calls) > MAX_TOOL_CALLS_PER_RESPONSE:
                tool_name_counts: dict[str, int] = {}
                for raw_call in raw_tool_calls:
                    tool_name = str((raw_call.get("function") or {}).get("name") or "unknown")
                    tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
                self.session.append(
                    "provider_tool_batch_truncated",
                    {
                        "reservation_key": reservation_key,
                        "reported_tool_calls": len(raw_tool_calls),
                        "retained_tool_calls": MAX_TOOL_CALLS_PER_RESPONSE,
                        "omitted_tool_calls": len(raw_tool_calls) - MAX_TOOL_CALLS_PER_RESPONSE,
                        "tool_name_counts": tool_name_counts,
                    },
                    "private_provider",
                )
                raw_tool_calls = raw_tool_calls[:MAX_TOOL_CALLS_PER_RESPONSE]
            output.stopReason = (
                "toolUse"
                if raw_tool_calls or finish_reason in {"tool_calls", "function_call"}
                else ("length" if finish_reason == "length" else "stop")
            )
            stream.push(StartEvent(partial=output))
            reasoning = message.get("reasoning")
            reasoning_details = message.get("reasoning_details")
            valid_reasoning_details = (
                [item for item in reasoning_details if isinstance(item, dict)]
                if isinstance(reasoning_details, list)
                else []
            )
            detail_text = "\n".join(
                str(item.get("summary") or item.get("text") or "")
                for item in valid_reasoning_details
                if item.get("summary") or item.get("text")
            )
            thinking_text = reasoning if isinstance(reasoning, str) else detail_text
            if thinking_text or valid_reasoning_details:
                block = ThinkingContent(
                    thinking=thinking_text or "[Provider reasoning state retained without visible text]",
                    thinkingSignature=(
                        _encode_reasoning_details(valid_reasoning_details)
                        if valid_reasoning_details
                        else "openrouter-reasoning"
                    ),
                    redacted=not bool(thinking_text),
                )
                output.content.append(block)
                index = len(output.content) - 1
                stream.push(ThinkingStartEvent(contentIndex=index, partial=output))
                stream.push(ThinkingEndEvent(contentIndex=index, content=block.thinking, partial=output))
            content = message.get("content")
            if isinstance(content, str) and content:
                block = TextContent(text=content)
                output.content.append(block)
                index = len(output.content) - 1
                stream.push(TextStartEvent(contentIndex=index, partial=output))
                stream.push(TextEndEvent(contentIndex=index, content=content, partial=output))
            for raw_call in raw_tool_calls:
                function = raw_call["function"]
                arguments, repair = _parse_tool_arguments(function.get("arguments"))
                if repair:
                    self.session.append(
                        "provider_tool_arguments_repaired",
                        {
                            "reservation_key": reservation_key,
                            "tool_call_id": raw_call["id"],
                            "tool_name": function["name"],
                            **repair,
                        },
                        "private_provider",
                    )
                block = ToolCall(id=raw_call["id"], name=function["name"], arguments=arguments)
                output.content.append(block)
                index = len(output.content) - 1
                stream.push(ToolCallStartEvent(contentIndex=index, partial=output))
                stream.push(ToolCallEndEvent(contentIndex=index, toolCall=block, partial=output))
            stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001
            account = self.ledger.read().accounts["inference"]
            if reservation_key in account.reservations:
                self.ledger.reconcile(
                    "inference",
                    reservation_key,
                    LedgerUsage(calls=1, request_bytes=requested.request_bytes),
                )
            self.session.append(
                "provider_error",
                {"reservation_key": reservation_key, "type": type(error).__name__, "message": str(error)},
                "private_provider",
            )
            output.stopReason = "error"
            output.errorMessage = str(error)
            stream.push(ErrorEvent(reason="error", error=output))
        finally:
            stream.end()
