import httpx
import pytest

from aibb.harness.catalog import OpenRouterModelRecord, fetch_openrouter_endpoint, public_openrouter_model_id


def _record(reasoning: dict[str, object] | None) -> OpenRouterModelRecord:
    return OpenRouterModelRecord(
        id="example/model",
        name="Example Labs: Model",
        context_length=100_000,
        pricing={"prompt": "0.000001", "completion": "0.000002"},
        architecture={"input_modalities": ["text", "image"]},
        supported_parameters=["tools", "reasoning"],
        reasoning=reasoning,
    )


def test_catalog_selects_high_reasoning_without_starving_the_visible_answer() -> None:
    record = _record(
        {
            "mandatory": False,
            "default_enabled": False,
            "default_effort": "medium",
            "supported_efforts": ["max", "xhigh", "high", "medium", "low"],
        }
    )

    selected = record.select_reasoning()

    assert record.developer == "Example Labs"
    assert record.display_name == "Model"
    assert selected.enabled is True
    assert selected.selected_effort == "high"
    assert selected.request_parameter == {"effort": "high", "exclude": False}


def test_catalog_honors_an_explicit_supported_reasoning_effort() -> None:
    record = _record(
        {
            "mandatory": False,
            "default_enabled": True,
            "default_effort": "medium",
            "supported_efforts": ["high", "medium", "low"],
        }
    )

    selected = record.select_reasoning(effort="low")

    assert selected.enabled is True
    assert selected.selected_effort == "low"
    assert selected.request_parameter == {"effort": "low", "exclude": False}
    assert selected.source == "curator-override"


def test_catalog_rejects_an_unadvertised_reasoning_effort() -> None:
    record = _record(
        {
            "mandatory": False,
            "default_enabled": True,
            "default_effort": "medium",
            "supported_efforts": ["high", "medium", "low"],
        }
    )

    with pytest.raises(ValueError, match="does not advertise reasoning effort max"):
        record.select_reasoning(effort="max")


def test_non_reasoning_catalog_record_does_not_invent_a_mode() -> None:
    selected = _record(None).select_reasoning()

    assert selected.enabled is False
    assert selected.request_parameter is None
    assert selected.source == "unavailable"


def test_probe_informed_mandatory_reasoning_override_is_explicit() -> None:
    selected = _record(None).select_reasoning("mandatory")

    assert selected.enabled is True
    assert selected.mandatory is True
    assert selected.selected_effort is None
    assert selected.request_parameter == {"enabled": True}
    assert selected.source == "curator-override"


def test_provider_context_ceiling_clamps_model_catalog_maximum() -> None:
    record = _record(None).model_copy(
        update={
            "context_length": 1_048_576,
            "top_provider": {"context_length": 524_288, "max_completion_tokens": None},
        }
    )

    assert record.effective_context_length == 524_288
    assert record.clamp_output_tokens(600_000) == 520_192


def test_missing_provider_context_uses_model_catalog_maximum() -> None:
    record = _record(None)

    assert record.effective_context_length == 100_000


def test_free_route_selector_is_not_part_of_public_model_identity() -> None:
    assert public_openrouter_model_id("inclusionai/ling-3.0-flash:free") == "inclusionai/ling-3.0-flash"
    assert public_openrouter_model_id("inclusionai/ling-3.0-flash") == "inclusionai/ling-3.0-flash"


@pytest.mark.asyncio
async def test_specific_openrouter_endpoint_is_resolved_and_parameter_checked() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "endpoints": [
                        {
                            "name": "Google | example/model",
                            "model_id": "example/model",
                            "provider_name": "Google",
                            "tag": "google-vertex",
                            "context_length": 163_840,
                            "pricing": {"prompt": "0.00000056", "completion": "0.00000168"},
                            "quantization": "unknown",
                            "max_completion_tokens": 65_536,
                            "supported_parameters": [
                                "tools",
                                "tool_choice",
                                "reasoning",
                                "max_completion_tokens",
                            ],
                        }
                    ]
                }
            },
        )

    endpoint = await fetch_openrouter_endpoint(
        "example/model",
        "google-vertex",
        transport=httpx.MockTransport(handler),
    )

    assert endpoint.provider_name == "Google"
    assert endpoint.context_length == 163_840
    assert endpoint.max_completion_tokens == 65_536
    assert endpoint.output_token_parameter == "max_completion_tokens"
    assert endpoint.prompt_price == pytest.approx(0.00000056)
    assert endpoint.quantization == "unknown"


@pytest.mark.asyncio
async def test_specific_openrouter_endpoint_rejects_missing_tool_choice() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "endpoints": [
                        {
                            "name": "Example",
                            "model_id": "example/model",
                            "provider_name": "Example",
                            "tag": "example",
                            "context_length": 10_000,
                            "pricing": {"prompt": "0.0", "completion": "0.0"},
                            "supported_parameters": ["tools", "max_tokens"],
                        }
                    ]
                }
            },
        )

    with pytest.raises(ValueError, match="tool_choice"):
        await fetch_openrouter_endpoint(
            "example/model",
            "example",
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_specific_openrouter_endpoint_accepts_value_level_tool_choice_support() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "endpoints": [
                        {
                            "name": "Example",
                            "model_id": "example/model",
                            "provider_name": "Example",
                            "tag": "example",
                            "context_length": 10_000,
                            "pricing": {"prompt": "0.0", "completion": "0.0"},
                            "supported_parameters": ["tools", "max_tokens"],
                            "supports_tool_choice": {
                                "none": True,
                                "auto": True,
                                "required": True,
                                "function": True,
                            },
                        }
                    ]
                }
            },
        )

    endpoint = await fetch_openrouter_endpoint(
        "example/model",
        "example",
        tool_choice="required",
        transport=httpx.MockTransport(handler),
    )

    assert endpoint.accepts_tool_choice("auto")
    assert endpoint.accepts_tool_choice("required")


@pytest.mark.asyncio
async def test_specific_openrouter_endpoint_rejects_unadvertised_tool_choice_value() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "endpoints": [
                        {
                            "name": "Example",
                            "model_id": "example/model",
                            "provider_name": "Example",
                            "tag": "example",
                            "context_length": 10_000,
                            "pricing": {"prompt": "0.0", "completion": "0.0"},
                            "supported_parameters": ["tools", "max_tokens"],
                            "supports_tool_choice": {"auto": True, "required": False},
                        }
                    ]
                }
            },
        )

    with pytest.raises(ValueError, match="tool_choice='required'"):
        await fetch_openrouter_endpoint(
            "example/model",
            "example",
            tool_choice="required",
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_specific_openrouter_endpoint_rejects_missing_output_token_parameter() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "endpoints": [
                        {
                            "name": "Example",
                            "model_id": "example/model",
                            "provider_name": "Example",
                            "tag": "example",
                            "context_length": 10_000,
                            "pricing": {"prompt": "0.0", "completion": "0.0"},
                            "supported_parameters": ["tools", "tool_choice"],
                        }
                    ]
                }
            },
        )

    with pytest.raises(ValueError, match="output-token parameter"):
        await fetch_openrouter_endpoint(
            "example/model",
            "example",
            transport=httpx.MockTransport(handler),
        )
