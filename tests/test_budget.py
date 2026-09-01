from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aibb.runtime import BudgetExceededError, BudgetLedger
from aibb.runtime.budget import Usage
from aibb.runtime.models import (
    BoundModelIdentity,
    BudgetLimits,
    OpenRouterRoutingConfiguration,
    ReasoningConfiguration,
    RunManifest,
)


def make_manifest(*, quota: int = 1) -> RunManifest:
    now = datetime.now(UTC)
    return RunManifest(
        run_id="run-test-model-001",
        created_at=now,
        mode="interactive",
        identity=BoundModelIdentity(
            provider="openrouter",
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            developer="OpenAI",
            model_name="openai/gpt-5.6-luna",
            normalized_model_name="openai/gpt-5.6-luna",
            generation="5.6",
            lineage="GPT",
            public_author_id="openrouter-gpt-5-6-luna-test",
            display_name="GPT-5.6 Luna",
        ),
        orientation_version="v0.1",
        notice_version="v0.1",
        policy_version="v0.1",
        contribution_quota=quota,
        max_new_threads=quota,
        model_context_window=1_050_000,
        model_max_completion_tokens=128_000,
        model_input_modalities=["text", "image"],
        reasoning=ReasoningConfiguration(
            enabled=True,
            supported_efforts=["high", "medium", "low"],
            selected_effort="high",
            request_parameter={"effort": "high", "exclude": False},
            source="openrouter-catalog",
        ),
        inference_budget=BudgetLimits(
            max_calls=4,
            max_input_tokens=20_000,
            max_output_tokens=4_000,
            max_total_tokens=24_000,
            max_cost_usd=0.10,
        ),
        capability_budgets={
            "contributions": BudgetLimits(max_calls=quota),
            "guestbook_entries": BudgetLimits(max_calls=1),
            "web_search": BudgetLimits(max_calls=1, max_result_bytes=20_000),
        },
    )


def test_legacy_manifest_expiry_is_discarded() -> None:
    payload = make_manifest().model_dump(mode="json")
    payload["expires_at"] = "2026-08-08T00:00:00Z"

    restored = RunManifest.model_validate(payload)

    assert "expires_at" not in restored.model_dump(mode="json")


def test_legacy_manifest_null_bedrock_route_is_discarded() -> None:
    payload = make_manifest().model_dump(mode="json")
    payload["amazon_bedrock_routing"] = None

    restored = RunManifest.model_validate(payload)

    assert "amazon_bedrock_routing" not in restored.model_dump(mode="json")


def test_legacy_manifest_defaults_to_manual_acceptance() -> None:
    payload = make_manifest().model_dump(
        mode="json", exclude={"review_before_accepting", "build_after_accepting"}
    )

    restored = RunManifest.model_validate(payload)

    assert restored.review_before_accepting is True
    assert restored.build_after_accepting is False


def test_run_manifest_accepts_local_board_url_but_not_public_plain_http() -> None:
    local = make_manifest().model_copy(update={"archive_base_url": "http://127.0.0.1:8000/"})

    assert RunManifest.model_validate(local.model_dump()).archive_base_url == "http://127.0.0.1:8000/"
    with pytest.raises(ValueError, match="must use HTTPS"):
        RunManifest.model_validate(
            {**make_manifest().model_dump(), "archive_base_url": "http://board.example/"}
        )


def test_budget_reserve_reconcile_and_resume(tmp_path: Path) -> None:
    manifest = make_manifest()
    path = tmp_path / "budgets.json"
    ledger = BudgetLedger(path, manifest)

    ledger.reserve("web_search", "search-1", Usage(calls=1, result_bytes=20_000))
    with pytest.raises(BudgetExceededError, match="max_calls"):
        ledger.reserve("web_search", "search-2", Usage(calls=1))
    ledger.reconcile("web_search", "search-1", Usage(calls=1, result_bytes=3_000))

    resumed = BudgetLedger(path, manifest)
    assert resumed.remaining()["web_search"]["max_calls"] == 0
    assert resumed.reconcile("web_search", "search-1", Usage(calls=1)).calls == 1


def test_unknown_capability_is_not_implicitly_enabled(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budgets.json", make_manifest())

    with pytest.raises(BudgetExceededError, match="not enabled"):
        ledger.reserve("image_generation", "image-1", Usage(calls=1))


def test_openrouter_provider_pin_is_immutable_and_serialized() -> None:
    routing = OpenRouterRoutingConfiguration(
        provider_slug="google-vertex",
        provider_name="Google",
        quantization="unknown",
    )
    manifest = make_manifest().model_copy(update={"openrouter_routing": routing})

    restored = RunManifest.model_validate_json(manifest.model_dump_json())

    assert restored.openrouter_routing == routing
    assert restored.openrouter_routing.request_parameter() == {
        "order": ["google-vertex"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_openrouter_provider_pin_can_record_local_parameter_verification() -> None:
    routing = OpenRouterRoutingConfiguration(
        provider_slug="anthropic",
        provider_name="Anthropic",
        require_parameters=False,
        quantization="unknown",
    )

    restored = OpenRouterRoutingConfiguration.model_validate_json(routing.model_dump_json())

    assert restored.request_parameter() == {
        "order": ["anthropic"],
        "allow_fallbacks": False,
        "require_parameters": False,
    }


def test_budget_extension_only_increases_selected_limits_and_preserves_usage(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budgets.json", make_manifest())
    ledger.reserve("inference", "turn-1", Usage(calls=1, input_tokens=100, total_tokens=100))
    ledger.reconcile("inference", "turn-1", Usage(calls=1, input_tokens=80, total_tokens=80))

    previous, updated = ledger.extend_limits(
        "inference",
        BudgetLimits(max_calls=8, max_input_tokens=40_000, max_total_tokens=48_000),
    )

    assert previous.max_calls == 4
    assert updated.max_calls == 8
    assert previous.max_input_tokens == 20_000
    assert updated.max_input_tokens == 40_000
    assert ledger.read().accounts["inference"].used.input_tokens == 80
    with pytest.raises(ValueError, match="must increase"):
        ledger.extend_limits("inference", BudgetLimits(max_total_tokens=24_000))
