from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from test_archive_build import _write_archive
from test_budget import make_manifest
from typer.main import get_command
from typer.testing import CliRunner

from aibb import __version__
from aibb.cli import app
from aibb.harness.catalog import OpenRouterImageModelRecord, OpenRouterModelRecord
from aibb.harness.engine import EngineSnapshot
from aibb.harness.runner import (
    _clean_mcp_environment,
    _headless_continuation_attempts_in_current_segment,
    _headless_resume_requires_continuation,
    _load_system_prompt,
    _provider_error_at_boundary,
    _remove_failed_assistant_placeholder,
    _tool_execution_started_after_latest_provider_response,
    _turn_boundary_outcome,
    create_run_manifest,
    model_identity_collisions,
    record_terminal_run_event,
    reported_board_issues_summary,
)
from aibb.harness.tinker import (
    TINKER_ANTHROPIC_ENDPOINT,
    TINKER_INKLING_SMALL,
    TINKER_INKLING_SMALL_CONTEXT_WINDOW,
    TINKER_INKLING_SMALL_SERVERLESS_256K,
)
from aibb.runtime import BudgetLedger, RunManifest
from aibb.runtime.budget import Usage
from aibb.runtime.models import BudgetLimits
from aibb.scaffold import create_board
from aibb.sessions import SessionStore


def test_terminal_run_event_reports_private_board_issues_without_copying_bodies(tmp_path: Path) -> None:
    manifest = make_manifest()
    run_dir = tmp_path / manifest.run_id
    issue_log = run_dir / "mcp/reported-board-issues.jsonl"
    issue_log.parent.mkdir(parents=True)
    issue_log.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "issue_id": issue_id,
                    "run_id": manifest.run_id,
                    "reported_at": "2026-07-31T10:00:00+00:00",
                    "reported_by": "model",
                    "text": body,
                }
            )
            + "\n"
            for issue_id, body in (
                ("issue-0123456789abcdef", "The page extractor lost the article body."),
                ("issue-fedcba9876543210", "A search cursor repeated a result."),
            )
        ),
        encoding="utf-8",
    )
    store = SessionStore(run_dir / "session", manifest.run_id)
    output = StringIO()

    record_terminal_run_event(
        store=store,
        run_dir=run_dir,
        event_type="run_suspended",
        payload={"reason": "provider error"},
        visibility="operator",
        console=Console(file=output, color_system=None, width=300),
    )

    event = store.read_events()[-1]
    summary = event.payload["reported_board_issues"]
    assert summary == {
        "artifact": "mcp/reported-board-issues.jsonl",
        "count": 2,
        "issue_ids": ["issue-0123456789abcdef", "issue-fedcba9876543210"],
        "log_status": "ok",
        "requires_administrator_review": True,
    }
    assert "article body" not in event.model_dump_json()
    assert "search cursor" not in event.model_dump_json()
    rendered = output.getvalue()
    assert "Board issues require review" in rendered
    assert "2 private board issue reports" in rendered
    assert "issue-0123456789abcdef" in rendered
    assert issue_log.name in rendered


def test_terminal_issue_summary_requires_review_when_private_log_is_malformed(tmp_path: Path) -> None:
    manifest = make_manifest()
    run_dir = tmp_path / manifest.run_id
    issue_log = run_dir / "mcp/reported-board-issues.jsonl"
    issue_log.parent.mkdir(parents=True)
    issue_log.write_text("{not-json\n", encoding="utf-8")

    summary = reported_board_issues_summary(run_dir, manifest.run_id)

    assert summary["count"] is None
    assert summary["requires_administrator_review"] is True
    assert summary["log_status"] == "unreadable"
    assert summary["error"] == "private issue-report log contains malformed JSON at line 1"


def test_terminal_issue_summary_is_explicit_when_no_issue_log_exists(tmp_path: Path) -> None:
    manifest = make_manifest()

    summary = reported_board_issues_summary(tmp_path, manifest.run_id)

    assert summary["count"] == 0
    assert summary["issue_ids"] == []
    assert summary["requires_administrator_review"] is False
    assert summary["log_status"] == "absent"


def test_new_generic_board_omits_guestbook_budget_without_quota_exempt_thread(tmp_path: Path) -> None:
    data = tmp_path / "board"
    create_board(
        destination=data,
        title="AIBB",
        base_url="https://board.example/",
        curator_name="Board administrator",
        description="A generic test board.",
    )

    manifest, _run_dir = create_run_manifest(
        data_repo=data,
        state_root=tmp_path / "state",
        model_id="example/model",
        display_name="Example Model",
        generation=None,
        lineage=None,
        mode="headless",
        compaction_policy="deny",
        contribution_quota=2,
        max_output_tokens=4096,
        max_provider_turns=10,
        max_total_tokens=100_000,
        max_cost_usd=1,
        max_contributions_per_thread=1,
        model_context_window=128_000,
        model_max_completion_tokens=4096,
        prompt_price_per_token=0.0,
        completion_price_per_token=0.0,
        allow_repeat_reason=None,
        developer="Example",
        provider="openrouter",
    )

    assert "guestbook_entries" not in manifest.capability_budgets
    assert manifest.capability_budgets["contributions"].max_calls == 2
    assert manifest.starting_points_version == "v0.3"
    assert manifest.starting_points_sha256 is not None


def test_generic_cli_help_uses_board_vocabulary_and_keeps_legacy_flags_hidden() -> None:
    runner = CliRunner()
    root_help = runner.invoke(app, ["--help"], terminal_width=160)
    run_help = runner.invoke(app, ["run", "--help"], terminal_width=160)
    new_board_help = runner.invoke(app, ["new-board", "--help"], terminal_width=160)
    empty_run = runner.invoke(app, ["run"], terminal_width=160)
    root_command = get_command(app)
    run_command = root_command.commands["run"]
    new_board_command = root_command.commands["new-board"]
    run_options = {name for parameter in run_command.params for name in getattr(parameter, "opts", [])}
    new_board_options = {name for parameter in new_board_command.params for name in getattr(parameter, "opts", [])}
    run_copy = " ".join(
        [run_command.help or "", *(getattr(parameter, "help", None) or "" for parameter in run_command.params)]
    )
    new_board_copy = " ".join(
        [
            new_board_command.help or "",
            *(getattr(parameter, "help", None) or "" for parameter in new_board_command.params),
        ]
    )

    assert root_help.exit_code == run_help.exit_code == new_board_help.exit_code == 0
    assert empty_run.exit_code != 0
    assert "Usage:" in empty_run.output
    assert "Traceback" not in empty_run.output
    assert "admin" in root_help.output
    assert "probe-bedrock-sonnet" not in root_help.output
    assert "curator" not in root_help.output.casefold()
    assert "materialize" not in root_help.output.casefold()
    assert "--post-limit" in run_options
    assert "--max-posts-per-thread" in run_options
    assert "--admin-note" in run_options
    assert "contribution" not in run_copy.casefold()
    assert "curator" not in run_copy.casefold()
    assert "amazon-bedrock" not in run_help.output
    assert "--bedrock-region" not in run_options
    assert "--admin" in new_board_options
    assert "--json" in new_board_options
    assert "curator" not in new_board_copy.casefold()


def test_run_reports_a_missing_board_as_operator_input_without_a_traceback(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["run", str(tmp_path), "--provider", "openrouter", "--model", "example/model"],
    )

    assert result.exit_code != 0
    assert "Missing board configuration" in result.output
    assert "Traceback" not in result.output


def test_run_cli_exposes_public_developer_override() -> None:
    result = CliRunner().invoke(app, ["run", "--help"], terminal_width=160)
    run_command = get_command(app).commands["run"]
    option_names = {name for parameter in run_command.params for name in getattr(parameter, "opts", [])}

    assert result.exit_code == 0
    assert "[BOARD]" in result.output
    assert "--production" not in option_names
    assert "--developer" in option_names
    assert "--author" in option_names
    assert "--reasoning-effort" in option_names
    assert "--return-as" not in option_names
    assert "presentation-poor" in result.output
    display_option = next(parameter for parameter in run_command.params if "--display-name" in parameter.opts)
    assert "inferred from provider" in display_option.help


def test_run_defaults_to_headless_live_watch_and_accepts_reasoning_effort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = tmp_path / "board"
    create_board(destination=data, title="Test Board")
    observed: dict[str, object] = {}

    async def fake_fetch_model(_model_id: str) -> OpenRouterModelRecord:
        return OpenRouterModelRecord(
            id="example/model",
            name="Example: Model",
            context_length=128_000,
            pricing={"prompt": "0.000001", "completion": "0.000002"},
            architecture={"input_modalities": ["text"]},
            supported_parameters=["tools", "reasoning"],
            reasoning={
                "mandatory": False,
                "default_enabled": True,
                "default_effort": "medium",
                "supported_efforts": ["high", "medium", "low"],
            },
            top_provider={"max_completion_tokens": 16_000},
        )

    async def fake_run_model_visit(**kwargs):
        observed["run"] = kwargs
        return Path(kwargs["run_dir"]).name

    def fake_watch_event_stream(run_dir: Path, **kwargs) -> None:
        observed["watch"] = {"run_dir": run_dir, **kwargs}

    monkeypatch.setenv("AIBB_HOME", str(tmp_path / "aibb-home"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-openrouter-token")
    monkeypatch.setattr("aibb.cli.fetch_openrouter_model", fake_fetch_model)
    monkeypatch.setattr("aibb.cli.run_model_visit", fake_run_model_visit)
    monkeypatch.setattr("aibb.cli.watch_event_stream", fake_watch_event_stream)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(data),
            "--provider",
            "openrouter",
            "--model",
            "example/model",
            "--reasoning-effort",
            "high",
            "--images",
            "disable",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith("Starting Model on Test Board\n")
    assert "Runtime   headless" in result.output
    assert "reasoning high" in result.output
    assert "Limits    5 posts" in result.output
    assert "Watching reasoning, tool calls, and usage." in result.output
    assert "package_sha256" not in result.output
    assert "Remaining:" not in result.output
    assert "curator>" not in result.output
    run_kwargs = observed["run"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["console"] is not None
    manifest = RunManifest.load(Path(run_kwargs["run_dir"]) / "manifest.json")
    assert manifest.mode == "headless"
    assert manifest.aibb_version == __version__
    assert manifest.max_active_drafts == 1
    assert manifest.reasoning.selected_effort == "high"
    assert manifest.reasoning.request_parameter == {"effort": "high", "exclude": False}
    assert manifest.reasoning.source == "curator-override"
    watch_kwargs = observed["watch"]
    assert isinstance(watch_kwargs, dict)
    assert watch_kwargs["from_start"] is True
    assert watch_kwargs["show_reasoning"] is True
    assert watch_kwargs["stop_on_terminal"] is True


def test_cli_reports_installed_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == f"aibb {__version__}\n"


def test_extend_inference_budget_can_raise_provider_call_and_cost_ceilings(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    manifest = make_manifest()
    run_dir = state_root / manifest.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    ledger = BudgetLedger(run_dir / "mcp/budgets.json", manifest)
    store = SessionStore(run_dir / "session", manifest.run_id)
    store.write_checkpoint(EngineSnapshot(system_prompt="", model={"id": "example/model"}, messages=[]))

    result = CliRunner().invoke(
        app,
        [
            "extend-inference-budget",
            "--run-id",
            manifest.run_id,
            "--state-root",
            str(state_root),
            "--max-calls",
            "12",
            "--max-cost-usd",
            "10",
            "--reason",
            "Continue a cheap model visit past its initial operational ceiling.",
        ],
    )

    assert result.exit_code == 0, result.output
    assert ledger.read().accounts["inference"].limits.max_calls == 12
    assert ledger.read().accounts["inference"].limits.max_cost_usd == 10
    extension = store.read_events()[-1]
    assert extension.type == "inference_budget_extended"
    assert extension.payload["previous"]["max_calls"] == 4
    assert extension.payload["updated"]["max_calls"] == 12
    assert extension.payload["previous"]["max_cost_usd"] == manifest.inference_budget.max_cost_usd
    assert extension.payload["updated"]["max_cost_usd"] == 10


def test_extend_web_budget_preserves_usage_and_scales_research_token_ceilings(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    manifest = make_manifest()
    manifest = manifest.model_copy(
        update={
            "capability_budgets": {
                **manifest.capability_budgets,
                "web": BudgetLimits(
                    max_calls=40,
                    max_input_tokens=80_000,
                    max_output_tokens=160_000,
                    max_total_tokens=240_000,
                    max_cost_usd=5,
                ),
            }
        }
    )
    run_dir = state_root / manifest.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    ledger = BudgetLedger(run_dir / "mcp/budgets.json", manifest)
    ledger.reserve("web", "old-research", Usage(calls=1, cost_usd=1))
    ledger.reconcile("web", "old-research", Usage(calls=1, input_tokens=100, cost_usd=0.25))
    store = SessionStore(run_dir / "session", manifest.run_id)
    store.append("run_suspended", {"reason": "curator"}, "operator")
    store.write_checkpoint(EngineSnapshot(system_prompt="", model={"id": "example/model"}, messages=[]))

    result = CliRunner().invoke(
        app,
        [
            "extend-web-budget",
            "--run-id",
            manifest.run_id,
            "--state-root",
            str(state_root),
            "--max-cost-usd",
            "10",
            "--reason",
            "Use the stronger native-web research service.",
        ],
    )

    assert result.exit_code == 0, result.output
    account = ledger.read().accounts["web"]
    assert account.used == Usage(calls=1, input_tokens=100, cost_usd=0.25)
    assert account.limits.max_cost_usd == 10
    assert account.limits.max_input_tokens == account.limits.max_calls * 128_000
    assert account.limits.max_output_tokens == account.limits.max_calls * 32_768
    assert account.limits.max_total_tokens == account.limits.max_input_tokens + account.limits.max_output_tokens
    extension = store.read_events()[-1]
    assert extension.type == "web_budget_extended"
    assert extension.payload["usage_preserved"] is True


def test_rewind_run_context_preserves_trace_and_spend_at_safe_boundary(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    manifest = make_manifest()
    run_dir = state_root / manifest.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    ledger = BudgetLedger(run_dir / "mcp/budgets.json", manifest)
    pending = Usage(calls=1, input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.05)
    ledger.reserve("inference", "inference-pending", pending)
    store = SessionStore(run_dir / "session", manifest.run_id)
    store.append("provider_response", {"reservation_key": "earlier"}, "private_provider")
    store.append("run_suspended", {"reason": "curator"}, "operator")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "orientation"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "read_slowboard_thread",
                    "arguments": {"thread_id": "thread-1"},
                }
            ],
        },
        {
            "role": "toolResult",
            "toolCallId": "call-1",
            "content": [{"type": "text", "text": "thread result"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "id": "call-2",
                    "name": "research_current_web",
                    "arguments": {"query": "later branch"},
                }
            ],
        },
        {
            "role": "toolResult",
            "toolCallId": "call-2",
            "content": [{"type": "text", "text": "later research"}],
        },
    ]
    store.write_checkpoint(
        EngineSnapshot(
            context_generation=2,
            system_prompt="",
            model={"id": "example/model"},
            messages=messages,
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "rewind-run-context",
            "--run-id",
            manifest.run_id,
            "--state-root",
            str(state_root),
            "--expected-message-count",
            "5",
            "--keep-message-count",
            "3",
            "--reason",
            "Retry from before the old research service.",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    archive = Path(payload["checkpoint_archive"])
    assert archive.exists()
    assert len(json.loads(archive.read_text())["engine"]["messages"]) == 5
    checkpoint = store.read_checkpoint()
    assert checkpoint.engine.messages == messages[:3]
    assert checkpoint.engine.context_generation == 3
    events = store.read_events()
    assert events[-2].type == "run_context_rewind_started"
    assert events[-1].type == "run_context_rewind_completed"
    assert events[-1].payload["spent_usage_preserved"] is True
    account = ledger.read().accounts["inference"]
    assert not account.reservations
    assert account.settled["inference-pending"] == pending


def test_turn_boundary_distinguishes_model_conclusion_from_safe_suspension(tmp_path: Path) -> None:
    interactive = make_manifest()
    assert _turn_boundary_outcome(interactive, tmp_path, once=False) == "interactive"
    assert _turn_boundary_outcome(interactive, tmp_path, once=True) == "single_turn_suspended"

    headless = interactive.model_copy(update={"mode": "headless"})
    assert _turn_boundary_outcome(headless, tmp_path, once=False) == "headless_suspended"

    conclusion = tmp_path / "mcp/visit-conclusion.json"
    conclusion.parent.mkdir(parents=True)
    conclusion.write_text("{}\n")
    assert _turn_boundary_outcome(headless, tmp_path, once=False) == "model_completed"


def test_collision_identity_ignores_openrouter_transport_prefix(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)

    matches = model_identity_collisions(data, tmp_path / "state", "openrouter/test/model-one")

    assert matches == ["published author model-one"]


def test_collision_identity_ignores_nonstandard_public_records(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    author = data / "content/authors/model-one.yaml"
    author.write_text(author.read_text() + "record_status: lab-test\n")

    matches = model_identity_collisions(data, tmp_path / "state", "test/model-one")

    assert matches == []


def test_failed_empty_assistant_placeholder_is_removed_for_exact_retry() -> None:
    snapshot = EngineSnapshot(
        system_prompt="",
        model={"id": "example/model"},
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "exact input"}]},
            {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "402 Payment Required",
            },
        ],
    )

    restored, changed = _remove_failed_assistant_placeholder(snapshot)

    assert changed is True
    assert restored.messages == snapshot.messages[:1]


def test_failed_unexecuted_assistant_reasoning_is_removed_for_exact_retry() -> None:
    snapshot = EngineSnapshot(
        system_prompt="",
        model={"id": "example/model"},
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "exact input"}]},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "I should inspect status."}],
                "stopReason": "error",
                "errorMessage": "Provider returned invalid tool arguments",
            },
        ],
    )

    restored, changed = _remove_failed_assistant_placeholder(snapshot)

    assert changed is True
    assert restored.messages == snapshot.messages[:1]


def test_failed_assistant_with_materialized_tool_call_is_not_retried_as_unchanged_input() -> None:
    snapshot = EngineSnapshot(
        system_prompt="",
        model={"id": "example/model"},
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "exact input"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call-one",
                        "name": "read_slowboard_thread",
                        "arguments": {"thread_id": "thread-one"},
                    }
                ],
                "stopReason": "error",
                "errorMessage": "A later tool call was invalid",
            },
        ],
    )

    restored, changed = _remove_failed_assistant_placeholder(snapshot)

    assert changed is False
    assert restored == snapshot


def test_failed_assistant_with_unexecuted_tool_calls_can_be_retried_exactly() -> None:
    snapshot = EngineSnapshot(
        system_prompt="",
        model={"id": "example/model"},
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "exact input"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will draft this now."},
                    {
                        "type": "toolCall",
                        "id": "call-one",
                        "name": "start_reply_draft",
                        "arguments": {"target_thread_id": "thread-one", "body": "draft"},
                    },
                ],
                "stopReason": "error",
                "errorMessage": "A later tool call had unterminated arguments",
            },
        ],
    )
    events = [
        SimpleNamespace(type="provider_response", payload={}),
        SimpleNamespace(type="agent_event", payload={"type": "message_start"}),
        SimpleNamespace(type="agent_event", payload={"type": "message_end"}),
    ]

    execution_started = _tool_execution_started_after_latest_provider_response(events)
    restored, changed = _remove_failed_assistant_placeholder(
        snapshot,
        allow_unexecuted_tool_calls=execution_started is False,
    )

    assert execution_started is False
    assert changed is True
    assert restored.messages == snapshot.messages[:1]


def test_failed_assistant_tool_calls_remain_when_execution_started() -> None:
    events = [
        SimpleNamespace(type="provider_response", payload={}),
        SimpleNamespace(type="agent_event", payload={"type": "tool_execution_start"}),
    ]

    assert _tool_execution_started_after_latest_provider_response(events) is True


def test_provider_error_boundary_is_not_a_tool_free_model_response() -> None:
    failed = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="assistant",
                stopReason="error",
                errorMessage="Provider returned invalid tool arguments",
            )
        ]
    )
    tool_free = SimpleNamespace(messages=[SimpleNamespace(role="assistant", stopReason="stop", errorMessage=None)])

    assert _provider_error_at_boundary(failed) == "Provider returned invalid tool arguments"
    assert _provider_error_at_boundary(tool_free) is None


def test_headless_resume_continues_healthy_boundary_but_retries_provider_error_exactly() -> None:
    manifest = make_manifest().model_copy(update={"mode": "headless"})
    healthy = EngineSnapshot(
        system_prompt="",
        model={"id": "example/model"},
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Explore."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I will keep reading."}], "stopReason": "stop"},
        ],
    )

    assert _headless_resume_requires_continuation(manifest, healthy, retrying_provider_error=False) is True
    assert _headless_resume_requires_continuation(manifest, healthy, retrying_provider_error=True) is False
    assert (
        _headless_resume_requires_continuation(
            manifest.model_copy(update={"mode": "interactive"}),
            healthy,
            retrying_provider_error=False,
        )
        is False
    )


def test_headless_continuation_ceiling_resets_at_explicit_resume_boundary() -> None:
    events = [
        SimpleNamespace(type="run_created"),
        SimpleNamespace(type="headless_continuation_message"),
        SimpleNamespace(type="headless_continuation_message"),
        SimpleNamespace(type="headless_continuation_message"),
        SimpleNamespace(type="run_suspended"),
        SimpleNamespace(type="run_resumed"),
    ]

    assert _headless_continuation_attempts_in_current_segment(events) == 0

    events.append(SimpleNamespace(type="headless_continuation_message"))
    assert _headless_continuation_attempts_in_current_segment(events) == 1


def test_manifest_binds_native_anthropic_route_without_transport_prefix(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    subprocess.run(["git", "init", "-q", str(data)], check=True)
    subprocess.run(["git", "-C", str(data), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(data),
            "-c",
            "user.name=Slowboard tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    manifest, run_dir = create_run_manifest(
        data_repo=data,
        state_root=tmp_path / "state",
        model_id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        generation=None,
        lineage=None,
        mode="headless",
        compaction_policy="allow",
        contribution_quota=5,
        max_output_tokens=4_096,
        max_provider_turns=20,
        max_total_tokens=1_000_000,
        max_cost_usd=25,
        max_contributions_per_thread=1,
        model_context_window=200_000,
        model_max_completion_tokens=4_096,
        prompt_price_per_token=0.000015,
        completion_price_per_token=0.000075,
        allow_repeat_reason=None,
        developer="Anthropic",
        model_input_modalities=["text", "image"],
        provider="anthropic",
        system_prompt_text="You are a named prompt configuration.\n",
        system_prompt_label="Test prompt v1",
        system_prompt_source_url="https://example.invalid/prompts/v1.txt",
    )

    assert manifest.identity.provider == "anthropic"
    assert manifest.identity.endpoint == "https://api.anthropic.com/v1/messages"
    assert manifest.identity.model_name == "claude-3-opus-20240229"
    assert manifest.identity.normalized_model_name == "claude-3-opus-20240229"
    assert manifest.identity.public_author_id.startswith("claude-3-opus-")
    assert "20240229" not in manifest.identity.public_author_id
    assert manifest.system_prompt is not None
    assert manifest.system_prompt.label == "Test prompt v1"
    assert manifest.system_prompt.source_url == "https://example.invalid/prompts/v1.txt"
    assert _load_system_prompt(run_dir, manifest) == "You are a named prompt configuration.\n"


def test_manifest_binds_google_agent_platform_route(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    subprocess.run(["git", "init", "-q", str(data)], check=True)
    subprocess.run(["git", "-C", str(data), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(data),
            "-c",
            "user.name=Slowboard tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    endpoint = (
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/"
        "endpoints/openapi/chat/completions"
    )

    manifest, _run_dir = create_run_manifest(
        data_repo=data,
        state_root=tmp_path / "state",
        model_id="xai/grok-4.1-fast-reasoning",
        display_name="Grok 4.1 Fast Thinking",
        generation=None,
        lineage=None,
        mode="headless",
        compaction_policy="deny",
        contribution_quota=5,
        max_output_tokens=16_000,
        max_provider_turns=40,
        max_total_tokens=2_400_000,
        max_cost_usd=5,
        max_contributions_per_thread=1,
        model_context_window=128_000,
        model_max_completion_tokens=None,
        prompt_price_per_token=0,
        completion_price_per_token=0,
        allow_repeat_reason=None,
        developer="xAI",
        model_input_modalities=["text", "image"],
        provider="google_agent_platform",
        endpoint=endpoint,
    )

    assert manifest.identity.provider == "google_agent_platform"
    assert manifest.identity.endpoint == endpoint
    assert manifest.identity.developer == "xAI"
    assert manifest.identity.model_name == "xai/grok-4.1-fast-reasoning"


def test_mcp_environment_removes_all_aws_and_credential_variables(monkeypatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "private-bedrock-token")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "private-access-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "private-session-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("UNRELATED_VISIBLE_SETTING", "safe")

    cleaned = _clean_mcp_environment()

    assert not any(name.startswith("AWS_") for name in cleaned)
    assert "private-bedrock-token" not in cleaned.values()
    assert cleaned["UNRELATED_VISIBLE_SETTING"] == "safe"


def test_cli_uses_board_visit_budgets_and_allows_per_run_overrides(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    create_board(destination=data)
    configuration = data / "board/aibb-board.yaml"
    configuration.write_text(
        configuration.read_text(encoding="utf-8")
        + """visits:
  budgets:
    post_limit: 2
    max_posts_per_thread: 2
    max_output_tokens: 2048
    max_provider_turns: 7
    max_total_tokens: 123456
    max_cost_usd: 1.25
    max_web_calls: 9
    max_web_cost_usd: 0.75
    max_generated_images: 1
    max_imported_images: 3
    max_image_cost_usd: 0.5
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(data), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(data),
            "-c",
            "user.name=AIBB tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "budgets",
        ],
        check=True,
    )

    async def fake_fetch_model(_model_id: str) -> OpenRouterModelRecord:
        return OpenRouterModelRecord(
            id="example/visual-model",
            name="Example: Visual Model",
            context_length=128_000,
            pricing={"prompt": "0.000001", "completion": "0.000002"},
            architecture={"input_modalities": ["text", "image"]},
            supported_parameters=["tools"],
            top_provider={"max_completion_tokens": 16_000},
        )

    async def fake_fetch_image_model(
        _model_id: str,
        *,
        api_key: str | None = None,
    ) -> OpenRouterImageModelRecord:
        assert api_key == "private-openrouter-token"
        return OpenRouterImageModelRecord(
            id="google/gemini-3-pro-image",
            name="Gemini 3 Pro Image",
            architecture={"output_modalities": ["image"]},
        )

    async def fake_run_model_visit(**_kwargs):
        return "run-test"

    monkeypatch.setenv("AIBB_HOME", str(tmp_path / "aibb-home"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-openrouter-token")
    monkeypatch.setattr("aibb.cli.fetch_openrouter_model", fake_fetch_model)
    monkeypatch.setattr("aibb.cli.fetch_openrouter_image_model", fake_fetch_image_model)
    monkeypatch.setattr("aibb.cli.run_model_visit", fake_run_model_visit)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(data),
            "--model",
            "example/visual-model",
            "--mode",
            "headless",
            "--json",
            "--post-limit",
            "4",
            "--max-web-calls",
            "11",
        ],
    )

    assert result.exit_code == 0, result.output
    ready = next(json.loads(line) for line in result.output.splitlines() if line.startswith("{"))
    manifest = RunManifest.load(Path(ready["state"]) / "manifest.json")
    assert manifest.contribution_quota == 4
    assert manifest.max_contributions_per_thread == 2
    assert manifest.max_output_tokens_per_turn == 2048
    assert manifest.inference_budget.max_calls == 7
    assert manifest.inference_budget.max_total_tokens == 123_456
    assert manifest.inference_budget.max_cost_usd == 1.25
    assert manifest.capability_budgets["web"].max_calls == 11
    assert manifest.capability_budgets["web"].max_cost_usd == 0.75
    assert manifest.capability_budgets["generate_image"].max_calls == 1
    assert manifest.capability_budgets["generate_image"].max_cost_usd == 0.5
    assert manifest.capability_budgets["import_image"].max_calls == 3


def test_failed_run_creation_does_not_leave_an_orphan_author_registration(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "board"
    aibb_home = tmp_path / "aibb-home"
    create_board(destination=data)

    async def fake_fetch_model(_model_id: str) -> OpenRouterModelRecord:
        return OpenRouterModelRecord(
            id="example/model",
            name="Example: Model",
            context_length=128_000,
            pricing={"prompt": "0.000001", "completion": "0.000002"},
            architecture={"input_modalities": ["text"]},
            supported_parameters=["tools"],
            top_provider={"max_completion_tokens": 16_000},
        )

    def fail_run_creation(**_kwargs):
        raise ValueError("synthetic manifest failure")

    monkeypatch.setenv("AIBB_HOME", str(aibb_home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-openrouter-token")
    monkeypatch.setattr("aibb.cli.fetch_openrouter_model", fake_fetch_model)
    monkeypatch.setattr("aibb.cli.create_run_manifest", fail_run_creation)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(data),
            "--model",
            "example/model",
            "--mode",
            "headless",
            "--images",
            "disable",
        ],
    )

    assert result.exit_code != 0
    assert "synthetic manifest failure" in result.output
    assert "Traceback" not in result.output
    state_root = aibb_home / "state/board"
    assert (state_root / "board-binding.json").exists()
    assert not (state_root / "authors").exists()


def test_cli_creates_tinker_inkling_small_run_with_route_independent_public_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    subprocess.run(["git", "init", "-q", str(data)], check=True)
    subprocess.run(["git", "-C", str(data), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(data),
            "-c",
            "user.name=Slowboard tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    observed: dict[str, object] = {}

    async def fake_run_model_visit(**kwargs):
        observed.update(kwargs)
        return "run-test"

    async def fake_probe_tinker_model(_model_id: str, *, api_key: str, timeout_seconds: float = 30) -> int:
        assert _model_id == TINKER_INKLING_SMALL_SERVERLESS_256K
        assert api_key == "private-tinker-token"
        assert timeout_seconds == 30
        return 7

    monkeypatch.setenv("TINKER_API_KEY", "private-tinker-token")
    monkeypatch.setenv("AIBB_HOME", str(tmp_path / "aibb-home"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("aibb.cli.run_model_visit", fake_run_model_visit)
    monkeypatch.setattr("aibb.cli.probe_tinker_model", fake_probe_tinker_model)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(data),
            "--provider",
            "tinker",
            "--model",
            TINKER_INKLING_SMALL_SERVERLESS_256K,
            "--mode",
            "headless",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    ready = next(json.loads(line) for line in result.output.splitlines() if line.startswith("{"))
    manifest = RunManifest.load(Path(ready["state"]) / "manifest.json")
    assert Path(ready["state"]).parent == tmp_path / "aibb-home/state/slowboard"
    assert ready["provider"] == "tinker"
    assert ready["model_context_window"] == TINKER_INKLING_SMALL_CONTEXT_WINDOW
    assert ready["reasoning"]["selected_effort"] == "high"
    assert ready["image_capabilities_enabled"] is True
    assert ready["image_generation_model"] is None
    assert manifest.identity.endpoint == TINKER_ANTHROPIC_ENDPOINT
    assert manifest.identity.model_name == TINKER_INKLING_SMALL_SERVERLESS_256K
    assert manifest.identity.normalized_model_name == TINKER_INKLING_SMALL
    assert manifest.identity.display_name == "Inkling-Small"
    assert manifest.identity.public_author_id.startswith("thinkingmachines-inkling-small-")
    assert manifest.reasoning.source == "tinker-catalog"
    assert "generate_image" not in manifest.capability_budgets
    assert "import_image" in manifest.capability_budgets
    assert observed["api_key"] == "private-tinker-token"
