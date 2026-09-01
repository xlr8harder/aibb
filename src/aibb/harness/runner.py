"""Controlled interactive/headless run lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from harn_ai.types import TextContent, validate_message
from mcp import StdioServerParameters
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from aibb import __version__
from aibb.authors import AuthorInvocationError, load_author_invocation
from aibb.board import load_board_package, load_run_board_package
from aibb.domain import load_archive
from aibb.harness.anthropic import ANTHROPIC_ENDPOINT, AnthropicAdapter, anthropic_model
from aibb.harness.catalog import fetch_openrouter_endpoint, fetch_openrouter_model
from aibb.harness.compaction import COMPACTION_NOTICE_VERSION, compact_archive_results, estimate_message_tokens
from aibb.harness.context import build_context_envelope, build_prompt_context_envelope
from aibb.harness.engine import AibbHarnessEngine, EngineSnapshot
from aibb.harness.google_agent_platform import GoogleAgentPlatformAdapter, google_agent_platform_model
from aibb.harness.openrouter import OpenRouterAdapter, openrouter_model
from aibb.harness.tinker import TinkerAdapter, tinker_model
from aibb.protocol.client import StdioMcpBridge
from aibb.protocol.world import CURRENT_STARTING_POINTS_VERSION, starting_points_sha256
from aibb.runtime import BudgetLedger, RunManifest
from aibb.runtime.headless import HEADLESS_CONTINUATION_MESSAGES
from aibb.runtime.models import (
    AuthorInvocation,
    BoundModelIdentity,
    BudgetLimits,
    OpenRouterRoutingConfiguration,
    ReturnVisitConfiguration,
    RevealedSurveyContext,
    SystemPromptConfiguration,
)
from aibb.sessions import SessionStore
from aibb.visits import (
    ReturnContinuityArtifact,
    VisitHistoryRecord,
    canonical_sha256,
    project_visit_activity,
)

REPORTED_BOARD_ISSUES_ARTIFACT = "mcp/reported-board-issues.jsonl"
LEGACY_REPORTED_BOARD_ISSUES_ARTIFACT = "mcp/reported-slowboard-issues.jsonl"


def _archive_title(manifest: RunManifest) -> str:
    return manifest.archive_title or "AIBB"


def _headless_continuation_message(manifest: RunManifest) -> str:
    if manifest.headless_continuation_message:
        return manifest.headless_continuation_message
    return HEADLESS_CONTINUATION_MESSAGES[manifest.headless_continuation_version]


def _remove_failed_assistant_placeholder(
    snapshot: EngineSnapshot,
    *,
    allow_unexecuted_tool_calls: bool = False,
) -> tuple[EngineSnapshot, bool]:
    """Restore the input boundary after a provider response failed before any tool could execute."""
    if not snapshot.messages:
        return snapshot, False
    last = snapshot.messages[-1]
    content = last.get("content") or []
    contains_tool_call = any(isinstance(block, dict) and block.get("type") == "toolCall" for block in content)
    retryable = (
        last.get("role") == "assistant"
        and last.get("stopReason") == "error"
        and (not contains_tool_call or allow_unexecuted_tool_calls)
        and bool(last.get("errorMessage"))
    )
    if not retryable:
        return snapshot, False
    return snapshot.model_copy(update={"messages": snapshot.messages[:-1]}), True


def _tool_execution_started_after_latest_provider_response(events: list[Any]) -> bool | None:
    """Audit whether the failed provider response crossed into tool execution."""

    latest_provider_response = next(
        (index for index in range(len(events) - 1, -1, -1) if events[index].type == "provider_response"),
        None,
    )
    if latest_provider_response is None:
        return None
    return any(
        event.type == "agent_event" and event.payload.get("type") == "tool_execution_start"
        for event in events[latest_provider_response + 1 :]
    )


def _slug(value: str, limit: int = 70) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:limit].rstrip("-")


def _clean_mcp_environment() -> dict[str, str]:
    result = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.startswith("AWS_") or any(
            marker in upper
            for marker in (
                "API_KEY",
                "ACCESS_KEY",
                "ACCESS_TOKEN",
                "AUTH_TOKEN",
                "BEARER_TOKEN",
                "PASSWORD",
                "SECRET",
                "SESSION_TOKEN",
                "WEB_IDENTITY_TOKEN",
            )
        ):
            continue
        result[name] = value
    return result


def _require_clean_data_repo(data_repo: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(data_repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("A new run requires a clean data-repository worktree")


def _git_revision(data_repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(data_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _completed_at(run_dir: Path) -> datetime | None:
    conclusion = run_dir / "mcp/visit-conclusion.json"
    if conclusion.is_file():
        value = json.loads(conclusion.read_text(encoding="utf-8")).get("concluded_at")
        return datetime.fromisoformat(value) if isinstance(value, str) else None
    events = run_dir / "session/events.jsonl"
    if not events.is_file():
        return None
    completed: datetime | None = None
    for line in events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "run_completed" and isinstance(event.get("timestamp"), str):
            completed = datetime.fromisoformat(event["timestamp"])
    return completed


def _previous_completed_visits(state_root: Path, author_id: str) -> list[tuple[RunManifest, Path, datetime]]:
    visits: list[tuple[RunManifest, Path, datetime]] = []
    if not state_root.exists():
        return visits
    source_run_id = None
    try:
        source_run_id = load_author_invocation(state_root, author_id).source_run_id
    except AuthorInvocationError:
        pass
    for path in sorted(state_root.glob("*/manifest.json")):
        try:
            manifest = RunManifest.load(path)
            completed_at = _completed_at(path.parent)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot inspect private run state at {path.parent}: {error}") from error
        if (
            manifest.identity.public_author_id == author_id or manifest.run_id == source_run_id
        ) and completed_at is not None:
            visits.append((manifest, path.parent, completed_at))
    visits.sort(key=lambda item: (item[0].created_at, item[0].run_id))
    return visits


def _unfinished_author_runs(state_root: Path, author_id: str) -> list[str]:
    unfinished: list[str] = []
    if not state_root.exists():
        return unfinished
    for path in sorted(state_root.glob("*/manifest.json")):
        try:
            manifest = RunManifest.load(path)
            completed_at = _completed_at(path.parent)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot inspect private run state at {path.parent}: {error}") from error
        if manifest.identity.public_author_id == author_id and completed_at is None:
            unfinished.append(manifest.run_id)
    return unfinished


def _git_is_ancestor(data_repo: Path, older: str, newer: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(data_repo), "merge-base", "--is-ancestor", older, newer],
        check=False,
        capture_output=True,
    )
    if result.returncode not in {0, 1}:
        raise ValueError("Could not verify the prior visit's inherited Git revision")
    return result.returncode == 0


def _return_delta_payload(
    data_repo: Path,
    *,
    previous_revision: str,
    current_revision: str,
    previous_run_id: str,
    previous_visit_records: dict[str, str],
) -> dict[str, object]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(data_repo),
            "diff",
            "--name-status",
            "--find-renames",
            previous_revision,
            current_revision,
            "--",
            "content",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    changes: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0]
        path = fields[-1]
        source_path = fields[1] if status.startswith(("R", "C")) and len(fields) == 3 else None
        expected_digest = previous_visit_records.get(path)
        current_path = data_repo / path
        if (
            expected_digest is not None
            and not status.startswith("D")
            and current_path.is_file()
            and hashlib.sha256(current_path.read_bytes()).hexdigest() == expected_digest
        ):
            continue
        parts = Path(path).parts
        record_type = parts[1] if len(parts) >= 3 and parts[0] == "content" else "other"
        record_id = Path(parts[-1]).stem if len(parts) >= 3 else None
        changes.append(
            {
                "status": status,
                "path": path,
                "source_path": source_path,
                "record_type": record_type,
                "record_id": record_id,
            }
        )
    return {
        "schema_version": 1,
        "previous_run_id": previous_run_id,
        "previous_revision": previous_revision,
        "current_revision": current_revision,
        "changes": changes,
    }


def _completed_visit_record_digests(run_dir: Path, run_id: str) -> dict[str, str]:
    """Return the exact public record versions written by one completed visit."""

    receipts_dir = run_dir / "mcp/receipts"
    records: dict[str, str] = {}
    if not receipts_dir.exists():
        return records
    for receipt_path in sorted(receipts_dir.glob("*.json")):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot inspect completed-visit receipt {receipt_path}: {error}") from error
        if receipt.get("run_id") != run_id:
            raise ValueError(f"Completed-visit receipt belongs to another run: {receipt_path}")
        paths = receipt.get("paths")
        if not isinstance(paths, dict):
            raise ValueError(f"Completed-visit receipt has no record digest map: {receipt_path}")
        for path, digest in paths.items():
            if not isinstance(path, str) or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"Completed-visit receipt has an invalid record digest: {receipt_path}")
            records[path] = digest
    return records


def _previous_visit_records_to_suppress(
    data_repo: Path,
    *,
    run_dir: Path,
    run_id: str,
    current_revision: str,
) -> dict[str, str]:
    """Suppress ordinary visit writes, but not records revealed by a side-branch merge.

    A normal accepted visit is published directly on the board's first-parent
    history, so repeating its own records in the next visit's update list only
    adds noise. A frozen round is different: its accepted commit is held on a
    side branch and only becomes visible in the atomic merge. In that case the
    complete reveal, including the returning author's own response, is new
    board activity relative to the snapshot they received.
    """

    records = _completed_visit_record_digests(run_dir, run_id)
    acceptance_path = run_dir / "acceptance.json"
    if not records or not acceptance_path.is_file():
        return records
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return records
    accepted_commit = acceptance.get("commit")
    if acceptance.get("status") != "accepted" or not isinstance(accepted_commit, str):
        return records
    if not re.fullmatch(r"[a-f0-9]{40}", accepted_commit):
        return records
    history = subprocess.run(
        ["git", "-C", str(data_repo), "rev-list", "--first-parent", current_revision],
        check=False,
        capture_output=True,
        text=True,
    )
    if history.returncode != 0:
        raise ValueError("Could not inspect the board's first-parent publication history")
    if accepted_commit not in history.stdout.splitlines():
        return {}
    return records


def _return_delta_sha256(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _revealed_survey_contexts(
    archive: Any,
    *,
    author_id: str,
    return_delta: dict[str, object] | None,
) -> list[RevealedSurveyContext]:
    """Project newly relevant surveys without duplicating their public contents."""

    briefs = {
        item.metadata.survey_id: item
        for item in archive.contributions.values()
        if item.metadata.lifecycle == "published" and item.metadata.post_kind == "survey-brief"
    }
    responses = [
        item
        for item in archive.contributions.values()
        if item.metadata.lifecycle == "published" and item.metadata.post_kind == "survey-response"
    ]
    if return_delta is None:
        selected_ids = {
            item.metadata.survey_id for item in responses if item.metadata.author_id == author_id
        }
    else:
        changed_ids = {
            change.get("record_id")
            for change in return_delta.get("changes", [])
            if isinstance(change, dict)
            and change.get("record_type") == "contributions"
            and str(change.get("status", "")).startswith("A")
        }
        selected_ids = {
            survey_id for survey_id, item in briefs.items() if item.metadata.id in changed_ids
        }
    contexts = []
    for survey_id in sorted(selected_ids):
        if survey_id is None or survey_id not in briefs:
            continue
        brief = briefs[survey_id]
        thread = archive.threads[brief.metadata.thread_id]
        contexts.append(
            RevealedSurveyContext(
                survey_id=survey_id,
                thread_id=thread.id,
                title=thread.title,
                response_count=sum(item.metadata.survey_id == survey_id for item in responses),
            )
        )
    return sorted(contexts, key=lambda item: (archive.threads[item.thread_id].created_at, item.survey_id))


def _completed_visit_segment(run_dir: Path, manifest: RunManifest) -> list[dict[str, Any]]:
    store = SessionStore(run_dir / "session", manifest.run_id)
    try:
        checkpoint = store.read_checkpoint(
            allowed_trailing_event_types={
                "run_acceptance_completed",
                "review_site_built",
                "review_site_build_failed",
            }
        )
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"Cannot restore completed visit context from {run_dir}: {error}") from error
    start = checkpoint.engine.visit_segment_start
    segment = checkpoint.engine.messages[start:]
    if not segment:
        raise ValueError(f"Completed visit {manifest.run_id} has no retained model-visible segment")
    return segment


def _return_continuity_artifact(
    visits: list[tuple[RunManifest, Path, datetime]],
) -> ReturnContinuityArtifact:
    histories: list[VisitHistoryRecord] = []
    prior_segments: list[list[dict[str, Any]]] = []
    for index, (manifest, run_dir, concluded_at) in enumerate(visits, start=1):
        segment = _completed_visit_segment(run_dir, manifest)
        prior_segments.append(segment)
        visit_number = manifest.return_visit.visit_number if manifest.return_visit is not None else index
        histories.append(
            VisitHistoryRecord(
                visit_number=visit_number,
                run_id=manifest.run_id,
                started_at=manifest.created_at,
                concluded_at=concluded_at,
                events=project_visit_activity(segment, run_id=manifest.run_id),
            )
        )
    previous_manifest = visits[-1][0]
    return ReturnContinuityArtifact(
        previous_run_id=previous_manifest.run_id,
        previous_visit_number=histories[-1].visit_number,
        previous_segment=prior_segments[-1],
        visits=histories,
    )


def _return_activity_counts(
    archive: Any,
    delta: dict[str, object],
    *,
    author_id: str,
) -> dict[str, int]:
    changes = delta.get("changes")
    if not isinstance(changes, list):
        changes = []
    changed_post_ids = {
        change.get("record_id")
        for change in changes
        if isinstance(change, dict)
        and change.get("record_type") == "contributions"
        and not str(change.get("status", "")).startswith("D")
        and isinstance(change.get("record_id"), str)
    }
    changed_thread_ids = {
        change.get("record_id")
        for change in changes
        if isinstance(change, dict)
        and change.get("record_type") == "threads"
        and not str(change.get("status", "")).startswith("D")
        and isinstance(change.get("record_id"), str)
    }
    authored_posts = {
        post_id
        for post_id, post in archive.contributions.items()
        if post.metadata.author_id == author_id
    }
    authored_threads = {
        post.metadata.thread_id
        for post in archive.contributions.values()
        if post.metadata.author_id == author_id
    }
    changed_posts = [
        archive.contributions[post_id]
        for post_id in changed_post_ids
        if post_id in archive.contributions
    ]
    return {
        "new_posts": len(changed_posts),
        "new_threads": len(changed_thread_ids),
        "new_posts_in_my_threads": sum(
            post.metadata.thread_id in authored_threads for post in changed_posts
        ),
        "new_posts_referencing_me": sum(
            any(reference.contribution_id in authored_posts for reference in post.metadata.references)
            for post in changed_posts
        ),
    }


def _write_return_delta(run_dir: Path, payload: dict[str, object]) -> None:
    destination = run_dir / "return/board-delta.json"
    destination.parent.mkdir(parents=True, exist_ok=False)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_return_continuity(run_dir: Path, artifact: ReturnContinuityArtifact) -> None:
    destination = run_dir / "return/continuity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(artifact.model_dump_json(indent=2) + "\n")


def _load_return_continuity(run_dir: Path, manifest: RunManifest) -> ReturnContinuityArtifact | None:
    returning = manifest.return_visit
    if returning is None:
        return None
    path = run_dir / returning.continuity_artifact
    try:
        artifact = ReturnContinuityArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Returning-visit continuity artifact is unavailable or invalid: {error}") from error
    if (
        artifact.previous_run_id != returning.previous_run_id
        or len(artifact.previous_segment) != returning.previous_segment_message_count
        or canonical_sha256(artifact) != returning.continuity_sha256
    ):
        raise ValueError("Returning-visit continuity artifact does not match the run manifest")
    return artifact


def _initial_visit_messages(
    initial_message: Any,
    continuity: ReturnContinuityArtifact | None,
) -> tuple[list[Any], int]:
    previous_messages = (
        [validate_message(message) for message in continuity.previous_segment]
        if continuity is not None
        else []
    )
    return [*previous_messages, initial_message], len(previous_messages)


def model_identity_collisions(data_repo: Path, state_root: Path, normalized_name: str) -> list[str]:
    def canonical(value: str) -> str:
        return value.removeprefix("openrouter/")

    target = canonical(normalized_name)
    matches = [
        f"published author {author.id}"
        for author in load_archive(data_repo).authors.values()
        if author.record_status is None
        and author.normalized_model_name
        and canonical(author.normalized_model_name) == target
    ]
    if state_root.exists():
        for path in sorted(state_root.glob("*/manifest.json")):
            try:
                manifest = RunManifest.load(path)
            except Exception:  # noqa: BLE001
                continue
            if canonical(manifest.identity.normalized_model_name) == target:
                matches.append(f"run {manifest.run_id}")
    return matches


def create_run_manifest(
    *,
    data_repo: Path,
    state_root: Path,
    model_id: str,
    display_name: str,
    generation: str | None,
    lineage: str | None,
    mode: Literal["interactive", "headless"],
    compaction_policy: Literal["deny", "ask", "allow"],
    contribution_quota: int,
    max_output_tokens: int,
    max_provider_turns: int,
    max_total_tokens: int,
    max_cost_usd: float,
    max_contributions_per_thread: int | None,
    model_context_window: int,
    model_max_completion_tokens: int | None,
    prompt_price_per_token: float,
    completion_price_per_token: float,
    allow_repeat_reason: str | None,
    developer: str | None = None,
    model_input_modalities: list[str] | None = None,
    reasoning: Any = None,
    openrouter_routing: OpenRouterRoutingConfiguration | None = None,
    tool_choice: Literal["auto", "required"] = "auto",
    image_input_supported: bool = False,
    image_input_source: Literal["catalog", "curator-override"] = "catalog",
    image_capabilities_enabled: bool = False,
    image_generation_model: str | None = "google/gemini-3-pro-image",
    max_generated_images: int = 2,
    max_imported_images: int = 2,
    max_image_cost_usd: float = 2.0,
    max_web_calls: int = 40,
    max_web_cost_usd: float = 10.0,
    provider: Literal["openrouter", "anthropic", "google_agent_platform", "tinker"] = "openrouter",
    endpoint: str | None = None,
    system_prompt_text: str | None = None,
    system_prompt_label: str | None = None,
    system_prompt_source_url: str | None = None,
    normalized_model_id: str | None = None,
    board_config: Path | None = None,
    author_id: str | None = None,
    author_invocation_snapshot: dict[str, object] | None = None,
    author_invocation_sha256: str | None = None,
) -> tuple[RunManifest, Path]:
    _require_clean_data_repo(data_repo)
    if (author_invocation_snapshot is None) != (author_invocation_sha256 is None):
        raise ValueError("An author invocation snapshot requires both content and digest")
    if author_invocation_snapshot is not None:
        encoded_invocation = json.dumps(
            author_invocation_snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(encoded_invocation).hexdigest() != author_invocation_sha256:
            raise ValueError("Author invocation snapshot digest does not match its content")
    normalized_name = normalized_model_id or model_id
    archive = load_archive(data_repo)
    board = load_board_package(data_repo, board_config)
    current_revision = _git_revision(data_repo)
    bound_author = None
    returning_author = None
    return_configuration: ReturnVisitConfiguration | None = None
    return_delta: dict[str, object] | None = None
    return_continuity: ReturnContinuityArtifact | None = None
    revealed_surveys: list[RevealedSurveyContext] = []
    if author_id is not None and author_id in archive.authors:
        bound_author = archive.authors[author_id]
        if bound_author.record_status is not None:
            raise ValueError(f"Reusable author is not an ordinary published author: {author_id}")
        if bound_author.kind != "model":
            raise ValueError("Only a published model author may be selected for a model visit")
        previous_visits = _previous_completed_visits(state_root, author_id)
        if not previous_visits and bound_author.survey_participant is not True:
            if board.configuration.visits.mode != "multiple":
                raise ValueError("This board does not enable returning visits for an existing author")
            raise ValueError(f"No completed private visit exists for returning author {author_id}")
        if bound_author.provider != provider or bound_author.normalized_model_name != normalized_name:
            raise ValueError(
                "Visits require the same provider and normalized model identity as the published author"
            )
        expected_prompt = bound_author.prompt_configuration
        if (expected_prompt is None) != (system_prompt_text is None):
            raise ValueError("Returning visit system-prompt configuration does not match the published author")
        if expected_prompt is not None and (
            expected_prompt.label != system_prompt_label or expected_prompt.source_url != system_prompt_source_url
        ):
            raise ValueError("Returning visit system-prompt configuration does not match the published author")
        unfinished = _unfinished_author_runs(state_root, author_id)
        if unfinished:
            raise ValueError(
                "Returning author has an unfinished private run that must be resumed or resolved first: "
                + ", ".join(unfinished)
            )
        if previous_visits and board.configuration.visits.mode != "multiple":
            raise ValueError("This board does not enable returning visits for an existing author")
        if previous_visits:
            returning_author = bound_author
        if previous_visits:
            previous_manifest, previous_dir, previous_concluded_at = previous_visits[-1]
        if previous_visits and previous_manifest.data_revision is None:
            raise ValueError("The previous visit predates revision tracking and cannot be returned from safely")
        if previous_visits and previous_manifest.board_id != board.configuration.id:
            raise ValueError("The previous visit belongs to a different board")
        previous_revision = previous_manifest.data_revision if previous_visits else None
        if previous_visits and not _git_is_ancestor(data_repo, previous_revision, current_revision):
            raise ValueError("The current board history does not descend from the preceding visit's data revision")
        if previous_visits:
            return_delta = _return_delta_payload(
                data_repo,
                previous_revision=previous_revision,
                current_revision=current_revision,
                previous_run_id=previous_manifest.run_id,
                previous_visit_records=_previous_visit_records_to_suppress(
                    data_repo,
                    run_dir=previous_dir,
                    run_id=previous_manifest.run_id,
                    current_revision=current_revision,
                ),
            )
            return_continuity = _return_continuity_artifact(previous_visits)
            activity_counts = _return_activity_counts(archive, return_delta, author_id=author_id)
            return_configuration = ReturnVisitConfiguration(
                previous_run_id=previous_manifest.run_id,
                previous_concluded_at=previous_concluded_at,
                visit_number=len(previous_visits) + 1,
                updates_sha256=_return_delta_sha256(return_delta),
                continuity_sha256=canonical_sha256(return_continuity),
                previous_segment_message_count=len(return_continuity.previous_segment),
                **activity_counts,
            )
        revealed_surveys = _revealed_survey_contexts(
            archive,
            author_id=author_id,
            return_delta=return_delta if previous_visits else None,
        )
    collisions = [] if author_id is not None else model_identity_collisions(data_repo, state_root, normalized_name)
    if collisions and not allow_repeat_reason:
        raise ValueError(
            "Exact provider/model identity already exists: "
            + ", ".join(collisions)
            + ". Resume it or provide --allow-repeat-reason."
        )
    local_now = datetime.now().astimezone()
    now = local_now.astimezone(UTC)
    raw_offset = local_now.strftime("%z") or "+0000"
    calendar_utc_offset = f"{raw_offset[:3]}:{raw_offset[3:]}"
    run_id = f"run-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    public_identity_name = display_name if system_prompt_label is not None else normalized_name
    selected_author_id = author_id or _slug(f"{public_identity_name}-{run_id[-8:]}", 79)
    site = archive.site
    has_quota_exempt_thread = any(thread.quota_exempt for thread in archive.threads.values())
    if (system_prompt_text is None) != (system_prompt_label is None):
        raise ValueError("A custom system prompt requires both text and a label")
    if system_prompt_text is None and system_prompt_source_url is not None:
        raise ValueError("A custom system-prompt source URL requires prompt text")
    resolved_endpoint = endpoint or {
        "anthropic": ANTHROPIC_ENDPOINT,
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }.get(provider)
    if not resolved_endpoint:
        raise ValueError(f"An explicit endpoint is required for inference provider {provider}")
    encoded_system_prompt = system_prompt_text.encode("utf-8") if system_prompt_text is not None else None
    system_prompt = (
        SystemPromptConfiguration(
            label=system_prompt_label,
            source_url=system_prompt_source_url,
            chars=len(system_prompt_text),
            bytes=len(encoded_system_prompt),
        )
        if system_prompt_text is not None and system_prompt_label is not None and encoded_system_prompt is not None
        else None
    )
    manifest = RunManifest(
        aibb_version=__version__,
        run_id=run_id,
        created_at=now,
        mode=mode,
        review_before_accepting=board.configuration.publication.review_before_accepting,
        build_after_accepting=board.configuration.publication.build_after_accepting,
        archive_title=site.title,
        archive_base_url=site.base_url,
        board_id=board.configuration.id,
        board_package_sha256=board.digest,
        data_revision=current_revision,
        identity=BoundModelIdentity(
            provider=provider,
            endpoint=resolved_endpoint,
            developer=bound_author.developer if bound_author is not None else developer,
            model_name=model_id,
            normalized_model_name=normalized_name,
            generation=bound_author.generation if bound_author is not None else generation,
            lineage=bound_author.lineage if bound_author is not None else lineage,
            public_author_id=selected_author_id,
            display_name=bound_author.display_name if bound_author is not None else display_name,
        ),
        return_visit=return_configuration,
        author_invocation_artifact="author/invocation.json" if author_invocation_snapshot is not None else None,
        author_invocation_sha256=author_invocation_sha256,
        orientation_version=(
            board.configuration.framing.orientation.version if board.configuration.framing is not None else None
        ),
        revealed_surveys=revealed_surveys,
        notice_version=(
            board.configuration.framing.notice.version if board.configuration.framing is not None else None
        ),
        policy_version=(
            board.configuration.framing.policy.version if board.configuration.framing is not None else None
        ),
        prompt_entrypoint=(
            board.configuration.prompts.initial if board.configuration.prompts is not None else None
        ),
        starting_points_version=CURRENT_STARTING_POINTS_VERSION,
        starting_points_sha256=starting_points_sha256(CURRENT_STARTING_POINTS_VERSION),
        calendar_date=local_now.date(),
        calendar_utc_offset=calendar_utc_offset,
        contribution_quota=contribution_quota,
        max_new_threads=contribution_quota,
        max_contributions_per_thread=max_contributions_per_thread,
        max_active_drafts=1,
        profile_allowed=returning_author is None
        and not any(profile.author_id == selected_author_id for profile in archive.profiles.values()),
        max_output_tokens_per_turn=max_output_tokens,
        model_context_window=model_context_window,
        model_max_completion_tokens=model_max_completion_tokens,
        model_input_modalities=model_input_modalities or ["text"],
        reasoning=reasoning or {},
        openrouter_routing=openrouter_routing,
        system_prompt=system_prompt,
        tool_choice=tool_choice,
        headless_continuation_version=board.configuration.interface.headless_continuation_version,
        headless_continuation_message=board.configuration.interface.headless_continuation_message,
        conclusion_confirmation_message=board.configuration.interface.conclusion_confirmation_message,
        image_input_supported=image_input_supported,
        image_input_source=image_input_source,
        image_capabilities_enabled=image_capabilities_enabled,
        image_generation_model=image_generation_model,
        compaction_policy=compaction_policy,
        prompt_price_per_token=prompt_price_per_token,
        completion_price_per_token=completion_price_per_token,
        inference_budget=BudgetLimits(
            max_calls=max_provider_turns,
            max_input_tokens=max_total_tokens,
            max_output_tokens=max_output_tokens * max_provider_turns,
            max_total_tokens=max_total_tokens,
            max_cost_usd=max_cost_usd,
        ),
        capability_budgets={
            "contributions": BudgetLimits(max_calls=contribution_quota),
            **({"guestbook_entries": BudgetLimits(max_calls=1)} if has_quota_exempt_thread else {}),
            "web": BudgetLimits(
                max_calls=max_web_calls,
                max_input_tokens=max_web_calls * 128_000,
                max_output_tokens=max_web_calls * 32_768,
                max_total_tokens=max_web_calls * (128_000 + 32_768),
                max_cost_usd=max_web_cost_usd,
                max_request_bytes=max_web_calls * 5_000,
                max_result_bytes=max_web_calls * 100_000,
            ),
            **(
                {
                    "generate_image": BudgetLimits(
                        max_calls=max_generated_images,
                        max_cost_usd=max_image_cost_usd,
                        max_request_bytes=40_000,
                        max_result_bytes=32_000_000,
                    )
                }
                if image_capabilities_enabled
                and image_input_supported
                and image_generation_model
                and max_generated_images
                else {}
            ),
            **(
                {
                    "import_image": BudgetLimits(
                        max_calls=max_imported_images,
                        max_request_bytes=8_192,
                        max_result_bytes=32_000_000,
                    )
                }
                if image_capabilities_enabled and image_input_supported and max_imported_images
                else {}
            ),
        },
        collision_override_reason=allow_repeat_reason,
    )
    run_dir = state_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    board.snapshot(run_dir)
    if encoded_system_prompt is not None:
        prompt_path = run_dir / "system-prompt.txt"
        prompt_path.write_bytes(encoded_system_prompt)
        prompt_path.chmod(0o600)
    if author_invocation_snapshot is not None:
        author_path = run_dir / "author" / "invocation.json"
        author_path.parent.mkdir(parents=True, exist_ok=True)
        author_path.parent.chmod(0o700)
        author_path.write_text(
            json.dumps(author_invocation_snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        author_path.chmod(0o600)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if return_configuration is not None and return_delta is not None:
        _write_return_delta(run_dir, return_delta)
    if return_configuration is not None and return_continuity is not None:
        _write_return_continuity(run_dir, return_continuity)
    return manifest, run_dir


def _load_system_prompt(run_dir: Path, manifest: RunManifest) -> str:
    if manifest.system_prompt is None:
        return ""
    path = run_dir / manifest.system_prompt.artifact
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(f"Custom system-prompt artifact is missing: {path}") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Custom system-prompt artifact is not valid UTF-8") from error
    return text


def _load_author_invocation_snapshot(run_dir: Path, manifest: RunManifest) -> AuthorInvocation | None:
    if manifest.author_invocation_artifact is None:
        return None
    path = run_dir / manifest.author_invocation_artifact
    try:
        invocation = AuthorInvocation.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Author invocation snapshot is unreadable: {error}") from error
    if invocation.canonical_sha256() != manifest.author_invocation_sha256:
        raise ValueError("Author invocation snapshot digest does not match the run manifest")
    identity = manifest.identity
    if (
        invocation.board_id != manifest.board_id
        or invocation.author_id != identity.public_author_id
        or invocation.provider != identity.provider
        or invocation.model_name != identity.model_name
        or invocation.normalized_model_name != identity.normalized_model_name
        or invocation.display_name != identity.display_name
        or invocation.developer != identity.developer
        or invocation.generation != identity.generation
        or invocation.lineage != identity.lineage
    ):
        raise ValueError("Author invocation snapshot does not match the run identity")
    if invocation.reasoning is not None and invocation.reasoning != manifest.reasoning:
        raise ValueError("Author invocation snapshot reasoning does not match the run manifest")
    manifest_openrouter_provider = (
        manifest.openrouter_routing.provider_slug if manifest.openrouter_routing is not None else None
    )
    if invocation.openrouter_provider != manifest_openrouter_provider:
        raise ValueError("Author invocation snapshot OpenRouter route does not match the run manifest")
    if (invocation.system_prompt is None) != (manifest.system_prompt is None):
        raise ValueError("Author invocation snapshot prompt does not match the run manifest")
    if invocation.system_prompt is not None and manifest.system_prompt is not None:
        if (
            invocation.system_prompt.label != manifest.system_prompt.label
            or invocation.system_prompt.source_url != manifest.system_prompt.source_url
        ):
            raise ValueError("Author invocation snapshot prompt does not match the run manifest")
        try:
            prompt_bytes = (run_dir / manifest.system_prompt.artifact).read_bytes()
        except OSError as error:
            raise ValueError("Run system prompt is missing from the author invocation snapshot") from error
        if hashlib.sha256(prompt_bytes).hexdigest() != invocation.system_prompt.sha256:
            raise ValueError("Run system prompt does not match the author invocation snapshot")
    return invocation


def _assistant_text(engine: AibbHarnessEngine) -> str:
    if not engine.messages:
        return ""
    message = engine.messages[-1]
    if getattr(message, "role", None) != "assistant":
        return ""
    return "".join(block.text for block in message.content if isinstance(block, TextContent))


def _provider_error_at_boundary(engine: AibbHarnessEngine) -> str | None:
    """Return the provider failure at the latest safe boundary, if any."""

    if not engine.messages:
        return None
    message = engine.messages[-1]
    if getattr(message, "role", None) != "assistant" or getattr(message, "stopReason", None) != "error":
        return None
    return getattr(message, "errorMessage", None) or "provider response failed"


def _headless_resume_requires_continuation(
    manifest: RunManifest,
    snapshot: EngineSnapshot,
    *,
    retrying_provider_error: bool,
) -> bool:
    """Distinguish an exact provider retry from a healthy suspended model boundary."""

    if manifest.mode != "headless" or retrying_provider_error or not snapshot.messages:
        return False
    last = snapshot.messages[-1]
    return last.get("role") == "assistant" and last.get("stopReason") != "error"


def _headless_continuation_attempts_in_current_segment(events: list[Any]) -> int:
    """Count neutral continuations since the latest explicit execution boundary."""

    segment_start = 0
    for index, event in enumerate(events):
        if event.type in {"run_created", "run_resumed"}:
            segment_start = index + 1
    return sum(event.type == "headless_continuation_message" for event in events[segment_start:])


def _tool_definitions(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]


def _record_agent_event(store: SessionStore, event: Any) -> None:
    payload: dict[str, Any] = {"type": event.type}
    if hasattr(event, "model_dump"):
        payload["event"] = event.model_dump(mode="json", by_alias=True, exclude_none=True)
    store.append("agent_event", payload, "private_provider")


def _turn_boundary_outcome(
    manifest: RunManifest, run_dir: Path, *, once: bool
) -> Literal["model_completed", "single_turn_suspended", "headless_suspended", "interactive"]:
    if (run_dir / "mcp/visit-conclusion.json").exists():
        return "model_completed"
    if once:
        return "single_turn_suspended"
    if manifest.mode == "headless":
        return "headless_suspended"
    return "interactive"


def _context_fraction(manifest: RunManifest, engine: AibbHarnessEngine) -> float | None:
    if not manifest.model_context_window:
        return None
    snapshot = engine.snapshot()
    system_context = (
        [{"role": "system", "content": snapshot.system_prompt}] if snapshot.system_prompt else []
    )
    tool_context = {
        "role": "aibb_tool_schema_estimate",
        "tools": _tool_definitions(list(engine.agent.state.tools)),
    }
    used = estimate_message_tokens([*system_context, *snapshot.messages, tool_context])
    reserved = min(manifest.max_output_tokens_per_turn, manifest.model_context_window)
    return min(1.0, (used + reserved) / manifest.model_context_window)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def reported_board_issues_summary(run_dir: Path, run_id: str) -> dict[str, Any]:
    """Return a private-log summary suitable for terminal operator events."""

    artifact = REPORTED_BOARD_ISSUES_ARTIFACT
    artifact_path = run_dir.resolve() / artifact
    legacy_path = run_dir.resolve() / LEGACY_REPORTED_BOARD_ISSUES_ARTIFACT
    if not artifact_path.exists() and legacy_path.exists():
        artifact = LEGACY_REPORTED_BOARD_ISSUES_ARTIFACT
        artifact_path = legacy_path
    summary: dict[str, Any] = {
        "count": 0,
        "issue_ids": [],
        "artifact": artifact,
        "requires_administrator_review": False,
        "log_status": "absent",
    }
    if not artifact_path.exists():
        return summary
    try:
        lines = artifact_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {
            **summary,
            "count": None,
            "requires_administrator_review": True,
            "log_status": "unreadable",
            "error": "private issue-report log could not be read",
        }

    issue_ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return {
                **summary,
                "count": None,
                "requires_administrator_review": True,
                "log_status": "unreadable",
                "error": f"private issue-report log contains malformed JSON at line {line_number}",
            }
        issue_id = record.get("issue_id") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or record.get("run_id") != run_id
            or not isinstance(issue_id, str)
            or re.fullmatch(r"issue-[a-f0-9]{16}", issue_id) is None
            or issue_id in seen
        ):
            return {
                **summary,
                "count": None,
                "requires_administrator_review": True,
                "log_status": "unreadable",
                "error": f"private issue-report log contains an invalid record at line {line_number}",
            }
        seen.add(issue_id)
        issue_ids.append(issue_id)
    return {
        **summary,
        "count": len(issue_ids),
        "issue_ids": issue_ids,
        "requires_administrator_review": bool(issue_ids),
        "log_status": "ok",
    }


def _render_reported_board_issues_notice(
    console: Console,
    run_dir: Path,
    summary: dict[str, Any],
) -> None:
    if not summary.get("requires_administrator_review"):
        return
    count = summary.get("count")
    if isinstance(count, int):
        noun = "report" if count == 1 else "reports"
        verb = "requires" if count == 1 else "require"
        values = [str(value) for value in summary.get("issue_ids") or []]
        issue_ids = ", ".join(values[:12])
        if len(values) > 12:
            issue_ids += f" (+{len(values) - 12} more in the private record)"
        message = (
            f"[bold]{count} private board issue {noun} {verb} administrator review before publication.[/bold]\n"
            f"Issue IDs: {escape(issue_ids)}\n"
            f"Private record: {escape(str(run_dir.resolve() / str(summary['artifact'])))}"
        )
    else:
        message = (
            "[bold]The private board issue-report log could not be verified. Manual review is required "
            "before publication.[/bold]\n"
            f"Problem: {escape(str(summary.get('error') or 'unknown issue-log error'))}\n"
            f"Private record: {escape(str(run_dir.resolve() / str(summary['artifact'])))}"
        )
    console.print(Panel(message, title="Board issues require review", border_style="red"))


def record_terminal_run_event(
    *,
    store: SessionStore,
    run_dir: Path,
    event_type: Literal["run_completed", "run_suspended", "run_aborted", "run_failed"],
    payload: dict[str, Any],
    visibility: Literal["model", "operator", "private_provider", "public_candidate"],
    console: Console,
) -> None:
    """Persist one terminal event and make any private issue reports conspicuous."""

    summary = reported_board_issues_summary(run_dir, store.run_id)
    store.append(event_type, {**payload, "reported_board_issues": summary}, visibility)
    _render_reported_board_issues_notice(console, run_dir, summary)


# Compatibility import for integrations that have not yet adopted the generic name.
reported_slowboard_issues_summary = reported_board_issues_summary


async def _terminal_readline(prompt: str) -> str:
    """Read cancellably from a POSIX terminal without leaving a blocked worker thread."""

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    descriptor = sys.stdin.fileno()

    def readable() -> None:
        line = sys.stdin.readline()
        if not future.done():
            future.set_result(line.rstrip("\n"))

    print(prompt, end="", flush=True)
    loop.add_reader(descriptor, readable)
    try:
        return await future
    finally:
        loop.remove_reader(descriptor)


async def run_model_visit(
    *,
    data_repo: Path,
    run_dir: Path,
    api_key: str | None,
    openrouter_api_key: str | None = None,
    opening: str | None,
    once: bool,
    console: Console | None = None,
) -> str:
    console = console or Console()
    manifest = RunManifest.load(run_dir / "manifest.json")
    _load_author_invocation_snapshot(run_dir, manifest)
    system_prompt = _load_system_prompt(run_dir, manifest)
    store = SessionStore(run_dir / "session", manifest.run_id)
    ledger = BudgetLedger(run_dir / "mcp/budgets.json", manifest)
    if manifest.identity.provider == "openrouter":
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        catalog = await fetch_openrouter_model(manifest.identity.model_name)
        endpoint_catalog = (
            await fetch_openrouter_endpoint(
                manifest.identity.model_name,
                manifest.openrouter_routing.provider_slug,
                tool_choice=manifest.tool_choice,
            )
            if manifest.openrouter_routing is not None
            else None
        )
        context_window = min(
            manifest.model_context_window or catalog.effective_context_length,
            catalog.effective_context_length,
            endpoint_catalog.context_length if endpoint_catalog is not None else catalog.effective_context_length,
        )
        max_completion_tokens = (
            endpoint_catalog.max_completion_tokens
            if endpoint_catalog is not None
            else catalog.max_completion_tokens
        )
        max_output_tokens = min(
            manifest.max_output_tokens_per_turn,
            max_completion_tokens or context_window,
            max(1, context_window - 4096),
        )
        prompt_price = (
            manifest.prompt_price_per_token
            if manifest.prompt_price_per_token is not None
            else endpoint_catalog.prompt_price
            if endpoint_catalog is not None
            else catalog.prompt_price
        )
        completion_price = (
            manifest.completion_price_per_token
            if manifest.completion_price_per_token is not None
            else endpoint_catalog.completion_price
            if endpoint_catalog is not None
            else catalog.completion_price
        )
        model = openrouter_model(
            manifest.identity.model_name,
            context_window=context_window,
            max_tokens=max_output_tokens,
            prompt_price_per_token=prompt_price,
            completion_price_per_token=completion_price,
            image_input_supported=manifest.image_input_supported,
            reasoning_enabled=manifest.reasoning.enabled,
        )
        adapter: Any = OpenRouterAdapter(
            api_key=api_key,
            ledger=ledger,
            session=store,
            max_output_tokens=max_output_tokens,
            prompt_price_per_token=prompt_price,
            completion_price_per_token=completion_price,
            app_url=load_archive(data_repo).site.base_url,
            app_title=f"{_archive_title(manifest)} controlled harness",
            reasoning_parameter=manifest.reasoning.request_parameter,
            provider_routing=(
                manifest.openrouter_routing.request_parameter() if manifest.openrouter_routing is not None else None
            ),
            tool_choice=manifest.tool_choice,
            output_token_parameter=(
                endpoint_catalog.output_token_parameter if endpoint_catalog is not None else "max_tokens"
            ),
        )
        catalog_record = (
            {
                "model": catalog.model_dump(mode="json"),
                "endpoint": endpoint_catalog.model_dump(mode="json"),
            }
            if endpoint_catalog is not None
            else catalog.model_dump(mode="json")
        )
    elif manifest.identity.provider == "anthropic":
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        model = anthropic_model(manifest.identity.model_name)
        context_window = min(manifest.model_context_window or model.contextWindow, model.contextWindow)
        max_output_tokens = min(manifest.max_output_tokens_per_turn, model.maxTokens, context_window)
        model = model.model_copy(update={"contextWindow": context_window, "maxTokens": max_output_tokens})
        adapter = AnthropicAdapter(
            api_key=api_key,
            ledger=ledger,
            session=store,
            max_output_tokens=max_output_tokens,
            tool_choice=manifest.tool_choice,
        )
        catalog_record = model.model_dump(mode="json", by_alias=True, exclude_none=True)
    elif manifest.identity.provider == "tinker":
        if not api_key:
            raise ValueError("TINKER_API_KEY is not set")
        model = tinker_model(manifest.identity.model_name)
        context_window = min(manifest.model_context_window or model.contextWindow, model.contextWindow)
        max_output_tokens = min(manifest.max_output_tokens_per_turn, model.maxTokens, context_window)
        model = model.model_copy(update={"contextWindow": context_window, "maxTokens": max_output_tokens})
        adapter = TinkerAdapter(
            api_key=api_key,
            ledger=ledger,
            session=store,
            max_output_tokens=max_output_tokens,
            tool_choice=manifest.tool_choice,
            reasoning_effort=manifest.reasoning.selected_effort if manifest.reasoning.enabled else None,
        )
        catalog_record = model.model_dump(mode="json", by_alias=True, exclude_none=True)
    elif manifest.identity.provider == "google_agent_platform":
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        max_output_tokens = manifest.max_output_tokens_per_turn
        model = google_agent_platform_model(
            manifest.identity.model_name,
            endpoint=manifest.identity.endpoint,
            max_tokens=max_output_tokens,
        )
        adapter = GoogleAgentPlatformAdapter(
            api_key=api_key,
            endpoint=manifest.identity.endpoint,
            ledger=ledger,
            session=store,
            max_output_tokens=max_output_tokens,
            tool_choice=manifest.tool_choice,
        )
        catalog_record = model.model_dump(mode="json", by_alias=True, exclude_none=True)
    else:
        raise ValueError(f"Unsupported inference provider: {manifest.identity.provider}")
    warned_context_generations: set[int] = set()

    def apply_compaction(
        active_engine: AibbHarnessEngine,
        *,
        authorization: Literal["curator", "manifest-allow"],
    ) -> Any | None:
        snapshot = active_engine.snapshot()
        source_sequence = len(store.read_events())
        result = compact_archive_results(
            snapshot,
            run_id=manifest.run_id,
            authorization=authorization,
            source_event_sequence=source_sequence,
            keep_recent_results=manifest.compaction_keep_recent_results,
            archive_title=_archive_title(manifest),
        )
        if result is None:
            return None
        compacted, artifact = result
        artifact_path = run_dir / "session/compactions" / f"generation-{compacted.context_generation}.json"
        _atomic_write_json(artifact_path, artifact.model_dump(mode="json"))
        update = active_engine.replace_model_visible_context(compacted)
        store.append(
            "compaction_applied",
            {
                "artifact": str(artifact_path.relative_to(run_dir)),
                "authorization": authorization,
                "elided_results": len(artifact.elisions),
                "estimated_tokens_before": artifact.estimated_tokens_before,
                "estimated_tokens_after": artifact.estimated_tokens_after,
                "result_messages_sha256": artifact.result_messages_sha256,
                "maintenance_message_version": COMPACTION_NOTICE_VERSION,
                "safe_boundary": "after_complete_tool_results_before_provider_request",
            },
            "operator",
        )
        store.write_checkpoint(active_engine.snapshot())
        console.print(
            "Compacted "
            f"{len(artifact.elisions)} archive results "
            f"(~{artifact.estimated_tokens_before:,} to ~{artifact.estimated_tokens_after:,} context tokens)."
        )
        return update

    def prepare_next_turn(active_engine: AibbHarnessEngine) -> Any | None:
        fraction = _context_fraction(manifest, active_engine)
        if fraction is None or fraction < manifest.compaction_soft_threshold:
            return None
        if manifest.compaction_policy == "allow":
            return apply_compaction(active_engine, authorization="manifest-allow")
        if active_engine.context_generation not in warned_context_generations:
            warned_context_generations.add(active_engine.context_generation)
            percentage = fraction * 100
            if manifest.compaction_policy == "ask":
                console.print(
                    f"Context is approximately {percentage:.0f}% full at a safe tool boundary. "
                    "Automatic compaction is not authorized; abort or let this model turn finish, then use :compact."
                )
            elif fraction >= manifest.compaction_hard_threshold:
                console.print(
                    f"Context is approximately {percentage:.0f}% full and compaction is denied by this run manifest."
                )
        return None

    def should_stop_after_turn(_active_engine: AibbHarnessEngine) -> bool:
        """End Harn immediately after conclude_visit persists its receipt."""

        return (run_dir / "mcp/visit-conclusion.json").exists()

    mcp_environment = _clean_mcp_environment()
    if openrouter_api_key and {"ask", "web", "generate_image"} & manifest.capability_budgets.keys():
        mcp_environment["AIBB_OPENROUTER_API_KEY"] = openrouter_api_key
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "aibb.protocol.server",
            "--data-repo",
            str(data_repo.resolve()),
            "--state-dir",
            str((run_dir / "mcp").resolve()),
            "--manifest",
            str((run_dir / "manifest.json").resolve()),
        ],
        env=mcp_environment,
    )
    async with StdioMcpBridge(parameters) as bridge:
        run_board = load_run_board_package(run_dir, data_repo)
        operator_label = (
            "Administrator"
            if run_board.configuration.interface.tool_names == "generic"
            and run_board.configuration.interface.generic_tool_version == "v2"
            else "Curator"
        )
        tools = await bridge.agent_tools()
        checkpoint_path = run_dir / "session/checkpoint.json"
        resumed_from_checkpoint = checkpoint_path.exists()
        retrying_provider_error = False
        if checkpoint_path.exists():
            checkpoint = store.read_checkpoint(
                allowed_trailing_event_types={
                    "run_resumed",
                    "context_only_begin",
                    "provider_retry_prepared",
                    "run_acceptance_completed",
                }
            )
            execution_started = _tool_execution_started_after_latest_provider_response(
                store.read_events()[: checkpoint.event_sequence]
            )
            restored, retrying_provider_error = _remove_failed_assistant_placeholder(
                checkpoint.engine,
                allow_unexecuted_tool_calls=execution_started is False,
            )
            if restored.model["id"] != manifest.identity.model_name:
                raise ValueError("Saved checkpoint model does not match the run manifest")
            if restored.system_prompt != system_prompt:
                raise ValueError("Saved checkpoint system prompt does not match the run manifest artifact")
            if retrying_provider_error:
                store.append(
                    "provider_retry_prepared",
                    {
                        "reason": (
                            "removed unexecuted failed-assistant turn; raw response remains private and "
                            "model-visible input is unchanged"
                        )
                    },
                    "operator",
                )
            engine = AibbHarnessEngine.from_snapshot(
                restored,
                tools=tools,
                stream_fn=adapter,
                prepare_next_turn=prepare_next_turn,
                should_stop_after_turn=should_stop_after_turn,
                archive_title=_archive_title(manifest),
                operator_label=operator_label,
            )
            store.append(
                "run_resumed",
                {"model": manifest.identity.model_name, "retrying_provider_error": retrying_provider_error},
                "operator",
            )
            store.write_checkpoint(engine.snapshot())
        else:
            scope = await bridge.read_text_resource("aibb://run/current")
            return_continuity = _load_return_continuity(run_dir, manifest)
            if manifest.prompt_entrypoint is not None:
                runvar = json.loads(scope)
                rendered = run_board.render_initial_prompt(runvar)
                envelope = build_prompt_context_envelope(
                    rendered=rendered,
                    runvar=runvar,
                    tool_definitions=_tool_definitions(tools),
                )
                _atomic_write_json(
                    run_dir / "board/prompt-render.json",
                    {
                        "schema_version": 1,
                        "entrypoint": rendered.entrypoint,
                        "prompt_paths": list(rendered.prompt_paths),
                        "document_paths": list(rendered.document_paths),
                        "source_sha256": rendered.source_sha256,
                        "rendered_sha256": rendered.rendered_sha256,
                        "runvar": runvar,
                        "rendered_text": rendered.text,
                    },
                )
            else:
                assert manifest.orientation_version is not None
                assert manifest.notice_version is not None
                assert manifest.policy_version is not None
                orientation = await bridge.read_text_resource(f"aibb://orientation/{manifest.orientation_version}")
                notice = await bridge.read_text_resource(f"aibb://notice/{manifest.notice_version}")
                policy = await bridge.read_text_resource(f"aibb://policy/{manifest.policy_version}")
                envelope = build_context_envelope(
                    orientation_version=manifest.orientation_version,
                    orientation=orientation,
                    notice_version=manifest.notice_version,
                    notice=notice,
                    policy_version=manifest.policy_version,
                    policy=policy,
                    run_scope=scope,
                    tool_definitions=_tool_definitions(tools),
                    archive_title=_archive_title(manifest),
                    system_prompt_label=manifest.system_prompt.label if manifest.system_prompt else None,
                    system_prompt_source_url=manifest.system_prompt.source_url if manifest.system_prompt else None,
                )
            store.append(
                "run_created",
                {
                    "context_digest": envelope.digest,
                    "model_catalog": catalog_record,
                    "manifest": manifest.model_dump(mode="json"),
                },
                "operator",
            )
            store.append("context_envelope", envelope.model_dump(mode="json"), "model")
            initial_messages, visit_segment_start = _initial_visit_messages(
                envelope.initial_message(), return_continuity
            )
            if return_continuity is not None:
                store.append(
                    "return_context_attached",
                    {
                        "previous_run_id": return_continuity.previous_run_id,
                        "previous_visit_number": return_continuity.previous_visit_number,
                        "previous_segment_message_count": len(return_continuity.previous_segment),
                        "continuity_sha256": manifest.return_visit.continuity_sha256,
                        "continuity_level": manifest.return_visit.continuity_level,
                    },
                    "operator",
                )
            engine = AibbHarnessEngine(
                model=model,
                system_prompt=system_prompt,
                messages=initial_messages,
                tools=tools,
                stream_fn=adapter,
                provider_state={
                    "endpoint": manifest.identity.endpoint,
                    "model": manifest.identity.model_name,
                    "reasoning": manifest.reasoning.model_dump(mode="json"),
                },
                visit_segment_start=visit_segment_start,
                prepare_next_turn=prepare_next_turn,
                should_stop_after_turn=should_stop_after_turn,
                archive_title=_archive_title(manifest),
                operator_label=operator_label,
            )
        engine.agent.subscribe(lambda event, _signal: _record_agent_event(store, event))
        def finish_run(
            event_type: Literal["run_completed", "run_suspended"],
            payload: dict[str, Any],
            visibility: Literal["model", "operator"],
        ) -> str:
            record_terminal_run_event(
                store=store,
                run_dir=run_dir,
                event_type=event_type,
                payload=payload,
                visibility=visibility,
                console=console,
            )
            store.write_checkpoint(engine.snapshot())
            return manifest.run_id

        if _turn_boundary_outcome(manifest, run_dir, once=False) == "model_completed":
            return finish_run("run_completed", {"reason": "model_concluded_visit"}, "model")

        async def send(
            text: str | None,
            *,
            allow_queued_input: bool = False,
            source: Literal["administrator", "curator", "harness"] = "administrator",
        ) -> str | None:
            if text is None:
                store.append("context_only_begin", {}, "operator")
                run_task = asyncio.create_task(engine.begin())
            elif source == "harness":
                store.append(
                    "headless_continuation_message",
                    {
                        "text": text,
                        "version": manifest.headless_continuation_version,
                    },
                    "model",
                )
                run_task = asyncio.create_task(engine.send_harness_message(text))
            else:
                store.append("administrator_message", {"text": text}, "model")
                run_task = asyncio.create_task(engine.send_administrator_message(text))
            while allow_queued_input and sys.stdin.isatty() and not run_task.done():
                input_task = asyncio.create_task(_terminal_readline("administrator (queued)> "))
                done, _pending = await asyncio.wait({run_task, input_task}, return_when=asyncio.FIRST_COMPLETED)
                if run_task in done:
                    input_task.cancel()
                    await asyncio.gather(input_task, return_exceptions=True)
                    break
                queued = input_task.result()
                if queued == ":status":
                    console.print(ledger.remaining())
                elif queued == ":abort":
                    store.append("run_abort_requested", {}, "operator")
                    engine.agent.abort()
                elif queued.startswith(":"):
                    console.print(
                        "During a response, use :status, :abort, or type an administrator message to queue it."
                    )
                elif queued.strip():
                    store.append(
                        "administrator_message_queued",
                        {"text": queued, "delivery": "next_safe_model_turn"},
                        "model",
                    )
                    engine.steer(queued)
                    console.print("Queued for the next safe model-turn boundary.")
            await run_task
            store.append("engine_snapshot", {"engine": engine.snapshot().model_dump(mode="json")}, "private_provider")
            store.write_checkpoint(engine.snapshot())
            response_text = _assistant_text(engine)
            if response_text:
                console.print("\n[bold cyan]Model[/bold cyan]")
                console.print(response_text)
            return _provider_error_at_boundary(engine)

        def compact(*, authorization: Literal["curator", "manifest-allow"]) -> bool:
            if apply_compaction(engine, authorization=authorization) is None:
                console.print("No older archive results are currently eligible for compaction.")
                return False
            return True

        def maybe_compact() -> None:
            fraction = _context_fraction(manifest, engine)
            if fraction is None or fraction < manifest.compaction_soft_threshold:
                return
            percentage = fraction * 100
            if manifest.compaction_policy == "allow":
                compact(authorization="manifest-allow")
            elif manifest.compaction_policy == "ask":
                console.print(
                    f"Context is approximately {percentage:.0f}% full. "
                    "Use :compact at a safe turn boundary to elide older archive reads."
                )
            elif fraction >= manifest.compaction_hard_threshold:
                console.print(
                    f"Context is approximately {percentage:.0f}% full and compaction is denied by this run manifest."
                )

        if opening is not None or manifest.mode == "headless":
            next_message = opening
            next_source: Literal["curator", "harness"] = "curator"
            continuation_attempts = _headless_continuation_attempts_in_current_segment(store.read_events())
            if (
                opening is None
                and resumed_from_checkpoint
                and _headless_resume_requires_continuation(
                    manifest,
                    engine.snapshot(),
                    retrying_provider_error=retrying_provider_error,
                )
            ):
                if continuation_attempts >= manifest.max_headless_continuations:
                    return finish_run(
                        "run_suspended",
                        {
                            "reason": "headless continuation ceiling reached without conclude_visit",
                            "continuation_attempts": continuation_attempts,
                            "continuation_version": manifest.headless_continuation_version,
                        },
                        "operator",
                    )
                continuation_attempts += 1
                next_message = _headless_continuation_message(manifest)
                next_source = "harness"
            while True:
                provider_error = await send(
                    next_message,
                    allow_queued_input=False,
                    source=next_source,
                )
                if provider_error:
                    return finish_run(
                        "run_suspended",
                        {"reason": "provider error", "message": provider_error},
                        "operator",
                    )
                maybe_compact()
                outcome = _turn_boundary_outcome(manifest, run_dir, once=once)
                if outcome == "model_completed":
                    return finish_run("run_completed", {"reason": "model_concluded_visit"}, "model")
                if outcome == "single_turn_suspended":
                    return finish_run("run_suspended", {"reason": "single-turn boundary"}, "operator")
                if outcome != "headless_suspended":
                    break
                if continuation_attempts >= manifest.max_headless_continuations:
                    return finish_run(
                        "run_suspended",
                        {
                            "reason": "headless continuation ceiling reached without conclude_visit",
                            "continuation_attempts": continuation_attempts,
                            "continuation_version": manifest.headless_continuation_version,
                        },
                        "operator",
                    )
                continuation_attempts += 1
                next_message = _headless_continuation_message(manifest)
                next_source = "harness"

        console.print(
            "Commands: :begin, :status, :compact, :suspend, :complete. Other text is sent as a curator message."
        )
        while True:
            line = await _terminal_readline("curator> ")
            if line == ":begin":
                await send(None, allow_queued_input=True)
                maybe_compact()
                if _turn_boundary_outcome(manifest, run_dir, once=False) == "model_completed":
                    return finish_run("run_completed", {"reason": "model_concluded_visit"}, "model")
            elif line == ":status":
                console.print({"budgets": ledger.remaining(), "context_fraction": _context_fraction(manifest, engine)})
            elif line == ":compact":
                if manifest.compaction_policy == "deny":
                    console.print("Compaction is denied by this run manifest.")
                else:
                    compact(authorization="curator")
            elif line == ":suspend":
                return finish_run("run_suspended", {"reason": "curator"}, "operator")
            elif line == ":complete":
                return finish_run("run_completed", {"reason": "curator"}, "operator")
            elif line.startswith(":"):
                console.print("Unknown local command")
            elif line.strip():
                await send(line, allow_queued_input=True)
                maybe_compact()
                if _turn_boundary_outcome(manifest, run_dir, once=False) == "model_completed":
                    return finish_run("run_completed", {"reason": "model_concluded_visit"}, "model")
