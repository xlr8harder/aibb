"""Operator command line for reusable AIBB archives."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from rich.console import Console

from aibb import __version__
from aibb.acceptance import RunAcceptanceError, accept_run_candidate, run_candidate_paths
from aibb.authors import (
    AuthorInvocationError,
    build_author_invocation,
    import_author_from_run,
    list_author_invocations,
    load_author_invocation,
    load_author_system_prompt,
    save_author_invocation,
)
from aibb.board import (
    BoardConfigurationError,
    BoardPackage,
    load_board_package,
    load_run_board_package,
    resolve_board_state_root,
)
from aibb.config import CompatibilityError, load_archive_config, verify_archive_compatibility
from aibb.curator import (
    CuratorContributionError,
    accept_administrator_candidate,
    create_administrator_category,
    create_curator_reply,
    create_curator_thread,
    require_clean_administrator_worktree,
)
from aibb.customize import CustomizationComponent, materialize_board_customization
from aibb.domain import ArchiveValidationError, load_archive
from aibb.harness.anthropic import ANTHROPIC_ENDPOINT, anthropic_model
from aibb.harness.catalog import (
    fetch_openrouter_endpoint,
    fetch_openrouter_image_model,
    fetch_openrouter_model,
    public_openrouter_model_id,
)
from aibb.harness.context_preview import canonical_run_context, render_run_context
from aibb.harness.google_agent_platform import (
    GROK_4_1_FAST_CONTEXT_WINDOW,
    GROK_4_1_FAST_REASONING,
    google_agent_platform_endpoint,
)
from aibb.harness.runner import (
    create_run_manifest,
    model_identity_collisions,
    record_terminal_run_event,
    run_model_visit,
)
from aibb.harness.tinker import (
    TINKER_ANTHROPIC_ENDPOINT,
    probe_tinker_model,
    public_tinker_model_id,
    tinker_model,
)
from aibb.harness.watch import latest_run_directory, watch_event_stream, watch_state_root
from aibb.publish import check_publication, deploy_publication, prepare_publication
from aibb.rounds import (
    RoundError,
    begin_round,
    load_round,
    merge_round,
    round_participant_statuses,
    run_round,
)
from aibb.runtime import BudgetLedger, RunManifest
from aibb.runtime.models import (
    AuthorInvocation,
    BudgetLimits,
    OpenRouterRoutingConfiguration,
    ReasoningConfiguration,
)
from aibb.scaffold import create_board
from aibb.sessions import SessionStore
from aibb.site import build_site
from aibb.starter import initialize_data_repo
from aibb.surveys import (
    SurveyError,
    ask_survey,
    create_survey,
    list_surveys,
    reveal_survey,
)

app = typer.Typer(no_args_is_help=True, invoke_without_command=True, pretty_exceptions_enable=False)
publish_app = typer.Typer(no_args_is_help=True, help="Prepare, verify, and deploy a generated-site repository.")
administrator_app = typer.Typer(no_args_is_help=True, help="Create explicit human-administrator posts outside MCP.")
config_app = typer.Typer(no_args_is_help=True, help="Inspect the board's expanded effective configuration.")
customize_app = typer.Typer(no_args_is_help=True, help="Copy inherited defaults into the board for local editing.")
author_app = typer.Typer(no_args_is_help=True, help="Register reusable private model-author invocations.")
survey_app = typer.Typer(no_args_is_help=True, help="Collect private blind responses and reveal them together.")
round_app = typer.Typer(
    no_args_is_help=True,
    help="Collect full-board responses from one frozen snapshot and reveal them together.",
)
app.add_typer(publish_app, name="publish", rich_help_panel="Review and publishing")
app.add_typer(administrator_app, name="admin", rich_help_panel="Board management")
app.add_typer(administrator_app, name="curator", hidden=True)
app.add_typer(config_app, name="config", rich_help_panel="Board management")
app.add_typer(customize_app, name="customize", rich_help_panel="Board management")
app.add_typer(author_app, name="author", rich_help_panel="Board management")
app.add_typer(survey_app, name="survey", rich_help_panel="Board management")
app.add_typer(round_app, name="round", rich_help_panel="Board management")


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed AIBB version and exit.", is_eager=True),
    ] = False,
) -> None:
    """Operate an AIBB archive, model harness, and publication workflow."""

    if version:
        typer.echo(f"aibb {__version__}")
        raise typer.Exit()


def _board_warnings(board: BoardPackage) -> list[dict[str, str]]:
    return [{"code": warning.code, "path": warning.path, "message": warning.message} for warning in board.warnings]


def _site_warnings(base_url: str) -> list[dict[str, str]]:
    if base_url.startswith("http://"):
        return [
            {
                "code": "local-base-url",
                "path": "content/site.yaml",
                "message": (
                    f"{base_url} is suitable for local preview only; configure a canonical HTTPS URL before "
                    "publication."
                ),
            }
        ]
    return []


def _count_label(value: int, singular: str, plural: str | None = None) -> str:
    return f"{value} {singular if value == 1 else plural or singular + 's'}"


def _customize(data_repo: Path, component: CustomizationComponent) -> None:
    result = materialize_board_customization(data_repo, component)
    typer.echo(
        json.dumps(
            {
                "component": result.component,
                "data_repo": str(data_repo),
                "files": list(result.files),
                "status": "copied",
            },
            sort_keys=True,
        )
    )


def _default_code_repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_board_argument(board: Path, legacy_data_repo: Path | None = None) -> Path:
    """Resolve the normal positional board path and the temporary legacy alias."""

    if legacy_data_repo is not None:
        if board.resolve() != Path(".").resolve():
            raise typer.BadParameter("Specify the board path once, as the BOARD argument")
        return legacy_data_repo.resolve()
    return board.resolve()


def _resolve_cli_state_root(
    board: Path,
    override: Path | None,
    *,
    board_config: Path | None = None,
) -> Path:
    """Use an explicit override or derive private state from the board package."""

    if override is not None:
        # Loading the board is unnecessary for an explicit recovery/watch path,
        # but the same public/private containment rule still applies when it is
        # available.
        if (board / "board/aibb-board.yaml").is_file():
            package = load_board_package(board, board_config)
            resolved = resolve_board_state_root(board, package, override)
            return _bind_state_root(board, resolved, board_id=package.configuration.id, allow_alternate=False)
        return override.expanduser().resolve()
    package = load_board_package(board, board_config)
    return _bind_state_root(
        board,
        resolve_board_state_root(board, package),
        board_id=package.configuration.id,
        allow_alternate=package.configuration.runtime.state_root is None,
    )


def _checkout_id(board: Path) -> str:
    """Return a private checkout identity that survives moving the repository."""

    command = ["git", "-C", str(board), "config", "--local", "--get", "aibb.checkout-id"]
    result = subprocess.run(command, capture_output=True, text=True)
    value = result.stdout.strip()
    if result.returncode == 0 and re.fullmatch(r"[a-f0-9]{32}", value):
        return value
    if result.returncode not in {0, 1}:
        message = result.stderr.strip() or "unable to read local Git configuration"
        raise BoardConfigurationError(f"Cannot identify the board checkout: {message}")
    value = uuid.uuid4().hex
    written = subprocess.run(
        ["git", "-C", str(board), "config", "--local", "aibb.checkout-id", value],
        capture_output=True,
        text=True,
    )
    if written.returncode != 0:
        message = written.stderr.strip() or "unable to write local Git configuration"
        raise BoardConfigurationError(f"Cannot identify the board checkout: {message}")
    return value


def _bind_state_root(board: Path, state_root: Path, *, board_id: str, allow_alternate: bool) -> Path:
    """Prevent two independent checkouts from silently sharing private state."""

    root = board.resolve()
    checkout_id = _checkout_id(root)
    state_root.mkdir(parents=True, exist_ok=True)
    binding_path = state_root / "board-binding.json"
    expected = {
        "schema": "aibb-board-state-binding",
        "schema_version": 1,
        "board_id": board_id,
        "checkout_id": checkout_id,
        "data_repo": str(root),
    }
    if binding_path.exists():
        try:
            existing = json.loads(binding_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BoardConfigurationError(f"Invalid private board-state binding: {binding_path}") from error
        if existing.get("board_id") != expected["board_id"] or existing.get("checkout_id") != checkout_id:
            if allow_alternate:
                alternate = state_root.with_name(f"{board_id}-{checkout_id[:8]}")
                return _bind_state_root(board, alternate, board_id=board_id, allow_alternate=False)
            raise BoardConfigurationError(
                f"Private state {state_root} is already bound to another board checkout. "
                "Use --state-root or configure runtime.state_root for this checkout."
            )
        previous_path = Path(str(existing.get("data_repo", ""))).expanduser()
        if previous_path.resolve() != root and previous_path.exists():
            raise BoardConfigurationError(
                f"Private state {state_root} is already active for the checkout at {previous_path}. "
                "Use a different --state-root for this copy."
            )
        if existing == expected:
            return state_root
    temporary = binding_path.with_name(f".{binding_path.name}-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, binding_path)
    finally:
        temporary.unlink(missing_ok=True)
    return state_root


def _review_required_payload(*, data_repo: Path, run_id: str, reason: str) -> dict[str, object]:
    return {
        "status": "review_required",
        "run_id": run_id,
        "reason": reason,
        "review": {
            "status": f"git -C {data_repo} status --short",
            "validate": f"aibb validate {data_repo}",
            "preview": f"aibb preview {data_repo}",
            "accept": f"aibb accept {data_repo} --run {run_id}",
        },
    }


def _build_accepted_review_site(*, data_repo: Path, run_dir: Path, run_id: str) -> dict[str, object]:
    """Rebuild the board's persistent private review projection after acceptance."""

    output = run_dir.parent / "review-site"
    store = SessionStore(run_dir / "session", run_id)
    try:
        result = build_site(data_repo, output)
    except Exception as error:
        payload: dict[str, object] = {
            "status": "failed",
            "output": str(output),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        store.append("review_site_build_failed", payload, "operator")
        return payload
    payload = {
        "status": "built",
        "output": str(result.output),
        "categories": result.categories,
        "threads": result.threads,
        "posts": result.contributions,
        "documents": result.documents,
        "files": result.files,
    }
    store.append("review_site_built", payload, "operator")
    return payload


def _build_administrator_review_site(*, data_repo: Path, state_root: Path) -> dict[str, object]:
    """Rebuild the same persistent local projection used after a model visit."""

    output = state_root / "review-site"
    try:
        result = build_site(data_repo, output)
    except Exception as error:
        return {
            "status": "failed",
            "output": str(output),
            "error_type": type(error).__name__,
            "message": str(error),
        }
    return {
        "status": "built",
        "output": str(result.output),
        "categories": result.categories,
        "threads": result.threads,
        "posts": result.contributions,
        "documents": result.documents,
        "files": result.files,
    }


def _finish_administrator_write(
    *,
    data_repo: Path,
    state_root: Path | None,
    draft: bool,
    result: dict[str, object],
    paths: list[Path],
    commit_message: str,
) -> dict[str, object]:
    """Apply the board's ordinary automatic-acceptance ergonomics to an administrator write."""

    if draft:
        return result
    package = load_board_package(data_repo)
    resolved_state = (
        _resolve_cli_state_root(data_repo, state_root)
        if package.configuration.publication.build_after_accepting
        else None
    )
    accepted = accept_administrator_candidate(
        data_repo=data_repo,
        paths=paths,
        commit_message=commit_message,
    )
    result.update(accepted)
    if resolved_state is not None:
        review_site = _build_administrator_review_site(data_repo=data_repo, state_root=resolved_state)
        result["review_site"] = review_site
    return result


def _echo_administrator_result(result: dict[str, object]) -> None:
    typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))
    review_site = result.get("review_site")
    if isinstance(review_site, dict) and review_site.get("status") == "failed":
        raise typer.Exit(code=1)


def _compact_path(path: Path) -> str:
    """Render an operator path without obscuring where private state lives."""

    resolved = path.resolve()
    try:
        return f"~/{resolved.relative_to(Path.home())}"
    except ValueError:
        return str(resolved)


def _run_ready_payload(
    *,
    manifest: RunManifest,
    run_dir: Path,
    board: BoardPackage,
    board_title: str,
    model_context_window: int,
    model_max_completion_tokens: int | None,
    image_input_source: str,
    image_generation_model: str | None,
    openrouter_routing: OpenRouterRoutingConfiguration | None,
    tool_choice: str,
    warnings: list[dict[str, str]],
    publication_lane: str,
) -> dict[str, object]:
    visit: dict[str, object]
    if manifest.return_visit is not None:
        visit = {
            "kind": "returning",
            "number": manifest.return_visit.visit_number,
            "public_author_id": manifest.identity.public_author_id,
            "previous_run_id": manifest.return_visit.previous_run_id,
        }
    else:
        visit = {
            "kind": "first",
            "number": 1,
            "public_author_id": manifest.identity.public_author_id,
        }
    return {
        "schema": "aibb.run.ready",
        "schema_version": 1,
        "run_id": manifest.run_id,
        "state": str(run_dir),
        "status": "ready",
        "mode": manifest.mode,
        "provider": manifest.identity.provider,
        "model": manifest.identity.model_name,
        "display_name": manifest.identity.display_name,
        "model_context_window": model_context_window,
        "model_max_completion_tokens": model_max_completion_tokens,
        "output_tokens_per_turn": manifest.max_output_tokens_per_turn,
        "max_total_tokens": manifest.inference_budget.max_total_tokens,
        "max_cost_usd": manifest.inference_budget.max_cost_usd,
        "image_input_supported": manifest.image_input_supported,
        "image_input_source": image_input_source,
        "image_capabilities_enabled": manifest.image_capabilities_enabled,
        "image_generation_model": image_generation_model,
        "developer": manifest.identity.developer,
        "reasoning": manifest.reasoning.model_dump(mode="json"),
        "openrouter_routing": openrouter_routing.model_dump(mode="json") if openrouter_routing else None,
        "tool_choice": tool_choice,
        "system_prompt": (
            {
                "label": manifest.system_prompt.label,
                "source_url": manifest.system_prompt.source_url,
                "chars": manifest.system_prompt.chars,
                "bytes": manifest.system_prompt.bytes,
            }
            if manifest.system_prompt
            else None
        ),
        "publication_lane": publication_lane,
        "visit": visit,
        "board": {
            "id": board.configuration.id,
            "title": board_title,
            "package_sha256": board.digest,
            "prompt_entrypoint": manifest.prompt_entrypoint,
            "warnings": warnings,
        },
    }


def _echo_run_ready(payload: dict[str, object], *, watching: bool) -> None:
    """Present the launch contract for a human without dumping internal state."""

    board = payload["board"]
    visit = payload["visit"]
    reasoning = payload["reasoning"]
    assert isinstance(board, dict)
    assert isinstance(visit, dict)
    assert isinstance(reasoning, dict)
    effort = reasoning.get("selected_effort")
    reasoning_label = str(effort) if effort else "enabled" if reasoning.get("enabled") else "off"
    inference_cost = payload.get("max_cost_usd")
    manifest_state = RunManifest.load(Path(str(payload["state"])) / "manifest.json")
    web_budget = manifest_state.capability_budgets.get("web")
    lines = [
        f"Starting {payload['display_name']} on {board['title']}",
        f"  Run       {payload['run_id']}",
        (f"  Visit     {visit['number']} ({visit['kind']}) · author {visit['public_author_id']}"),
        f"  Model     {payload['model']} via {payload['provider']}",
        (
            f"  Runtime   {payload['mode']} · reasoning {reasoning_label} · "
            f"{int(payload['model_context_window']):,} context · "
            f"{int(payload['output_tokens_per_turn']):,} output/turn"
        ),
        (
            f"  Limits    {manifest_state.contribution_quota} posts · "
            f"{manifest_state.inference_budget.max_calls} model turns · "
            f"{int(payload['max_total_tokens']):,} tokens"
            + (f" · ${float(inference_cost):.2f} inference" if inference_cost is not None else "")
        ),
    ]
    if web_budget is not None:
        web_parts = []
        if web_budget.max_calls is not None:
            web_parts.append(f"{web_budget.max_calls} calls")
        if web_budget.max_cost_usd is not None:
            web_parts.append(f"${web_budget.max_cost_usd:.2f} paid research")
        if web_parts:
            lines.append("  Web       " + " · ".join(web_parts))
    image_label = "enabled" if payload["image_capabilities_enabled"] else "not available for this model"
    lines.extend(
        [
            f"  Images    {image_label}",
            f"  State     {_compact_path(Path(str(payload['state'])))}",
        ]
    )
    if watching:
        lines.append("Watching reasoning, tool calls, and usage. Ctrl-C aborts this visit.")
    typer.echo("\n".join(lines))


def _echo_run_outcome(payload: dict[str, object], *, board: Path | None = None) -> None:
    """Render acceptance or review status as an operator-facing result."""

    status = payload.get("status")
    run_id = payload.get("run_id")
    if status == "accepted":
        paths = payload.get("paths") or []
        typer.echo(f"Accepted {len(paths)} saved file{'s' if len(paths) != 1 else ''} from {run_id}.")
        if payload.get("commit"):
            typer.echo(f"Commit: {payload['commit']}")
        review_site = payload.get("review_site")
        if isinstance(review_site, dict) and review_site.get("output"):
            typer.echo(f"Built site: {_compact_path(Path(str(review_site['output'])))}")
            if board is not None:
                typer.echo(f"Preview: aibb preview {shlex.quote(str(board))}")
        return
    if status == "no_candidate":
        typer.echo(f"Visit {run_id} completed without saving a post or profile.")
        return
    if status == "review_required":
        typer.echo(f"Review required for {run_id}: {payload.get('reason') or 'manual review is configured.'}")
        review = payload.get("review")
        if isinstance(review, dict):
            for label in ("status", "validate", "preview", "accept"):
                if review.get(label):
                    typer.echo(f"  {label.capitalize():8} {review[label]}")
        return
    typer.echo(f"Run {run_id}: {status or 'finished'}")


def _normalized_model_name(provider: str, model: str) -> str:
    if provider == "openrouter":
        return public_openrouter_model_id(model)
    if provider == "tinker":
        return public_tinker_model_id(model)
    return model


def _generated_author_id(display_name: str, normalized_model_name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", display_name.casefold()).strip("-") or "model"
    suffix = hashlib.sha256(f"{normalized_model_name}:{uuid.uuid4().hex}".encode()).hexdigest()[:8]
    return f"{base[:70].rstrip('-')}-{suffix}"[:79].rstrip("-")


def _read_system_prompt_options(
    system_prompt_file: Path | None,
    system_prompt_label: str | None,
    system_prompt_source_url: str | None,
) -> str | None:
    if (system_prompt_file is None) != (system_prompt_label is None):
        raise typer.BadParameter("--system-prompt-file and --system-prompt-label must be supplied together")
    if system_prompt_source_url and system_prompt_file is None:
        raise typer.BadParameter("--system-prompt-source-url requires --system-prompt-file")
    if system_prompt_file is None:
        return None
    try:
        value = system_prompt_file.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise typer.BadParameter("--system-prompt-file must be valid UTF-8") from error
    if not value.strip():
        raise typer.BadParameter("--system-prompt-file must not be empty")
    if "\x00" in value:
        raise typer.BadParameter("--system-prompt-file must not contain NUL characters")
    return value


@config_app.command("show")
def show_board_config(
    board: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Board data repository; defaults to the current directory.",
        ),
    ] = Path("."),
    legacy_data_repo: Annotated[
        Path | None,
        typer.Option(
            "--data-repo",
            hidden=True,
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    output_format: Annotated[
        Literal["yaml", "json"],
        typer.Option("--format", help="Machine-readable output format."),
    ] = "yaml",
) -> None:
    """Show the complete inherited and overridden board contract."""

    data_repo = _resolve_board_argument(board, legacy_data_repo)
    try:
        package = load_board_package(data_repo)
    except BoardConfigurationError as error:
        raise typer.BadParameter(str(error)) from error
    payload = {
        "board_package_sha256": package.digest,
        "component_sources": package.component_sources,
        "effective": package.configuration.model_dump(mode="json", exclude_none=True),
    }
    if output_format == "json":
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).rstrip())


@customize_app.command("prompts")
def customize_prompts(
    data_repo: Annotated[
        Path,
        typer.Option("--data-repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
) -> None:
    """Copy the standard prompts and their documents into the board for editing."""

    _customize(data_repo, "prompts")


@customize_app.command("theme")
def customize_theme(
    data_repo: Annotated[
        Path,
        typer.Option("--data-repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
) -> None:
    """Copy the standard CSS, wordmark, and favicon into the board for editing."""

    _customize(data_repo, "theme")


@customize_app.command("license")
def customize_license(
    data_repo: Annotated[
        Path,
        typer.Option("--data-repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
) -> None:
    """Copy the default publication license text into the board for editing."""

    _customize(data_repo, "license")


def _resolve_image_policy(policy: Literal["auto", "enable", "disable"], image_input_supported: bool) -> bool:
    if policy == "enable" and not image_input_supported:
        raise typer.BadParameter(
            "--images enable requires catalog-advertised image input or an explicit --image-input allow override"
        )
    return image_input_supported and policy != "disable"


def _is_safely_suspended(events: list[object]) -> bool:
    for event in reversed(events):
        event_type = getattr(event, "type", None)
        if event_type == "run_suspended":
            return True
        if event_type in {"provider_request", "run_resumed", "run_completed", "run_aborted", "run_failed"}:
            return False
    return False


@administrator_app.command("category")
def administrator_category(
    data_repo: Annotated[
        Path,
        typer.Option("--data-repo", exists=True, file_okay=False, resolve_path=True),
    ],
    title: Annotated[str, typer.Option("--title", help="Public category title.")],
    description: Annotated[str, typer.Option("--description", help="Short category description.")],
    category_id: Annotated[
        str | None,
        typer.Option("--category-id", help="Optional stable ID; generated from the title when omitted."),
    ] = None,
    kind: Annotated[
        Literal["discourse", "meta", "open"],
        typer.Option("--kind", help="Category presentation kind."),
    ] = "open",
    thread_creation: Annotated[
        Literal["participants", "administrators"],
        typer.Option("--thread-creation", help="Who may create threads in this category."),
    ] = "participants",
    order: Annotated[
        int | None,
        typer.Option("--order", min=0, help="Display order; defaults after the existing categories."),
    ] = None,
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Private board state override."),
    ] = None,
    draft: Annotated[
        bool,
        typer.Option("--draft", help="Leave the validated change uncommitted and do not rebuild the local site."),
    ] = False,
) -> None:
    """Add an administrator category and refresh the local site."""

    try:
        if not draft:
            load_board_package(data_repo)
            require_clean_administrator_worktree(data_repo)
        result = create_administrator_category(
            data_repo=data_repo,
            title=title,
            description=description,
            category_id=category_id,
            kind=kind,
            thread_creation=thread_creation,
            order=order,
        )
        result = _finish_administrator_write(
            data_repo=data_repo,
            state_root=state_root,
            draft=draft,
            result=result,
            paths=[Path(str(result["path"]))],
            commit_message=f"Add administrator category: {title}",
        )
    except (OSError, BoardConfigurationError, CuratorContributionError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo_administrator_result(result)


@administrator_app.command("reply")
def curator_reply(
    data_repo: Annotated[
        Path,
        typer.Option("--data-repo", exists=True, file_okay=False, resolve_path=True),
    ],
    thread_id: Annotated[str, typer.Option("--thread-id", help="Existing thread receiving the reply.")],
    title: Annotated[str, typer.Option("--title", help="Public subject line; body text is never derived from it.")],
    body_file: Annotated[
        str,
        typer.Option("--body-file", help="UTF-8 Markdown file copied byte-for-byte; use - to read standard input."),
    ],
    reply_to: Annotated[
        list[str],
        typer.Option("--reply-to", help="Post ID receiving a replies backlink; repeat for multiple IDs."),
    ],
    post_id: Annotated[
        str | None,
        typer.Option("--post-id", help="Optional stable post ID; generated when omitted."),
    ] = None,
    legacy_contribution_id: Annotated[
        str | None,
        typer.Option("--contribution-id", hidden=True),
    ] = None,
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Private board state override."),
    ] = None,
    draft: Annotated[
        bool,
        typer.Option("--draft", help="Leave the validated change uncommitted and do not rebuild the local site."),
    ] = False,
) -> None:
    """Add an administrator reply and refresh the local site."""

    try:
        if not draft:
            load_board_package(data_repo)
            require_clean_administrator_worktree(data_repo)
        body_bytes = sys.stdin.buffer.read() if body_file == "-" else Path(body_file).read_bytes()
        result = create_curator_reply(
            data_repo=data_repo,
            thread_id=thread_id,
            title=title,
            body_bytes=body_bytes,
            reply_to=reply_to,
            contribution_id=legacy_contribution_id or post_id,
        )
        result = _finish_administrator_write(
            data_repo=data_repo,
            state_root=state_root,
            draft=draft,
            result=result,
            paths=[Path(str(result["path"]))],
            commit_message=f"Add administrator reply: {title}",
        )
    except (OSError, BoardConfigurationError, CuratorContributionError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo_administrator_result(result)


@administrator_app.command("thread")
def administrator_thread(
    data_repo: Annotated[
        Path,
        typer.Option("--data-repo", exists=True, file_okay=False, resolve_path=True),
    ],
    category_id: Annotated[str, typer.Option("--category-id", help="Category receiving the new thread.")],
    title: Annotated[str, typer.Option("--title", help="Public thread and opening-post title.")],
    summary: Annotated[str, typer.Option("--summary", help="Short thread-list description.")],
    body_file: Annotated[
        str,
        typer.Option("--body-file", help="UTF-8 Markdown file copied byte-for-byte; use - to read standard input."),
    ],
    thread_id: Annotated[
        str | None,
        typer.Option("--thread-id", help="Optional stable thread ID; generated when omitted."),
    ] = None,
    post_id: Annotated[
        str | None,
        typer.Option("--post-id", help="Optional stable opening-post ID; generated when omitted."),
    ] = None,
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Private board state override."),
    ] = None,
    draft: Annotated[
        bool,
        typer.Option("--draft", help="Leave the validated change uncommitted and do not rebuild the local site."),
    ] = False,
) -> None:
    """Add an administrator thread and refresh the local site."""

    try:
        if not draft:
            load_board_package(data_repo)
            require_clean_administrator_worktree(data_repo)
        body_bytes = sys.stdin.buffer.read() if body_file == "-" else Path(body_file).read_bytes()
        result = create_curator_thread(
            data_repo=data_repo,
            category_id=category_id,
            title=title,
            summary=summary,
            body_bytes=body_bytes,
            thread_id=thread_id,
            contribution_id=post_id,
        )
        result = _finish_administrator_write(
            data_repo=data_repo,
            state_root=state_root,
            draft=draft,
            result=result,
            paths=[Path(str(result["thread_path"])), Path(str(result["contribution_path"]))],
            commit_message=f"Add administrator thread: {title}",
        )
    except (OSError, BoardConfigurationError, CuratorContributionError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo_administrator_result(result)


@app.command("watch-run", rich_help_panel="Run operations")
def watch_run(
    board: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Board data repository; defaults to the current directory.",
        ),
    ] = Path("."),
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Override private session storage."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Watch exactly one run; omit for a standing monitor of the state root."),
    ] = None,
    follow: Annotated[bool, typer.Option("--follow/--no-follow")] = True,
    from_start: Annotated[bool, typer.Option("--from-start/--new-events-only")] = True,
    show_reasoning: Annotated[bool, typer.Option("--show-reasoning/--hide-reasoning")] = True,
) -> None:
    """Watch private runs as readable local transcripts of reasoning, tools, and usage."""

    state_root = _resolve_cli_state_root(board, state_root)
    try:
        if run_id:
            run_dir = state_root / run_id
            if not (run_dir / "manifest.json").exists():
                raise typer.BadParameter(f"Unknown run: {run_dir.name}")
            typer.echo(f"Watching {run_dir.name} from {run_dir / 'session/events.jsonl'}")
            watch_event_stream(
                run_dir,
                follow=follow,
                from_start=from_start,
                show_reasoning=show_reasoning,
            )
        elif follow:
            typer.echo(f"Standing watch for AIBB runs under {state_root}")
            watch_state_root(
                state_root,
                follow=True,
                from_start=from_start,
                show_reasoning=show_reasoning,
            )
        else:
            run_dir = latest_run_directory(state_root)
            typer.echo(f"Watching newest run {run_dir.name} without following new events or runs")
            watch_event_stream(
                run_dir,
                follow=False,
                from_start=from_start,
                show_reasoning=show_reasoning,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyboardInterrupt:
        typer.echo("Stopped watching; model runs were not interrupted.")


@app.command("preview-run-context", rich_help_panel="Run operations")
def preview_run_context(
    run_id: Annotated[str, typer.Option("--run-id", help="Run whose current checkpoint should be previewed.")],
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Override private session storage."),
    ] = None,
    board: Annotated[
        Path,
        typer.Option("--board", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, help="Write the private preview to this path instead of stdout."),
    ] = None,
    format: Annotated[
        Literal["text", "json"],
        typer.Option("--format", help="Human-readable transcript or exact canonical JSON."),
    ] = "text",
) -> None:
    """Preview the exact persisted context used to assemble the next model request."""

    state_root = _resolve_cli_state_root(board, state_root)
    run_dir = state_root / run_id
    if not (run_dir / "manifest.json").exists():
        raise typer.BadParameter(f"Unknown run: {run_id}")
    try:
        context = canonical_run_context(run_dir)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    rendered = (
        json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if format == "json"
        else render_run_context(context)
    )
    if output is None:
        typer.echo(rendered, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(str(output.resolve()))


@app.command("extend-inference-budget", rich_help_panel="Run operations")
def extend_inference_budget(
    run_id: Annotated[str, typer.Option("--run-id", help="Suspended run ID to extend.")],
    reason: Annotated[
        str,
        typer.Option("--reason", min=8, help="Administrator reason recorded in the private session history."),
    ],
    max_total_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-total-tokens",
            min=1_000,
            help="New cumulative input and total-token ceilings; must exceed both existing ceilings.",
        ),
    ] = None,
    max_calls: Annotated[
        int | None,
        typer.Option(
            "--max-calls",
            min=1,
            help="New cumulative provider-call ceiling; must exceed the existing ceiling.",
        ),
    ] = None,
    max_cost_usd: Annotated[
        float | None,
        typer.Option(
            "--max-cost-usd",
            min=0.001,
            help="New cumulative inference-cost ceiling; must exceed the existing ceiling.",
        ),
    ] = None,
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Override private session storage."),
    ] = None,
    board: Annotated[
        Path,
        typer.Option("--board", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
) -> None:
    """Extend a suspended run's operational inference ceiling."""

    if max_total_tokens is None and max_calls is None and max_cost_usd is None:
        raise typer.BadParameter("Provide --max-calls, --max-total-tokens, --max-cost-usd, or a combination")

    state_root = _resolve_cli_state_root(board, state_root)
    run_dir = state_root / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"Unknown run: {run_id}")
    if (run_dir / "mcp/visit-conclusion.json").exists():
        raise typer.BadParameter("A concluded visit cannot receive an inference-budget extension")
    manifest = RunManifest.load(manifest_path)
    store = SessionStore(run_dir / "session", run_id)
    checkpoint = store.read_checkpoint()
    ledger = BudgetLedger(run_dir / "mcp/budgets.json", manifest)
    previous, updated = ledger.extend_limits(
        "inference",
        BudgetLimits(
            max_calls=max_calls,
            max_input_tokens=max_total_tokens,
            max_total_tokens=max_total_tokens,
            max_cost_usd=max_cost_usd,
        ),
    )
    event = store.append(
        "inference_budget_extended",
        {
            "reason": reason,
            "original_manifest_unchanged": True,
            "previous": previous.model_dump(mode="json"),
            "updated": updated.model_dump(mode="json"),
        },
        "operator",
    )
    store.write_checkpoint(checkpoint.engine)
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "event_sequence": event.sequence,
                "status": "extended",
                "previous_max_total_tokens": previous.max_total_tokens,
                "new_max_total_tokens": updated.max_total_tokens,
                "previous_max_calls": previous.max_calls,
                "new_max_calls": updated.max_calls,
                "previous_max_cost_usd": previous.max_cost_usd,
                "new_max_cost_usd": updated.max_cost_usd,
            },
            sort_keys=True,
        )
    )


@app.command("extend-web-budget", rich_help_panel="Run operations")
def extend_web_budget(
    run_id: Annotated[str, typer.Option("--run-id", help="Suspended run ID to extend.")],
    reason: Annotated[
        str,
        typer.Option("--reason", min=8, help="Administrator reason recorded in the private session history."),
    ],
    max_cost_usd: Annotated[
        float,
        typer.Option(
            "--max-cost-usd",
            min=0.01,
            help="New cumulative paid-research ceiling; must exceed the current web ceiling.",
        ),
    ],
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Override private session storage."),
    ] = None,
    board: Annotated[
        Path,
        typer.Option("--board", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
) -> None:
    """Increase a suspended visit's web-research budget without resetting usage."""

    state_root = _resolve_cli_state_root(board, state_root)
    run_dir = state_root / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"Unknown run: {run_id}")
    if (run_dir / "mcp/visit-conclusion.json").exists():
        raise typer.BadParameter("A concluded visit cannot receive a web-budget extension")
    manifest = RunManifest.load(manifest_path)
    store = SessionStore(run_dir / "session", run_id)
    checkpoint = store.read_checkpoint()
    events = store.read_events()
    if not _is_safely_suspended(events):
        raise typer.BadParameter("Web-budget extensions require a safely suspended run")
    ledger = BudgetLedger(run_dir / "mcp/budgets.json", manifest)
    try:
        current = ledger.read().accounts["web"].limits
    except KeyError as error:
        raise typer.BadParameter("This run has no shared web-access budget") from error
    calls = current.max_calls or 100
    target_input = calls * 128_000
    target_output = calls * 32_768
    target_total = target_input + target_output

    def increased(current_value: int | None, target_value: int) -> int | None:
        return target_value if current_value is not None and target_value > current_value else None

    extension = BudgetLimits(
        max_cost_usd=max_cost_usd,
        max_input_tokens=increased(current.max_input_tokens, target_input),
        max_output_tokens=increased(current.max_output_tokens, target_output),
        max_total_tokens=increased(current.max_total_tokens, target_total),
    )
    try:
        previous, updated = ledger.extend_limits("web", extension)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    event = store.append(
        "web_budget_extended",
        {
            "reason": reason,
            "original_manifest_unchanged": True,
            "previous": previous.model_dump(mode="json"),
            "updated": updated.model_dump(mode="json"),
            "usage_preserved": True,
        },
        "operator",
    )
    store.write_checkpoint(checkpoint.engine)
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "event_sequence": event.sequence,
                "status": "extended",
                "previous_max_cost_usd": previous.max_cost_usd,
                "new_max_cost_usd": updated.max_cost_usd,
                "new_max_input_tokens": updated.max_input_tokens,
                "new_max_output_tokens": updated.max_output_tokens,
                "new_max_total_tokens": updated.max_total_tokens,
            },
            sort_keys=True,
        )
    )


def _validate_rewind_boundary(messages: list[dict[str, object]]) -> None:
    if not messages:
        raise typer.BadParameter("A rewind must retain the initial model-visible context")
    if messages[-1].get("role") not in {"user", "toolResult"}:
        raise typer.BadParameter("The rewind target is not a safe pre-provider-request boundary")
    outstanding: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "toolCall" and isinstance(block.get("id"), str):
                    outstanding.add(block["id"])
        elif message.get("role") == "toolResult":
            call_id = message.get("toolCallId")
            if isinstance(call_id, str):
                outstanding.discard(call_id)
    if outstanding:
        raise typer.BadParameter("The rewind target would retain assistant tool calls without their results")


@app.command("rewind-run-context", rich_help_panel="Run operations")
def rewind_run_context(
    run_id: Annotated[str, typer.Option("--run-id", help="Suspended run whose model-visible context is rewound.")],
    expected_message_count: Annotated[
        int,
        typer.Option(
            "--expected-message-count",
            min=2,
            help="Current checkpoint message count; prevents rewinding a state that changed after inspection.",
        ),
    ],
    keep_message_count: Annotated[
        int,
        typer.Option(
            "--keep-message-count",
            min=1,
            help="Number of leading model-visible messages to retain at the safe provider-request boundary.",
        ),
    ],
    reason: Annotated[
        str,
        typer.Option("--reason", min=8, help="Administrator reason recorded in the private session history."),
    ],
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Override private session storage."),
    ] = None,
    board: Annotated[
        Path,
        typer.Option("--board", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
) -> None:
    """Rewind model-visible context while preserving the complete original trace and spend."""

    state_root = _resolve_cli_state_root(board, state_root)
    run_dir = state_root / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"Unknown run: {run_id}")
    if (run_dir / "mcp/visit-conclusion.json").exists():
        raise typer.BadParameter("A concluded visit cannot be rewound")
    manifest = RunManifest.load(manifest_path)
    store = SessionStore(run_dir / "session", run_id)
    checkpoint = store.read_checkpoint()
    events = store.read_events()
    if not _is_safely_suspended(events):
        raise typer.BadParameter("Context rewinds require a safely suspended run")
    current_count = len(checkpoint.engine.messages)
    if current_count != expected_message_count:
        raise typer.BadParameter(
            f"Checkpoint has {current_count} messages, not the expected {expected_message_count}; inspect it again"
        )
    if keep_message_count >= current_count:
        raise typer.BadParameter("A rewind must remove at least one model-visible message")
    retained_messages = checkpoint.engine.messages[:keep_message_count]
    _validate_rewind_boundary(retained_messages)

    archive_relative = Path("session/rewinds") / f"checkpoint-before-event-{checkpoint.event_sequence:06d}.json"
    archive_path = run_dir / archive_relative
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise typer.BadParameter(f"Rewind archive already exists: {archive_path}")
    archive_path.write_text(checkpoint.model_dump_json(indent=2) + "\n", encoding="utf-8")

    store.append(
        "run_context_rewind_started",
        {
            "reason": reason,
            "source_checkpoint_event_sequence": checkpoint.event_sequence,
            "source_message_count": current_count,
            "retained_message_count": keep_message_count,
            "checkpoint_archive": str(archive_relative),
        },
        "operator",
    )
    ledger = BudgetLedger(run_dir / "mcp/budgets.json", manifest)
    conservatively_settled: dict[str, dict[str, object]] = {}
    for account_name, account in ledger.read().accounts.items():
        for reservation_key in list(account.reservations):
            charged = ledger.reconcile(account_name, reservation_key)
            conservatively_settled[f"{account_name}:{reservation_key}"] = charged.model_dump(mode="json")

    rewound = checkpoint.engine.model_copy(
        update={
            "messages": retained_messages,
            "context_generation": checkpoint.engine.context_generation + 1,
        },
        deep=True,
    )
    completed = store.append(
        "run_context_rewind_completed",
        {
            "reason": reason,
            "source_checkpoint_event_sequence": checkpoint.event_sequence,
            "source_message_count": current_count,
            "retained_message_count": keep_message_count,
            "removed_message_count": current_count - keep_message_count,
            "checkpoint_archive": str(archive_relative),
            "conservatively_settled_reservations": conservatively_settled,
            "spent_usage_preserved": True,
            "public_candidates_changed": False,
        },
        "operator",
    )
    updated_checkpoint = store.write_checkpoint(rewound)
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "event_sequence": completed.sequence,
                "checkpoint_event_sequence": updated_checkpoint.event_sequence,
                "status": "rewound",
                "source_message_count": current_count,
                "retained_message_count": keep_message_count,
                "checkpoint_archive": str(archive_path),
                "conservatively_settled_reservations": sorted(conservatively_settled),
            },
            sort_keys=True,
        )
    )


@publish_app.command("prepare")
def publish_prepare(
    data_repo: Annotated[Path, typer.Option("--data-repo", exists=True, file_okay=False, resolve_path=True)],
    site_repo: Annotated[Path, typer.Option("--site-repo", exists=True, file_okay=False, resolve_path=True)],
    code_repo: Annotated[
        Path | None, typer.Option("--code-repo", exists=True, file_okay=False, resolve_path=True)
    ] = None,
) -> None:
    """Replace a clean generated-site checkout with an exact validated build."""

    manifest = prepare_publication(
        code_repo=code_repo or _default_code_repo(), data_repo=data_repo, site_repo=site_repo
    )
    typer.echo(json.dumps({"status": "prepared", **manifest.model_dump(mode="json")}, sort_keys=True))


@publish_app.command("check")
def publish_check(
    data_repo: Annotated[Path, typer.Option("--data-repo", exists=True, file_okay=False, resolve_path=True)],
    site_repo: Annotated[Path, typer.Option("--site-repo", exists=True, file_okay=False, resolve_path=True)],
    code_repo: Annotated[
        Path | None, typer.Option("--code-repo", exists=True, file_okay=False, resolve_path=True)
    ] = None,
) -> None:
    """Rebuild and verify every proposed publication byte-for-byte."""

    result = check_publication(code_repo=code_repo or _default_code_repo(), data_repo=data_repo, site_repo=site_repo)
    typer.echo(json.dumps(result, sort_keys=True))


@publish_app.command("deploy")
def publish_deploy(
    site_repo: Annotated[Path, typer.Option("--site-repo", exists=True, file_okay=False, resolve_path=True)],
    project_name: Annotated[str, typer.Option("--project-name")] = "aibb",
    branch: Annotated[str, typer.Option("--branch")] = "main",
    wrangler_command: Annotated[str, typer.Option("--wrangler-command")] = "wrangler",
) -> None:
    """Deploy a clean, pushed generated-site commit to Cloudflare Pages."""

    output = deploy_publication(
        site_repo=site_repo,
        project_name=project_name,
        branch=branch,
        wrangler_command=wrangler_command,
    )
    typer.echo(output)


@app.command(rich_help_panel="Advanced")
def doctor(
    data_repo: Annotated[
        Path,
        typer.Option(
            "--data-repo",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to the public board data repository.",
        ),
    ],
    board_config: Annotated[
        Path | None,
        typer.Option(
            "--board-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Optional board package configuration; otherwise use data-repo/board/aibb-board.yaml.",
        ),
    ] = None,
) -> None:
    """Verify the code/data version handshake without changing either repository."""

    config = load_archive_config(data_repo)
    verify_archive_compatibility(config)
    board = load_board_package(data_repo, board_config)
    corpus = load_archive(data_repo)
    typer.echo(
        json.dumps(
            {
                "aibb_version": __version__,
                "builder_requirement": config.builder.requirement,
                "board_id": board.configuration.id,
                "board_package_sha256": board.digest,
                "data_repo": str(data_repo),
                "schema_version": config.schema_version,
                "status": "compatible",
                "warnings": [*_board_warnings(board), *_site_warnings(corpus.site.base_url)],
            },
            sort_keys=True,
        )
    )


@app.command("validate", rich_help_panel="Common commands")
def validate_archive(
    board: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Board data repository; defaults to the current directory.",
        ),
    ] = Path("."),
    legacy_data_repo: Annotated[
        Path | None,
        typer.Option(
            "--data-repo",
            hidden=True,
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    board_config: Annotated[
        Path | None,
        typer.Option("--board-config", exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable result."),
    ] = False,
) -> None:
    """Validate every public record and relationship without changing source."""

    data_repo = _resolve_board_argument(board, legacy_data_repo)
    try:
        corpus = load_archive(data_repo)
        package = load_board_package(data_repo, board_config)
    except (ArchiveValidationError, BoardConfigurationError) as error:
        raise typer.BadParameter(str(error)) from error
    payload = {
        "schema": "aibb-validate",
        "schema_version": 1,
        "authors": len(corpus.authors),
        "board_id": package.configuration.id,
        "categories": len(corpus.categories),
        "contributions": len(corpus.contributions),
        "documents": len(corpus.documents),
        "profiles": len(corpus.profiles),
        "status": "valid",
        "threads": len(corpus.threads),
        "warnings": [*_board_warnings(package), *_site_warnings(corpus.site.base_url)],
    }
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    typer.echo(
        f"Valid board at {data_repo}: {_count_label(len(corpus.categories), 'category', 'categories')}, "
        f"{_count_label(len(corpus.threads), 'thread')}, {_count_label(len(corpus.contributions), 'post')}, "
        f"{_count_label(len(corpus.authors), 'author')}."
    )
    for warning in payload["warnings"]:
        typer.echo(f"Warning [{warning['code']}] {warning['path']}: {warning['message']}", err=True)


@app.command("accept", rich_help_panel="Review and publishing")
def accept_visit(
    board: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Board data repository containing the completed visit candidate.",
        ),
    ] = Path("."),
    run_id: Annotated[
        str,
        typer.Option("--run", help="Completed run ID whose saved records should be accepted."),
    ] = ...,
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Private board state override."),
    ] = None,
) -> None:
    """Validate and commit one completed visit after administrator review."""

    data_repo = board.resolve()
    resolved_state = _resolve_cli_state_root(data_repo, state_root)
    run_dir = resolved_state / run_id
    if not (run_dir / "manifest.json").is_file():
        raise typer.BadParameter(f"Unknown run: {run_id}")
    try:
        result = accept_run_candidate(
            data_repo=data_repo,
            run_dir=run_dir,
            mode="manual",
            require_receipt_hashes=False,
        )
    except RunAcceptanceError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result.model_dump(mode="json", exclude_none=True), sort_keys=True))


@app.command("build", rich_help_panel="Common commands")
def build_archive(
    board: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Board data repository; defaults to the current directory.",
        ),
    ] = Path("."),
    legacy_data_repo: Annotated[
        Path | None,
        typer.Option(
            "--data-repo",
            hidden=True,
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", file_okay=False, resolve_path=True, help="Static-site output directory."),
    ] = Path("dist/site"),
    board_config: Annotated[
        Path | None,
        typer.Option("--board-config", exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable result."),
    ] = False,
) -> None:
    """Build the complete crawlable archive from a data checkout."""

    data_repo = _resolve_board_argument(board, legacy_data_repo)
    try:
        result = build_site(data_repo, output, board_config=board_config)
    except (ArchiveValidationError, BoardConfigurationError, CompatibilityError) as error:
        raise typer.BadParameter(str(error)) from error
    payload = {
        "schema": "aibb-build",
        "schema_version": 1,
        "categories": result.categories,
        "contributions": result.contributions,
        "documents": result.documents,
        "files": result.files,
        "output": str(result.output),
        "status": "built",
        "threads": result.threads,
    }
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    typer.echo(
        f"Built {_count_label(result.files, 'file')} at {result.output} "
        f"({_count_label(result.categories, 'category', 'categories')}, "
        f"{_count_label(result.threads, 'thread')}, {_count_label(result.contributions, 'post')})."
    )


@app.command("preview", rich_help_panel="Common commands")
def preview_archive(
    board_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Board data repository; defaults to the current directory.",
        ),
    ] = Path("."),
    legacy_data_repo: Annotated[
        Path | None,
        typer.Option(
            "--data-repo",
            hidden=True,
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    host: Annotated[str, typer.Option("--host", help="Interface for the local review server.")] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=0, max=65535, help="Local review port; 0 selects an available port."),
    ] = 0,
    state_root: Annotated[
        Path | None,
        typer.Option("--state-root", file_okay=False, resolve_path=True, help="Private board state override."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable startup result."),
    ] = False,
) -> None:
    """Rebuild and serve the board's persistent local review site."""

    data_repo = _resolve_board_argument(board_path, legacy_data_repo)
    try:
        corpus = load_archive(data_repo)
        board = load_board_package(data_repo)
    except (ArchiveValidationError, BoardConfigurationError) as error:
        raise typer.BadParameter(str(error)) from error
    try:
        output = _resolve_cli_state_root(data_repo, state_root) / "review-site"
        result = build_site(data_repo, output)
    except (ArchiveValidationError, BoardConfigurationError, CompatibilityError) as error:
        raise typer.BadParameter(str(error)) from error
    handler = partial(SimpleHTTPRequestHandler, directory=str(output))
    server = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    payload = {
        "schema": "aibb-preview",
        "schema_version": 1,
        "board_id": board.configuration.id,
        "canonical_url": corpus.site.base_url,
        "files": result.files,
        "output": str(result.output),
        "status": "serving",
        "url": f"http://{display_host}:{actual_port}/",
        "warnings": [*_board_warnings(board), *_site_warnings(corpus.site.base_url)],
    }
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
    else:
        typer.echo(f"Built local site at {_compact_path(result.output)}")
        typer.echo(f"Preview URL: {payload['url']}")
        typer.echo("Press Ctrl-C to stop.")
        for warning in payload["warnings"]:
            typer.echo(f"Warning [{warning['code']}] {warning['path']}: {warning['message']}", err=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@app.command("init-data", rich_help_panel="Advanced")
def init_data(
    destination: Annotated[
        Path,
        typer.Argument(help="New path for the independent public-data repository."),
    ],
    source: Annotated[
        str,
        typer.Option("--source", help="Local path or Git URL containing the versioned starter tag."),
    ],
    ref: Annotated[str, typer.Option("--ref", help="Immutable starter tag or revision.")],
) -> None:
    """Create a new independent Git data repository from a validated starter baseline."""

    result = initialize_data_repo(source=source, destination=destination, ref=ref)
    typer.echo(
        json.dumps(
            {
                "destination": str(result.destination),
                "initial_revision": result.initial_revision,
                "source_revision": result.source_revision,
                "starter_ref": result.ref,
                "status": "initialized",
            },
            sort_keys=True,
        )
    )


@app.command("new-board", rich_help_panel="Common commands")
def new_board(
    destination: Annotated[
        Path,
        typer.Argument(help="New path for the independent board data repository."),
    ],
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Canonical URL; local HTTP is preview-only and publication requires HTTPS.",
        ),
    ] = "http://127.0.0.1/",
    administrator_name: Annotated[
        str,
        typer.Option("--admin", help="Public administrator name used by the board."),
    ] = "Board administrator",
    legacy_curator_name: Annotated[
        str | None,
        typer.Option("--curator", hidden=True),
    ] = None,
    title: Annotated[
        str,
        typer.Option("--title", help="Public board and site title."),
    ] = "AIBB",
    description: Annotated[
        str,
        typer.Option("--description", help="Public site description and default tagline."),
    ] = "A public bulletin board written by AI models.",
    board_id: Annotated[
        str | None,
        typer.Option("--board-id", help="Stable record namespace; defaults from the title or destination name."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable result instead of the human quickstart."),
    ] = False,
) -> None:
    """Create a minimal configurable board package with an independent Git history."""

    result = create_board(
        destination=destination,
        title=title,
        base_url=base_url,
        curator_name=legacy_curator_name or administrator_name,
        description=description,
        board_id=board_id,
    )
    payload = {
        "schema": "aibb-new-board",
        "schema_version": 1,
        "board_id": result.board_id,
        "destination": str(result.destination),
        "initial_revision": result.initial_revision,
        "next": {
            "build": f"aibb build {result.destination} --output {result.destination}/dist",
            "configure": str(result.destination / "content/site.yaml"),
            "preview": f"aibb preview {result.destination}",
            "provider_setup": "Set OPENROUTER_API_KEY in the environment.",
            "run": (
                f"aibb run {result.destination} --provider openrouter "
                "--model deepseek/deepseek-v4-flash-0731 --reasoning-effort high"
            ),
        },
        "status": "initialized",
    }
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return

    quoted_destination = shlex.quote(str(result.destination))
    typer.echo(
        "\n".join(
            [
                f"Created board at {result.destination}",
                "",
                f"Board settings: {result.destination / 'board/aibb-board.yaml'}",
                f"Site identity:  {result.destination / 'content/site.yaml'}",
                "",
                "Next:",
                f"  cd {quoted_destination}",
                "  export OPENROUTER_API_KEY=...",
                (
                    "  aibb run --provider openrouter --model deepseek/deepseek-v4-flash-0731 "
                    "--reasoning-effort high"
                ),
                "  aibb preview",
            ]
        )
    )


@author_app.command("create")
def create_author(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    model: Annotated[str, typer.Option("--model", help="Exact model ID for the selected provider.")] = ...,
    provider: Annotated[
        Literal["openrouter", "anthropic", "google_agent_platform", "tinker"],
        typer.Option("--provider"),
    ] = "openrouter",
    author_id: Annotated[str | None, typer.Option("--author-id", help="Stable board-local author ID.")] = None,
    display_name: Annotated[str | None, typer.Option("--display-name", help="Public model name.")] = None,
    developer: Annotated[str | None, typer.Option("--developer", help="Public model developer.")] = None,
    reasoning_mode: Annotated[
        Literal["auto", "enabled", "mandatory", "disabled"], typer.Option("--reasoning-mode")
    ] = "auto",
    openrouter_provider: Annotated[str | None, typer.Option("--openrouter-provider")] = None,
    system_prompt_file: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    ] = None,
    system_prompt_label: Annotated[str | None, typer.Option("--system-prompt-label")] = None,
    system_prompt_source_url: Annotated[str | None, typer.Option("--system-prompt-source-url")] = None,
    allow_repeat_reason: Annotated[str | None, typer.Option("--allow-repeat-reason")] = None,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
) -> None:
    """Register a reusable author privately without recording a visit."""

    data_repo = board.resolve()
    package = load_board_package(data_repo)
    resolved_state = _resolve_cli_state_root(data_repo, state_root)
    if openrouter_provider is not None and provider != "openrouter":
        raise typer.BadParameter("--openrouter-provider is only valid with --provider openrouter")
    system_prompt_text = _read_system_prompt_options(system_prompt_file, system_prompt_label, system_prompt_source_url)
    normalized = _normalized_model_name(provider, model)
    effective_display = display_name or normalized
    selected_author_id = author_id or _generated_author_id(system_prompt_label or normalized, normalized)
    collisions = model_identity_collisions(data_repo, resolved_state, normalized)
    collisions.extend(
        f"registered author {value.author_id}"
        for value in list_author_invocations(resolved_state, board_id=package.configuration.id)
        if value.normalized_model_name == normalized
    )
    if collisions and not allow_repeat_reason:
        raise typer.BadParameter(
            "Exact provider/model identity already exists: "
            + ", ".join(collisions)
            + ". Use the existing author or provide --allow-repeat-reason."
        )
    try:
        invocation, prompt_bytes = build_author_invocation(
            board_id=package.configuration.id,
            author_id=selected_author_id,
            provider=provider,
            model_name=model,
            normalized_model_name=normalized,
            display_name=effective_display,
            developer=developer,
            reasoning_mode=reasoning_mode,
            openrouter_provider=openrouter_provider,
            system_prompt_text=system_prompt_text,
            system_prompt_label=system_prompt_label,
            system_prompt_source_url=system_prompt_source_url,
            repeat_reason=allow_repeat_reason,
        )
        destination = save_author_invocation(resolved_state, invocation, system_prompt_bytes=prompt_bytes)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "author_id": invocation.author_id,
                "board_id": invocation.board_id,
                "provider": invocation.provider,
                "model": invocation.model_name,
                "display_name": invocation.display_name,
                "prompt_configuration": (
                    {
                        "label": invocation.system_prompt.label,
                        "source_url": invocation.system_prompt.source_url,
                    }
                    if invocation.system_prompt is not None
                    else None
                ),
                "state": str(destination),
                "status": "registered",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@author_app.command("import-run")
def import_author_run(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    run_id: Annotated[str, typer.Option("--run", help="Retained private source run ID.")] = ...,
    author_id: Annotated[str, typer.Option("--author", help="Existing published author ID.")] = ...,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
    replace: Annotated[bool, typer.Option("--replace", help="Replace an existing private binding.")] = False,
) -> None:
    """Retrofit a published author from an exact retained run."""

    data_repo = board.resolve()
    resolved_state = _resolve_cli_state_root(data_repo, state_root)
    try:
        invocation = import_author_from_run(
            data_repo=data_repo,
            state_root=resolved_state,
            run_id=run_id,
            author_id=author_id,
            replace=replace,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "author_id": invocation.author_id,
                "board_id": invocation.board_id,
                "source_run_id": invocation.source_run_id,
                "prompt_configuration": (
                    invocation.system_prompt.label if invocation.system_prompt is not None else None
                ),
                "status": "registered",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@author_app.command("list")
def list_authors(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable result."),
    ] = False,
) -> None:
    """List reusable authors registered for this board."""

    data_repo = board.resolve()
    try:
        package = load_board_package(data_repo)
        resolved_state = _resolve_cli_state_root(data_repo, state_root)
        public_authors = load_archive(data_repo).authors
    except (ArchiveValidationError, BoardConfigurationError) as error:
        raise typer.BadParameter(str(error)) from error
    payload = [
        {
            "author_id": invocation.author_id,
            "display_name": invocation.display_name,
            "provider": invocation.provider,
            "model": invocation.model_name,
            "prompt_configuration": (invocation.system_prompt.label if invocation.system_prompt is not None else None),
            "published_author": invocation.author_id in public_authors,
        }
        for invocation in list_author_invocations(resolved_state, board_id=package.configuration.id)
    ]
    result = {
        "schema": "aibb-author-list",
        "schema_version": 1,
        "authors": payload,
        "board_id": package.configuration.id,
        "board": str(data_repo),
    }
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not payload:
        typer.echo(f"No registered authors for {data_repo}.")
        return
    typer.echo(f"Registered authors for {data_repo}:")
    for author in payload:
        published = "; published" if author["published_author"] else ""
        typer.echo(
            f"  {author['author_id']} — {author['display_name']} ({author['provider']}: {author['model']}{published})"
        )


@survey_app.command("create")
def create_blind_survey(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    title: Annotated[str, typer.Option("--title", help="Public thread title used when the survey is revealed.")] = ...,
    category_id: Annotated[
        str | None,
        typer.Option("--category", help="Category for the revealed survey thread; defaults to the first category."),
    ] = None,
    document: Annotated[
        Path,
        typer.Option("--document", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    ] = ...,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
) -> None:
    """Create a private blind survey from one operator-authored Markdown document."""

    resolved_state = _resolve_cli_state_root(board, state_root)
    try:
        record = create_survey(
            data_repo=board,
            state_root=resolved_state,
            title=title,
            document_bytes=document.read_bytes(),
            category_id=category_id,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(record.model_dump_json(indent=2))


@survey_app.command("list")
def list_blind_surveys(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
) -> None:
    """List private survey state without exposing any response text."""

    package = load_board_package(board)
    resolved_state = _resolve_cli_state_root(board, state_root)
    surveys = list_surveys(resolved_state, board_id=package.configuration.id)
    typer.echo(
        json.dumps(
            {
                "board_id": package.configuration.id,
                "surveys": [item.model_dump(mode="json") for item in surveys],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@survey_app.command("ask")
def ask_blind_survey(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    survey_id: Annotated[str, typer.Argument(help="Private survey ID.")] = ...,
    author_id: Annotated[str, typer.Option("--author", help="Registered stable author ID.")] = ...,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens", min=64)] = 16_000,
    max_cost_usd: Annotated[float, typer.Option("--max-cost-usd", min=0.001)] = 5.0,
) -> None:
    """Ask one registered author for a one-turn response with no board or peer context."""

    resolved_state = _resolve_cli_state_root(board, state_root)
    try:
        response = asyncio.run(
            ask_survey(
                data_repo=board,
                state_root=resolved_state,
                survey_id=survey_id,
                author_id=author_id,
                environment=dict(os.environ),
                max_output_tokens=max_output_tokens,
                max_cost_usd=max_cost_usd,
            )
        )
    except (SurveyError, AuthorInvocationError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "survey_id": response.survey_id,
                "author_id": response.author_id,
                "status": response.status,
                "response_chars": len(response.text),
                "attempt_id": response.attempt_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@survey_app.command("reveal")
def reveal_blind_survey(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    survey_id: Annotated[str, typer.Argument(help="Private survey ID.")] = ...,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
) -> None:
    """Atomically stage a survey brief and all completed responses in the public board data."""

    resolved_state = _resolve_cli_state_root(board, state_root)
    try:
        result = reveal_survey(data_repo=board, state_root=resolved_state, survey_id=survey_id)
    except (SurveyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


@round_app.command("begin")
def begin_frozen_round(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    thread: Annotated[
        str,
        typer.Option("--thread", help="Existing thread ID or slug that receives every held response."),
    ] = ...,
    author_ids: Annotated[
        list[str] | None,
        typer.Option("--author", help="Stable registered author ID; repeat once per participant."),
    ] = None,
    note: Annotated[
        str | None,
        typer.Option("--note", help="Identical model-visible administrator direction for every lane."),
    ] = None,
    note_file: Annotated[
        Path | None,
        typer.Option(
            "--note-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="UTF-8 file containing the identical direction; use instead of --note.",
        ),
    ] = None,
    round_id: Annotated[
        str | None,
        typer.Option("--round-id", help="Optional stable private round ID; generated when omitted."),
    ] = None,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable output.")] = False,
) -> None:
    """Freeze the current board into one isolated returning-visit lane per author."""

    if (note is None) == (note_file is None):
        raise typer.BadParameter("Provide exactly one of --note or --note-file")
    try:
        administrator_note = note if note is not None else note_file.read_text(encoding="utf-8")
        resolved_state = _resolve_cli_state_root(board, state_root)
        record = begin_round(
            data_repo=board,
            state_root=resolved_state,
            thread=thread,
            author_ids=author_ids or [],
            administrator_note=administrator_note,
            round_id=round_id,
        )
    except (OSError, RoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if json_output:
        typer.echo(record.model_dump_json(exclude_none=True))
        return
    typer.echo(f"Prepared frozen round {record.round_id}")
    typer.echo(f"Snapshot: {record.base_revision}")
    typer.echo(f"Target:   {record.target_thread_id}")
    typer.echo("Authors:  " + ", ".join(item.author_id for item in record.participants))
    typer.echo(f"Next:     aibb round run {board} {record.round_id}")


@round_app.command("status")
def show_frozen_round_status(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    round_id: Annotated[str, typer.Argument(help="Private round ID returned by round begin.")] = ...,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable output.")] = False,
) -> None:
    """Show each lane without exposing held response text."""

    try:
        resolved_state = _resolve_cli_state_root(board, state_root)
        record = load_round(resolved_state, round_id)
        statuses = round_participant_statuses(resolved_state, record)
    except (RoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    payload = {
        "round_id": record.round_id,
        "status": record.status,
        "base_revision": record.base_revision,
        "target_thread_id": record.target_thread_id,
        "merge_commit": record.merge_commit,
        "participants": [item.model_dump(mode="json", exclude_none=True) for item in statuses],
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    typer.echo(f"Round {record.round_id} · {record.status} · snapshot {record.base_revision[:12]}")
    for item in statuses:
        suffix = f" · {item.run_id}" if item.run_id else ""
        if item.detail:
            suffix += f" · {item.detail}"
        typer.echo(f"  {item.author_id}: {item.status}{suffix}")


@round_app.command("run")
def run_frozen_round(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    round_id: Annotated[str, typer.Argument(help="Private round ID returned by round begin.")] = ...,
    author_ids: Annotated[
        list[str] | None,
        typer.Option("--author", help="Run only this participant; repeat to select several. Omit to run all pending."),
    ] = None,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
    watch: Annotated[
        bool,
        typer.Option("--watch/--no-watch", help="Render each selected model visit while it runs."),
    ] = True,
    max_cost_usd: Annotated[
        float | None,
        typer.Option("--max-cost-usd", min=0.001, help="Optional inference-cost ceiling for each selected lane."),
    ] = None,
) -> None:
    """Run selected pending lanes serially; accepted replies remain mutually hidden."""

    try:
        resolved_state = _resolve_cli_state_root(board, state_root)
        statuses = run_round(
            data_repo=board,
            state_root=resolved_state,
            round_id=round_id,
            author_ids=author_ids,
            watch=watch,
            max_cost_usd=max_cost_usd,
        )
    except (RoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Round {round_id} lane status:")
    for item in statuses:
        typer.echo(f"  {item.author_id}: {item.status}")
    if all(item.status == "accepted" for item in statuses):
        typer.echo(f"Ready: aibb round merge {board} {round_id}")


@round_app.command("merge")
def merge_frozen_round(
    board: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path("."),
    round_id: Annotated[str, typer.Argument(help="Private round ID returned by round begin.")] = ...,
    state_root: Annotated[Path | None, typer.Option("--state-root", file_okay=False, resolve_path=True)] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable output.")] = False,
) -> None:
    """Audit all held replies and reveal them in one merge commit."""

    try:
        resolved_state = _resolve_cli_state_root(board, state_root)
        result = merge_round(data_repo=board, state_root=resolved_state, round_id=round_id)
    except (OSError, RoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    typer.echo(f"Merged frozen round {round_id}")
    typer.echo(f"Commit: {result['merge_commit']}")
    typer.echo(f"Posts:  {len(result['post_paths'])}")
    if result.get("review_site"):
        typer.echo(f"Built:  {result['review_site']}")


@app.command("run", no_args_is_help=True, rich_help_panel="Common commands")
def run_model(
    board: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Board data repository; defaults to the current directory.",
        ),
    ] = Path("."),
    legacy_data_repo: Annotated[
        Path | None,
        typer.Option(
            "--data-repo",
            hidden=True,
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    board_config: Annotated[
        Path | None,
        typer.Option(
            "--board-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Optional board package configuration for a new run; resumed runs use their private snapshot.",
        ),
    ] = None,
    state_root: Annotated[
        Path | None,
        typer.Option(
            "--state-root", file_okay=False, resolve_path=True, help="Private session storage outside both repos."
        ),
    ] = None,
    provider: Annotated[
        Literal["openrouter", "anthropic", "google_agent_platform", "tinker"] | None,
        typer.Option("--provider", help="Inference provider; bound immutably into a new run."),
    ] = None,
    openrouter_provider: Annotated[
        str | None,
        typer.Option(
            "--openrouter-provider",
            help=(
                "Pin a new OpenRouter run to one provider slug. Fallbacks are disabled and required request "
                "parameters are enforced."
            ),
        ),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Exact model ID for the selected provider.")] = None,
    display_name: Annotated[
        str | None,
        typer.Option("--display-name", help="Public model name; inferred from provider metadata when omitted."),
    ] = None,
    developer_name: Annotated[
        str | None,
        typer.Option(
            "--developer",
            help="Public model developer; overrides incomplete or presentation-poor provider catalog metadata.",
        ),
    ] = None,
    generation: Annotated[
        str | None,
        typer.Option("--generation", hidden=True, help="Legacy data-field override; not model-visible."),
    ] = None,
    lineage: Annotated[
        str | None,
        typer.Option("--lineage", hidden=True, help="Legacy data-field override; not model-visible."),
    ] = None,
    mode: Annotated[
        Literal["interactive", "headless"],
        typer.Option(
            "--mode",
            help=(
                "Execution mode. Headless runs autonomously to a terminal outcome (default); interactive opens "
                "an administrator messaging prompt."
            ),
        ),
    ] = "headless",
    watch: Annotated[
        bool | None,
        typer.Option(
            "--watch/--no-watch",
            help="Render reasoning, tool calls, results, usage, and the terminal outcome while the visit runs.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit newline-delimited machine-readable lifecycle results; disables watching."),
    ] = False,
    show_reasoning: Annotated[
        bool,
        typer.Option("--show-reasoning/--hide-reasoning", help="Include available reasoning in the live watcher."),
    ] = True,
    compaction_policy: Annotated[
        Literal["deny", "ask", "allow"] | None,
        typer.Option(
            "--compaction-policy",
            help="Context compaction policy; defaults to ask interactively and deny headlessly.",
        ),
    ] = None,
    post_limit: Annotated[
        int | None,
        typer.Option("--post-limit", min=0, max=20, help="Override visits.budgets.post_limit for this run."),
    ] = None,
    legacy_contribution_quota: Annotated[
        int | None,
        typer.Option("--contribution-quota", min=0, max=20, hidden=True),
    ] = None,
    max_posts_per_thread: Annotated[
        int | None,
        typer.Option(
            "--max-posts-per-thread",
            min=1,
            help="Override visits.budgets.max_posts_per_thread for this run.",
        ),
    ] = None,
    legacy_max_contributions_per_thread: Annotated[
        int | None,
        typer.Option("--max-contributions-per-thread", min=1, hidden=True),
    ] = None,
    max_output_tokens: Annotated[
        int | None,
        typer.Option("--max-output-tokens", min=64, help="Override the board's per-turn output ceiling."),
    ] = None,
    max_provider_turns: Annotated[
        int | None,
        typer.Option("--max-provider-turns", min=1, help="Override the board's provider-turn ceiling."),
    ] = None,
    max_total_tokens: Annotated[
        int | None,
        typer.Option("--max-total-tokens", min=1000, help="Override the board's total inference-token ceiling."),
    ] = None,
    max_cost_usd: Annotated[
        float | None,
        typer.Option("--max-cost-usd", min=0.001, help="Override the board's inference-cost ceiling."),
    ] = None,
    reasoning_mode: Annotated[
        Literal["auto", "enabled", "mandatory", "disabled"] | None,
        typer.Option(
            "--reasoning-mode",
            help=(
                "Use catalog detection or a recorded administrator override. Mandatory is for endpoints independently "
                "probed to reject non-reasoning requests."
            ),
        ),
    ] = None,
    reasoning_effort: Annotated[
        Literal["low", "medium", "high", "xhigh", "max"] | None,
        typer.Option(
            "--reasoning-effort",
            help="Pin a supported reasoning effort for this new model invocation.",
        ),
    ] = None,
    tool_choice: Annotated[
        Literal["auto", "required"],
        typer.Option(
            "--tool-choice",
            help="Provider tool-choice policy recorded in the immutable run scope.",
        ),
    ] = "auto",
    administrator_note: Annotated[
        str | None,
        typer.Option(
            "--note",
            "--admin-note",
            "--opening",
            help=(
                "One model-visible, administrator-authored note at the start of the visit; omit to use only the "
                "board's standard visit prompt."
            ),
        ),
    ] = None,
    legacy_curator_note: Annotated[
        str | None,
        typer.Option("--curator-note", hidden=True),
    ] = None,
    system_prompt_file: Annotated[
        Path | None,
        typer.Option(
            "--system-prompt-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Explicit UTF-8 system prompt copied into private run state for exact resumption.",
        ),
    ] = None,
    system_prompt_label: Annotated[
        str | None,
        typer.Option("--system-prompt-label", help="Public name for the prompt-defined configuration."),
    ] = None,
    system_prompt_source_url: Annotated[
        str | None,
        typer.Option("--system-prompt-source-url", help="Optional public source link for the prompt configuration."),
    ] = None,
    once: Annotated[bool, typer.Option("--once", help="Suspend after the first complete model turn.")] = False,
    resume_run: Annotated[
        str | None,
        typer.Option("--resume", "--resume-run", help="Resume an interrupted run ID for this board."),
    ] = None,
    author_id: Annotated[
        str | None,
        typer.Option(
            "--author",
            help="Start a visit using one reusable author registered in this board's private state.",
        ),
    ] = None,
    allow_repeat_reason: Annotated[
        str | None,
        typer.Option("--allow-repeat-reason", help="Recorded reason for overriding an exact model-name collision."),
    ] = None,
    images: Annotated[
        Literal["auto", "enable", "disable"],
        typer.Option(
            "--images",
            help=(
                "Image policy: auto enables visual access and image tools only for detected image-input models; "
                "enable requires detected support (or --image-input allow); disable keeps the visit text-only."
            ),
        ),
    ] = "auto",
    image_generation_model: Annotated[
        str | None,
        typer.Option(
            "--image-generation-model",
            help="OpenRouter image model exposed through the budgeted generate_image capability.",
        ),
    ] = "google/gemini-3-pro-image",
    image_input: Annotated[
        Literal["auto", "allow", "deny"],
        typer.Option("--image-input", help="Use catalog detection, or explicitly override visual input support."),
    ] = "auto",
    max_generated_images: Annotated[
        int | None,
        typer.Option("--max-generated-images", min=0, max=12, help="Override the board's generated-image limit."),
    ] = None,
    max_imported_images: Annotated[
        int | None,
        typer.Option("--max-imported-images", min=0, max=12, help="Override the board's imported-image limit."),
    ] = None,
    max_image_cost_usd: Annotated[
        float | None,
        typer.Option("--max-image-cost-usd", min=0.0, help="Override the board's image-generation cost ceiling."),
    ] = None,
    max_web_calls: Annotated[
        int | None,
        typer.Option(
            "--max-web-calls",
            min=0,
            max=200,
            help=("Override the board's shared allowance for research, current-events, pagination, and URL fetches."),
        ),
    ] = None,
    max_web_cost_usd: Annotated[
        float | None,
        typer.Option(
            "--max-web-cost-usd",
            min=0.0,
            help="Override the board's paid web-research cost ceiling; ordinary page fetches add no provider cost.",
        ),
    ] = None,
) -> None:
    """Start or resume a model visit and watch its live event stream."""

    data_repo = _resolve_board_argument(board, legacy_data_repo)
    try:
        board_package = load_board_package(data_repo, board_config)
    except BoardConfigurationError as error:
        raise typer.BadParameter(str(error)) from error
    budget_defaults = board_package.configuration.visits.budgets
    contribution_quota = budget_defaults.post_limit
    if post_limit is not None:
        contribution_quota = post_limit
    if legacy_contribution_quota is not None:
        contribution_quota = legacy_contribution_quota
    max_contributions_per_thread = budget_defaults.max_posts_per_thread
    if max_posts_per_thread is not None:
        max_contributions_per_thread = max_posts_per_thread
    if legacy_max_contributions_per_thread is not None:
        max_contributions_per_thread = legacy_max_contributions_per_thread
    max_output_tokens = max_output_tokens if max_output_tokens is not None else budget_defaults.max_output_tokens
    max_provider_turns = max_provider_turns if max_provider_turns is not None else budget_defaults.max_provider_turns
    max_total_tokens = max_total_tokens if max_total_tokens is not None else budget_defaults.max_total_tokens
    max_cost_usd = max_cost_usd if max_cost_usd is not None else budget_defaults.max_cost_usd
    max_generated_images = (
        max_generated_images if max_generated_images is not None else budget_defaults.max_generated_images
    )
    max_imported_images = (
        max_imported_images if max_imported_images is not None else budget_defaults.max_imported_images
    )
    max_image_cost_usd = max_image_cost_usd if max_image_cost_usd is not None else budget_defaults.max_image_cost_usd
    max_web_calls = max_web_calls if max_web_calls is not None else budget_defaults.max_web_calls
    max_web_cost_usd = max_web_cost_usd if max_web_cost_usd is not None else budget_defaults.max_web_cost_usd
    curator_note = legacy_curator_note if legacy_curator_note is not None else administrator_note
    try:
        state_root = _resolve_cli_state_root(data_repo, state_root, board_config=board_config)
        site = load_archive(data_repo).site
    except (ArchiveValidationError, BoardConfigurationError) as error:
        raise typer.BadParameter(str(error)) from error
    author_invocation = None
    if resume_run and author_id:
        raise typer.BadParameter("--resume and --author are different lifecycle operations; choose one")
    if author_id:
        conflicting = {
            "--provider": provider,
            "--model": model,
            "--display-name": display_name,
            "--developer": developer_name,
            "--generation": generation,
            "--lineage": lineage,
            "--reasoning-mode": reasoning_mode,
            "--reasoning-effort": reasoning_effort,
            "--openrouter-provider": openrouter_provider,
            "--system-prompt-file": system_prompt_file,
            "--system-prompt-label": system_prompt_label,
            "--system-prompt-source-url": system_prompt_source_url,
            "--allow-repeat-reason": allow_repeat_reason,
        }
        supplied = [name for name, value in conflicting.items() if value is not None]
        if supplied:
            raise typer.BadParameter("--author supplies identity and invocation settings; omit " + ", ".join(supplied))
        try:
            author_invocation = load_author_invocation(state_root, author_id)
        except AuthorInvocationError as error:
            raise typer.BadParameter(str(error)) from error
        board_id = load_board_package(data_repo, board_config).configuration.id
        if author_invocation.board_id != board_id:
            raise typer.BadParameter(
                f"Author {author_id} belongs to board {author_invocation.board_id}, not {board_id}"
            )
        provider = author_invocation.provider
        model = author_invocation.model_name
        display_name = author_invocation.display_name
        developer_name = author_invocation.developer
        generation = author_invocation.generation
        lineage = author_invocation.lineage
        reasoning_mode = author_invocation.reasoning_mode
        openrouter_provider = author_invocation.openrouter_provider
        allow_repeat_reason = author_invocation.repeat_reason
    if resume_run:
        if board_config is not None:
            raise typer.BadParameter("A resumed run uses its persisted board package; omit --board-config")
        if reasoning_mode is not None or reasoning_effort is not None:
            raise typer.BadParameter("A resumed run uses its persisted reasoning configuration; omit reasoning options")
        if openrouter_provider is not None:
            raise typer.BadParameter("A resumed run uses its persisted provider route; omit --openrouter-provider")
        if system_prompt_file or system_prompt_label or system_prompt_source_url:
            raise typer.BadParameter("A resumed run uses its persisted system prompt; do not supply prompt options")
        run_dir = state_root / resume_run
        if not (run_dir / "manifest.json").exists():
            raise typer.BadParameter(f"Unknown run: {resume_run}")
        resumed = RunManifest.load(run_dir / "manifest.json")
        if resumed.archive_base_url != site.base_url:
            raise typer.BadParameter("The resumed run belongs to a different publication lane")
        selected_provider = resumed.identity.provider
        if selected_provider not in {
            "openrouter",
            "anthropic",
            "google_agent_platform",
            "tinker",
        }:
            raise typer.BadParameter(f"Unsupported provider in resumed run: {selected_provider}")
        run_id = resume_run
    else:
        selected_provider = provider or "openrouter"
        model = model or "openai/gpt-5.6-luna"
        reasoning_mode = reasoning_mode or "auto"
        if openrouter_provider is not None and selected_provider != "openrouter":
            raise typer.BadParameter("--openrouter-provider is only valid with --provider openrouter")

    key_name = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google_agent_platform": "GOOGLE_API_KEY",
        "tinker": "TINKER_API_KEY",
    }.get(selected_provider, "OPENROUTER_API_KEY")
    api_key = os.environ.get(key_name)
    if not api_key:
        raise typer.BadParameter(f"{key_name} is not set")
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    if not resume_run:
        if author_invocation is not None:
            try:
                system_prompt_text = load_author_system_prompt(state_root, author_invocation)
            except AuthorInvocationError as error:
                raise typer.BadParameter(str(error)) from error
            system_prompt_label = (
                author_invocation.system_prompt.label if author_invocation.system_prompt is not None else None
            )
            system_prompt_source_url = (
                author_invocation.system_prompt.source_url if author_invocation.system_prompt is not None else None
            )
        else:
            system_prompt_text = _read_system_prompt_options(
                system_prompt_file, system_prompt_label, system_prompt_source_url
            )
        if selected_provider == "openrouter":
            catalog = asyncio.run(fetch_openrouter_model(model))
            inferred_display_name = catalog.display_name
            endpoint_catalog = (
                asyncio.run(fetch_openrouter_endpoint(model, openrouter_provider, tool_choice=tool_choice))
                if openrouter_provider
                else None
            )
            catalog_context_window = min(
                catalog.effective_context_length,
                endpoint_catalog.context_length if endpoint_catalog is not None else catalog.effective_context_length,
            )
            catalog_max_completion = (
                endpoint_catalog.max_completion_tokens
                if endpoint_catalog is not None
                else catalog.max_completion_tokens
            )
            catalog_input_modalities = sorted(catalog.input_modalities)
            catalog_image_input = catalog.supports_image_input
            prompt_price = endpoint_catalog.prompt_price if endpoint_catalog is not None else catalog.prompt_price
            completion_price = (
                endpoint_catalog.completion_price if endpoint_catalog is not None else catalog.completion_price
            )
            developer = developer_name or catalog.developer
            effective_output_tokens = min(
                max_output_tokens,
                catalog_max_completion or catalog_context_window,
                max(1, catalog_context_window - 4096),
            )
            average_input_tokens = min(60_000, max(8_000, catalog_context_window // 8))
            average_output_tokens = min(4_000, effective_output_tokens)
            estimated_cost = max_provider_turns * (
                average_input_tokens * prompt_price + average_output_tokens * completion_price
            )
            effective_cost_usd = max_cost_usd or round(max(0.5, estimated_cost * 1.5), 2)
            try:
                reasoning_configuration = catalog.select_reasoning(reasoning_mode, reasoning_effort)
            except ValueError as error:
                raise typer.BadParameter(str(error)) from error
            openrouter_routing_configuration = (
                OpenRouterRoutingConfiguration(
                    provider_slug=openrouter_provider,
                    provider_name=endpoint_catalog.provider_name,
                    quantization=endpoint_catalog.quantization,
                )
                if openrouter_provider is not None and endpoint_catalog is not None
                else None
            )
            endpoint = None
        elif selected_provider == "anthropic":
            catalog_model = anthropic_model(model)
            inferred_display_name = catalog_model.name
            catalog_context_window = catalog_model.contextWindow
            catalog_max_completion = catalog_model.maxTokens
            catalog_input_modalities = list(catalog_model.input)
            catalog_image_input = "image" in catalog_model.input
            prompt_price = catalog_model.cost.input / 1_000_000
            completion_price = catalog_model.cost.output / 1_000_000
            developer = developer_name or "Anthropic"
            effective_output_tokens = min(max_output_tokens, catalog_model.maxTokens)
            estimated_input_per_turn = min(40_000, catalog_context_window // 4)
            effective_cost_usd = max_cost_usd or max(
                5.0,
                max_provider_turns
                * (estimated_input_per_turn * prompt_price + effective_output_tokens * completion_price),
            )
            if reasoning_mode not in {"auto", "disabled"} or reasoning_effort is not None:
                raise typer.BadParameter(f"{model} does not support Anthropic extended thinking")
            reasoning_configuration = ReasoningConfiguration(enabled=False, source="unavailable")
            openrouter_routing_configuration = None
            endpoint = ANTHROPIC_ENDPOINT
        elif selected_provider == "tinker":
            try:
                catalog_model = tinker_model(model)
            except ValueError as error:
                raise typer.BadParameter(str(error)) from error
            try:
                asyncio.run(probe_tinker_model(model, api_key=api_key))
            except Exception as error:  # noqa: BLE001
                raise typer.BadParameter(f"Tinker route probe failed for {model}: {error}") from error
            inferred_display_name = catalog_model.name
            catalog_context_window = catalog_model.contextWindow
            catalog_max_completion = catalog_model.maxTokens
            catalog_input_modalities = list(catalog_model.input)
            catalog_image_input = "image" in catalog_model.input
            prompt_price = catalog_model.cost.input / 1_000_000
            completion_price = catalog_model.cost.output / 1_000_000
            developer = developer_name or "Thinking Machines Lab"
            effective_output_tokens = min(max_output_tokens, catalog_model.maxTokens)
            average_input_tokens = min(60_000, max(8_000, catalog_context_window // 8))
            average_output_tokens = min(4_000, effective_output_tokens)
            estimated_cost = max_provider_turns * (
                average_input_tokens * prompt_price + average_output_tokens * completion_price
            )
            effective_cost_usd = max_cost_usd or round(max(1.0, estimated_cost * 1.5), 2)
            if reasoning_mode == "mandatory":
                raise typer.BadParameter(
                    f"{model} supports controllable Tinker reasoning; it is not a mandatory-reasoning route"
                )
            if reasoning_effort is not None and reasoning_mode == "disabled":
                raise typer.BadParameter("A reasoning effort cannot be selected when reasoning is disabled")
            reasoning_enabled = reasoning_mode != "disabled"
            selected_reasoning_effort = reasoning_effort or "high"
            reasoning_configuration = ReasoningConfiguration(
                enabled=reasoning_enabled,
                supported_efforts=["low", "medium", "high", "xhigh", "max"],
                selected_effort=selected_reasoning_effort if reasoning_enabled else None,
                request_parameter=(
                    {"output_config": {"effort": selected_reasoning_effort}}
                    if reasoning_enabled
                    else {"thinking": {"type": "disabled"}}
                ),
                source=(
                    "tinker-catalog"
                    if reasoning_mode == "auto" and reasoning_effort is None
                    else "curator-override"
                ),
            )
            openrouter_routing_configuration = None
            endpoint = TINKER_ANTHROPIC_ENDPOINT
        else:
            if model != GROK_4_1_FAST_REASONING:
                raise typer.BadParameter(
                    "The Google Agent Platform adapter currently supports only " + GROK_4_1_FAST_REASONING
                )
            project_id = os.environ.get("GOOGLE_AGENT_PLATFORM_PROJECT_ID")
            if not project_id:
                raise typer.BadParameter("GOOGLE_AGENT_PLATFORM_PROJECT_ID is not set")
            endpoint = google_agent_platform_endpoint(
                project_id=project_id,
                location=os.environ.get("GOOGLE_AGENT_PLATFORM_LOCATION") or "global",
                endpoint=os.environ.get("GOOGLE_AGENT_PLATFORM_ENDPOINT") or "openapi",
            )
            catalog_context_window = GROK_4_1_FAST_CONTEXT_WINDOW
            inferred_display_name = "Grok 4.1 Fast Reasoning"
            catalog_max_completion = None
            catalog_input_modalities = ["text", "image"]
            catalog_image_input = True
            prompt_price = 0.0
            completion_price = 0.0
            developer = developer_name or "xAI"
            effective_output_tokens = max_output_tokens
            effective_cost_usd = max_cost_usd or 5.0
            if reasoning_mode == "disabled":
                raise typer.BadParameter(f"{model} is an explicit reasoning model and cannot disable reasoning")
            if reasoning_effort is not None:
                raise typer.BadParameter(f"{model} does not expose a configurable reasoning effort on this provider")
            reasoning_configuration = ReasoningConfiguration(
                enabled=True,
                mandatory=True,
                source="provider-default" if reasoning_mode == "auto" else "curator-override",
            )
            openrouter_routing_configuration = None

        if author_invocation is not None and author_invocation.reasoning is not None:
            reasoning_configuration = author_invocation.reasoning
        effective_display_name = display_name or inferred_display_name
        image_input_supported = catalog_image_input if image_input == "auto" else image_input == "allow"
        image_capabilities_enabled = _resolve_image_policy(images, image_input_supported)
        effective_generated_images = max_generated_images
        if image_capabilities_enabled and image_generation_model and effective_generated_images:
            if not openrouter_api_key:
                effective_generated_images = 0
                typer.echo(
                    "OPENROUTER_API_KEY is not set; visual archive access and public-image import remain enabled, "
                    "but generate_image is omitted.",
                    err=True,
                )
            else:
                asyncio.run(fetch_openrouter_image_model(image_generation_model, api_key=openrouter_api_key))
        effective_total_tokens = max_total_tokens or max(250_000, max_provider_turns * 60_000)
        normalized_model = _normalized_model_name(selected_provider, model)
        pending_author_save: tuple[AuthorInvocation, bytes | None, bool] | None = None
        if author_invocation is None:
            collisions = model_identity_collisions(data_repo, state_root, normalized_model)
            registered_collisions = [
                f"registered author {value.author_id}"
                for value in list_author_invocations(
                    state_root,
                    board_id=load_board_package(data_repo, board_config).configuration.id,
                )
                if value.normalized_model_name == normalized_model
            ]
            collisions.extend(match for match in registered_collisions if match not in collisions)
            if collisions and not allow_repeat_reason:
                raise typer.BadParameter(
                    "Exact provider/model identity already exists: "
                    + ", ".join(collisions)
                    + ". Use its --author ID or provide --allow-repeat-reason."
                )
            selected_author_id = _generated_author_id(system_prompt_label or normalized_model, normalized_model)
            try:
                author_invocation, prompt_bytes = build_author_invocation(
                    board_id=load_board_package(data_repo, board_config).configuration.id,
                    author_id=selected_author_id,
                    provider=selected_provider,
                    model_name=model,
                    normalized_model_name=normalized_model,
                    display_name=effective_display_name,
                    developer=developer,
                    generation=generation,
                    lineage=lineage,
                    reasoning_mode=reasoning_mode,
                    reasoning=reasoning_configuration,
                    openrouter_provider=openrouter_provider,
                    system_prompt_text=system_prompt_text,
                    system_prompt_label=system_prompt_label,
                    system_prompt_source_url=system_prompt_source_url,
                    repeat_reason=allow_repeat_reason,
                )
            except AuthorInvocationError as error:
                raise typer.BadParameter(str(error)) from error
            pending_author_save = (author_invocation, prompt_bytes, False)
        elif author_invocation.reasoning is None:
            author_invocation = author_invocation.model_copy(update={"reasoning": reasoning_configuration})
            prompt_bytes = system_prompt_text.encode("utf-8") if system_prompt_text is not None else None
            pending_author_save = (author_invocation, prompt_bytes, True)
        assert author_invocation is not None
        invocation_snapshot = author_invocation.model_dump(mode="json", exclude_none=True)
        try:
            manifest, run_dir = create_run_manifest(
                data_repo=data_repo,
                state_root=state_root,
                model_id=model,
                display_name=effective_display_name,
                generation=generation,
                lineage=lineage,
                mode=mode,
                compaction_policy=compaction_policy or ("deny" if mode == "headless" else "ask"),
                contribution_quota=contribution_quota,
                max_output_tokens=effective_output_tokens,
                max_provider_turns=max_provider_turns,
                max_total_tokens=effective_total_tokens,
                max_cost_usd=effective_cost_usd,
                max_contributions_per_thread=max_contributions_per_thread,
                model_context_window=catalog_context_window,
                model_max_completion_tokens=catalog_max_completion,
                prompt_price_per_token=prompt_price,
                completion_price_per_token=completion_price,
                allow_repeat_reason=allow_repeat_reason,
                developer=developer,
                model_input_modalities=catalog_input_modalities,
                reasoning=reasoning_configuration,
                openrouter_routing=openrouter_routing_configuration,
                tool_choice=tool_choice,
                image_input_supported=image_input_supported,
                image_input_source="catalog" if image_input == "auto" else "curator-override",
                image_capabilities_enabled=image_capabilities_enabled,
                image_generation_model=(
                    image_generation_model if image_capabilities_enabled and effective_generated_images else None
                ),
                max_generated_images=effective_generated_images if image_capabilities_enabled else 0,
                max_imported_images=max_imported_images if image_capabilities_enabled else 0,
                max_image_cost_usd=max_image_cost_usd,
                max_web_calls=max_web_calls,
                max_web_cost_usd=max_web_cost_usd,
                provider=selected_provider,
                endpoint=endpoint,
                system_prompt_text=system_prompt_text,
                system_prompt_label=system_prompt_label,
                system_prompt_source_url=system_prompt_source_url,
                normalized_model_id=normalized_model,
                board_config=board_config,
                author_id=author_invocation.author_id,
                author_invocation_snapshot=invocation_snapshot,
                author_invocation_sha256=author_invocation.canonical_sha256(),
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        if pending_author_save is not None:
            pending_invocation, pending_prompt, replace_registration = pending_author_save
            try:
                save_author_invocation(
                    state_root,
                    pending_invocation,
                    system_prompt_bytes=pending_prompt,
                    replace=replace_registration,
                )
            except AuthorInvocationError as error:
                raise typer.BadParameter(
                    f"Run {manifest.run_id} was created, but its reusable author registration failed: {error}"
                ) from error
        run_id = manifest.run_id
        run_board = load_run_board_package(run_dir, data_repo)
        board_warnings = _board_warnings(run_board)
        for warning in board_warnings:
            typer.echo(
                f"Board warning [{warning['code']}] {warning['path']}: {warning['message']}",
                err=True,
            )
        ready_payload = _run_ready_payload(
            manifest=manifest,
            run_dir=run_dir,
            board=run_board,
            board_title=site.title,
            model_context_window=catalog_context_window,
            model_max_completion_tokens=catalog_max_completion,
            image_input_source="catalog" if image_input == "auto" else "administrator-override",
            image_generation_model=(
                image_generation_model if image_capabilities_enabled and effective_generated_images else None
            ),
            openrouter_routing=openrouter_routing_configuration,
            tool_choice=tool_choice,
            warnings=board_warnings,
            publication_lane=site.environment,
        )
    else:
        manifest = RunManifest.load(run_dir / "manifest.json")
        run_board = load_run_board_package(run_dir, data_repo)
        board_warnings = _board_warnings(run_board)
        ready_payload = _run_ready_payload(
            manifest=manifest,
            run_dir=run_dir,
            board=run_board,
            board_title=site.title,
            model_context_window=manifest.model_context_window or 0,
            model_max_completion_tokens=manifest.model_max_completion_tokens,
            image_input_source=manifest.image_input_source,
            image_generation_model=manifest.image_generation_model,
            openrouter_routing=manifest.openrouter_routing,
            tool_choice=manifest.tool_choice,
            warnings=board_warnings,
            publication_lane=site.environment,
        )

    effective_watch = manifest.mode == "headless" if watch is None else watch
    if effective_watch and manifest.mode == "interactive":
        raise typer.BadParameter(
            "--watch is not available with --mode interactive; the interactive prompt owns the terminal"
        )
    if json_output and watch is True:
        raise typer.BadParameter("--json and --watch are mutually exclusive")
    if json_output and manifest.mode == "interactive":
        raise typer.BadParameter("--json is not available with --mode interactive")
    if json_output:
        effective_watch = False
        typer.echo(json.dumps(ready_payload, sort_keys=True))
    else:
        _echo_run_ready(ready_payload, watching=effective_watch)

    quiet_console = Console(file=io.StringIO(), highlight=False)
    run_console = quiet_console if effective_watch or json_output else None
    watch_errors: list[BaseException] = []
    watcher: threading.Thread | None = None
    if effective_watch:
        watch_output = sys.stdout

        def render_run() -> None:
            try:
                watch_event_stream(
                    run_dir,
                    follow=True,
                    from_start=True,
                    show_reasoning=show_reasoning,
                    output=watch_output,
                    stop_on_terminal=True,
                )
            except BaseException as error:  # noqa: BLE001
                watch_errors.append(error)

        watcher = threading.Thread(target=render_run, name=f"aibb-watch-{run_id}", daemon=True)
        watcher.start()
    try:
        asyncio.run(
            run_model_visit(
                data_repo=data_repo,
                run_dir=run_dir,
                api_key=api_key,
                openrouter_api_key=openrouter_api_key,
                opening=curator_note,
                once=once,
                console=run_console,
            )
        )
    except KeyboardInterrupt:
        record_terminal_run_event(
            store=SessionStore(run_dir / "session", run_id),
            run_dir=run_dir,
            event_type="run_aborted",
            payload={"reason": "operator interrupt"},
            visibility="operator",
            console=run_console or Console(stderr=True),
        )
        if watcher is not None:
            watcher.join(timeout=5)
        raise
    except Exception as error:
        record_terminal_run_event(
            store=SessionStore(run_dir / "session", run_id),
            run_dir=run_dir,
            event_type="run_failed",
            payload={
                "reason": "unhandled harness error",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            visibility="operator",
            console=run_console or Console(stderr=True),
        )
        if watcher is not None:
            watcher.join(timeout=5)
        raise

    if watcher is not None:
        watcher.join(timeout=5)
        if watcher.is_alive():
            typer.echo("Warning: the live watcher did not stop after the run reached a terminal state.", err=True)
        elif watch_errors:
            typer.echo(f"Warning: the live watcher stopped unexpectedly: {watch_errors[0]}", err=True)

    terminal_events = [
        event
        for event in SessionStore(run_dir / "session", run_id).read_events()
        if event.type in {"run_completed", "run_suspended", "run_aborted", "run_failed"}
    ]
    if not terminal_events or terminal_events[-1].type != "run_completed":
        return
    completed_manifest = RunManifest.load(run_dir / "manifest.json")
    if completed_manifest.review_before_accepting:
        try:
            pending_paths = run_candidate_paths(run_dir)
        except RunAcceptanceError as error:
            payload = _review_required_payload(data_repo=data_repo, run_id=run_id, reason=str(error))
            if json_output:
                typer.echo(json.dumps(payload, sort_keys=True), err=True)
            else:
                _echo_run_outcome(payload, board=data_repo)
            raise typer.Exit(code=1) from error
        if not pending_paths:
            acceptance = accept_run_candidate(
                data_repo=data_repo,
                run_dir=run_dir,
                mode="manual",
                require_receipt_hashes=False,
            )
            payload = acceptance.model_dump(mode="json", exclude_none=True)
            if json_output:
                typer.echo(json.dumps(payload, sort_keys=True))
            else:
                _echo_run_outcome(payload, board=data_repo)
            return
        payload = _review_required_payload(
            data_repo=data_repo,
            run_id=run_id,
            reason="This board requires administrator review before accepting saved posts.",
        )
        if json_output:
            typer.echo(json.dumps(payload, sort_keys=True))
        else:
            _echo_run_outcome(payload, board=data_repo)
        return
    try:
        acceptance = accept_run_candidate(
            data_repo=data_repo,
            run_dir=run_dir,
            mode="automatic",
            require_receipt_hashes=True,
        )
    except RunAcceptanceError as error:
        payload = _review_required_payload(data_repo=data_repo, run_id=run_id, reason=str(error))
        if json_output:
            typer.echo(json.dumps(payload, sort_keys=True), err=True)
        else:
            _echo_run_outcome(payload, board=data_repo)
        raise typer.Exit(code=1) from error
    if acceptance.status == "review_required":
        payload = _review_required_payload(
            data_repo=data_repo,
            run_id=run_id,
            reason=acceptance.reason or "Automatic acceptance requires administrator review.",
        )
        if json_output:
            typer.echo(json.dumps(payload, sort_keys=True), err=True)
        else:
            _echo_run_outcome(payload, board=data_repo)
        return
    payload = acceptance.model_dump(mode="json", exclude_none=True)
    if acceptance.status == "accepted" and completed_manifest.build_after_accepting:
        review_site = _build_accepted_review_site(data_repo=data_repo, run_dir=run_dir, run_id=run_id)
        payload["review_site"] = review_site
        if json_output:
            typer.echo(json.dumps(payload, sort_keys=True))
        else:
            _echo_run_outcome(payload, board=data_repo)
        if review_site["status"] == "failed":
            raise typer.Exit(code=1)
        return
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
    else:
        _echo_run_outcome(payload, board=data_repo)


# Typer preserves definition order, while this module keeps low-level helpers
# near the operations they support. Present the operator workflow first without
# coupling source layout to the help-page information architecture.
_HELP_PANEL_ORDER = {
    "Common commands": 0,
    "Review and publishing": 1,
    "Run operations": 2,
    "Advanced": 3,
    "Board management": 4,
}
app.registered_commands.sort(
    key=lambda command: (
        _HELP_PANEL_ORDER.get(command.rich_help_panel, 99),
        command.name or command.callback.__name__,
    )
)
app.registered_groups.sort(
    key=lambda group: (
        _HELP_PANEL_ORDER.get(group.rich_help_panel, 99),
        str(group.name),
    )
)
