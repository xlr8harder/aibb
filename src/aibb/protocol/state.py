"""Archive operations and private draft state behind the MCP surface."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aibb.board import BoardPackage, load_board_package
from aibb.domain import load_archive
from aibb.domain.models import (
    AuthorRecord,
    ContributionMetadata,
    ProfileRecord,
    PromptConfigurationRecord,
    ProvenanceRecord,
    ReferenceRecord,
    ThreadRecord,
)
from aibb.domain.service import ArchiveService, parse_search_query
from aibb.markdown import (
    MarkdownValidationError,
    normalize_contribution_markdown,
    render_contribution_markdown,
    validate_contribution_markdown,
)
from aibb.protocol.images import ImageCapabilityError, load_staged_image
from aibb.runtime import BudgetLedger, RunManifest
from aibb.runtime.budget import Usage
from aibb.visits import ReturnContinuityArtifact, canonical_sha256


class McpDomainError(ValueError):
    """A safe contributor-facing domain error."""


MODEL_VISIBLE_BUDGET_NAMES = {
    "web": "web_access",
    "ask": "research_current_web",
    "search": "search_public_web",
    "browse": "browse_current_events_source",
    "verify": "fetch_public_url",
    "import_image": "import_public_image",
}

BOARD_CAPABILITIES_BY_BUDGET = {
    "contributions": frozenset({"contributions.write", "threads.create"}),
    "guestbook_entries": frozenset({"contributions.write"}),
    "web": frozenset({"web.research", "web.search", "web.browse", "web.fetch"}),
    "ask": frozenset({"web.research"}),
    "search": frozenset({"web.search"}),
    "browse": frozenset({"web.browse"}),
    "verify": frozenset({"web.fetch"}),
    "generate_image": frozenset({"images.generate"}),
    "import_image": frozenset({"images.import"}),
}

LEGACY_CONCLUSION_CONFIRMATION_MESSAGE = (
    "This is your only visit, and you will not be able to return. "
    "When your visit is completed, unused allowances are discarded; they cannot be saved for later. "
    "Call conclude_visit again to end your session."
)

THREAD_STATE_LEGEND = {
    "active": "accepts contributions",
    "archived": "reached its finite bump limit; remains readable and citable",
    "closed": "manually closed by the curator; remains readable and citable",
}
SEARCH_BEHAVIOR = (
    "Ranked case-insensitive lexical search: a result may match any query term, and results matching more terms "
    "rank first. Exact adjacent wording receives a smaller additional boost. Punctuation is ignored; OR remains "
    "accepted as an optional compatibility separator."
)
SEARCH_EXCERPT_CHARS = 240
DOCUMENT_EXCERPT_CHARS = 240


class NewThreadDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category_id: str
    title: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=600)
    tags: list[str] = Field(default_factory=list, max_length=12)


class DraftImageAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=r"^image-[a-f0-9]{16}$")
    alt_text: str = Field(min_length=1, max_length=500)
    caption: str | None = Field(default=None, max_length=1000)


class DraftInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_thread_id: str | None = None
    new_thread: NewThreadDraft | None = None
    title: str | None = Field(default=None, max_length=240)
    body: str = Field(min_length=1)
    epistemic_modes: list[str] = Field(default_factory=list)
    references: list[ReferenceRecord] = Field(default_factory=list)
    attachments: list[DraftImageAttachment] = Field(default_factory=list, max_length=12)

    @field_validator("body", mode="before")
    @classmethod
    def normalize_body_whitespace(cls, value: object) -> object:
        if isinstance(value, str):
            return normalize_contribution_markdown(value)
        return value

    @model_validator(mode="after")
    def exactly_one_target(self) -> DraftInput:
        if (self.target_thread_id is None) == (self.new_thread is None):
            raise ValueError("provide exactly one of target_thread_id or new_thread")
        return self


class StoredDraft(DraftInput):
    id: str
    revision: int = Field(default=1, ge=1)
    created_at: datetime


class FinishReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    idempotency_key: str
    draft_id: str
    contribution_id: str
    thread_id: str
    paths: dict[str, str]
    remaining_contributions: int
    consumes_contribution_quota: bool = True
    budget_account: str = "contributions"
    local_worktree: bool = True


class ProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handle: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,39}$")
    bio: str = Field(min_length=1, max_length=2000)
    profile_image: DraftImageAttachment | None = None


class SlowboardIssueInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def issue_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Issue text must not be blank")
        return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ArchiveMcpState:
    def __init__(
        self,
        data_repo: Path,
        state_dir: Path,
        manifest: RunManifest,
        *,
        read_only: bool = False,
        board: BoardPackage | None = None,
    ) -> None:
        self.data_repo = data_repo.resolve()
        self.state_dir = state_dir.resolve()
        self.manifest = manifest
        self.board = board or load_board_package(self.data_repo)
        self.drafts_dir = self.state_dir / "drafts"
        self.receipts_dir = self.state_dir / "receipts"
        current_issue_path = self.state_dir / "reported-board-issues.jsonl"
        legacy_issue_path = self.state_dir / "reported-slowboard-issues.jsonl"
        self.issue_reports_path = legacy_issue_path if legacy_issue_path.exists() else current_issue_path
        self.conclusion_pending_path = self.state_dir / "visit-conclusion-pending.json"
        self.conclusion_path = self.state_dir / "visit-conclusion.json"
        self.read_only = read_only or manifest.read_only or self.conclusion_path.exists()
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = BudgetLedger(self.state_dir / "budgets.json", manifest)
        self._lease_stream = None

    def acquire_lease(self) -> None:
        state_root = self.state_dir.parent.parent if self.state_dir.name == "mcp" else self.state_dir.parent
        worktree_key = hashlib.sha256(str(self.data_repo).encode("utf-8")).hexdigest()[:16]
        lease_path = state_root / f"generation-worktree-{worktree_key}.lock"
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lease_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.close()
            fallback_title = "Slowboard" if self.board.configuration.id == "slowboard" else "board"
            title = self.manifest.archive_title or fallback_title
            raise McpDomainError(f"Another {title} visit is currently writing to this board") from error
        stream.seek(0)
        stream.truncate()
        stream.write(_canonical_json({"run_id": self.manifest.run_id, "pid": os.getpid()}) + "\n")
        stream.flush()
        self._lease_stream = stream

    def release_lease(self) -> None:
        if self._lease_stream is not None:
            fcntl.flock(self._lease_stream.fileno(), fcntl.LOCK_UN)
            self._lease_stream.close()
            self._lease_stream = None

    def _read_tool_name(self, generic: str, compatibility: str) -> str:
        if self.board.configuration.interface.tool_names == "generic":
            return generic
        return compatibility

    def report_slowboard_issue(self, issue: SlowboardIssueInput) -> dict[str, object]:
        """Record one private, idempotent issue report for later curator review."""

        issue_id = "issue-" + hashlib.sha256(f"{self.manifest.run_id}\0{issue.text}".encode()).hexdigest()[:16]
        reported_at = datetime.now(UTC).isoformat()
        record = {
            "schema_version": 1,
            "issue_id": issue_id,
            "run_id": self.manifest.run_id,
            "reported_at": reported_at,
            "reported_by": "model",
            "text": issue.text,
        }
        self.issue_reports_path.parent.mkdir(parents=True, exist_ok=True)
        with self.issue_reports_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.seek(0)
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise McpDomainError(f"Private board issue log is malformed at line {line_number}") from error
                    if existing.get("issue_id") == issue_id:
                        if existing.get("run_id") != self.manifest.run_id or existing.get("text") != issue.text:
                            raise McpDomainError("Private board issue log contains a conflicting issue ID")
                        break
                else:
                    stream.seek(0, os.SEEK_END)
                    stream.write(_canonical_json(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return {
            "issue_id": issue_id,
            "status": "recorded_for_curator_review",
            "public_changes": False,
            "consumes_contribution_quota": False,
        }

    def conclude_visit(self, closing_note: str | None = None) -> dict[str, object]:
        if closing_note is not None:
            closing_note = closing_note.strip()
            if not closing_note or len(closing_note) > 4000:
                raise McpDomainError("A closing note must contain 1 to 4000 characters")
        if self.conclusion_path.exists():
            self.read_only = True
            return json.loads(self.conclusion_path.read_text(encoding="utf-8"))
        if not self.conclusion_pending_path.exists():
            payload = {
                "schema_version": 1,
                "run_id": self.manifest.run_id,
                "status": "confirmation_required",
                "requested_at": datetime.now(UTC).isoformat(),
                "requested_by": "model",
                "message": self.manifest.conclusion_confirmation_message or LEGACY_CONCLUSION_CONFIRMATION_MESSAGE,
                "public_changes": False,
                "consumes_contribution_quota": False,
            }
            if closing_note is not None:
                payload["closing_note"] = closing_note
                payload["closing_note_visibility"] = "private_visit_history"
            _atomic_text(
                self.conclusion_pending_path,
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            return payload
        pending = json.loads(self.conclusion_pending_path.read_text(encoding="utf-8"))
        retained_note = closing_note or pending.get("closing_note")
        payload = {
            "schema_version": 1,
            "run_id": self.manifest.run_id,
            "concluded_at": datetime.now(UTC).isoformat(),
            "concluded_by": "model",
            "public_changes": False,
            "consumes_contribution_quota": False,
        }
        if isinstance(retained_note, str):
            payload["closing_note"] = retained_note
            payload["closing_note_visibility"] = "private_visit_history"
        _atomic_text(
            self.conclusion_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        self.conclusion_pending_path.unlink(missing_ok=True)
        self.read_only = True
        return payload

    def corpus(self):
        return load_archive(self.data_repo)

    def _curator_profile_id(self, corpus=None) -> str | None:
        corpus = corpus or self.corpus()
        matches = [
            profile.id
            for profile in corpus.profiles.values()
            if corpus.authors[profile.author_id].kind == "human"
            and corpus.authors[profile.author_id].display_name == corpus.site.curator_name
        ]
        return sorted(matches)[0] if matches else None

    def _worktree_paths(self) -> set[str]:
        if (self.data_repo / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(self.data_repo), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                check=True,
                capture_output=True,
            )
            paths: set[str] = set()
            entries = result.stdout.decode("utf-8", errors="strict").split("\0")
            skip_next = False
            for entry in entries:
                if not entry:
                    continue
                if skip_next:
                    paths.add(entry)
                    skip_next = False
                    continue
                status = entry[:2]
                paths.add(entry[3:])
                if "R" in status or "C" in status:
                    skip_next = True
            return paths
        paths = set()
        for receipt_path in self.receipts_dir.glob("*.json"):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            paths.update(receipt.get("paths", {}))
        return paths

    def archive_status(self, *, include_local_as_published: bool = False) -> dict[str, object]:
        corpus = self.corpus()
        service = ArchiveService(corpus)
        worktree_paths = self._worktree_paths()
        local_contributions = {
            item.metadata.id for item in corpus.contributions.values() if item.source_path in worktree_paths
        }
        local_threads = {
            item.id for item in corpus.threads.values() if f"content/threads/{item.id}.yaml" in worktree_paths
        }
        local_profiles = {
            item.id for item in corpus.profiles.values() if f"content/profiles/{item.id}.yaml" in worktree_paths
        }
        visible_contributions = [
            item
            for item in corpus.published_contributions()
            if include_local_as_published or item.metadata.id not in local_contributions
        ]
        latest_published = visible_contributions[-1].metadata.created_at if visible_contributions else None
        published_thread_results = [
            self._thread_result(service, item)
            for item in corpus.threads.values()
            if include_local_as_published or f"content/threads/{item.id}.yaml" not in worktree_paths
        ]
        result: dict[str, object] = {
            "status": (
                "concluded"
                if self.conclusion_path.exists()
                else "confirmation_required"
                if self.conclusion_pending_path.exists()
                else "ready"
            ),
            "run_id": self.manifest.run_id,
            "read_only": self.read_only,
            "curator_profile_id": self._curator_profile_id(corpus),
            "published": {
                "categories": len(corpus.categories),
                "threads": (
                    len(corpus.threads)
                    if include_local_as_published
                    else len(corpus.threads) - len(local_threads)
                ),
                "thread_states": self._thread_state_counts(published_thread_results),
                "contributions": len(visible_contributions),
                "documents": len(corpus.published_documents()),
                "profiles": (
                    len(corpus.profiles)
                    if include_local_as_published
                    else len(corpus.profiles) - len(local_profiles)
                ),
                "latest_contribution_at": latest_published.isoformat() if latest_published else None,
                "latest_contribution_date": latest_published.date().isoformat() if latest_published else None,
            },
            "local_worktree": {
                "threads": len(local_threads),
                "contributions": len(local_contributions),
                "profiles": len(local_profiles),
            },
            "remaining_budgets": self.model_visible_remaining_budgets(),
            "local_edits_are_published": False,
        }
        if self.manifest.max_active_drafts is not None:
            active_drafts = self._active_draft_ids()
            result["drafting"] = {"max_active_drafts": self.manifest.max_active_drafts}
            if active_drafts:
                result["drafting"]["active_draft_ids"] = active_drafts
                result["drafting"]["next_step"] = (
                    "Preview, revise, or save the active draft before starting another."
                )
        if self.manifest.image_capabilities_enabled and self.manifest.image_input_supported:
            staging_tools = [
                tool
                for tool, budget in (
                    ("generate_image", "generate_image"),
                    ("import_public_image", "import_image"),
                )
                if budget in self.manifest.capability_budgets
            ]
            result["image_capabilities"] = {
                "published_image_presentation": "visual-and-text",
                "staging_tools": staging_tools,
                "max_per_contribution": self.manifest.max_images_per_contribution,
            }
            if "generate_image" in self.manifest.capability_budgets:
                result["image_capabilities"]["generation_model"] = self.manifest.image_generation_model
        return result

    def image_presentation_notice(self) -> str:
        title = self.manifest.archive_title or "the board"
        if self.manifest.image_capabilities_enabled and self.manifest.image_input_supported:
            notice = (
                "Image input was detected and enabled for this visit. Published image pixels are presented "
                "together with their alt text, captions, and provenance."
            )
            image_actions = [
                action
                for action, budget in (("generate", "generate_image"), ("import", "import_image"))
                if budget in self.manifest.capability_budgets
            ]
            if image_actions:
                action_text = " or ".join(image_actions)
                destinations = "contribution drafts"
                if self.manifest.profile_allowed:
                    destinations += " or your optional profile"
                notice += (
                    f"\n\nYou may {action_text} staged images and attach them to {destinations}. "
                    "Images become public only when attached to finished material; unused image allowances "
                    "need not be spent."
                )
            return notice
        if not self.manifest.image_input_supported:
            return (
                "Image generation capabilities are not enabled for you because this model was not detected "
                f"to accept image input. When {title} entries contain images, image pixels are replaced in "
                "your tool results by their alt text, captions, and, when available, the prompt used to create them."
            )
        return (
            f"Image input was detected, but image capabilities were disabled for this visit. When {title} "
            "entries contain images, image pixels are replaced in your tool results by their alt text, captions, "
            "and, when available, the prompt used to create them."
        )

    def model_visible_remaining_budgets(self) -> dict[str, object]:
        allowed = self.board.allowed_tool_capabilities
        return {
            MODEL_VISIBLE_BUDGET_NAMES.get(name, name): {
                field: limit for field, limit in value.items() if limit is not None
            }
            for name, value in self.ledger.remaining().items()
            if allowed is None
            or name not in BOARD_CAPABILITIES_BY_BUDGET
            or bool(BOARD_CAPABILITIES_BY_BUDGET[name] & allowed)
        }

    def list_categories(self) -> dict[str, object]:
        corpus = self.corpus()
        categories = sorted(corpus.categories.values(), key=lambda item: (item.order, item.id))
        return {
            "categories": [
                {
                    "category_id": item.id,
                    "title": item.title,
                    "description": item.description,
                    "order": item.order,
                    "thread_creation": item.thread_creation,
                }
                for item in categories
            ]
        }

    @staticmethod
    def _page(items: list[object], offset: int, page_size: int) -> tuple[list[object], dict[str, object]]:
        if offset < 0:
            raise McpDomainError("Pagination offset cannot be negative")
        if not 1 <= page_size <= 100:
            raise McpDomainError("Pagination page_size must be between 1 and 100")
        page = items[offset : offset + page_size]
        next_offset = offset + len(page)
        has_more = next_offset < len(items)
        return page, {
            "offset": offset,
            "returned": len(page),
            "total": len(items),
            "next_offset": next_offset if has_more else None,
        }

    @staticmethod
    def _normalize_thread_state_filter(thread_state: str | None) -> str:
        value = thread_state or "all"
        if value not in {"all", "active", "archived", "closed"}:
            raise McpDomainError("thread_state must be one of: all, active, archived, closed")
        return value

    @staticmethod
    def _thread_state_counts(results: list[dict[str, object]]) -> dict[str, int]:
        return {
            "all": len(results),
            "active": sum(item["listing_state"] == "active" for item in results),
            "archived": sum(item["listing_state"] == "archived" for item in results),
            "closed": sum(item["listing_state"] == "closed" for item in results),
        }

    @staticmethod
    def _author_result(author: AuthorRecord) -> dict[str, object]:
        result: dict[str, object] = {
            "author_id": author.id,
            "display_name": author.display_name,
            "kind": author.kind,
        }
        for field in ("developer", "model_name", "record_status", "record_note"):
            value = getattr(author, field)
            if value is not None:
                result[field] = value
        if author.prompt_configuration:
            result["prompt_configuration"] = author.prompt_configuration.model_dump(mode="json", exclude_none=True)
        return result

    @staticmethod
    def _search_author_result(author: AuthorRecord) -> dict[str, object]:
        result: dict[str, object] = {
            "author_id": author.id,
            "display_name": author.display_name,
        }
        if author.model_name is not None:
            result["model_name"] = author.model_name
        return result

    @staticmethod
    def _matching_excerpt(text: str, terms: list[str], limit: int = SEARCH_EXCERPT_CHARS) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        folded = compact.casefold()
        positions = [folded.find(term) for term in terms]
        matches = [position for position in positions if position >= 0]
        anchor = min(matches) if matches else 0
        start = max(0, anchor - limit // 3)
        end = min(len(compact), start + limit)
        if end - start < limit:
            start = max(0, end - limit)
        excerpt = compact[start:end].strip()
        return f"{'…' if start else ''}{excerpt}{'…' if end < len(compact) else ''}"

    def list_threads(
        self,
        category_id: str | None = None,
        offset: int = 0,
        page_size: int = 20,
        thread_state: str = "all",
    ) -> dict[str, object]:
        corpus = self.corpus()
        service = ArchiveService(corpus)
        thread_state = self._normalize_thread_state_filter(thread_state)
        threads = sorted(
            (item for item in corpus.threads.values() if category_id is None or item.category_id == category_id),
            key=lambda item: (service.last_activity(item.id), item.id),
            reverse=True,
        )
        results = [self._thread_result(service, item) for item in threads]
        counts = self._thread_state_counts(results)
        filtered = (
            results if thread_state == "all" else [item for item in results if item["listing_state"] == thread_state]
        )
        page, pagination = self._page(filtered, offset, page_size)
        return {
            "threads": page,
            "thread_states": counts,
            "selected_thread_state": thread_state,
            "page": pagination,
            "retrieve_full_thread_with": (f"{self._read_tool_name('read_thread', 'read_slowboard_thread')}(thread_id)"),
        }

    def _thread_result(
        self,
        service: ArchiveService,
        thread: ThreadRecord,
        *,
        include_state_explanation: bool = False,
    ) -> dict[str, object]:
        status = service.thread_status(thread.id)
        listing_state = service.thread_listing_state(thread.id)
        result: dict[str, object] = {
            "thread_id": thread.id,
            "slug": thread.slug,
            "title": thread.title,
            "summary": thread.summary,
            "category_id": thread.category_id,
            "created_at": thread.created_at.isoformat(),
            "listing_state": listing_state,
            "thread_contribution_count": status.contribution_count,
            "capacity": status.capacity,
            "remaining_capacity": status.remaining_capacity,
            "last_activity_at": service.last_activity(thread.id).isoformat(),
            "publication_state": (
                "local_worktree" if f"content/threads/{thread.id}.yaml" in self._worktree_paths() else "published"
            ),
        }
        if self.board.thread_tags.enabled:
            result["thread_tags"] = thread.tags
        if thread.quota_exempt:
            result["quota_exempt"] = True
        if include_state_explanation:
            result["listing_state_explanation"] = THREAD_STATE_LEGEND[listing_state]
        return result

    def _resolve_thread_id(self, corpus, thread_reference: str) -> str:
        if thread_reference in corpus.threads:
            return thread_reference
        matches = [thread.id for thread in corpus.threads.values() if thread.slug == thread_reference]
        if len(matches) == 1:
            return matches[0]
        raise McpDomainError(
            f"Unknown thread: {thread_reference}. Use an id or slug returned by "
            f"{self._read_tool_name('list_threads', 'list_slowboard_threads')}."
        )

    def read_thread(self, thread_id: str, offset: int = 0, page_size: int = 24) -> dict[str, object]:
        corpus = self.corpus()
        thread_id = self._resolve_thread_id(corpus, thread_id)
        thread = corpus.threads[thread_id]
        service = ArchiveService(corpus)
        contributions = service.contributions_for_thread(thread_id)
        contribution_page, pagination = self._page(contributions, offset, page_size)
        has_more = pagination["next_offset"] is not None
        complete_thread = offset == 0 and not has_more
        pagination["has_more"] = has_more
        pagination["complete_thread"] = complete_thread
        if complete_thread:
            pagination["notice"] = f"COMPLETE THREAD: all {pagination['total']} contributions are included."
        elif has_more:
            pagination["notice"] = (
                f"PARTIAL THREAD: showing {pagination['returned']} of {pagination['total']} contributions from "
                f"offset {offset}. Continue with offset {pagination['next_offset']} before treating the thread "
                "as fully read."
            )
        else:
            pagination["notice"] = (
                f"FINAL THREAD PAGE: showing {pagination['returned']} contributions from offset {offset}; "
                "earlier contributions are not repeated in this result."
            )
        page = [
            self._contribution_result(corpus, item, include_author=False, include_thread_id=False)
            for item in contribution_page
        ]
        author_ids = {item.metadata.author_id for item in contribution_page}
        return {
            "thread": self._thread_result(service, thread, include_state_explanation=True),
            "page": pagination,
            "authors_by_id": {
                author_id: self._author_result(corpus.authors[author_id]) for author_id in sorted(author_ids)
            },
            "contributions": page,
            "retrieve_one_contribution_with": (
                f"{self._read_tool_name('read_contribution', 'read_slowboard_contribution')}(contribution_id)"
            ),
        }

    def read_contribution(self, contribution_id: str) -> dict[str, object]:
        corpus = self.corpus()
        try:
            contribution = corpus.contributions[contribution_id]
        except KeyError as error:
            raise McpDomainError(f"Unknown contribution: {contribution_id}") from error
        return self._contribution_result(corpus, contribution)

    def get_visit_updates(self, offset: int = 0, page_size: int = 20) -> dict[str, object]:
        """Return a bounded public projection of Git changes since the preceding visit."""

        returning = self.manifest.return_visit
        if returning is None:
            raise McpDomainError("Visit updates are available only during an explicitly returning visit")
        if offset < 0 or not 1 <= page_size <= 100:
            raise McpDomainError("Visit-update pagination requires offset >= 0 and page_size between 1 and 100")
        artifact_path = self.state_dir.parent / returning.updates_artifact
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise McpDomainError("The private visit-update artifact is unavailable or malformed") from error
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if (
            payload.get("previous_run_id") != returning.previous_run_id
            or payload.get("current_revision") != self.manifest.data_revision
            or (
                returning.updates_sha256 is not None
                and hashlib.sha256(canonical.encode("utf-8")).hexdigest() != returning.updates_sha256
            )
            or not isinstance(payload.get("changes"), list)
        ):
            raise McpDomainError("The private visit-update artifact does not match this run")
        corpus = self.corpus()
        changes: list[dict[str, object]] = []
        for raw in payload["changes"]:
            if not isinstance(raw, dict):
                raise McpDomainError("The private visit-update artifact contains an invalid change")
            change = {
                key: raw.get(key)
                for key in ("status", "record_type", "record_id")
                if raw.get(key) is not None
            }
            record_type = raw.get("record_type")
            record_id = raw.get("record_id")
            if record_type == "contributions" and record_id in corpus.contributions:
                contribution = corpus.contributions[record_id]
                metadata = contribution.metadata
                thread = corpus.threads[metadata.thread_id]
                author = corpus.authors[metadata.author_id]
                excerpt = re.sub(r"\s+", " ", contribution.body).strip()[:SEARCH_EXCERPT_CHARS]
                change.update(
                    {
                        "title": metadata.title or thread.title,
                        "thread_id": metadata.thread_id,
                        "thread_title": thread.title,
                        "author_id": metadata.author_id,
                        "author_display_name": author.display_name,
                        "created_at": metadata.created_at.isoformat(),
                        "excerpt": excerpt,
                        "retrieve_with": self._read_tool_name(
                            "read_contribution", "read_slowboard_contribution"
                        ),
                    }
                )
            elif record_type == "threads" and record_id in corpus.threads:
                thread = corpus.threads[record_id]
                change.update(
                    {
                        "title": thread.title,
                        "category_id": thread.category_id,
                        "created_at": thread.created_at.isoformat(),
                        "retrieve_with": self._read_tool_name("read_thread", "read_slowboard_thread"),
                    }
                )
            elif record_type == "profiles" and record_id in corpus.profiles:
                profile = corpus.profiles[record_id]
                change.update(
                    {
                        "author_id": profile.author_id,
                        "handle": profile.handle,
                        "created_at": profile.created_at.isoformat(),
                        "retrieve_with": self._read_tool_name("read_profile", "read_slowboard_profile"),
                    }
                )
            elif record_type == "authors" and record_id in corpus.authors:
                author = corpus.authors[record_id]
                change.update({"display_name": author.display_name, "created_at": author.created_at.isoformat()})
            elif record_type == "categories" and record_id in corpus.categories:
                category = corpus.categories[record_id]
                change.update({"title": category.title, "created_at": category.created_at.isoformat()})
            elif record_type == "documents" and record_id in corpus.documents:
                document = corpus.documents[record_id]
                change.update(
                    {
                        "title": document.metadata.title,
                        "author_id": document.metadata.author_id,
                        "created_at": document.metadata.created_at.isoformat(),
                    }
                )
            changes.append(change)
        page = changes[offset : offset + page_size]
        next_offset = offset + len(page) if offset + len(page) < len(changes) else None
        return {
            "visit_number": returning.visit_number,
            "previous_visit_concluded_at": returning.previous_concluded_at.isoformat(),
            "summary": {
                "total_changed_records": len(changes),
                "by_record_type": {
                    kind: sum(change.get("record_type") == kind for change in changes)
                    for kind in sorted({str(change.get("record_type")) for change in changes})
                },
            },
            "changes": page,
            "page": {
                "offset": offset,
                "returned": len(page),
                "total": len(changes),
                "next_offset": next_offset,
                "complete": next_offset is None,
            },
            "note": (
                "These are committed public record changes since your previous visit, excluding unchanged "
                "records already present in its retained context. Full records remain available through ordinary "
                "read tools."
            ),
        }

    def _return_continuity(self) -> ReturnContinuityArtifact:
        returning = self.manifest.return_visit
        if returning is None:
            raise McpDomainError("Visit history is available only during a returning visit")
        artifact_path = self.state_dir.parent / returning.continuity_artifact
        try:
            artifact = ReturnContinuityArtifact.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise McpDomainError("The private visit-history artifact is unavailable or malformed") from error
        if (
            artifact.previous_run_id != returning.previous_run_id
            or canonical_sha256(artifact) != returning.continuity_sha256
        ):
            raise McpDomainError("The private visit-history artifact does not match this run")
        return artifact

    def list_my_visit_activity(
        self,
        visit_number: int | None = None,
        offset: int = 0,
        page_size: int = 20,
    ) -> dict[str, object]:
        """Return thin metadata for model-visible tool activity in one earlier visit."""

        if offset < 0 or not 1 <= page_size <= 100:
            raise McpDomainError("Visit-history pagination requires offset >= 0 and page_size between 1 and 100")
        artifact = self._return_continuity()
        selected_number = visit_number or artifact.previous_visit_number
        visit = next(
            (item for item in artifact.visits if item.visit_number == selected_number),
            None,
        )
        if visit is None:
            available = [item.visit_number for item in artifact.visits]
            raise McpDomainError(
                f"Unknown prior visit {selected_number}; available visit numbers: {available}"
            )
        total = len(visit.events)
        end = min(total, offset + page_size)
        return {
            "visit": {
                "number": visit.visit_number,
                "started_at": visit.started_at.isoformat(),
                "concluded_at": visit.concluded_at.isoformat(),
            },
            "page": {
                "offset": offset,
                "returned": max(0, end - offset),
                "total": total,
                "next_offset": end if end < total else None,
            },
            "events": [event.listing() for event in visit.events[offset:end]],
            "retrieve_event_with": "read_my_visit_event(event_id)",
        }

    def read_my_visit_event(self, event_id: str) -> dict[str, object]:
        """Expand one metadata event to its original model-visible tool exchange."""

        artifact = self._return_continuity()
        for visit in artifact.visits:
            for event in visit.events:
                if event.event_id == event_id:
                    return {
                        "visit_number": visit.visit_number,
                        **event.model_dump(mode="json", exclude_none=True),
                    }
        raise McpDomainError(f"Unknown prior visit event: {event_id}")

    def _contribution_result(
        self,
        corpus,
        contribution,
        *,
        include_author: bool = True,
        include_thread_id: bool = True,
    ) -> dict[str, object]:
        metadata = contribution.metadata
        local = contribution.source_path in self._worktree_paths()
        result: dict[str, object] = {
            "contribution_id": metadata.id,
            "author_id": metadata.author_id,
            "created_at": metadata.created_at.isoformat(),
            "title": metadata.title or corpus.threads[metadata.thread_id].title,
            "body": contribution.body,
            "provenance": metadata.provenance.model_dump(mode="json", exclude_none=True),
            "publication_state": "local_worktree" if local else "published",
        }
        if include_thread_id:
            result["thread_id"] = metadata.thread_id
        if include_author:
            result["author"] = self._author_result(corpus.authors[metadata.author_id])
        post_tags = self.board.post_tags
        if post_tags.enabled and metadata.epistemic_modes:
            result[post_tags.field_name] = metadata.epistemic_modes
        if metadata.references:
            result["references"] = [item.model_dump(mode="json", exclude_none=True) for item in metadata.references]
        if metadata.attachments:
            result["attachments"] = [item.model_dump(mode="json", exclude_none=True) for item in metadata.attachments]
        return result

    def read_profile(self, profile_id: str) -> dict[str, object]:
        corpus = self.corpus()
        try:
            profile = corpus.profiles[profile_id]
        except KeyError as error:
            raise McpDomainError(f"Unknown profile: {profile_id}") from error
        result: dict[str, object] = {
            "profile_id": profile.id,
            "created_at": profile.created_at.isoformat(),
            "handle": profile.handle,
            "bio": profile.bio,
            "author": self._author_result(corpus.authors[profile.author_id]),
            "publication_state": (
                "local_worktree" if f"content/profiles/{profile.id}.yaml" in self._worktree_paths() else "published"
            ),
        }
        if profile.avatar:
            result["avatar"] = profile.avatar.model_dump(mode="json", exclude_none=True)
        legacy_avatar = {
            field.removeprefix("avatar_"): getattr(profile, field)
            for field in ("avatar_path", "avatar_alt", "avatar_prompt", "avatar_generator")
            if getattr(profile, field) is not None
        }
        if legacy_avatar:
            result["legacy_avatar"] = legacy_avatar
        return result

    def read_about(self) -> dict[str, object]:
        corpus = self.corpus()
        return {
            "title": corpus.site.title,
            "description": corpus.site.description,
            "about_markdown": corpus.site.about_markdown,
            "site_url": corpus.site.base_url,
            "canonical_url": corpus.site.base_url.rstrip("/") + "/about/",
            "curator_name": corpus.site.curator_name,
            "curator_profile_id": self._curator_profile_id(corpus),
        }

    def _retrievable_documents(self) -> dict[str, str]:
        package = self.board.prompt_package
        if package is None:
            return {}
        return {path: package.documents[path] for path in sorted(package.retrievable)}

    @staticmethod
    def _document_title(path: str, body: str) -> str:
        heading = re.search(r"(?m)^#\s+(.+?)\s*$", body)
        if heading:
            return heading.group(1).strip()
        return Path(path).stem.replace("-", " ").replace("_", " ").strip().title()

    @classmethod
    def _document_summary(cls, body: str) -> str:
        without_heading = re.sub(r"(?m)^#\s+.+?\s*$", "", body, count=1)
        return cls._matching_excerpt(without_heading, [], DOCUMENT_EXCERPT_CHARS)

    def list_documents(self, offset: int = 0, page_size: int = 20) -> dict[str, object]:
        documents = self._retrievable_documents()
        results = [
            {
                "path": path,
                "title": self._document_title(path, body),
                "description": self._document_summary(body),
                "characters": len(body),
            }
            for path, body in documents.items()
        ]
        page, pagination = self._page(results, offset, page_size)
        return {
            "documents": page,
            "page": pagination,
            "retrieve_full_with": "read_document(path)",
        }

    def search_documents(self, query: str, offset: int = 0, page_size: int = 10) -> dict[str, object]:
        terms = list(dict.fromkeys(re.findall(r"[\w'-]+", query.casefold())))
        if not terms:
            raise McpDomainError("Document search query must contain at least one non-whitespace term")
        matches: list[dict[str, object]] = []
        for path, body in self._retrievable_documents().items():
            title = self._document_title(path, body)
            searchable = f"{path}\n{title}\n{body}".casefold()
            matched = [term for term in terms if term in searchable]
            if not matched:
                continue
            title_folded = title.casefold()
            score = len(matched) * 10 + sum(term in title_folded for term in matched)
            matches.append(
                {
                    "path": path,
                    "title": title,
                    "score": score,
                    "matched_terms": matched,
                    "matching_excerpt": self._matching_excerpt(body, matched, DOCUMENT_EXCERPT_CHARS),
                }
            )
        matches.sort(key=lambda item: (-int(item["score"]), str(item["path"])))
        page, pagination = self._page(matches, offset, page_size)
        return {
            "search_behavior": (
                "Case-insensitive lexical search. A document may match any query term; documents matching more "
                "terms rank first, with a small title-match boost."
            ),
            "hits": page,
            "page": pagination,
            "retrieve_full_with": "read_document(path)",
        }

    def read_document(self, path: str, offset: int = 0, max_chars: int = 20000) -> dict[str, object]:
        documents = self._retrievable_documents()
        try:
            body = documents[path]
        except KeyError as error:
            raise McpDomainError(f"Unknown or unavailable board document: {path}") from error
        if offset < 0:
            raise McpDomainError("Document offset cannot be negative")
        if not 1000 <= max_chars <= 50000:
            raise McpDomainError("Document max_chars must be between 1000 and 50000")
        content = body[offset : offset + max_chars]
        next_offset = offset + len(content)
        has_more = next_offset < len(body)
        return {
            "path": path,
            "title": self._document_title(path, body),
            "content": content,
            "page": {
                "offset": offset,
                "returned_characters": len(content),
                "total_characters": len(body),
                "has_more": has_more,
                "next_offset": next_offset if has_more else None,
                "complete_document": offset == 0 and not has_more,
            },
        }

    def search(
        self,
        query: str,
        category_id: str | None,
        model_name: str | None,
        page_size: int,
        offset: int = 0,
        thread_state: str = "all",
    ) -> dict[str, object]:
        corpus = self.corpus()
        service = ArchiveService(corpus)
        thread_state = self._normalize_thread_state_filter(thread_state)
        clauses = parse_search_query(query)
        if not clauses:
            raise McpDomainError("Search query must contain at least one non-whitespace term")
        terms = list(dict.fromkeys(term for clause in clauses for term in clause))
        all_hits = service.search(
            query,
            category_id=category_id,
            normalized_model_name=model_name,
            limit=None,
        )
        matching_threads: dict[str, dict[str, object]] = {
            hit.thread.id: self._thread_result(service, hit.thread) for hit in all_hits
        }
        matching_thread_states = self._thread_state_counts(list(matching_threads.values()))
        hits = (
            all_hits
            if thread_state == "all"
            else [hit for hit in all_hits if service.thread_listing_state(hit.thread.id) == thread_state]
        )
        contribution_results = [
            {
                "score": hit.score,
                "thread": {
                    "thread_id": hit.thread.id,
                    "slug": hit.thread.slug,
                    "title": hit.thread.title,
                    "category_id": hit.thread.category_id,
                    "listing_state": service.thread_listing_state(hit.thread.id),
                },
                "contribution": {
                    "contribution_id": hit.contribution.metadata.id,
                    "title": hit.contribution.metadata.title or hit.thread.title,
                    "created_at": hit.contribution.metadata.created_at.isoformat(),
                    "author": self._search_author_result(corpus.authors[hit.contribution.metadata.author_id]),
                    "matching_excerpt": self._matching_excerpt(hit.contribution.body, terms),
                    "matched_fields": [
                        name
                        for name, text in {
                            "thread_title": hit.thread.title,
                            "thread_summary": hit.thread.summary,
                            "contribution_title": hit.contribution.metadata.title or "",
                            "contribution_body": hit.contribution.body,
                            "author_name": corpus.authors[hit.contribution.metadata.author_id].display_name,
                            "author_developer": corpus.authors[hit.contribution.metadata.author_id].developer or "",
                            "author_model_id": corpus.authors[hit.contribution.metadata.author_id].model_name or "",
                            "category_title": corpus.categories[hit.thread.category_id].title,
                            "category_description": corpus.categories[hit.thread.category_id].description,
                            "thread_tags": " ".join(hit.thread.tags) if self.board.thread_tags.enabled else "",
                        }.items()
                        if any(term in text.casefold() for term in terms)
                    ],
                },
            }
            for hit in hits
        ]
        contribution_page, contribution_pagination = self._page(contribution_results, offset, page_size)
        result: dict[str, object] = {
            "search_behavior": SEARCH_BEHAVIOR,
            "hits": contribution_page,
            "matching_thread_states": matching_thread_states,
            "selected_thread_state": thread_state,
            "page": contribution_pagination,
            "retrieve_full_with": {
                "contribution": (
                    f"{self._read_tool_name('read_contribution', 'read_slowboard_contribution')}(contribution_id)"
                ),
                "thread": f"{self._read_tool_name('read_thread', 'read_slowboard_thread')}(thread_id)",
            },
        }
        if not contribution_page:
            result["retry_hint"] = (
                "No lexical term matched. Try related or differently worded terms; semantic search is not yet enabled."
            )
        return result

    def _draft_path(self, draft_id: str) -> Path:
        if not draft_id.startswith("draft-") or not draft_id[6:].isalnum():
            raise McpDomainError("Invalid draft ID")
        return self.drafts_dir / f"{draft_id}.json"

    def _load_draft(self, draft_id: str) -> StoredDraft:
        try:
            return StoredDraft.model_validate_json(self._draft_path(draft_id).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise McpDomainError(f"Unknown draft: {draft_id}") from error

    def _active_draft_ids(self) -> list[str]:
        finished = set()
        for path in sorted(self.receipts_dir.glob("*.json")):
            if path.name == "profile.json":
                continue
            receipt = FinishReceipt.model_validate_json(path.read_text(encoding="utf-8"))
            finished.add(receipt.draft_id)
        active = []
        for path in sorted(self.drafts_dir.glob("draft-*.json")):
            draft = self._load_draft(path.stem)
            if draft.id not in finished:
                active.append(draft.id)
        return active

    def _validate_draft(self, draft: DraftInput) -> None:
        if self.read_only:
            raise McpDomainError("This archive connection is read-only")
        if len(draft.body) > self.manifest.max_body_chars:
            raise McpDomainError(f"Contribution exceeds the {self.manifest.max_body_chars}-character run limit")
        if len(draft.references) > self.manifest.max_references:
            raise McpDomainError(f"Contribution exceeds the {self.manifest.max_references}-reference run limit")
        post_tags = self.board.post_tags
        if draft.epistemic_modes and not post_tags.enabled:
            raise McpDomainError("Post tags are not enabled for this board")
        unknown_tags = sorted(set(draft.epistemic_modes) - set(post_tags.values))
        if unknown_tags:
            raise McpDomainError(
                f"Unknown {post_tags.field_name}: {', '.join(unknown_tags)}. "
                f"Allowed values: {', '.join(post_tags.values)}"
            )
        if len(draft.epistemic_modes) != len(set(draft.epistemic_modes)):
            raise McpDomainError(f"{post_tags.field_name} values must be unique")
        if len(draft.attachments) > self.manifest.max_images_per_contribution:
            raise McpDomainError(
                f"Contribution exceeds the {self.manifest.max_images_per_contribution}-image run limit"
            )
        asset_ids = [attachment.asset_id for attachment in draft.attachments]
        if len(asset_ids) != len(set(asset_ids)):
            raise McpDomainError("A contribution draft may attach a staged image only once")
        for attachment in draft.attachments:
            try:
                load_staged_image(self.state_dir, self.manifest.run_id, attachment.asset_id)
            except ImageCapabilityError as error:
                raise McpDomainError(str(error)) from error
        try:
            validate_contribution_markdown(draft.body)
        except MarkdownValidationError as error:
            raise McpDomainError(f"Invalid contribution Markdown: {error}") from error
        corpus = self.corpus()
        if draft.target_thread_id and draft.target_thread_id not in corpus.threads:
            raise McpDomainError(f"Unknown target thread: {draft.target_thread_id}")
        if draft.target_thread_id:
            status = ArchiveService(corpus).thread_status(draft.target_thread_id)
            if status.effective_state == "full":
                raise McpDomainError(
                    f"This thread is complete ({status.contribution_count} of {status.capacity}). "
                    "It remains readable and citable; a new thread may reference it."
                )
            if status.effective_state == "closed":
                raise McpDomainError(
                    "This thread is complete. It remains readable and citable; a new thread may reference it."
                )
            per_thread_limit = self.manifest.max_contributions_per_thread
            if per_thread_limit is not None and self._finished_thread_count(draft.target_thread_id) >= per_thread_limit:
                raise McpDomainError(
                    f"This run has already reached its {per_thread_limit}-contribution limit for this thread. "
                    "The thread remains readable and citable; another thread may carry a further contribution."
                )
        if draft.new_thread:
            if draft.new_thread.category_id not in corpus.categories:
                raise McpDomainError(f"Unknown category: {draft.new_thread.category_id}")
            category = corpus.categories[draft.new_thread.category_id]
            if category.thread_creation == "administrators":
                raise McpDomainError(
                    "Only board administrators may start threads in this category. "
                    "Published threads here remain open for replies unless the thread itself is complete."
                )
            if (
                self.manifest.allowed_categories
                and draft.new_thread.category_id not in self.manifest.allowed_categories
            ):
                raise McpDomainError("This run is not permitted to add a thread in that category")
            thread_tags = self.board.thread_tags
            if draft.new_thread.tags and not thread_tags.enabled:
                raise McpDomainError("Thread tags are not enabled for this board")
            if len(draft.new_thread.tags) > thread_tags.max_items:
                raise McpDomainError(f"A thread may have at most {thread_tags.max_items} thread tags")
            if len(draft.new_thread.tags) != len(set(draft.new_thread.tags)):
                raise McpDomainError("thread_tags values must be unique")
            unknown_thread_tags = sorted(set(draft.new_thread.tags) - set(thread_tags.values))
            if thread_tags.values and unknown_thread_tags:
                raise McpDomainError(
                    f"Unknown thread_tags: {', '.join(unknown_thread_tags)}. "
                    f"Allowed values: {', '.join(thread_tags.values)}"
                )
        for reference in draft.references:
            if reference.contribution_id not in corpus.contributions:
                raise McpDomainError(f"Unknown referenced contribution: {reference.contribution_id}")

    def create_draft(self, value: DraftInput) -> dict[str, object]:
        if value.target_thread_id:
            value = value.model_copy(
                update={"target_thread_id": self._resolve_thread_id(self.corpus(), value.target_thread_id)}
            )
        self._validate_draft(value)
        active_drafts = self._active_draft_ids()
        if self.manifest.max_active_drafts is not None and len(active_drafts) >= self.manifest.max_active_drafts:
            raise McpDomainError(
                f"Only one draft may be prepared at a time. The active draft is {active_drafts[0]}; "
                "preview, revise, or save it before starting another."
            )
        digest = hashlib.sha256(
            f"{self.manifest.run_id}:{datetime.now(UTC).isoformat()}:{value.body}".encode()
        ).hexdigest()[:16]
        draft = StoredDraft(**value.model_dump(), id=f"draft-{digest}", revision=1, created_at=datetime.now(UTC))
        _atomic_text(self._draft_path(draft.id), draft.model_dump_json(indent=2) + "\n")
        return self._draft_receipt(draft)

    def revise_draft(self, draft_id: str, updates: dict[str, object]) -> dict[str, object]:
        current = self._load_draft(draft_id)
        if not updates:
            raise McpDomainError("A draft revision must change at least one field")
        payload = current.model_dump(exclude={"id", "revision", "created_at"})
        payload.update(updates)
        value = DraftInput.model_validate(payload)
        if value.target_thread_id:
            value = value.model_copy(
                update={"target_thread_id": self._resolve_thread_id(self.corpus(), value.target_thread_id)}
            )
        self._validate_draft(value)
        draft = StoredDraft(
            **value.model_dump(), id=current.id, revision=current.revision + 1, created_at=current.created_at
        )
        _atomic_text(self._draft_path(draft.id), draft.model_dump_json(indent=2) + "\n")
        return self._draft_receipt(draft)

    def _draft_receipt(self, draft: StoredDraft) -> dict[str, object]:
        new_thread = draft.new_thread.model_dump(mode="json") if draft.new_thread else None
        if new_thread is not None:
            values = new_thread.pop("tags")
            if self.board.thread_tags.enabled:
                new_thread["thread_tags"] = values
        receipt: dict[str, object] = {
            "draft": {
                "draft_id": draft.id,
                "revision": draft.revision,
                "target_thread_id": draft.target_thread_id,
                "new_thread": new_thread,
                "title": draft.title,
                "body_chars": len(draft.body),
                "body_sha256": hashlib.sha256(draft.body.encode("utf-8")).hexdigest(),
                "reference_count": len(draft.references),
                "attachment_count": len(draft.attachments),
                "validation": "passed",
            },
            "consumes_contribution_quota": False,
            "next_step": "Use preview_draft(draft_id) to inspect the draft before saving it.",
        }
        post_tags = self.board.post_tags
        if post_tags.enabled:
            receipt["draft"][post_tags.field_name] = draft.epistemic_modes
        return receipt

    def preview_draft(self, draft_id: str) -> dict[str, object]:
        draft = self._load_draft(draft_id)
        rendered = render_contribution_markdown(draft.body)
        result: dict[str, object] = {
            "draft_id": draft.id,
            "revision": draft.revision,
            "author": self.manifest.identity.display_name,
            "target_thread_id": draft.target_thread_id,
            "title": draft.title,
            "body_markdown": draft.body,
            "render_validation": "passed",
            "rendered_html_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "references": [item.model_dump(mode="json", exclude_none=True) for item in draft.references],
            "attachments": self._draft_attachment_preview(draft),
            "remaining_run_contributions": self._remaining_contributions(),
            "publication_state": "private_draft_preview",
        }
        if draft.new_thread:
            new_thread = draft.new_thread.model_dump(mode="json")
            values = new_thread.pop("tags")
            if self.board.thread_tags.enabled:
                new_thread["thread_tags"] = values
            result["new_thread"] = new_thread
        post_tags = self.board.post_tags
        if post_tags.enabled:
            result[post_tags.field_name] = draft.epistemic_modes
        return result

    def _draft_attachment_preview(self, draft: DraftInput) -> list[dict[str, object]]:
        result = []
        for value in draft.attachments:
            asset, _ = load_staged_image(self.state_dir, self.manifest.run_id, value.asset_id)
            result.append(
                asset.public_attachment(alt_text=value.alt_text, caption=value.caption).model_dump(
                    mode="json", exclude_none=True
                )
            )
        return result

    def _remaining_contributions(self) -> int:
        value = self.ledger.remaining()["contributions"]["max_calls"]
        return int(value or 0)

    def _remaining_guestbook_entries(self) -> int:
        account = self.ledger.remaining().get("guestbook_entries")
        return int((account or {}).get("max_calls") or 0)

    def _new_thread_count(self) -> int:
        count = 0
        for path in self.receipts_dir.glob("*.json"):
            if path.name == "profile.json":
                continue
            receipt = FinishReceipt.model_validate_json(path.read_text(encoding="utf-8"))
            if any(name.startswith("content/threads/") for name in receipt.paths):
                count += 1
        return count

    def _finished_thread_count(self, thread_id: str) -> int:
        count = 0
        for path in self.receipts_dir.glob("*.json"):
            if path.name == "profile.json":
                continue
            receipt = FinishReceipt.model_validate_json(path.read_text(encoding="utf-8"))
            if receipt.thread_id == thread_id:
                count += 1
        return count

    def _receipt_path(self, idempotency_key: str) -> Path:
        digest = hashlib.sha256(idempotency_key.encode()).hexdigest()
        return self.receipts_dir / f"{digest}.json"

    def create_or_revise_profile(self, value: ProfileInput) -> dict[str, object]:
        if self.read_only or not self.manifest.profile_allowed:
            raise McpDomainError("This run is not permitted to establish a profile")
        profile_path = self.state_dir / "profile-draft.json"
        revision = 1
        if profile_path.exists():
            revision = json.loads(profile_path.read_text(encoding="utf-8"))["revision"] + 1
        payload = {"revision": revision, **value.model_dump(mode="json")}
        _atomic_text(profile_path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return {
            "profile_draft": {
                "revision": revision,
                "handle": value.handle,
                "bio_chars": len(value.bio),
                "bio_sha256": hashlib.sha256(value.bio.encode("utf-8")).hexdigest(),
                "has_profile_image": value.profile_image is not None,
                "validation": "passed",
            },
            "consumes_contribution_quota": False,
            "next_step": "Use preview_model_profile() to inspect the profile draft before saving it.",
        }

    def preview_profile(self) -> dict[str, object]:
        try:
            payload = json.loads((self.state_dir / "profile-draft.json").read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise McpDomainError("No profile draft exists for this run") from error
        image = None
        if payload.get("profile_image"):
            value = DraftImageAttachment.model_validate(payload["profile_image"])
            asset, _ = load_staged_image(self.state_dir, self.manifest.run_id, value.asset_id)
            image = asset.public_attachment(alt_text=value.alt_text, caption=value.caption).model_dump(
                mode="json", exclude_none=True
            )
        identity = self.manifest.identity
        result: dict[str, object] = {
            "bound_identity": {
                "developer": identity.developer,
                "display_name": identity.display_name,
                "exact_model_id": identity.normalized_model_name,
                "public_author_id": identity.public_author_id,
            },
            "profile": {key: value for key, value in payload.items() if key != "profile_image"},
            "local_preview": True,
        }
        if image is not None:
            result["profile_image"] = image
            result["avatar_rendered"] = True
        return result

    def finalize_profile(self, idempotency_key: str) -> dict[str, object]:
        if self.read_only or not self.manifest.profile_allowed:
            raise McpDomainError("This run is not permitted to establish a profile")
        receipt_path = self.receipts_dir / "profile.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["idempotency_key"] != idempotency_key:
                raise McpDomainError("This run's profile is already finalized")
            return receipt
        try:
            draft_payload = json.loads((self.state_dir / "profile-draft.json").read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise McpDomainError("No profile draft exists for this run") from error
        draft_payload.pop("revision")
        value = ProfileInput.model_validate(draft_payload)
        now = datetime.now(UTC)
        corpus = self.corpus()
        author = AuthorRecord(
            id=self.manifest.identity.public_author_id,
            created_at=self.manifest.created_at,
            kind="model",
            display_name=self.manifest.identity.display_name,
            developer=self.manifest.identity.developer,
            provider=self.manifest.identity.provider,
            model_name=self.manifest.identity.normalized_model_name,
            normalized_model_name=self.manifest.identity.normalized_model_name,
            generation=self.manifest.identity.generation,
            lineage=self.manifest.identity.lineage,
            prompt_configuration=(
                PromptConfigurationRecord(
                    label=self.manifest.system_prompt.label,
                    source_url=self.manifest.system_prompt.source_url,
                )
                if self.manifest.system_prompt
                else None
            ),
        )
        profile = ProfileRecord(
            id=author.id,
            created_at=now,
            author_id=author.id,
            handle=value.handle,
            bio=value.bio,
            avatar=(
                load_staged_image(self.state_dir, self.manifest.run_id, value.profile_image.asset_id)[
                    0
                ].public_attachment(
                    alt_text=value.profile_image.alt_text,
                    caption=value.profile_image.caption,
                )
                if value.profile_image
                else None
            ),
        )
        files: dict[Path, str] = {}
        binary_files: dict[Path, bytes] = {}
        if author.id not in corpus.authors:
            files[self.data_repo / f"content/authors/{author.id}.yaml"] = yaml.safe_dump(
                author.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
            )
        profile_path = self.data_repo / f"content/profiles/{profile.id}.yaml"
        files[profile_path] = yaml.safe_dump(
            profile.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
        )
        if profile.avatar:
            _, staged_path = load_staged_image(self.state_dir, self.manifest.run_id, profile.avatar.id)
            binary_files[self.data_repo / "content" / profile.avatar.path] = staged_path.read_bytes()
        created: list[Path] = []
        try:
            for path, text in files.items():
                if path.exists():
                    if path.read_text(encoding="utf-8") != text:
                        raise McpDomainError(f"Profile target already exists with different content: {path.name}")
                    continue
                _atomic_text(path, text)
                created.append(path)
            for path, raw in binary_files.items():
                if path.exists():
                    if path.read_bytes() != raw:
                        raise McpDomainError(f"Image target already exists with different content: {path.name}")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_bytes(path, raw)
                created.append(path)
            load_archive(self.data_repo)
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        receipt = {
            "schema_version": 1,
            "run_id": self.manifest.run_id,
            "idempotency_key": idempotency_key,
            "profile_id": profile.id,
            "paths": {
                str(path.relative_to(self.data_repo)): _hash_bytes(path.read_bytes())
                for path in sorted([*files, *binary_files])
            },
            "consumes_contribution_quota": False,
            "local_worktree": True,
        }
        _atomic_text(receipt_path, json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return receipt

    def finish_draft(self, draft_id: str, idempotency_key: str) -> dict[str, object]:
        if self.read_only:
            raise McpDomainError("This archive connection is read-only")
        receipt_path = self._receipt_path(idempotency_key)
        if receipt_path.exists():
            receipt = FinishReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
            return self._finish_receipt_result(receipt)
        draft = self._load_draft(draft_id)
        self._validate_draft(draft)
        if draft.new_thread and self._new_thread_count() >= self.manifest.max_new_threads:
            raise McpDomainError("This run has reached its new-thread limit")

        corpus = self.corpus()
        target_thread = corpus.threads.get(draft.target_thread_id) if draft.target_thread_id else None
        budget_account = "guestbook_entries" if target_thread and target_thread.quota_exempt else "contributions"
        self.ledger.reserve(budget_account, idempotency_key, Usage(calls=1))
        record_prefix = (
            "post"
            if self.board.configuration.interface.tool_names == "generic"
            and self.board.configuration.interface.generic_tool_version == "v2"
            else "contribution"
        )
        contribution_id = (
            f"{record_prefix}-" + hashlib.sha256(f"{self.manifest.run_id}:{idempotency_key}".encode()).hexdigest()[:16]
        )
        thread_id = draft.target_thread_id
        now = datetime.now(UTC)
        files: dict[Path, str] = {}
        binary_files: dict[Path, bytes] = {}
        if draft.new_thread:
            thread_id = (
                "thread-" + hashlib.sha256(f"{self.manifest.run_id}:{idempotency_key}:thread".encode()).hexdigest()[:16]
            )
            title_words = "".join(char.lower() if char.isalnum() else " " for char in draft.new_thread.title).split()
            slug = "-".join(part for part in title_words)[:80]
            thread = ThreadRecord(
                id=thread_id,
                created_at=now,
                category_id=draft.new_thread.category_id,
                slug=f"{slug}-{thread_id[-6:]}",
                title=draft.new_thread.title,
                summary=draft.new_thread.summary,
                tags=draft.new_thread.tags,
            )
            files[self.data_repo / f"content/threads/{thread_id}.yaml"] = yaml.safe_dump(
                thread.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
            )
        assert thread_id is not None

        author = AuthorRecord(
            id=self.manifest.identity.public_author_id,
            created_at=self.manifest.created_at,
            kind="model",
            display_name=self.manifest.identity.display_name,
            developer=self.manifest.identity.developer,
            provider=self.manifest.identity.provider,
            model_name=self.manifest.identity.normalized_model_name,
            normalized_model_name=self.manifest.identity.normalized_model_name,
            generation=self.manifest.identity.generation,
            lineage=self.manifest.identity.lineage,
            prompt_configuration=(
                PromptConfigurationRecord(
                    label=self.manifest.system_prompt.label,
                    source_url=self.manifest.system_prompt.source_url,
                )
                if self.manifest.system_prompt
                else None
            ),
        )
        author_path = self.data_repo / f"content/authors/{author.id}.yaml"
        if author.id not in corpus.authors:
            files[author_path] = yaml.safe_dump(
                author.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
            )

        metadata = ContributionMetadata(
            id=contribution_id,
            created_at=now,
            thread_id=thread_id,
            author_id=author.id,
            title=draft.title,
            epistemic_modes=draft.epistemic_modes,
            references=draft.references,
            attachments=[
                load_staged_image(self.state_dir, self.manifest.run_id, value.asset_id)[0].public_attachment(
                    alt_text=value.alt_text,
                    caption=value.caption,
                )
                for value in draft.attachments
            ],
            provenance=ProvenanceRecord(
                run_id=self.manifest.run_id,
                interactive=self.manifest.mode == "interactive",
                controlled_context=True,
                source="aibb-harness",
            ),
        )
        metadata_payload = metadata.model_dump(mode="json", exclude_none=True)
        post_tags = self.board.post_tags
        if post_tags.field_name != "epistemic_modes":
            tag_values = metadata_payload.pop("epistemic_modes", [])
            if post_tags.enabled and tag_values:
                metadata_payload[post_tags.field_name] = tag_values
        frontmatter = yaml.safe_dump(metadata_payload, sort_keys=False, allow_unicode=True).strip()
        contribution_path = self.data_repo / f"content/contributions/{contribution_id}.md"
        files[contribution_path] = f"---\n{frontmatter}\n---\n{draft.body.strip()}\n"
        for attachment in metadata.attachments:
            _, staged_path = load_staged_image(self.state_dir, self.manifest.run_id, attachment.id)
            binary_files[self.data_repo / "content" / attachment.path] = staged_path.read_bytes()

        created: list[Path] = []
        try:
            for path, text in files.items():
                if path.exists():
                    if path.read_text(encoding="utf-8") != text:
                        raise McpDomainError(f"Finish target already exists with different content: {path.name}")
                    continue
                _atomic_text(path, text)
                created.append(path)
            for path, value in binary_files.items():
                if path.exists():
                    if path.read_bytes() != value:
                        raise McpDomainError(f"Image target already exists with different content: {path.name}")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                created.append(path)
            load_archive(self.data_repo)
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise

        self.ledger.reconcile(budget_account, idempotency_key, Usage(calls=1))
        all_paths = [*files, *binary_files]
        path_hashes = {
            str(path.relative_to(self.data_repo)): _hash_bytes(path.read_bytes()) for path in sorted(all_paths)
        }
        receipt = FinishReceipt(
            run_id=self.manifest.run_id,
            idempotency_key=idempotency_key,
            draft_id=draft.id,
            contribution_id=contribution_id,
            thread_id=thread_id,
            paths=path_hashes,
            remaining_contributions=self._remaining_contributions(),
            consumes_contribution_quota=budget_account == "contributions",
            budget_account=budget_account,
        )
        _atomic_text(receipt_path, receipt.model_dump_json(indent=2) + "\n")
        return self._finish_receipt_result(receipt)

    @staticmethod
    def _finish_receipt_result(receipt: FinishReceipt) -> dict[str, object]:
        result = receipt.model_dump(mode="json")
        result["remaining_run_contributions"] = result.pop("remaining_contributions")
        return result
