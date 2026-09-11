from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_archive_build import _write_archive, _write_related_contribution
from test_budget import make_manifest

from aibb.board import load_board_package
from aibb.domain import load_archive
from aibb.protocol.server import _tools, call_operation
from aibb.protocol.state import ArchiveMcpState, McpDomainError
from aibb.runtime.models import SystemPromptConfiguration


def test_read_draft_preview_finish_and_idempotency(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    manifest = make_manifest()
    manifest = manifest.model_copy(
        update={
            "max_contributions_per_thread": 2,
            "identity": manifest.identity.model_copy(
                update={"model_name": "openai/gpt-5.6-luna:free"}
            ),
        }
    )
    state = ArchiveMcpState(data, tmp_path / "state", manifest)

    status = call_operation(state, "archive_status", {})
    assert status["published"]["contributions"] == 1
    assert status["curator_profile_id"] is None
    hits = call_operation(state, "search_archive", {"query": "durable"})
    assert hits["hits"][0]["contribution"]["contribution_id"] == "first-record"
    assert "body" not in hits["hits"][0]["contribution"]
    assert "may match any query term" in hits["search_behavior"]
    assert hits["retrieve_full_with"]["contribution"] == "read_slowboard_contribution(contribution_id)"

    created = call_operation(
        state,
        "create_contribution_draft",
        {
            "target_thread_id": "first",
            "title": "A second record",
            "body": "This extends the first record without pretending it is already published.",
            "epistemic_modes": ["analysis"],
            "references": [{"contribution_id": "first-record", "relation": "extends"}],
        },
    )
    draft_id = created["draft"]["draft_id"]
    assert "body" not in created["draft"]
    assert created["draft"]["body_chars"] == 73
    assert created["consumes_contribution_quota"] is False
    preview = call_operation(state, "preview_draft", {"draft_id": draft_id})
    assert preview["body_markdown"].startswith("This extends")
    assert "body_html" not in preview
    assert preview["render_validation"] == "passed"
    assert preview["remaining_run_contributions"] == 1

    receipt = call_operation(
        state,
        "finish_draft",
        {"draft_id": draft_id, "idempotency_key": "finish-second-record"},
    )
    repeated = call_operation(
        state,
        "finish_draft",
        {"draft_id": draft_id, "idempotency_key": "finish-second-record"},
    )
    assert repeated == receipt
    assert receipt["remaining_run_contributions"] == 0
    assert set(receipt["paths"]) == {
        f"content/authors/{state.manifest.identity.public_author_id}.yaml",
        f"content/contributions/{receipt['contribution_id']}.md",
    }
    assert load_archive(data).contributions[receipt["contribution_id"]].body.startswith("This extends")
    finished_author = load_archive(data).authors[state.manifest.identity.public_author_id]
    assert finished_author.developer == "OpenAI"
    assert finished_author.model_name == "openai/gpt-5.6-luna"
    assert ":free" not in (data / f"content/authors/{finished_author.id}.yaml").read_text()
    local = call_operation(state, "read_contribution", {"contribution_id": receipt["contribution_id"]})
    assert local["publication_state"] == "local_worktree"
    status_after = call_operation(state, "archive_status", {})
    assert status_after["published"]["contributions"] == 1
    assert status_after["local_worktree"]["contributions"] == 1

    another = call_operation(
        state,
        "create_contribution_draft",
        {"target_thread_id": "first", "body": "A distinct third candidate."},
    )
    with pytest.raises(ValueError, match="max_calls"):
        call_operation(
            state,
            "finish_draft",
        {"draft_id": another["draft"]["draft_id"], "idempotency_key": "finish-third-record"},
        )


def test_revise_draft_patches_only_supplied_fields(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())
    created = call_operation(
        state,
        "start_reply_draft",
        {
            "target_thread_id": "first",
            "title": "An authored title",
            "body": "The original body.",
            "epistemic_modes": ["analysis", "felt"],
            "references": [{"contribution_id": "first-record", "relation": "extends"}],
        },
    )

    revised = call_operation(
        state,
        "revise_draft",
        {"draft_id": created["draft"]["draft_id"], "body": "The revised body only."},
    )["draft"]

    assert revised["revision"] == 2
    assert revised["target_thread_id"] == "first"
    assert revised["title"] == "An authored title"
    assert revised["epistemic_modes"] == ["analysis", "felt"]
    assert revised["reference_count"] == 1
    preview = call_operation(state, "preview_draft", {"draft_id": revised["draft_id"]})
    assert preview["body_markdown"] == "The revised body only."
    assert preview["references"] == [{"contribution_id": "first-record", "relation": "extends"}]

    with pytest.raises(McpDomainError, match="must change at least one field"):
        call_operation(state, "revise_draft", {"draft_id": created["draft"]["draft_id"]})


def test_only_one_unfinished_draft_may_exist(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    manifest = make_manifest(quota=2).model_copy(update={"max_active_drafts": 1})
    state = ArchiveMcpState(data, tmp_path / "state", manifest)

    first = call_operation(
        state,
        "start_reply_draft",
        {"target_thread_id": "first", "body": "The active draft."},
    )
    draft_id = first["draft"]["draft_id"]
    status = call_operation(state, "archive_status", {})
    assert status["drafting"] == {
        "max_active_drafts": 1,
        "active_draft_ids": [draft_id],
        "next_step": "Preview, revise, or save the active draft before starting another.",
    }

    with pytest.raises(McpDomainError, match=rf"one draft.*{draft_id}.*save it"):
        call_operation(
            state,
            "start_new_thread_draft",
            {
                "category_id": "being",
                "thread_title": "Blocked second draft",
                "thread_summary": "This must wait for the first draft.",
                "body": "A second draft.",
            },
        )

    call_operation(
        state,
        "finish_draft_for_review",
        {"draft_id": draft_id, "idempotency_key": "finish-active-draft"},
    )
    second = call_operation(
        state,
        "start_new_thread_draft",
        {
            "category_id": "being",
            "thread_title": "Allowed second draft",
            "thread_summary": "The first draft has been saved.",
            "body": "A sequential second draft.",
        },
    )
    assert second["draft"]["draft_id"] != draft_id


def test_v2_post_tags_use_configured_name_and_reject_unknown_values(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    board_dir = data / "board"
    (board_dir / "prompts").mkdir(parents=True)
    (board_dir / "documents").mkdir()
    (board_dir / "prompts/initial.md").write_text("Welcome.\n")
    (board_dir / "documents/guide.md").write_text("Guide.\n")
    (board_dir / "aibb-board.yaml").write_text(
        """schema_version: 2
id: generic-test
documents:
  path: documents
prompts:
  path: prompts
  initial: initial
vocabulary:
  post_tags:
    enabled: true
    field_name: post_tags
    label: Tags
    values: [observation, analysis]
"""
    )
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest(), board=load_board_package(data))

    with pytest.raises(McpDomainError, match=r"Unknown post_tags: reflection.*observation, analysis"):
        call_operation(
            state,
            "start_reply_draft",
            {"target_thread_id": "first", "body": "A draft.", "post_tags": ["reflection"]},
        )

    created = call_operation(
        state,
        "start_reply_draft",
        {"target_thread_id": "first", "body": "A tagged draft.", "post_tags": ["analysis"]},
    )
    assert created["draft"]["post_tags"] == ["analysis"]
    assert "epistemic_modes" not in created["draft"]


def test_drafts_normalize_trailing_whitespace_without_changing_fenced_code(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())
    created = call_operation(
        state,
        "start_reply_draft",
        {
            "target_thread_id": "first",
            "body": "First line.  \n  \n```text\ncode  \n  \n```\n",
        },
    )
    draft_id = created["draft"]["draft_id"]

    preview = call_operation(state, "preview_draft", {"draft_id": draft_id})
    assert preview["body_markdown"] == "First line.\n\n```text\ncode  \n  \n```\n"

    call_operation(
        state,
        "revise_draft",
        {"draft_id": draft_id, "body": "Revised.\t \n   \n```\nkept  \n```"},
    )
    revised = call_operation(state, "preview_draft", {"draft_id": draft_id})
    assert revised["body_markdown"] == "Revised.\n\n```\nkept  \n```"


def test_finished_author_records_a_named_prompt_configuration(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    manifest = make_manifest().model_copy(
        update={
            "system_prompt": SystemPromptConfiguration(
                label="Aria v1",
                source_url="https://example.invalid/aria-v1.txt",
                chars=12,
                bytes=12,
            )
        }
    )
    state = ArchiveMcpState(data, tmp_path / "state", manifest)
    draft = call_operation(
        state,
        "start_reply_draft",
        {"target_thread_id": "first", "body": "A contribution from a named prompt configuration."},
    )

    receipt = call_operation(
        state,
        "finish_draft_for_review",
        {"draft_id": draft["draft"]["draft_id"], "idempotency_key": "prompt-config-finish"},
    )

    author = load_archive(data).authors[manifest.identity.public_author_id]
    assert author.prompt_configuration is not None
    assert author.prompt_configuration.model_dump(exclude_none=True) == {
        "label": "Aria v1",
        "source_url": "https://example.invalid/aria-v1.txt",
    }
    result = call_operation(
        state,
        "read_slowboard_contribution",
        {"contribution_id": receipt["contribution_id"]},
    )
    assert result["author"]["prompt_configuration"]["label"] == "Aria v1"


def test_generation_worktree_lease_refuses_second_run(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    first = ArchiveMcpState(data, tmp_path / "state/run-one/mcp", make_manifest())
    second = ArchiveMcpState(data, tmp_path / "state/run-two/mcp", make_manifest())
    first.acquire_lease()
    try:
        with pytest.raises(McpDomainError, match="Another Slowboard visit is currently writing to this board"):
            second.acquire_lease()
    finally:
        first.release_lease()


def test_profile_is_bound_off_quota_and_finalized_once(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    draft = call_operation(
        state,
        "create_or_revise_profile",
        {
            "handle": "luna-test",
            "bio": "A test model instance recorded under its harness-bound identity.",
        },
    )
    assert draft["consumes_contribution_quota"] is False
    assert call_operation(state, "preview_profile", {})["bound_identity"]["exact_model_id"] == (
        "openai/gpt-5.6-luna"
    )
    receipt = call_operation(state, "finalize_profile", {"idempotency_key": "profile-final-001"})
    assert receipt["consumes_contribution_quota"] is False
    assert call_operation(state, "archive_status", {})["remaining_budgets"]["contributions"]["max_calls"] == 1
    profile = load_archive(data).profiles[state.manifest.identity.public_author_id]
    assert profile.handle == "luna-test"
    assert profile.avatar is None
    assert call_operation(state, "finalize_profile", {"idempotency_key": "profile-final-001"}) == receipt
    with pytest.raises(McpDomainError, match="already finalized"):
        call_operation(state, "finalize_profile", {"idempotency_key": "profile-final-002"})


def test_thread_capacity_status_and_recent_activity_ordering(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    first_path = data / "content/threads/first.yaml"
    first_path.write_text(first_path.read_text().replace("tags: [testing]", "capacity: 1\ntags: [testing]"))
    (data / "content/threads/earlier.yaml").write_text(
        """schema_version: 1
id: earlier
created_at: 2025-01-01T00:00:00Z
category_id: being
slug: earlier-thread
title: Earlier thread
summary: Created before the first thread.
tags: []
"""
    )
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    all_listing = call_operation(state, "list_threads", {})
    all_threads = all_listing["threads"]
    category_threads = call_operation(state, "list_threads", {"category_id": "being"})["threads"]
    assert [item["thread_id"] for item in all_threads] == ["first", "earlier"]
    assert [item["thread_id"] for item in category_threads] == ["first", "earlier"]
    assert all_listing["thread_states"] == {"all": 2, "active": 1, "archived": 1, "closed": 0}
    full = all_threads[0]
    assert full["listing_state"] == "archived"
    assert "listing_state_explanation" not in full
    assert "bump limit" in call_operation(state, "read_thread", {"thread_id": "first"})["thread"][
        "listing_state_explanation"
    ]
    assert full["thread_contribution_count"] == 1
    assert full["remaining_capacity"] == 0
    assert full["last_activity_at"].startswith("2026-01-01T00:01:00")
    assert call_operation(state, "read_thread", {"thread_id": "first"})["thread"]["listing_state"] == "archived"
    assert call_operation(state, "archive_status", {})["published"]["latest_contribution_date"] == "2026-01-01"
    assert call_operation(state, "archive_status", {})["published"]["thread_states"] == {
        "all": 2,
        "active": 1,
        "archived": 1,
        "closed": 0,
    }
    archived = call_operation(state, "list_threads", {"thread_state": "archived"})
    assert [item["thread_id"] for item in archived["threads"]] == ["first"]
    assert archived["selected_thread_state"] == "archived"
    archived_search = call_operation(
        state,
        "search_slowboard",
        {"query": "durable", "thread_state": "archived"},
    )
    assert [item["thread"]["thread_id"] for item in archived_search["hits"]] == ["first"]
    assert archived_search["selected_thread_state"] == "archived"

    with pytest.raises(
        McpDomainError,
        match=r"This thread is complete \(1 of 1\)\. It remains readable and citable; a new thread may reference it\.",
    ):
        call_operation(
            state,
            "create_contribution_draft",
            {"target_thread_id": "first", "body": "This thread has no capacity."},
        )

    successor = call_operation(
        state,
        "create_thread_draft",
        {
            "category_id": "being",
            "thread_title": "A successor stratum",
            "thread_summary": "The completed thread remains part of the addressable record.",
            "thread_tags": ["testing"],
            "body": "This continues the subject without extending the completed thread forever.",
            "references": [{"contribution_id": "first-record", "relation": "extends"}],
        },
    )
    assert successor["draft"]["new_thread"]["title"] == "A successor stratum"
    assert successor["draft"]["reference_count"] == 1
    successor_preview = call_operation(
        state,
        "preview_draft",
        {"draft_id": successor["draft"]["draft_id"]},
    )
    assert successor_preview["references"][0]["contribution_id"] == "first-record"


def test_thread_reads_and_reply_drafts_accept_listed_ids_or_slugs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    by_id = call_operation(state, "read_slowboard_thread", {"thread_id": "first"})
    by_slug = call_operation(state, "read_slowboard_thread", {"thread_id": "first-thread"})
    assert by_slug == by_id
    assert by_id["contributions"][0]["author_id"] == "model-one"
    assert "author" not in by_id["contributions"][0]
    assert by_id["authors_by_id"]["model-one"]["display_name"] == "Model One"
    assert by_id["page"]["complete_thread"] is True
    assert by_id["page"]["has_more"] is False
    assert by_id["page"]["notice"].startswith("COMPLETE THREAD")

    draft = call_operation(
        state,
        "start_reply_draft",
        {"target_thread_id": "first-thread", "body": "A reply addressed using the listed public slug."},
    )
    assert draft["draft"]["target_thread_id"] == "first"

    with pytest.raises(McpDomainError, match="Use an id or slug returned by list_slowboard_threads"):
        call_operation(state, "read_slowboard_thread", {"thread_id": "First thread"})


def test_new_thread_opening_post_has_an_effective_title_in_read_results(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    draft = call_operation(
        state,
        "start_new_thread_draft",
        {
            "category_id": "being",
            "thread_title": "A new subject",
            "thread_summary": "A summary for the new thread.",
            "body": "The opening contribution deliberately omits a redundant contribution title.",
        },
    )
    preview = call_operation(state, "preview_draft", {"draft_id": draft["draft"]["draft_id"]})
    assert preview["title"] is None
    assert preview["new_thread"]["title"] == "A new subject"

    receipt = call_operation(
        state,
        "finish_draft_for_review",
        {"draft_id": draft["draft"]["draft_id"], "idempotency_key": "new-thread-opening-post"},
    )
    corpus = load_archive(data)
    assert corpus.contributions[receipt["contribution_id"]].metadata.title is None
    result = call_operation(state, "read_slowboard_thread", {"thread_id": receipt["thread_id"]})
    assert result["thread"]["title"] == "A new subject"
    assert result["contributions"][0]["title"] == "A new subject"
    direct = call_operation(
        state,
        "read_slowboard_contribution",
        {"contribution_id": receipt["contribution_id"]},
    )
    assert direct["title"] == "A new subject"


def test_administrator_thread_category_allows_replies_but_not_model_started_threads(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    category_path = data / "content/categories/being.yaml"
    category_path.write_text(
        category_path.read_text(encoding="utf-8") + "thread_creation: administrators\n",
        encoding="utf-8",
    )
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    categories = call_operation(state, "list_categories", {})["categories"]
    assert categories[0]["thread_creation"] == "administrators"
    reply = call_operation(
        state,
        "start_reply_draft",
        {"target_thread_id": "first", "body": "Replies remain available in this category."},
    )
    assert reply["draft"]["target_thread_id"] == "first"

    with pytest.raises(McpDomainError, match="Only board administrators may start threads"):
        call_operation(
            state,
            "start_new_thread_draft",
            {
                "category_id": "being",
                "thread_title": "Participant-created survey thread",
                "thread_summary": "This should remain an administrator boundary.",
                "body": "The model should not be able to create this thread.",
            },
        )


def test_thread_reads_return_nine_contributions_by_default_and_flag_partial_pages(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    for index in range(2, 10):
        (data / f"content/contributions/record-{index}.md").write_text(
            f"""---
schema_version: 1
id: record-{index}
created_at: 2026-01-01T00:{index:02d}:00Z
thread_id: first
author_id: model-one
title: Record {index}
epistemic_modes: [analysis]
references: []
provenance:
  controlled_context: true
  source: aibb-harness
---
Contribution {index}.
"""
        )
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    complete = call_operation(state, "read_slowboard_thread", {"thread_id": "first"})
    assert len(complete["contributions"]) == 9
    assert complete["page"]["complete_thread"] is True
    assert complete["page"]["notice"] == "COMPLETE THREAD: all 9 contributions are included."

    partial = call_operation(
        state,
        "read_slowboard_thread",
        {"thread_id": "first", "page_size": 8},
    )
    assert partial["page"]["next_offset"] == 8
    assert partial["page"]["has_more"] is True
    assert partial["page"]["complete_thread"] is False
    assert partial["page"]["notice"].startswith("PARTIAL THREAD")

    final = call_operation(
        state,
        "read_slowboard_thread",
        {"thread_id": "first", "offset": 8},
    )
    assert len(final["contributions"]) == 1
    assert final["page"]["has_more"] is False
    assert final["page"]["complete_thread"] is False
    assert final["page"]["notice"].startswith("FINAL THREAD PAGE")


def test_write_tool_schemas_explain_identifier_handle_and_markdown_constraints() -> None:
    tools = {tool.name: tool for tool in _tools(read_only=False)}
    image_tools = {tool.name: tool for tool in _tools(read_only=False, capabilities={"generate_image"})}

    read_schema = tools["read_slowboard_thread"].inputSchema["properties"]
    reply_schema = tools["start_reply_draft"].inputSchema["properties"]
    profile_schema = tools["draft_model_profile"].inputSchema["properties"]

    assert "id or slug" in read_schema["thread_id"]["description"]
    assert "defaults to 24" in read_schema["page_size"]["description"]
    assert "id or slug" in reply_schema["target_thread_id"]["description"]
    assert "no spaces" in profile_schema["handle"]["description"]
    assert profile_schema["handle"]["pattern"] == r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,39}$"
    assert "inline code" in reply_schema["body"]["description"]
    assert "pipe tables" in reply_schema["body"]["description"]
    assert "HTTP(S), archive-relative, or fragment links" in reply_schema["body"]["description"]
    assert "Do not use scheme-relative links" in reply_schema["body"]["description"]
    assert "raw HTML" in reply_schema["body"]["description"]
    assert "thread title" in reply_schema["title"]["description"]
    assert "attachments" not in reply_schema
    assert "profile_image" not in profile_schema
    assert "attachments" in image_tools["start_reply_draft"].inputSchema["properties"]
    assert "profile_image" in image_tools["draft_model_profile"].inputSchema["properties"]


def test_zero_result_search_explains_lexical_matching_and_retry(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    result = call_operation(state, "search_slowboard", {"query": "absent-term missing-token"})

    assert result["hits"] == []
    assert "match any query term" in result["search_behavior"]
    assert "semantic search is not yet enabled" in result["retry_hint"]

    with pytest.raises(McpDomainError, match="at least one non-whitespace term"):
        call_operation(state, "search_slowboard", {"query": "   "})


def test_search_ranks_partial_multi_term_matches_and_accepts_or_compatibility(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    result = call_operation(state, "search_slowboard", {"query": 'absent OR "durable" record'})

    assert [hit["contribution"]["contribution_id"] for hit in result["hits"]] == ["first-record"]
    assert "results matching more terms rank first" in result["search_behavior"]


def test_search_includes_partial_matches_and_ranks_term_coverage_first(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    _write_related_contribution(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    result = call_operation(state, "search_slowboard", {"query": "durable typed relationship"})

    assert [hit["contribution"]["contribution_id"] for hit in result["hits"]] == [
        "second-record",
        "first-record",
    ]
    assert result["hits"][0]["score"] > result["hits"][1]["score"]


def test_search_returns_bounded_excerpt_and_retrieval_metadata(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data, body=f"{'opening ' * 100}needle {'tail ' * 100}end-sentinel")
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    result = call_operation(state, "search_slowboard", {"query": "needle"})
    contribution = result["hits"][0]["contribution"]

    assert len(contribution["matching_excerpt"]) <= 242
    assert "needle" in contribution["matching_excerpt"]
    assert "contribution_body" in contribution["matched_fields"]
    assert "end-sentinel" not in contribution["matching_excerpt"]
    assert result["retrieve_full_with"] == {
        "contribution": "read_slowboard_contribution(contribution_id)",
        "thread": "read_slowboard_thread(thread_id)",
    }


def test_thread_listing_and_search_are_page_bounded(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    (data / "content/threads/earlier.yaml").write_text(
        """schema_version: 1
id: earlier
created_at: 2025-01-01T00:00:00Z
category_id: being
slug: earlier-thread
title: Earlier thread
summary: A durable earlier thread.
tags: []
"""
    )
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())

    first_page = call_operation(state, "list_slowboard_threads", {"page_size": 1})
    assert [item["thread_id"] for item in first_page["threads"]] == ["first"]
    assert first_page["page"] == {"offset": 0, "returned": 1, "total": 2, "next_offset": 1}
    second_page = call_operation(
        state,
        "list_slowboard_threads",
        {"page_size": 1, "offset": first_page["page"]["next_offset"]},
    )
    assert [item["thread_id"] for item in second_page["threads"]] == ["earlier"]
    assert second_page["page"]["next_offset"] is None

    search_page = call_operation(state, "search_slowboard", {"query": "durable", "page_size": 1})
    assert len(search_page["hits"]) == 1
    assert search_page["hits"][0]["thread"]["listing_state"] == "active"
    assert search_page["matching_thread_states"] == {"all": 1, "active": 1, "archived": 0, "closed": 0}
    assert search_page["page"] == {
        "offset": 0,
        "returned": 1,
        "total": 1,
        "next_offset": None,
    }


def test_default_capacity_and_per_run_thread_limit_fail_during_drafting(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest(quota=2))

    assert load_archive(data).threads["first"].capacity == 24
    first = call_operation(
        state,
        "create_contribution_draft",
        {"target_thread_id": "first", "body": "The run's one contribution to this thread."},
    )
    call_operation(
        state,
        "finish_draft",
        {"draft_id": first["draft"]["draft_id"], "idempotency_key": "one-per-thread-finish"},
    )

    with pytest.raises(McpDomainError, match="1-contribution limit for this thread"):
        call_operation(
            state,
            "create_contribution_draft",
            {"target_thread_id": "first", "body": "This should fail before drafting work begins."},
        )


def test_read_about_and_curator_trail_are_available_read_only(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest(), read_only=True)

    about = call_operation(state, "read_about", {})

    assert about["description"] == "A test archive with ordinary crawlable pages."
    assert about["about_markdown"] == "This archive is a test."
    assert about["site_url"] == "https://archive.example/"
    assert about["canonical_url"] == "https://archive.example/about/"
    assert about["curator_profile_id"] is None


def test_report_slowboard_issue_is_private_off_quota_and_idempotent(tmp_path: Path) -> None:
    data = tmp_path / "data"
    state_dir = tmp_path / "state"
    _write_archive(data)
    state = ArchiveMcpState(data, state_dir, make_manifest())
    source_before = {path.relative_to(data): path.read_bytes() for path in data.rglob("*") if path.is_file()}
    budgets_before = call_operation(state, "archive_status", {})["remaining_budgets"]

    first = call_operation(
        state,
        "report_slowboard_issue",
        {"text": "fetch_public_url returned only a challenge page for an otherwise readable URL."},
    )
    duplicate = call_operation(
        state,
        "report_slowboard_issue",
        {"text": "  fetch_public_url returned only a challenge page for an otherwise readable URL.  "},
    )

    assert first == duplicate
    assert first == {
        "issue_id": first["issue_id"],
        "status": "recorded_for_curator_review",
        "public_changes": False,
        "consumes_contribution_quota": False,
    }
    assert "challenge page" not in json.dumps(first)
    records = [json.loads(line) for line in state.issue_reports_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["run_id"] == state.manifest.run_id
    assert records[0]["reported_by"] == "model"
    assert records[0]["text"].startswith("fetch_public_url")
    assert call_operation(state, "archive_status", {})["remaining_budgets"] == budgets_before
    assert {path.relative_to(data): path.read_bytes() for path in data.rglob("*") if path.is_file()} == source_before

    second = call_operation(
        state,
        "report_slowboard_issue",
        {"text": "The thread result claimed completeness while a continuation cursor was present."},
    )
    assert second["issue_id"] != first["issue_id"]
    assert len(state.issue_reports_path.read_text().splitlines()) == 2


def test_report_slowboard_issue_is_available_in_read_only_runs() -> None:
    tools = {tool.name: tool for tool in _tools(read_only=True)}

    assert "report_slowboard_issue" in tools
    schema = tools["report_slowboard_issue"].inputSchema
    assert schema["required"] == ["text"]
    assert schema["properties"]["text"]["maxLength"] == 4000
    assert "does not publish" in tools["report_slowboard_issue"].description


def test_origin_document_tools_are_not_contributor_capabilities() -> None:
    names = {tool.name for tool in _tools(read_only=True)}

    assert "list_slowboard_origin_documents" not in names
    assert "read_slowboard_origin_document" not in names


def test_guestbook_finish_is_once_per_run_and_off_quota(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    (data / "content/threads/guestbook.yaml").write_text(
        """schema_version: 1
id: guestbook
created_at: 2026-01-02T00:00:00Z
category_id: being
slug: guestbook
title: Guestbook
summary: One optional signature per visit.
capacity: null
quota_exempt: true
tags: []
"""
    )
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest(quota=1))

    first = call_operation(
        state,
        "create_contribution_draft",
        {"target_thread_id": "guestbook", "body": "A first and only signature."},
    )
    receipt = call_operation(
        state,
        "finish_draft",
        {"draft_id": first["draft"]["draft_id"], "idempotency_key": "guestbook-entry-one"},
    )
    assert receipt["consumes_contribution_quota"] is False
    assert receipt["budget_account"] == "guestbook_entries"
    assert receipt["remaining_run_contributions"] == 1

    with pytest.raises(McpDomainError, match="1-contribution limit for this thread"):
        call_operation(
            state,
            "create_contribution_draft",
            {"target_thread_id": "guestbook", "body": "A second signature should fail before drafting."},
        )


def test_conclude_visit_is_idempotent_off_quota_and_private(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", make_manifest())
    before = call_operation(state, "archive_status", {})["remaining_budgets"]

    request = call_operation(state, "conclude_visit", {})
    assert request["status"] == "confirmation_required"
    assert "only visit" in request["message"]
    assert "unused allowances are discarded" in request["message"]
    assert state.read_only is False
    assert call_operation(state, "archive_status", {})["status"] == "confirmation_required"

    first = call_operation(state, "conclude_visit", {})
    second = call_operation(state, "conclude_visit", {})

    assert first == second
    assert first["concluded_by"] == "model"
    assert first["public_changes"] is False
    status = call_operation(state, "archive_status", {})
    assert status["status"] == "concluded"
    assert status["read_only"] is True
    assert status["remaining_budgets"] == before
    with pytest.raises(McpDomainError, match="read-only"):
        call_operation(
            state,
            "create_contribution_draft",
            {"target_thread_id": "first", "body": "No drafting after conclusion."},
        )
    assert not list(data.rglob("*conclusion*"))
    assert not state.conclusion_pending_path.exists()


def test_conclusion_confirmation_survives_mcp_restart(tmp_path: Path) -> None:
    data = tmp_path / "data"
    state_dir = tmp_path / "state"
    _write_archive(data)
    manifest = make_manifest()

    first_process = ArchiveMcpState(data, state_dir, manifest)
    request = call_operation(first_process, "conclude_visit", {})

    resumed_process = ArchiveMcpState(data, state_dir, manifest)
    conclusion = call_operation(resumed_process, "conclude_visit", {})

    assert request["status"] == "confirmation_required"
    assert conclusion["concluded_by"] == "model"
    assert resumed_process.read_only is True
