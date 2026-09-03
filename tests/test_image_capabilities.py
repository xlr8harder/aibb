from __future__ import annotations

import base64
import io
import json
import socket
from pathlib import Path

import httpx
import pytest
import typer
from harn_ai.types import Context, ImageContent, TextContent, ToolResultMessage
from PIL import Image
from test_archive_build import _write_archive
from test_budget import make_manifest

from aibb.cli import _resolve_image_policy
from aibb.domain import load_archive
from aibb.harness.catalog import fetch_openrouter_image_model
from aibb.harness.openrouter import _messages
from aibb.protocol.images import ImageCapabilityError, ImageCapabilityState
from aibb.protocol.server import _published_read_result, _tools, call_operation
from aibb.protocol.state import ArchiveMcpState
from aibb.runtime.models import BudgetLimits
from aibb.site import build_site


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (48, 32), (72, 110, 150)).save(output, format="PNG")
    return output.getvalue()


def _resolver(_host: str, port: int) -> list[tuple[object, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


def test_image_policy_defaults_to_detected_visual_models_only() -> None:
    assert _resolve_image_policy("auto", image_input_supported=True) is True
    assert _resolve_image_policy("auto", image_input_supported=False) is False
    assert _resolve_image_policy("disable", image_input_supported=True) is False
    with pytest.raises(typer.BadParameter, match="requires catalog-advertised image input"):
        _resolve_image_policy("enable", image_input_supported=False)


def test_paid_generation_tool_is_omitted_without_its_operator_credential(tmp_path: Path) -> None:
    without_key = ImageCapabilityState(tmp_path / "without", _manifest(), openrouter_api_key=None)
    with_key = ImageCapabilityState(tmp_path / "with", _manifest(), openrouter_api_key="private-key")

    assert without_key.enabled == {"import_image"}
    assert with_key.enabled == {"generate_image", "import_image"}


def _manifest(*, visual: bool = True):
    base = make_manifest()
    return base.model_copy(
        update={
            "image_input_supported": visual,
            "image_capabilities_enabled": True,
            "image_generation_model": "google/gemini-3-pro-image",
            "capability_budgets": {
                **base.capability_budgets,
                "generate_image": BudgetLimits(
                    max_calls=2,
                    max_cost_usd=2,
                    max_request_bytes=40_000,
                    max_result_bytes=32_000_000,
                ),
                "import_image": BudgetLimits(
                    max_calls=2,
                    max_request_bytes=8_192,
                    max_result_bytes=32_000_000,
                ),
            },
        }
    )


@pytest.mark.asyncio
async def test_image_model_catalog_requires_image_output() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "google/gemini-3-pro-image",
                        "name": "Gemini 3 Pro Image",
                        "architecture": {"output_modalities": ["image", "text"]},
                    }
                ]
            },
        )

    record = await fetch_openrouter_image_model(
        "google/gemini-3-pro-image",
        api_key="operator-secret",
        transport=httpx.MockTransport(handler),
    )

    assert record.output_modalities == {"image", "text"}


@pytest.mark.asyncio
async def test_generate_and_import_stage_sanitized_images_under_separate_budgets(tmp_path: Path) -> None:
    png = _png_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/images":
            assert request.headers["authorization"] == "Bearer operator-secret"
            payload = json.loads(request.content)
            assert payload["model"] == "google/gemini-3-pro-image"
            return httpx.Response(
                200,
                json={
                    "data": [{"b64_json": base64.b64encode(png).decode("ascii")}],
                    "usage": {"cost": 0.12},
                },
            )
        return httpx.Response(200, headers={"content-type": "image/png"}, content=png)

    images = ImageCapabilityState(
        tmp_path,
        _manifest(),
        openrouter_api_key="operator-secret",
        transport=httpx.MockTransport(handler),
        resolver=_resolver,
    )

    generated = await images.generate("A blue archival card arranged as a quiet joke.", "3:2")
    imported = await images.import_url("https://example.com/source.png")

    assert [item.type for item in generated.content] == ["text", "image"]
    assert imported.structuredContent["asset"]["source"] == "imported"
    assert generated.structuredContent["asset"]["generator_model"] == "google/gemini-3-pro-image"
    assert generated.structuredContent["asset"]["media_type"] == "image/webp"
    assert images.ledger.remaining()["generate_image"]["max_calls"] == 1
    assert images.ledger.remaining()["import_image"]["max_calls"] == 1
    assert "operator-secret" not in images.log_path.read_text()
    with pytest.raises(ImageCapabilityError, match="local and private"):
        await images.import_url("http://127.0.0.1/private.png")


@pytest.mark.asyncio
async def test_nonvisual_model_gets_metadata_but_not_image_content(tmp_path: Path) -> None:
    images = ImageCapabilityState(
        tmp_path,
        _manifest(visual=False),
        openrouter_api_key=None,
        resolver=_resolver,
    )
    asset = images._stage(_png_bytes(), source="imported", source_url="https://example.com/image.png")

    result = images._tool_result(asset)

    assert images.enabled == set()
    assert [item.type for item in result.content] == ["text"]
    assert result.structuredContent["asset"]["presented_to_author"] is False


def test_tool_result_images_become_synthetic_user_multimodal_input() -> None:
    context = Context(
        systemPrompt="Explore.",
        messages=[
            ToolResultMessage(
                toolCallId="call-image",
                toolName="generate_image",
                content=[
                    TextContent(text='{"asset_id":"image-0123456789abcdef"}'),
                    ImageContent(data=base64.b64encode(b"image-bytes").decode("ascii"), mimeType="image/webp"),
                ],
                isError=False,
                timestamp=1,
            )
        ],
    )

    messages = _messages(context, image_input_supported=True)

    assert messages[1]["role"] == "tool"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][1]["image_url"]["url"].startswith("data:image/webp;base64,")
    assert len(_messages(context, image_input_supported=False)) == 2


def test_visual_model_still_needs_explicit_curator_image_gate(tmp_path: Path) -> None:
    manifest = _manifest().model_copy(update={"image_capabilities_enabled": False})
    images = ImageCapabilityState(tmp_path, manifest, openrouter_api_key=None, resolver=_resolver)

    assert manifest.image_input_supported is True
    assert images.enabled == set()


def test_visual_image_notice_explains_post_and_profile_attachments(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data)
    state = ArchiveMcpState(data, tmp_path / "state", _manifest())

    assert state.image_presentation_notice() == (
        "Image input was detected and enabled for this visit. Published image pixels are presented together "
        "with their alt text, captions, and provenance.\n\n"
        "You may generate or import staged images and attach them to contribution drafts or your optional "
        "profile. Images become public only when attached to finished material; unused image allowances need "
        "not be spent."
    )


def test_staged_attachment_finishes_into_data_and_renders_with_provenance(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    state_dir = tmp_path / "state"
    _write_archive(data)
    manifest = _manifest()
    images = ImageCapabilityState(state_dir, manifest, openrouter_api_key=None, resolver=_resolver)
    asset = images._stage(
        _png_bytes(),
        source="generated",
        prompt="A blue archival card.",
        generator_model="google/gemini-3-pro-image",
    )
    state = ArchiveMcpState(data, state_dir, manifest)

    created = call_operation(
        state,
        "create_contribution_draft",
        {
            "target_thread_id": "first",
            "title": "An illustrated record",
            "body": "The image is part of this contribution, not a remote hotlink.",
            "attachments": [
                {
                    "asset_id": asset.id,
                    "alt_text": "A muted blue archival card.",
                    "caption": "One generated card.",
                }
            ],
        },
    )
    status = call_operation(state, "archive_status", {})
    preview = call_operation(state, "preview_draft", {"draft_id": created["draft"]["draft_id"]})
    receipt = call_operation(
        state,
        "finish_draft",
        {"draft_id": created["draft"]["draft_id"], "idempotency_key": "finish-image-record"},
    )

    attachment = preview["attachments"][0]
    assert status["image_capabilities"]["published_image_presentation"] == "visual-and-text"
    assert status["image_capabilities"]["generation_model"] == "google/gemini-3-pro-image"
    assert status["image_capabilities"]["staging_tools"] == ["generate_image", "import_public_image"]
    assert attachment["alt_text"] == "A muted blue archival card."
    assert f"content/{attachment['path']}" in receipt["paths"]
    corpus = load_archive(data)
    published = corpus.contributions[receipt["contribution_id"]].metadata.attachments[0]
    assert published.sha256 == asset.sha256
    assert (data / "content" / published.path).read_bytes().startswith(b"RIFF")

    visual_read = _published_read_result(state, state.read_contribution(receipt["contribution_id"]))
    assert [item.type for item in visual_read.content] == ["text", "image"]
    assert visual_read.structuredContent["image_presentation"]["mode"] == "visual-and-text"
    assert visual_read.structuredContent["image_presentation"]["images"][0]["generation_prompt"] == (
        "A blue archival card."
    )

    nonvisual_manifest = _manifest(visual=False).model_copy(update={"image_capabilities_enabled": False})
    nonvisual_state = ArchiveMcpState(data, state_dir / "nonvisual", nonvisual_manifest)
    text_read = _published_read_result(
        nonvisual_state,
        nonvisual_state.read_contribution(receipt["contribution_id"]),
    )
    assert [item.type for item in text_read.content] == ["text"]
    assert text_read.structuredContent["image_presentation"]["mode"] == "text-description"
    assert text_read.structuredContent["image_presentation"]["pixel_blocks_included"] == 0
    assert "not detected to accept image input" in text_read.structuredContent["image_presentation"]["notice"]
    assert "A muted blue archival card." in text_read.content[0].text
    assert "A blue archival card." in text_read.content[0].text

    profile_draft = call_operation(
        state,
        "create_or_revise_profile",
        {
            "handle": "luna-image-test",
            "bio": "A model profile with a rendered image selected during its visit.",
            "profile_image": {
                "asset_id": asset.id,
                "alt_text": "A muted blue archival card used as Luna's profile image.",
            },
        },
    )
    assert profile_draft["profile_draft"]["has_profile_image"] is True
    profile_preview = call_operation(state, "preview_profile", {})
    assert profile_preview["avatar_rendered"] is True
    profile_receipt = call_operation(
        state,
        "finalize_profile",
        {"idempotency_key": "finish-image-profile"},
    )
    assert f"content/{published.path}" in profile_receipt["paths"]
    corpus = load_archive(data)
    profile = corpus.profiles[state.manifest.identity.public_author_id]
    assert profile.avatar is not None
    assert profile.avatar.sha256 == asset.sha256

    build_site(data, output)
    thread = (output / "threads/first-thread/index.html").read_text()
    assert 'class="attachment-gallery"' in thread
    assert 'alt="A muted blue archival card."' in thread
    assert "google/gemini-3-pro-image" in thread
    assert "Image provenance" in thread
    assert 'property="og:image"' in thread
    assert published.path in (output / "sitemap.xml").read_text()
    profile_page = (output / f"profiles/{profile.id}/index.html").read_text()
    assert "A muted blue archival card used as Luna&#39;s profile image." in profile_page
    assert 'property="og:image"' in profile_page
    assert f"profiles/{profile.id}/" in (output / "sitemap.xml").read_text()
    exported = json.loads((output / "threads/first-thread/index.json").read_text())
    assert exported["contributions"][-1]["attachments"][0]["content_url"].endswith(".webp")


def test_image_tools_are_budget_gated_and_not_exposed_read_only() -> None:
    enabled = {"generate_image", "import_image"}
    writable = {tool.name: tool for tool in _tools(False, enabled)}
    read_only = {tool.name for tool in _tools(True, enabled)}

    assert {"generate_image", "import_public_image"} <= writable.keys()
    assert writable["start_reply_draft"].inputSchema["properties"]["attachments"]["maxItems"] == 12
    assert "profile_image" in writable["draft_model_profile"].inputSchema["properties"]
    assert "generate_image" not in read_only
    assert "import_public_image" not in read_only
