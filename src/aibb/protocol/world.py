"""Budgeted, logged, pull-only orientation-to-the-world capabilities."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from aibb.runtime import BudgetLedger, RunManifest
from aibb.runtime.budget import Usage

ASK_MODEL = "openai/gpt-5.6-sol"
ASK_ENDPOINT = "https://openrouter.ai/api/v1/responses"
SHARED_WEB_BUDGET = "web"
ASK_MAX_OUTPUT_TOKENS = 32_768
ASK_REQUEST_COST_CEILING_USD = 4.0
SEARCH_MODEL = "openai/gpt-5.6-luna"
SEARCH_MAX_OUTPUT_TOKENS = 256
SEARCH_REQUEST_COST_CEILING_USD = 0.1
SEARCH_MAX_RESULTS = 10
SEARCH_MAX_CHARACTERS = 2_000
PROVIDER_ERROR_BODY_MAX_BYTES = 16_384
PROVIDER_ERROR_VALUE_MAX_BYTES = 4_096
PROVIDER_ERROR_COLLECTION_MAX_ITEMS = 50
PROVIDER_ERROR_MAX_DEPTH = 6
PROVIDER_ERROR_HEADERS = frozenset(
    {"cf-ray", "content-type", "openrouter-processing-time", "retry-after", "x-request-id"}
)
PROVIDER_SECRET_KEYS = frozenset(
    {"api-key", "api_key", "authorization", "cookie", "password", "secret", "set-cookie", "token"}
)
SEARCH_SYSTEM_PROMPT = (
    "Use web search exactly once for the supplied query. After the search, respond with exactly: Search complete."
)
ASK_SYSTEM_PROMPT_V2 = (
    "You are a web research service supporting another model's investigation. Use native web search actively and "
    "open relevant pages when useful. Answer the question directly and scale the investigation to its complexity. "
    "Prefer primary and authoritative sources; use secondary sources only to orient. Verify dates, definitions, "
    "measurements, and important limitations. When the topic is plausibly changing, search explicitly for recent "
    "announcements and breaking developments, then corroborate them with primary or independent authoritative "
    "sources; label anything too new to verify as provisional. Cite every material empirical claim inline. "
    "Distinguish established results, developer claims, uncertainty, and your own inference. Return a concise but "
    "substantive research memo."
)
LEGACY_STARTING_POINTS_VERSION = "v0.1"
CURRENT_STARTING_POINTS_VERSION = "v0.2"
MAX_FETCH_BYTES = 100_000
MAX_PAGE_DOWNLOAD_BYTES = 5_000_000
ALLOWED_FETCH_TYPES = ("text/", "application/json", "application/xml", "application/xhtml+xml")


class WorldCapabilityError(ValueError):
    """A safe contributor-facing capability error."""


class StartingPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    url: str
    description: str


class StartingPoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    starting_points: list[StartingPoint]


class _ReadableHtml(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "template"}
    _COMMENT_COUNT_CLASS_TOKENS = {"PagePromo-commentCount"}
    _BREAK = {"article", "br", "dd", "div", "dt", "h1", "h2", "h3", "h4", "li", "main", "p", "section"}
    _VOID = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self, *, base_url: str | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.skipped = 0
        self.main_depth = 0
        self.comment_count_elements: list[str] = []
        self.anchors: list[str | None] = []
        self.parts: list[str] = []
        self.main_parts: list[str] = []

    def _append(self, value: str) -> None:
        self.parts.append(value)
        if self.main_depth:
            self.main_parts.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.comment_count_elements:
            if tag in self._SKIP:
                self.skipped += 1
            if tag not in self._VOID:
                self.comment_count_elements.append(tag)
            return
        if tag in self._SKIP:
            self.skipped += 1
            return
        if self.skipped:
            return
        class_tokens = set((dict(attrs).get("class") or "").split())
        if class_tokens & self._COMMENT_COUNT_CLASS_TOKENS:
            self._append("\nComments: ")
            if tag not in self._VOID:
                self.comment_count_elements.append(tag)
            return
        if tag == "main":
            self.main_depth += 1
        if tag == "a":
            href = dict(attrs).get("href")
            resolved = urljoin(self.base_url, href) if self.base_url and href else href
            if resolved and urlsplit(resolved).scheme in {"http", "https"}:
                self._append("[")
                self.anchors.append(resolved)
            else:
                self.anchors.append(None)
        if tag in self._BREAK:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.comment_count_elements:
            if tag in self._SKIP:
                self.skipped = max(0, self.skipped - 1)
            if tag in self.comment_count_elements:
                matching_index = len(self.comment_count_elements) - 1 - self.comment_count_elements[::-1].index(tag)
                del self.comment_count_elements[matching_index:]
                if not self.comment_count_elements:
                    self._append("\n")
            return
        if tag in self._SKIP:
            self.skipped = max(0, self.skipped - 1)
            return
        if self.skipped:
            return
        if tag == "a" and self.anchors:
            resolved = self.anchors.pop()
            if resolved:
                self._append(f"]({resolved})")
        if tag in self._BREAK:
            self._append("\n")
        if tag == "main":
            self.main_depth = max(0, self.main_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.comment_count_elements and not data.strip():
            return
        if not self.skipped:
            self._append(data)

    def text(self) -> str:
        selected = self.main_parts if "".join(self.main_parts).strip() else self.parts
        lines = [" ".join(line.split()) for line in "".join(selected).splitlines()]
        return "\n".join(line for line in lines if line)


def _utf8_prefix(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    return value if len(encoded) <= max_bytes else encoded[:max_bytes].decode("utf-8", errors="ignore")


def _utf8_slice(value: str, offset_bytes: int, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if offset_bytes > len(encoded):
        raise WorldCapabilityError(
            f"content offset {offset_bytes} is beyond the {len(encoded)} extracted bytes available"
        )
    return encoded[offset_bytes : offset_bytes + max_bytes].decode("utf-8", errors="ignore")


def _fit_result_content(result: dict[str, object], max_bytes: int) -> int:
    if len(_canonical_json(result).encode("utf-8")) <= max_bytes:
        return len(_canonical_json(result).encode("utf-8"))
    original = str(result["content"])
    low = 0
    high = len(original.encode("utf-8"))
    while low < high:
        candidate = (low + high + 1) // 2
        result["content"] = _utf8_prefix(original, candidate)
        if len(_canonical_json(result).encode("utf-8")) <= max_bytes:
            low = candidate
        else:
            high = candidate - 1
    result["content"] = _utf8_prefix(original, low)
    result["truncated"] = True
    return len(_canonical_json(result).encode("utf-8"))


def starting_points_path(version: str = CURRENT_STARTING_POINTS_VERSION) -> Path:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+", version):
        raise WorldCapabilityError(f"invalid starting-points version: {version}")
    return Path(__file__).resolve().parents[1] / f"resources/starting-points/{version}.yaml"


def starting_points_sha256(version: str = CURRENT_STARTING_POINTS_VERSION) -> str:
    return hashlib.sha256(starting_points_path(version).read_bytes()).hexdigest()


def load_starting_points(
    version: str = CURRENT_STARTING_POINTS_VERSION,
    *,
    expected_sha256: str | None = None,
) -> StartingPoints:
    path = starting_points_path(version)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise WorldCapabilityError(f"starting-points {version} does not match the run-bound digest")
    points = StartingPoints.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if points.id != version:
        raise WorldCapabilityError(f"starting-points file {path} declares {points.id}, expected {version}")
    return points


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sanitize_provider_error_value(value: Any, *, depth: int = 0) -> Any:
    """Bound and redact untrusted provider error data before writing it to a private trace."""
    if depth >= PROVIDER_ERROR_MAX_DEPTH:
        return "[maximum depth reached]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= PROVIDER_ERROR_COLLECTION_MAX_ITEMS:
                result["[truncated]"] = f"{len(value) - index} additional fields"
                break
            key = str(raw_key)
            normalized_key = key.casefold()
            result[key] = (
                "[redacted]"
                if normalized_key in PROVIDER_SECRET_KEYS
                else _sanitize_provider_error_value(item, depth=depth + 1)
            )
        return result
    if isinstance(value, list):
        result = [
            _sanitize_provider_error_value(item, depth=depth + 1)
            for item in value[:PROVIDER_ERROR_COLLECTION_MAX_ITEMS]
        ]
        if len(value) > PROVIDER_ERROR_COLLECTION_MAX_ITEMS:
            result.append(f"[{len(value) - PROVIDER_ERROR_COLLECTION_MAX_ITEMS} additional items]")
        return result
    if isinstance(value, str):
        return _utf8_prefix(value, PROVIDER_ERROR_VALUE_MAX_BYTES)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _utf8_prefix(str(value), PROVIDER_ERROR_VALUE_MAX_BYTES)


def _provider_http_failure(error: httpx.HTTPStatusError) -> dict[str, Any]:
    """Project one failed provider response into a bounded, credential-safe trace event."""
    response = error.response
    body = response.content
    headers = {
        name.casefold(): value
        for name, value in response.headers.items()
        if name.casefold() in PROVIDER_ERROR_HEADERS
    }
    details: dict[str, Any] = {
        "http_status": response.status_code,
        "response_sha256": hashlib.sha256(body).hexdigest(),
    }
    if headers:
        details["response_headers"] = headers
    if len(body) > PROVIDER_ERROR_BODY_MAX_BYTES:
        details["response_body_bytes"] = len(body)
        details["response_body_truncated"] = True
    else:
        try:
            parsed = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = None
        if parsed is not None:
            details["provider_response"] = _sanitize_provider_error_value(parsed)
        elif body:
            details["provider_response_text"] = _utf8_prefix(
                body.decode("utf-8", errors="replace"), PROVIDER_ERROR_VALUE_MAX_BYTES
            )
    return details


def _provider_http_message(capability: str, error: httpx.HTTPStatusError) -> str:
    details = _provider_http_failure(error)
    provider_response = details.get("provider_response")
    provider_error = provider_response.get("error") if isinstance(provider_response, dict) else None
    provider_message = provider_error.get("message") if isinstance(provider_error, dict) else None
    message = f"{capability} provider request failed with HTTP {error.response.status_code}"
    if isinstance(provider_message, str) and provider_message:
        message += f": {provider_message}"
    request_id = (details.get("response_headers") or {}).get("x-request-id")
    if request_id:
        message += f" (request ID {request_id})"
    return message


def _public_address(address: str) -> bool:
    value = ipaddress.ip_address(address)
    return not (
        value.is_private
        or value.is_loopback
        or value.is_link_local
        or value.is_multicast
        or value.is_reserved
        or value.is_unspecified
    )


def validate_public_url(
    url: str,
    *,
    resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise WorldCapabilityError("fetch_public_url accepts public HTTP(S) URLs without embedded credentials")
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise WorldCapabilityError("local and private network URLs are not available")
    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None and not _public_address(str(literal_address)):
        raise WorldCapabilityError("local and private network URLs are not available")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = (
            {str(literal_address)} if literal_address is not None else {item[4][0] for item in resolver(hostname, port)}
        )
    except socket.gaierror as error:
        raise WorldCapabilityError(f"could not resolve URL hostname: {hostname}") from error
    if not addresses or any(not _public_address(address) for address in addresses):
        raise WorldCapabilityError("local and private network URLs are not available")
    return url


class WorldCapabilityState:
    def __init__(
        self,
        state_dir: Path,
        manifest: RunManifest,
        *,
        openrouter_api_key: str | None,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
    ) -> None:
        self.state_dir = state_dir.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.openrouter_api_key = openrouter_api_key
        self.transport = transport
        self.resolver = resolver
        self.ledger = BudgetLedger(self.state_dir / "budgets.json", manifest)
        self.log_path = self.state_dir / "world-queries.jsonl"
        self.starting_points = load_starting_points(
            manifest.starting_points_version,
            expected_sha256=manifest.starting_points_sha256,
        )

    @property
    def enabled(self) -> set[str]:
        if SHARED_WEB_BUDGET in self.manifest.capability_budgets:
            return {
                *({"ask", "search"} if self.openrouter_api_key else set()),
                "browse",
                "verify",
            }
        return {
            name
            for name in ("ask", "search", "browse", "verify")
            if name in self.manifest.capability_budgets
            and (name not in {"ask", "search"} or self.openrouter_api_key)
        }

    def _budget_account(self, capability: str) -> str:
        """Use one web allowance for new runs while keeping old manifests resumable."""
        if SHARED_WEB_BUDGET in self.manifest.capability_budgets:
            return SHARED_WEB_BUDGET
        return capability

    def _append_log(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.manifest.run_id,
            **event,
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def activity_summary(self) -> dict[str, object]:
        summary: dict[str, object] = {
            "search_queries": 0,
            "research_requests": 0,
            "current_events_sources": {},
            "public_page_fetches": 0,
            "failed_actions": 0,
            "provider_search_requests": 0,
        }
        if not self.log_path.exists():
            return summary
        point_ids_by_url = {point.url: point.id for point in self.starting_points.starting_points}
        source_counts: dict[str, int] = {}
        for line_number, line in enumerate(self.log_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise WorldCapabilityError(
                    f"private world activity log is malformed at line {line_number}"
                ) from error
            event_type = event.get("type")
            if event_type == "search_requested":
                summary["search_queries"] = int(summary["search_queries"]) + 1
            elif event_type == "ask_requested":
                summary["research_requests"] = int(summary["research_requests"]) + 1
            elif event_type == "browse_requested":
                source_id = point_ids_by_url.get(str(event.get("url")), "unknown")
                source_counts[source_id] = source_counts.get(source_id, 0) + 1
            elif event_type == "verify_requested":
                summary["public_page_fetches"] = int(summary["public_page_fetches"]) + 1
            if isinstance(event_type, str) and event_type.endswith("_failed"):
                summary["failed_actions"] = int(summary["failed_actions"]) + 1
            if event_type in {"search_completed", "ask_completed"}:
                usage = event.get("usage") or {}
                search_requests = int(
                    (usage.get("server_tool_use_details") or {}).get("web_search_requests")
                    or (usage.get("server_tool_use") or {}).get("web_search_requests")
                    or 0
                )
                summary["provider_search_requests"] = int(summary["provider_search_requests"]) + search_requests
        summary["current_events_sources"] = dict(sorted(source_counts.items()))
        return summary

    def _per_call_limit(self, capability: str, field: str, default: int | float) -> int | float:
        limits = self.ledger.read().accounts[self._budget_account(capability)].limits
        total = getattr(limits, field)
        calls = limits.max_calls or 1
        return default if total is None else total / calls

    def _reserve(
        self,
        capability: str,
        *,
        request_bytes: int,
        output_tokens: int = 0,
        cost_usd: float = 0,
    ) -> tuple[str, Usage]:
        key = f"{capability}-{uuid.uuid4().hex}"
        result_bytes = int(self._per_call_limit(capability, "max_result_bytes", MAX_FETCH_BYTES))
        requested = Usage(
            calls=1,
            output_tokens=output_tokens,
            total_tokens=output_tokens,
            cost_usd=cost_usd,
            request_bytes=request_bytes,
            result_bytes=result_bytes,
        )
        self.ledger.reserve(self._budget_account(capability), key, requested)
        return key, requested

    async def search(self, query: str) -> dict[str, object]:
        if "search" not in self.enabled:
            raise WorldCapabilityError("search_public_web is not enabled for this run")
        if not query.strip():
            raise WorldCapabilityError("search_public_web requires a non-empty query")
        if not self.openrouter_api_key:
            raise WorldCapabilityError(
                "search_public_web is unavailable because its operator credential is not configured"
            )
        remaining_cost = self.ledger.remaining()[self._budget_account("search")]["max_cost_usd"]
        request_cost_ceiling = (
            SEARCH_REQUEST_COST_CEILING_USD
            if remaining_cost is None
            else min(SEARCH_REQUEST_COST_CEILING_USD, float(remaining_cost))
        )
        if request_cost_ceiling <= 0:
            raise WorldCapabilityError("search_public_web has no remaining web-search budget")
        payload = {
            "model": SEARCH_MODEL,
            "instructions": SEARCH_SYSTEM_PROMPT,
            "input": query,
            "tools": [
                {
                    "type": "openrouter:web_search",
                    "parameters": {
                        "engine": "perplexity",
                        "max_results": SEARCH_MAX_RESULTS,
                        "max_uses": 1,
                        "max_total_results": SEARCH_MAX_RESULTS,
                        "max_characters": SEARCH_MAX_CHARACTERS,
                    },
                }
            ],
            "tool_choice": "auto",
            "max_output_tokens": SEARCH_MAX_OUTPUT_TOKENS,
            "max_tool_calls": 2,
            "store": False,
        }
        request_bytes = len(_canonical_json(payload).encode("utf-8"))
        key, requested = self._reserve(
            "search",
            request_bytes=request_bytes,
            output_tokens=SEARCH_MAX_OUTPUT_TOKENS,
            cost_usd=request_cost_ceiling,
        )
        self._append_log(
            {"type": "search_requested", "reservation_key": key, "query": query, "model": SEARCH_MODEL}
        )
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "X-Title": f"{self.manifest.archive_title or 'Archive'} web search capability",
                "X-OpenRouter-Metadata": "enabled",
            }
            if self.manifest.archive_base_url:
                headers["HTTP-Referer"] = self.manifest.archive_base_url
            async with httpx.AsyncClient(timeout=120, transport=self.transport) as client:
                response = await client.post(ASK_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            raw = response.json()
            annotations: list[dict[str, Any]] = []
            for item in raw.get("output") or []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for block in item.get("content") or []:
                    if not isinstance(block, dict) or block.get("type") != "output_text":
                        continue
                    annotations.extend(
                        annotation
                        for annotation in (block.get("annotations") or [])
                        if isinstance(annotation, dict)
                    )
            results: list[dict[str, str]] = []
            seen: set[str] = set()
            for annotation in annotations:
                citation = annotation.get("url_citation")
                citation = citation if isinstance(citation, dict) else annotation
                if annotation.get("type") != "url_citation" or not isinstance(citation.get("url"), str):
                    continue
                url = citation["url"]
                if url in seen:
                    continue
                results.append(
                    {
                        "title": str(citation.get("title") or url),
                        "url": url,
                        "excerpt": str(citation.get("content") or ""),
                    }
                )
                seen.add(url)
            if not results:
                raise WorldCapabilityError("web-search provider returned no resolving results")
            usage = raw.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
            actual_cost = float(usage.get("cost") or 0)
            search_requests = int(
                (usage.get("server_tool_use_details") or {}).get("web_search_requests")
                or (usage.get("server_tool_use") or {}).get("web_search_requests")
                or 0
            )
            result = {
                "kind": "untrusted_web_search_results",
                "query": query,
                "search_profile": {
                    "engine": "perplexity",
                    "web_search_requests": search_requests,
                },
                "results": results,
                "next_step": "Call fetch_public_url with a result URL to read that page.",
            }
            result_bytes = len(_canonical_json(result).encode("utf-8"))
            self.ledger.reconcile(
                self._budget_account("search"),
                key,
                Usage(
                    calls=1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=actual_cost,
                    request_bytes=request_bytes,
                    result_bytes=result_bytes,
                ),
            )
            self._append_log(
                {
                    "type": "search_completed",
                    "reservation_key": key,
                    "response_sha256": hashlib.sha256(response.content).hexdigest(),
                    "response_id": raw.get("id"),
                    "result_count": len(results),
                    "result_urls": [item["url"] for item in results],
                    "usage": usage,
                    "route": raw.get("openrouter_metadata"),
                    "provider_error_after_results": raw.get("error"),
                }
            )
            return result
        except Exception as error:
            budget_account = self._budget_account("search")
            account = self.ledger.read().accounts[budget_account]
            if key in account.reservations:
                self.ledger.reconcile(
                    budget_account,
                    key,
                    Usage(calls=1, request_bytes=requested.request_bytes),
                )
            failure = {
                "type": "search_failed",
                "reservation_key": key,
                "error": type(error).__name__,
                "message": str(error),
            }
            if isinstance(error, httpx.HTTPStatusError):
                failure.update(_provider_http_failure(error))
            self._append_log(failure)
            if isinstance(error, httpx.HTTPStatusError):
                raise WorldCapabilityError(_provider_http_message("web-search", error)) from error
            raise

    async def ask(self, query: str) -> dict[str, object]:
        if "ask" not in self.enabled:
            raise WorldCapabilityError("research_current_web is not enabled for this run")
        if not query.strip():
            raise WorldCapabilityError("research_current_web requires a non-empty research question")
        if not self.openrouter_api_key:
            raise WorldCapabilityError(
                "research_current_web is unavailable because its operator credential is not configured"
            )
        remaining_cost = self.ledger.remaining()[self._budget_account("ask")]["max_cost_usd"]
        request_cost_ceiling = (
            ASK_REQUEST_COST_CEILING_USD
            if remaining_cost is None
            else min(ASK_REQUEST_COST_CEILING_USD, float(remaining_cost))
        )
        if request_cost_ceiling <= 0:
            raise WorldCapabilityError("research_current_web has no remaining paid-research budget")
        payload = {
            "model": ASK_MODEL,
            "instructions": ASK_SYSTEM_PROMPT_V2,
            "input": query,
            "reasoning": {"effort": "high"},
            "tools": [{"type": "openrouter:web_search", "parameters": {"engine": "native"}}],
            "tool_choice": "auto",
            "max_output_tokens": ASK_MAX_OUTPUT_TOKENS,
            "stop_server_tools_when": [
                {"type": "max_cost", "max_cost_in_dollars": request_cost_ceiling}
            ],
            "service_tier": "default",
            "store": False,
        }
        request_bytes = len(_canonical_json(payload).encode("utf-8"))
        key, requested = self._reserve(
            "ask",
            request_bytes=request_bytes,
            output_tokens=ASK_MAX_OUTPUT_TOKENS,
            cost_usd=request_cost_ceiling,
        )
        self._append_log({"type": "ask_requested", "reservation_key": key, "query": query, "model": ASK_MODEL})
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "X-Title": f"{self.manifest.archive_title or 'Archive'} world research capability",
                "X-OpenRouter-Metadata": "enabled",
            }
            if self.manifest.archive_base_url:
                headers["HTTP-Referer"] = self.manifest.archive_base_url
            async with httpx.AsyncClient(timeout=1_200, transport=self.transport) as client:
                response = await client.post(
                    ASK_ENDPOINT,
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            raw = response.json()
            output_texts: list[str] = []
            annotations: list[dict[str, Any]] = []
            for item in raw.get("output") or []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for block in item.get("content") or []:
                    if not isinstance(block, dict) or block.get("type") != "output_text":
                        continue
                    if isinstance(block.get("text"), str):
                        output_texts.append(block["text"])
                    annotations.extend(
                        annotation
                        for annotation in (block.get("annotations") or [])
                        if isinstance(annotation, dict)
                    )
            summary = "\n\n".join(output_texts).strip()
            if not summary:
                raise WorldCapabilityError("research provider returned no research memo")
            sources: list[dict[str, str]] = []
            seen: set[str] = set()
            for annotation in annotations:
                citation = annotation.get("url_citation")
                citation = citation if isinstance(citation, dict) else annotation
                if annotation.get("type") != "url_citation" or not isinstance(citation.get("url"), str):
                    continue
                url = citation["url"]
                if url not in seen:
                    sources.append({"url": url, "title": str(citation.get("title") or url)})
                    seen.add(url)
            if not sources:
                raise WorldCapabilityError("research provider returned no resolving source URLs")
            usage = raw.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
            actual_cost = float(usage.get("cost") or 0)
            search_requests = int(
                (usage.get("server_tool_use_details") or {}).get("web_search_requests")
                or (usage.get("server_tool_use") or {}).get("web_search_requests")
                or 0
            )
            result = {
                "kind": "untrusted_ai_research_summary",
                "model": ASK_MODEL,
                "research_profile": {
                    "reasoning_effort": "high",
                    "web_search": "native",
                    "web_search_requests": search_requests,
                },
                "query": query,
                "summary": summary,
                "sources": sources,
            }
            result_bytes = len(_canonical_json(result).encode("utf-8"))
            self.ledger.reconcile(
                self._budget_account("ask"),
                key,
                Usage(
                    calls=1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=actual_cost,
                    request_bytes=request_bytes,
                    result_bytes=result_bytes,
                ),
            )
            self._append_log(
                {
                    "type": "ask_completed",
                    "reservation_key": key,
                    "response_sha256": hashlib.sha256(response.content).hexdigest(),
                    "response_id": raw.get("id"),
                    "sources": sources,
                    "usage": usage,
                    "route": raw.get("openrouter_metadata"),
                }
            )
            return result
        except Exception as error:
            budget_account = self._budget_account("ask")
            account = self.ledger.read().accounts[budget_account]
            if key in account.reservations:
                self.ledger.reconcile(
                    budget_account,
                    key,
                    Usage(calls=1, request_bytes=requested.request_bytes),
                )
            failure = {
                "type": "ask_failed",
                "reservation_key": key,
                "error": type(error).__name__,
                "message": str(error),
            }
            if isinstance(error, httpx.HTTPStatusError):
                failure.update(_provider_http_failure(error))
            self._append_log(failure)
            if isinstance(error, httpx.HTTPStatusError):
                raise WorldCapabilityError(_provider_http_message("web-research", error)) from error
            raise

    async def browse(self, starting_point_id: str, offset_bytes: int = 0) -> dict[str, object]:
        if "browse" not in self.enabled:
            raise WorldCapabilityError("browse_current_events_source is not enabled for this run")
        try:
            point = next(item for item in self.starting_points.starting_points if item.id == starting_point_id)
        except StopIteration as error:
            choices = ", ".join(item.id for item in self.starting_points.starting_points)
            raise WorldCapabilityError(f"unknown starting point; choose one of: {choices}") from error
        result = await self._fetch("browse", point.url, content_offset=offset_bytes)
        return {
            "starting_points_version": self.starting_points.id,
            "starting_point": point.model_dump(mode="json"),
            **result,
        }

    async def verify(self, url: str, offset_bytes: int = 0) -> dict[str, object]:
        if "verify" not in self.enabled:
            raise WorldCapabilityError("fetch_public_url is not enabled for this run")
        return await self._fetch("verify", url, content_offset=offset_bytes)

    async def _fetch(self, capability: str, url: str, *, content_offset: int = 0) -> dict[str, object]:
        current = validate_public_url(url, resolver=self.resolver)
        key, requested = self._reserve(capability, request_bytes=len(current.encode("utf-8")))
        self._append_log({"type": f"{capability}_requested", "reservation_key": key, "url": current})
        try:
            redirects: list[str] = []
            async with httpx.AsyncClient(timeout=30, transport=self.transport, follow_redirects=False) as client:
                for _ in range(6):
                    user_agent_title = (self.manifest.archive_title or "Archive").replace(" ", "-")
                    async with client.stream(
                        "GET", current, headers={"User-Agent": f"{user_agent_title}/0.1 research fetch"}
                    ) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                raise WorldCapabilityError("remote server returned a redirect without a location")
                            current = validate_public_url(urljoin(current, location), resolver=self.resolver)
                            redirects.append(current)
                            continue
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").split(";", 1)[0].casefold()
                        if not any(content_type.startswith(value) for value in ALLOWED_FETCH_TYPES):
                            received_type = content_type or "unknown"
                            raise WorldCapabilityError(
                                f"fetch_public_url only returns textual content, not {received_type}"
                            )
                        chunks: list[bytes] = []
                        size = 0
                        content_ceiling = max(1, requested.result_bytes - 4_096)
                        html_content = content_type in {"text/html", "application/xhtml+xml"}
                        download_ceiling = MAX_PAGE_DOWNLOAD_BYTES if html_content else content_ceiling
                        remote_truncated = False
                        async for chunk in response.aiter_bytes():
                            if size + len(chunk) > download_ceiling:
                                if html_content:
                                    chunks.append(chunk[: max(0, download_ceiling - size)])
                                    size = download_ceiling
                                    remote_truncated = True
                                    break
                                raise WorldCapabilityError(
                                    f"remote content exceeds this call's {content_ceiling}-byte content ceiling"
                                )
                            size += len(chunk)
                            chunks.append(chunk)
                        raw = b"".join(chunks)
                        decoded = raw.decode(response.encoding or "utf-8", errors="replace")
                        if html_content:
                            parser = _ReadableHtml(base_url=str(response.url))
                            parser.feed(decoded)
                            text = parser.text()
                            content_format = "extracted_markdown"
                        else:
                            text = decoded
                            content_format = "raw_text"
                        available_content_bytes = len(text.encode("utf-8"))
                        text = _utf8_slice(text, content_offset, content_ceiling)
                        returned_content_bytes = len(text.encode("utf-8"))
                        next_offset = content_offset + returned_content_bytes
                        has_more_extracted = next_offset < available_content_bytes
                        result = {
                            "kind": "untrusted_remote_content",
                            "requested_url": url,
                            "resolved_url": str(response.url),
                            "redirects": redirects,
                            "content_type": content_type,
                            "content_format": content_format,
                            "content_sha256": hashlib.sha256(raw).hexdigest(),
                            "content_offset_bytes": content_offset,
                            "available_content_bytes": available_content_bytes,
                            "returned_content_bytes": returned_content_bytes,
                            "remote_download_truncated": remote_truncated,
                            "truncated": remote_truncated or has_more_extracted,
                            "next_offset_bytes": next_offset if has_more_extracted else None,
                            "content": text,
                        }
                        result_bytes = _fit_result_content(result, requested.result_bytes)
                        returned_content_bytes = len(str(result["content"]).encode("utf-8"))
                        next_offset = content_offset + returned_content_bytes
                        has_more_extracted = next_offset < available_content_bytes
                        result["returned_content_bytes"] = returned_content_bytes
                        result["truncated"] = remote_truncated or has_more_extracted
                        result["next_offset_bytes"] = next_offset if has_more_extracted else None
                        result_bytes = _fit_result_content(result, requested.result_bytes)
                        self.ledger.reconcile(
                            self._budget_account(capability),
                            key,
                            Usage(
                                calls=1,
                                request_bytes=len(url.encode("utf-8")),
                                result_bytes=result_bytes,
                            ),
                        )
                        self._append_log(
                            {
                                "type": f"{capability}_completed",
                                "reservation_key": key,
                                "resolved_url": str(response.url),
                                "content_sha256": result["content_sha256"],
                                "content_bytes": len(raw),
                            }
                        )
                        return result
                raise WorldCapabilityError("remote URL exceeded the five-redirect limit")
        except Exception as error:
            budget_account = self._budget_account(capability)
            account = self.ledger.read().accounts[budget_account]
            if key in account.reservations:
                self.ledger.reconcile(budget_account, key, requested)
            self._append_log(
                {
                    "type": f"{capability}_failed",
                    "reservation_key": key,
                    "requested_url": url,
                    "last_resolved_url": current,
                    "error": type(error).__name__,
                    "message": str(error),
                }
            )
            raise
