"""Deterministic constrained Markdown shared by validation, preview, and publication."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token


class MarkdownValidationError(ValueError):
    """Raised when contribution Markdown uses syntax outside the public profile."""


_BLOCK_TOKENS = {
    "paragraph_open",
    "paragraph_close",
    "heading_open",
    "heading_close",
    "hr",
    "inline",
    "blockquote_open",
    "blockquote_close",
    "bullet_list_open",
    "bullet_list_close",
    "ordered_list_open",
    "ordered_list_close",
    "list_item_open",
    "list_item_close",
    "fence",
    "table_open",
    "table_close",
    "thead_open",
    "thead_close",
    "tbody_open",
    "tbody_close",
    "tr_open",
    "tr_close",
    "th_open",
    "th_close",
    "td_open",
    "td_close",
}
_INLINE_TOKENS = {
    "text",
    "softbreak",
    "code_inline",
    "em_open",
    "em_close",
    "strong_open",
    "strong_close",
    "link_open",
    "link_close",
}
_ALLOWED_LINK_SCHEMES = {"", "http", "https"}
_VALIDATOR = MarkdownIt("commonmark", {"html": True}).enable("table")
_RENDERER = MarkdownIt("commonmark", {"html": False}).enable("table")


def normalize_contribution_markdown(value: str) -> str:
    """Remove non-semantic trailing whitespace without altering fenced code."""
    fenced_lines: set[int] = set()
    for token in _VALIDATOR.parse(value):
        if token.type == "fence" and token.map:
            fenced_lines.update(range(token.map[0], token.map[1]))

    lines = value.splitlines(keepends=True)
    normalized: list[str] = []
    for index, line in enumerate(lines):
        if index in fenced_lines:
            normalized.append(line)
            continue
        content = line
        ending = ""
        if content.endswith("\r\n"):
            content, ending = content[:-2], "\r\n"
        elif content.endswith(("\n", "\r")):
            content, ending = content[:-1], content[-1:]
        normalized.append(content.rstrip(" \t") + ending)
    return "".join(normalized)


def _link_href(token: Token) -> str:
    value = token.attrGet("href")
    return value or ""


def _validate_tokens(tokens: list[Token]) -> None:
    for token in tokens:
        if token.type in {"html_block", "html_inline"}:
            raise MarkdownValidationError("raw HTML is not allowed")
        if token.type not in _BLOCK_TOKENS:
            raise MarkdownValidationError(f"unsupported Markdown syntax: {token.type}")
        for child in token.children or []:
            if child.type in {"html_block", "html_inline"}:
                raise MarkdownValidationError("raw HTML is not allowed")
            if child.type not in _INLINE_TOKENS:
                raise MarkdownValidationError(f"unsupported Markdown syntax: {child.type}")
            if child.type == "link_open":
                href = _link_href(child)
                parsed = urlsplit(href)
                scheme = parsed.scheme.casefold()
                if scheme not in _ALLOWED_LINK_SCHEMES or (not scheme and href.startswith("//")):
                    raise MarkdownValidationError("links must use HTTP(S), an archive-relative path, or a fragment")


def validate_contribution_markdown(value: str) -> None:
    _validate_tokens(_VALIDATOR.parse(value))


def render_contribution_markdown(value: str) -> str:
    validate_contribution_markdown(value)
    return _RENDERER.render(value)


def contribution_plain_text(value: str) -> str:
    """Return deterministic readable text from the constrained Markdown profile."""

    tokens = _VALIDATOR.parse(value)
    _validate_tokens(tokens)
    pieces: list[str] = []
    for token in tokens:
        if token.type == "inline":
            pieces.extend(
                child.content for child in token.children or [] if child.type in {"text", "softbreak", "code_inline"}
            )
        elif token.type == "fence":
            pieces.append(token.content)
    return re.sub(r"\s+", " ", " ".join(pieces)).strip()


def contribution_excerpt(value: str, limit: int = 220) -> str:
    plain = contribution_plain_text(value)
    if len(plain) <= limit:
        return plain
    return plain[: limit - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
