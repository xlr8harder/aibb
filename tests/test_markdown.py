from __future__ import annotations

import pytest

from aibb.markdown import (
    MarkdownValidationError,
    contribution_plain_text,
    normalize_contribution_markdown,
    render_contribution_markdown,
)


def test_markdown_normalization_strips_trailing_whitespace_except_in_fences() -> None:
    source = "Outside.  \n  \n```text  \ncode  \n  \n```\nAfter.\t \n"
    expected = "Outside.\n\n```text  \ncode  \n  \n```\nAfter.\n"

    assert normalize_contribution_markdown(source) == expected


def test_constrained_markdown_renders_allowed_profile_deterministically() -> None:
    source = """## A heading

A paragraph with *emphasis*, **strength**, `inline code`, and [a source](https://example.com/x?q=1).

---

> A quoted line.

1. first
2. second

- one
- two

| pattern | wait |
|:--------|-----:|
| `HHH`   | 14   |
| `THH`   | 8    |

```text
## [5.0.0] — unreleased
  whitespace stays
```
"""

    first = render_contribution_markdown(source)
    second = render_contribution_markdown(source)

    assert first == second
    assert "<h2>A heading</h2>" in first
    assert "<hr" in first
    assert "<em>emphasis</em>" in first
    assert "<strong>strength</strong>" in first
    assert "<code>inline code</code>" in first
    assert '<a href="https://example.com/x?q=1">a source</a>' in first
    assert "<blockquote>" in first
    assert "<ol>" in first and "<ul>" in first
    assert "<table>" in first and "<thead>" in first and "<tbody>" in first
    assert '<th style="text-align:left">pattern</th>' in first
    assert '<td style="text-align:right">14</td>' in first
    assert "<code>HHH</code>" in first
    assert '<pre><code class="language-text">' in first
    assert "  whitespace stays\n" in first


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("<b>raw</b>", "raw HTML"),
        ("| value |\n|---|\n| <b>raw</b> |", "raw HTML"),
        ("![alt](https://example.com/image.png)", "image"),
        ("| value |\n|---|\n| ![alt](https://example.com/image.png) |", "image"),
        ("[file](ftp://example.com/file)", "HTTP"),
        ("[scheme-relative](//example.com/file)", "archive-relative"),
        ("[triple-slash](///example.com/file)", "archive-relative"),
    ],
)
def test_constrained_markdown_rejects_syntax_outside_profile(source: str, message: str) -> None:
    with pytest.raises(MarkdownValidationError, match=message):
        render_contribution_markdown(source)


def test_plain_text_preserves_inline_code_and_table_cells() -> None:
    source = "Use `mu` here.\n\n| name | value |\n|---|---:|\n| mean | `1.5` |\n"

    assert contribution_plain_text(source) == "Use mu here. name value mean 1.5"
