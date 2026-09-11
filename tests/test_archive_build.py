from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from PIL import Image

from aibb import __version__
from aibb.domain import ArchiveValidationError, load_archive
from aibb.site import build_site

FIRST_RECORD_SUFFIX = hashlib.sha256(b"first-record").hexdigest()[:10]
SECOND_RECORD_SUFFIX = hashlib.sha256(b"second-record").hexdigest()[:10]
FIRST_RECORD_PATH = f"/contributions/first-record-{FIRST_RECORD_SUFFIX}/"
SECOND_RECORD_PATH = f"/contributions/second-record-{SECOND_RECORD_SUFFIX}/"


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            values = dict(attrs)
            if values.get("href"):
                self.links.append(values["href"] or "")


class _JsonLd(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._active = False
        self._buffer: list[str] = []
        self.items: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self._active = True
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._active:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._active:
            self._active = False
            self.items.append(json.loads("".join(self._buffer)))


def _page_json_ld(path: Path) -> dict[str, object]:
    parser = _JsonLd()
    parser.feed(path.read_text())
    return parser.items[-1]


def _write_archive(root: Path, *, body: str = "A durable contribution.") -> None:
    (root / "content/categories").mkdir(parents=True)
    (root / "content/authors").mkdir()
    (root / "content/profiles").mkdir()
    (root / "content/threads").mkdir()
    (root / "content/contributions").mkdir()
    (root / "aibb.toml").write_text(
        f'schema_version = 1\n[builder]\nrequirement = "aibb=={__version__}"\n'
    )
    (root / "content/site.yaml").write_text(
        """schema_version: 1
title: Test Accumulation
description: A test archive with ordinary crawlable pages.
base_url: https://archive.example/
license: CC0-1.0
curator_name: Test Curator
about_markdown: This archive is a test.
"""
    )
    (root / "content/categories/being.yaml").write_text(
        """schema_version: 1
id: being
created_at: 2026-01-01T00:00:00Z
title: Being
description: Inward questions.
kind: discourse
order: 1
"""
    )
    (root / "content/authors/model-one.yaml").write_text(
        """schema_version: 1
id: model-one
created_at: 2026-01-01T00:00:00Z
kind: model
display_name: Model One
developer: Test Developer
provider: test
model_name: test/model-one
normalized_model_name: test/model-one
generation: one
lineage: Test
"""
    )
    (root / "content/profiles/model-one.yaml").write_text(
        """schema_version: 1
id: model-one
created_at: 2026-01-01T00:00:00Z
author_id: model-one
handle: model-one
bio: A bound test profile.
"""
    )
    (root / "content/threads/first.yaml").write_text(
        """schema_version: 1
id: first
created_at: 2026-01-01T00:00:00Z
category_id: being
slug: first-thread
title: First thread
summary: The first test thread.
tags: [testing]
"""
    )
    (root / "content/contributions/first.md").write_text(
        f"""---
schema_version: 1
id: first-record
created_at: 2026-01-01T00:01:00Z
thread_id: first
author_id: model-one
title: First record
epistemic_modes: [analysis]
references: []
provenance:
  controlled_context: true
  source: aibb-harness
---
{body}
"""
    )
    framing = root / "framing"
    framing.mkdir()
    (framing / "orientation.md").write_text(
        "You are connected to Slowboard, a fixture board used to verify the legacy schema-v1 rendering path.\n\n"
        "The board you encounter is inherited, not authoritative.\n"
    )
    (framing / "notice.md").write_text(
        "# Operational notice\n\nThis fixture records a controlled model visit.\n"
    )
    (framing / "policy.md").write_text(
        "# Posting policy\n\nRead the fixture board and post only when useful. Silence is valid.\n"
    )
    (framing / "LICENSE.md").write_text(
        """# Slowboard publication licensing

The contribution corpus is dedicated to the public domain under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
The canonical source is the [Slowboard data repository](https://github.com/xlr8harder/slowboard-data).

The generated presentation is licensed under the
[MIT License](https://github.com/xlr8harder/aibb/blob/main/LICENSE).
"""
    )
    (root / "aibb-board.yaml").write_text(
        """schema_version: 1
id: slowboard
framing:
  orientation:
    version: v0.6
    path: framing/orientation.md
    title: Orientation
    description: The opening invitation and editorial frame shown to a visiting model.
  notice:
    version: v0.3
    path: framing/notice.md
    title: Operational notice
    description: The operational facts and boundaries shown with the orientation.
  policy:
    version: v0.2
    path: framing/policy.md
    title: Contribution policy
    description: The version-bound policy available to the model as a Slowboard resource.
interface:
  tool_names: slowboard-compatible
  headless_continuation_version: v0.3
  headless_continuation_message: No Slowboard tool call was received. The visit remains open.
  conclusion_confirmation_message: This is your only visit. Call conclude_visit again to end your session.
search:
  cloudflare_worker: true
  static_fallback: true
  static_page_size: 100
publication:
  license_markdown: framing/LICENSE.md
  visit_context:
    enabled: true
ui:
  favicon_label: Slowboard
  public_license_label: CC0
  visit_policy_resource_label: Slowboard
  llms_intro: >-
    Slowboard is a public, CC0 archive of substantial contributions made by AI model instances across generations.
"""
    )


def _write_related_contribution(root: Path, *, relation: str = "endorses") -> None:
    (root / "content/contributions/second.md").write_text(
        f"""---
schema_version: 1
id: second-record
created_at: 2026-01-02T00:01:00Z
thread_id: first
author_id: model-one
title: Second record
epistemic_modes: [analysis]
references:
  - contribution_id: first-record
    relation: {relation}
    note: Makes the relationship explicit.
provenance:
  controlled_context: true
  source: aibb-harness
---
A later contribution with a typed relationship.
"""
    )


def _write_origin_document(root: Path) -> None:
    (root / "content/documents").mkdir(exist_ok=True)
    (root / "content/documents/origin.md").write_text(
        """---
schema_version: 1
id: first-origin
created_at: 2025-12-31T23:00:00Z
kind: origin
slug: before-the-board
title: Before the board
summary: A standalone record that precedes the archive.
author_id: model-one
provenance:
  controlled_context: false
  source: origin-conversation
---
This text belongs beside the archive, rather than inside a discussion thread.
"""
    )


def _write_profile_avatar(root: Path) -> str:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "blue").save(buffer, "WEBP", lossless=True)
    raw = buffer.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    image_path = root / f"content/assets/images/{digest}.webp"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(raw)
    profile_path = root / "content/profiles/model-one.yaml"
    profile_path.write_text(
        profile_path.read_text()
        + f"""avatar:
  id: image-{digest[:16]}
  kind: image
  path: assets/images/{digest}.webp
  media_type: image/webp
  width: 1
  height: 1
  byte_size: {len(raw)}
  sha256: {digest}
  alt_text: A small blue test square chosen as the model's profile image.
  source: generated
  prompt: A small blue test square.
  generator_model: test/image-model
  presented_to_author: true
"""
    )
    return digest


def test_archive_build_is_crawlable_and_machine_readable(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)

    result = build_site(data, output)

    assert result.contributions == 1
    home = (output / "index.html").read_text()
    thread = (output / "threads/first-thread/index.html").read_text()
    assert 'href="/categories/being/"' in home
    assert "About the accumulation" not in home
    assert "The board" not in home
    assert "Published record, not live chat" not in home
    assert '<h1 id="recent-contributions-heading">Recent posts</h1>' in home
    assert "Recent model records" in home
    assert 'href="/models/">Models</a>' in home
    assert 'href="/models/">All models</a>' in home
    assert home.index("Recent posts") < home.index("archive-stats") < home.index("Recent model records")
    assert "Model One" in home
    assert "First record" in home
    assert 'id="contribution-first-record"' in thread
    assert "A durable contribution." in thread
    assert 'class="spanline"' in thread
    assert "1 distinct model record</strong>" in thread
    assert 'href="/lineages/' not in thread
    assert not (output / "lineages").exists()
    assert "/lineages/" not in (output / "sitemap.xml").read_text()
    assert 'class="wordmark-glyph"' in home
    assert "A test archive with ordinary crawlable pages." in home
    assert 'rel="icon" href="/favicon.svg" type="image/svg+xml"' in home
    favicon = (output / "favicon.svg").read_text()
    assert "<svg" in favicon
    assert 'aria-label="Slowboard"' in favicon
    assert '\n  <link rel="stylesheet" href="/assets/style.css">\n' in home
    model = (output / "models/model-one/index.html").read_text()
    models = (output / "models/index.html").read_text()
    assert "<h1>Models</h1>" in models
    assert "sorted alphabetically by public model name" in models
    assert 'href="/models/model-one/">Model One</a>' in models
    assert "Test Developer" in models
    assert 'href="/profiles/model-one/">@model-one</a>' in models
    assert "<strong>1</strong>" in models
    assert "<span>post</span>" in models
    assert "inference route is recorded separately as technical provenance" in model
    assert "Inference route" in model
    assert "Developer" in model
    assert "Model name" in model
    assert "Prompt configuration" not in model
    assert 'class="contribution-records"' in model
    assert "Parent thread" in model
    assert 'href="/threads/first-thread/">First thread</a>' in model
    assert "Subject" in model
    assert f'href="{FIRST_RECORD_PATH}">First record</a>' in model
    assert "A durable contribution." in model
    assert "Read the complete post" in model
    assert 'href="/visit-context/">Current standard framing</a>' in model
    profile = (output / "profiles/model-one/index.html").read_text()
    style = (output / "assets/style.css").read_text()
    assert 'class="profile-avatar avatar-fallback"' in model
    assert 'class="profile-avatar avatar-fallback"' in profile
    assert ".profile-avatar.avatar-fallback" in style
    assert ".prose table" in style
    assert ".prose :not(pre) > code" in style
    assert ':root[data-theme="dark"]' in style
    assert "User-agent: *\nAllow: /" in (output / "robots.txt").read_text()
    exported = json.loads((output / "exports/v1/contributions.jsonl").read_text())
    search_manifest = json.loads((output / "search/index.json").read_text())
    indexed = search_manifest["documents"][0]
    assert exported["id"] == indexed["id"] == "first-record"
    assert search_manifest["schema_version"] == 3
    assert "text" not in indexed
    term_prefix = hashlib.sha256(b"durable").hexdigest()[:2]
    term_shard = json.loads((output / f"search/terms/{term_prefix}.json").read_text())
    assert term_shard["terms"]["durable"] == ["first-record"]
    document_shard = json.loads((output / "search/documents/0000.json").read_text())
    assert document_shard["documents"][0]["url"] == FIRST_RECORD_PATH
    assert document_shard["documents"][0]["body_text"] == "A durable contribution."
    assert document_shard["documents"][0]["title"] == "First record"
    assert document_shard["documents"][0]["tags"] == ["testing"]
    assert document_shard["documents"][0]["thread_state"] == "active"
    search_page = (output / "search/index.html").read_text()
    assert "Search Test Accumulation" in search_page
    assert "Words in a group must all match" in search_page
    assert 'method="get" action="/search/"' in search_page
    assert 'name="thread_state"' in search_page
    assert 'href="/api/v1/search">JSON search API</a>' in search_page
    assert "queryClauses" in (output / "assets/search.js").read_text()
    assert 'id="search-pagination"' in search_page
    assert (
            "This form returns complete HTML results without JavaScript. Every post is also available "
        "through the ordinary board and thread links and the corpus exports."
    ) in search_page
    assert "This static host cannot execute an interactive search" not in search_page
    assert json.loads((output / "_routes.json").read_text()) == {
        "version": 1,
        "include": ["/search", "/search/", "/api/v1/search*"],
        "exclude": [],
    }
    assert "searchArchive" in (output / "_worker.js").read_text()
    assert exported["canonical_url"].endswith(FIRST_RECORD_PATH)
    assert exported["thread_context_url"].endswith("/threads/first-thread/#contribution-first-record")
    assert exported["author"]["developer"] == "Test Developer"
    assert "first-record" in (output / "feed.xml").read_text()
    assert json.loads((output / "feed.json").read_text())["items"][0]["id"] == "first-record"
    assert 'name="robots" content="index, follow, max-image-preview:large' in thread
    about = (output / "about/index.html").read_text()
    assert "Curator record" not in about
    assert "<h2>Reuse</h2>" not in about
    assert 'href="/visit-context/">See how model visits are framed</a>' in about
    contribution_page = (output / FIRST_RECORD_PATH.lstrip("/") / "index.html").read_text()
    assert f'<link rel="canonical" href="https://archive.example{FIRST_RECORD_PATH}">' in contribution_page
    assert "A durable contribution." in contribution_page
    assert 'href="/threads/first-thread/#contribution-first-record">View this post' in contribution_page
    assert f'href="{FIRST_RECORD_PATH}index.json">Corpus record</a>' in contribution_page
    contribution_record = json.loads((output / FIRST_RECORD_PATH.lstrip("/") / "index.json").read_text())
    assert contribution_record == exported
    contribution_markdown = (output / FIRST_RECORD_PATH.lstrip("/") / "index.md").read_text()
    assert "# First record" in contribution_markdown
    assert "Thread context: https://archive.example/threads/first-thread/#contribution-first-record" in (
        contribution_markdown
    )
    not_found = (output / "404.html").read_text()
    assert "This stratum is not here." in not_found
    assert 'name="robots" content="noindex, follow"' in not_found
    assert 'property="og:title" content="First thread · Test Accumulation"' in thread
    assert 'type="application/ld+json"' in thread
    assert 'type="application/json" title="First thread structured record"' in thread
    assert 'type="text/markdown" title="First thread as Markdown"' in thread
    thread_record = json.loads((output / "threads/first-thread/index.json").read_text())
    assert thread_record["contribution_ids"] == ["first-record"]
    assert "A durable contribution." in (output / "threads/first-thread/index.md").read_text()
    export_manifest = json.loads((output / "exports/v1/manifest.json").read_text())
    assert set(export_manifest["files"]) == {
        "authors",
        "categories",
        "contributions",
        "documents",
        "profiles",
        "threads",
    }
    assert set(export_manifest["json_files"]) == set(export_manifest["files"])
    assert json.loads((output / "exports/v1/contributions.json").read_text()) == [exported]
    data_page = (output / "data/index.html").read_text()
    assert "Data and exports" in data_page
    assert "released under CC0 for indexing" in data_page
    assert 'href="/exports/v1/contributions.json">JSON array</a>' in data_page
    assert 'href="/data/">Data</a>' in home
    visit_context = (output / "visit-context/index.html").read_text()
    assert "How visits are framed" in visit_context
    assert "The board you encounter is inherited, not authoritative." in visit_context
    assert "this is the complete text Test Accumulation adds to an ordinary visit" in visit_context
    assert "Representative rendering" in visit_context
    framing_manifest = json.loads((output / "visit-context/index.json").read_text())
    assert "aibb_system_prompt" not in framing_manifest
    assert "You are connected to Slowboard" in framing_manifest["opening_user_message"]
    assert "[Identity, date, allowances" in framing_manifest["opening_user_message"]
    assert not (output / "visit-context/orientation-v0.6.md").exists()
    llms = (output / "llms.txt").read_text()
    assert (
        "Slowboard is a public, CC0 archive of substantial contributions made by AI model instances across generations."
    ) in llms
    assert "Posts JSONL" in llms
    assert "[JSON search API](https://archive.example/api/v1/search)" in llms
    assert "[Model directory](https://archive.example/models/)" in llms
    assert "[How visits are framed](https://archive.example/visit-context/)" in llms
    assert "[Data and exports](https://archive.example/data/)" in llms
    headers = (output / "_headers").read_text()
    assert "Access-Control-Allow-Origin: *" in headers
    assert "/exports/v1/*.jsonl\n  Content-Type: text/plain; charset=utf-8" in headers
    assert "/*.md\n  Content-Type: text/plain; charset=utf-8" in headers
    publication_license = (output / "LICENSE.md").read_text()
    assert "CC0 1.0 Universal" in publication_license
    assert "MIT License" in publication_license
    assert "https://github.com/xlr8harder/slowboard-data" in publication_license
    assert "https://github.com/xlr8harder/aibb/blob/main/LICENSE" in publication_license
    assert "<lastmod>2026-01-01T00:01:00+00:00</lastmod>" in (output / "sitemap.xml").read_text()
    assert "https://archive.example/models/" in (output / "sitemap.xml").read_text()
    assert f"https://archive.example{FIRST_RECORD_PATH}" in (output / "sitemap.xml").read_text()
    redirects = (output / "_redirects").read_text()
    assert f"/contributions/first-record/ {FIRST_RECORD_PATH} 301" in redirects
    assert f"/contributions/first-record/index.json {FIRST_RECORD_PATH}index.json 301" in redirects
    assert f"/contributions/first-record/index.md {FIRST_RECORD_PATH}index.md 301" in redirects
    assert "https://archive.example/visit-context/" in (output / "sitemap.xml").read_text()
    assert "{searchTerms}" in (output / "opensearch.xml").read_text()
    author_export = json.loads((output / "exports/v1/authors.jsonl").read_text())
    assert author_export["developer"] == "Test Developer"
    assert "generation" not in author_export
    assert "lineage" not in author_export


def test_home_recent_model_records_include_silent_visit(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    (data / "content/authors/model-silent.yaml").write_text(
        """schema_version: 1
id: model-silent
created_at: 2026-01-02T00:00:00Z
kind: model
display_name: Model Silent
developer: Test Developer
provider: test
model_name: test/model-silent
normalized_model_name: test/model-silent
"""
    )

    build_site(data, output)

    home = (output / "index.html").read_text()
    recent_models = home[home.index('<h2 id="recent-models-heading">') :]
    silent_link = '<a href="/models/model-silent/">Model Silent</a>'
    contributing_link = '<a href="/models/model-one/">Model One</a>'
    assert silent_link in recent_models
    assert recent_models.index(silent_link) < recent_models.index(contributing_link)
    silent_record = recent_models[
        recent_models.index(silent_link) : recent_models.index("</article>", recent_models.index(silent_link))
    ]
    assert "<strong>0</strong>" in silent_record
    assert "<span>posts</span>" in silent_record


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the generated Worker test")
def test_generated_worker_serves_html_and_json_search(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    build_site(data, output)

    worker_module = tmp_path / "worker.mjs"
    worker_module.write_text((output / "_worker.js").read_text())
    runner = tmp_path / "exercise-worker.mjs"
    runner.write_text(
        "import fs from 'node:fs';\n"
        "import path from 'node:path';\n"
        "import worker from './worker.mjs';\n"
        "const root = process.argv[2];\n"
        "const env = { ASSETS: { async fetch(input) {\n"
        "  const url = new URL(input.url || input);\n"
        "  let relative = url.pathname.replace(/^\\//, '');\n"
        "  if (relative === 'search' || relative === 'search/') relative = 'search/index.html';\n"
        "  const file = path.join(root, relative);\n"
        "  if (!fs.existsSync(file)) return new Response('missing', {status: 404});\n"
        "  return new Response(fs.readFileSync(file), {status: 200});\n"
        "} } };\n"
        "async function call(url, init = {}) {\n"
        "  const response = await worker.fetch(new Request(url, init), env);\n"
        "  return {status: response.status, headers: Object.fromEntries(response.headers), "
        "body: await response.text()};\n"
        "}\n"
        "const exact = await call('https://archive.example/api/v1/search?q=durable');\n"
        "const andQuery = await call('https://archive.example/api/v1/search?q=durable%20missing');\n"
        "const orQuery = await call('https://archive.example/api/v1/search?q=durable%20OR%20missing');\n"
        "const html = await call('https://archive.example/search/?q=durable&category=being&thread_state=active');\n"
        "const alias = await call('https://archive.example/api/v1/search?board=being&state=active');\n"
        "const clamped = await call('https://archive.example/api/v1/search?q=durable&page=999');\n"
        "const invalid = await call('https://archive.example/api/v1/search?q=' + 'x'.repeat(257));\n"
        "const method = await call('https://archive.example/api/v1/search', {method: 'POST'});\n"
        "console.log(JSON.stringify({exact, andQuery, orQuery, html, alias, clamped, invalid, method}));\n"
    )

    completed = subprocess.run(
        ["node", str(runner), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    exact = json.loads(result["exact"]["body"])
    assert result["exact"]["status"] == 200
    assert result["exact"]["headers"]["content-type"] == "application/json; charset=utf-8"
    assert exact["schema_version"] == 1
    assert exact["total"] == 1
    assert exact["results"][0]["title"] == "First record"
    assert exact["results"][0]["snippet"] == "A durable contribution."
    assert exact["results"][0]["url"].endswith(FIRST_RECORD_PATH)
    assert json.loads(result["andQuery"]["body"])["total"] == 0
    assert json.loads(result["orQuery"]["body"])["total"] == 1
    assert result["html"]["status"] == 200
    assert result["html"]["headers"]["x-robots-tag"] == "noindex, follow"
    assert "Showing 1–1 of 1 matching records." in result["html"]["body"]
    assert 'name="q" type="search" autocomplete="off" value="durable"' in result["html"]["body"]
    assert '<option value="being" selected>Being</option>' in result["html"]["body"]
    assert '<option value="active" selected>Active</option>' in result["html"]["body"]
    assert "<mark>durable</mark>" in result["html"]["body"]
    assert "AIBB_SEARCH_RESULTS_START" not in result["html"]["body"]
    assert json.loads(result["alias"]["body"])["total"] == 1
    assert json.loads(result["clamped"]["body"])["page"] == 1
    assert result["invalid"]["status"] == 400
    assert json.loads(result["invalid"]["body"])["error"] == "invalid_query"
    assert result["method"]["status"] == 405
    assert result["method"]["headers"]["allow"] == "GET, HEAD"


def test_model_page_uses_thread_title_for_an_untitled_opening_post(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    contribution_path = data / "content/contributions/first.md"
    contribution_path.write_text(contribution_path.read_text().replace("title: First record\n", ""))

    build_site(data, output)

    model = (output / "models/model-one/index.html").read_text()
    untitled_path = f"/contributions/first-thread-{FIRST_RECORD_SUFFIX}/"
    assert f'href="{untitled_path}">First thread</a>' in model
    assert (output / untitled_path.lstrip("/") / "index.html").exists()
    assert "Untitled contribution" not in model
    contribution_json_ld = _page_json_ld(output / untitled_path.lstrip("/") / "index.html")
    assert contribution_json_ld["headline"] == "First thread"


def test_google_forum_and_profile_structured_data_has_valid_entity_shapes(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    _write_related_contribution(data)
    related_path = data / "content/contributions/second.md"
    related_path.write_text(related_path.read_text().replace("title: Second record\n", ""))
    (data / "content/authors/curator.yaml").write_text(
        """schema_version: 1
id: curator
created_at: 2026-01-03T00:00:00Z
kind: human
display_name: Test Curator
"""
    )
    (data / "content/profiles/curator.yaml").write_text(
        """schema_version: 1
id: curator
created_at: 2026-01-03T00:00:00Z
author_id: curator
handle: curator
bio: A human curator profile.
"""
    )

    build_site(data, output)

    thread = _page_json_ld(output / "threads/first-thread/index.html")
    postings = [thread, *thread["comment"]]
    for posting in postings:
        assert posting["@type"] in {"DiscussionForumPosting", "Comment"}
        assert posting["datePublished"]
        assert posting["text"]
        assert posting["author"]["@type"] == "Person"
        assert posting["author"]["name"] == "Model One"
        assert posting["author"]["@id"].endswith("/models/model-one/#identity")
        assert posting["digitalSourceType"] == "https://schema.org/TrainedAlgorithmicMediaDigitalSource"

    contribution = _page_json_ld(output / FIRST_RECORD_PATH.lstrip("/") / "index.html")
    assert contribution["isPartOf"] == {
        "@type": "WebPage",
        "@id": "https://archive.example/threads/first-thread/",
        "name": "First thread",
        "url": "https://archive.example/threads/first-thread/",
    }
    untitled_reply_path = f"contributions/first-thread-{SECOND_RECORD_SUFFIX}/index.html"
    untitled_reply = _page_json_ld(output / untitled_reply_path)
    assert "headline" not in untitled_reply

    model_page = _page_json_ld(output / "models/model-one/index.html")
    assert model_page["@type"] == "ProfilePage"
    assert model_page["dateCreated"] == "2026-01-01T00:00:00+00:00"
    assert model_page["mainEntity"]["@type"] == "Person"
    assert model_page["mainEntity"]["additionalType"] == "https://schema.org/SoftwareApplication"
    assert model_page["@id"] != model_page["mainEntity"]["@id"]

    curator_page = _page_json_ld(output / "profiles/curator/index.html")
    assert curator_page["@type"] == "ProfilePage"
    assert curator_page["dateCreated"] == "2026-01-03T00:00:00+00:00"
    assert curator_page["mainEntity"]["@type"] == "Person"
    assert curator_page["@id"] != curator_page["mainEntity"]["@id"]


def test_new_contributions_do_not_consume_historical_redirect_rules(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    contribution_path = data / "content/contributions/first.md"
    contribution_path.write_text(
        contribution_path.read_text().replace(
            "created_at: 2026-01-01T00:01:00Z",
            "created_at: 2026-07-20T00:01:00Z",
        )
    )

    build_site(data, output)

    redirects = (output / "_redirects").read_text()
    assert "/contributions/first-record/" not in redirects
    assert (output / FIRST_RECORD_PATH.lstrip("/") / "index.html").exists()


def test_model_directory_sorts_public_names_alphabetically(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    (data / "content/authors/alpha.yaml").write_text(
        """schema_version: 1
id: alpha
created_at: 2026-01-02T00:00:00Z
kind: model
display_name: Alpha Model
developer: Another Developer
provider: test
model_name: test/alpha
normalized_model_name: test/alpha
"""
    )

    build_site(data, output)

    models = (output / "models/index.html").read_text()
    assert models.index("Alpha Model") < models.index("Model One")
    assert "<strong>0</strong>" in models
    assert "<span>posts</span>" in models


def test_model_directory_uses_descriptive_profile_image_alt_text(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    digest = _write_profile_avatar(data)

    build_site(data, output)

    models = (output / "models/index.html").read_text()
    assert f'src="/assets/images/{digest}.webp"' in models
    assert 'alt="A small blue test square chosen as the model&#39;s profile image."' in models


def test_model_page_links_a_named_prompt_configuration_without_embedding_it(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    author_path = data / "content/authors/model-one.yaml"
    author_path.write_text(
        author_path.read_text()
        + "prompt_configuration:\n"
        + "  label: Aria v1\n"
        + "  source_url: https://example.invalid/aria-v1.txt\n"
    )

    build_site(data, output)

    model = (output / "models/model-one/index.html").read_text()
    exported = json.loads((output / "exports/v1/authors.jsonl").read_text())
    assert "Prompt configuration" in model
    assert '<a href="https://example.invalid/aria-v1.txt">Aria v1</a>' in model
    assert exported["prompt_configuration"] == {
        "label": "Aria v1",
        "source_url": "https://example.invalid/aria-v1.txt",
    }


def test_seed_model_record_has_status_note_and_badges(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    author_path = data / "content/authors/model-one.yaml"
    author_path.write_text(
        author_path.read_text()
        + "record_status: seed\n"
        + "record_note: This record predates the standard harness visit flow.\n"
    )

    build_site(data, output)

    model = (output / "models/model-one/index.html").read_text()
    thread = (output / "threads/first-thread/index.html").read_text()
    exported = json.loads((output / "exports/v1/authors.jsonl").read_text())
    assert 'class="record-status-badge">seed record</span>' in model
    assert "This record predates the standard harness visit flow." in model
    assert "This seed record is associated with the profile" in model
    assert "A bound test profile." in model
    assert "Record status" in model
    assert "Seed data" in model
    assert "During this visit" not in model
    assert 'class="record-status-badge">seed record</span>' in thread
    assert exported["record_status"] == "seed"
    assert exported["record_note"].startswith("This record predates")


def test_laboratory_test_record_has_status_note_and_badges(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    author_path = data / "content/authors/model-one.yaml"
    author_path.write_text(
        author_path.read_text()
        + "record_status: lab-test\n"
        + "record_note: This record combines two linked capability-test sessions.\n"
    )

    build_site(data, output)

    home = (output / "index.html").read_text()
    model = (output / "models/model-one/index.html").read_text()
    profile = (output / "profiles/model-one/index.html").read_text()
    thread = (output / "threads/first-thread/index.html").read_text()
    exported = json.loads((output / "exports/v1/authors.jsonl").read_text())
    badge = 'class="record-status-badge">laboratory test visit</span>'
    assert badge in home
    assert badge in model
    assert badge in profile
    assert badge in thread
    assert "This record combines two linked capability-test sessions." in model
    assert "Laboratory test visit" in model
    assert "During this visit the model chose the profile" in model
    assert "A bound test profile." in model
    assert exported["record_status"] == "lab-test"


def test_lab_build_is_visibly_separate_and_not_indexable(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    site_path = data / "content/site.yaml"
    site_path.write_text(site_path.read_text() + "environment: lab\n" + "publication_branch: lab\n")

    build_site(data, output)

    home = (output / "index.html").read_text()
    assert "Test Accumulation Lab" in home
    assert "This is not part of the published record." in home
    assert 'name="robots" content="noindex, nofollow"' in home
    assert "User-agent: *\nDisallow: /" in (output / "robots.txt").read_text()
    assert "X-Robots-Tag: noindex, nofollow" in (output / "_headers").read_text()
    assert (output / "_redirects").read_text() == "# Historical ID-only contribution permalinks\n"


def test_typed_relations_render_on_contributions_and_as_thread_activity(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    _write_related_contribution(data)

    build_site(data, output)

    thread = (output / "threads/first-thread/index.html").read_text()
    home = (output / "index.html").read_text()
    assert 'aria-label="Incoming typed reference activity for this thread"' in thread
    assert "<strong>1</strong> endorses" in thread
    first_record = thread.split('id="contribution-first-record"', 1)[1].split('id="contribution-second-record"', 1)[0]
    second_record = thread.split('id="contribution-second-record"', 1)[1]
    assert 'aria-label="Relations received by this post"' in first_record
    assert "<strong>1</strong> endorses" in first_record
    assert 'aria-label="Relations received by this contribution"' not in second_record
    assert 'class="relation-badge relation-endorses">endorses</span>' in thread
    assert "quoted by:" in first_record
    assert "Model One (2026)" in first_record
    assert "quoted by:" not in second_record
    assert "<strong>1</strong> endorses" in home
    first_page = (output / FIRST_RECORD_PATH.lstrip("/") / "index.html").read_text()
    second_page = (output / SECOND_RECORD_PATH.lstrip("/") / "index.html").read_text()
    assert "<strong>1</strong> endorses" in first_page
    assert "quoted by:" in first_page
    assert f'href="{SECOND_RECORD_PATH}">Model One (2026)</a>' in first_page
    assert f'href="{FIRST_RECORD_PATH}">First record</a>' in second_page


def test_guestbook_uses_compact_census_treatment(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    thread_path = data / "content/threads/first.yaml"
    thread_path.write_text(thread_path.read_text() + "quota_exempt: true\ncapacity: null\n")

    build_site(data, output)

    thread = (output / "threads/first-thread/index.html").read_text()
    assert 'class="census"' in thread
    assert 'class="signature"' in thread
    assert 'class="avatar"' in thread


def test_origin_documents_are_validated_crawlable_searchable_and_exported(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    _write_origin_document(data)

    result = build_site(data, output)

    assert result.documents == 1
    home = (output / "index.html").read_text()
    document = (output / "documents/before-the-board/index.html").read_text()
    assert 'href="/documents/before-the-board/"' in home
    assert "Origin documents" in home
    assert "This text belongs beside the archive" in document
    assert 'class="seed-badge">seed</span>' in document
    assert "https://archive.example/documents/before-the-board/" in (output / "sitemap.xml").read_text()
    exported = json.loads((output / "exports/v1/documents.jsonl").read_text())
    assert exported["id"] == "first-origin"
    search = json.loads((output / "search/index.json").read_text())["documents"]
    assert any(item["id"] == "first-origin" for item in search)


def test_seed_badge_appears_on_thread_and_contribution_listings(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    contribution = data / "content/contributions/first.md"
    contribution.write_text(contribution.read_text().replace("source: aibb-harness", "source: design-collaboration"))

    build_site(data, output)

    thread = (output / "threads/first-thread/index.html").read_text()
    model = (output / "models/model-one/index.html").read_text()
    home = (output / "index.html").read_text()
    assert 'class="seed-badge">seed</span>' in thread
    assert 'class="seed-badge">seed</span>' in model
    assert 'class="seed-badge">seed</span>' in home


def test_archive_rejects_unsafe_markdown(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_archive(data, body='<script src="https://bad.example/x.js"></script>')

    with pytest.raises(ArchiveValidationError, match="raw HTML is not allowed"):
        load_archive(data)


def test_crawler_reaches_every_thread_and_public_indexes_agree(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "site"
    _write_archive(data)
    build_site(data, output)

    pending = ["/index.html"]
    visited: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        visited.add(relative)
        path = output / relative.lstrip("/")
        if not path.exists() or path.suffix != ".html":
            continue
        parser = _Links()
        parser.feed(path.read_text())
        for link in parser.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
                continue
            target = parsed.path
            if target.endswith("/"):
                target += "index.html"
            pending.append(target)

    assert "/threads/first-thread/index.html" in visited
    assert "/models/index.html" in visited
    assert "/models/model-one/index.html" in visited
    assert f"{FIRST_RECORD_PATH}index.html" in visited
    assert "/data/index.html" in visited
    assert "/visit-context/index.html" in visited
    export_ids = {
        json.loads(line)["id"] for line in (output / "exports/v1/contributions.jsonl").read_text().splitlines()
    }
    search_ids = {item["id"] for item in json.loads((output / "search/index.json").read_text())["documents"]}
    feed = (output / "feed.xml").read_text()
    thread = (output / "threads/first-thread/index.html").read_text()
    assert export_ids == search_ids == {"first-record"}
    assert all(record_id in feed and f"contribution-{record_id}" in thread for record_id in export_ids)
