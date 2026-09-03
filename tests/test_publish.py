from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_archive_build import _write_archive
from typer.testing import CliRunner

from aibb.cli import app
from aibb.publish import PublicationError, check_publication, deploy_publication, prepare_publication


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(path: Path, *, remote: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Slowboard Test")
    _git(path, "config", "user.email", "slowboard-test@users.noreply.github.com")
    _git(path, "remote", "add", "origin", remote)


def _commit_all(path: Path, message: str) -> str:
    _git(path, "add", "-A")
    _git(path, "commit", "-m", message)
    return _git(path, "rev-parse", "HEAD")


def _repositories(tmp_path: Path) -> tuple[Path, Path, Path]:
    code = tmp_path / "code"
    data = tmp_path / "data"
    site = tmp_path / "site"
    _init_repo(code, remote="https://github.com/example/slowboard.git")
    (code / "tracked.txt").write_text("builder revision\n")
    _commit_all(code, "code")
    _init_repo(data, remote="https://github.com/example/slowboard-data.git")
    _write_archive(data)
    _commit_all(data, "data")
    _init_repo(site, remote="https://github.com/example/slowboard-site.git")
    (site / ".github/workflows").mkdir(parents=True)
    (site / ".github/workflows/validate.yml").write_text("name: keep me\n")
    _commit_all(site, "site")
    return code, data, site


def _deployable_site(tmp_path: Path) -> tuple[Path, str]:
    site = tmp_path / "site"
    bare = tmp_path / "site.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    _init_repo(site, remote=str(bare))
    (site / "index.html").write_text("published\n")
    (site / ".github/workflows").mkdir(parents=True)
    (site / ".github/workflows/validate.yml").write_text("name: private CI metadata\n")
    (site / "publication.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site": "https://archive.example/",
                "builder": {
                    "repository": "https://github.com/example/slowboard",
                    "revision": "a" * 40,
                },
                "data": {
                    "repository": "https://github.com/example/slowboard-data",
                    "revision": "b" * 40,
                },
            }
        )
    )
    head = _commit_all(site, "publication")
    _git(site, "push", "-u", "origin", "main")
    return site, head


def test_prepare_and_check_publication_are_revision_bound_and_deterministic(tmp_path: Path) -> None:
    code, data, site = _repositories(tmp_path)

    manifest = prepare_publication(code_repo=code, data_repo=data, site_repo=site)

    assert manifest.builder.revision == _git(code, "rev-parse", "HEAD")
    assert manifest.data.revision == _git(data, "rev-parse", "HEAD")
    assert manifest.site == "https://archive.example/"
    assert manifest.channel == "production"
    assert manifest.branch == "main"
    assert (site / ".github/workflows/validate.yml").exists()
    assert (site / "threads/first-thread/index.html").exists()
    checked = check_publication(code_repo=code, data_repo=data, site_repo=site)
    assert checked["status"] == "valid"
    assert checked["files"] > 10

    _git(site, "switch", "--detach")
    assert check_publication(code_repo=code, data_repo=data, site_repo=site)["status"] == "valid"
    _git(site, "switch", "main")

    (site / "index.html").write_text("tampered\n")
    with pytest.raises(PublicationError, match="not the deterministic build"):
        check_publication(code_repo=code, data_repo=data, site_repo=site)


def test_prepare_refuses_dirty_source_or_output_repositories(tmp_path: Path) -> None:
    code, data, site = _repositories(tmp_path)
    (data / "untracked.txt").write_text("not public yet\n")

    with pytest.raises(PublicationError, match="Data repository must be clean"):
        prepare_publication(code_repo=code, data_repo=data, site_repo=site)


def test_prepare_refuses_publication_lane_branch_mismatch(tmp_path: Path) -> None:
    code, data, site = _repositories(tmp_path)
    site_config = data / "content/site.yaml"
    site_config.write_text(site_config.read_text() + "environment: lab\npublication_branch: lab\n")
    _commit_all(data, "configure lab lane")

    with pytest.raises(PublicationError, match="Lab data must publish from the 'lab'.*not 'main'"):
        prepare_publication(code_repo=code, data_repo=data, site_repo=site)

    _git(site, "switch", "-c", "lab")
    manifest = prepare_publication(code_repo=code, data_repo=data, site_repo=site)
    assert manifest.channel == "lab"
    assert manifest.branch == "lab"


def test_prepare_refuses_local_preview_base_url(tmp_path: Path) -> None:
    code, data, site = _repositories(tmp_path)
    site_config = data / "content/site.yaml"
    site_config.write_text(
        site_config.read_text().replace("https://archive.example/", "http://127.0.0.1:8000/")
    )
    _commit_all(data, "use local preview URL")

    with pytest.raises(PublicationError, match="canonical HTTPS base URL.*local-preview"):
        prepare_publication(code_repo=code, data_repo=data, site_repo=site)


def test_deploy_uses_pushed_commit_archive_and_cleans_wrangler_cache(tmp_path: Path) -> None:
    site, head = _deployable_site(tmp_path)
    arguments_path = tmp_path / "wrangler-arguments.json"
    fake = tmp_path / "fake_wrangler.py"
    fake.write_text(
        "import json, os, pathlib, sys\n"
        "root = pathlib.Path(sys.argv[3])\n"
        "result = {'arguments': sys.argv[1:], 'workflow_uploaded': (root / '.github').exists()}\n"
        "pathlib.Path(os.environ['ARGS_PATH']).write_text(json.dumps(result))\n"
        "pathlib.Path('.wrangler/cache').mkdir(parents=True)\n"
        "pathlib.Path('.wrangler/cache/account.json').write_text('private')\n"
        "print('https://test.slowboard.pages.dev')\n"
    )
    old = os.environ.get("ARGS_PATH")
    os.environ["ARGS_PATH"] = str(arguments_path)
    try:
        output = deploy_publication(
            site_repo=site,
            project_name="slowboard",
            wrangler_command=f"{sys.executable} {fake}",
        )
    finally:
        if old is None:
            os.environ.pop("ARGS_PATH", None)
        else:
            os.environ["ARGS_PATH"] = old

    invocation = json.loads(arguments_path.read_text())
    arguments = invocation["arguments"]
    assert arguments[:2] == ["pages", "deploy"]
    assert arguments[arguments.index("--commit-hash") + 1] == head
    assert output == "https://test.slowboard.pages.dev"
    assert invocation["workflow_uploaded"] is False
    assert not (site / ".wrangler").exists()


def test_deploy_reports_missing_wrangler_executable(tmp_path: Path) -> None:
    site, _ = _deployable_site(tmp_path)

    with pytest.raises(PublicationError, match="Wrangler executable 'not-a-real-wrangler'.*--wrangler-command"):
        deploy_publication(
            site_repo=site,
            project_name="slowboard",
            wrangler_command="not-a-real-wrangler",
        )


def test_deploy_reports_wrangler_stderr_and_cleans_cache(tmp_path: Path) -> None:
    site, _ = _deployable_site(tmp_path)
    fake = tmp_path / "failing_wrangler.py"
    fake.write_text(
        "import pathlib, sys\n"
        "pathlib.Path('.wrangler/cache').mkdir(parents=True)\n"
        "print('\\x1b[31mProject does not exist: wrong-project\\x1b[0m', file=sys.stderr)\n"
        "raise SystemExit(1)\n"
    )

    with pytest.raises(
        PublicationError,
        match="Wrangler deployment failed with exit code 1: Project does not exist: wrong-project",
    ) as raised:
        deploy_publication(
            site_repo=site,
            project_name="wrong-project",
            wrangler_command=f"{sys.executable} {fake}",
        )

    assert "\x1b" not in str(raised.value)
    assert not (site / ".wrangler").exists()


def test_publish_deploy_cli_renders_publication_errors_without_traceback(tmp_path: Path) -> None:
    site, _ = _deployable_site(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "publish",
            "deploy",
            "--site-repo",
            str(site),
            "--project-name",
            "slowboard",
            "--wrangler-command",
            "not-a-real-wrangler",
        ],
    )

    assert result.exit_code == 2
    assert "Wrangler executable 'not-a-real-wrangler' was not found" in result.output
    assert "Traceback" not in result.output
