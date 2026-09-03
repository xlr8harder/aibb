"""Reproducible publication preparation, verification, and Cloudflare deployment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aibb.domain import load_archive
from aibb.site import build_site


class PublicationError(ValueError):
    """Raised when a publication cannot be tied to clean, exact revisions."""


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class RepositoryRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")


class PublicationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    site: str = Field(pattern=r"^https://")
    channel: Literal["production", "lab"] = "production"
    branch: str = "main"
    builder: RepositoryRevision
    data: RepositoryRevision


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "git command failed").strip()
        raise PublicationError(f"Git failed for {repo}: {detail}") from error
    return result.stdout.strip()


def _revision(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _repository_url(repo: Path) -> str:
    value = _git(repo, "remote", "get-url", "origin")
    if value.startswith("git@github.com:"):
        return "https://github.com/" + value.removeprefix("git@github.com:").removesuffix(".git")
    return value.removesuffix(".git")


def _require_clean(repo: Path, *, role: str, include_untracked: bool = True) -> None:
    if not (repo / ".git").exists():
        raise PublicationError(f"{role} is not a Git worktree: {repo}")
    untracked = "all" if include_untracked else "no"
    status = _git(repo, "status", "--porcelain=v1", f"--untracked-files={untracked}")
    if status:
        raise PublicationError(f"{role} must be clean before publication: {repo}")


def _replace_site_tree(site_repo: Path, build: Path) -> None:
    for path in site_repo.iterdir():
        if path.name in {".git", ".github"}:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    for source in build.iterdir():
        target = site_repo / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _digest_files(root: Path, *, publication_tree: bool) -> dict[str, str]:
    excluded_top_level = {".git", ".github"}
    if publication_tree:
        excluded_top_level.add("publication.json")
    result: dict[str, str] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if relative.parts[0] in excluded_top_level:
            continue
        result[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _require_publishable_site_url(base_url: str) -> None:
    if not base_url.startswith("https://"):
        raise PublicationError(
            f"Publication requires a canonical HTTPS base URL; {base_url!r} is local-preview configuration"
        )


def prepare_publication(*, code_repo: Path, data_repo: Path, site_repo: Path) -> PublicationManifest:
    code_repo = code_repo.resolve()
    data_repo = data_repo.resolve()
    site_repo = site_repo.resolve()
    _require_clean(code_repo, role="Builder repository", include_untracked=False)
    _require_clean(data_repo, role="Data repository")
    _require_clean(site_repo, role="Generated-site repository")
    corpus = load_archive(data_repo)
    _require_publishable_site_url(corpus.site.base_url)
    site_branch = _git(site_repo, "branch", "--show-current")
    if site_branch != corpus.site.publication_branch:
        raise PublicationError(
            f"{corpus.site.environment.title()} data must publish from the "
            f"{corpus.site.publication_branch!r} generated-site branch, not {site_branch!r}"
        )

    with tempfile.TemporaryDirectory(prefix="slowboard-publication-") as temporary:
        output = Path(temporary) / "site"
        build = build_site(data_repo, output)
        manifest = PublicationManifest(
            site=corpus.site.base_url,
            channel=corpus.site.environment,
            branch=corpus.site.publication_branch,
            builder=RepositoryRevision(repository=_repository_url(code_repo), revision=_revision(code_repo)),
            data=RepositoryRevision(repository=_repository_url(data_repo), revision=_revision(data_repo)),
        )
        _replace_site_tree(site_repo, build.output)
    (site_repo / "publication.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def check_publication(*, code_repo: Path, data_repo: Path, site_repo: Path) -> dict[str, object]:
    code_repo = code_repo.resolve()
    data_repo = data_repo.resolve()
    site_repo = site_repo.resolve()
    _require_clean(code_repo, role="Builder repository", include_untracked=False)
    _require_clean(data_repo, role="Data repository")
    manifest = PublicationManifest.model_validate_json((site_repo / "publication.json").read_text(encoding="utf-8"))
    corpus = load_archive(data_repo)
    _require_publishable_site_url(corpus.site.base_url)
    site_branch = _git(site_repo, "branch", "--show-current")
    if site_branch and site_branch != manifest.branch:
        raise PublicationError("Publication branch does not match the generated-site worktree branch")
    if manifest.channel != corpus.site.environment or manifest.branch != corpus.site.publication_branch:
        raise PublicationError("Publication channel does not match the data repository configuration")
    if manifest.site != corpus.site.base_url:
        raise PublicationError("Publication site URL does not match the data repository configuration")
    if manifest.builder.revision != _revision(code_repo):
        raise PublicationError("Publication builder revision does not match the checked-out code revision")
    if manifest.data.revision != _revision(data_repo):
        raise PublicationError("Publication data revision does not match the checked-out data revision")
    if manifest.builder.repository != _repository_url(code_repo):
        raise PublicationError("Publication builder repository does not match the checked-out code repository")
    if manifest.data.repository != _repository_url(data_repo):
        raise PublicationError("Publication data repository does not match the checked-out data repository")

    with tempfile.TemporaryDirectory(prefix="slowboard-publication-check-") as temporary:
        output = Path(temporary) / "site"
        build_site(data_repo, output)
        expected = _digest_files(output, publication_tree=False)
    actual = _digest_files(site_repo, publication_tree=True)
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        changed = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])
        raise PublicationError(
            "Generated-site tree is not the deterministic build "
            f"(missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]})"
        )
    return {
        "status": "valid",
        "channel": manifest.channel,
        "branch": manifest.branch,
        "builder_revision": manifest.builder.revision,
        "data_revision": manifest.data.revision,
        "files": len(actual) + 1,
    }


def deploy_publication(
    *,
    site_repo: Path,
    project_name: str,
    branch: str = "main",
    wrangler_command: str = "wrangler",
) -> str:
    site_repo = site_repo.resolve()
    _require_clean(site_repo, role="Generated-site repository")
    try:
        manifest = PublicationManifest.model_validate_json(
            (site_repo / "publication.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise PublicationError("Generated-site repository does not contain a valid publication.json") from error
    if branch != manifest.branch:
        raise PublicationError(
            f"Deployment branch {branch!r} does not match publication branch {manifest.branch!r}"
        )
    if _git(site_repo, "branch", "--show-current") != manifest.branch:
        raise PublicationError("Generated-site worktree is not on the publication branch")
    head = _revision(site_repo)
    upstream = _git(site_repo, "rev-parse", "@{upstream}")
    if upstream != head:
        raise PublicationError("Generated-site HEAD must be pushed to its upstream before deployment")

    command = shlex.split(wrangler_command)
    if not command:
        raise PublicationError("Wrangler command cannot be empty")
    with tempfile.TemporaryDirectory(prefix="slowboard-deploy-") as temporary:
        archive = Path(temporary) / "site.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "-C", str(site_repo), "archive", "--format=tar", "HEAD"],
                check=True,
                stdout=stream,
            )
        deploy_tree = Path(temporary) / "site"
        deploy_tree.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(deploy_tree, filter="data")
        shutil.rmtree(deploy_tree / ".github", ignore_errors=True)
        try:
            try:
                result = subprocess.run(
                    [
                        *command,
                        "pages",
                        "deploy",
                        str(deploy_tree),
                        "--project-name",
                        project_name,
                        "--branch",
                        branch,
                        "--commit-hash",
                        head,
                        "--commit-dirty=false",
                    ],
                    check=True,
                    cwd=site_repo,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
            except FileNotFoundError as error:
                raise PublicationError(
                    f"Wrangler executable {command[0]!r} was not found; install Wrangler or pass "
                    "--wrangler-command with an available command"
                ) from error
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "Wrangler returned no error output").strip()
                detail = _ANSI_ESCAPE.sub("", detail)
                raise PublicationError(
                    f"Wrangler deployment failed with exit code {error.returncode}: {detail}"
                ) from error
        finally:
            shutil.rmtree(site_repo / ".wrangler", ignore_errors=True)
    return result.stdout.strip()
