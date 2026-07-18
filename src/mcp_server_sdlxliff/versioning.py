"""
Invisible, local-only version history for SDLXLIFF files.

Every save snapshots the file into a repository stored in a hidden `.git`
folder next to the file, using dulwich (pure Python — no system git binary
required). History never leaves the local machine: no remotes are ever
configured and nothing is ever pushed anywhere (client confidentiality).

Version numbers exposed to users are 1-based and chronological:
version 1 is the oldest snapshot, the highest number is the newest.
"""

import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dulwich.repo import Repo

logger = logging.getLogger("sdlxliff-versioning")

# Identity used for snapshot commits (dulwich requires one; we never read
# or write the user's global git configuration)
_IDENTITY = b"SDLXLIFF Translation Editor <history@localhost>"


class VersioningError(Exception):
    """Raised when a version-history operation cannot be completed."""


@dataclass
class Version:
    """A single saved version of a file."""
    number: int          # 1-based, chronological (1 = oldest)
    commit_id: str
    date: str            # Human-readable local time
    description: str     # What changed in this version


def _hide_directory(path: Path) -> None:
    """Mark a directory hidden on Windows (dot-prefix already hides it elsewhere)."""
    if os.name == 'nt':
        try:
            import ctypes
            FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
        except Exception as e:
            logger.debug(f"Could not mark {path} hidden: {e}")


def _open_or_init_repo(folder: Path) -> Repo:
    """
    Open the version-history repository in a folder, creating it on first use.

    Gracefully reuses an existing repository if one is already present
    (whether created by us or by the user).
    """
    git_dir = folder / ".git"
    if git_dir.exists():
        return Repo(str(folder))

    repo = Repo.init(str(folder))
    _hide_directory(git_dir)
    logger.info(f"Initialized local version history in {folder}")
    return repo


def _relative_file_bytes(repo: Repo, file_path: Path) -> bytes:
    """Get the repo-relative path of a file as bytes (git's internal form)."""
    repo_root = Path(repo.path).resolve()
    rel = file_path.resolve().relative_to(repo_root)
    return str(rel).replace(os.sep, '/').encode('utf-8')


def _blob_at_commit(repo: Repo, commit, rel_path: bytes) -> Optional[bytes]:
    """Return the file's content at a given commit, or None if absent."""
    try:
        tree = repo[commit.tree]
        mode, sha = tree.lookup_path(repo.get_object, rel_path)
        return repo[sha].data
    except KeyError:
        return None


def _list_versions(repo: Repo, rel_path: bytes) -> List[Version]:
    """All versions of a file, oldest first."""
    entries = []
    try:
        walker = repo.get_walker(paths=[rel_path])
    except KeyError:
        return []  # No commits yet

    for entry in walker:
        commit = entry.commit
        when = datetime.fromtimestamp(commit.commit_time)
        entries.append((
            commit.id.decode('ascii'),
            when.strftime('%Y-%m-%d %H:%M'),
            commit.message.decode('utf-8', errors='replace').strip(),
        ))

    entries.reverse()  # Walker yields newest first; we number oldest = 1
    return [
        Version(number=i + 1, commit_id=cid, date=date, description=msg)
        for i, (cid, date, msg) in enumerate(entries)
    ]


def _commit_file(repo: Repo, file_path: Path, message: str) -> Optional[str]:
    """
    Snapshot the file's current on-disk content. Returns the commit id,
    or None if the content is identical to the latest snapshot.
    """
    rel_path = _relative_file_bytes(repo, file_path)

    # Skip if content is unchanged from the last snapshot of this file
    try:
        head = repo[repo.head()]
        previous = _blob_at_commit(repo, head, rel_path)
        if previous is not None and previous == file_path.read_bytes():
            return None
    except KeyError:
        pass  # No commits yet

    # Stage and commit via the low-level API. sign=False and an explicit
    # identity keep behavior independent of any user git configuration
    # (no signing, no reliance on user.name/user.email being set).
    worktree = repo.get_worktree()
    worktree.stage([rel_path])
    commit_id = worktree.commit(
        message=message.encode('utf-8'),
        author=_IDENTITY,
        committer=_IDENTITY,
        sign=False,
    )
    return commit_id.decode('ascii')


def ensure_baseline(file_path: Path) -> Optional[str]:
    """
    Make sure the file's current on-disk state is snapshotted before it is
    overwritten. On the first save in a folder this initializes the local
    history and captures the original file.

    Returns a warning string if history could not be recorded (the save
    itself must still proceed), otherwise None.
    """
    if not file_path.exists():
        return None
    try:
        repo = _open_or_init_repo(file_path.parent)
        try:
            rel_path = _relative_file_bytes(repo, file_path)
            if not _list_versions(repo, rel_path):
                _commit_file(repo, file_path,
                             f"Original version of {file_path.name}")
            else:
                # File changed outside our tools since the last snapshot?
                # Capture that state too so nothing is ever lost.
                _commit_file(repo, file_path,
                             f"State of {file_path.name} before this save")
        finally:
            repo.close()
        return None
    except Exception as e:
        logger.warning(f"Could not record pre-save version history: {e}")
        return f"Version history unavailable for this save: {e}"


def record_save(file_path: Path, message: str) -> Optional[str]:
    """
    Snapshot the file's new content after a save.

    Returns a warning string if history could not be recorded, otherwise None.
    """
    try:
        repo = _open_or_init_repo(file_path.parent)
        try:
            _commit_file(repo, file_path, message)
        finally:
            repo.close()
        return None
    except Exception as e:
        logger.warning(f"Could not record version history: {e}")
        return f"Version history unavailable for this save: {e}"


def get_history(file_path: Path) -> List[Version]:
    """
    List all saved versions of a file, oldest first.

    Raises VersioningError if no history exists yet.
    """
    git_dir = file_path.parent / ".git"
    if not git_dir.exists():
        raise VersioningError(
            f"No saved versions exist yet for files in {file_path.parent}. "
            "Versions are recorded automatically each time a file is saved."
        )
    repo = Repo(str(file_path.parent))
    try:
        versions = _list_versions(repo, _relative_file_bytes(repo, file_path))
    finally:
        repo.close()
    if not versions:
        raise VersioningError(
            f"No saved versions exist yet for {file_path.name}. "
            "Versions are recorded automatically each time the file is saved."
        )
    return versions


def get_version_content(file_path: Path, version_number: int) -> bytes:
    """
    Get the exact file content of a saved version.

    Raises VersioningError if the version does not exist.
    """
    versions = get_history(file_path)
    match = next((v for v in versions if v.number == version_number), None)
    if match is None:
        available = ', '.join(str(v.number) for v in versions)
        raise VersioningError(
            f"Version {version_number} does not exist for {file_path.name}. "
            f"Available versions: {available}."
        )

    repo = Repo(str(file_path.parent))
    try:
        commit = repo[match.commit_id.encode('ascii')]
        content = _blob_at_commit(repo, commit, _relative_file_bytes(repo, file_path))
    finally:
        repo.close()
    if content is None:
        raise VersioningError(
            f"Version {version_number} of {file_path.name} could not be read."
        )
    return content


def restore_version(file_path: Path, version_number: int) -> Version:
    """
    Replace the file's current content with a saved version.

    The current content is snapshotted first, so a restore can always be
    undone the same way. Returns the Version that was restored.

    Raises VersioningError if the version does not exist.
    """
    versions = get_history(file_path)
    match = next((v for v in versions if v.number == version_number), None)
    if match is None:
        available = ', '.join(str(v.number) for v in versions)
        raise VersioningError(
            f"Version {version_number} does not exist for {file_path.name}. "
            f"Available versions: {available}."
        )

    content = get_version_content(file_path, version_number)

    # Snapshot the current state first so the restore itself is undoable
    warning = ensure_baseline(file_path)
    if warning:
        raise VersioningError(
            f"Cannot restore safely because the current file state could not "
            f"be preserved: {warning}"
        )

    # Atomic write of the restored content (temp file in the same folder)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp', prefix='.sdlxliff_', dir=file_path.parent
    )
    try:
        os.write(temp_fd, content)
        os.close(temp_fd)
        temp_fd = None
        os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temp_path, str(file_path))
    except Exception:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise

    record_save(
        file_path,
        f"Restored {file_path.name} to the version from {match.date}",
    )
    return match
