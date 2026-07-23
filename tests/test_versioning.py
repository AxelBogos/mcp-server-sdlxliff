"""
Tests for the invisible local-only version history (versioning.py).
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_server_sdlxliff import versioning
from mcp_server_sdlxliff.versioning import VersioningError
from mcp_server_sdlxliff.parser import SDLXLIFFParser
from mcp_server_sdlxliff.server import _diff_segment_lists


SDLXLIFF_V1 = '''<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" xmlns:sdl="http://sdl.com/FileTypes/SdlXliff/1.0" version="1.2">
  <file source-language="en-US" target-language="fr-FR">
    <body>
      <trans-unit id="1">
        <source>Hello world.</source>
        <seg-source><mrk mtype="seg" mid="1">Hello world.</mrk></seg-source>
        <target><mrk mtype="seg" mid="1">Bonjour le monde.</mrk></target>
        <sdl:seg-defs><sdl:seg id="1" conf="Translated"/></sdl:seg-defs>
      </trans-unit>
    </body>
  </file>
</xliff>'''

SDLXLIFF_V2 = SDLXLIFF_V1.replace("Bonjour le monde.", "Bonjour tout le monde.")


@pytest.fixture
def sdlxliff_file(tmp_path):
    f = tmp_path / "projet.sdlxliff"
    f.write_text(SDLXLIFF_V1, encoding='utf-8')
    return f


class TestSnapshotOnSave:
    def test_baseline_initializes_hidden_repo(self, sdlxliff_file):
        versioning.ensure_baseline(sdlxliff_file)
        assert (sdlxliff_file.parent / ".git").exists()

    def test_baseline_captures_original(self, sdlxliff_file):
        versioning.ensure_baseline(sdlxliff_file)
        versions = versioning.get_history(sdlxliff_file)
        assert len(versions) == 1
        assert versions[0].number == 1
        assert "Original version" in versions[0].description

    def test_record_save_adds_version(self, sdlxliff_file):
        versioning.ensure_baseline(sdlxliff_file)
        sdlxliff_file.write_text(SDLXLIFF_V2, encoding='utf-8')
        versioning.record_save(sdlxliff_file, "Updated 1 segment in projet.sdlxliff (EN→FR)")

        versions = versioning.get_history(sdlxliff_file)
        assert len(versions) == 2
        assert versions[1].description == "Updated 1 segment in projet.sdlxliff (EN→FR)"

    def test_unchanged_content_is_not_duplicated(self, sdlxliff_file):
        versioning.ensure_baseline(sdlxliff_file)
        versioning.record_save(sdlxliff_file, "No-op save")
        versions = versioning.get_history(sdlxliff_file)
        assert len(versions) == 1  # Identical content -> no new version

    def test_existing_repo_is_reused(self, sdlxliff_file):
        # A repo already initialized in the folder must be reused, not broken
        from dulwich.repo import Repo
        Repo.init(str(sdlxliff_file.parent)).close()

        assert versioning.ensure_baseline(sdlxliff_file) is None
        versions = versioning.get_history(sdlxliff_file)
        assert len(versions) == 1

    def test_no_remotes_configured(self, sdlxliff_file):
        """History must be local-only: no remote is ever configured."""
        versioning.ensure_baseline(sdlxliff_file)
        config_text = (sdlxliff_file.parent / ".git" / "config").read_text()
        assert "remote" not in config_text
        assert "url" not in config_text


class TestHistoryAndVersions:
    def test_history_without_repo_raises(self, sdlxliff_file):
        with pytest.raises(VersioningError, match="No saved versions"):
            versioning.get_history(sdlxliff_file)

    def test_history_for_untracked_file_raises(self, sdlxliff_file, tmp_path):
        versioning.ensure_baseline(sdlxliff_file)
        other = tmp_path / "autre.sdlxliff"
        other.write_text(SDLXLIFF_V1, encoding='utf-8')
        with pytest.raises(VersioningError, match="No saved versions"):
            versioning.get_history(other)

    def test_get_version_content_exact_bytes(self, sdlxliff_file):
        original_bytes = sdlxliff_file.read_bytes()
        versioning.ensure_baseline(sdlxliff_file)
        sdlxliff_file.write_text(SDLXLIFF_V2, encoding='utf-8')
        versioning.record_save(sdlxliff_file, "Update")

        assert versioning.get_version_content(sdlxliff_file, 1) == original_bytes

    def test_unknown_version_raises_with_available_list(self, sdlxliff_file):
        versioning.ensure_baseline(sdlxliff_file)
        with pytest.raises(VersioningError, match="Available versions: 1"):
            versioning.get_version_content(sdlxliff_file, 42)


class TestRestore:
    def test_restore_reverts_file(self, sdlxliff_file):
        versioning.ensure_baseline(sdlxliff_file)
        sdlxliff_file.write_text(SDLXLIFF_V2, encoding='utf-8')
        versioning.record_save(sdlxliff_file, "Update")

        restored = versioning.restore_version(sdlxliff_file, 1)
        assert restored.number == 1
        assert "Bonjour le monde." in sdlxliff_file.read_text(encoding='utf-8')

    def test_restore_is_undoable(self, sdlxliff_file):
        """Restoring snapshots the pre-restore state, so it can be undone."""
        versioning.ensure_baseline(sdlxliff_file)
        sdlxliff_file.write_text(SDLXLIFF_V2, encoding='utf-8')
        versioning.record_save(sdlxliff_file, "Update")

        versioning.restore_version(sdlxliff_file, 1)
        versions = versioning.get_history(sdlxliff_file)
        # original + update + restore-commit
        assert len(versions) == 3
        assert "Restored" in versions[-1].description

        # Undo the restore: go back to the updated state (version 2)
        versioning.restore_version(sdlxliff_file, 2)
        assert "Bonjour tout le monde." in sdlxliff_file.read_text(encoding='utf-8')

    def test_restore_unknown_version_raises(self, sdlxliff_file):
        versioning.ensure_baseline(sdlxliff_file)
        with pytest.raises(VersioningError, match="does not exist"):
            versioning.restore_version(sdlxliff_file, 99)


class TestSegmentDiff:
    def test_diff_between_versions(self, sdlxliff_file):
        versioning.ensure_baseline(sdlxliff_file)
        sdlxliff_file.write_text(SDLXLIFF_V2, encoding='utf-8')
        versioning.record_save(sdlxliff_file, "Update")

        old_content = versioning.get_version_content(sdlxliff_file, 1)
        old_segments = SDLXLIFFParser.from_bytes(old_content).extract_segments()
        new_segments = SDLXLIFFParser(str(sdlxliff_file)).extract_segments()

        changes = _diff_segment_lists(old_segments, new_segments)
        assert changes == [{
            'segment_id': '1',
            'old_target': 'Bonjour le monde.',
            'new_target': 'Bonjour tout le monde.',
        }]

    def test_diff_identical_versions_is_empty(self, sdlxliff_file):
        segments = SDLXLIFFParser(str(sdlxliff_file)).extract_segments()
        assert _diff_segment_lists(segments, segments) == []
