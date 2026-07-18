"""
Tests for the french_typography QA check (FR-FR and FR-CA conventions).
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_server_sdlxliff.qa import (
    NBSP,
    NNBSP,
    check_french_typography,
    run_qa_checks,
)


def messages(text, convention='fr-FR'):
    return [i.message for i in check_french_typography('1', text, convention)]


class TestPunctuationSpacingFrFr:
    def test_breaking_space_before_punctuation_flagged(self):
        msgs = messages("Attention : ceci est un test !")
        assert len(msgs) == 2
        assert any("':'" in m and "Breaking space" in m for m in msgs)
        assert any("'!'" in m and "Breaking space" in m for m in msgs)

    def test_nbsp_and_nnbsp_accepted(self):
        assert messages(f"Attention{NBSP}: parfait{NNBSP}!") == []

    def test_missing_space_flagged(self):
        msgs = messages("Attention: pas d'espace!")
        assert len(msgs) == 2
        assert all("Missing non-breaking space" in m for m in msgs)

    def test_time_not_flagged(self):
        assert messages("Rendez-vous à 10:30 demain.") == []

    def test_url_not_flagged(self):
        assert messages("Voir https://example.com pour info.") == []

    def test_repeated_punctuation_reported_once(self):
        msgs = messages("Quoi ?!")
        assert len(msgs) == 1  # Only the '?' is reported, not the '!' after it

    def test_empty_target(self):
        assert messages("") == []


class TestPunctuationSpacingFrCa:
    def test_colon_still_requires_nbsp(self):
        msgs = messages("Attention : oui", 'fr-CA')
        assert len(msgs) == 1
        assert "':'" in msgs[0]

    def test_no_space_before_exclamation_is_correct(self):
        assert messages("Bravo! Continue!", 'fr-CA') == []

    def test_missing_space_before_colon_flagged(self):
        msgs = messages("Attention: oui", 'fr-CA')
        assert len(msgs) == 1
        assert "Missing non-breaking space before ':'" in msgs[0]

    def test_breaking_space_before_exclamation_flagged(self):
        msgs = messages("Bravo !", 'fr-CA')
        assert len(msgs) == 1
        assert "Canadian French" in msgs[0]

    def test_nnbsp_before_exclamation_tolerated(self):
        assert messages(f"Bravo{NNBSP}!", 'fr-CA') == []


class TestQuotes:
    def test_straight_quotes_flagged(self):
        msgs = messages('Il a dit "bonjour" hier.')
        assert any("Straight quotes" in m for m in msgs)

    def test_curly_quotes_flagged(self):
        msgs = messages('Il a dit “bonjour” hier.')
        assert any("curly quotes" in m for m in msgs)

    def test_correct_guillemets_pass(self):
        assert messages(f'Il a dit «{NBSP}bonjour{NBSP}» hier.') == []

    def test_guillemets_missing_inner_spaces_flagged(self):
        msgs = messages('Il a dit «bonjour» hier.')
        assert any("after '«'" in m for m in msgs)
        assert any("before '»'" in m for m in msgs)


class TestNumberFormatting:
    def test_english_thousands_flagged(self):
        msgs = messages("Le total est 1,234.56 dollars.")
        assert len(msgs) == 1
        assert "English number format" in msgs[0]

    def test_decimal_point_flagged(self):
        msgs = messages("Pi vaut 3.14 environ.")
        assert len(msgs) == 1
        assert "Decimal point" in msgs[0]
        assert "3,14" in msgs[0]

    def test_french_format_passes(self):
        assert messages(f"Le total est 1{NBSP}234,56 dollars.") == []

    def test_version_number_not_flagged(self):
        assert messages("Utilisez la version 2.5.1 du logiciel.") == []


class TestLanguageGating:
    SEGMENTS = [{'segment_id': '1', 'source': 'Note: yes', 'target': 'Note : oui'}]

    def test_runs_for_french_target(self):
        report = run_qa_checks(self.SEGMENTS, target_lang='fr-FR')
        assert report.summary.get('french_typography') == 1

    def test_runs_for_canadian_french_target(self):
        report = run_qa_checks(self.SEGMENTS, target_lang='fr-CA')
        assert report.summary.get('french_typography') == 1

    def test_skipped_for_non_french_target(self):
        report = run_qa_checks(self.SEGMENTS, target_lang='de-DE')
        assert 'french_typography' not in report.summary

    def test_skipped_without_target_language(self):
        report = run_qa_checks(self.SEGMENTS, target_lang=None)
        assert 'french_typography' not in report.summary

    def test_convention_derived_from_fr_ca_target(self):
        # 'Bravo !' with a breaking space: flagged under both conventions but
        # with the Canadian message when target is fr-CA
        segments = [{'segment_id': '1', 'source': 'Great!', 'target': 'Bravo !'}]
        report = run_qa_checks(segments, target_lang='fr-CA')
        typo = [i for i in report.issues if i.check == 'french_typography']
        assert len(typo) == 1
        assert "Canadian French" in typo[0].message

    def test_explicit_convention_overrides_target_lang(self):
        segments = [{'segment_id': '1', 'source': 'Great!', 'target': 'Bravo !'}]
        report = run_qa_checks(segments, target_lang='fr-FR', french_convention='fr-CA')
        typo = [i for i in report.issues if i.check == 'french_typography']
        assert len(typo) == 1
        assert "Canadian French" in typo[0].message
