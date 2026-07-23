"""
Language configuration for offline spellchecking.

Maps BCP-47 language codes (used in XLIFF/SDLXLIFF files) to pyspellchecker
language codes. All spellchecking runs fully offline using the bundled
pyspellchecker dictionaries — no network access is ever required.

Note: Russian and Ukrainian were previously checked via the online
Yandex.Speller API and are no longer supported (pyspellchecker's
frequency-based approach handles highly inflected languages poorly).
"""

from typing import Optional

# Languages with adequate offline dictionary support in pyspellchecker.
# Maps ISO 639-1 base code to pyspellchecker language code.
LANGUAGE_CONFIG = {
    'en': 'en',  # English
    'de': 'de',  # German
    'es': 'es',  # Spanish
    'fr': 'fr',  # French
    'it': 'it',  # Italian
    'pt': 'pt',  # Portuguese
    'nl': 'nl',  # Dutch
}


def get_spellcheck_lang(xliff_lang: str) -> Optional[str]:
    """
    Get the pyspellchecker language code for an XLIFF language.

    Args:
        xliff_lang: BCP-47 language tag (e.g., 'fr-FR', 'fr-CA', 'en-US')

    Returns:
        pyspellchecker language code (e.g., 'fr') or None if unsupported
    """
    if not xliff_lang:
        return None
    # Extract base language code (e.g., 'fr-CA' -> 'fr')
    base_lang = xliff_lang.split('-')[0].lower()
    return LANGUAGE_CONFIG.get(base_lang)


def is_language_supported(xliff_lang: str) -> bool:
    """
    Check if language is supported for offline spellchecking.

    Args:
        xliff_lang: BCP-47 language tag (e.g., 'fr-FR', 'en-US')

    Returns:
        True if the language has an offline dictionary available
    """
    return get_spellcheck_lang(xliff_lang) is not None
