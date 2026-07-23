"""
Caching and path resolution for the SDLXLIFF MCP server.

Provides:
- LRU-style parser cache with modification time validation
- File extension validation
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .constants import ALLOWED_EXTENSIONS, CACHE_MAX_SIZE

if TYPE_CHECKING:
    from .parser import SDLXLIFFParser

logger = logging.getLogger("sdlxliff-server")


@dataclass
class CachedParser:
    """Cache entry for parser with modification time tracking."""
    parser: "SDLXLIFFParser"
    mtime: float


# Module-level cache state
_parser_cache: dict[str, CachedParser] = {}


def validate_file_extension(file_path: str) -> None:
    """
    Validate that the file has an allowed extension.

    Args:
        file_path: The file path to validate

    Raises:
        ValueError: If the file extension is not allowed
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Invalid file type: '{suffix}'. "
            f"This tool only supports SDLXLIFF files ({', '.join(ALLOWED_EXTENSIONS)})"
        )


def resolve_file_path(file_path: str) -> Path:
    """
    Resolve a file path to the exact file requested.

    Only the given path is considered - no directory scanning or filename
    guessing is performed.

    Args:
        file_path: The file path to resolve

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If the file does not exist at the given path
        ValueError: If the file extension is not allowed
    """
    # Validate file extension first
    validate_file_extension(file_path)

    path = Path(file_path)

    try:
        if path.exists() and path.is_file():
            return path.resolve()
    except (OSError, ValueError) as e:
        logger.debug(f"Path check failed: {e}")

    raise FileNotFoundError(f"File not found: {file_path}")


def get_parser(file_path: str) -> "SDLXLIFFParser":
    """
    Get or create a parser instance for the given file.

    Uses LRU-style caching with modification time validation to ensure
    fresh data and bounded memory usage.

    Args:
        file_path: Path to the SDLXLIFF file

    Returns:
        SDLXLIFFParser instance
    """
    # Import here to avoid circular dependency
    from .parser import SDLXLIFFParser

    # Resolve path with sandbox awareness
    path = resolve_file_path(file_path)
    normalized_path = str(path)

    # Get current file modification time
    current_mtime = path.stat().st_mtime

    # Check if cached and still valid
    if normalized_path in _parser_cache:
        cached = _parser_cache[normalized_path]
        if cached.mtime == current_mtime:
            # Move to end for LRU behavior (most recently used)
            _parser_cache.pop(normalized_path)
            _parser_cache[normalized_path] = cached
            return cached.parser
        else:
            # File modified, remove stale cache
            logger.debug(f"Cache invalidated for {normalized_path} (file modified)")
            _parser_cache.pop(normalized_path)

    # Evict oldest entry if cache is full
    if len(_parser_cache) >= CACHE_MAX_SIZE:
        oldest_key = next(iter(_parser_cache))
        logger.debug(f"Evicting oldest cache entry: {oldest_key}")
        _parser_cache.pop(oldest_key)

    # Create new parser and cache it
    parser = SDLXLIFFParser(normalized_path)
    _parser_cache[normalized_path] = CachedParser(parser=parser, mtime=current_mtime)

    return parser


def clear_parser_cache(file_path: Optional[str] = None) -> None:
    """
    Clear parser cache for a specific file or all files.

    Args:
        file_path: Optional specific file path. If None, clears all cache.
    """
    if file_path:
        normalized_path = str(Path(file_path).resolve())
        _parser_cache.pop(normalized_path, None)
    else:
        _parser_cache.clear()