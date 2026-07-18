"""
File I/O operations for SDLXLIFF files.

Handles loading, saving, atomic writes, and BOM preservation.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

from lxml import etree

from .constants import MAX_FILE_SIZE

logger = logging.getLogger("sdlxliff-parser")


def create_secure_parser() -> etree.XMLParser:
    """
    Create a secure XML parser with XXE and XML bomb protection.

    Returns:
        Configured lxml XMLParser instance
    """
    return etree.XMLParser(
        remove_blank_text=False,
        strip_cdata=False,
        resolve_entities=False,  # Prevent XXE attacks
        no_network=True,         # Block external network access
        huge_tree=False,         # Prevent billion laughs / memory exhaustion
    )


def load_sdlxliff(file_path: Path) -> Tuple[etree._ElementTree, etree._Element]:
    """
    Load and parse an SDLXLIFF file with security checks.

    Args:
        file_path: Path to the SDLXLIFF file

    Returns:
        Tuple of (ElementTree, root Element)

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file exceeds size limit
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file size to prevent memory exhaustion
    file_size = file_path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large: {file_size / (1024*1024):.1f}MB "
            f"(max: {MAX_FILE_SIZE / (1024*1024):.0f}MB)"
        )

    parser = create_secure_parser()
    tree = etree.parse(str(file_path), parser)
    root = tree.getroot()

    return tree, root


def detect_bom(file_path: Path) -> bool:
    """
    Check if a file has a UTF-8 BOM.

    Args:
        file_path: Path to check

    Returns:
        True if file starts with UTF-8 BOM
    """
    try:
        with open(file_path, 'rb') as f:
            return f.read(3) == b'\xef\xbb\xbf'
    except (IOError, OSError):
        return False


def save_sdlxliff(
    root: etree._Element,
    output_path: Path,
    original_path: Path,
) -> None:
    """
    Save an SDLXLIFF file using atomic write pattern.

    Uses atomic write pattern (write to temp file, then rename) to prevent
    file corruption if the process crashes during write. Previous file
    states are preserved by the version history (versioning.py), so no
    .bak backup file is created.

    Preserves:
    - UTF-8 BOM if present in original
    - XML declaration format

    Args:
        root: The root XML element to save
        output_path: Where to save the file
        original_path: Original file path (for BOM detection)

    Raises:
        IOError: If file cannot be written
    """
    # Check if original file had BOM
    has_bom = detect_bom(original_path)

    # Generate XML content
    xml_content = etree.tostring(
        root,
        encoding='unicode',
        pretty_print=False,
    )

    # Prepare the content to write
    content_bytes = b''
    if has_bom:
        content_bytes += b'\xef\xbb\xbf'
    content_bytes += b'<?xml version="1.0" encoding="utf-8"?>'
    content_bytes += xml_content.encode('utf-8')

    # Write to temp file first (atomic write pattern)
    # Create temp file in the same directory to ensure same filesystem for rename
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='.sdlxliff_',
            dir=output_path.parent
        )
        os.write(temp_fd, content_bytes)
        os.close(temp_fd)
        temp_fd = None  # Mark as closed

        # Atomic rename (on same filesystem, this is atomic on POSIX)
        os.replace(temp_path, str(output_path))
        logger.debug(f"Saved file: {output_path}")

    except Exception:
        # Clean up temp file on error
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise