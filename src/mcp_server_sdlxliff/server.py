"""
MCP Server for SDLXLIFF File Operations

This server exposes tools for reading, analyzing, and modifying SDLXLIFF files
through the Model Context Protocol (MCP).
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
from mcp.server.stdio import stdio_server
import logging

from .cache import (
    get_parser,
    clear_parser_cache,
    resolve_file_path,
    validate_file_extension,
)
from .parser import SDLXLIFFParser
from . import versioning
from .versioning import VersioningError
from .qa import (
    run_qa_checks,
    QAReport,
    load_glossary,
    discover_glossary,
    load_custom_dictionary,
    discover_custom_dictionary,
)
from .languages import is_language_supported


def setup_logging():
    """
    Set up logging to stderr only.

    No log files are ever written: translation content is client-confidential
    and must not be persisted outside the user's working folder. Log messages
    never include tool arguments or segment text.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    return logging.getLogger("sdlxliff-server")


logger = setup_logging()
logger.info("=== MCP Server Starting ===")

# Create the MCP server instance
app = Server("sdlxliff-server")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """
    List available SDLXLIFF files as resources.

    Note: Returns empty list as file discovery should use the built-in
    filesystem server's search_files capability.
    """
    logger.info("list_resources called - returning empty (use filesystem search)")
    return []


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    logger.info(f"read_resource called with URI: {uri}")

    # Extract file path from URI
    if uri.startswith("sdlxliff:///"):
        file_path = uri.replace("sdlxliff:///", "")
        parser = get_parser(file_path)
        segments = parser.extract_segments()

        return json.dumps({
            "file": file_path,
            "segments": segments,
        }, indent=2, ensure_ascii=False)

    raise ValueError(f"Unknown resource URI: {uri}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available SDLXLIFF tools."""
    return [
        Tool(
            name="read_sdlxliff",
            description=(
                "Extract translation segments from an SDLXLIFF file. "
                "Returns segment IDs, source text, target text, status, locked state, percent (TM match), and origin. "
                "Maximum 50 segments per request (enforced). Use offset parameter to paginate through large files. "
                "Filtering: use max_percent to exclude high TM matches (e.g., max_percent=99 excludes 100% matches), "
                "use skip_cm=true to exclude Context Matches. "
                "Use include_tags=true only when you need to UPDATE segments with formatting tags. "
                "ALWAYS use this tool to read SDLXLIFF files - DO NOT write Python code to parse XML."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Full path to the SDLXLIFF file",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting segment index (0-based). Use with limit for pagination. Default: 0",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Number of segments to return (max 50, enforced). Default: 50."
                        ),
                    },
                    "include_tags": {
                        "type": "boolean",
                        "description": (
                            "If true, includes source_tagged/target_tagged fields with tag placeholders. "
                            "Only needed when planning to update segments with formatting tags. "
                            "Default: false (smaller output)."
                        ),
                        "default": False,
                    },
                    "max_percent": {
                        "type": "integer",
                        "description": (
                            "Filter to exclude high-match segments. Only returns segments with "
                            "match percent <= this value (or no percent). "
                            "Example: max_percent=99 excludes 100% TM matches. "
                            "Use when client requests not to touch pre-translated/approved 100% segments. "
                            "Default: no filtering (returns all segments)."
                        ),
                    },
                    "skip_cm": {
                        "type": "boolean",
                        "description": (
                            "Skip Context Matches (CM). CMs are 100% matches where both source, "
                            "target AND surrounding context match the TM. "
                            "Use when client says 'skip CMs' or 'don't touch context matches'. "
                            "Default: false."
                        ),
                        "default": False,
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="get_sdlxliff_segment",
            description=(
                "Get a specific segment from an SDLXLIFF file by its segment ID. "
                "Returns the segment's source text, target text, status, locked state, and tag information. "
                "For segments with inline tags, both clean and tagged versions are provided."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the SDLXLIFF file (can be relative or absolute)",
                    },
                    "segment_id": {
                        "type": "string",
                        "description": "The segment ID to retrieve",
                    },
                },
                "required": ["file_path", "segment_id"],
            },
        ),
        Tool(
            name="update_sdlxliff_segment",
            description=(
                "Update a segment's target text and set status to RejectedTranslation. "
                "Use this to correct translations. The segment_id is the mrk mid (e.g., '1', '2', '42'). "
                "IMPORTANT: For segments with formatting tags (has_tags=true), you MUST include "
                "tag placeholders in target_text to preserve formatting. "
                "Format: {id}text{/id} for paired tags, {x:id} for self-closing. "
                "Example: '{5}Acme{/5}{6}&{/6}{7} Events{/7}'. "
                "If tags are missing or malformed, the update will be rejected with an error. "
                "Changes are made in memory; you must call save_sdlxliff to persist changes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the SDLXLIFF file",
                    },
                    "segment_id": {
                        "type": "string",
                        "description": "The segment ID (mrk mid) to update",
                    },
                    "target_text": {
                        "type": "string",
                        "description": (
                            "New target text for the segment. For segments with tags, "
                            "include placeholders like {5}text{/5} or {x:5}"
                        ),
                    },
                    "preserve_tags": {
                        "type": "boolean",
                        "description": (
                            "If true (default), validates and restores tags from placeholders. "
                            "If false, strips all tags and uses plain text."
                        ),
                        "default": True,
                    },
                },
                "required": ["file_path", "segment_id", "target_text"],
            },
        ),
        Tool(
            name="save_sdlxliff",
            description=(
                "Save changes made to an SDLXLIFF file. All modifications from "
                "update_sdlxliff_segment are kept in memory until this tool is called. "
                "Can optionally save to a different file path. "
                "A snapshot of the file is kept automatically on every save, so any "
                "earlier state can be reviewed or brought back later "
                "(see list_file_history, diff_versions, restore_version)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the SDLXLIFF file to save",
                    },
                    "output_path": {
                        "type": "string",
                        "description": (
                            "Optional output path. If not provided, overwrites the original file."
                        ),
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="get_sdlxliff_statistics",
            description=(
                "Get statistics and metadata about an SDLXLIFF file. Returns source/target "
                "language codes (e.g., 'en-US' -> 'de-DE'), total segment count, counts by "
                "status, and locked segment count. Call this first to understand the file "
                "before reading segments."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the SDLXLIFF file (can be relative or absolute)",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="validate_sdlxliff_segment",
            description=(
                "Validate proposed changes to a segment before updating. "
                "Checks that all required tags are present and properly formatted. "
                "Use this to pre-validate translations before calling update_sdlxliff_segment. "
                "Returns validation result with any errors or warnings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the SDLXLIFF file",
                    },
                    "segment_id": {
                        "type": "string",
                        "description": "The segment ID (mrk mid) to validate against",
                    },
                    "target_text": {
                        "type": "string",
                        "description": (
                            "Proposed target text with tag placeholders to validate. "
                            "Format: {id}text{/id} for paired tags, {x:id} for self-closing."
                        ),
                    },
                },
                "required": ["file_path", "segment_id", "target_text"],
            },
        ),
        Tool(
            name="qa_check_sdlxliff",
            description=(
                "Run quality assurance checks on an SDLXLIFF file. "
                "ALWAYS use this tool (not custom scripts) for QA tasks. "
                "Default checks: trailing punctuation mismatches, missing/extra numbers, "
                "double spaces, whitespace mismatches, bracket mismatches, "
                "inconsistent repetitions (same source text translated differently), "
                "terminology (glossary compliance), and french_typography (only when the "
                "file's target language is French: non-breaking spaces before : ; ! ?, "
                "« guillemets » instead of English quotes, French number formatting like "
                "1 234,56; FR-FR vs FR-CA conventions via french_convention parameter). "
                "OPT-IN checks: spelling (must be explicitly requested via checks parameter). "
                "Spelling uses target language from file metadata; supports: en, de, es, fr, it, pt, nl (offline dictionaries, no network). "
                "For terminology check: auto-discovers glossary.tsv/txt in same folder as SDLXLIFF, "
                "or specify explicit glossary_path. "
                "For spelling check: auto-discovers dictionary.txt/custom_words.txt/spelling.txt in same folder, "
                "or specify explicit dictionary_path. "
                "Filtering: use max_percent to skip high TM matches (e.g., max_percent=99 excludes 100% matches), "
                "use skip_cm=true to exclude Context Matches. "
                "Use for: 'check translation quality', 'find errors', 'are translations consistent', "
                "'run QA', 'verify before delivery', 'check terminology', 'verify glossary', 'check spelling'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the SDLXLIFF file",
                    },
                    "segment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of segment IDs to check. "
                            "If not provided, checks all segments."
                        ),
                    },
                    "checks": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "trailing_punctuation",
                                "numbers",
                                "double_spaces",
                                "whitespace",
                                "brackets",
                                "inconsistent_repetitions",
                                "terminology",
                                "french_typography",
                                "spelling",
                            ],
                        },
                        "description": (
                            "Optional list of specific checks to run. "
                            "If not provided, runs default checks (all except spelling). "
                            "Spelling is OPT-IN: must be explicitly listed to run. "
                            "french_typography runs only when the file's target language is French. "
                            "Available: trailing_punctuation, numbers, double_spaces, "
                            "whitespace, brackets, inconsistent_repetitions, terminology, "
                            "french_typography, spelling."
                        ),
                    },
                    "french_convention": {
                        "type": "string",
                        "enum": ["fr-FR", "fr-CA"],
                        "description": (
                            "Typography convention for the french_typography check. "
                            "fr-FR (France): non-breaking space before : ; ! ?. "
                            "fr-CA (Canada/OQLF): non-breaking space before : only; "
                            "no space before ; ! ?. "
                            "Default: derived from the file's target language "
                            "(fr-CA target uses Canadian rules, other French uses fr-FR)."
                        ),
                    },
                    "glossary_path": {
                        "type": "string",
                        "description": (
                            "Optional path to glossary file (tab-delimited: source_term<TAB>target_term). "
                            "If not provided, auto-discovers glossary.tsv/glossary.txt/terminology.tsv/terminology.txt "
                            "in same directory as SDLXLIFF file."
                        ),
                    },
                    "dictionary_path": {
                        "type": "string",
                        "description": (
                            "Optional path to custom dictionary file (one word per line, # for comments). "
                            "Used by spelling check to ignore domain-specific terms. "
                            "If not provided, auto-discovers dictionary.txt/custom_words.txt/spelling.txt "
                            "in same directory as SDLXLIFF file."
                        ),
                    },
                    "max_percent": {
                        "type": "integer",
                        "description": (
                            "Filter to exclude high-match segments from QA. Only checks segments with "
                            "match percent <= this value (or no percent). "
                            "Example: max_percent=99 skips QA on 100% TM matches. "
                            "Use when client requests not to touch pre-translated segments."
                        ),
                    },
                    "skip_cm": {
                        "type": "boolean",
                        "description": (
                            "Skip Context Matches (CM) from QA. CMs are 100% matches where both source, "
                            "target AND surrounding context match the TM. "
                            "Use when client says 'skip CMs' or 'don't touch context matches'. "
                            "Default: false."
                        ),
                        "default": False,
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="list_file_history",
            description=(
                "Show the saved versions of an SDLXLIFF file. A snapshot is kept "
                "automatically every time the file is saved, so earlier states can "
                "always be reviewed or brought back - nothing is ever lost. "
                "Returns a dated list of versions (oldest first) with a summary of "
                "what changed in each. All history stays on this computer only. "
                "Use for: 'show the history of this file', 'what versions are there', "
                "'when was this file last saved'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the SDLXLIFF file",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="diff_versions",
            description=(
                "Compare a saved version of an SDLXLIFF file with its current "
                "content, translation by translation. For each changed segment, "
                "shows the segment ID, the translation as it was in that version, "
                "and the translation as it is now (readable text, not raw XML). "
                "Use list_file_history first to see the available version numbers. "
                "Use for: 'what changed since version 2', 'what did my last save "
                "change', 'review my edits'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the SDLXLIFF file",
                    },
                    "version": {
                        "type": "integer",
                        "description": (
                            "Version number to compare against the current file "
                            "(from list_file_history; 1 is the oldest)"
                        ),
                    },
                },
                "required": ["file_path", "version"],
            },
        ),
        Tool(
            name="restore_version",
            description=(
                "Bring back an earlier version of an SDLXLIFF file, replacing its "
                "current content in one step. The current state is snapshotted "
                "first, so a restore can itself be undone the same way - nothing is "
                "ever lost. To undo the most recent save, restore the version just "
                "before it (see list_file_history). "
                "Use for: 'undo my last save', 'go back to yesterday's version', "
                "'restore version 3'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the SDLXLIFF file",
                    },
                    "version": {
                        "type": "integer",
                        "description": (
                            "Version number to restore "
                            "(from list_file_history; 1 is the oldest)"
                        ),
                    },
                },
                "required": ["file_path", "version"],
            },
        ),
    ]


def _language_pair_suffix(metadata: dict) -> str:
    """Build a language-pair suffix like ' (EN→FR)' from file metadata."""
    source = metadata.get('source_language')
    target = metadata.get('target_language')
    if not source or not target:
        return ""
    return f" ({source.split('-')[0].upper()}→{target.split('-')[0].upper()})"


def _diff_segment_lists(old_segments: list, new_segments: list) -> list:
    """
    Compute a segment-level diff between two versions of a file.

    Returns a list of {segment_id, old_target, new_target} for every segment
    whose target text differs, in current-file order.
    """
    old_map = {seg['segment_id']: seg for seg in old_segments}
    new_ids = set()
    changes = []

    for seg in new_segments:
        seg_id = seg['segment_id']
        new_ids.add(seg_id)
        old_seg = old_map.get(seg_id)
        if old_seg is None:
            changes.append({
                'segment_id': seg_id,
                'old_target': None,
                'new_target': seg.get('target', ''),
            })
        elif old_seg.get('target', '') != seg.get('target', ''):
            changes.append({
                'segment_id': seg_id,
                'old_target': old_seg.get('target', ''),
                'new_target': seg.get('target', ''),
            })

    # Segments that existed in the old version but are gone now (rare)
    for seg in old_segments:
        if seg['segment_id'] not in new_ids:
            changes.append({
                'segment_id': seg['segment_id'],
                'old_target': seg.get('target', ''),
                'new_target': None,
            })

    return changes


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""

    # Log only the tool name - arguments contain file paths and client text
    logger.info(f"call_tool: {name}")

    try:
        if name == "read_sdlxliff":
            file_path = arguments["file_path"]
            include_tags = arguments.get("include_tags", False)
            offset = arguments.get("offset", 0)
            limit = arguments.get("limit")  # None means all
            max_percent = arguments.get("max_percent")  # None means no filtering
            skip_cm = arguments.get("skip_cm", False)  # Skip Context Matches

            parser = get_parser(file_path)
            all_segments = parser.extract_segments()
            total_count = len(all_segments)

            # Apply percent filter if specified
            if max_percent is not None:
                all_segments = [
                    seg for seg in all_segments
                    if seg.get('percent') is None or seg.get('percent') <= max_percent
                ]
                logger.info(f"After max_percent={max_percent} filter: {len(all_segments)} segments (was {total_count})")

            # Apply CM filter if specified (text_match="SourceAndTarget" indicates CM)
            if skip_cm:
                all_segments = [
                    seg for seg in all_segments
                    if seg.get('text_match') != 'SourceAndTarget'
                ]
                logger.info(f"After skip_cm filter: {len(all_segments)} segments")
            logger.info(f"Extracted {total_count} segments")

            # Enforce maximum limit to prevent token overflow
            MAX_SEGMENTS_PER_REQUEST = 50
            if limit is None or limit > MAX_SEGMENTS_PER_REQUEST:
                limit = MAX_SEGMENTS_PER_REQUEST

            # Apply pagination
            segments = all_segments[offset:offset + limit]

            # Strip tagged fields to reduce output size
            for seg in segments:
                if not include_tags or not seg.get('has_tags', False):
                    seg.pop('source_tagged', None)
                    seg.pop('target_tagged', None)

            # Build response with pagination metadata
            filtered_count = len(all_segments) if max_percent is not None else total_count
            response = {
                "total_segments": total_count,
                "filtered_segments": filtered_count if max_percent is not None else None,
                "offset": offset,
                "count": len(segments),
                "has_more": (offset + len(segments)) < filtered_count,
                "segments": segments,
            }
            # Remove null fields to save tokens
            response = {k: v for k, v in response.items() if v is not None}

            return [
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "get_sdlxliff_segment":
            file_path = arguments["file_path"]
            segment_id = arguments["segment_id"]
            parser = get_parser(file_path)
            segment = parser.get_segment_by_id(segment_id)

            if segment is None:
                return [
                    TextContent(
                        type="text",
                        text=f"Segment with ID '{segment_id}' not found.",
                    )
                ]

            # Strip tagged fields if segment has no tags (saves tokens)
            if not segment.get('has_tags', False):
                segment.pop('source_tagged', None)
                segment.pop('target_tagged', None)

            return [
                TextContent(
                    type="text",
                    text=json.dumps(segment, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "update_sdlxliff_segment":
            file_path = arguments["file_path"]
            segment_id = arguments["segment_id"]
            target_text = arguments["target_text"]
            preserve_tags = arguments.get("preserve_tags", True)

            parser = get_parser(file_path)
            result = parser.update_segment_with_tags(
                segment_id, target_text, preserve_tags=preserve_tags
            )

            if result['success']:
                response = {
                    "status": "success",
                    "message": f"Successfully updated segment '{segment_id}' (status set to RejectedTranslation). "
                               f"Remember to call save_sdlxliff to persist changes.",
                }
                if result.get('warnings'):
                    response["warnings"] = result['warnings']
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(response, indent=2, ensure_ascii=False),
                    )
                ]
            else:
                response = {
                    "status": "error",
                    "message": result['message'],
                }
                if result.get('validation'):
                    response["validation"] = result['validation']
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(response, indent=2, ensure_ascii=False),
                    )
                ]

        elif name == "save_sdlxliff":
            file_path = arguments["file_path"]
            output_path = arguments.get("output_path")

            # Validate output_path extension if provided
            if output_path:
                validate_file_extension(output_path)

            parser = get_parser(file_path)
            n_modified = len(parser.modified_segment_ids)
            lang_suffix = _language_pair_suffix(parser.get_file_metadata())

            save_location = Path(output_path).resolve() if output_path else parser.file_path
            history_warnings = []

            # Snapshot the file's current on-disk state before overwriting it
            warning = versioning.ensure_baseline(save_location)
            if warning:
                history_warnings.append(warning)

            parser.save(output_path)

            # Snapshot the newly saved state
            seg_word = "segment" if n_modified == 1 else "segments"
            warning = versioning.record_save(
                save_location,
                f"Updated {n_modified} {seg_word} in {save_location.name}{lang_suffix}",
            )
            if warning:
                history_warnings.append(warning)

            # Clear cache after saving
            clear_parser_cache(file_path)

            message = f"Successfully saved SDLXLIFF file to: {save_location}"
            if history_warnings:
                message += "\nNote: " + " ".join(history_warnings)
            else:
                message += "\nA snapshot of this version was kept automatically (see list_file_history)."
            return [
                TextContent(
                    type="text",
                    text=message,
                )
            ]

        elif name == "list_file_history":
            file_path = str(resolve_file_path(arguments["file_path"]))

            try:
                versions = versioning.get_history(Path(file_path))
            except VersioningError as e:
                return [TextContent(type="text", text=str(e))]

            response = {
                "file": file_path,
                "versions": [
                    {
                        "version": v.number,
                        "date": v.date,
                        "description": v.description,
                    }
                    for v in versions
                ],
            }
            return [
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "diff_versions":
            file_path = str(resolve_file_path(arguments["file_path"]))
            version = arguments["version"]

            try:
                versions = versioning.get_history(Path(file_path))
                old_content = versioning.get_version_content(Path(file_path), version)
            except VersioningError as e:
                return [TextContent(type="text", text=str(e))]

            version_info = next(v for v in versions if v.number == version)
            old_parser = SDLXLIFFParser.from_bytes(old_content, name=file_path)
            old_segments = old_parser.extract_segments()
            new_segments = get_parser(file_path).extract_segments()

            changes = _diff_segment_lists(old_segments, new_segments)

            response = {
                "file": file_path,
                "compared_to_version": version,
                "version_date": version_info.date,
                "version_description": version_info.description,
                "segments_changed": len(changes),
                "changes": changes,
            }
            return [
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "restore_version":
            file_path = str(resolve_file_path(arguments["file_path"]))
            version = arguments["version"]

            try:
                restored = versioning.restore_version(Path(file_path), version)
            except VersioningError as e:
                return [TextContent(type="text", text=str(e))]

            # The file on disk changed; drop any cached parser
            clear_parser_cache(file_path)

            return [
                TextContent(
                    type="text",
                    text=(
                        f"Restored {Path(file_path).name} to version {restored.number} "
                        f"from {restored.date} ({restored.description}). "
                        f"The previous state was kept as a new version, so this "
                        f"can be undone with restore_version if needed."
                    ),
                )
            ]

        elif name == "get_sdlxliff_statistics":
            file_path = arguments["file_path"]
            parser = get_parser(file_path)
            stats = parser.get_statistics()

            return [
                TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "validate_sdlxliff_segment":
            file_path = arguments["file_path"]
            segment_id = arguments["segment_id"]
            target_text = arguments["target_text"]

            parser = get_parser(file_path)
            validation = parser.validate_tagged_text(segment_id, target_text)

            # Get the original tagged text for reference
            segment = parser.get_segment_by_id(segment_id)
            if segment:
                validation['original_tagged'] = segment.get('target_tagged', '')
                validation['has_tags'] = segment.get('has_tags', False)

            return [
                TextContent(
                    type="text",
                    text=json.dumps(validation, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "qa_check_sdlxliff":
            file_path = arguments["file_path"]
            segment_ids = arguments.get("segment_ids")
            checks = arguments.get("checks")
            glossary_path = arguments.get("glossary_path")
            dictionary_path = arguments.get("dictionary_path")
            french_convention = arguments.get("french_convention")
            max_percent = arguments.get("max_percent")
            skip_cm = arguments.get("skip_cm", False)

            parser = get_parser(file_path)
            all_segments = parser.extract_segments()
            total_count = len(all_segments)

            # Get target language from file metadata (for spelling check)
            metadata = parser.get_file_metadata()
            target_lang = metadata.get('target_language')
            logger.info(f"QA: Target language from metadata: {target_lang}")

            # Apply percent filter if specified
            if max_percent is not None:
                all_segments = [
                    seg for seg in all_segments
                    if seg.get('percent') is None or seg.get('percent') <= max_percent
                ]
                logger.info(f"QA: After max_percent={max_percent} filter: {len(all_segments)} segments (was {total_count})")

            # Apply CM filter if specified
            if skip_cm:
                all_segments = [
                    seg for seg in all_segments
                    if seg.get('text_match') != 'SourceAndTarget'
                ]
                logger.info(f"QA: After skip_cm filter: {len(all_segments)} segments")

            # Filter segments if specific IDs provided
            if segment_ids:
                segment_id_set = set(segment_ids)
                segments_to_check = [
                    s for s in all_segments
                    if s['segment_id'] in segment_id_set
                ]
            else:
                segments_to_check = all_segments

            # Load glossary for terminology check
            glossary_terms = None
            used_glossary_path = None

            # If glossary_path provided, use it; otherwise auto-discover
            if glossary_path:
                glossary_terms = load_glossary(glossary_path)
                if glossary_terms:
                    used_glossary_path = glossary_path
            else:
                discovered = discover_glossary(file_path)
                if discovered:
                    glossary_terms = load_glossary(discovered)
                    if glossary_terms:
                        used_glossary_path = discovered

            # Load custom dictionary for spelling check (only if spelling requested)
            custom_words = None
            used_dictionary_path = None
            spelling_requested = checks and 'spelling' in checks

            if spelling_requested:
                if dictionary_path:
                    custom_words = load_custom_dictionary(dictionary_path)
                    if custom_words:
                        used_dictionary_path = dictionary_path
                else:
                    discovered_dict = discover_custom_dictionary(file_path)
                    if discovered_dict:
                        custom_words = load_custom_dictionary(discovered_dict)
                        if custom_words:
                            used_dictionary_path = discovered_dict
                logger.info(f"QA: Spelling check requested, custom dictionary: {used_dictionary_path}, words: {len(custom_words) if custom_words else 0}")

            # Run QA checks with glossary terms and spelling support
            report = run_qa_checks(
                segments_to_check,
                checks,
                glossary_terms,
                target_lang=target_lang,
                custom_words=custom_words,
                french_convention=french_convention,
            )

            # Convert to JSON-serializable format
            response = {
                "total_segments": total_count,
                "segments_checked": report.segments_checked,
                "segments_with_issues": report.segments_with_issues,
                "issues": [
                    {
                        "segment_id": issue.segment_id,
                        "check": issue.check,
                        "severity": issue.severity,
                        "message": issue.message,
                        "source_excerpt": issue.source_excerpt,
                        "target_excerpt": issue.target_excerpt,
                    }
                    for issue in report.issues
                ],
                "summary": report.summary,
            }

            # Add filter info if applied
            if max_percent is not None or skip_cm:
                response["segments_excluded"] = total_count - len(all_segments)
                if max_percent is not None:
                    response["filtered_by_max_percent"] = max_percent
                if skip_cm:
                    response["skipped_context_matches"] = True

            # Add glossary info to response
            if used_glossary_path:
                response["glossary_used"] = used_glossary_path
                response["glossary_terms_count"] = len(glossary_terms) if glossary_terms else 0

            # Add spelling info to response
            if spelling_requested:
                if target_lang:
                    response["target_language"] = target_lang
                if not is_language_supported(target_lang):
                    response["spelling_skipped"] = f"Language '{target_lang}' not supported for spelling check"
                if used_dictionary_path:
                    response["dictionary_used"] = used_dictionary_path
                    response["custom_words_count"] = len(custom_words) if custom_words else 0

            return [
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, ensure_ascii=False),
                )
            ]

        else:
            return [
                TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ]

    except FileNotFoundError as e:
        # Try to provide more helpful error message
        file_path = arguments.get("file_path", "unknown")
        resolved_path = str(Path(file_path).resolve())
        return [TextContent(
            type="text",
            text=f"File not found.\nRequested: {file_path}\nResolved to: {resolved_path}\nError: {str(e)}"
        )]
    except Exception as e:
        # Provide detailed error for debugging
        error_details = traceback.format_exc()
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}\n\nDetails:\n{error_details}"
        )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())