"""Normalize R_TABLE headers for compact retrieval text."""

from __future__ import annotations

from collections.abc import Sequence


PATH_SEPARATOR = " > "


def normalize_header_path(path: Sequence[str]) -> tuple[str, ...]:
    """Strip blanks and collapse adjacent duplicates without changing canonical data."""
    if not isinstance(path, (list, tuple)) or not all(
        isinstance(part, str) for part in path
    ):
        raise ValueError("R_TABLE header path must be a sequence of strings")

    normalized: list[str] = []
    for part in path:
        cleaned = part.strip()
        if cleaned and (not normalized or normalized[-1] != cleaned):
            normalized.append(cleaned)
    return tuple(normalized)


def normalize_header_paths(headers: object) -> list[tuple[str, ...]]:
    if not isinstance(headers, list) or not all(
        isinstance(header, list)
        and all(isinstance(part, str) for part in header)
        for header in headers
    ):
        raise ValueError("R_TABLE payload.headers must be a list of string lists")
    return [normalize_header_path(header) for header in headers]


def record_header_keys(headers: object) -> list[str]:
    """Return one stable key per source column, disambiguating exact duplicates."""
    paths = normalize_header_paths(headers)
    counts: dict[str, int] = {}
    keys: list[str] = []
    for column_index, path in enumerate(paths):
        base = PATH_SEPARATOR.join(path) or f"컬럼 {column_index + 1}"
        occurrence = counts.get(base, 0) + 1
        counts[base] = occurrence
        keys.append(base if occurrence == 1 else f"{base} [{occurrence}]")
    return keys


def grouped_header_lines(headers: object) -> list[str]:
    """Group sibling columns by their common parent for compact display."""
    unique_paths = list(dict.fromkeys(
        path for path in normalize_header_paths(headers) if path
    ))
    output: list[str | tuple[str, ...]] = []
    grouped_leaves: dict[tuple[str, ...], list[str]] = {}
    group_positions: dict[tuple[str, ...], int] = {}

    for path in unique_paths:
        if len(path) == 1:
            output.append(path)
            continue
        parent = path[:-1]
        if parent not in grouped_leaves:
            grouped_leaves[parent] = []
            group_positions[parent] = len(output)
            output.append(parent)
        leaf = path[-1]
        if leaf not in grouped_leaves[parent]:
            grouped_leaves[parent].append(leaf)

    lines: list[str] = []
    for position, item in enumerate(output):
        if item in grouped_leaves and group_positions[item] == position:
            leaves = grouped_leaves[item]
            if len(leaves) > 1:
                lines.append(
                    f"{PATH_SEPARATOR.join(item)} : {' | '.join(leaves)}"
                )
            else:
                lines.append(PATH_SEPARATOR.join((*item, leaves[0])))
        else:
            lines.append(PATH_SEPARATOR.join(item))
    return lines


__all__ = [
    "PATH_SEPARATOR",
    "grouped_header_lines",
    "normalize_header_path",
    "normalize_header_paths",
    "record_header_keys",
]
