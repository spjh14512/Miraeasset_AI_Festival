"""Select an embedding strategy for one complete R_TABLE."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from vector_db.bgem3_token_counter import count_bge_m3_tokens


DEFAULT_CONFIG_PATH = Path(__file__).with_name("r_table_embedding.yaml")
CONFIG_SCHEMA_VERSION = "r-table-embedding.v1"

TokenCounter = Callable[[str], int]

_DATE_PATTERNS = (
    re.compile(r"^\d{4}[-./]\d{1,2}(?:[-./]\d{1,2})?$"),
    re.compile(r"^\d{4}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?$"),
    re.compile(r"^\d{8}$"),
    re.compile(r"^\d{4}\s*[Qq][1-4]$"),
)
_NUMERIC_PATTERN = re.compile(
    r"""
    ^\s*
    [\(\[]?
    [+-△▲▼]?
    \s*[₩$€¥]?\s*
    (?:\d{1,3}(?:,\d{3})+|\d+)
    (?:\.\d+)?
    \s*
    (?:%|퍼센트|원|천원|백만원|억원|조원|주|개|명|건|톤|kg|g|km|m|㎡)?
    \s*[\)\]]?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_NATURAL_LANGUAGE_PATTERN = re.compile(r"[A-Za-z가-힣]")


class RTableEmbeddingStrategy(StrEnum):
    WHOLE_TABLE = "whole_table"
    DESCRIPTOR = "descriptor"
    ROW_GROUP = "row_group"


@dataclass(frozen=True, slots=True)
class RTableStrategyConfig:
    whole_table_max_tokens: int
    row_group_max_tokens: int
    long_text_min_characters: int
    semantic_text_ratio_threshold: float

    def __post_init__(self) -> None:
        for name, value in (
            ("whole_table_max_tokens", self.whole_table_max_tokens),
            ("row_group_max_tokens", self.row_group_max_tokens),
            ("long_text_min_characters", self.long_text_min_characters),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        ratio = self.semantic_text_ratio_threshold
        if (
            not isinstance(ratio, (int, float))
            or isinstance(ratio, bool)
            or not 0 <= float(ratio) <= 1
        ):
            raise ValueError(
                "semantic_text_ratio_threshold must be between 0 and 1"
            )


@dataclass(frozen=True, slots=True)
class RTableFeatures:
    token_count: int
    row_count: int
    column_count: int
    non_empty_cell_count: int
    numeric_cell_ratio: float
    date_cell_ratio: float
    long_text_cell_ratio: float
    avg_cell_length: float
    max_cell_length: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RTableStrategyDecision:
    strategy: RTableEmbeddingStrategy
    reason: str
    features: RTableFeatures

    def to_dict(self) -> dict[str, Any]:
        features = self.features.to_dict()
        features.pop("token_count")
        return {
            "embedding_strategy": self.strategy.value,
            "embedding_token_count": self.features.token_count,
            "embedding_strategy_reason": self.reason,
            "features": features,
        }


def load_r_table_strategy_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> RTableStrategyConfig:
    """Load and validate classifier thresholds from YAML."""
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Failed to load R_TABLE strategy config: {config_path}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("R_TABLE strategy config must be a mapping")
    if loaded.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"R_TABLE strategy config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    values = loaded.get("r_table_embedding")
    if not isinstance(values, Mapping):
        raise ValueError("r_table_embedding config must be a mapping")
    expected_fields = {
        "whole_table_max_tokens",
        "row_group_max_tokens",
        "long_text_min_characters",
        "semantic_text_ratio_threshold",
    }
    if set(values) != expected_fields:
        raise ValueError(
            "r_table_embedding config fields must be exactly: "
            + ", ".join(sorted(expected_fields))
        )
    return RTableStrategyConfig(
        whole_table_max_tokens=values["whole_table_max_tokens"],
        row_group_max_tokens=values["row_group_max_tokens"],
        long_text_min_characters=values["long_text_min_characters"],
        semantic_text_ratio_threshold=values["semantic_text_ratio_threshold"],
    )


def is_date_cell(value: str) -> bool:
    return any(pattern.fullmatch(value) for pattern in _DATE_PATTERNS)


def is_numeric_cell(value: str) -> bool:
    return _NUMERIC_PATTERN.fullmatch(value) is not None


def _validated_cells(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, list[str]]:
    if evidence.get("evidence_type") != "TABLE":
        raise ValueError("Expected evidence_type to be TABLE")
    if evidence.get("table_type") != "R_TABLE":
        raise ValueError("Expected table_type to be R_TABLE")

    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("R_TABLE Evidence payload must be a mapping")
    table_id = payload.get("table_id")
    if not isinstance(table_id, str) or not table_id.strip():
        raise ValueError("R_TABLE payload.table_id must be a non-empty string")
    headers = payload.get("headers")
    if not isinstance(headers, list) or not all(
        isinstance(header, list)
        and all(isinstance(part, str) for part in header)
        for header in headers
    ):
        raise ValueError("R_TABLE payload.headers must be a list of string lists")

    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 0
    ):
        raise ValueError("R_TABLE payload.record_count must be a non-negative integer")
    if not isinstance(records, (list, tuple)):
        raise ValueError("R_TABLE records must be a sequence")
    if record_count != len(records):
        raise ValueError("R_TABLE payload.record_count must match the records length")

    cells: list[str] = []
    record_indexes: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each R_TABLE record must be a mapping")
        if record.get("table_id") != table_id:
            raise ValueError("R_TABLE Evidence and record table_id must match")
        record_index = record.get("record_index")
        if (
            not isinstance(record_index, int)
            or isinstance(record_index, bool)
            or record_index < 0
        ):
            raise ValueError("R_TABLE record_index must be a non-negative integer")
        if record_index in record_indexes:
            raise ValueError("R_TABLE record_index must be unique within a table")
        record_indexes.add(record_index)

        values = record.get("values")
        if not isinstance(values, list) or not all(
            value is None or isinstance(value, str) for value in values
        ):
            raise ValueError("R_TABLE record.values must be a list of strings or nulls")
        if len(values) != len(headers):
            raise ValueError("R_TABLE headers and values must have the same length")
        cells.extend(
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
    return len(headers), cells


def calculate_r_table_features(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    contextual_text: str,
    *,
    config: RTableStrategyConfig,
    token_counter: TokenCounter = count_bge_m3_tokens,
) -> RTableFeatures:
    """Calculate deterministic features without changing canonical records."""
    if not isinstance(contextual_text, str) or not contextual_text.strip():
        raise ValueError("contextual_text must be a non-empty string")
    if not callable(token_counter):
        raise ValueError("token_counter must be callable")

    column_count, cells = _validated_cells(evidence, records)
    token_count = token_counter(contextual_text)
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count < 0
    ):
        raise ValueError("token_counter must return a non-negative integer")

    non_empty_cell_count = len(cells)
    lengths = [len(value) for value in cells]
    numeric_count = sum(is_numeric_cell(value) for value in cells)
    date_count = sum(is_date_cell(value) for value in cells)
    long_text_count = sum(
        len(value) >= config.long_text_min_characters
        and _NATURAL_LANGUAGE_PATTERN.search(value) is not None
        for value in cells
    )
    denominator = non_empty_cell_count or 1
    return RTableFeatures(
        token_count=token_count,
        row_count=len(records),
        column_count=column_count,
        non_empty_cell_count=non_empty_cell_count,
        numeric_cell_ratio=numeric_count / denominator,
        date_cell_ratio=date_count / denominator,
        long_text_cell_ratio=long_text_count / denominator,
        avg_cell_length=sum(lengths) / denominator,
        max_cell_length=max(lengths, default=0),
    )


class RTableEmbeddingStrategySelector:
    """Classify an R_TABLE without building descriptor or row-group texts."""

    def __init__(
        self,
        *,
        config: RTableStrategyConfig | None = None,
        token_counter: TokenCounter = count_bge_m3_tokens,
    ) -> None:
        self.config = config or load_r_table_strategy_config()
        if not callable(token_counter):
            raise ValueError("token_counter must be callable")
        self.token_counter = token_counter

    def select(
        self,
        evidence: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        contextual_text: str,
    ) -> RTableStrategyDecision:
        features = calculate_r_table_features(
            evidence,
            records,
            contextual_text,
            config=self.config,
            token_counter=self.token_counter,
        )
        if features.token_count <= self.config.whole_table_max_tokens:
            return RTableStrategyDecision(
                strategy=RTableEmbeddingStrategy.WHOLE_TABLE,
                reason="within_whole_table_token_limit",
                features=features,
            )
        if (
            features.long_text_cell_ratio
            >= self.config.semantic_text_ratio_threshold
        ):
            return RTableStrategyDecision(
                strategy=RTableEmbeddingStrategy.ROW_GROUP,
                reason="semantic_oversized",
                features=features,
            )
        return RTableStrategyDecision(
            strategy=RTableEmbeddingStrategy.DESCRIPTOR,
            reason="structured_oversized",
            features=features,
        )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "RTableEmbeddingStrategy",
    "RTableEmbeddingStrategySelector",
    "RTableFeatures",
    "RTableStrategyConfig",
    "RTableStrategyDecision",
    "calculate_r_table_features",
    "is_date_cell",
    "is_numeric_cell",
    "load_r_table_strategy_config",
]
