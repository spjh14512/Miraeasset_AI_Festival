from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import system_prompts as sp
from .llm import (
    COMPACTOR_MAX_COMPLETION_TOKENS,
    MAX_LLM_RETRIES,
    bind_structured_output,
    build_output_retry_message,
    get_llm,
    invoke_with_rate_limit_retry,
)
from .state import Plan


DEFAULT_CONTENT_CHARACTER_LIMIT = 4_000
DEFAULT_ITEM_COUNT_LIMIT = 30
COMPACTABLE_POINT_KINDS = {"KV_TABLE", "R_TABLE"}


class CompactorOutput(BaseModel):
    """Compactor가 point에서 보존하기로 선택한 item ID 목록입니다."""

    item_ids: list[int] = Field(
        default_factory=list,
        description=(
            "보존할 item ID 목록. KV_TABLE은 entry index, "
            "R_TABLE은 record_index를 사용합니다."
        ),
    )

    @field_validator("item_ids")
    @classmethod
    def validate_item_ids(cls, item_ids: list[int]) -> list[int]:
        if any(item_id < 0 for item_id in item_ids):
            raise ValueError("item_ids는 0 이상의 정수여야 합니다.")
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("item_ids에 중복 값을 사용할 수 없습니다.")
        return item_ids


@lru_cache(maxsize=1)
def _get_compactor_llm() -> Any:
    return bind_structured_output(
        get_llm(COMPACTOR_MAX_COMPLETION_TOKENS),
        CompactorOutput,
    )


def _point_value(point: Any, key: str) -> Any:
    if isinstance(point, Mapping):
        return point.get(key)
    return getattr(point, key, None)


def _point_payload(point: Any) -> Mapping[str, Any]:
    payload = _point_value(point, "payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Qdrant point payload must be a mapping")
    return payload


def _point_kind(payload: Mapping[str, Any]) -> str:
    point_kind = payload.get("point_kind")
    if not isinstance(point_kind, str):
        raise ValueError("Qdrant payload.point_kind must be a string")
    return point_kind


def _canonical_items(payload: Mapping[str, Any]) -> list[Any] | None:
    point_kind = _point_kind(payload)
    if point_kind not in COMPACTABLE_POINT_KINDS:
        return None
    canonical = payload.get("canonical")
    if not isinstance(canonical, Mapping):
        raise ValueError("Qdrant payload.canonical must be a mapping")
    field = "entries" if point_kind == "KV_TABLE" else "records"
    items = canonical.get(field)
    if not isinstance(items, list):
        raise ValueError(f"{point_kind} canonical.{field} must be a list")
    return items


def point_requires_compaction(
    point: Any,
    *,
    content_character_limit: int = DEFAULT_CONTENT_CHARACTER_LIMIT,
    item_count_limit: int = DEFAULT_ITEM_COUNT_LIMIT,
) -> bool:
    """Point의 직렬화 길이나 item 수가 기준을 초과하는지 판단합니다.

    입력 예시:
        point_requires_compaction(
            kv_point,
            content_character_limit=4000,
            item_count_limit=30,
        )

    출력 예시:
        True
    """

    if content_character_limit <= 0:
        raise ValueError("content_character_limit는 양수여야 합니다.")
    if item_count_limit <= 0:
        raise ValueError("item_count_limit는 양수여야 합니다.")
    items = _canonical_items(_point_payload(point))
    if items is None:
        return False
    if len(items) > item_count_limit:
        return True
    serialized = json.dumps(
        items,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(serialized) > content_character_limit


def _compactor_input(point: Any, plan: Plan) -> tuple[dict[str, Any], list[int]]:
    payload = _point_payload(point)
    point_kind = _point_kind(payload)
    if point_kind not in COMPACTABLE_POINT_KINDS:
        raise ValueError(f"Compactor가 지원하지 않는 point_kind입니다: {point_kind}")

    canonical = payload.get("canonical")
    if not isinstance(canonical, Mapping):
        raise ValueError("Qdrant payload.canonical must be a mapping")

    table_context: dict[str, Any] = {}
    table_metadata = canonical.get("table_metadata")
    if table_metadata is not None:
        if not isinstance(table_metadata, Mapping):
            raise ValueError("canonical.table_metadata must be a mapping")
        table_context["table_metadata"] = dict(table_metadata)

    if point_kind == "KV_TABLE":
        entries = _canonical_items(payload)
        assert entries is not None
        items = []
        for item_id, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"KV_TABLE canonical.entries[{item_id}] must be a mapping")
            items.append({"item_id": item_id, **dict(entry)})
        item_ids = list(range(len(entries)))
    else:
        headers = canonical.get("headers")
        if not isinstance(headers, list):
            raise ValueError("R_TABLE canonical.headers must be a list")
        table_context["headers"] = headers
        records = _canonical_items(payload)
        assert records is not None
        items = []
        item_ids = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"R_TABLE canonical.records[{index}] must be a mapping")
            record_index = record.get("record_index")
            if not isinstance(record_index, int) or isinstance(record_index, bool):
                raise ValueError(
                    f"R_TABLE canonical.records[{index}].record_index must be an integer"
                )
            item_ids.append(record_index)
            items.append({"item_id": record_index, "values": record.get("values")})

    contextual_text = payload.get("contextual_text")
    if isinstance(contextual_text, str):
        table_context["retrieval_context"] = contextual_text.split("\n\n", 1)[0]

    return (
        {
            "plan": plan.model_dump(mode="json"),
            "point_kind": point_kind,
            **table_context,
            "items": items,
        },
        item_ids,
    )


def compact_qdrant_point(
    point: Any,
    plan: Plan,
    *,
    llm: Any | None = None,
) -> list[int]:
    """하나의 KV/R_TABLE point에서 보존할 item ID를 선택합니다.

    입력 예시:
        compact_qdrant_point(point, Plan(...))

    출력 예시:
        KV_TABLE이면 [0, 3], R_TABLE이면 [12, 15]
    """

    compactor_input, available_item_ids = _compactor_input(point, plan)
    compactor_llm = llm or _get_compactor_llm()
    messages = [
        SystemMessage(content=sp.QDRANT_POINT_COMPACTOR_SYSTEM_PROMPT),
        HumanMessage(
            content=json.dumps(
                compactor_input,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
    ]
    available = set(available_item_ids)
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            result = invoke_with_rate_limit_retry(compactor_llm, messages)
            output = (
                result
                if isinstance(result, CompactorOutput)
                else CompactorOutput.model_validate(result)
            )
            invalid_item_ids = [
                item_id
                for item_id in output.item_ids
                if item_id not in available
            ]
            if invalid_item_ids:
                raise ValueError(
                    "Compactor가 point에 없는 item ID를 반환했습니다: "
                    + ", ".join(map(str, invalid_item_ids))
                )
            selected = set(output.item_ids)
            break
        except (ValidationError, ValueError, TypeError) as error:
            if attempt == MAX_LLM_RETRIES:
                raise
            messages.append(HumanMessage(content=build_output_retry_message(
                "CompactorOutput",
                error,
            )))

    selected_item_ids = [
        item_id for item_id in available_item_ids if item_id in selected
    ]
    print("[Compactor 호출 결과]")
    print("선택 :", selected_item_ids)

    return selected_item_ids


__all__ = [
    "CompactorOutput",
    "DEFAULT_CONTENT_CHARACTER_LIMIT",
    "DEFAULT_ITEM_COUNT_LIMIT",
    "compact_qdrant_point",
    "point_requires_compaction",
]
