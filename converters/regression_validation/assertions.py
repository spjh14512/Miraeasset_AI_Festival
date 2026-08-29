"""Semantic assertions for fixed converter regression fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from converters.regression_validation.normalizer import normalize_duplicate_value


@dataclass(frozen=True, slots=True)
class AssertionFailure:
    case_id: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "code": self.code,
            "message": self.message,
        }


def _evidence(fragment: Mapping[str, Any], evidence_id: str) -> Mapping[str, Any]:
    for item in fragment.get("evidence_list", []):
        if item.get("evidence_id") == evidence_id:
            return item
    raise KeyError(f"Evidence not found: {evidence_id}")


def _table(fragment: Mapping[str, Any], table_id: str) -> Mapping[str, Any]:
    for item in fragment.get("evidence_list", []):
        payload = item.get("payload", {})
        if isinstance(payload, Mapping) and payload.get("table_id") == table_id:
            return item
    raise KeyError(f"Table evidence not found: {table_id}")


def _record(
    fragment: Mapping[str, Any], table_id: str, record_index: int
) -> Mapping[str, Any]:
    for item in fragment.get("records", []):
        if item.get("table_id") == table_id and item.get("record_index") == record_index:
            return item
    raise KeyError(f"Record not found: {table_id}#{record_index}")


def _path(value: Any, path: Sequence[str | int]) -> Any:
    current = value
    for part in path:
        current = current[part]
    return current


def evaluate_case(
    case: Mapping[str, Any],
    fragments: Mapping[str, Mapping[str, Any]],
) -> list[AssertionFailure]:
    case_id = str(case["case_id"])
    failures: list[AssertionFailure] = []
    for assertion in case.get("assertions", []):
        operation = str(assertion["op"])
        try:
            section_id = str(assertion.get("section_id") or case["section_ids"][0])
            fragment = fragments[section_id]
            if operation == "section_counts":
                actual = {
                    "evidence": len(fragment.get("evidence_list", [])),
                    "records": len(fragment.get("records", [])),
                }
                expected = {
                    "evidence": int(assertion["evidence"]),
                    "records": int(assertion["records"]),
                }
            elif operation == "evidence_path_equals":
                item = _evidence(fragment, str(assertion["evidence_id"]))
                actual = _path(item, assertion["path"])
                expected = assertion["value"]
            elif operation == "evidence_path_not_contains":
                item = _evidence(fragment, str(assertion["evidence_id"]))
                expected = assertion["value"]
                try:
                    actual = _path(item, assertion["path"])
                except (KeyError, IndexError):
                    continue
                if expected not in actual:
                    continue
                raise AssertionError(f"path unexpectedly contains {expected!r}")
            elif operation == "table_path_equals":
                item = _table(fragment, str(assertion["table_id"]))
                actual = _path(item, assertion["path"])
                expected = assertion["value"]
            elif operation == "kv_field_equals":
                item = _evidence(fragment, str(assertion["evidence_id"]))
                expected_paths = assertion["key_paths"]
                matching = [
                    field
                    for field in item.get("payload", {}).get("fields", [])
                    if field.get("key_paths") == expected_paths
                ]
                actual = matching[0].get("raw_value") if matching else None
                expected = assertion["raw_value"]
            elif operation == "record_equals":
                item = _record(
                    fragment,
                    str(assertion["table_id"]),
                    int(assertion["record_index"]),
                )
                actual = {
                    key: item.get(key)
                    for key in ("row_type", "row_context", "values")
                    if key in assertion
                }
                expected = {
                    key: assertion[key]
                    for key in ("row_type", "row_context", "values")
                    if key in assertion
                }
            elif operation == "text_contains":
                item = _evidence(fragment, str(assertion["evidence_id"]))
                actual = str(item.get("payload", {}).get("text", ""))
                expected = str(assertion["value"])
                if expected in actual:
                    continue
                raise AssertionError(f"text does not contain {expected!r}")
            elif operation == "table_title_not_navigation":
                item = _table(fragment, str(assertion["table_id"]))
                actual = item.get("payload", {}).get("title")
                if actual is None or not re.search(r"본문\s*위치로\s*이동", str(actual)):
                    continue
                raise AssertionError(f"navigation text became table title: {actual!r}")
            elif operation == "pair_relation":
                left = fragments[str(assertion["left_section_id"])]
                right = fragments[str(assertion["right_section_id"])]
                left_value = normalize_duplicate_value(left)
                right_value = normalize_duplicate_value(right)
                actual = left_value == right_value
                expected = assertion["relation"] == "SEMANTIC_EQUAL"
            else:
                raise ValueError(f"Unsupported assertion operation: {operation}")
            if actual != expected:
                raise AssertionError(f"expected {expected!r}, got {actual!r}")
        except Exception as error:
            failures.append(
                AssertionFailure(
                    case_id=case_id,
                    code="SEMANTIC_ASSERTION_FAILED",
                    message=f"{operation}: {type(error).__name__}: {error}",
                )
            )
    return failures


__all__ = ["AssertionFailure", "evaluate_case"]
