from __future__ import annotations

import json
from pathlib import Path

from converters.regression_validation.assertions import evaluate_case
from converters.regression_validation.normalizer import normalize_duplicate_value
from converters.regression_validation.runner import _compare_golden, _write_candidate, run_regression


def _fragment(text: str = "원문") -> dict:
    return {
        "section_id": "section:1:src0:s1",
        "evidence_list": [{
            "evidence_id": "evidence:1:src0:s1:e0",
            "evidence_type": "TEXT",
            "order": 0,
            "payload": {"text": text},
        }],
        "records": [],
    }


def test_semantic_mutation_is_detected() -> None:
    case = {
        "case_id": "MUTATION",
        "section_ids": ["section:1:src0:s1"],
        "assertions": [{
            "op": "text_contains",
            "evidence_id": "evidence:1:src0:s1:e0",
            "value": "기대 문자열",
        }],
    }
    failures = evaluate_case(case, {"section:1:src0:s1": _fragment("변경됨")})
    assert [failure.code for failure in failures] == ["SEMANTIC_ASSERTION_FAILED"]


def test_negative_path_assertion_guards_against_stale_heading_context() -> None:
    fragment = _fragment()
    fragment["evidence_list"][0]["payload"]["heading_path"] = ["다. 담보제공 내역"]
    case = {
        "case_id": "STALE_HEADING",
        "section_ids": ["section:1:src0:s1"],
        "assertions": [
            {
                "op": "evidence_path_not_contains",
                "evidence_id": "evidence:1:src0:s1:e0",
                "path": ["payload", "heading_path"],
                "value": "(2) 해외법인",
            }
        ],
    }

    assert evaluate_case(case, {"section:1:src0:s1": fragment}) == []


def test_negative_path_assertion_accepts_an_absent_optional_context_path() -> None:
    case = {
        "case_id": "NO_HEADING",
        "section_ids": ["section:1:src0:s1"],
        "assertions": [
            {
                "op": "evidence_path_not_contains",
                "evidence_id": "evidence:1:src0:s1:e0",
                "path": ["payload", "heading_path"],
                "value": "잘못된 제목",
            }
        ],
    }

    assert evaluate_case(case, {"section:1:src0:s1": _fragment()}) == []


def test_duplicate_normalizer_ignores_source_copy_whitespace() -> None:
    left = _fragment("보고기 간 말 (단위 : 백만원) .")
    right = _fragment("보고기간 말 (단위 : 백만원).")
    assert normalize_duplicate_value(left) == normalize_duplicate_value(right)


def test_candidate_generation_never_overwrites(tmp_path: Path) -> None:
    case = {"case_id": "FIXED", "section_ids": ["section:1:src0:s1"]}
    fragments = {"section:1:src0:s1": _fragment()}
    candidate_dir = tmp_path / "candidate"
    paths = _write_candidate(candidate_dir, [case], fragments)
    assert len(paths) == 1
    try:
        _write_candidate(candidate_dir, [case], fragments)
    except FileExistsError:
        pass
    else:
        raise AssertionError("candidate snapshot was overwritten")


def test_snapshot_mutation_is_detected(tmp_path: Path) -> None:
    case = {"case_id": "FIXED", "section_ids": ["section:1:src0:s1"]}
    fragments = {"section:1:src0:s1": _fragment()}
    golden = tmp_path / "golden"
    candidate = tmp_path / "candidate"
    _write_candidate(candidate, [case], fragments)
    golden.mkdir()
    value = json.loads((candidate / "FIXED.json").read_text(encoding="utf-8"))
    value["sections"][0]["summary"]["evidence_count"] = 99
    (golden / "FIXED.json").write_text(json.dumps(value), encoding="utf-8")
    failures = _compare_golden(golden, [case], fragments)
    assert [failure["code"] for failure in failures] == ["SNAPSHOT_MISMATCH"]


def test_quick_profile_is_isolated() -> None:
    result = run_regression(profile="quick", semantic_only=True)
    assert result["status"] == "PASSED"
    assert result["structural_summary"]["errors_count"] == 0
    assert not any(
        item["code"] == "PRODUCTION_WRITE_DETECTED"
        for item in result["failures"]
    )
