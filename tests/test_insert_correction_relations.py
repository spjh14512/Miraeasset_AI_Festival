import json
from pathlib import Path
from typing import Any

import pytest

import knowledge_graph.insertDSE as insert_dse
from knowledge_graph import insert_correction_relations as loader


def _correction_record(**overrides: Any) -> dict[str, Any]:
    correction = {
        "correction_date": "2024-02-02",
        "original_submission_date": "2024-01-01",
        "target_document_name": "주요사항보고서",
        "reason": "기재 내용 정정",
        "target_rcept_no": "20240101000001",
        "raw_text": "정정신고(보고) 원문 발췌",
    }
    correction.update(overrides.pop("correction", {}))
    record = {
        "schema_version": "correction.v1",
        "source_document": {"rcept_no": "20240202000002"},
        "status": "FOUND",
        "correction": correction,
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_relation_rows_use_existing_correction_v1_contract(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        path,
        [
            _correction_record(
                source_document={"rcept_no": "20240303000003"},
                correction={"target_rcept_no": None},
            ),
            _correction_record(),
        ],
    )

    assert loader.correction_relation_rows(path) == [
        {
            "source_id": "d20240202000002",
            "target_id": "d20240101000001",
            "correction_date": "2024-02-02",
            "original_submission_date": "2024-01-01",
            "target_document_name": "주요사항보고서",
            "reason": "기재 내용 정정",
            "def": "정정신고(보고) 원문 발췌",
        }
    ]


def test_relation_rows_build_chronological_chain_for_same_original(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        path,
        [
            _correction_record(
                source_document={"rcept_no": "20240303000003"},
                correction={"target_rcept_no": "20240101000001"},
            ),
            _correction_record(
                source_document={"rcept_no": "20240202000002"},
                correction={"target_rcept_no": "20240101000001"},
            ),
        ],
    )

    rows = loader.correction_relation_rows(path)

    assert [(row["source_id"], row["target_id"]) for row in rows] == [
        ("d20240202000002", "d20240101000001"),
        ("d20240303000003", "d20240202000002"),
    ]


def test_relation_rows_reject_conflicting_originals(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        path,
        [
            _correction_record(),
            _correction_record(correction={"target_rcept_no": "20240101000999"}),
        ],
    )

    with pytest.raises(ValueError, match="Conflicting original disclosures"):
        loader.correction_relation_rows(path)


def test_insert_dse_only_adds_correction_identity_properties(tmp_path: Path):
    schema = insert_dse.load_schema(Path("knowledge_graph/neo4j_schema.yaml"))
    output_path = "canonical_section/major/sections_20240202000002.json"
    canonical = tmp_path / output_path
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"sections": []}', encoding="utf-8")
    selected = [
        (
            {
                "rcept_no": "20240202000002",
                "doc_group": "major",
                "source_path": "raw/major/20240202000002",
                "output_path": output_path,
            },
            {"output_paths": []},
        )
    ]
    metadata = {
        "20240202000002": {
            "corp_code": "00123456",
            "rcept_dt": "20240202",
            "is_correction": True,
            "report_nm": "[기재정정]주요사항보고서",
        }
    }

    rows = insert_dse.build_rows(schema, tmp_path, selected, metadata)

    assert rows["disclosures"][0]["rcept_no"] == "20240202000002"
    assert rows["disclosures"][0]["rcept_dt"] == "20240202"
    assert rows["disclosures"][0]["is_correction"] is True
    assert rows["disclosures"][0]["is_latest_version"] is True
    assert "corrects" not in rows


def test_select_correction_pairs_returns_complete_unique_endpoints(tmp_path: Path):
    pairs = [
        ("periodic", "20240202000002", "20240101000001", "00123456"),
        ("major", "20240404000004", "20240303000003", "00987654"),
    ]
    section_rows = []
    evidence_rows = []
    metadata: dict[str, dict[str, Any]] = {}
    correction_records = []
    for group, source_rcept_no, target_rcept_no, corp_code in pairs:
        for rcept_no, is_correction in (
            (source_rcept_no, True),
            (target_rcept_no, False),
        ):
            section_output = f"canonical_section/{group}/{rcept_no}.json"
            evidence_output = f"evidence_fragment/{group}/{rcept_no}.json"
            section_path = tmp_path / section_output
            evidence_path = tmp_path / evidence_output
            section_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            section_path.write_text("{}", encoding="utf-8")
            evidence_path.write_text("{}", encoding="utf-8")
            section_rows.append(
                {
                    "doc_group": group,
                    "rcept_no": rcept_no,
                    "status": "SUCCESS",
                    "error": None,
                    "output_path": section_output,
                }
            )
            evidence_rows.append(
                {
                    "doc_group": group,
                    "rcept_no": rcept_no,
                    "status": "SUCCESS",
                    "error": None,
                    "output_paths": [evidence_output],
                }
            )
            metadata[rcept_no] = {
                "corp_code": corp_code,
                "is_correction": is_correction,
            }
        correction_records.append(
            _correction_record(
                source_document={
                    "rcept_no": source_rcept_no,
                    "doc_group": group,
                },
                correction={"target_rcept_no": target_rcept_no},
            )
        )

    later_source = "20240505000005"
    later_group = "periodic"
    later_section_output = f"canonical_section/{later_group}/{later_source}.json"
    later_evidence_output = f"evidence_fragment/{later_group}/{later_source}.json"
    later_section_path = tmp_path / later_section_output
    later_evidence_path = tmp_path / later_evidence_output
    later_section_path.parent.mkdir(parents=True, exist_ok=True)
    later_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    later_section_path.write_text("{}", encoding="utf-8")
    later_evidence_path.write_text("{}", encoding="utf-8")
    section_rows.append({
        "doc_group": later_group,
        "rcept_no": later_source,
        "status": "SUCCESS",
        "error": None,
        "output_path": later_section_output,
    })
    evidence_rows.append({
        "doc_group": later_group,
        "rcept_no": later_source,
        "status": "SUCCESS",
        "error": None,
        "output_paths": [later_evidence_output],
    })
    metadata[later_source] = {
        "corp_code": "00123456",
        "is_correction": True,
    }
    correction_records.append(
        _correction_record(
            source_document={
                "rcept_no": later_source,
                "doc_group": later_group,
            },
            correction={"target_rcept_no": "20240101000001"},
        )
    )

    (tmp_path / "canonical_section").mkdir(exist_ok=True)
    (tmp_path / "evidence_fragment").mkdir(exist_ok=True)
    (tmp_path / "correction").mkdir(exist_ok=True)
    _write_jsonl(tmp_path / "canonical_section" / "manifest.jsonl", section_rows)
    _write_jsonl(tmp_path / "evidence_fragment" / "manifest.jsonl", evidence_rows)
    _write_jsonl(
        tmp_path / "correction" / "manifest.jsonl",
        correction_records,
    )

    selected, records = insert_dse.select_correction_pairs(
        tmp_path,
        pair_count=2,
        random_seed=42,
        disclosure_metadata=metadata,
    )

    assert len(selected) == 5
    assert len(records) == 3
    assert len({row[0]["rcept_no"] for row in selected}) == 5
    selection_path = tmp_path / "correction" / "selected_pairs.jsonl"
    insert_dse.write_jsonl_rows(selection_path, records)
    relation_rows = loader.correction_relation_rows(selection_path)
    assert len(relation_rows) == 3
    assert (
        "d20240505000005",
        "d20240202000002",
    ) in {(row["source_id"], row["target_id"]) for row in relation_rows}


def test_select_all_disclosures_returns_every_complete_receipt_in_order(
    monkeypatch,
    tmp_path: Path,
):
    rows = {
        "20240303000003": ({"rcept_no": "20240303000003"}, {"id": "e3"}),
        "20240101000001": ({"rcept_no": "20240101000001"}, {"id": "e1"}),
        "20240202000002": ({"rcept_no": "20240202000002"}, {"id": "e2"}),
    }
    monkeypatch.setattr(
        insert_dse,
        "complete_disclosures_by_receipt",
        lambda _data_root: rows,
    )

    selected = insert_dse.select_all_disclosures(tmp_path)

    assert [section["rcept_no"] for section, _ in selected] == [
        "20240101000001",
        "20240202000002",
        "20240303000003",
    ]


class _Result:
    def __init__(self, rows: list[dict[str, Any]] | None = None):
        self.rows = rows or []

    def single(self, *, strict: bool) -> dict[str, Any]:
        assert strict is True
        return {"rows": self.rows}

    def consume(self) -> None:
        return None


class _Session:
    def __init__(self, *, missing: list[dict[str, Any]] | None = None):
        self.missing = missing or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def run(self, query: str, **parameters: Any) -> _Result:
        self.calls.append((query, parameters))
        if "source IS NULL" in query:
            return _Result(self.missing)
        return _Result()


class _Driver:
    def __init__(self, session: _Session):
        self.session_value = session
        self.verified = False

    def verify_connectivity(self) -> None:
        self.verified = True

    def session(self, *, database: str | None):
        assert database is None
        return self.session_value


def _relation_rows() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "d20240202000002",
            "target_id": "d20240101000001",
            "correction_date": "2024-02-02",
            "original_submission_date": "2024-01-01",
            "target_document_name": "주요사항보고서",
            "reason": "기재 내용 정정",
            "def": "정정신고(보고) 원문 발췌",
        }
    ]


def test_neo4j_loader_replaces_relationship_and_updates_latest_flags():
    session = _Session()
    driver = _Driver(session)

    loader.insert_correction_relations(
        driver,
        _relation_rows(),
        database=None,
        batch_size=10,
    )

    assert driver.verified is True
    merge_query = next(query for query, _ in session.calls if "MERGE (source)" in query)
    assert "MERGE (source)-[relation:CORRECTS]->(target)" in merge_query
    assert "relation.def = row.def" in merge_query
    assert any("DELETE existing" in query for query, _ in session.calls)
    assert any(
        "SET disclosure.is_latest_version = false" in query
        for query, _ in session.calls
    )
    latest_call = next(
        parameters
        for query, parameters in session.calls
        if "SET disclosure.is_latest_version = true" in query
    )
    assert latest_call["ids"] == ["d20240202000002"]


def test_neo4j_loader_stops_if_a_disclosure_node_is_missing():
    session = _Session(
        missing=[
            {
                "source_id": "d20240202000002",
                "target_id": "d20240101000001",
            }
        ]
    )

    with pytest.raises(RuntimeError, match="endpoints must already exist"):
        loader.insert_correction_relations(
            _Driver(session),
            _relation_rows(),
            database=None,
            batch_size=10,
        )

    assert not any(":CORRECTS" in query for query, _ in session.calls)
