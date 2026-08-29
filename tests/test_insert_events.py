from datetime import date
import json
from pathlib import Path
from typing import Any

import pytest

from knowledge_graph import insert_events as loader
from knowledge_graph.insertDSE import load_schema


def _record() -> dict[str, Any]:
    return {
        "schema_version": "event.v1",
        "source_document": {
            "rcept_no": "20240102000001",
            "disclosure_id": "d20240102000001",
            "is_latest_version": True,
        },
        "event": {
            "id": "event:20240102000001",
            "event_type": "자본·증권발행",
            "event_subtype": "유상증자결정",
            "event_date": "2024-01-01",
            "content": "핵심 내용",
        },
        "relations": {
            "reports": {
                "type": "REPORTS",
                "source_id": "d20240102000001",
                "target_id": "event:20240102000001",
            },
            "is_supported_by": [
                {
                    "type": "IS_SUPPORTED_BY",
                    "source_id": "event:20240102000001",
                    "target_id": "d20240102000001:src0:s0:e0",
                }
            ],
        },
    }


def _write(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def test_event_rows_validate_and_flatten_graph_contract(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    _write(path, _record())

    assert loader.event_rows(path) == [
        {
            "id": "event:20240102000001",
            "event_type": "자본·증권발행",
            "event_subtype": "유상증자결정",
            "event_date": date(2024, 1, 1),
            "content": "핵심 내용",
            "disclosure_id": "d20240102000001",
            "evidence_ids": ["d20240102000001:src0:s0:e0"],
        }
    ]


def test_event_rows_reject_non_latest_source(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    record = _record()
    record["source_document"]["is_latest_version"] = False
    _write(path, record)

    with pytest.raises(ValueError, match="not the latest"):
        loader.event_rows(path)


def test_schema_declares_event_and_relationships():
    schema = load_schema(Path("knowledge_graph/neo4j_schema.yaml"))

    assert set(schema["entities"]["Event"]["properties"]) == {
        "id",
        "event_type",
        "event_subtype",
        "event_date",
        "content",
    }
    assert schema["relations"]["REPORTS"]["endpoints"] == {
        "source": ["Disclosure"],
        "target": ["Event"],
    }
    assert schema["relations"]["IS_SUPPORTED_BY"]["endpoints"] == {
        "source": ["Event"],
        "target": ["Evidence:Text", "Evidence:Table"],
    }


class _Result:
    def __init__(self, rows: list[str] | None = None):
        self.rows = rows or []

    def single(self, *, strict: bool) -> dict[str, Any]:
        assert strict is True
        return {"rows": self.rows}

    def consume(self) -> None:
        return None


class _Session:
    def __init__(
        self,
        *,
        invalid_disclosures: list[str] | None = None,
        missing_evidence: list[str] | None = None,
    ):
        self.invalid_disclosures = invalid_disclosures or []
        self.missing_evidence = missing_evidence or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def run(self, query: str, **parameters: Any) -> _Result:
        self.calls.append((query, parameters))
        if "disclosure.is_latest_version" in query:
            return _Result(self.invalid_disclosures)
        if "WHERE evidence IS NULL" in query:
            return _Result(self.missing_evidence)
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


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "event:20240102000001",
            "event_type": "자본·증권발행",
            "event_subtype": "유상증자결정",
            "event_date": date(2024, 1, 1),
            "content": "핵심 내용",
            "disclosure_id": "d20240102000001",
            "evidence_ids": ["d20240102000001:src0:s0:e0"],
        }
    ]


def test_insert_events_uses_required_relationship_directions():
    session = _Session()
    driver = _Driver(session)

    loader.insert_events(driver, _rows(), database=None, batch_size=10)

    assert driver.verified is True
    queries = "\n".join(query for query, _ in session.calls)
    assert "(disclosure)-[:REPORTS]->(event)" in queries
    assert "(event)-[:IS_SUPPORTED_BY]->(evidence)" in queries
    assert "REQUIRE event.id IS UNIQUE" in queries


def test_insert_events_stops_for_nonlatest_disclosure():
    session = _Session(invalid_disclosures=["d20240102000001"])

    with pytest.raises(RuntimeError, match="existing latest Disclosure"):
        loader.insert_events(_Driver(session), _rows(), database=None, batch_size=10)

    assert not any("MERGE (event:Event" in query for query, _ in session.calls)
