from pathlib import Path
from typing import Any

import pytest

from knowledge_graph import metric_definitions as loader
from knowledge_graph.insertDSE import load_schema


def test_metric_definition_rows_validate_seed_list():
    rows = loader.metric_definition_rows()

    ids = [row["id"] for row in rows]
    assert ids == [
        "revenue",
        "operating_income",
        "net_income",
        "total_assets",
        "total_equity",
        "roa",
    ]
    roa = next(row for row in rows if row["id"] == "roa")
    assert roa["higher_is_better"] is True
    assert roa["required_metrics"] == ["net_income", "total_assets"]
    assert roa["unit"] == "%"

    revenue = next(row for row in rows if row["id"] == "revenue")
    assert revenue["formula"] is None
    assert revenue["required_metrics"] is None


def test_metric_definition_rows_reject_duplicate_id():
    with pytest.raises(ValueError, match="Duplicate"):
        loader.metric_definition_rows(
            [
                {"id": "revenue", "name": "매출액"},
                {"id": "revenue", "name": "매출액(중복)"},
            ]
        )


def test_metric_definition_rows_reject_blank_name():
    with pytest.raises(ValueError, match="name"):
        loader.metric_definition_rows([{"id": "revenue", "name": "  "}])


def test_metric_definition_rows_reject_non_boolean_higher_is_better():
    with pytest.raises(ValueError, match="higher_is_better"):
        loader.metric_definition_rows(
            [{"id": "revenue", "name": "매출액", "higher_is_better": "yes"}]
        )


def test_metric_definition_rows_reject_invalid_required_metrics():
    with pytest.raises(ValueError, match="required_metrics"):
        loader.metric_definition_rows(
            [{"id": "roa", "name": "ROA", "required_metrics": ["net_income", ""]}]
        )


def test_schema_declares_metric_definition_and_relations():
    schema = load_schema(Path("knowledge_graph/neo4j_schema.yaml"))

    assert set(schema["entities"]["MetricDefinition"]["properties"]) == {
        "id",
        "name",
        "formula",
        "unit",
        "higher_is_better",
        "formula_version",
        "required_metrics",
    }
    assert schema["relations"]["INSTANCE_OF"]["endpoints"] == {
        "source": ["MetricObservation"],
        "target": ["MetricDefinition"],
    }


class _Result:
    def consume(self) -> None:
        return None


class _Session:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def run(self, query: str, **parameters: Any) -> _Result:
        self.calls.append((query, parameters))
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


def test_insert_metric_definitions_merges_by_natural_key():
    session = _Session()
    driver = _Driver(session)

    loader.insert_metric_definitions(driver, loader.metric_definition_rows(), database=None)

    assert driver.verified is True
    queries = "\n".join(query for query, _ in session.calls)
    assert "MERGE (definition:MetricDefinition {id: row.id})" in queries
    assert "REQUIRE definition.id IS UNIQUE" in queries


def test_insert_metric_definitions_skips_when_no_rows():
    session = _Session()
    driver = _Driver(session)

    loader.insert_metric_definitions(driver, [], database=None)

    assert driver.verified is False
    assert session.calls == []
