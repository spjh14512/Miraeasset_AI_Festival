from __future__ import annotations

from neo4j import Record
from neo4j.graph import Graph, Node, Path

from agent_graph.retrieval_result_parser import (
    parse_aggregate,
    parse_neo4j_response,
    parse_node,
    parse_path,
    parse_record,
    parse_relationship,
)


def _graph_values():
    graph = Graph()
    company = Node(
        graph,
        "company-element",
        1,
        ["Company"],
        {"corp_code": "00126380", "corp_name": "삼성전자"},
    )
    disclosure = Node(
        graph,
        "disclosure-element",
        2,
        ["Disclosure"],
        {"disclosure_id": "d1"},
    )
    relationship_class = graph.relationship_type("PUBLISHES")
    relationship = relationship_class(
        graph,
        "relationship-element",
        3,
        {"published_at": "2025-03-31"},
    )
    relationship._start_node = company
    relationship._end_node = disclosure
    return company, disclosure, relationship


def test_parse_node_preserves_labels_and_properties():
    company, _, _ = _graph_values()

    assert parse_node(company) == {
        "type": "node",
        "labels": ["Company"],
        "properties": {
            "corp_code": "00126380",
            "corp_name": "삼성전자",
        },
    }


def test_parse_relationship_preserves_type_endpoints_and_properties():
    company, disclosure, relationship = _graph_values()

    parsed = parse_relationship(relationship)

    assert parsed["relationship_type"] == "PUBLISHES"
    assert parsed["start_node"] == parse_node(company)
    assert parsed["end_node"] == parse_node(disclosure)
    assert parsed["properties"] == {"published_at": "2025-03-31"}


def test_parse_path_preserves_graph_order():
    company, _, relationship = _graph_values()
    path = Path(company, relationship)

    parsed = parse_path(path)

    assert parsed["type"] == "path"
    assert [node["labels"] for node in parsed["nodes"]] == [
        ["Company"],
        ["Disclosure"],
    ]
    assert [item["relationship_type"] for item in parsed["relationships"]] == [
        "PUBLISHES"
    ]


def test_parse_aggregate_recursively_parses_graph_values():
    company, _, _ = _graph_values()

    parsed = parse_aggregate([company, {"count": 3}])

    assert parsed == [parse_node(company), {"count": 3}]


def test_parse_record_preserves_aliases():
    company, _, _ = _graph_values()
    record = Record([("company", company), ("disclosure_count", 3)])

    parsed = parse_record(record)

    assert parsed == {
        "type": "record",
        "fields": {
            "company": parse_node(company),
            "disclosure_count": 3,
        },
    }


def test_parse_neo4j_response_returns_one_result_per_plan():
    company, _, _ = _graph_values()
    records = [
        Record([("company", company), ("disclosure_count", 3)]),
        Record([("company_name", "삼성전자"), ("disclosure_count", 5)]),
    ]

    result = parse_neo4j_response(
        records,
        plan_id="plan_1",
        query="MATCH (c:Company) RETURN c",
        parameters={"corp_name": "삼성전자"},
    )

    assert result.result_id == "retrieval:plan_1"
    assert result.plan_id == "plan_1"
    assert result.source == "neo4j"
    assert result.result_count == 2
    assert len(result.items) == 2
    assert result.metadata == {"parameters": {"corp_name": "삼성전자"}}
    result.model_dump_json()
