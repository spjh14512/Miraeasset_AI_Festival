from __future__ import annotations

import json

import pytest

from agent_graph.state import Citation
from agent_graph.utils import CITATION_CONTEXT_QUERY, format_citations


class _Neo4jSession:
    def __init__(self, records):
        self.records = records
        self.call = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, cypher, parameters):
        self.call = (cypher, parameters)
        return self.records


class _Neo4jDriver:
    def __init__(self, records):
        self.opened_session = _Neo4jSession(records)

    def session(self):
        return self.opened_session


def _write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )


def _paths(tmp_path):
    document_manifest_path = tmp_path / "data" / "manifest.jsonl"
    _write_jsonl(document_manifest_path, [{
        "rcept_no": "20240306000686",
        "corp_name": "LG유플러스",
        "report_nm": "사업보고서 (2023.12)",
        "rcept_dt": "20240306",
    }])
    driver = _Neo4jDriver([
        {
            "disclosure_id": "d20240306000686",
            "section_id": "d20240306000686:src0:s27",
            "requested_evidence_id": "d20240306000686:src0:s27:e8",
            "evidence_id": "d20240306000686:src0:s27:e8",
            "section_path": ["IV. 이사의 경영진단 및 분석의견"],
            "heading_path": [
                "3. 재무상태 및 영업실적(연결기준)",
                "가. 연결 재무상태",
            ],
        },
        {
            "disclosure_id": "d20240306000686",
            "section_id": "d20240306000686:src0:s27",
            "requested_evidence_id": "d20240306000686:src0:s27:e9",
            "evidence_id": "d20240306000686:src0:s27:e9",
            "section_path": ["IV. 이사의 경영진단 및 분석의견"],
            "heading_path": [
                "3. 재무상태 및 영업실적(연결기준)",
                "가. 연결 재무상태",
            ],
        },
    ])
    return document_manifest_path, driver


def test_formats_and_deduplicates_citations_by_document_and_section(tmp_path):
    document_manifest_path, driver = _paths(tmp_path)
    citations = [
        Citation(
            disclosure_id="d20240306000686",
            section_id="d20240306000686:src0:s27",
            evidence_id="d20240306000686:src0:s27:e8",
        ),
        Citation(
            disclosure_id="d20240306000686",
            section_id="d20240306000686:src0:s27",
            evidence_id="d20240306000686:src0:s27:e9",
        ),
    ]

    assert format_citations(
        citations,
        document_manifest_path=document_manifest_path,
        driver=driver,
    ) == [
        "LG유플러스가 2024년 3월 6일에 발행한 「사업보고서 (2023.12)」"
        "(접수번호 20240306000686)의 「IV. 이사의 경영진단 및 분석의견」 "
        "섹션 중 「3. 재무상태 및 영업실적(연결기준) > 가. 연결 재무상태」를 "
        "근거로 사용했습니다."
    ]
    assert driver.opened_session.call == (
        CITATION_CONTEXT_QUERY,
        {"citations": [
            {
                "disclosure_id": "d20240306000686",
                "section_id": "d20240306000686:src0:s27",
                "evidence_id": "d20240306000686:src0:s27:e8",
            },
            {
                "disclosure_id": "d20240306000686",
                "section_id": "d20240306000686:src0:s27",
                "evidence_id": "d20240306000686:src0:s27:e9",
            },
        ]},
    )


def test_formats_citation_as_user_visible_path(tmp_path):
    document_manifest_path, driver = _paths(tmp_path)

    assert format_citations(
        [Citation(
            disclosure_id="d20240306000686",
            section_id="d20240306000686:src0:s27",
            evidence_id="d20240306000686:src0:s27:e8",
        )],
        document_manifest_path=document_manifest_path,
        driver=driver,
        style="path",
    ) == [
        "[사업보고서 (2023.12)(20240306000686) > "
        "IV. 이사의 경영진단 및 분석의견 > "
        "3. 재무상태 및 영업실적(연결기준) > 가. 연결 재무상태]"
    ]


def test_formats_citation_as_answer_source_label(tmp_path):
    document_manifest_path, driver = _paths(tmp_path)

    assert format_citations(
        [Citation(
            disclosure_id="d20240306000686",
            section_id="d20240306000686:src0:s27",
            evidence_id="d20240306000686:src0:s27:e8",
        )],
        document_manifest_path=document_manifest_path,
        driver=driver,
        style="label",
    ) == ["[근거: 사업보고서 (2023.12), 2024-03-06]"]
    assert driver.opened_session.call is None


def test_answer_source_labels_are_deduplicated_by_disclosure(tmp_path):
    document_manifest_path, driver = _paths(tmp_path)

    assert format_citations(
        [
            Citation(
                disclosure_id="d20240306000686",
                section_id="d20240306000686:src0:s27",
                evidence_id="d20240306000686:src0:s27:e8",
            ),
            Citation(
                disclosure_id="d20240306000686",
                section_id="d20240306000686:src0:s27",
                evidence_id="d20240306000686:src0:s27:e9",
            ),
        ],
        document_manifest_path=document_manifest_path,
        driver=driver,
        style="label",
    ) == ["[근거: 사업보고서 (2023.12), 2024-03-06]"]


def test_formats_disclosure_level_citation_without_section(tmp_path):
    document_manifest_path, driver = _paths(tmp_path)

    assert format_citations(
        [Citation(disclosure_id="d20240306000686")],
        document_manifest_path=document_manifest_path,
        driver=driver,
    ) == [
        "LG유플러스가 2024년 3월 6일에 발행한 「사업보고서 (2023.12)」"
        "(접수번호 20240306000686)입니다."
    ]


def test_falls_back_to_section_when_heading_path_is_missing(tmp_path):
    document_manifest_path, _ = _paths(tmp_path)
    driver = _Neo4jDriver([{
        "disclosure_id": "d20240306000686",
        "section_id": "d20240306000686:src0:s27",
        "requested_evidence_id": "d20240306000686:src0:s27:e8",
        "evidence_id": "d20240306000686:src0:s27:e8",
        "section_path": ["IV. 이사의 경영진단 및 분석의견"],
        "heading_path": [],
    }])

    assert format_citations(
        [Citation(
            disclosure_id="d20240306000686",
            section_id="d20240306000686:src0:s27",
            evidence_id="d20240306000686:src0:s27:e8",
        )],
        document_manifest_path=document_manifest_path,
        driver=driver,
    ) == [
        "LG유플러스가 2024년 3월 6일에 발행한 「사업보고서 (2023.12)」"
        "(접수번호 20240306000686)의 「IV. 이사의 경영진단 및 분석의견」 "
        "섹션입니다."
    ]


def test_rejects_citation_missing_from_document_manifest(tmp_path):
    document_manifest_path, driver = _paths(tmp_path)

    with pytest.raises(ValueError, match="manifest에서 공시를 찾지 못했습니다"):
        format_citations(
            [Citation(disclosure_id="d20240307000001")],
            document_manifest_path=document_manifest_path,
            driver=driver,
        )
