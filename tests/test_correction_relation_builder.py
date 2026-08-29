import json
from pathlib import Path

import pytest

from converters.common.source_models import DocumentContext, DocumentSyntax, SourceRef
from converters.correction_extractor import correction_relation_builder as builder
from converters.correction_extractor.correction_models import (
    CorrectionExtraction,
    CorrectionMetadata,
    CorrectionStatus,
)


def _manifest_row(
    rcept_no: str,
    *,
    is_correction: bool,
    rcept_dt: str,
    report_nm: str = "사업보고서",
    corp_code: str = "00123456",
    doc_group: str = "periodic",
    flr_nm: str = "예시회사",
) -> dict[str, object]:
    return {
        "doc_id": f"{doc_group}_{rcept_no}",
        "rcept_no": rcept_no,
        "rcept_dt": rcept_dt,
        "report_nm": report_nm,
        "corp_code": corp_code,
        "doc_group": doc_group,
        "flr_nm": flr_nm,
        "is_correction": is_correction,
        "file_path": f"raw/{doc_group}/{rcept_no}",
    }


def _found(
    context: DocumentContext,
    *,
    target_document_name: str = "사업보고서",
) -> CorrectionExtraction:
    return CorrectionExtraction(
        source_document=context,
        syntax=DocumentSyntax.DART_XML,
        status=CorrectionStatus.FOUND,
        correction=CorrectionMetadata(
            title="정정신고(보고)",
            correction_date="2024-08-01",
            target_document_name=target_document_name,
            original_submission_date="2024-07-25",
            reason="기재 내용 정정",
            target_rcept_no=None,
            source_ref=SourceRef(
                syntax=DocumentSyntax.DART_XML,
                element_path="//CORRECTION[1]",
            ),
        ),
    )


def _write_source(data_root: Path, row: dict[str, object]) -> Path:
    directory = data_root / str(row["file_path"])
    directory.mkdir(parents=True)
    primary = directory / f"{row['rcept_no']}.xml"
    primary.write_text("primary", encoding="utf-8")
    (directory / f"{row['rcept_no']}_00760.xml").write_text(
        "attachment",
        encoding="utf-8",
    )
    return primary


def test_builder_uses_existing_resolver_and_compact_manifest_date(
    tmp_path: Path,
    monkeypatch,
):
    correction = _manifest_row(
        "20240801000001",
        is_correction=True,
        rcept_dt="20240801",
        report_nm="[기재정정]사업보고서",
    )
    original = _manifest_row(
        "20240725000001",
        is_correction=False,
        rcept_dt="20240725",
    )
    _write_source(tmp_path, correction)
    extracted_documents: list[str] = []

    def fake_extract(document: bytes, *, context: DocumentContext):
        extracted_documents.append(document.decode())
        return _found(context)

    monkeypatch.setattr(builder, "extract_correction", fake_extract)
    results = builder.build_correction_results(
        (correction, original),
        data_root=tmp_path,
    )

    assert len(results) == 1
    assert results[0].correction is not None
    assert results[0].correction.target_rcept_no == "20240725000001"
    assert extracted_documents == ["primary"]


def test_builder_does_not_add_document_name_guessing(tmp_path: Path, monkeypatch):
    correction = _manifest_row(
        "20240801000001",
        is_correction=True,
        rcept_dt="20240801",
        report_nm="[기재정정]제51기 사업보고서",
    )
    original = _manifest_row(
        "20240725000001",
        is_correction=False,
        rcept_dt="20240725",
        report_nm="사업보고서 (2023.12)",
    )
    _write_source(tmp_path, correction)
    monkeypatch.setattr(
        builder,
        "extract_correction",
        lambda document, *, context: _found(
            context,
            target_document_name="제51기 사업보고서",
        ),
    )

    result = builder.build_correction_results(
        (correction, original),
        data_root=tmp_path,
    )[0]

    assert result.correction is not None
    assert result.correction.target_rcept_no is None
    assert result.issues[-1].code == "CORRECTION_TARGET_NOT_FOUND"


def test_holding_resolution_uses_filer_identity(tmp_path: Path, monkeypatch):
    correction = _manifest_row(
        "20240801000001",
        is_correction=True,
        rcept_dt="20240801",
        report_nm="대량보유상황보고서(일반)",
        doc_group="holding",
        flr_nm="국민연금공단",
    )
    same_filer = _manifest_row(
        "20240725000001",
        is_correction=False,
        rcept_dt="20240725",
        report_nm="대량보유상황보고서(일반)",
        doc_group="holding",
        flr_nm="국민연금공단",
    )
    other_filer = _manifest_row(
        "20240725000002",
        is_correction=False,
        rcept_dt="20240725",
        report_nm="대량보유상황보고서(일반)",
        doc_group="holding",
        flr_nm="다른보고자",
    )
    _write_source(tmp_path, correction)
    monkeypatch.setattr(
        builder,
        "extract_correction",
        lambda document, *, context: _found(
            context,
            target_document_name="대량보유상황보고서(일반)",
        ),
    )

    result = builder.build_correction_results(
        (correction, same_filer, other_filer),
        data_root=tmp_path,
    )[0]

    assert result.correction is not None
    assert result.correction.target_rcept_no == "20240725000001"


def test_materializer_keeps_correction_v1_contract(tmp_path: Path, monkeypatch):
    correction = _manifest_row(
        "20240801000001",
        is_correction=True,
        rcept_dt="20240801",
    )
    original = _manifest_row(
        "20240725000001",
        is_correction=False,
        rcept_dt="20240725",
    )
    _write_source(tmp_path, correction)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (correction, original)
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "correction" / "manifest.jsonl"
    monkeypatch.setattr(
        builder,
        "extract_correction",
        lambda document, *, context: _found(context),
    )

    summary = builder.materialize_correction_results(
        manifest_path=manifest_path,
        data_root=tmp_path,
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary == {"total": 1, "resolved": 1, "unresolved": 0, "failed": 0}
    assert payload["schema_version"] == "correction.v1"
    assert payload["source_document"]["rcept_no"] == "20240801000001"
    assert payload["correction"]["target_rcept_no"] == "20240725000001"


def test_manifest_json_error_reports_line_number(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"doc_id":"valid"}\n{"doc_id": broken}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"manifest\.jsonl:2"):
        builder.read_manifest_rows(manifest_path)
