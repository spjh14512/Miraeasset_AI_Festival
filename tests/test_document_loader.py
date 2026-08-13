from __future__ import annotations

from converters.common.document_loader import DocumentLoadStatus, load_document
from converters.common.source_models import DocumentSyntax


def test_load_valid_xml_without_recovery():
    loaded = load_document("<DOCUMENT><P>본문</P></DOCUMENT>")

    assert loaded.status == DocumentLoadStatus.SUCCESS
    assert loaded.syntax == DocumentSyntax.DART_XML
    assert loaded.root is not None
    assert loaded.root.tag == "DOCUMENT"
    assert loaded.issues == ()


def test_load_xml_with_bare_ampersand_records_recovery():
    loaded = load_document("<DOCUMENT><P>A & B</P></DOCUMENT>")

    assert loaded.status == DocumentLoadStatus.RECOVERED
    assert loaded.root is not None
    assert "".join(loaded.root.itertext()) == "A & B"
    assert [issue.code for issue in loaded.issues] == ["XML_RECOVERED"]


def test_unrecoverable_xml_returns_failed_result():
    loaded = load_document("<NOT_DOCUMENT><P>본문</NOT_DOCUMENT_EXTRA>")

    assert loaded.status == DocumentLoadStatus.FAILED
    assert loaded.root is None
    assert loaded.issues[0].code == "XML_LOAD_FAILED"


def test_malformed_attributes_and_html_entities_use_tolerant_xml_recovery():
    loaded = load_document(
        '<DOCUMENT><SECTION-1><TITLE>본문</TITLE><P>Intel &reg;</P>'
        '<TABLE><TR><TH ENG=""Other receivables">값</TH></TR></TABLE>'
        '</SECTION-1></DOCUMENT>'
    )

    assert loaded.status == DocumentLoadStatus.RECOVERED
    assert loaded.root is not None
    assert loaded.root.tag == "DOCUMENT"
    assert loaded.issues[0].code == "XML_TOLERANT_RECOVERY"
    assert "Intel ®" in "".join(loaded.root.itertext())


def test_loose_exchange_html_is_loaded_with_omitted_cell_end_tags():
    loaded = load_document(
        "<html><body><table><tr><td>A<td>B</table></body></html>",
        syntax=DocumentSyntax.HTML,
    )

    assert loaded.status == DocumentLoadStatus.SUCCESS
    assert loaded.root is not None
    cells = [node for node in loaded.root.iter() if node.tag == "TD"]
    assert ["".join(cell.itertext()) for cell in cells] == ["A", "B"]
