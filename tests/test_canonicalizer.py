from pathlib import Path

from converters.canonicalizer import canonicalize_document, canonicalize_table


SAMPLE = """<TABLE ACLASS="EXTRACTION"><TBODY>
<TR><TD COLSPAN="4">요약정보</TD></TR>
<TR><TD>발행회사명</TD><TE ACODE="CRP_NM">삼성전자주식회사</TE><TD>발행회사와의 관계</TD><TU AUNIT="FLT_CRP_RLT" AUNITVALUE="14">계열회사등</TU></TR>
<TR><TD>보고구분</TD><TU COLSPAN="3" AUNIT="RPT_DST1" AUNITVALUE="4">변동ㆍ변경</TU></TR>
<TR><TD ROWSPAN="3">보유주식등의 수 및 보유비율</TD><TD></TD><TD>보유주식등의 수</TD><TD>보유비율</TD></TR>
<TR><TD>직전 보고서</TD><TE ACODE="SUM_BMT_CNT">1,238,767,819</TE><TE ACODE="SUM_BMT_RT">20.75</TE></TR>
<TR><TD>이번 보고서</TD><TE ACODE="SUM_TMT_CNT">1,237,964,708</TE><TE ACODE="SUM_TMT_RT">20.74</TE></TR>
<TR><TD>보고사유</TD><TE COLSPAN="3" ACODE="SUM_CHN_RWN">- 보유주식수 변동\n- 보유계약 변경</TE></TR>
</TBODY></TABLE>"""


def by_code(result, code):
    return [field for field in result["fields"] if field["code"] == code]


def test_holding_summary_context_and_normalization():
    result = canonicalize_table(SAMPLE)
    assert result["table_class"] == "EXTRACTION"
    assert result["title"] == "요약정보"
    assert result["dimensions"] == {"rows": 7, "columns": 4}
    assert by_code(result, "CRP_NM")[0]["label"] == "발행회사명"
    relation = by_code(result, "FLT_CRP_RLT")[0]
    assert (relation["label"], relation["raw_code_value"], relation["value"]) == ("발행회사와의 관계", "14", "계열회사등")
    report_type = by_code(result, "RPT_DST1")[0]
    assert report_type["raw_code_value"] == "4"
    assert report_type["value"] == "CHANGE_AND_MODIFICATION"
    previous_count = by_code(result, "SUM_BMT_CNT")[0]
    assert previous_count["value"] == 1238767819
    assert previous_count["group_label"] == "보유주식등의 수 및 보유비율"
    assert previous_count["row_label"] == "직전 보고서"
    assert previous_count["column_label"] == "보유주식등의 수"
    assert by_code(result, "SUM_BMT_RT")[0]["value"] == 20.75
    assert by_code(result, "SUM_TMT_CNT")[0]["row_label"] == "이번 보고서"
    assert by_code(result, "SUM_TMT_RT")[0]["column_label"] == "보유비율"
    assert by_code(result, "SUM_CHN_RWN")[0]["raw_value"] == "- 보유주식수 변동\n- 보유계약 변경"


def test_unknown_empty_dash_and_multiple_pairs_are_preserved():
    xml = """<TABLE ACLASS="X"><TR><TD>빈 제목</TD><TE ACODE="EMPTY"></TE><TD>미상</TD><TU AUNIT="UNKNOWN" AUNITVALUE="Y">-</TU></TR></TABLE>"""
    fields = canonicalize_table(xml)["fields"]
    assert fields[0]["raw_value"] == ""
    assert fields[0]["known_field"] is False
    assert fields[1]["label"] == "미상"
    assert fields[1]["raw_code_value"] == "Y"
    assert fields[1]["value"] == "-"


def test_date_failure_is_local_and_custom_registry_is_extensible():
    registry = {"DATE": {"semantic_name": "event_date", "data_type": "date", "unit": None}}
    valid = canonicalize_table('<TABLE><TR><TD>날짜</TD><TU AUNIT="DATE" AUNITVALUE="20230116">표시일</TU></TR></TABLE>', field_definitions=registry)["fields"][0]
    assert valid["value"] == "2023-01-16"
    invalid = canonicalize_table('<TABLE><TR><TD>날짜</TD><TU AUNIT="DATE" AUNITVALUE="bad">잘못된 날짜</TU></TR></TABLE>', field_definitions=registry)["fields"][0]
    assert invalid["value"] == "잘못된 날짜"
    assert invalid["normalization_status"] == "failed"
    assert "normalization_error" in invalid


def test_input_not_mutated_order_and_duplicate_codes():
    raw = '<TABLE><TR><TD>A</TD><TE ACODE="X">1</TE></TR><TR><TD>B</TD><TE ACODE="X">2</TE></TR></TABLE>'
    original = raw[:]
    result = canonicalize_table(raw)
    assert raw == original
    assert [(f["label"], f["raw_value"]) for f in result["fields"]] == [("A", "1"), ("B", "2")]


def test_document_keeps_table_order_and_group_context():
    xml = '<DOCUMENT><TABLE-GROUP ACLASS="G1"><TABLE ACLASS="A"><TR><TD>A</TD><TE ACODE="X">1</TE></TR></TABLE></TABLE-GROUP><TABLE ACLASS="B"><TR><TD>B</TD><TE ACODE="X">2</TE></TR></TABLE></DOCUMENT>'
    result = canonicalize_document(xml)
    assert [table["table_class"] for table in result["tables"]] == ["A", "B"]
    assert result["tables"][0]["table_group_class"] == "G1"
    assert [table["source_index"] for table in result["tables"]] == [0, 1]


def test_malformed_fragment_is_returned_instead_of_raising():
    result = canonicalize_table("<TABLE><TR>")
    assert result["parse_status"] == "failed"
    assert result["raw_fragment"] == "<TABLE><TR>"


def test_html_like_text_is_recovered_without_source_mutation():
    raw = '<TABLE><TR><TD>이름</TD><TE ACODE="X">C&T</TE></TR><TR><TD><표1> 참조</TD></TR></TABLE>'
    result = canonicalize_table(raw)
    assert result["parse_status"] == "recovered"
    assert result["fields"][0]["raw_value"] == "C&T"
    assert raw.endswith("</TABLE>")


def test_real_holding_document_smoke():
    path = Path("data/raw/holding/효성중공업/20251001000631/20251001000631.xml")
    if not path.exists():
        return
    result = canonicalize_document(path)
    assert result["parse_status"] == "success"
    assert len(result["tables"]) > 1
    codes = [field["code"] for table in result["tables"] for field in table["fields"]]
    assert "CRP_NM" in codes
    assert "SUM_BMT_CNT" in codes
