"""Document-family parsers that build concise, retrieval-oriented Event content."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Sequence
import re


_NULL_VALUES = {"", "-", "--"}
_ADMINISTRATIVE_KEYS = (
    "사외이사참석여부",
    "감사참석여부",
    "감사(사외이사가아닌감사위원)참석여부",
    "작성책임자",
)
_CORRECTION_MARKERS = (
    "정정관련공시서류",
    "정정사유",
    "정정사항",
)
_DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*(?:년|[-./])\s*(?P<month>\d{1,2})\s*"
    r"(?:월|[-./])\s*(?P<day>\d{1,2})\s*일?"
)
_COMPACT_DATE = re.compile(r"(?<!\d)(?P<date>20\d{6})(?!\d)")


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _compact(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣%]", "", _clean(value)).casefold()


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = _clean(value)
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _iso_date(value: object) -> str | None:
    text = _clean(value)
    match = _DATE_PATTERN.search(text)
    if match:
        parts = tuple(int(match.group(name)) for name in ("year", "month", "day"))
    else:
        compact_match = _COMPACT_DATE.search(text)
        if compact_match is None:
            return None
        raw = compact_match.group("date")
        parts = (int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    try:
        return date(*parts).isoformat()
    except ValueError:
        return None


def neo4j_evidence_id(source_evidence_id: str, rcept_no: str) -> str:
    prefix = f"evidence:{rcept_no}"
    if not source_evidence_id.startswith(prefix):
        raise ValueError(
            f"Evidence ID is outside disclosure {rcept_no}: {source_evidence_id}"
        )
    return f"d{rcept_no}{source_evidence_id[len(prefix):]}"


@dataclass(frozen=True, slots=True)
class FieldFact:
    key_path: tuple[str, ...]
    value: str
    evidence_id: str
    order: int

    @property
    def key(self) -> str:
        return " > ".join(self.key_path)


@dataclass(frozen=True, slots=True)
class TextFact:
    heading_path: tuple[str, ...]
    text: str
    evidence_id: str
    order: int


@dataclass(frozen=True, slots=True)
class EvidenceFacts:
    fields: tuple[FieldFact, ...]
    texts: tuple[TextFact, ...]
    candidate_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedEventContent:
    content: str
    event_date: str
    evidence_ids: tuple[str, ...]
    parser_name: str
    used_receipt_date: bool


def _section_order(fragment: Mapping[str, Any]) -> tuple[int, int]:
    section_id = str(fragment.get("section_id", ""))
    source = re.search(r":src(?P<source>\d+)", section_id)
    section = re.search(r":s(?P<section>\d+)$", section_id)
    return (
        int(source.group("source")) if source else 0,
        int(section.group("section")) if section else 0,
    )


def _selected_fragments(
    fragments: Sequence[Mapping[str, Any]],
    doc_group: str,
) -> tuple[Mapping[str, Any], ...]:
    ordered = sorted(fragments, key=_section_order)
    nonempty = tuple(
        fragment
        for fragment in ordered
        if fragment.get("evidence_list") or fragment.get("records")
    )
    if doc_group == "major":
        return nonempty[:1]
    return nonempty


def _record_fields(
    records: Sequence[Mapping[str, Any]],
    evidence_id: str,
    start_order: int,
) -> list[FieldFact]:
    result: list[FieldFact] = []
    for offset, record in enumerate(records):
        values = _unique(
            [str(item) for item in record.get("row_context", [])]
            + [str(item) for item in record.get("values", [])]
        )
        if len(values) < 2:
            continue
        result.append(
            FieldFact(
                key_path=tuple(values[:-1]),
                value=values[-1],
                evidence_id=evidence_id,
                order=start_order + offset,
            )
        )
    return result


def _header_field(
    payload: Mapping[str, Any],
    evidence_id: str,
    order: int,
) -> FieldFact | None:
    cells: list[str] = []
    for header in payload.get("headers", []):
        if isinstance(header, list):
            cells.append(" > ".join(_clean(item) for item in header if _clean(item)))
        else:
            cells.append(_clean(header))
    values = _unique(cells)
    if len(values) < 2:
        return None
    return FieldFact(tuple(values[:-1]), values[-1], evidence_id, order)


def _is_correction_table(
    payload: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> bool:
    text = _compact(
        " ".join(str(value) for value in payload.values())
        + " "
        + " ".join(str(record.get("values", [])) for record in records[:8])
    )
    return any(_compact(marker) in text for marker in _CORRECTION_MARKERS)


def collect_evidence_facts(
    fragments: Sequence[Mapping[str, Any]],
    *,
    doc_group: str,
) -> EvidenceFacts:
    fields: list[FieldFact] = []
    texts: list[TextFact] = []
    candidate_ids: list[str] = []
    order = 0

    for fragment in _selected_fragments(fragments, doc_group):
        records_by_table: dict[str, list[Mapping[str, Any]]] = {}
        for record in fragment.get("records", []):
            records_by_table.setdefault(str(record.get("table_id", "")), []).append(record)

        for evidence in fragment.get("evidence_list", []):
            evidence_id = str(evidence.get("evidence_id", "")).strip()
            if not evidence_id:
                continue
            payload = evidence.get("payload", {})
            if not isinstance(payload, Mapping):
                continue
            evidence_type = str(evidence.get("evidence_type", ""))
            if evidence_type == "TEXT":
                text = _clean(payload.get("text"))
                if text:
                    texts.append(
                        TextFact(
                            heading_path=tuple(
                                _clean(item) for item in payload.get("heading_path", [])
                                if _clean(item)
                            ),
                            text=text,
                            evidence_id=evidence_id,
                            order=order,
                        )
                    )
                    candidate_ids.append(evidence_id)
                    order += 1
                continue
            if evidence_type != "TABLE":
                continue

            table_id = str(payload.get("table_id", ""))
            table_records = records_by_table.get(table_id, [])
            if _is_correction_table(payload, table_records):
                continue

            before = len(fields)
            for field in payload.get("fields", []):
                if not isinstance(field, Mapping):
                    continue
                value = _clean(field.get("raw_value"))
                key_paths = field.get("key_paths", [])
                if not key_paths:
                    continue
                first_path = key_paths[0]
                if not isinstance(first_path, list):
                    continue
                key_path = tuple(_clean(item) for item in first_path if _clean(item))
                if key_path and value:
                    fields.append(FieldFact(key_path, value, evidence_id, order))
                    order += 1

            if doc_group == "exchange":
                header = _header_field(payload, evidence_id, order)
                if header is not None:
                    fields.append(header)
                    order += 1
            record_facts = _record_fields(table_records, evidence_id, order)
            fields.extend(record_facts)
            order += len(record_facts)
            if len(fields) > before:
                candidate_ids.append(evidence_id)

    return EvidenceFacts(
        fields=tuple(fields),
        texts=tuple(texts),
        candidate_evidence_ids=tuple(dict.fromkeys(candidate_ids)),
    )


_DATE_KEYS_BY_SUBTYPE: dict[str, tuple[str, ...]] = {
    "단일판매ㆍ공급계약체결": ("계약수주일자", "계약수주일"),
    "단일판매ㆍ공급계약해지": ("해지일자",),
    "신규시설투자등": ("이사회결의일결정일",),
    "투자판단관련주요경영사항": (
        "이사회결의일결정일또는사실확인일",
        "사실발생확인일",
        "결정일",
    ),
    "자기주식처분결정": ("처분결정일",),
    "자기주식취득결정": ("취득결정일",),
    "자기주식취득신탁계약체결결정": ("이사회결의일결정일", "계약체결예정일자"),
    "자기주식취득신탁계약해지결정": ("이사회결의일결정일", "해지예정일자"),
    "유상증자결정": ("이사회결의일결정일",),
    "무상증자결정": ("이사회결의일결정일",),
    "감자결정": ("이사회결의일결정일",),
    "상각형조건부자본증권발행결정": ("이사회결의일결정일",),
    "전환사채권발행결정": ("이사회결의일결정일",),
    "교환사채권발행결정": ("이사회결의일결정일",),
    "자본으로인정되는채무증권발행결정": ("이사회결의일결정일",),
    "자기전환사채매도결정": ("매도결정일",),
    "제3자의전환사채매수선택권행사": ("전환사채매수선택권행사일",),
    "타법인주식및출자증권양수결정": ("이사회결의일결정일", "양수예정일자"),
    "타법인주식및출자증권양도결정": ("이사회결의일결정일", "양도예정일자"),
    "유형자산양수결정": ("이사회결의일결정일", "계약체결일"),
    "유형자산양도결정": ("이사회결의일결정일", "계약체결일"),
    "회사합병결정": ("이사회결의일결정일", "합병계약일"),
    "회사분할결정": ("이사회결의일",),
    "회사분할합병결정": ("이사회결의일결정일", "분할합병계약일"),
    "주식교환·이전결정": ("이사회결의일결정일", "교환이전계약일"),
    "주식교환ㆍ이전결정": ("이사회결의일결정일", "교환이전계약일"),
    "영업양수결정": ("이사회결의일결정일", "계약체결일"),
    "소송등의제기": ("확인일자", "제기일자"),
    "영업정지": ("이사회결의일결정일", "영업정지일자"),
    "해외증권시장주권등상장폐지결정": ("이사회결의일확인일", "폐지예정일자"),
    "해외증권시장주권등상장폐지": ("확인일자", "매매거래종료일"),
    "해외증권시장주권등상장결정": ("이사회결의일결정일", "상장예정일자"),
    "해외증권시장주권등상장": ("확인일자", "상장일자"),
}

_GENERIC_DATE_KEYS = (
    "이사회결의일결정일",
    "이사회결의일",
    "사실발생확인일",
    "사실확인일",
    "확인일자",
    "결정일",
)


class BaseContentParser:
    name = "generic"
    priority_keywords: tuple[str, ...] = ()

    def date_keys(self, event_subtype: str) -> tuple[str, ...]:
        return _DATE_KEYS_BY_SUBTYPE.get(event_subtype, _GENERIC_DATE_KEYS)

    def _priority(self, fact: FieldFact) -> tuple[int, int]:
        key = _compact(fact.key)
        for index, keyword in enumerate(self.priority_keywords):
            if _compact(keyword) in key:
                return (index, fact.order)
        return (len(self.priority_keywords), fact.order)

    def _use_field(self, fact: FieldFact) -> bool:
        key = _compact(fact.key)
        return (
            _clean(fact.value) not in _NULL_VALUES
            and not any(_compact(marker) in key for marker in _ADMINISTRATIVE_KEYS)
            and not any(_compact(marker) in key for marker in _CORRECTION_MARKERS)
        )

    def _event_date(
        self,
        facts: EvidenceFacts,
        event_subtype: str,
        receipt_date: str,
    ) -> tuple[str, bool]:
        for candidate in self.date_keys(event_subtype):
            compact_candidate = _compact(candidate)
            for fact in facts.fields:
                if compact_candidate in _compact(fact.key):
                    parsed = _iso_date(fact.value)
                    if parsed is not None:
                        return parsed, False
        parsed_receipt = _iso_date(receipt_date)
        if parsed_receipt is None:
            raise ValueError(f"Invalid receipt date: {receipt_date}")
        return parsed_receipt, True

    def parse(
        self,
        facts: EvidenceFacts,
        *,
        event_type: str,
        event_subtype: str,
        receipt_date: str,
        company_name: str,
    ) -> ParsedEventContent:
        fields = sorted(
            (fact for fact in facts.fields if self._use_field(fact)),
            key=self._priority,
        )
        seen: set[tuple[str, str]] = set()
        field_lines: list[str] = []
        used_ids: list[str] = []
        for fact in fields:
            identity = (_compact(fact.key), _clean(fact.value))
            if identity in seen:
                continue
            seen.add(identity)
            field_lines.append(f"- {fact.key}: {fact.value}")
            used_ids.append(fact.evidence_id)

        text_lines: list[str] = []
        seen_text: set[str] = set()
        for fact in sorted(facts.texts, key=lambda item: item.order):
            text = _clean(fact.text)
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            heading = " > ".join(fact.heading_path)
            text_lines.append(f"- {heading}: {text}" if heading else f"- {text}")
            used_ids.append(fact.evidence_id)

        sections = [
            f"{company_name}의 {event_subtype} 공시 사건",
            f"이벤트 분류: {event_type}",
        ]
        if field_lines:
            sections.append("핵심 사실\n" + "\n".join(field_lines))
        if text_lines:
            sections.append("보충 설명\n" + "\n".join(text_lines))
        if not field_lines and not text_lines:
            sections.append("공시 본문에서 구조화된 핵심 사실을 추출하지 못했습니다.")

        event_date, used_receipt_date = self._event_date(
            facts,
            event_subtype,
            receipt_date,
        )
        evidence_ids = tuple(dict.fromkeys(used_ids or facts.candidate_evidence_ids))
        return ParsedEventContent(
            content="\n\n".join(sections),
            event_date=event_date,
            evidence_ids=evidence_ids,
            parser_name=self.name,
            used_receipt_date=used_receipt_date,
        )


class ContractContentParser(BaseContentParser):
    name = "contract"
    priority_keywords = (
        "체결계약명",
        "판매ㆍ공급계약구분",
        "계약금액",
        "계약상대",
        "판매ㆍ공급지역",
        "계약기간",
        "계약수주일",
        "주요계약조건",
        "공시유보",
        "기타투자판단",
    )


class ContractTerminationContentParser(BaseContentParser):
    name = "contract_termination"
    priority_keywords = (
        "해지구분",
        "해지금액",
        "계약상대",
        "해지주요사유",
        "해지일자",
        "계약기간",
        "기타투자판단",
    )


class ManagementJudgmentContentParser(BaseContentParser):
    name = "management_judgment"
    priority_keywords = (
        "제목",
        "주요내용",
        "임상시험",
        "시험결과",
        "사실발생확인일",
        "이사회결의일결정일또는사실확인일",
        "향후계획",
        "투자유의사항",
        "기타투자판단",
    )


class FacilityInvestmentContentParser(BaseContentParser):
    name = "facility_investment"
    priority_keywords = (
        "투자구분",
        "투자대상",
        "투자금액",
        "자기자본대비",
        "투자목적",
        "투자기간",
        "이사회결의일",
        "기타투자판단",
    )


class CapitalSecuritiesContentParser(BaseContentParser):
    name = "capital_securities"
    priority_keywords = (
        "발행",
        "증자",
        "사채",
        "주식수",
        "발행가액",
        "자금조달목적",
        "자금의사용목적",
        "배정",
        "전환",
        "교환",
        "청약",
        "납입일",
        "이사회결의일",
    )


class TreasuryStockContentParser(BaseContentParser):
    name = "treasury_stock"
    priority_keywords = (
        "취득예정주식",
        "처분예정주식",
        "취득예정금액",
        "처분예정금액",
        "계약금액",
        "취득목적",
        "처분목적",
        "취득방법",
        "처분방법",
        "계약기간",
        "결정일",
    )


class AssetTransactionContentParser(BaseContentParser):
    name = "asset_transaction"
    priority_keywords = (
        "양수",
        "양도",
        "투자대상",
        "투자금액",
        "거래상대방",
        "목적",
        "예정일자",
        "이사회결의일",
        "기타투자판단",
    )


class RestructuringContentParser(BaseContentParser):
    name = "corporate_restructuring"
    priority_keywords = (
        "합병방법",
        "분할방법",
        "교환이전",
        "상대방",
        "합병비율",
        "분할비율",
        "목적",
        "계약일",
        "효력발생일",
        "일정",
        "이사회결의일",
        "기타투자판단",
    )


class LegalRiskContentParser(BaseContentParser):
    name = "legal_risk"
    priority_keywords = (
        "사건명",
        "원고",
        "피고",
        "청구내용",
        "청구금액",
        "영업정지",
        "영향",
        "대응",
        "제기일자",
        "확인일자",
        "이사회결의일",
    )


class MarketDisclosureContentParser(BaseContentParser):
    name = "market_disclosure"
    priority_keywords = (
        "증권시장",
        "상장",
        "상장폐지",
        "종목명",
        "주식수",
        "사유",
        "예정일",
        "확인일",
        "향후일정",
    )


_GENERIC = BaseContentParser()
_BY_TYPE: dict[str, BaseContentParser] = {
    "계약·영업거래": ContractContentParser(),
    "자본·증권발행": CapitalSecuritiesContentParser(),
    "자기주식·주주환원": TreasuryStockContentParser(),
    "투자·자산거래": AssetTransactionContentParser(),
    "M&A·기업구조개편": RestructuringContentParser(),
    "법률·규제·사고·리스크": LegalRiskContentParser(),
    "시장·공시·해외": MarketDisclosureContentParser(),
}


def content_parser_for(event_type: str, event_subtype: str) -> BaseContentParser:
    if event_subtype == "단일판매ㆍ공급계약체결":
        return ContractContentParser()
    if event_subtype == "단일판매ㆍ공급계약해지":
        return ContractTerminationContentParser()
    if event_subtype == "신규시설투자등":
        return FacilityInvestmentContentParser()
    if event_subtype == "투자판단관련주요경영사항":
        return ManagementJudgmentContentParser()
    return _BY_TYPE.get(event_type, _GENERIC)


__all__ = [
    "EvidenceFacts",
    "FieldFact",
    "ParsedEventContent",
    "TextFact",
    "collect_evidence_facts",
    "content_parser_for",
    "neo4j_evidence_id",
]
