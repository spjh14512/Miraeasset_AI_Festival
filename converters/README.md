# Converters

이 문서는 `converters` 아래에 있는 변환 도구들의 명세와 사용법을 모아 둔다.
새 도구를 추가할 때는 아래 도구 목록에 항목을 추가하고, 각 도구를 독립된 2단계
제목으로 문서화한다.

## 도구 목록

| 도구 | 역할 | 상태 |
|---|---|---|
| `correction_extractor` | 정정공시 영역과 관계 해석용 메타데이터 추출 | 구현됨 |
| `section_canonicalizer` | 문서를 계층형 section과 원문 순서 block으로 분할 | 구현됨 |
| `paragraph_parser` | 본문의 `P`와 독립 `SPAN` 묶음을 문단으로 변환 | 구현됨 |
| `table_parser` | DART 공시의 `TABLE` 구조화 | 구현됨 |
| `table_context_resolver` | 표와 인접한 제목·단위·캡션·주석 연결 | 구현됨 |

## `correction_extractor`

`converters/correction_extractor`는 현재 공시의 정정신고 영역을 탐지하고, 관계 해석에
필요한 메타데이터와 제외할 원본 범위를 반환한다. 원문에서 정정 영역을 삭제하거나
최초 공시의 접수번호를 임의로 추론하지 않는다.

### 파일 구성

```text
converters/
├── common/
│   └── source_models.py
└── correction_extractor/
    ├── correction_models.py
    └── correction_extractor.py
```

### 입력 계약

현재 문서의 식별자는 정정 본문에서 추론하지 않고 ingestion 단계에서
`DocumentContext`로 전달한다. `doc_id`와 `rcept_no`는 필수다.

```python
from pathlib import Path

from converters.common.source_models import DocumentContext
from converters.correction_extractor.correction_extractor import extract_correction

rcept_no = "20241118000171"
path = Path("data/raw/major/삼성전자/20241118000171/20241118000171.xml")

result = extract_correction(
    path.read_bytes(),
    context=DocumentContext(
        doc_id=f"major_{rcept_no}",
        rcept_no=rcept_no,
        source_path=str(path),
        doc_group="major",
    ),
)
```

`extract_correction()`의 입력은 `xml.etree.ElementTree.Element`, `str`, `bytes` 중
하나다. 문자열과 bytes는 기본적으로 XML/HTML 구문을 자동 판별한다.

### 탐지 경계

| 구문 | 정정 영역 | 대체 기준 |
|---|---|---|
| DART XML | `<CORRECTION>` 서브트리 | 없음 |
| 거래소 HTML | `div#LIB_LC000` | `XFormD8_*` 표 |

XML에서는 `//CORRECTION[1]`을 제외 경로로 반환한다. 거래소 HTML에서는
`LIB_LC000`을 반환하고, 해당 컨테이너가 없으면 발견된 `XFormD8_*` 표 ID를 각각
반환한다.

이 경계는 section chunker가 같은 원문에서 정정 영역을 건너뛰는 데 사용한다. 원문
문자열이나 `Element`는 변경되지 않는다.

### 출력 예시

```json
{
  "schema_version": "correction.v1",
  "source_document": {
    "doc_id": "major_20241118000171",
    "rcept_no": "20241118000171",
    "source_path": "data/raw/major/삼성전자/20241118000171/20241118000171.xml",
    "doc_group": "major"
  },
  "syntax": "DART_XML",
  "status": "FOUND",
  "has_correction": true,
  "correction": {
    "title": "정 정 신 고 (보고)",
    "correction_date": "2024-11-18",
    "target_document_name": "주요사항보고서(자기주식취득결정)",
    "original_submission_date": "2024-11-18",
    "reason": "기재 내용 정정",
    "target_rcept_no": null,
    "source_ref": {
      "syntax": "DART_XML",
      "element_path": "//CORRECTION[1]",
      "html_id": null
    }
  },
  "correction_blocks": [],
  "excluded_source_refs": [
    {
      "syntax": "DART_XML",
      "element_path": "//CORRECTION[1]",
      "html_id": null
    }
  ],
  "issues": []
}
```

`source_document.rcept_no`는 현재 정정공시의 접수번호다.
`correction.target_rcept_no`는 원 공시를 찾는 `CorrectionResolver`가 나중에
채울 값이므로 extractor 단계에서는 항상 `null`이다.

### 모델 명세

| 모델 | 역할 |
|---|---|
| `DocumentContext` | 현재 공시의 확정 식별자와 출처 |
| `SourceRef` | 원문을 변경하지 않고 요소를 다시 찾기 위한 참조 |
| `CorrectionMetadata` | 정정일자, 대상 서류, 최초 제출일, 사유 |
| `CorrectionBlockRef` | 정정 영역 안의 제목·문단·표 참조와 원문 순서 |
| `CorrectionExtraction` | 추출 상태, 메타데이터, 제외 범위, 이슈를 묶은 결과 |

`correction_blocks`는 정정 전·후 표의 내용을 별도 형식으로 복제하지 않는다. 이후
`table_parser`, `paragraph_parser`가 원본 참조를 이용해 canonical evidence로
변환할 수 있도록 블록 종류, 순서, 위치만 보존한다.

### 상태와 이슈

| 상태 | 의미 |
|---|---|
| `FOUND` | 정정 영역을 정상적으로 추출함 |
| `NOT_FOUND` | 정정 영역이 없는 문서 |
| `RECOVERED` | XML 텍스트를 보수적으로 복구한 뒤 추출함 |
| `FAILED` | 정정 영역을 구조화하지 못함 |

메타데이터 일부가 원문에 없거나 잘못 기재돼도 정정 경계를 찾았다면 결과는 유지하고
`MISSING_*` 이슈를 추가한다. 정정일자와 현재 `rcept_no`의 접수일이 달라도 값을
변경하지 않는다.

### 현재 코퍼스 검증

활성 `data/manifest.jsonl`을 기준으로 확인한 결과는 다음과 같다.

```text
정정공시 탐지: 1,002 / 1,002
비정정 오탐:   0 / 3,200
정상 추출:     922
XML 복구 추출: 80
```

원문 품질 문제로 대상 서류명이 비어 있는 문서 1건과 최초 제출일이
`2025년 08년 28일`로 잘못 적힌 문서 1건은 값을 추정하지 않고 `MISSING_*` 이슈로
남긴다.

## DART `section_canonicalizer`

`converters/section_canonicalizer`는 correction 영역을 제외한 문서를 계층형 section으로
나누고, 각 section에 직접 속한 paragraph, table, image의 원본 참조를 DOM 순서대로
배치한다. 블록 내용을 직접 변환하지 않으며 이후 paragraph/table parser가 사용할
경계와 소유권만 결정한다.

### 파일 구성

```text
converters/section_canonicalizer/
├── section_models.py    # section, block 참조, 상태 모델
└── section_canonicalizer.py   # 명시적 계층과 보수적 fallback 탐지
```

공개 함수는 `chunk_sections()`다.

```python
from converters.section_canonicalizer.section_canonicalizer import chunk_sections
```

### 경계와 계층 규칙

1. DART XML의 `SECTION-1`, `SECTION-2`, ... 태그를 가장 신뢰도 높은 경계로 사용한다.
2. section의 직접 자식 `TITLE`을 제목으로 사용하고 `section_path`에 상위 제목부터
   누적한다.
3. wrapper인 `LIBRARY` 아래의 section도 가장 가까운 상위 section의 자식으로 둔다.
4. 명시적 section이 없으면 `H1`~`H6`, 명시적인 heading style/class를 사용한다.
5. 번호형 `P` 또는 한 셀 표는 같은 부모에서 같은 번호 체계가 두 번 이상 반복될 때만
   암시적 경계로 인정한다.
6. 경계를 신뢰할 수 없는 문서는 하나의 `SYNTHETIC` root section으로 유지한다.

거래소 HTML처럼 굵은 문서 제목과 하나의 본문 표로 구성된 문서는 굵은 제목을 synthetic
section의 metadata로 사용하고, 제목 `SPAN`을 별도 block으로 중복 출력하지 않는다.

### 블록 소유와 순서

| `block_type` | 대상 |
|---|---|
| `P` | 비어 있지 않은 단일 `P` |
| `SPAN_RUN` | `P` 밖에서 직접 연속된 비어 있지 않은 `SPAN` |
| `TABLE` | `TABLE-GROUP`에 속하지 않는 본문 표 |
| `TABLE_GROUP` | 내부 보조 표를 포함한 전체 표 그룹 |
| `IMAGE` | `IMAGE` 또는 `IMG` |

각 원본 block은 가장 깊은 section 하나에만 속한다. 부모 section은 자식 section의
block을 다시 소유하지 않으며, `TABLE_GROUP`을 한 block으로 다루므로 내부 표도 별도
`TABLE` block으로 중복되지 않는다. section 내부의 `order`는 0부터 시작하는 DOM
순서다.

`TITLE`은 section metadata이므로 paragraph block으로 만들지 않는다. `PGBRK`,
`SCRIPT`, `STYLE`은 evidence block에서 제외한다.

### correction extractor와 함께 사용

```python
from pathlib import Path

from converters.common.source_models import DocumentContext
from converters.correction_extractor.correction_extractor import extract_correction
from converters.section_canonicalizer.section_canonicalizer import chunk_sections

path = Path("data/raw/major/회사/20241118000171/20241118000171.xml")
raw = path.read_bytes()
document_context = DocumentContext(
    doc_id="major_20241118000171",
    rcept_no="20241118000171",
    source_path=str(path),
    doc_group="major",
)

correction = extract_correction(raw, context=document_context)
sections = chunk_sections(
    raw,
    document_context=document_context,
    excluded_source_refs=correction.excluded_source_refs,
)
```

XML의 `CORRECTION` 서브트리는 항상 제외한다. 거래소 HTML처럼 태그명만으로 경계를
알 수 없는 경우에는 correction extractor의 `excluded_source_refs`에 포함된 HTML ID
또는 element path를 이용해 해당 컨테이너 전체를 건너뛴다. 원본 DOM은 변경하지 않는다.

### 출력 예시

```json
{
  "schema_version": "section-collection.v1",
  "source_document": {
    "doc_id": "periodic_20230515002335",
    "rcept_no": "20230515002335",
    "source_path": "sample.xml",
    "doc_group": "periodic"
  },
  "syntax": "DART_XML",
  "parse_status": "SUCCESS",
  "sections": [
    {
      "id": "s2",
      "parent_section_id": "s1",
      "order": 2,
      "level": 2,
      "boundary_kind": "EXPLICIT",
      "title": "1. 회사의 개요",
      "section_path": ["I. 회사의 개요", "1. 회사의 개요"],
      "source_ref": {
        "syntax": "DART_XML",
        "element_path": "/DOCUMENT[1]/BODY[1]/SECTION-1[1]/SECTION-2[1]",
        "html_id": null
      },
      "title_source_ref": {
        "syntax": "DART_XML",
        "element_path": "/DOCUMENT[1]/BODY[1]/SECTION-1[1]/SECTION-2[1]/TITLE[1]",
        "html_id": null
      },
      "blocks": [
        {
          "block_type": "P",
          "order": 0,
          "source_refs": [
            {
              "syntax": "DART_XML",
              "element_path": "/DOCUMENT[1]/BODY[1]/SECTION-1[1]/SECTION-2[1]/P[1]",
              "html_id": null
            }
          ]
        }
      ],
      "issues": []
    }
  ],
  "issues": []
}
```

`boundary_kind`는 원문의 `SECTION-*`를 사용한 `EXPLICIT`, 보수적으로 추론한
`IMPLICIT`, 문서 잔여부나 fallback을 담는 `SYNTHETIC` 중 하나다. 명시적 section에
제목이 없으면 값을 추정하지 않고 `MISSING_SECTION_TITLE` 이슈를 남긴다.

### 중간 산출물 생성

전체 manifest를 청킹한 결과는 다음 명령으로 `data/canonical_section`에 저장한다.

```powershell
uv run python -m scripts.build_canonical_sections --workers 4
```

```text
data/canonical_section/
├── periodic/<rcept_no>.json
├── major/<rcept_no>.json
├── holding/<rcept_no>.json
├── exchange/<rcept_no>.json
└── manifest.jsonl
```

출력 경로에는 인코딩 영향을 받는 회사명을 사용하지 않는다. 하나의 공시에 본문,
감사보고서, 별도·연결 재무제표 등 XML이 여러 개 있으면 공시 JSON 하나의 `sources`
배열에 모두 저장한다.

```json
{
  "schema_version": "canonical-section-document.v1",
  "source_document": {
    "doc_id": "periodic_20240312000736",
    "rcept_no": "20240312000736",
    "source_path": "raw/periodic/회사/20240312000736_annual_2023_12",
    "doc_group": "periodic"
  },
  "sources": [
    {
      "source_path": "raw/periodic/회사/20240312000736_annual_2023_12/20240312000736.xml",
      "source_sha256": "...",
      "section_collection": {
        "schema_version": "section-collection.v1",
        "source_document": {
          "doc_id": "periodic_20240312000736",
          "rcept_no": "20240312000736",
          "source_path": "raw/periodic/회사/20240312000736_annual_2023_12/20240312000736.xml",
          "doc_group": "periodic"
        },
        "syntax": "DART_XML",
        "parse_status": "RECOVERED",
        "sections": [],
        "issues": []
      }
    }
  ]
}
```

빌더는 각 원문의 SHA-256을 계산해 이전 manifest 및 기존 JSON과 비교한다. 입력과
스키마가 같고 상태가 `SUCCESS` 또는 `RECOVERED`이면 기존 파일을 재사용한다.
`FAILED`와 `PARTIAL`은 다음 실행에서 자동 재시도한다. JSON과 manifest는 임시 파일에
쓴 뒤 원자적으로 교체하며, 중단 후에도 완성된 개별 JSON을 다시 사용할 수 있다.

`manifest.jsonl`에는 공시별 결합 입력 해시, 원문·출력 경로, 건너뛴 파일, 상태,
원문 수, section 수, block 수를 기록한다. PDF+HTML 문서는 현재 HTML만 청킹하고
PDF는 `skipped_files`에 남긴다. 전체 `data` 디렉터리는 Git 제외 대상이다.

현재 활성 manifest 전체 생성 결과는 다음과 같다.

```text
공시:       4,202
원문:       4,617
section:   90,581
block:  1,519,194
SUCCESS:    2,694
RECOVERED:  1,508
FAILED:         0
PARTIAL:        0
```

### 후속 파서 연결

각 `SectionBlockRef.source_refs`로 원본 요소를 다시 찾은 뒤 다음처럼 전달한다.

- `P`, `SPAN_RUN` → paragraph parser
- `TABLE` → `parse_table()`
- `TABLE_GROUP` → `parse_table_group()`
- section의 canonical block 순서 → `table_context_resolver`

이때 section의 `section_path`를 `EvidenceContext` 또는
`resolve_table_contexts(section_path=...)`에 전달한다. 표 문맥으로 소비된 paragraph는
resolver의 `consumed_source_refs`를 이용해 독립 evidence로 다시 출력하지 않는다.

### 상태와 제한사항

| 상태 | 의미 |
|---|---|
| `SUCCESS` | 복구 없이 section 생성 |
| `RECOVERED` | XML 텍스트 복구 후 section 생성 |
| `FAILED` | 문서 트리를 만들지 못함 |

- 번호처럼 보이는 문장 하나만으로는 section을 만들지 않는다.
- 한 셀 제목 표가 `TABLE-GROUP` 안에 있으면 section 제목이 아니라 table context다.
- synthetic root의 제목이 없으면 `title`과 `section_path`를 비워 둔다.
- block parser 실행과 markdown 렌더링은 chunker의 책임이 아니다.

## DART `paragraph_parser`

`converters/paragraph_parser`는 문서의 텍스트 블록을 원문 위치와 인라인 조각을
추적할 수 있는 `CanonicalParagraph`로 변환한다. 화면 너비에 따른 자동 줄바꿈은
문단 경계로 보지 않으며, 하나의 비어 있지 않은 `P`를 하나의 문단으로 유지한다.

### 파일 구성

```text
converters/
├── common/
│   ├── document_loader.py    # XML/HTML 로드와 보수적 복구
│   └── source_models.py      # 문서·evidence 문맥과 원본 참조
└── paragraph_parser/
    ├── paragraph_models.py   # Enum과 불변 데이터 모델
    └── paragraph_parser.py   # 문단 탐지와 canonical 변환
```

공개 함수는 다음 두 개다.

```python
from converters.paragraph_parser.paragraph_parser import (
    parse_paragraph_element,
    parse_paragraphs,
)
```

### 문단 탐지 규칙

- 비어 있지 않은 `P` 하나를 `P` 유형 문단 하나로 만든다.
- 연속된 `P`는 서로 합치지 않는다. 각각 별개의 의미 단위일 수 있기 때문이다.
- `P` 내부의 `SPAN`은 새 문단이 아니라 인라인 `fragment`로 보존한다.
- `P` 밖에서 직접 연속된 `SPAN`만 방어적으로 묶어 `SPAN_RUN` 문단으로 만든다.
- 빈 `P`는 결과에서 제외한다.
- `TABLE`, `TABLE-GROUP`, `CORRECTION`, `TITLE`, `SCRIPT`, `STYLE` 서브트리는
  중복 evidence를 막기 위해 순회하지 않는다.
- 거래소 HTML의 굵은 독립 `SPAN` 제목은 레이아웃 요소로 보고 문단에서 제외한다.

문장 종결 부호, 목록 기호 또는 화면 줄바꿈으로 하나의 `P`를 다시 나누지는 않는다.
문단이 길어 청킹이 필요하면 paragraph parser 이후 단계에서 처리한다.

### 공통 문맥 계약

paragraph parser와 table parser는 모두 `EvidenceContext`를 받아 같은 `context`
구조로 출력한다.

| 필드 | 의미 |
|---|---|
| `section_path` | 상위부터 현재까지의 섹션 제목 경로 |
| `blocks` | 역할, 앞뒤 위치, 텍스트, 원본 참조를 가진 문맥 블록 목록 |

문맥 블록의 `role`은 `TITLE`, `UNIT`, `CAPTION`, `NOTE`, `position`은 `BEFORE`,
`AFTER` 중 하나다. `source_ref`도 함께 저장하므로 markdown에 포함된 문맥을 원문에서
다시 찾을 수 있다. `block_title`, `units`, `captions`, `notes`는 중복 저장하지 않고
`blocks`에서 계산하는 Python property로만 제공한다.

문맥은 이후 `section_canonicalizer`와 `table_context_resolver`가 결정한다. paragraph와 table
parser는 전달받은 문맥을 손실 없이 보존하며, `TABLE-GROUP` 내부의 레이아웃 표와
명시적인 HTML `CAPTION`만 table parser가 직접 추가한다.

### 사용법

```python
from pathlib import Path

from converters.common.source_models import DocumentContext, EvidenceContext
from converters.paragraph_parser.paragraph_parser import parse_paragraphs

path = Path("data/raw/major/셀트리온/20250528000403/20250528000403.xml")
result = parse_paragraphs(
    path.read_bytes(),
    document_context=DocumentContext(
        doc_id="major_20250528000403",
        rcept_no="20250528000403",
        source_path=str(path),
        doc_group="major",
    ),
    content_context=EvidenceContext(
        section_path=("11. 기타 투자판단에 참고할 사항",),
    ),
)

paragraph = result.paragraphs[0]
print(paragraph.text)
```

`parse_paragraphs()`는 `LoadedDocument`, `xml.etree.ElementTree.Element`, `str`,
`bytes`를 받는다. `str`과 `bytes`는 DART XML과 거래소 HTML을 자동 판별한다.
동일한 문서를 correction extractor에서 먼저 처리했다면 HTML 정정 영역처럼 태그명만으로
제외할 수 없는 범위를 `excluded_source_refs=result.excluded_source_refs`로 전달한다.

이미 선택한 단일 `P`만 변환할 때는 원본 위치를 명시한다.

```python
from converters.common.source_models import DocumentSyntax, SourceRef
from converters.paragraph_parser.paragraph_parser import parse_paragraph_element

paragraph = parse_paragraph_element(
    p_element,
    document_context=document_context,
    source_ref=SourceRef(
        syntax=DocumentSyntax.DART_XML,
        element_path="/DOCUMENT[1]/BODY[1]/P[3]",
    ),
    content_context=evidence_context,
)
```

### 출력 예시

```json
{
  "schema_version": "paragraph.v2",
  "parse_status": "SUCCESS",
  "source_document": {
    "doc_id": "major_20250528000403",
    "rcept_no": "20250528000403",
    "source_path": "data/raw/major/셀트리온/20250528000403/20250528000403.xml",
    "doc_group": "major"
  },
  "source_kind": "P",
  "source_refs": [
    {
      "syntax": "DART_XML",
      "element_path": "/DOCUMENT[1]/BODY[1]/P[3]",
      "html_id": null
    }
  ],
  "paragraph_index": 0,
  "context": {
    "section_path": ["11. 기타 투자판단에 참고할 사항"],
    "blocks": []
  },
  "raw_text": "- 상기 '1. 취득예정주식(주)'은 취득후 전량 소각할 계획임.",
  "text": "- 상기 '1. 취득예정주식(주)'은 취득후 전량 소각할 계획임.",
  "leading_marker": {"text": "-", "kind": "BULLET"},
  "source_attributes": {"USERMARK": "F-1"},
  "fragments": [
    {
      "id": "f0",
      "kind": "TEXT",
      "raw_text": "- 상기 '1. 취득예정주식(주)'은 ",
      "text": "- 상기 '1. 취득예정주식(주)'은",
      "source_attributes": {},
      "source_ref": {
        "syntax": "DART_XML",
        "element_path": "/DOCUMENT[1]/BODY[1]/P[3]",
        "html_id": null
      }
    }
  ],
  "issues": []
}
```

`raw_text`는 원본 조각의 문자열을 순서대로 이은 값이고, `text`는 연속 공백을 하나로
정리한 검색·표시용 값이다. `fragments`는 `P.text`, 내부 `SPAN`, 자식의 tail을 DOM
순서대로 보존한다. 명시적인 `BR`은 `raw_text`와 fragment에 줄바꿈으로 보존하지만,
화면 폭 때문에 생긴 줄바꿈은 만들지 않는다. `USERMARK`, `CLASS`, `STYLE`은 해석하지
않고 원본 속성으로 둔다.

`leading_marker`는 문단 맨 앞에 명시된 `-`, `ㆍ`, `※`, `주1)`, `①`, `(1)`,
`1.` 같은 기호만 `BULLET`, `NOTE`, `FOOTNOTE`, `ENUMERATION`으로 분류한다.

### 상태와 제한사항

| 상태 | 의미 |
|---|---|
| `SUCCESS` | 복구 없이 정상 변환 |
| `RECOVERED` | XML 텍스트를 보수적으로 복구한 뒤 변환 |
| `FAILED` | 문서 트리를 만들지 못함 |

- 문단 제목과 소속 섹션은 파서가 추론하지 않는다.
- 시각적 줄바꿈과 들여쓰기는 원문 DOM에 없으면 복원하지 않는다.
- 표 셀의 문단은 table parser가 담당하므로 paragraph 결과에 중복되지 않는다.
- 독립 `SPAN` 묶음은 비표준 원문을 위한 방어 규칙이며 `P`보다 낮은 신뢰도의 출처다.

## `table_context_resolver`

`converters/table_context_resolver`는 한 섹션 안에서 원문 순서대로 정렬된 canonical
paragraph와 table을 받아 표에 인접한 제목, 단위, 캡션, 주석을 연결한다. table parser가
형제 요소를 직접 추측하지 않게 하면서도 table evidence에 렌더링 문맥을 포함하기 위한
조립 단계다.

### 파일 구성

```text
converters/table_context_resolver/
├── resolver_models.py          # 결과 bundle과 소비 참조 모델
└── table_context_resolver.py   # 인접 문맥 판정과 연결
```

### 연결 규칙

- 표 앞의 대괄호·꺾쇠형 한 줄 제목은 `TITLE/BEFORE`로 연결한다.
- `(단위: 원)`, `(단위 : 주)` 형태는 `UNIT/BEFORE`로 연결한다.
- `table-caption`, `tbl_caption` 등의 명시적인 class는 `CAPTION`으로 연결한다.
- `※`, `*`, `주1)`, `- 상기`로 시작하는 인접 문단은 `NOTE`로 연결한다.
- 표 뒤에서는 `CAPTION`과 `NOTE`만 연결하며 일반 서술 문단을 만나면 즉시 멈춘다.
- 다른 표를 가로질러 문맥을 연결하지 않고, 모호한 일반 문장은 소비하지 않는다.

resolver는 canonical block의 입력 순서가 실제 DOM 순서라고 가정한다. 이 순서는 추후
`section_canonicalizer`가 제공해야 한다.

### 사용법과 출력

```python
from converters.table_context_resolver.table_context_resolver import (
    resolve_table_contexts,
)

ordered_blocks = (
    title_paragraph,
    unit_paragraph,
    canonical_table,
    first_note_paragraph,
    second_note_paragraph,
)

resolution = resolve_table_contexts(
    ordered_blocks,
    section_path=("III. 재무에 관한 사항",),
)
bundle = resolution.tables[0]

assert bundle.table.title == "【자기주식 보유현황】"
assert bundle.table.units == ("(단위 : 주)",)
assert len(bundle.table.notes) == 2
```

핵심 출력은 다음과 같다.

```json
{
  "context": {
    "section_path": ["III. 재무에 관한 사항"],
    "blocks": [
      {
        "role": "TITLE",
        "position": "BEFORE",
        "text": "【자기주식 보유현황】",
        "source_ref": {
          "syntax": "DART_XML",
          "element_path": "/DOCUMENT[1]/P[1]",
          "html_id": null
        }
      },
      {
        "role": "UNIT",
        "position": "BEFORE",
        "text": "(단위 : 주)",
        "source_ref": {
          "syntax": "DART_XML",
          "element_path": "/DOCUMENT[1]/P[2]",
          "html_id": null
        }
      },
      {
        "role": "NOTE",
        "position": "AFTER",
        "text": "- 상기 '기초수량'은 사업연도 개시일 기준임.",
        "source_ref": {
          "syntax": "DART_XML",
          "element_path": "/DOCUMENT[1]/P[3]",
          "html_id": null
        }
      }
    ]
  }
}
```

`TableContextResolution.consumed_source_refs`는 표 문맥으로 사용된 paragraph의 원본
참조다. evidence assembler는 이 참조에 해당하는 paragraph를 독립 evidence로 다시
출력하지 않아야 한다. canonical paragraph 자체는 삭제하거나 변경하지 않는다.

## DART `table_parser`

`converters/table_parser`는 DART 공시 XML 또는 거래소 HTML의 `<TABLE>`을
원본 셀 추적이 가능한 구조화 데이터로 변환한다.

이 파서는 숫자 변환이나 업무 코드 해석보다 표의 구조를 보존하는 데 초점을 둔다.
따라서 `1,000`, `-`, 빈 문자열 같은 값은 문자열 그대로 유지하며, `ACODE`,
`AUNIT`, `AUNITVALUE`는 별도 코드 필드로 승격하지 않고 원본 속성으로만 보존한다.

### 파일 구성

```text
converters/
├── README.md
└── table_parser/
    ├── table_models.py   # Enum과 불변 데이터 모델
    └── table_parser.py   # 격자 구성, 분류, 파싱, 복구
```

직접 사용할 공개 함수는 다음과 같다.

```python
from converters.table_parser.table_parser import (
    build_logical_grid,
    classify_table_type,
    extract_cell_text,
    parse_table,
    parse_table_fragment,
    parse_table_group,
)
```

### 처리 범위

지원하는 주요 입력은 다음과 같다.

- DART XML의 `TABLE`, `TR`, `THEAD`, `TBODY`, `TH`, `TD`, `TE`, `TU`
- `ROWSPAN`과 `COLSPAN`이 있는 병합 셀
- `P`, `SPAN` 등 셀 내부의 중첩 텍스트
- `TABLE-GROUP` 안의 제목, 단위, 본문 표 조합
- 닫는 태그가 생략된 거래소 HTML 표
- bare ampersand 등 일부 잘못된 XML 텍스트의 보수적 복구

현재 처리 범위에 포함되지 않는 것은 다음과 같다.

- 공시 맨 앞의 정정공시 안내 표 식별 및 자동 제외
- 숫자, 날짜, 백분율 등의 자료형 변환
- `ACODE`, `AUNIT`에 대한 업무 의미 해석
- 공시 전체에서 여러 표를 자동 순회하고 결과를 묶는 작업

정정공시 안내 표가 포함된 문서는 호출자가 대상 `TABLE` 또는 대상 표 조각을 먼저
선택해야 한다. `parse_table_fragment()`에 여러 표가 들어 있는 전체 문서를 넘기면
첫 번째 표만 변환한다.

### 처리 과정

파서는 다음 순서로 표를 처리한다.

1. 각 원본 셀에 `c0`, `c1`, ... 순서로 ID를 부여한다.
2. `ROWSPAN`과 `COLSPAN`을 펼쳐 직사각형 논리 격자를 만든다.
3. 표를 `KV_TABLE`, `R_TABLE`, `LAYOUT_TABLE`, `UNKNOWN` 중 하나로 분류한다.
4. 셀 역할과 유형별 `content`를 생성한다.
5. 원본 위치와 속성을 포함한 `CanonicalTable`을 반환한다.

병합 셀이 여러 격자 좌표를 차지하더라도 새로운 셀을 복제하지 않는다. 모든 점유
좌표가 동일한 `SourceCell`을 가리키므로 원본 셀과 결과 사이의 대응 관계가 유지된다.

### 표 유형

#### `KV_TABLE`

키와 값의 관계를 표현하는 표다. 다음 중 하나를 만족하는 셀을 강한 값 셀로 본다.

- `TE`이면서 `ACODE`가 있음
- `TU`이면서 `AUNIT`가 있음
- 거래소 HTML에서 `xforms_input` 클래스를 가짐

각 값은 `content.fields`에 기록된다.

```json
{
  "value_cell": "c3",
  "raw_value": "2,967,759",
  "key_paths": [
    {
      "path": [
        "보유주식등의 수 및 보유비율",
        "이번 보고서",
        "보유주식등의 수"
      ],
      "cells": ["c0", "c2", "c1"]
    }
  ],
  "context_status": "RESOLVED"
}
```

`key_paths`는 별도의 행 경로와 열 경로를 최종 결합한 결과다. 하나의 병합 값이 여러
문맥에 속하면 경로를 모두 보존하고 `context_status`를 `MULTI_CONTEXT`로 설정한다.

#### `R_TABLE`

열 머리글과 반복되는 본문 행으로 구성된 레코드 표다. 다음 순서로 머리글을 찾는다.

1. `THEAD`가 있으면 그 안의 행을 머리글로 사용한다.
2. `THEAD`가 없으면 첫 행과 뒤따르는 반복 행의 폭을 비교해 첫 행을 추론한다.

다단 머리글은 열별 `header_path`로 보존한다. 각 레코드는 값을 복사하지 않고
`cells_by_column`에서 열 ID와 원본 셀 ID를 연결한다.

```json
{
  "columns": [
    {
      "id": "col_0",
      "index": 0,
      "header_path": ["구분"],
      "header_cells": ["c0"]
    },
    {
      "id": "col_1",
      "index": 1,
      "header_path": ["내용"],
      "header_cells": ["c1"]
    }
  ],
  "records": [
    {
      "record_index": 0,
      "source_row": 1,
      "row_type": "DATA",
      "row_context": [],
      "row_context_cells": [],
      "cells_by_column": {
        "col_0": "c2",
        "col_1": "c3"
      }
    }
  ]
}
```

본문에 `TE` 또는 `TU` 값 셀이 있으면 그 앞의 일반 셀을 `row_context`로 보존한다.
`소계`와 `총계` 형태의 행은 각각 `SUBTOTAL`, `TOTAL`로 표시한다.

#### `LAYOUT_TABLE`

본문 데이터가 아니라 제목, 단위 또는 주석 역할을 하는 작은 표다.

```json
{
  "layout_role": "UNIT",
  "values": ["(단위 : 주)"]
}
```

`layout_role`은 다음 중 하나다.

- `TITLE`: 표 제목
- `UNIT`: 단위 표기
- `NOTE`: 제목이나 단위로 확정되지 않은 보조 문구

#### `UNKNOWN`

다른 유형으로 안전하게 판단할 수 없는 표다. 의미를 억지로 추론하지 않고 논리 격자의
셀 ID를 그대로 남긴다.

```json
{
  "rows": [
    ["c0", null],
    ["c1", "c2"]
  ]
}
```

### 데이터 모델 명세

모든 모델은 `frozen=True`, `slots=True`인 dataclass다. 파서는 입력
`Element`를 변경하지 않는다.

#### Enum

| Enum | 값 |
|---|---|
| `TableType` | `KV_TABLE`, `R_TABLE`, `LAYOUT_TABLE`, `UNKNOWN` |
| `ParseStatus` | `SUCCESS`, `RECOVERED`, `PARTIAL`, `FAILED` |
| `CellRole` | `EMPTY`, `HEADER`, `KEY`, `VALUE`, `CONTEXT`, `UNKNOWN` |
| `ContextStatus` | `RESOLVED`, `MULTI_CONTEXT`, `NONE` |
| `RowType` | `DATA`, `SUBTOTAL`, `TOTAL`, `UNKNOWN` |
| `LayoutRole` | `TITLE`, `UNIT`, `NOTE` |
| `SourceSyntax` | `AUTO`, `DART_XML`, `HTML` |

#### `TableContext`

호출자가 알고 있는 출처 정보를 결과에 전달한다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `source_path` | `str \| None` | 원본 공시 파일 경로 |
| `table_index` | `int \| None` | 원본 문서 안의 표 순번 |
| `table_group_class` | `str \| None` | 상위 `TABLE-GROUP`의 `ACLASS` |

#### `SourceCell`

하나의 원본 셀을 나타낸다.

| 필드 | 설명 |
|---|---|
| `id` | 표 안에서 부여한 셀 ID |
| `tag` | 원본 태그: `TD`, `TH`, `TE`, `TU` |
| `row_start`, `col_start` | 셀이 시작하는 0 기반 좌표 |
| `row_end`, `col_end` | 셀이 끝나는 배타적 좌표 |
| `raw_value` | 공백을 정리하고 문단을 줄바꿈으로 연결한 문자열 |
| `text_segments` | 문단 경계를 보존한 문자열 튜플 |
| `source_attributes` | 허용 목록에 포함된 원본 속성 |

보존하는 원본 속성은 다음과 같다.

```text
ACODE, AUNIT, AUNITVALUE, ACONTEXT, ADECIMAL, AUPDATECONT,
ENG, ROWSPAN, COLSPAN, XFORMS_INPUT
```

`AUPDATECONT="N"`은 키/값 판정 규칙이 아니라 출처 메타데이터로만 취급한다.

#### `LogicalGrid`

| 필드 | 설명 |
|---|---|
| `rows` | 병합 셀을 펼친 직사각형 셀 격자 |
| `cells` | 중복되지 않은 원본 셀 목록 |
| `issues` | 격자 구성 중 발견한 문제 |
| `height`, `width` | 논리 격자의 행·열 크기 |

#### `CanonicalTable`

파싱이 완료된 단일 표다. `to_dict()`를 호출하면 다음 공통 구조가 된다.

```json
{
  "schema_version": "table.v3",
  "table_type": "KV_TABLE",
  "parse_status": "SUCCESS",
  "source": {
    "source_path": "data/raw/holding/example.xml",
    "table_index": 1,
    "table_group_class": null,
    "aclass": "EXTRACTION",
    "syntax": "DART_XML"
  },
  "dimensions": {
    "rows": 1,
    "columns": 2
  },
  "context": {
    "section_path": [],
    "blocks": []
  },
  "cells": [],
  "content": {},
  "issues": []
}
```

`cells`와 `content`의 상세 구조는 표 유형에 따라 달라진다. 모든 셀에는
`id`, `role`, `tag`, 좌표, `raw_value`, `text_segments`,
`source_attributes`가 포함된다.

#### `CanonicalTableGroup`

`TABLE-GROUP`의 보조 표와 본문 표를 묶은 결과다.

| 필드 | 설명 |
|---|---|
| `table_group_class` | 그룹의 `ACLASS` |
| `context_blocks` | 제목, 단위, 주석 표의 내용과 출처 |
| `tables` | 보조 표가 연결된 본문 `CanonicalTable` 목록 |
| `issues` | 그룹 수준 문제 목록 |

앞에 있는 제목·단위 레이아웃 표는 다음 본문 표에 `BEFORE`로 연결한다. 본문 표 뒤의
주석 레이아웃 표는 이전 본문 표에 `AFTER`로 연결한다. 모든 항목은
`context.blocks`에 원본 순서로 저장하며 Python 객체의 `title`, `units`, `captions`,
`notes` property로 역할별 텍스트를 조회할 수 있다.

### 사용법

#### 이미 파싱한 XML `Element` 변환

```python
import json
import xml.etree.ElementTree as ET

from converters.common.source_models import (
    ContextPosition,
    ContextRole,
    EvidenceContext,
    EvidenceContextBlock,
)
from converters.table_parser.table_models import TableContext
from converters.table_parser.table_parser import parse_table

xml = """
<TABLE ACLASS="EXTRACTION">
  <TR>
    <TD>발행회사명</TD>
    <TE ACODE="CRP_NM">(주)에스엠엔터테인먼트</TE>
  </TR>
</TABLE>
"""

element = ET.fromstring(xml)
result = parse_table(
    element,
    context=TableContext(
        source_path="sample.xml",
        table_index=0,
    ),
    content_context=EvidenceContext(
        section_path=("III. 재무에 관한 사항", "1. 요약재무정보"),
        blocks=(
            EvidenceContextBlock(
                role=ContextRole.TITLE,
                position=ContextPosition.BEFORE,
                text="요약재무정보",
            ),
            EvidenceContextBlock(
                role=ContextRole.UNIT,
                position=ContextPosition.BEFORE,
                text="(단위: 원)",
            ),
        ),
    ),
)

print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
```

`parse_table()`은 반드시 `TABLE` 요소를 받아야 하며, 다른 요소를 전달하면
`ValueError`를 발생시킨다.

#### XML 문자열 또는 bytes 변환

```python
from converters.table_parser.table_parser import parse_table_fragment

result = parse_table_fragment(
    '<TABLE><TR><TD>금액</TD>'
    '<TE ACODE="AMOUNT">320,939,749,998</TE></TR></TABLE>'
)

field = result.content["fields"][0]
assert field["raw_value"] == "320,939,749,998"
assert field["key_paths"][0]["path"] == ["금액"]
```

`SourceSyntax.AUTO`가 기본값이다. `<html>` 또는 `xforms` 흔적이 있으면 HTML로,
그렇지 않으면 DART XML로 처리한다. 구문을 명시하려면 다음처럼 호출한다.

```python
from converters.table_parser.table_models import SourceSyntax
from converters.table_parser.table_parser import parse_table_fragment

result = parse_table_fragment(raw_html, syntax=SourceSyntax.HTML)
```

#### 거래소 HTML 변환

```python
html = """
<html><body><table>
  <tr>
    <td><span>1. 제목</span>
    <td><span class="xforms_input">아티스트 전속계약 체결의 건</span>
</table></body></html>
"""

result = parse_table_fragment(html)

assert result.source["syntax"] == "HTML"
assert result.content["fields"][0]["key_paths"][0]["path"] == ["1. 제목"]
```

HTML 모드에서는 생략된 `TR`/`TD` 종료 태그를 허용한다. 여러 표가 들어 있으면 첫
번째 표만 변환하고 `MULTIPLE_TABLES` 이슈를 추가한다.

#### `TABLE-GROUP` 변환

```python
import xml.etree.ElementTree as ET

from converters.table_parser.table_parser import parse_table_group

group = ET.fromstring("""
<TABLE-GROUP ACLASS="TBL_OWN_STK">
  <TABLE><TR><TD>【자기주식 보유현황】</TD></TR></TABLE>
  <TABLE><TR><TU AUNIT="STOCK" AUNITVALUE="1">(단위 : 주)</TU></TR></TABLE>
  <TABLE>
    <THEAD><TR><TH>구분</TH><TH>수량</TH></TR></THEAD>
    <TBODY><TR><TD>보통주식</TD><TE ACODE="COUNT">10</TE></TR></TBODY>
  </TABLE>
</TABLE-GROUP>
""")

result = parse_table_group(group)
table = result.tables[0]

assert table.title == "【자기주식 보유현황】"
assert table.units == ("(단위 : 주)",)
assert table.table_type.value == "R_TABLE"
```

#### 논리 격자만 사용

```python
import xml.etree.ElementTree as ET

from converters.table_parser.table_parser import build_logical_grid

table = ET.fromstring("""
<TABLE>
  <TR><TD ROWSPAN="2">A</TD><TD>B</TD></TR>
  <TR><TD>C</TD></TR>
</TABLE>
""")

grid = build_logical_grid(table)

assert (grid.height, grid.width) == (2, 2)
assert grid.rows[0][0] is grid.rows[1][0]
```

### 입출력 예시

#### KV 표

입력:

```xml
<TABLE>
  <TR>
    <TD>보고구분</TD>
    <TU AUNIT="RPT" AUNITVALUE="2">변동</TU>
  </TR>
</TABLE>
```

핵심 출력:

```json
{
  "table_type": "KV_TABLE",
  "content": {
    "fields": [
      {
        "value_cell": "c1",
        "raw_value": "변동",
        "key_paths": [
          {"path": ["보고구분"], "cells": ["c0"]}
        ],
        "context_status": "RESOLVED"
      }
    ]
  }
}
```

#### 레코드 표

입력:

```xml
<TABLE>
  <TR><TD>구분</TD><TD>내용</TD></TR>
  <TR><TD>발행회사의 경영</TD><TD>의결권을 공동으로 행사</TD></TR>
  <TR><TD>주식처분제한</TD><TD>제3자 처분 제한</TD></TR>
</TABLE>
```

핵심 출력:

```json
{
  "table_type": "R_TABLE",
  "content": {
    "columns": [
      {"id": "col_0", "header_path": ["구분"]},
      {"id": "col_1", "header_path": ["내용"]}
    ],
    "records": [
      {
        "source_row": 1,
        "row_type": "DATA",
        "cells_by_column": {"col_0": "c2", "col_1": "c3"}
      },
      {
        "source_row": 2,
        "row_type": "DATA",
        "cells_by_column": {"col_0": "c4", "col_1": "c5"}
      }
    ]
  }
}
```

#### 레이아웃 표

입력:

```xml
<TABLE><TR><TD>【자기주식 취득 결정 전 자기주식 보유현황】</TD></TR></TABLE>
```

핵심 출력:

```json
{
  "table_type": "LAYOUT_TABLE",
  "content": {
    "layout_role": "TITLE",
    "values": ["【자기주식 취득 결정 전 자기주식 보유현황】"]
  }
}
```

#### 판정 불가 표

입력:

```xml
<TABLE>
  <TR><TD>A</TD></TR>
  <TR><TD>B</TD><TD>C</TD></TR>
</TABLE>
```

핵심 출력:

```json
{
  "table_type": "UNKNOWN",
  "content": {
    "rows": [
      ["c0", null],
      ["c1", "c2"]
    ]
  }
}
```

### 텍스트와 값 보존 규칙

- 연속 공백은 하나의 공백으로 정리한다.
- 셀 안의 최상위 `P` 요소는 `text_segments`에서 별도 문단으로 보존한다.
- 여러 문단의 `raw_value`는 줄바꿈 문자로 연결한다.
- 빈 값 `""`과 대시 값 `"-"`를 서로 다르게 보존한다.
- 숫자의 쉼표, 괄호, 단위 기호를 제거하거나 변환하지 않는다.
- 중첩된 하위 `TABLE`의 행과 셀은 상위 표 소유 셀로 포함하지 않는다.

### 파싱 상태와 이슈

| 상태 | 의미 |
|---|---|
| `SUCCESS` | 복구 없이 정상 변환 |
| `RECOVERED` | XML 텍스트를 보수적으로 복구한 후 변환 |
| `PARTIAL` | 결과는 생성했지만 격자 겹침 등 오류가 있음 |
| `FAILED` | XML/HTML을 표로 구성하지 못함 |

`issues`의 각 항목은 다음 구조를 갖는다.

```json
{
  "code": "XML_RECOVERED",
  "severity": "WARNING",
  "message": "The fragment required conservative text repair before parsing."
}
```

현재 생성될 수 있는 주요 이슈는 다음과 같다.

| 코드 | 의미 |
|---|---|
| `INVALID_SPAN` | 잘못된 병합 크기를 1로 대체함 |
| `GRID_OVERLAP` | 논리 격자에서 서로 다른 셀이 겹침 |
| `XML_RECOVERED` | 잘못된 XML 텍스트를 복구함 |
| `MULTIPLE_TABLES` | 조각에 여러 표가 있어 첫 표만 처리함 |
| `PARSE_FAILED` | 복구 후에도 표를 만들지 못함 |

### 제한사항과 호출자 책임

- 분류는 구조 기반 휴리스틱이므로 생소한 표는 `UNKNOWN`이 될 수 있다.
- `THEAD` 없는 레코드 표는 최소 3행, 2열 이상이며 반복 행이 확인될 때만 추론한다.
- `parse_table_fragment()`는 문서 전체의 표 목록을 반환하는 API가 아니다.
- 정정공시 안내 표는 자동 제외하지 않는다. 호출자가 실제 본문 표를 골라 전달한다.
- `source_attributes`는 원본 추적용이다. 업무 코드로 사용하려면 별도의 해석 계층이 필요하다.
- 셀 값의 자료형 변환과 정규화는 이 파서 이후 단계에서 수행한다.

### 테스트

저장소 루트에서 다음 명령으로 전체 회귀 테스트를 실행한다.

```powershell
uv run pytest -q
```

## `evidence_builder`

`converters/evidence_builder`는 graph-facing `data/canonical_section`과 원문을
다시 대조하여 section 하나당 canonical Evidence Fragment JSON 하나를 생성한다.
경량화된 canonical section에는 block ref가 없으므로 원문을 동일한 section chunker로
재분할하고 graph projection이 저장된 section 목록과 완전히 일치할 때만 생성한다.

### Evidence 생성 규칙

| 입력 | 최종 Evidence |
|---|---|
| `P`, `SPAN_RUN` | paragraph 하나당 `TEXT` 하나 |
| `KV_TABLE` | table 전체를 `TABLE` 하나 |
| 모든 `R_TABLE` | table metadata Evidence 하나, row는 최상위 `records`에 저장 |
| `UNKNOWN` | layout이면 제외하고, 나머지는 허용된 TEXT/KV/R schema로 보존 |
| `LAYOUT_TABLE` | 독립 Evidence를 만들지 않고 인접 table의 제목·단위·주석으로 사용 |
| `IMAGE` | 현재 생성하지 않고 `IMAGE_SKIPPED` 통계에 기록 |

표 제목·단위·캡션·주석으로 소비된 paragraph와 layout table은 별도 Evidence로
중복 생성하지 않는다. section 정보는 최상위 `section_id`로만 참조하며 title/path와
원문 provenance는 `canonical_section` 및 build manifest에서 관리한다.

### 최종 Evidence 예시

```json
{
  "schema_version": "evidence-fragment.v2",
  "section_id": "section:20250101000001:src0:s12",
  "evidence_list": [
    {
      "evidence_id": "evidence:20250101000001:src0:s12:e0",
      "evidence_type": "TABLE",
      "table_type": "R_TABLE",
      "storage_mode": "SECTION_RECORDS",
      "order": 0,
      "payload": {
        "table_id": "rtable:20250101000001:src0:s12:t0",
        "title": "주요 제품 현황",
        "units": ["(단위 : 억원, %)"],
        "headers": [["부문"], ["매출액"]],
        "record_count": 720
      }
    }
  ],
  "records": [
    {
      "table_id": "rtable:20250101000001:src0:s12:t0",
      "record_index": 0,
      "row_type": "DATA",
      "row_context": [],
      "values": ["DX 부문", "102,300"]
    }
  ]
}
```

모든 R-table의 Evidence와 row는 `table_id`로 연결한다. header는 table metadata에
한 번만 저장하고 각 record의 `values`는 header 순서와 대응하는 문자열 배열이다.
`markdown`, hash, source path와
source cell은 Fragment에 저장하지 않는다.

### 전체 생성과 검증

```powershell
uv run python -m scripts.build_evidence_fragments --workers 4
uv run python -m scripts.validate_evidence_fragments
```

출력은 다음 구조로 원자적으로 저장한다.

```text
data/evidence_fragment/
├── manifest.jsonl
├── periodic/{rcept_no}/src{source_index}__s{section_index}.json
├── major/{rcept_no}/src{source_index}__s{section_index}.json
├── holding/{rcept_no}/src{source_index}__s{section_index}.json
└── exchange/{rcept_no}/src{source_index}__s{section_index}.json
```

각 JSON은 `section_id`, 순서가 보존된 `evidence_list`, 대형 R-table의 `records`를
저장한다. 동일한 canonical section 및 원문 SHA-256 출력은 재사용한다. validator는
section 1:1 coverage, ID와 순서, 금지 field, table-record 연결, manifest 개수와
타입별 통계를 검사한다.

테스트는 병합 셀, 계층형 키, 다중 문맥, 다단 머리글, `THEAD` 없는 레코드 표,
소계·합계, 제목·단위 연결, 거래소 HTML, XML 복구, 빈 값 보존을 포함한다.
