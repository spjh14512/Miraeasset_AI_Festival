# 미래에셋 AI 페스티벌 / Team Sortie
## 공시 분석 Agent

# DART Disclosure Analyst

DART 공시 원문을 구조화하고 **Neo4j Knowledge Graph + Vector Search**를 활용해 공시를 분석하는 AI Agent 프로젝트입니다.

최종 목표는 자연어 질문에 대해 공시 데이터를 분석하고, **답변에 사용한 공시와 원문 근거까지 추적 가능한 답변**을 제공하는 것입니다.

## Architecture

```text
DART XML
    ↓
Raw Fragment Extraction
    ↓
Canonicalization
    ↓
Canonical JSON
    ├──────────────┬───────────────┐
    ↓              ↓               ↓
 Markdown       Evidence      Entity / Event
                   ↓               ↓
               Embedding          Neo4j
                   │               │
                   └───────┬───────┘
                           ↓
                    Hybrid Retrieval
                           ↓
                         Agent
```

`Canonical JSON`을 구조화된 기준 데이터로 사용하며 Markdown, Evidence, Entity, Event는 여기서 파생합니다.

---

## Document Processing

### 1. Raw Extraction ✅

DART XML에서 분석에 필요한 Section, Table 등의 원문 구조를 추출합니다.

주요 DART 태그:

| 태그      | 의미             |
| ------- | -------------- |
| `TABLE` | 표              |
| `TR`    | 행              |
| `TD`    | label / header |
| `TE`    | 입력 데이터         |
| `TU`    | 코드형 데이터        |
| `P`     | 문단             |
| `SPAN`  | 텍스트 단위         |

`ACODE`, `AUNIT`, `AUNITVALUE`, `ACLASS` 등 의미 있는 metadata는 유지합니다.

---

### 2. Canonicalization ✅

Raw XML/HTML-like 구조를 정규화된 JSON으로 변환합니다.

```text
Raw Table
    ↓
ROWSPAN / COLSPAN 복원
    ↓
Field Context 추출
    ↓
Value Normalization
    ↓
Canonical JSON
```

예:

```json
{
  "label": "보유주식등의 수",
  "code": "SUM_BMT_CNT",
  "raw_value": "1,238,767,819",
  "value": 1238767819,
  "data_type": "integer",
  "unit": "share",
  "context": {
    "group": "보유주식등의 수 및 보유비율",
    "row": "직전 보고서"
  }
}
```

원본 값과 DART field code는 항상 유지합니다.

---

### 3. Markdown Rendering 🚧

Canonical JSON을 사람과 LLM이 읽기 쉬운 Markdown으로 변환합니다.

```markdown
## 보유주식등의 수 및 보유비율

| 구분 | 보유주식등의 수 | 보유비율 |
|---|---:|---:|
| 직전 보고서 | 1,238,767,819주 | 20.75% |
| 이번 보고서 | 1,237,964,708주 | 20.74% |
```

Markdown은 기준 데이터가 아니라 Canonical JSON에서 생성되는 View입니다.

---

## Evidence

Evidence는 Agent가 답변의 근거로 사용하는 **공시 원문의 최소 의미 단위**입니다.

```text
작은 표     → Table 단위
반복 표     → Row 단위
서술형 내용 → Paragraph 단위
```

각 Evidence는 원문 위치와 Canonical 데이터를 참조합니다.

```text
Event / Fact
     ↓
SUPPORTED_BY
     ↓
Evidence
     ↓
Section
     ↓
Disclosure
     ↓
DART XML
```

Evidence 단위로 embedding을 생성하여 Vector Search에도 사용합니다.

---

## Knowledge Graph

Graph DB는 Neo4j를 사용합니다.

현재 주요 노드:

```text
Entity
├── Person
└── Organization
     └── Company

Disclosure
Section
Evidence
Event
HoldingPosition
ShareContract
```

주요 관계:

```text
(Entity)-[:FILES]->(Disclosure)

(Disclosure)-[:TARGETS]->(Entity)
(Disclosure)-[:REPORTS]->(Event)
(Disclosure)-[:HAS_SECTION]->(Section)
(Disclosure)-[:CORRECTS]->(Disclosure)

(Section)-[:HAS_SECTION]->(Section)
(Section)-[:HAS_EVIDENCE]->(Evidence)

(Event)-[:SUPPORTED_BY]->(Evidence)

(Entity)-[:HAS_POSITION]->(HoldingPosition)
(HoldingPosition)-[:SUPPORTED_BY]->(Evidence)
```

같은 의미의 관계는 노드 타입이 달라도 동일한 Relation Type을 재사용합니다.

---

## Event Extraction

4,000개 이상의 공시를 모두 LLM으로 분석하지 않습니다.

기본 전략은 **Rule-based Extraction**입니다.

```text
Canonical JSON
      ↓
공시 유형별 Extraction Rule
      ↓
Validation
      ↓
Event
```

DART의 `ACODE`, `AUNIT`, 표 구조 등을 활용해 정형 Event를 추출합니다.

LLM은 다음과 같은 경우에만 제한적으로 사용합니다.

* 자유 서술형 의미 분석
* Rule-based extraction 실패
* 희귀하거나 복잡한 공시 유형

---

## Retrieval

최종 검색은 하나의 방식에 의존하지 않는 Hybrid Retrieval을 목표로 합니다.

```text
User Query
    ├── Graph Search
    ├── Vector Search
    └── Structured / Keyword Search
              ↓
           Evidence
              ↓
            Agent
```

Graph는 Entity와 Event 관계 탐색에, Vector Search는 의미 기반 원문 검색에 사용합니다.

---

## Roadmap

### Document Processing

* [x] DART XML 구조 분석
* [x] Raw fragment extraction
* [x] Canonicalizer 구현
* [x] ROWSPAN / COLSPAN 처리
* [x] Value normalization
* [ ] Markdown renderer
* [ ] Regression test

### Evidence & Retrieval

* [ ] Evidence schema 및 chunking
* [ ] Embedding text 생성
* [ ] Evidence embedding
* [ ] Vector Search
* [ ] Hybrid Retrieval

### Knowledge Graph

* [ ] Ontology 확정
* [ ] Entity resolution
* [ ] Disclosure / Section / Evidence 적재
* [ ] Event extractor
* [ ] Event ↔ Evidence 연결
* [ ] Neo4j index / constraint

### Agent

* [ ] Query routing
* [ ] Graph-first retrieval
* [ ] Vector-first retrieval
* [ ] Context construction
* [ ] Evidence citation
* [ ] LangGraph workflow
* [ ] End-to-end evaluation

---

## Design Principles

* **Preserve the source** — 원본 값과 위치를 유지합니다.
* **Canonical JSON first** — 파생 데이터의 기준은 Canonical JSON입니다.
* **Evidence before inference** — 모든 주요 판단은 원문 근거까지 추적할 수 있어야 합니다.
* **Prefer deterministic extraction** — 구조화 데이터는 가능한 한 규칙 기반으로 처리합니다.
* **Hybrid retrieval** — Graph와 Vector Search의 장점을 함께 활용합니다.
