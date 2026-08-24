# 미래에셋 AI 페스티벌 / Team Sortie

## DART Disclosure Analyst

DART 공시 원문을 구조화해 Neo4j Knowledge Graph와 Vector Search로 분석하는 프로젝트입니다. 자연어 답변에 사용된 Evidence를 Section과 Disclosure까지 추적할 수 있는 구조를 목표로 합니다.

## Architecture

```text
DART XML / HTML
        ↓
Section Canonicalizer
        ↓
data/canonical_section
        ↓
Paragraph / Table Parser
        ↓
Evidence Fragment
   ├── evidence_list           → Neo4j Evidence
   └── evidence_list + records → Qdrant Points
        ↓
Hybrid Retrieval → Agent
```

`canonical_section`은 그래프에 필요한 section 정보만 저장합니다. 원문은 Evidence 생성 시 다시 파싱하며, section 하나당 `evidence_fragment` JSON 하나를 만듭니다.

## Evidence Fragment

각 Fragment의 최상위 구조는 다음과 같습니다.

```json
{
  "schema_version": "evidence-fragment.v2",
  "section_id": "section:20250101000001:src0:s12",
  "evidence_list": [],
  "records": []
}
```

- `TEXT`: 의미 단위 본문입니다. `BODY`, `NOTE`, `REFERENCE_NOTICE` 역할을 가집니다.
- `KV_TABLE`: 표 전체를 Evidence 하나로 저장합니다.
- `R_TABLE`: table metadata만 Evidence에 저장하고 모든 row는 `records`에 둡니다.
- `LAYOUT_TABLE`: 독립 Evidence로 만들지 않고 제목·단위·주석 문맥으로만 사용합니다.

본문의 소제목은 별도 Evidence 대신 `heading_path`로 전파하고, 표 설명문은 `captions` 배열로 연결합니다. 다른 공시를 가리키는 DART 링크는 Evidence의 `references`에 보존합니다.

Evidence에는 `section_id`를 통한 소속 관계만 저장합니다. 문서 정보와 section 경로는 `canonical_section`에서 조회하며, markdown·원문 경로·hash·cell provenance는 Fragment에 중복 저장하지 않습니다.

## Knowledge Graph

핵심 관계는 다음과 같습니다.

```text
(Disclosure)-[:HAS_SECTION]->(Section)
(Section)-[:HAS_SECTION]->(Section)
(Section)-[:HAS_EVIDENCE]->(Evidence)
(Event)-[:SUPPORTED_BY]->(Evidence)
```

Neo4j에는 Disclosure, Section, Evidence, Entity, Event와 그 관계를 저장합니다. R-table의 row는 Evidence Fragment의 `records`에 보존한 뒤, R_TABLE 적재 전략에 따라 생성된 Qdrant Point의 `canonical.records` payload에 포함합니다. 여러 Point로 나뉜 경우에도 `table_id`와 row 범위로 같은 표의 record를 추적할 수 있습니다.

## Retrieval

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

Graph Search는 관계 탐색에, Vector Search는 의미 기반 원문 검색에 사용합니다. 정형 데이터는 가능한 한 규칙 기반으로 추출하고 LLM은 자유 서술형·예외 사례에 제한적으로 사용합니다.

## Pipeline Commands

```powershell
uv run python -m scripts.build_canonical_sections --workers 4
uv run python -m scripts.build_evidence_fragments --workers 4
uv run python -m scripts.validate_evidence_fragments
uv run python scripts/validate_converter_pipeline.py --profile quick --semantic-only
```

## Current Status

- [x] DART XML/HTML 로드 및 보수적 복구
- [x] Section canonicalization
- [x] Paragraph·table parsing
- [x] Semantic TEXT segmentation
- [x] Section 단위 Evidence Fragment 생성
- [x] R-table record 분리 및 보존
- [x] 고정 공시 회귀 검증
- [ ] Neo4j·Qdrant 적재
- [ ] Embedding 및 Hybrid Retrieval
- [ ] Event extraction과 Agent workflow

## Design Principles

- 원문 값을 임의로 보정하지 않습니다.
- 중복 정보는 저장하지 않고 ID와 관계로 연결합니다.
- 구조화 가능한 데이터는 결정론적으로 처리합니다.
- 모든 주요 판단은 Evidence에서 원문 공시까지 추적할 수 있어야 합니다.


## Quick Commands
#### run at root directory
#### DB insert는 --dry-run 옵션 가능

### Section canonicalize
```powershell
uv run python -m scripts.build_canonical_sections --force --workers 4 --progress-every 100
```

### Evidence builder
```powershell
uv run python -m scripts.build_evidence_fragments --workers 4 --progress-every 100
```

### Neo4j insert
#### Company nodes insert (from universe.csv)
```powershell
uv run python -m knowledge_graph.insertUniverse
```
#### Disclosure, Section, Evidence insert
```powershell
uv run python -m knowledge_graph.insertDSE --limit 100 --random-seed 42
```

### Qdrant insert
```powershell
uv run python -m vector_db.insert_points --limit 4 --random-seed 42 --batch-size 100
```

