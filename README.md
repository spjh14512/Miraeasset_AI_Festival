# 미래에셋 AI 페스티벌 / Team Sortie

## 평가용 API End-point

> Public API End-point: `https://101.79.22.226/answer` (배포 후 실제 주소로 교체)

평가 API는 별도 인증 header 없이 다음 요청을 처리합니다.

```http
GET /answer?question_id=Q-001&question=평가%20질의
```

응답은 `application/json`이며 모든 필드는 문자열입니다.

```json
{
  "question_id": "Q-001",
  "question": "평가 질의",
  "retrieved_context": "{\"citations\": [...], \"results\": [...]}",
  "think_trace": "{\"query_text\": \"...\", \"question_analysis\": {...}, ...}",
  "answer": "자연어 답변 + 인용"
}
```

- `retrieved_context`: 검색된 공시와 섹션 정보 (JSON 형식)
- `think_trace`: 질의 분석, 검색 이력, 계산 이력, 검증 상태
- `answer`: 근거 문서와 구간까지 추적 가능한 자연어 답변

로컬 서버 실행:

```powershell
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

로컬 호출 예시:

```powershell
curl.exe --get "http://localhost:8000/answer" `
  --data-urlencode "question_id=Q-001" `
  --data-urlencode "question=삼성전자의 설비 투자를 알려줘"
```

---

## Project Overview

DART Disclosure Analyst는 공시 원문 구조화부터 에이전트 응답까지 전체 파이프라인을 통합한 금융 정보 검색 시스템입니다.

**핵심 특징:**

- **정확한 출처 추적**: 답변의 근거를 공시 문서 → 섹션 → 증거 조각까지 명확하게 표시
- **보수적 파싱**: 원문 값을 임의로 보정하지 않으며, 구조화 가능한 데이터는 결정론적으로 처리
- **하이브리드 검색**: 그래프 탐색 + 벡터 검색 + 키워드/정형 검색 조합으로 다양한 쿼리 커버
- **제한 인식**: 데이터 부족 또는 모호한 질의는 명확하게 선택지를 제시하거나 역질문

---

## Architecture & Pipeline

```text
DART XML / HTML 공시
        ↓
   [1] Section Canonicalizer (원문 구간 식별)
        ↓
   data/canonical_section/ (구조화된 섹션)
        ↓
   [2] Paragraph / Table Parser (단락·표 추출)
        ↓
   [3] Evidence Fragment Generator (의미 단위 증거 생성)
        ├── evidence_list      → Neo4j (증거 노드 + 관계)
        └── records            → Qdrant (벡터 임베딩)
        ↓
   [4] Knowledge Graph + Vector Index (검색 인덱스)
        ↓
   [5] Hybrid Retrieval (그래프 + 벡터 + 정형 검색)
        ↓
   [6] Agent Workflow (LLM + Tool Calling)
        ↓
     JSON 답변 + 근거
```

**각 단계 설명:**

1. **Section Canonicalizer**: DART XML/HTML을 파싱하여 문서 계층 구조를 식별하고 canonical section으로 정규화
2. **Parser**: 섹션 내 단락, 표, 주석, 참조를 구분하여 추출
3. **Evidence Fragment**: 의미 단위(TEXT, KV_TABLE, R_TABLE)로 증거 조각을 분리. R_TABLE은 행 단위로 보존
4. **Knowledge Graph (Neo4j)**: Disclosure → Section → Evidence → Event 관계 저장. Entity와 Correction 관계도 포함
5. **Vector Index (Qdrant)**: 텍스트 증거는 벡터 임베딩. R_TABLE 행은 메타데이터로 추적 가능하게 저장
6. **Retrieval**: 자연어 쿼리를 기업명 정규화, 범위 축소, 계산 식별 등으로 전처리한 뒤 하이브리드 검색
7. **Agent**: 검색 결과를 LLM으로 해석하여 자연어 답변 생성. 부족하거나 모호한 경우 명확히 고지

---

## Repository Directory Structure

```
Miraeasset_AI_Festival/
├── main.py                              # FastAPI 평가 API 엔드포인트
│
├── agent_graph/                         # 에이전트 워크플로우 (LangGraph)
│   ├── graph.py                         # 그래프 노드 정의 및 워크플로우
│   ├── state.py                         # 상태 정의 (Pydantic)
│   ├── tools.py                         # 에이전트 도구 (검색, 계산, 종료)
│   ├── llm.py                           # LLM 설정 및 Tool binding
│   ├── utils.py                         # 질의 분석, 검색 실행, 답변 생성
│   ├── calculation.py                   # 표 데이터 집계 및 계산
│   ├── retrieval_result_parser.py       # 검색 결과 파싱
│   ├── system_prompts.py                # LLM 시스템 프롬프트
│   └── qdrant_query_schema.yaml         # Qdrant 필터 스키마
│
├── scripts/                             # 데이터 파이프라인 (멀티프로세싱)
│   ├── build_canonical_sections.py      # Section canonicalization
│   ├── build_evidence_fragments.py      # Evidence fragment 생성
│   ├── validate_evidence_fragments.py   # Fragment 검증
│   ├── validate_converter_pipeline.py   # 파이프라인 회귀 테스트
│   └── approve_converter_golden.py      # 테스트 케이스 승인
│
├── converters/                          # XML/HTML 파싱 및 구조화
│   ├── section_canonicalizer/           # Section 정규화
│   ├── paragraph_parser/                # 단락 추출 및 정규화
│   ├── table_parser/                    # 표 파싱 (KV_TABLE, R_TABLE, LAYOUT_TABLE)
│   ├── table_context_resolver/          # 표 제목·단위·주석 연결
│   ├── evidence_builder/                # Evidence fragment 생성 및 검증
│   ├── event_extractor/                 # 이벤트 추출 및 분류
│   ├── correction_extractor/            # 정정 정보 추출
│   ├── regression_validation/           # 고정 공시 회귀 검증
│   └── common/                          # 공통 데이터 모델 및 유틸
│
├── knowledge_graph/                     # Neo4j 그래프 구축 및 쿼리
│   ├── insertUniverse.py                # Company 노드 삽입 (universe.csv)
│   ├── insertDSE.py                     # Disclosure/Section/Evidence 삽입
│   ├── insert_events.py                 # Event 노드 및 관계 삽입
│   ├── insert_correction_relations.py   # Correction 관계 삽입
│   ├── neo4j_schema.yaml                # Neo4j 스키마 정의
│   └── metric_definitions.py            # 금융 지표 정의
│
├── vector_db/                           # Qdrant 벡터 DB 구축 및 쿼리
│   ├── insert_points.py                 # 벡터 포인트 삽입
│   ├── point_builder.py                 # 포인트 구성 및 메타데이터
│   ├── text2vector.py                   # 텍스트 벡터 임베딩
│   ├── bgem3_token_counter.py           # 토큰 계산
│   ├── gpu_thermal_guard.py             # GPU 온도 관리
│   ├── r_table_embedding.yaml           # R_TABLE 임베딩 전략
│   ├── r_table_column_profiler.py       # 컬럼 프로파일링
│   └── r_table_strategy_selector.py     # 컬럼 선택 전략
│
├── data/                                # 파이프라인 입출력
│   ├── raw/                             # DART 원본 XML/HTML
│   ├── canonical_section/               # 정규화된 섹션
│   ├── evidence_fragment/               # 생성된 증거 조각
│   ├── correction/                      # 추출된 정정 정보
│   └── model_cache/                     # 임베딩 모델 캐시
│
├── DOCS/                                # 공시 문서 및 메타데이터
│   ├── universe.xlsx                    # 회사 목록 (70개사)
│   └── 기타 관련 문서
│
├── .env.example                         # 환경변수 템플릿
├── pyproject.toml                       # 프로젝트 설정 (uv)
└── uv.lock                              # 의존성 Lock 파일
```

**핵심 모듈:**

- **agent_graph**: LangGraph 기반 멀티 에이전트 워크플로우. 질의 분석 → 검색 계획 → 하이브리드 검색 → 답변 생성
- **scripts**: 완전 병렬화된 데이터 파이프라인. 원본 공시 수백 개를 분 단위에 처리
- **converters**: 도메인 특화 XML/HTML 파서. 정형/반정형 데이터 구조를 보존
- **knowledge_graph**: Neo4j로 관계 기반 검색 인덱스 구축
- **vector_db**: Qdrant로 의미 기반 검색 인덱스 구축. GPU 임베딩 및 온도 관리 포함

---

## Key Achievements & Features

### ✅ Data Pipeline (완성)

- DART XML/HTML 보수적 파싱 및 구조 복구
- Section 자동 정규화 (장 → 절 → 항 계층 추출)
- 단락·표·주석 의미 단위 분리
- 표 행 단위 레코드 분리 및 추적 가능 저장
- 시맨틱 기반 텍스트 세그먼테이션
- 고정 공시 회귀 테스트 스위트 (정확도 검증)

**파이프라인 복잡도:**
- 입력: 공시 XML/HTML 문서
- 출력: 공시 → 섹션 → 증거 조각 계층 구조 (JSON)
- 멀티프로세싱으로 수백 개 공시를 분 단위에 처리

### ✅ Knowledge Graph (완성)

- **Node Types**: Disclosure (공시), Section (섹션), Evidence (증거), Entity (기업/개인), Event (사건)
- **Relations**: HAS_SECTION, HAS_EVIDENCE, SUPPORTED_BY, Correction (정정), Related (연관)
- Neo4j 자동 적재 스크립트 (제약 조건, 인덱스 포함)
- 이벤트 추출 및 분류 (사건 속성 메타데이터)

### ✅ Vector Search Index (완성)

- 텍스트 증거 벡터 임베딩 (Hugging Face BGE-M3)
- R_TABLE 행 메타데이터 보존 (테이블 ID, 행 범위, 컬럼명)
- GPU 자동 온도 관리 (82°C 일시정지, 72°C 재개)
- Qdrant 스키마 정의 및 자동 적재

### ✅ Agent Workflow (완성)

**노드별 기능:**

1. **Question Analyzer**: 자연어 질의 분석 → 의도, 범위, 기업 식별
2. **Scope Narrower**: 기업 해석 → 단일 기업 또는 다중 기업으로 범위 축소
3. **Retriever**: 검색 도구 선택 및 실행
   - `retrieve_search`: 벡터 + 그래프 + 키워드 하이브리드 검색
   - `retrieve_correction_history`: 정정 정보 검색
   - `calculate_table_statistic`: 표 데이터 집계 (합계, 평균, 최대값 등)
   - `combine_numeric_results`: 계산 결과 병합
4. **Answer Generator**: 검색 결과를 LLM으로 해석 → 자연어 + 인용 생성
5. **Answer Validator**: 생성 답변 검증 (구간, 값 범위, 논리 일관성)
6. **Fallback**: 정보 부족 또는 모호한 경우 명확한 고지 또는 역질문

### ✅ Evaluation API (완성)

- FastAPI 기반 `GET /answer` 엔드포인트
- 평가자 스펙 완전 준수 (question_id, question, retrieved_context, think_trace, answer)
- 상세 think_trace: 질의 분석, 검색 이력, 계산 이력, 답변 검증 상태
- 외부 서비스 장애 시 고정 오류 응답 유지

---

## Design Principles

- **보존**: 원문 값을 임의로 보정하지 않습니다. 데이터는 DART 원본 그대로 저장됩니다.
- **정규화**: 중복 정보는 저장하지 않고 ID와 관계로만 연결합니다.
- **결정론적**: 구조화 가능한 데이터(표 행, 기업명, 섹션 경로)는 규칙 기반으로 처리합니다.
- **추적성**: 모든 답변은 공시 문서 → 섹션 → 증거 조각까지 역추적할 수 있습니다.

---

## Getting Started

### Prerequisites

- **Python**: 3.14 이상
- **uv**: 패키지 관리자 (https://docs.astral.sh/uv/getting-started/)
- **Neo4j**: 실행 중인 인스턴스 (localhost:7687 또는 원격)
- **Qdrant**: 실행 중인 벡터 DB 인스턴스 (localhost:6333 또는 원격)
- **CLOVA Studio API**: Naver LLM (API 키 필요)

### Environment Variables

`.env` 파일을 프로젝트 루트에 생성하고 다음 값을 설정합니다:

```bash
# LLM (Naver CLOVA Studio)
CLOVASTUDIO_API_KEY=<your-clova-api-key>
CLOVAX_MODEL_NAME=HCX-007

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>

# Qdrant Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedding Model
MODEL_NAME=BAAI/bge-m3
VECTOR_DIMENSION=1024
EMBEDDING_DEVICE=auto
EMBEDDING_MODEL_CACHE_DIR=./data/model_cache
EMBEDDING_API_URL=http://localhost:8001/embed  # 선택: 로컬 임베딩 서버 URL
```

### Quick Start Commands

모든 명령은 프로젝트 루트에서 실행합니다.

#### 1. 환경 확인 및 의존성 설치

```powershell
uv sync
```

#### 2. 데이터 파이프라인 실행

**2-1. Section Canonicalization**
```powershell
uv run python -m scripts.build_canonical_sections `
  --force `
  --workers 4 `
  --progress-every 100
```

입력: `data/raw/` (DART XML/HTML)  
출력: `data/canonical_section/` (정규화된 섹션)

**2-2. Evidence Fragment 생성**
```powershell
uv run python -m scripts.build_evidence_fragments `
  --workers 4 `
  --progress-every 100
```

입력: `data/canonical_section/`  
출력: `data/evidence_fragment/` (의미 단위 증거)

**2-3. Fragment 검증**
```powershell
uv run python -m scripts.validate_evidence_fragments
```

**2-4. 파이프라인 회귀 테스트** (선택)
```powershell
uv run python -m scripts.validate_converter_pipeline `
  --profile quick `
  --semantic-only
```

#### 3. 이벤트 추출

```powershell
uv run python -m converters.event_extractor.event_pipeline
```

출력: `data/event/manifest.jsonl` (추출된 이벤트)

#### 4. Knowledge Graph (Neo4j) 적재

**4-1. Company 노드 삽입** (데이터 초기화 필요 시)
```powershell
uv run python -m knowledge_graph.insertUniverse
```

**4-2. Disclosure/Section/Evidence 삽입**
```powershell
uv run python -m knowledge_graph.insertDSE `
  --limit 100 `
  --random-seed 42
```

`--limit`: 처리할 공시 개수  
`--random-seed`: 재현성을 위한 시드값

**4-3. Event 노드 및 관계 삽입**
```powershell
uv run python -m knowledge_graph.insert_events `
  --input data/event/manifest.jsonl
```

**4-4. Correction 관계 삽입**
```powershell
uv run python knowledge_graph/insert_correction_relations.py `
  --input data/correction/manifest.jsonl
```

#### 5. Vector Index (Qdrant) 적재

```powershell
uv run python -m vector_db.insert_points `
  --limit 100 `
  --random-seed 42 `
  --batch-size 128 `
  --embedding-buffer-size 512
```

`--limit`: 처리할 공시 개수  
`--batch-size`: 배치당 포인트 수  
`--embedding-buffer-size`: 임베딩 버퍼 크기

GPU 임베딩 시 자동으로 온도를 관리합니다 (82°C 일시정지, 72°C 재개).

#### 6. API 서버 실행

```powershell
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

API는 http://localhost:8000에서 실행됩니다.

#### 7. API 호출 테스트

```powershell
curl.exe --get "http://localhost:8000/answer" `
  --data-urlencode "question_id=Q-001" `
  --data-urlencode "question=삼성전자의 설비 투자를 알려줘"
```

응답 예시:
```json
{
  "question_id": "Q-001",
  "question": "삼성전자의 설비 투자를 알려줘",
  "retrieved_context": "{\"citations\": [...], \"results\": [...]}",
  "think_trace": "{\"query_text\": \"...\", \"question_analysis\": {...}, ...}",
  "answer": "삼성전자의 2024년 설비 투자는 약 XX조 원입니다. [공시명 (공시 ID) > 섹션명]"
}
```

---

## Execution Timeline (Typical)

단일 머신에서 70개사, 약 200개 공시 기준:

| 단계 | 시간 | 설명 |
|------|------|------|
| Section Canonicalization | 5-10분 | 멀티프로세싱, 정규화 |
| Evidence Fragment 생성 | 10-15분 | 의미 분리, 계층 구조 |
| Event 추출 | 5-10분 | 이벤트 분류 |
| Neo4j 적재 | 5-10분 | 노드·관계 삽입 |
| Vector 임베딩 | 30-60분 | GPU 의존 (온도 관리) |
| Qdrant 적재 | 5-10분 | 벡터 포인트 삽입 |
| **총 소요 시간** | **60-105분** | 병렬 실행 가능 (적재 제외) |

API 서버는 설정 후 즉시 응답 가능합니다. 평가 시간은 문제당 1-5초입니다.

---

## Testing & Validation

```powershell
# 파이프라인 회귀 테스트
uv run python -m scripts.validate_converter_pipeline --profile quick

# Evidence 검증
uv run python -m scripts.validate_evidence_fragments

# 고정 공시 정확도 검증
uv run python scripts/approve_converter_golden.py --check-only
```

---

## API Response Contract

평가자 스펙 완전 준수. 모든 필드는 JSON 문자열입니다.

### Success Response (HTTP 200)

```json
{
  "question_id": "Q-001",
  "question": "삼성전자의 매출을 알려줘",
  "retrieved_context": "JSON string: {\"citations\": [...], \"results\": [...]}",
  "think_trace": "JSON string: {\"query_text\": \"...\", \"question_analysis\": {...}, ...}",
  "answer": "자연어 답변 + 인용\n\n[공시명 (공시 ID) > 섹션명]\n[다른 공시 (공시 ID) > 다른 섹션]"
}
```

### Error Response (HTTP 200, 안정성 유지)

외부 서비스 장애 시에도 평가 API 계약 유지:

```json
{
  "question_id": "Q-001",
  "question": "삼성전자의 매출을 알려줘",
  "retrieved_context": "{\"citations\": [], \"results\": []}",
  "think_trace": "{\"query_text\": \"...\", \"warnings\": {...}, ...}",
  "answer": "확인할 수 없습니다. 요청 처리 중 외부 서비스 또는 내부 처리 오류가 발생했습니다."
}
```

---

## Key Implementation Notes

### Evidence Fragment Schema

```json
{
  "schema_version": "evidence-fragment.v2",
  "section_id": "section:20250101000001:src0:s12",
  "evidence_list": [
    {
      "evidence_id": "ev:...",
      "type": "TEXT|KV_TABLE|R_TABLE|LAYOUT_TABLE",
      "content": "원문 또는 메타데이터",
      "heading_path": ["장", "절", "항"],
      "captions": ["설명 문구"]
    }
  ],
  "records": [
    { "table_id": "...", "row_index": 0, "columns": {...} }
  ]
}
```

### Neo4j Schema

```cypher
(Disclosure)-[:HAS_SECTION]->(Section)
(Section)-[:HAS_SECTION]->(Section)
(Section)-[:HAS_EVIDENCE]->(Evidence)
(Event)-[:SUPPORTED_BY]->(Evidence)
(Company)-[:MADE]->(Disclosure)
(Event)-[:INVOLVES]->(Entity)
```

### Qdrant Metadata

```json
{
  "canonical": {
    "disclosure_id": "...",
    "section_id": "...",
    "evidence_id": "...",
    "records": [...]  // R_TABLE 행 메타데이터
  },
  "hierarchy": {
    "document_title": "...",
    "section_path": ["장", "절"]
  }
}
```

---

## Troubleshooting

### Neo4j 연결 실패
```
확인: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
Neo4j 서버 상태: neo4j status
```

### Qdrant 연결 실패
```
확인: QDRANT_HOST, QDRANT_PORT
Qdrant 상태: curl http://localhost:6333/health
```

### CLOVA API 오류
```
확인: CLOVASTUDIO_API_KEY, CLOVAX_MODEL_NAME
Naver AI API 콘솔: https://console.naver.com/
```

### GPU 메모리 부족
```
EMBEDDING_DEVICE=cpu 로 설정하여 CPU 임베딩으로 전환
또는 --batch-size를 줄임
```

---

## References

- DART 공시 XML/HTML: https://dart.fss.or.kr/
- Neo4j 문서: https://neo4j.com/docs/
- Qdrant 문서: https://qdrant.tech/documentation/
- LangGraph: https://langchain-ai.github.io/langgraph/
- BGE-M3 모델: https://huggingface.co/BAAI/bge-m3

---

## License & Attribution

Team Sortie / Miraeasset AI Festival 2024

최종 수정: 2026년 9월
