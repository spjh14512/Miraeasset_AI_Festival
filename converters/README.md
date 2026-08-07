# Converters

`converters`는 원본 공시 데이터를 후속 처리에 사용할 수 있는 표준 형식으로 변환하는 도구를 모아두는 폴더입니다.

새 도구를 추가할 때는 이 문서에 도구의 목적, 입력·출력, 실행 명령을 함께 기록합니다.

## canonicalizer.py

DART XML 문서를 중첩된 canonical JSON 트리로 변환합니다.

현재 지원하는 문서 유형은 다음과 같습니다.

- `holding`: 주식 등의 대량보유상황보고서
- `major`: 주요사항보고서

주요 처리 내용:

- `COVER`, `SECTION-*`, `TABLE` 구조 및 원본 순서 보존
- `ROWSPAN`, `COLSPAN`을 고려한 표 문맥 복원
- `ACODE`, `AUNIT`, `AUNITVALUE` 보존
- 숫자, 비율, 날짜, enum 등 등록된 필드 타입 정규화
- `P`, `SPAN` 기반 장문 텍스트의 문단과 줄바꿈 보존
- major 문서의 표지 정보를 회사명, 대표이사, 본점소재지, 작성책임자 필드로 변환
- 원본 XML 오류를 가능한 범위에서 복구하고 `parse_status`에 기록

### 기본 실행

저장소 루트에서 실행합니다.

```powershell
uv run python converters/canonicalizer.py
```

기본 실행은 holding과 major를 모두 처리합니다.

```text
data/raw/holding/**/*.xml -> data/canonical/holding/**/*.json
data/raw/major/**/*.xml   -> data/canonical/major/**/*.json
```

원본 기준 상대 디렉터리 구조와 파일명은 유지되고 확장자만 `.json`으로 변경됩니다.

### 문서 유형별 실행

holding만 변환:

```powershell
uv run python converters/canonicalizer.py --category holding
```

major만 변환:

```powershell
uv run python converters/canonicalizer.py --category major
```

두 유형을 명시적으로 지정:

```powershell
uv run python converters/canonicalizer.py --category holding --category major
```

### 입력·출력 경로 지정

사용자 지정 경로는 하나의 `--category`와 함께 사용해야 합니다.

```powershell
uv run python converters/canonicalizer.py `
  --category major `
  --input-root path/to/raw/major `
  --output-root path/to/canonical/major
```

### Python API

단일 표 변환:

```python
from converters.canonicalizer import canonicalize_table

canonical_table = canonicalize_table(raw_table_fragment)
```

단일 문서 변환:

```python
from pathlib import Path

from converters.canonicalizer import canonicalize_document

canonical_document = canonicalize_document(
    Path("data/raw/major/company/report/report.xml"),
    document_type="major",
)
```

디렉터리 일괄 변환:

```python
from converters.canonicalizer import canonicalize_holding_tree

counts = canonicalize_holding_tree(
    "data/raw/major",
    "data/canonical/major",
    document_type="major",
    progress=True,
)
```

### 변환 상태

문서의 `parse_status`는 다음 중 하나입니다.

- `success`: 원본 XML을 정상적으로 파싱함
- `recovered`: 원본 XML의 비정상 문자 등을 보정한 뒤 파싱함
- `failed`: 변환하지 못했으며 오류 정보가 결과 JSON에 기록됨

하나의 파일에서 오류가 발생해도 나머지 파일 변환은 계속됩니다. 실패한 문서가 하나라도 있으면 CLI는 종료 코드 `1`을 반환합니다.

### 새 도구 문서화 형식

새 스크립트를 추가할 때는 아래 항목을 간략히 기록합니다.

```markdown
## tool_name.py

도구의 목적

- 입력: 입력 형식과 기본 경로
- 출력: 출력 형식과 기본 경로
- 실행: `uv run python converters/tool_name.py`
- 주요 옵션: 필요한 CLI 옵션
```
