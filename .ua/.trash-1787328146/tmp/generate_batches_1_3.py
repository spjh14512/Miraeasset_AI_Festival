import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UA = ROOT / ".ua"


FILE_SUMMARIES = {
    "knowledge_graph/insert_DSE.py": "Canonical Section과 Evidence Fragment 산출물을 Neo4j용 Disclosure-Section-Evidence 그래프 row로 변환하고 batch 삽입하는 CLI입니다.",
    "vector_db/bgem3_token_counter.py": "BGE-M3 tokenizer를 지연 로딩하여 contextual text의 token 수를 계산합니다.",
    "vector_db/contextual_text_builders/__init__.py": "TEXT, KV-table, R-table evidence용 contextual text builder의 공개 API를 모아 재노출합니다.",
    "vector_db/contextual_text_builders/kv_table_context_builder.py": "KV-table의 문서·section·table 문맥과 key path/value를 결합해 embedding용 contextual text를 만듭니다.",
    "vector_db/contextual_text_builders/r_table_descriptor_builder.py": "대형 R-table의 header, dimension value, 문맥을 압축한 descriptor embedding text를 생성합니다.",
    "vector_db/contextual_text_builders/r_table_record_context_builder.py": "R-table 전체 또는 개별 record를 column header와 table 문맥에 연결한 embedding text로 렌더링합니다.",
    "vector_db/contextual_text_builders/r_table_row_group_builder.py": "token 제한을 지키면서 R-table record를 누락·중복 없이 greedy row group으로 분할하고 렌더링합니다.",
    "vector_db/contextual_text_builders/text_context_builder.py": "일반 TEXT evidence에 disclosure, section, heading 문맥을 덧붙인 embedding text를 생성합니다.",
    "vector_db/insert_points.py": "Evidence Fragment를 읽어 PointInput을 만들고 local embedding을 계산한 뒤 Qdrant collection과 payload index에 batch upsert하는 CLI입니다.",
    "vector_db/point_builder.py": "TEXT, KV-table, R-table evidence를 strategy별 Qdrant point payload와 deterministic ID로 조립합니다.",
    "vector_db/r_table_column_profiler.py": "R-table column 값을 통계적으로 profile하여 dimension, measure, identifier, date, long-text role을 판정합니다.",
    "vector_db/r_table_strategy_selector.py": "R-table의 token 크기와 cell 특성을 바탕으로 whole-table, descriptor, row-group embedding strategy를 선택합니다.",
    "vector_db/text2vector.py": "CUDA 기반 BGE-M3 model을 지연 로딩하고 batch text를 고정 차원 dense vector로 변환합니다.",
    "converters/common/document_loader.py": "DART XML/HTML source를 decode하고 tolerant recovery를 적용하여 추적 가능한 LoadedDocument로 정규화합니다.",
    "converters/common/source_models.py": "document syntax, source reference, evidence context를 표현하는 공통 Enum과 dataclass model을 정의합니다.",
    "converters/common/source_resolver.py": "LoadedDocument element를 stable path로 index하고 SourceRef에서 원본 element와 table 위치를 역참조합니다.",
    "converters/correction_extractor/correction_extractor.py": "XML과 거래소 HTML에서 정정공시 metadata, 대상 접수번호, 정정 사유 및 block reference를 복원·추출합니다.",
    "converters/correction_extractor/correction_models.py": "정정공시 추출 상태, issue, metadata, block reference 결과를 직렬화 가능한 model로 정의합니다.",
    "converters/evidence_builder/fragment_assembler.py": "canonical section의 paragraph와 table을 semantic evidence로 조립하고 외부 R-table record를 연결한 Evidence Fragment를 생성합니다.",
    "converters/evidence_builder/semantic_text_segmenter.py": "paragraph 내부 heading, marker, caption 경계를 감지해 원문 reference를 보존한 semantic text segment로 분할합니다.",
    "converters/paragraph_parser/paragraph_models.py": "paragraph parse 상태, marker, fragment, canonical paragraph collection을 나타내는 model을 정의합니다.",
    "converters/paragraph_parser/paragraph_parser.py": "XML/HTML의 P, SPAN, BR, anchor 구조를 순서와 source attribute를 보존한 canonical paragraph로 변환합니다.",
    "converters/table_context_resolver/resolver_models.py": "table evidence bundle과 table context resolution 결과를 표현하는 dataclass model을 정의합니다.",
    "converters/table_context_resolver/table_context_resolver.py": "주변 paragraph와 layout table에서 title, unit, caption, note를 찾아 각 canonical table에 안전하게 연결합니다.",
    "converters/table_parser/table_models.py": "table type, cell/row role, logical grid, canonical table/group을 표현하는 parser domain model을 정의합니다.",
    "converters/table_parser/table_parser.py": "병합 cell logical grid를 만들고 DART XML/HTML table을 KV, record, layout, unknown canonical table로 분류·파싱합니다.",
    "converters/evidence_builder/fragment_models.py": "Evidence Fragment의 evidence type, storage mode, external table record 및 직렬화 형식을 정의합니다.",
    "converters/evidence_builder/fragment_pipeline.py": "canonical section manifest를 graph section으로 투영하고 disclosure별 Evidence Fragment를 원자적으로 생성·재사용하는 pipeline입니다.",
    "converters/evidence_builder/fragment_validator.py": "Evidence Fragment schema, reference 정합성, external record 폭, manifest hash와 금지 필드를 검증합니다.",
    "converters/section_canonicalizer/section_canonicalizer.py": "명시적 section과 암시적 heading을 탐지해 모든 source block을 단일 계층 section에 귀속시키는 canonicalizer입니다.",
    "converters/section_canonicalizer/section_models.py": "section boundary, block reference, issue, canonical section collection의 직렬화 model을 정의합니다.",
    "scripts/build_canonical_sections.py": "source manifest를 읽어 canonical section JSON과 manifest를 병렬 생성하고 hash 기반 재사용·실패 추적을 수행하는 CLI입니다.",
    "scripts/build_evidence_fragments.py": "Evidence Fragment build pipeline의 main entry point를 실행하는 얇은 CLI wrapper입니다.",
    "scripts/validate_evidence_fragments.py": "Evidence Fragment validator의 main entry point를 실행하는 얇은 CLI wrapper입니다.",
}

TEST_SUMMARIES = {
    "test_insert_points.py": "Qdrant schema 로딩, collection/index 생성, disclosure 선택 재사용, embedding deduplication과 batch upsert 계약을 검증합니다.",
    "test_kv_context_builder.py": "KV-table contextual text가 table 문맥과 key path를 정확히 보존하고 invalid evidence를 거부하는지 검증합니다.",
    "test_point_builder.py": "evidence 유형과 R-table strategy별 Qdrant payload, point ID, embedding 조립 경계를 폭넓게 검증합니다.",
    "test_r_table_column_profiler.py": "R-table column role profiling, threshold, value cardinality와 multi-level header 직렬화를 검증합니다.",
    "test_r_table_descriptor_builder.py": "R-table descriptor가 context, header, dimension value를 선택적으로 구성하는지 검증합니다.",
    "test_r_table_record_context_builder.py": "R-table record/whole-table contextual text 렌더링과 입력 정합성 검사를 검증합니다.",
    "test_r_table_row_group_builder.py": "token 제한 기반 greedy row grouping이 누락·중복 없이 oversized record와 오류를 처리하는지 검증합니다.",
    "test_r_table_strategy_selector.py": "table 크기와 content 특성에 따른 embedding strategy 선택 및 설정 threshold를 검증합니다.",
    "test_text2vector.py": "BGE-M3 local model 설정, CUDA/FP16 사용, batch API와 embedding dimension 오류 처리를 검증합니다.",
    "test_text_contextual_builder.py": "TEXT evidence contextual text의 document/section/heading 결합과 중복 제거를 검증합니다.",
    "test_correction_extractor.py": "XML 및 거래소 HTML 정정공시 metadata 추출, recovery, false-positive 방지를 검증합니다.",
    "test_document_loader.py": "정상 XML, bare ampersand, malformed attribute와 loose HTML의 tolerant loading 상태를 검증합니다.",
    "test_paragraph_parser.py": "inline span, anchor, BR, exclusion, recovery를 보존하는 canonical paragraph parsing을 검증합니다.",
    "test_semantic_text_segmenter.py": "bold heading, marker chain, BR, caption 경계가 원문을 훼손하지 않고 분할되는지 검증합니다.",
    "test_table_context_resolver.py": "table 주변 title/unit/note 결합과 ambiguous narrative 및 table boundary 차단을 검증합니다.",
    "test_table_parser.py": "merged cell, KV/record/layout/unknown table과 DART XML/HTML recovery를 포괄적으로 검증합니다.",
    "test_build_canonical_sections.py": "canonical section builder의 multi-source flattening, hash 재사용, 실패 manifest와 graph projection을 검증합니다.",
    "test_evidence_fragments.py": "Evidence Fragment assembly, table record 외부화, context deduplication, validator와 end-to-end pipeline을 검증합니다.",
    "test_section_canonicalizer.py": "명시적·암시적 section hierarchy, exclusion, orphan block, recovery 상태 추적을 검증합니다.",
}


def file_summary(path: str) -> str:
    if path in FILE_SUMMARIES:
        return FILE_SUMMARIES[path]
    name = Path(path).name
    if name in TEST_SUMMARIES:
        return TEST_SUMMARIES[name]
    return f"{name}의 핵심 동작과 project 내부 연동 계약을 구현합니다."


def tags_for_file(path: str) -> list[str]:
    if path.startswith("tests/"):
        return ["test", "regression", "validation", "pytest"]
    if path.startswith("scripts/"):
        return ["entry-point", "cli", "data-pipeline", "automation"]
    if "models.py" in path:
        return ["data-model", "schema-definition", "serialization", "type-definition"]
    if "parser" in path or "loader" in path or "canonicalizer" in path:
        return ["parser", "normalization", "dart-xml", "data-pipeline"]
    if "validator" in path:
        return ["validation", "schema", "data-integrity", "data-pipeline"]
    if "context" in path:
        return ["context-builder", "embedding", "serialization", "data-pipeline"]
    if path.startswith("vector_db/"):
        return ["vector-db", "embedding", "qdrant", "retrieval"]
    if path.startswith("knowledge_graph/"):
        return ["knowledge-graph", "neo4j", "data-pipeline", "cli"]
    return ["converter", "data-pipeline", "normalization", "dart-disclosure"]


def complexity(nonempty: int) -> str:
    return "simple" if nonempty < 50 else "moderate" if nonempty <= 200 else "complex"


def words(name: str) -> str:
    return " ".join(w for w in re.sub(r"^_+", "", name).split("_") if w) or name


def symbol_summary(name: str, kind: str, path: str) -> str:
    label = words(name)
    if name.startswith("test_"):
        return f"{label} 시나리오의 기대 동작과 regression 조건을 검증합니다."
    if name.startswith("_") and kind == "function":
        return f"{label} 처리를 담당하며 상위 pipeline에서 재사용되는 내부 helper입니다."
    if name in {"main", "run"}:
        return f"{Path(path).name} workflow를 초기화하고 전체 실행 순서를 조정하는 entry point입니다."
    if name.startswith("build_") or name.startswith("assemble_"):
        return f"검증된 입력으로 {label} 결과를 조립하여 downstream 단계에 전달합니다."
    if name.startswith("load_"):
        return f"{label} 설정 또는 source를 읽고 유효한 runtime 객체로 변환합니다."
    if name.startswith("parse_") or name.startswith("extract_"):
        return f"source 구조를 분석하여 {label} canonical 결과와 trace 정보를 추출합니다."
    if name.startswith("validate_"):
        return f"{label} 구조와 cross-reference 정합성을 검사하고 오류를 보고합니다."
    if name.startswith("select_") or name == "select":
        return f"입력 feature와 설정 기준을 평가해 {label} 대상을 결정합니다."
    if name.startswith("to_dict"):
        return f"{label} model을 안정적인 JSON 직렬화 dict로 변환합니다."
    if kind == "class":
        if name.endswith("Status") or name.endswith("Type") or name.endswith("Role") or name.endswith("Kind") or name.endswith("Syntax") or name.endswith("Mode") or name.endswith("Strategy"):
            return f"{label} domain 값의 허용 범위를 명시하는 Enum입니다."
        if name.endswith("Error") or name.endswith("Issue"):
            return f"{label} 실패 원인과 진단 context를 표현하는 error model입니다."
        return f"{label} domain data와 관련 동작을 캡슐화하는 model입니다."
    return f"{label} 처리 규칙을 구현하여 {Path(path).stem} workflow의 한 단계를 수행합니다."


def symbol_tags(name: str, kind: str, path: str) -> list[str]:
    if name.startswith("test_"):
        return ["test", "regression", "validation"]
    if kind == "class":
        return ["data-model", "type-definition", "serialization"]
    if name.startswith("_"):
        return ["internal-helper", "data-pipeline", "normalization"]
    if name in {"main", "run"}:
        return ["entry-point", "orchestration", "cli"]
    if "validate" in name:
        return ["validation", "data-integrity", "schema"]
    if "build" in name or "assemble" in name:
        return ["factory", "data-pipeline", "serialization"]
    if "parse" in name or "extract" in name:
        return ["parser", "normalization", "dart-xml"]
    return ["service", "data-pipeline", "utility"]


def node_for_file(result: dict) -> dict:
    path = result["path"]
    node = {
        "id": f"file:{path}",
        "type": "file",
        "name": Path(path).name,
        "filePath": path,
        "summary": file_summary(path),
        "tags": tags_for_file(path),
        "complexity": complexity(result.get("nonEmptyLines", 0)),
    }
    if path.endswith("__init__.py"):
        node["languageNotes"] = "Python package barrel로 public builder API를 명시적으로 re-export합니다."
    elif result.get("nonEmptyLines", 0) > 400:
        node["languageNotes"] = "복잡한 DART source 변형과 traceability를 다루는 다단계 Python pipeline입니다."
    return node


def build_batch(index: int) -> tuple[list[dict], list[dict], list[str], dict]:
    batches = json.loads((UA / "intermediate/batches.json").read_text(encoding="utf-8"))
    batch = next(item for item in batches["batches"] if item["batchIndex"] == index)
    extracted = json.loads((UA / f"tmp/ua-file-extract-results-{index}.json").read_text(encoding="utf-8"))
    by_path = {item["path"]: item for item in extracted["results"]}
    nodes: list[dict] = []
    edges: list[dict] = []
    for file_info in batch["files"]:
        path = file_info["path"]
        result = by_path[path]
        nodes.append(node_for_file(result))
        exported = {item["name"] for item in result.get("exports", [])}
        for kind, items in (("function", result.get("functions", [])), ("class", result.get("classes", []))):
            for item in items:
                line_count = item["endLine"] - item["startLine"] + 1
                significant = item["name"] in exported or line_count >= (10 if kind == "function" else 20) or (kind == "class" and len(item.get("methods", [])) >= 2)
                if not significant:
                    continue
                node_id = f"{kind}:{path}:{item['name']}"
                nodes.append({
                    "id": node_id,
                    "type": kind,
                    "name": item["name"],
                    "filePath": path,
                    "lineRange": [item["startLine"], item["endLine"]],
                    "summary": symbol_summary(item["name"], kind, path),
                    "tags": symbol_tags(item["name"], kind, path),
                    "complexity": complexity(line_count),
                })
                edges.append({"source": f"file:{path}", "target": node_id, "type": "contains", "direction": "forward", "weight": 1.0})
                if item["name"] in exported:
                    edges.append({"source": f"file:{path}", "target": node_id, "type": "exports", "direction": "forward", "weight": 0.8})
        for target in batch["batchImportData"].get(path, []):
            edges.append({"source": f"file:{path}", "target": f"file:{target}", "type": "imports", "direction": "forward", "weight": 0.7})
    return nodes, edges, extracted.get("filesSkipped", []), batch


def write_parts(index: int, nodes: list[dict], edges: list[dict], batch: dict) -> list[Path]:
    parts = math.ceil(max(len(nodes) / 60, len(edges) / 120))
    parts = max(parts, 1)
    paths = sorted(item["path"] for item in batch["files"])
    group_size = math.ceil(len(paths) / parts)
    outputs = []
    if parts == 1:
        out = UA / f"intermediate/batch-{index}.json"
        out.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return [out]
    node_by_path = {}
    for node in nodes:
        node_by_path.setdefault(node["filePath"], []).append(node)
    for part_no in range(1, parts + 1):
        group = paths[(part_no - 1) * group_size : part_no * group_size]
        part_nodes = [node for path in group for node in node_by_path.get(path, [])]
        sources = {node["id"] for node in part_nodes}
        part_edges = [edge for edge in edges if edge["source"] in sources]
        out = UA / f"intermediate/batch-{index}-part-{part_no}.json"
        out.write_text(json.dumps({"nodes": part_nodes, "edges": part_edges}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs.append(out)
    return outputs


def validate(index: int, outputs: list[Path], batch: dict) -> None:
    allowed_files = set(batch["neighborMap"]) | {p for values in batch["batchImportData"].values() for p in values}
    parsed = []
    for out in outputs:
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list)
        parsed.append((out, data))
    for out, data in parsed:
        local_ids = {node["id"] for node in data["nodes"]}
        for edge in data["edges"]:
            for endpoint in (edge["source"], edge["target"]):
                if endpoint in local_ids:
                    continue
                if endpoint.startswith("file:") and endpoint[5:] in allowed_files:
                    continue
                raise AssertionError(f"{out.name}: 허용되지 않은 edge endpoint {endpoint}")
    import_expected = sum(len(batch["batchImportData"].get(item["path"], [])) for item in batch["files"])
    import_actual = sum(1 for _, data in parsed for edge in data["edges"] if edge["type"] == "imports")
    assert import_actual == import_expected, (index, import_expected, import_actual)


for batch_index in (1, 2, 3):
    batch_nodes, batch_edges, skipped, batch_data = build_batch(batch_index)
    written = write_parts(batch_index, batch_nodes, batch_edges, batch_data)
    validate(batch_index, written, batch_data)
    print(json.dumps({"batch": batch_index, "parts": [p.name for p in written], "nodes": len(batch_nodes), "edges": len(batch_edges), "skipped": skipped}, ensure_ascii=False))
