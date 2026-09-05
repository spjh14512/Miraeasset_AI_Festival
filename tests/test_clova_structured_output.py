from __future__ import annotations

import json

from agent_graph.llm import (
    DEFAULT_CLOVA_MODEL,
    bind_structured_output,
    build_clova_json_schema,
)
from agent_graph.state import QuestionAnalyzerOutput
from agent_graph.utils import NarrowScopeAction


class _FakeLlm:
    def __init__(self):
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self


def test_default_model_is_hcx_007():
    assert DEFAULT_CLOVA_MODEL == "HCX-007"


def test_clova_schema_keeps_nested_contract_and_removes_unsupported_keywords():
    schema = build_clova_json_schema(QuestionAnalyzerOutput)
    serialized = json.dumps(schema)

    assert schema["title"] == "QuestionAnalyzerOutput"
    assert "question_analysis" in schema["properties"]
    assert "sub_questions" in schema["properties"]["question_analysis"]["properties"]
    for keyword in (
        "$defs",
        "$ref",
        "additionalProperties",
        "default",
        "minLength",
        '"type": "null"',
    ):
        assert keyword not in serialized


def test_structured_output_uses_json_schema_method():
    llm = _FakeLlm()

    assert bind_structured_output(llm, QuestionAnalyzerOutput) is llm
    assert llm.method == "json_schema"
    assert llm.schema["title"] == "QuestionAnalyzerOutput"


def test_narrow_scope_action_uses_clova_compatible_schema():
    schema = build_clova_json_schema(NarrowScopeAction)

    assert set(schema["properties"]) == {
        "action",
        "cypher",
        "parameters_json",
        "level",
        "corp_names",
        "corp_codes",
        "disclosure_ids",
        "section_ids",
        "reason",
    }
    assert schema["properties"]["section_ids"]["items"]["type"] == "string"
