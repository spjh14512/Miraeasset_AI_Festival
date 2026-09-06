import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import APIConnectionError, APIStatusError, BadRequestError, RateLimitError
from pydantic import BaseModel


load_dotenv()

MAX_LLM_RETRIES = 2
MAX_RATE_LIMIT_RETRIES = 2
RATE_LIMIT_RESET_BUFFER_SECONDS = 0.5
UNSUPPORTED_FUNCTION_ERROR_CODE = "40009"
DEFAULT_CLOVA_MODEL = "HCX-007"
QUESTION_ANALYZER_MAX_COMPLETION_TOKENS = 1024
NARROW_SCOPE_MAX_COMPLETION_TOKENS = 1024
RETRIEVER_MAX_TOKENS = 1024
CYPHER_BUILDER_MAX_COMPLETION_TOKENS = 768
QDRANT_QUERY_BUILDER_MAX_COMPLETION_TOKENS = 512
COMPACTOR_MAX_COMPLETION_TOKENS = 512
ANSWER_GENERATOR_MAX_COMPLETION_TOKENS = 2048
ANSWER_VALIDATOR_MAX_COMPLETION_TOKENS = 1024
DEFAULT_MAX_COMPLETION_TOKENS = 512
_SUPPORTED_JSON_SCHEMA_KEYS = {
    "type",
    "description",
    "format",
    "enum",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "required",
}


def _sanitize_clova_json_schema(value: Any) -> Any:
    """CLOVA Structured Outputs가 지원하는 JSON Schema keyword만 남깁니다."""

    if not isinstance(value, dict):
        return value

    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if key == "properties" and isinstance(item, dict):
            sanitized[key] = {
                name: _sanitize_clova_json_schema(property_schema)
                for name, property_schema in item.items()
            }
        elif key == "items":
            sanitized[key] = _sanitize_clova_json_schema(item)
        elif key == "anyOf" and isinstance(item, list):
            alternatives = [
                _sanitize_clova_json_schema(schema)
                for schema in item
                if not isinstance(schema, dict) or schema.get("type") != "null"
            ]
            if len(alternatives) == 1:
                sanitized.update(alternatives[0])
            elif alternatives:
                sanitized[key] = alternatives
        elif key in _SUPPORTED_JSON_SCHEMA_KEYS:
            sanitized[key] = item
    return sanitized


def build_clova_json_schema(output_model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic model을 CLOVA 호환 Structured Outputs schema로 변환합니다."""

    function = convert_to_openai_tool(output_model)["function"]
    parameters = _sanitize_clova_json_schema(function["parameters"])
    return {
        "title": function["name"],
        **parameters,
    }


def bind_structured_output(llm: Any, output_model: type[BaseModel]) -> Any:
    """HCX-007에 CLOVA 호환 JSON Schema structured output을 binding합니다."""

    return llm.with_structured_output(
        build_clova_json_schema(output_model),
        method="json_schema",
    )


def format_output_error(error: BaseException) -> str:
    """검증 오류에서 field, 원인과 짧은 invalid input만 추출합니다."""

    if hasattr(error, "errors"):
        try:
            details = error.errors(include_url=False)
        except TypeError:
            details = error.errors()
        messages = []
        for detail in details:
            location = ".".join(map(str, detail.get("loc", ()))) or "output"
            message = f"{location}: {detail.get('msg', '')}"
            invalid_input = detail.get("input")
            if isinstance(invalid_input, (str, int, float, bool)):
                rendered_input = repr(invalid_input)
                if len(rendered_input) <= 120:
                    message += f" (input={rendered_input})"
            messages.append(message)
        return "; ".join(messages)
    return str(error)


def build_output_retry_message(component: str, error: BaseException) -> str:
    """LLM 출력 검증 오류를 해당 LLM의 재생성 요청으로 변환합니다."""

    return (
        f"{component} 검증 실패: {format_output_error(error)}\n"
        "오류를 수정해 요구된 전체 출력을 다시 생성하세요."
    )


def _retry_after_seconds(error: BaseException) -> float | None:
    """API 응답의 Retry-After header를 초 단위로 해석합니다."""

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(
            0.0,
            (retry_at - datetime.now(timezone.utc)).total_seconds(),
        )


def _token_rate_limit_info(
    error: BaseException,
) -> tuple[int | None, float | None]:
    """CLOVA 응답 header에서 남은 token과 TPM reset 대기 시간을 읽습니다."""

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None, None

    remaining_value = headers.get("x-ratelimit-remaining-tokens")
    reset_value = headers.get("x-ratelimit-reset-tokens")

    try:
        remaining_tokens = (
            int(remaining_value) if remaining_value is not None else None
        )
    except (TypeError, ValueError):
        remaining_tokens = None

    if reset_value is None:
        return remaining_tokens, None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", str(reset_value))
    if match is None:
        return remaining_tokens, None
    return remaining_tokens, float(match.group(1))


def _is_unsupported_function_error(error: BadRequestError) -> bool:
    """CLOVA의 일시적인 Unsupported function 오류인지 확인합니다."""

    code = getattr(error, "code", None)
    if code is None:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            code = body.get("code")
            nested_error = body.get("error")
            if code is None and isinstance(nested_error, dict):
                code = nested_error.get("code")
    return str(code) == UNSUPPORTED_FUNCTION_ERROR_CODE


def invoke_with_rate_limit_retry(
    runnable: Any,
    input_value: Any,
    *,
    max_retries: int = MAX_RATE_LIMIT_RETRIES,
) -> Any:
    """429, 40009, 5xx 및 연결 오류를 최대 횟수만큼 재호출합니다.

    입력 예시:
        invoke_with_rate_limit_retry(structured_llm, messages)

    출력 예시:
        runnable.invoke(messages)의 성공 응답
    """

    if max_retries < 0:
        raise ValueError("max_retries는 0 이상이어야 합니다.")
    for attempt in range(max_retries + 1):
        try:
            return runnable.invoke(input_value)
        except RateLimitError as error:
            if attempt == max_retries:
                raise
            remaining_tokens, reset_seconds = _token_rate_limit_info(error)
            if reset_seconds is not None:
                delay = reset_seconds + RATE_LIMIT_RESET_BUFFER_SECONDS
            else:
                retry_after = _retry_after_seconds(error)
                delay = retry_after if retry_after is not None else 2 ** attempt
            remaining_text = (
                f", 남은 token: {remaining_tokens}"
                if remaining_tokens is not None
                else ""
            )
            print(
                "LLM rate limit으로 재시도합니다: "
                f"{delay:g}초 후 ({attempt + 1}/{max_retries})"
                f"{remaining_text}"
            )
            time.sleep(delay)
        except BadRequestError as error:
            if not _is_unsupported_function_error(error) or attempt == max_retries:
                raise
            retry_after = _retry_after_seconds(error)
            delay = retry_after if retry_after is not None else 2 ** attempt
            print(
                "LLM Unsupported function 오류로 재시도합니다: "
                f"{delay:g}초 후 ({attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
        except APIStatusError as error:
            if error.status_code < 500 or attempt == max_retries:
                raise
            retry_after = _retry_after_seconds(error)
            delay = retry_after if retry_after is not None else 2 ** attempt
            print(
                f"LLM 서버 오류({error.status_code})로 재시도합니다: "
                f"{delay:g}초 후 ({attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
        except APIConnectionError as error:
            if attempt == max_retries:
                raise
            delay = 2 ** attempt
            print(
                f"LLM 연결 오류({type(error).__name__})로 재시도합니다: "
                f"{delay:g}초 후 ({attempt + 1}/{max_retries})"
            )
            time.sleep(delay)


@lru_cache(maxsize=None)
def get_llm(
    max_output_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    *,
    output_token_parameter: Literal[
        "max_completion_tokens",
        "max_tokens",
    ] = "max_completion_tokens",
) -> Any:
    """출력 토큰 상한과 API parameter 조합별 ChatClovaX를 재사용합니다."""

    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens는 양수여야 합니다.")

    api_key = os.getenv("CLOVASTUDIO_API_KEY")
    if not api_key:
        raise RuntimeError("CLOVASTUDIO_API_KEY 환경변수가 필요합니다.")

    # Import가 무거우므로 실제 LLM 호출이 필요한 시점까지 지연한다.
    from langchain_naver import ChatClovaX

    token_options = {output_token_parameter: max_output_tokens}
    return ChatClovaX(
        model=os.getenv("CLOVAX_MODEL_NAME") or DEFAULT_CLOVA_MODEL,
        api_key=api_key,
        **token_options,
        reasoning_effort="none",
        temperature=0.0,
        top_p=0.8,
        top_k=0,
        repetition_penalty=1.1,
        disabled_params={"parallel_tool_calls": None},
    )


__all__ = [
    "ANSWER_GENERATOR_MAX_COMPLETION_TOKENS",
    "ANSWER_VALIDATOR_MAX_COMPLETION_TOKENS",
    "COMPACTOR_MAX_COMPLETION_TOKENS",
    "CYPHER_BUILDER_MAX_COMPLETION_TOKENS",
    "DEFAULT_CLOVA_MODEL",
    "DEFAULT_MAX_COMPLETION_TOKENS",
    "MAX_LLM_RETRIES",
    "MAX_RATE_LIMIT_RETRIES",
    "NARROW_SCOPE_MAX_COMPLETION_TOKENS",
    "QDRANT_QUERY_BUILDER_MAX_COMPLETION_TOKENS",
    "QUESTION_ANALYZER_MAX_COMPLETION_TOKENS",
    "RATE_LIMIT_RESET_BUFFER_SECONDS",
    "RETRIEVER_MAX_TOKENS",
    "UNSUPPORTED_FUNCTION_ERROR_CODE",
    "bind_structured_output",
    "build_clova_json_schema",
    "build_output_retry_message",
    "get_llm",
    "invoke_with_rate_limit_retry",
]
