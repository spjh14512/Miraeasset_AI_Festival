import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from openai import RateLimitError


load_dotenv()

MAX_LLM_RETRIES = 2
MAX_RATE_LIMIT_RETRIES = 2


def build_output_retry_message(component: str, error: BaseException) -> str:
    """LLM 출력 검증 오류를 해당 LLM의 재생성 요청으로 변환합니다."""

    return (
        f"방금 생성한 {component} 출력이 application 검증을 통과하지 못했습니다.\n"
        f"오류: {type(error).__name__}: {error}\n"
        "오류 원인을 수정하여 전체 출력을 요구된 schema 또는 tool call 형식으로 "
        "다시 생성하세요. 오류가 난 출력을 그대로 반복하지 마세요."
    )


def _retry_after_seconds(error: RateLimitError) -> float | None:
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


def invoke_with_rate_limit_retry(
    runnable: Any,
    input_value: Any,
    *,
    max_retries: int = MAX_RATE_LIMIT_RETRIES,
) -> Any:
    """429 응답에 한해 Retry-After 또는 exponential backoff로 재호출합니다.

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
            retry_after = _retry_after_seconds(error)
            delay = retry_after if retry_after is not None else 2 ** attempt
            print(
                "LLM rate limit으로 재시도합니다: "
                f"{delay:g}초 후 ({attempt + 1}/{max_retries})"
            )
            time.sleep(delay)


@lru_cache(maxsize=1)
def get_llm() -> Any:
    """환경 설정으로 공용 ChatClovaX를 한 번 생성한다."""

    api_key = os.getenv("CLOVASTUDIO_API_KEY")
    if not api_key:
        raise RuntimeError("CLOVASTUDIO_API_KEY 환경변수가 필요합니다.")

    # Import가 무거우므로 실제 LLM 호출이 필요한 시점까지 지연한다.
    from langchain_naver import ChatClovaX

    return ChatClovaX(
        model=os.getenv("CLOVAX_MODEL_NAME", "HCX-005"),
        api_key=api_key,
        max_tokens=1024,
        temperature=0.0,
        top_p=0.8,
        top_k=0,
        repetition_penalty=1.1,
        disabled_params={"parallel_tool_calls": None},
    )


__all__ = [
    "MAX_LLM_RETRIES",
    "MAX_RATE_LIMIT_RETRIES",
    "build_output_retry_message",
    "get_llm",
    "invoke_with_rate_limit_retry",
]
