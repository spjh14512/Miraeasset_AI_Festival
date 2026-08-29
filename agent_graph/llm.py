import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv


load_dotenv()

MAX_LLM_RETRIES = 2


def build_output_retry_message(component: str, error: BaseException) -> str:
    """LLM 출력 검증 오류를 해당 LLM의 재생성 요청으로 변환합니다."""

    return (
        f"방금 생성한 {component} 출력이 application 검증을 통과하지 못했습니다.\n"
        f"오류: {type(error).__name__}: {error}\n"
        "오류 원인을 수정하여 전체 출력을 요구된 schema 또는 tool call 형식으로 "
        "다시 생성하세요. 오류가 난 출력을 그대로 반복하지 마세요."
    )


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


__all__ = ["MAX_LLM_RETRIES", "build_output_retry_message", "get_llm"]
