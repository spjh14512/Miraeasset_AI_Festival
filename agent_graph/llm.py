import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv


load_dotenv()


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


__all__ = ["get_llm"]
