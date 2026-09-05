"""Tier 3 배치 추출기 파일럿 대상 범위(기업 20개 × 최근 3개 회계연도).

여기 나열된 corp_code는 data/universe.csv(70개 기업 마스터)의 실제 값이며,
이 파일 자체는 그 값을 다시 베끼는 것이 아니라 항상 universe.csv를 기준으로
검증합니다 — corp_code가 하나라도 universe.csv에 없거나 회사명이 어긋나면
즉시 에러를 냅니다(잘못된 기업으로 조용히 파일럿을 도는 것을 방지).

파일럿 회사는 업종 다양성(반도체·자동차·2차전지·철강·비철금속·금융·플랫폼·
통신·게임·바이오·조선·방산·유통)과 안정적인 공시 이력(n_periodic 두 자릿수
이상, 최근 상장 종목 제외)을 기준으로 골랐습니다. 모두 결산월이 12월이라
기간 정렬 문제가 없습니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_graph.insertUniverse import load_companies

# corp_code -> universe.csv 기준 corp_name (교차검증용, 실제 값의 출처는 아님)
PILOT_COMPANIES: dict[str, str] = {
    "00126380": "삼성전자",
    "00164779": "SK하이닉스",
    "00164742": "현대자동차",
    "00106641": "기아",
    "01515323": "LG에너지솔루션",
    "00126362": "삼성SDI",
    "00155319": "POSCO홀딩스",
    "00102858": "고려아연",
    "00126256": "삼성생명",
    "00688996": "KB금융",
    "00382199": "신한지주",
    "00266961": "NAVER",
    "00258801": "카카오",
    "00159023": "SK텔레콤",
    "00760971": "크래프톤",
    "00877059": "삼성바이오로직스",
    "00413046": "셀트리온",
    "01390344": "HD현대중공업",
    "00126566": "한화에어로스페이스",
    "00583424": "아모레퍼시픽",
}

PILOT_CORP_CODES: tuple[str, ...] = tuple(PILOT_COMPANIES)

# 최근 3개 회계연도. 회계연도 Y의 사업보고서는 보통 Y+1년 3월에 제출되므로
# 셋 다 코퍼스 범위(2023-01-02 ~ 2026-06-01) 안의 사업보고서로 커버됩니다.
PILOT_FISCAL_YEARS: tuple[int, ...] = (2023, 2024, 2025)


def pilot_companies(
    csv_path: Path = Path("data/universe.csv"),
) -> list[dict[str, Any]]:
    """PILOT_CORP_CODES를 data/universe.csv와 대조해 회사 정보를 반환합니다.

    corp_code가 universe.csv에 없거나 corp_name이 PILOT_COMPANIES와
    다르면 잘못된 기업을 파일럿에 넣는 것을 막기 위해 즉시 예외를 냅니다.
    """

    if len(PILOT_CORP_CODES) != len(set(PILOT_CORP_CODES)):
        raise ValueError("PILOT_CORP_CODES에 중복된 corp_code가 있습니다.")

    universe_by_code = {
        company["corp_code"]: company for company in load_companies(csv_path)
    }

    missing = [code for code in PILOT_CORP_CODES if code not in universe_by_code]
    if missing:
        raise ValueError(
            f"data/universe.csv에 없는 corp_code입니다: {missing}"
        )

    mismatched = [
        code
        for code in PILOT_CORP_CODES
        if universe_by_code[code]["corp_name"] != PILOT_COMPANIES[code]
    ]
    if mismatched:
        raise ValueError(
            "PILOT_COMPANIES의 회사명이 data/universe.csv와 다릅니다: "
            f"{mismatched}"
        )

    return [universe_by_code[code] for code in PILOT_CORP_CODES]


__all__ = [
    "PILOT_COMPANIES",
    "PILOT_CORP_CODES",
    "PILOT_FISCAL_YEARS",
    "pilot_companies",
]
