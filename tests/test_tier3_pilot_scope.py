from pathlib import Path

import pytest

from knowledge_graph.tier3_pilot_scope import (
    PILOT_COMPANIES,
    PILOT_CORP_CODES,
    PILOT_FISCAL_YEARS,
    pilot_companies,
)

UNIVERSE_CSV = Path("data/universe.csv")


def test_pilot_scope_is_twenty_companies_three_years():
    assert len(PILOT_CORP_CODES) == 20
    assert len(set(PILOT_CORP_CODES)) == 20
    assert PILOT_FISCAL_YEARS == (2023, 2024, 2025)


def test_pilot_companies_match_real_universe_csv():
    companies = pilot_companies(UNIVERSE_CSV)

    assert len(companies) == 20
    resolved = {company["corp_code"]: company["corp_name"] for company in companies}
    assert resolved == PILOT_COMPANIES


def test_pilot_companies_span_multiple_sectors():
    companies = pilot_companies(UNIVERSE_CSV)

    sectors = {company["sector"] for company in companies}
    assert len(sectors) >= 10


def test_pilot_companies_rejects_unknown_corp_code(monkeypatch):
    import knowledge_graph.tier3_pilot_scope as scope

    monkeypatch.setattr(scope, "PILOT_CORP_CODES", ("99999999",))

    with pytest.raises(ValueError, match="corp_code"):
        pilot_companies(UNIVERSE_CSV)


def test_pilot_companies_rejects_name_mismatch(monkeypatch):
    import knowledge_graph.tier3_pilot_scope as scope

    first_code = PILOT_CORP_CODES[0]
    monkeypatch.setattr(
        scope, "PILOT_COMPANIES", {first_code: "존재하지않는이름"}
    )
    monkeypatch.setattr(scope, "PILOT_CORP_CODES", (first_code,))

    with pytest.raises(ValueError, match="다릅니다"):
        pilot_companies(UNIVERSE_CSV)
