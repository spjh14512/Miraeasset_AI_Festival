"""Insert Company nodes from data/universe.csv into Neo4j."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase


EXPECTED_COMPANY_COUNT = 70
COMPANY_FIELDS = (
    "corp_code",
    "stock_code",
    "corp_name",
    "corp_eng_name",
    "market",
    "industry",
    "sector",
    "listing_date",
    "market_cap",
)

CREATE_COMPANY_CONSTRAINT = """
CREATE CONSTRAINT company_corp_code IF NOT EXISTS
FOR (company:Company)
REQUIRE company.corp_code IS UNIQUE
"""

UPSERT_COMPANIES = """
UNWIND $rows AS row
MERGE (company:Company {corp_code: row.corp_code})
SET company.stock_code = row.stock_code,
    company.corp_name = row.corp_name,
    company.corp_eng_name = row.corp_eng_name,
    company.market = row.market,
    company.industry = row.industry,
    company.sector = row.sector,
    company.listing_date =
        CASE
            WHEN row.listing_date IS NULL THEN null
            ELSE date(row.listing_date)
        END,
    company.market_cap = row.market_cap
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/universe.csv"),
        help="Universe CSV path",
    )
    parser.add_argument("--database", default=None, help="Neo4j database name")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def parse_market_cap(value: str | None, *, row_number: int) -> int | None:
    normalized = optional_string(value)
    if normalized is None:
        return None
    try:
        return int(normalized.replace(",", ""))
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: market_cap must be an integer, got {value!r}"
        ) from exc


def load_companies(
    csv_path: Path, *, expected_count: int = EXPECTED_COMPANY_COUNT
) -> list[dict[str, Any]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or ())
        missing_fields = sorted(set(COMPANY_FIELDS) - fieldnames)
        if missing_fields:
            raise ValueError(
                f"Missing required CSV columns: {', '.join(missing_fields)}"
            )

        companies: list[dict[str, Any]] = []
        for row_number, source_row in enumerate(reader, start=2):
            corp_code = optional_string(source_row.get("corp_code"))
            if corp_code is None:
                raise ValueError(f"Row {row_number}: corp_code is required")
            listing_date = optional_string(source_row.get("listing_date"))
            companies.append(
                {
                    "corp_code": corp_code,
                    "stock_code": optional_string(source_row.get("stock_code")),
                    "corp_name": optional_string(source_row.get("corp_name")),
                    "corp_eng_name": optional_string(
                        source_row.get("corp_eng_name")
                    ),
                    "market": optional_string(source_row.get("market")),
                    "industry": optional_string(source_row.get("industry")),
                    "sector": optional_string(source_row.get("sector")),
                    "listing_date": listing_date,
                    "market_cap": parse_market_cap(
                        source_row.get("market_cap"), row_number=row_number
                    ),
                }
            )

    if len(companies) != expected_count:
        raise ValueError(
            f"Expected {expected_count} companies, found {len(companies)}"
        )

    code_counts = Counter(company["corp_code"] for company in companies)
    duplicate_codes = sorted(
        code for code, count in code_counts.items() if count > 1
    )
    if duplicate_codes:
        raise ValueError(
            f"Duplicate corp_code values: {', '.join(duplicate_codes)}"
        )
    return companies


def chunks(
    rows: Sequence[dict[str, Any]], batch_size: int
) -> Iterable[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def insert_companies(
    driver: Driver,
    companies: Sequence[dict[str, Any]],
    *,
    database: str | None,
    batch_size: int,
) -> None:
    driver.verify_connectivity()
    with driver.session(database=database) as session:
        session.run(CREATE_COMPANY_CONSTRAINT).consume()
        for batch in chunks(companies, batch_size):
            session.run(UPSERT_COMPANIES, rows=batch).consume()


def build_summary(companies: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "companies": len(companies),
        "markets": dict(
            sorted(
                Counter(
                    str(company["market"])
                    for company in companies
                    if company["market"] is not None
                ).items()
            )
        ),
        "fields": list(COMPANY_FIELDS),
    }


def main() -> int:
    args = parse_args()
    companies = load_companies(args.csv)
    print(json.dumps(build_summary(companies), ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all((uri, username, password)):
        raise RuntimeError(
            "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be configured"
        )

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        insert_companies(
            driver,
            companies,
            database=args.database,
            batch_size=args.batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
