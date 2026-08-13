"""Review and explicitly promote converter regression snapshot candidates."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "fixtures" / "converter_pipeline" / "golden"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    candidates = sorted(args.candidate_dir.resolve().glob("*.json"))
    if args.cases:
        selected = set(args.cases)
        candidates = [path for path in candidates if path.stem in selected]
    if not candidates:
        raise SystemExit("No matching candidate snapshots were found.")

    golden_dir = args.golden_dir.resolve()
    for candidate in candidates:
        golden = golden_dir / candidate.name
        old_lines = (
            golden.read_text(encoding="utf-8").splitlines(keepends=True)
            if golden.exists()
            else []
        )
        new_lines = candidate.read_text(encoding="utf-8").splitlines(keepends=True)
        print(
            "".join(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=str(golden),
                    tofile=str(candidate),
                )
            )
        )

    if not args.apply:
        raise SystemExit(
            "Review only: no golden files changed. Re-run with --apply to approve."
        )

    golden_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        shutil.copyfile(candidate, golden_dir / candidate.name)
        print(f"approved={candidate.stem}")


if __name__ == "__main__":
    main()
