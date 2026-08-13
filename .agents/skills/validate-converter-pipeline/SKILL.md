---
name: validate-converter-pipeline
description: Run and diagnose this project's fixed DART XML to Canonical Section to Evidence Fragment regression corpus. Use after changing section chunking, paragraph parsing, table parsing, evidence assembly, schemas, validators, or normalization; use before accepting converter changes that could alter evidence or external records.
---

# Validate Converter Pipeline

Validate the current converter code against reviewed real-world fixtures without writing to production `data/canonical_section` or `data/evidence_fragment`.

## Workflow

1. Run the quick profile while iterating:

   `uv run python scripts/validate_converter_pipeline.py --profile quick`

2. Run the full profile before declaring converter work complete:

   `uv run python scripts/validate_converter_pipeline.py --profile full`

3. For a focused failure, pass `--case CASE_ID` one or more times. Read `tests/fixtures/converter_pipeline/corpus.json` for case IDs and assertions.

4. Report structural, semantic, and snapshot failures separately. Read [failure-codes.md](references/failure-codes.md) when a failure needs interpretation.

## Golden changes

Never modify approved golden files as part of an ordinary validation run.

Generate non-overwriting candidates in a new directory:

`uv run python scripts/validate_converter_pipeline.py --profile full --write-candidate tmp/converter-golden-candidate`

Display candidate-versus-golden diffs without changing files:

`uv run python scripts/approve_converter_golden.py --candidate-dir tmp/converter-golden-candidate`

Only after the user explicitly approves the displayed change, run the same command with `--apply`. Never infer approval from a request to run validation.

## Required report

Summarize:

- overall pass/fail and elapsed time;
- documents, sections, evidence, and external-record counts;
- each failing case and failure code;
- whether production output remained untouched;
- whether golden files were read, proposed, or explicitly changed.

The full suite intentionally exposes known converter defects when their semantic assertion fails. Do not hide those failures by weakening assertions.
