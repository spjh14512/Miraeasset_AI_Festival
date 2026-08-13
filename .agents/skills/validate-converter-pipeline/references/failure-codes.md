# Failure codes

- `SOURCE_HASH_MISMATCH`: a compressed fixture no longer matches the reviewed raw source.
- `STRUCTURE_VALIDATION_FAILED`: the existing Evidence Fragment validator rejected generated output.
- `SEMANTIC_ASSERTION_FAILED`: a human-readable invariant in `corpus.json` failed.
- `GOLDEN_MISSING`: an approved normalized snapshot does not exist.
- `SNAPSHOT_MISMATCH`: generated normalized output differs from the approved golden.
- `PRODUCTION_WRITE_DETECTED`: production canonical-section or evidence-fragment output changed during an isolated run.

Do not update a golden merely to make the suite pass. Generate candidates, inspect the diff, explain why the change is correct, and obtain explicit user approval before promotion.
