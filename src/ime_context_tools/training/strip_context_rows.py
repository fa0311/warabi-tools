"""Derive no-context KD rows from materialized candidate rows.

The teacher ranks candidates well even with empty context (its pretrained
lexical knowledge tells real word compositions from broken ones, e.g.
アメリカンな over アメリカんな for あめりかんな). Materialized training data
is ~99% contextful, so that ability never transfers. This tool re-emits
existing rows with `context: ""` and no `target_index` (the gold target is
context-dependent, the empty-context ranking is the teacher's to decide);
`score_teacher_cross_encoder` then caches the teacher's no-context logits
and the student learns them through the KL/MSE terms.

Candidate lists depend only on the reading, so rows are deduplicated by
reading (first occurrence wins) and readings that already appear with empty
context in the inputs are skipped. Candidates are kept verbatim — kaomoji
included: readings like しょぼーん legitimately convert to faces and the
model needs the teacher's ranking on them (runtime composition abuse is
prevented structurally by the lattice's kaomoji policy instead).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="materialized candidate JSONL; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    seen: set[str] = set()
    for source in args.input:
        with source.open("rb") as handle:
            for line in handle:
                if line.strip():
                    row = orjson.loads(line)
                    if not row["context"]:
                        seen.add(row["reading"])

    written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        for source in args.input:
            with source.open("rb") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = orjson.loads(line)
                    if row["reading"] in seen:
                        continue
                    seen.add(row["reading"])
                    out.write(
                        orjson.dumps(
                            {
                                "context": "",
                                "reading": row["reading"],
                                "source_file": row.get("source_file"),
                                "candidates": row["candidates"],
                            }
                        )
                        + b"\n"
                    )
                    written += 1
    print(f"{args.output}: wrote {written} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
