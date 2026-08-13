"""Generate unlabeled KD-augmentation rows from the dictionary itself.

The from-scratch students lack the lexical common sense a pretrained teacher
has (which candidate is plausible at all, independent of context). KD needs
no gold labels: sample ambiguous readings from the dictionary, materialize
their candidate lists with the Rust engine, and let the teacher's context-free
scores supervise the soft losses. Rows carry no `target_index`; the trainer
skips the hard CE term for them.

With `--suffixes`, each sampled reading is composed with one particle
(word+particle readings dominate real IME conversion — あめりかんな — and the
teacher tells real compositions like アメリカンな from broken segmentations
like アメリカんな; plain dictionary readings never exercise that). Composed
bases should be common words the teacher knows: pair with `--max-entry-cost`.

Output rows: {"context": "", "reading", "candidates": [...]} — same candidate
shape as materialize_candidates.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

import orjson

from ..dictionary.kaomoji_filter import is_kaomoji_like
from .materialize_candidates import STORED_FIELDS, dump_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--ime-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-reading-chars", type=int, default=2)
    parser.add_argument("--max-reading-chars", type=int, default=10)
    parser.add_argument(
        "--min-entry-candidates",
        type=int,
        default=2,
        help="sample only readings with at least this many dictionary entries (ambiguity)",
    )
    parser.add_argument("--store-limit", type=int, default=128)
    parser.add_argument(
        "--min-candidates", type=int, default=32, help="drop rows with fewer generated candidates"
    )
    parser.add_argument(
        "--suffixes",
        help="comma-separated particle suffixes; each sampled reading is "
        "composed with one (rng-chosen) instead of used bare",
    )
    parser.add_argument(
        "--max-entry-cost",
        type=int,
        help="sample only readings whose cheapest entry is at most this cost (common words)",
    )
    return parser


def sample_readings(args: argparse.Namespace) -> list[str]:
    connection = sqlite3.connect(args.pack)
    query = (
        "SELECT reading FROM entries "
        "WHERE LENGTH(reading) BETWEEN ? AND ? "
        "GROUP BY reading HAVING COUNT(*) >= ?"
    )
    parameters: list[object] = [
        args.min_reading_chars,
        args.max_reading_chars,
        args.min_entry_candidates,
    ]
    if args.max_entry_cost is not None:
        query += " AND MIN(cost) <= ?"
        parameters.append(args.max_entry_cost)
    readings = [row[0] for row in connection.execute(query, parameters)]
    if args.suffixes:
        # Composition bases must be real word readings (the engine contract:
        # readings are hiragana); the dictionary also carries symbol/kaomoji
        # trigger readings, which compose into nonsense.
        readings = [
            reading
            for reading in readings
            if all("ぁ" <= char <= "ゖ" or char == "ー" for char in reading)
        ]
    rng = random.Random(args.seed)
    if len(readings) > args.count:
        readings = rng.sample(readings, args.count)
    if args.suffixes:
        suffixes = args.suffixes.split(",")
        readings = list({reading + rng.choice(suffixes) for reading in readings})
    return sorted(readings)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    readings = sample_readings(args)
    print(f"sampled {len(readings)} readings")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "candidates.cache.jsonl"
    dump_candidates(args.ime_cli, args.pack, readings, args.store_limit, cache)

    written = 0
    skipped = 0
    with (args.output_dir / "train.jsonl").open("wb") as out, cache.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = orjson.loads(line)
            candidates = entry["candidates"]
            if args.suffixes:
                # The teacher's MeCab tokenizer shreds kaomoji and scores
                # them erratically on empty-context rows (top-1 in 13-30%
                # of composed rows) — the student then learned kaomoji as
                # a class preference. Kaomoji stay a dictionary concern.
                candidates = [
                    candidate
                    for candidate in candidates
                    if not is_kaomoji_like(candidate[0])
                ]
            if len(candidates) < args.min_candidates:
                skipped += 1
                continue
            record = {
                "context": "",
                "reading": entry["reading"],
                "candidates": [
                    dict(zip(STORED_FIELDS, candidate[:4])) for candidate in candidates
                ],
            }
            out.write(orjson.dumps(record) + b"\n")
            written += 1
    cache.unlink()

    report = {
        "rows": written,
        "skipped_thin_readings": skipped,
        "seed": args.seed,
        "store_limit": args.store_limit,
        "pack": str(args.pack),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
