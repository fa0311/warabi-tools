from __future__ import annotations

import argparse
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

import orjson

from .curated_context import load_curated_context, load_excluded_targets
from .paths import ARTIFACTS_DIR, DATASETS_DIR, DEFAULT_DICTIONARY_PATH, DEFAULT_IME_CLI

from .build_ime_dictionary import (
    DICT,
    Entry,
    iter_ipadic,
    iter_jmdict,
    iter_kanjidic,
    iter_mozc,
    iter_original,
    iter_skk,
    iter_sudachi,
)

CORE_ORIGINAL_SOURCES = {"original:dict.bin", "original:kernel_dict.bin"}
EXTENDED_ORIGINAL_SOURCES = CORE_ORIGINAL_SOURCES | {"original:white_dict.bin"}


def _filtered(entries: Iterable[Entry], allowed: set[str]) -> Iterable[Entry]:
    for entry in entries:
        if entry[3] in allowed:
            yield entry


def build_database(path: Path, entries: Iterable[Entry]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE entries (
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            cost INTEGER NOT NULL,
            sources TEXT NOT NULL,
            PRIMARY KEY (reading, candidate)
        ) WITHOUT ROWID;
        """
    )
    sql = """
        INSERT INTO entries(reading, candidate, cost, sources) VALUES (?, ?, ?, ?)
        ON CONFLICT(reading, candidate) DO UPDATE SET
            cost = min(entries.cost, excluded.cost),
            sources = excluded.sources
    """
    count = 0
    batch: list[Entry] = []
    for entry in entries:
        batch.append(entry)
        if len(batch) == 20_000:
            connection.executemany(sql, batch)
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(sql, batch)
        count += len(batch)
    connection.execute("CREATE INDEX entries_reading_cost ON entries(reading, cost, candidate)")
    connection.commit()
    unique = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
    connection.close()
    return unique


def evaluate(
    path: Path,
    cases_path: Path,
    candidate_count: int,
    ime_cli: Path = DEFAULT_IME_CLI,
) -> dict[str, object]:
    """Rank curated cases with the Rust engine (`ime-cli convert-dump`,
    byte-parity verified against the retired Python engine)."""
    invalid = DATASETS_DIR / "evaluation" / "public" / "ime_context_test_curated_invalid.tsv"
    excluded = load_excluded_targets(invalid) if invalid.is_file() else frozenset()
    dataset = load_curated_context(cases_path, excluded_targets=excluded)
    readings = sorted({case.reading for case in dataset.cases})
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        readings_path = Path(tmp) / "readings.txt"
        readings_path.write_text("\n".join(readings) + "\n", encoding="utf-8", newline="\n")
        result = subprocess.run(
            [str(ime_cli), "convert-dump", str(path), str(readings_path), str(candidate_count)],
            capture_output=True,
            check=True,
        )
    candidates_by_reading = {
        row["reading"]: [candidate[0] for candidate in row["candidates"]]
        for row in map(orjson.loads, result.stdout.splitlines())
        if row
    }
    ranks: list[int | None] = []
    for case in dataset.cases:
        texts = candidates_by_reading.get(case.reading, [])
        matching = [texts.index(value) + 1 for value in case.expected if value in texts]
        ranks.append(min(matching) if matching else None)
    total = len(ranks)
    return {
        "cases": total,
        "recall": sum(rank is not None for rank in ranks) / total,
        "top1": sum(rank == 1 for rank in ranks) / total,
        "top5": sum(rank is not None and rank <= 5 for rank in ranks) / total,
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / total,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw source dictionaries at Recall@N")
    parser.add_argument(
        "--cases", type=Path, default=DATASETS_DIR / "source" / "ime_context_curated.txt"
    )
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--ime-cli", type=Path, default=DEFAULT_IME_CLI)
    parser.add_argument(
        "--output", type=Path, default=ARTIFACTS_DIR / "cache" / "source-benchmark"
    )
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--variants",
        default="mozc,original_core,original_extended,current_merged",
        help="comma-separated variants to build/evaluate",
    )
    args = parser.parse_args()

    variants: list[tuple[str, Path, Iterable[Entry] | None]] = [
        ("mozc", args.output / "mozc.sqlite3", iter_mozc()),
        (
            "original_core",
            args.output / "original-core.sqlite3",
            _filtered(iter_original(predictions=False), CORE_ORIGINAL_SOURCES),
        ),
        (
            "original_extended",
            args.output / "original-extended.sqlite3",
            _filtered(iter_original(predictions=False), EXTENDED_ORIGINAL_SOURCES),
        ),
        ("current_merged", DEFAULT_DICTIONARY_PATH, None),
        ("skk", args.output / "skk.sqlite3", iter_skk()),
        (
            "jmdict",
            args.output / "jmdict.sqlite3",
            iter_jmdict(DICT / "jmdict" / "xml" / "JMdict_e", "jmdict", 6_000),
        ),
        (
            "jmnedict",
            args.output / "jmnedict.sqlite3",
            iter_jmdict(DICT / "jmdict" / "xml" / "JMnedict.xml", "jmnedict", 6_800),
        ),
        ("kanjidic2", args.output / "kanjidic2.sqlite3", iter_kanjidic()),
        ("ipadic", args.output / "ipadic.sqlite3", iter_ipadic()),
        ("sudachi", args.output / "sudachi.sqlite3", iter_sudachi()),
    ]
    selected = {value.strip() for value in args.variants.split(",") if value.strip()}
    for label, database, entries in variants:
        if label not in selected:
            continue
        unique = None
        if entries is not None and (not args.reuse or not database.is_file()):
            unique = build_database(database, entries)
        result = evaluate(database, args.cases, args.candidate_count, args.ime_cli)
        print(orjson.dumps({"source": label, "unique_entries": unique, **result}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
