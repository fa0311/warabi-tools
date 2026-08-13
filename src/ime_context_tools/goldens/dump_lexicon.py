"""Dump Python-reference lexicon lookups for sampled readings.

Produces the same JSONL shape as `ime-cli lexicon-dump` so the two can be
diffed byte-for-byte:
  {"reading": ..., "exact": [[reading, text, cost, sources], ...],
   "exact_by_cost": [...], "predictions": [...]}

Readings are sampled from the eval files plus their leading substrings (which
exercises the prediction table).
"""

from __future__ import annotations

import argparse
import sqlite3
import unicodedata
from pathlib import Path

import jaconv
import orjson
import romkan2


def normalize_reading(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    return jaconv.kata2hira(romkan2.to_hiragana(normalized))


class ReferenceLexicon:
    """Trimmed copy of `tui_ime.lexicon.Lexicon` (kept dependency-free)."""

    def __init__(self, database_path: Path) -> None:
        uri = database_path.resolve().as_uri() + "?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)
        tables = {
            row[0]
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self._has_predictions = "predictions" in tables
        self._entries_have_sources = "sources" in {
            row[1] for row in self._connection.execute("PRAGMA table_info(entries)")
        }
        self._predictions_have_sources = self._has_predictions and "sources" in {
            row[1] for row in self._connection.execute("PRAGMA table_info(predictions)")
        }

    def _dedupe(self, rows) -> list[list]:
        result: list[list] = []
        seen: set[str] = set()
        for reading, text, cost, sources in rows:
            if text in seen:
                continue
            seen.add(text)
            result.append([reading, text, cost, sources])
        return result

    def exact(self, reading: str, *, by_cost: bool) -> list[list]:
        reading = normalize_reading(reading)
        order = "cost, candidate COLLATE BINARY" if by_cost else "candidate COLLATE BINARY"
        sources = "sources" if self._entries_have_sources else "'' AS sources"
        rows = self._connection.execute(
            f"SELECT reading, candidate, cost, {sources} FROM entries "
            f"WHERE reading = ? ORDER BY {order}",
            (reading,),
        )
        return self._dedupe(rows)

    def predictions(self, reading: str) -> list[list]:
        reading = normalize_reading(reading)
        if not self._has_predictions:
            return []
        if self._predictions_have_sources:
            rows = self._connection.execute(
                "SELECT reading, candidate, cost, sources FROM predictions "
                "WHERE reading = ? "
                "  AND (instr(',' || sources || ',', ',original:predict.bin,') > 0 "
                "       OR instr(',' || sources || ',', ',original:predict_model_dict.bin,') > 0) "
                "ORDER BY cost, candidate COLLATE BINARY",
                (reading,),
            )
        else:
            rows = self._connection.execute(
                "SELECT reading, candidate, cost, '' AS sources FROM predictions "
                "WHERE reading = ? ORDER BY cost, candidate COLLATE BINARY",
                (reading,),
            )
        return self._dedupe(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--readings-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    readings: list[str] = []
    seen: set[str] = set()
    for path in args.eval_file:
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                reading = str(orjson.loads(line).get("reading", ""))
                for variant in (reading, reading[:2], reading[:4]):
                    if variant and variant not in seen:
                        seen.add(variant)
                        readings.append(variant)
        if len(readings) >= args.limit:
            break
    readings = readings[: args.limit]

    lexicon = ReferenceLexicon(args.pack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.readings_output.write_text(
        "\n".join(readings) + "\n", encoding="utf-8", newline="\n"
    )
    with args.output.open("wb") as out:
        for reading in readings:
            row = {
                "reading": reading,
                "exact": lexicon.exact(reading, by_cost=False),
                "exact_by_cost": lexicon.exact(reading, by_cost=True),
                "predictions": lexicon.predictions(reading),
            }
            out.write(orjson.dumps(row) + b"\n")
    print(f"dumped {len(readings)} readings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
