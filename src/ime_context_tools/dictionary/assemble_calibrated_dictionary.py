from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
from .paths import ARTIFACTS_DIR, DICTIONARY_ARTIFACTS_DIR

from .build_ime_dictionary import iter_original
from .calibrate_dictionary_sources import isotonic_mapping, load_overlap

DEFAULT_SUPPLEMENTS = (
    ("jmdict.sqlite3", 4_000),
    ("ipadic.sqlite3", 5_000),
    ("skk.sqlite3", 5_000),
    ("jmnedict.sqlite3", 4_000),
    ("kanjidic2.sqlite3", 6_000),
    ("sudachi.sqlite3", 6_000),
)


def add_supplement(database: Path, source: Path, penalty: int) -> dict[str, object]:
    source_cost, reference_cost = load_overlap(database, source)
    transform, mapping = isotonic_mapping(source_cost, reference_cost)
    connection = sqlite3.connect(database)
    connection.execute("ATTACH DATABASE ? AS source", (str(source),))
    cursor = connection.execute(
        """
        SELECT reading, candidate, cost, sources
        FROM source.entries
        WHERE NOT EXISTS (
            SELECT 1 FROM main.entries
            WHERE main.entries.reading = source.entries.reading
              AND main.entries.candidate = source.entries.candidate
        )
        """
    )
    added = 0
    while rows := cursor.fetchmany(20_000):
        raw = np.fromiter((row[2] for row in rows), dtype=np.float64, count=len(rows))
        costs = np.rint(transform(raw)).astype(int) + penalty
        connection.executemany(
            "INSERT INTO entries(reading, candidate, cost, sources) VALUES (?, ?, ?, ?)",
            (
                (row[0], row[1], int(cost), row[3])
                for row, cost in zip(rows, costs, strict=True)
            ),
        )
        added += len(rows)
    connection.commit()
    connection.close()
    return {
        "source": source.name,
        "penalty": penalty,
        "overlap_anchors": len(source_cost),
        "added_entries": added,
        "mapping_points": len(mapping["x_thresholds"]),
    }


def build_predictions(database: Path) -> int:
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE predictions (
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            cost INTEGER NOT NULL,
            sources TEXT NOT NULL,
            PRIMARY KEY (reading, candidate)
        ) WITHOUT ROWID;
        """
    )
    sql = """
        INSERT INTO predictions(reading, candidate, cost, sources) VALUES (?, ?, ?, ?)
        ON CONFLICT(reading, candidate) DO UPDATE SET
            cost = min(predictions.cost, excluded.cost),
            sources = excluded.sources
    """
    batch = []
    for entry in iter_original(predictions=True):
        batch.append(entry)
        if len(batch) == 20_000:
            connection.executemany(sql, batch)
            batch.clear()
    if batch:
        connection.executemany(sql, batch)
    connection.execute(
        "CREATE INDEX predictions_reading_cost ON predictions(reading, cost, candidate)"
    )
    count = connection.execute("SELECT count(*) FROM predictions").fetchone()[0]
    connection.commit()
    connection.close()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the production IME dictionary on the calibrated original ranking"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ARTIFACTS_DIR
        / "cache"
        / "source-benchmark"
        / "original-mozc-mozcscale-quantile.sqlite3",
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=ARTIFACTS_DIR / "cache" / "source-benchmark",
    )
    parser.add_argument(
        "--output", type=Path, default=DICTIONARY_ARTIFACTS_DIR / "ime.calibrated.sqlite3"
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    shutil.copyfile(args.base, args.output)
    reports = [
        add_supplement(args.output, args.source_directory / filename, penalty)
        for filename, penalty in DEFAULT_SUPPLEMENTS
    ]
    predictions = build_predictions(args.output)
    connection = sqlite3.connect(args.output)
    total = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('assembly', ?)",
        (
            json.dumps(
                {
                    "base": str(args.base),
                    "reference_order": "original_extended",
                    "numeric_scale": "mozc_quantile",
                    "supplements": reports,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
    )
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    print(
        json.dumps(
            {"entries": total, "predictions": predictions, "supplements": reports},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
