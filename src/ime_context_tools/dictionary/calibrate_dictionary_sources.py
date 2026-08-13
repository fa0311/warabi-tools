from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


def load_overlap(reference: Path, source: Path) -> tuple[np.ndarray, np.ndarray]:
    connection = sqlite3.connect(":memory:")
    connection.execute("ATTACH DATABASE ? AS reference", (str(reference),))
    connection.execute("ATTACH DATABASE ? AS source", (str(source),))
    rows = connection.execute(
        """
        SELECT source.entries.cost, reference.entries.cost
        FROM source.entries
        JOIN reference.entries USING (reading, candidate)
        """
    ).fetchall()
    connection.close()
    values = np.asarray(rows, dtype=np.float64)
    if len(values) < 1_000:
        raise ValueError(f"too few exact overlap anchors: {len(values)}")
    return values[:, 0], values[:, 1]


Transform = Callable[[np.ndarray], np.ndarray]


def isotonic_mapping(x: np.ndarray, y: np.ndarray) -> tuple[Transform, dict[str, object]]:
    model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)

    def transform(values: np.ndarray) -> np.ndarray:
        return model.predict(values)

    metadata = {
        "method": "isotonic",
        "x_thresholds": model.X_thresholds_.astype(int).tolist(),
        "y_thresholds": np.rint(model.y_thresholds_).astype(int).tolist(),
    }
    return transform, metadata


def quantile_mapping(x: np.ndarray, y: np.ndarray) -> tuple[Transform, dict[str, object]]:
    quantiles = np.linspace(0.0, 1.0, 257)
    x_points = np.quantile(x, quantiles)
    y_points = np.quantile(y, quantiles)
    unique_x, indices = np.unique(x_points, return_index=True)
    unique_y = y_points[indices]

    def transform(values: np.ndarray) -> np.ndarray:
        return np.interp(values, unique_x, unique_y, left=unique_y[0], right=unique_y[-1])

    metadata = {
        "method": "quantile",
        "x_thresholds": np.rint(unique_x).astype(int).tolist(),
        "y_thresholds": np.rint(unique_y).astype(int).tolist(),
    }
    return transform, metadata


def merge(
    reference: Path,
    source: Path,
    output: Path,
    transform: Transform,
    metadata: dict[str, object],
    addition_penalty: int,
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
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
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        """
    )
    connection.execute("ATTACH DATABASE ? AS reference", (str(reference),))
    connection.execute("ATTACH DATABASE ? AS source", (str(source),))
    connection.execute("INSERT INTO entries SELECT * FROM reference.entries")
    cursor = connection.execute(
        """
        SELECT reading, candidate, cost, sources
        FROM source.entries
        WHERE NOT EXISTS (
            SELECT 1 FROM reference.entries
            WHERE reference.entries.reading = source.entries.reading
              AND reference.entries.candidate = source.entries.candidate
        )
        """
    )
    added = 0
    while rows := cursor.fetchmany(20_000):
        raw = np.fromiter((row[2] for row in rows), dtype=np.float64, count=len(rows))
        calibrated = np.rint(transform(raw)).astype(int) + addition_penalty
        connection.executemany(
            "INSERT INTO entries(reading, candidate, cost, sources) VALUES (?, ?, ?, ?)",
            (
                (row[0], row[1], int(cost), row[3])
                for row, cost in zip(rows, calibrated, strict=True)
            ),
        )
        added += len(rows)
    connection.execute("CREATE INDEX entries_reading_cost ON entries(reading, cost, candidate)")
    connection.execute(
        "INSERT INTO metadata VALUES ('cost_calibration', ?)",
        (json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),),
    )
    total = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
    reference_count = connection.execute("SELECT count(*) FROM reference.entries").fetchone()[0]
    connection.commit()
    connection.close()
    return {"reference_entries": reference_count, "added_entries": added, "total_entries": total}


def merge_on_source_scale(
    reference: Path,
    source: Path,
    output: Path,
    transform: Transform,
    metadata: dict[str, object],
    addition_penalty: int,
) -> dict[str, int]:
    """Keep reference ordering while expressing it on the source's numeric scale."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
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
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        """
    )
    connection.execute("ATTACH DATABASE ? AS reference", (str(reference),))
    connection.execute("ATTACH DATABASE ? AS source", (str(source),))
    cursor = connection.execute("SELECT reading, candidate, cost, sources FROM reference.entries")
    reference_count = 0
    while rows := cursor.fetchmany(20_000):
        raw = np.fromiter((row[2] for row in rows), dtype=np.float64, count=len(rows))
        calibrated = np.rint(transform(raw)).astype(int)
        connection.executemany(
            "INSERT INTO entries(reading, candidate, cost, sources) VALUES (?, ?, ?, ?)",
            (
                (row[0], row[1], int(cost), row[3])
                for row, cost in zip(rows, calibrated, strict=True)
            ),
        )
        reference_count += len(rows)
    cursor = connection.execute(
        """
        SELECT reading, candidate, cost, sources
        FROM source.entries
        WHERE NOT EXISTS (
            SELECT 1 FROM reference.entries
            WHERE reference.entries.reading = source.entries.reading
              AND reference.entries.candidate = source.entries.candidate
        )
        """
    )
    added = 0
    while rows := cursor.fetchmany(20_000):
        connection.executemany(
            "INSERT INTO entries(reading, candidate, cost, sources) VALUES (?, ?, ?, ?)",
            (
                (row[0], row[1], int(row[2]) + addition_penalty, row[3])
                for row in rows
            ),
        )
        added += len(rows)
    connection.execute("CREATE INDEX entries_reading_cost ON entries(reading, cost, candidate)")
    connection.execute(
        "INSERT INTO metadata VALUES ('cost_calibration', ?)",
        (json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),),
    )
    total = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
    connection.commit()
    connection.close()
    return {"reference_entries": reference_count, "added_entries": added, "total_entries": total}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate a supplementary dictionary onto a reference cost scale"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("isotonic", "quantile"), default="isotonic")
    parser.add_argument(
        "--addition-penalty",
        type=int,
        default=0,
        help="fallback penalty applied only to source-only entries",
    )
    parser.add_argument(
        "--output-scale",
        choices=("reference", "source"),
        default="reference",
        help="numeric cost scale used by the resulting database",
    )
    args = parser.parse_args()

    source_cost, reference_cost = load_overlap(args.reference, args.source)
    if args.output_scale == "reference":
        x, y = source_cost, reference_cost
    else:
        x, y = reference_cost, source_cost
    factory = isotonic_mapping if args.method == "isotonic" else quantile_mapping
    transform, metadata = factory(x, y)
    metadata.update(
        {
            "reference": str(args.reference),
            "source": str(args.source),
            "overlap_anchors": len(x),
        }
    )
    metadata["addition_penalty"] = args.addition_penalty
    metadata["output_scale"] = args.output_scale
    merge_function = merge if args.output_scale == "reference" else merge_on_source_scale
    result = merge_function(
        args.reference, args.source, args.output, transform, metadata, args.addition_penalty
    )
    print(
        json.dumps(
            {
                "method": metadata["method"],
                "reference": metadata["reference"],
                "source": metadata["source"],
                "overlap_anchors": metadata["overlap_anchors"],
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
