"""Materialize candidate lists for raw context-dataset rows.

Self-contained data path: raw rows (fa0311/ime-context-dataset) + the
dictionary pack (fa0311/ime-dictionary-core) + the Rust engine (`ime-cli
convert-dump`) produce train/dev files in the same shape the trainer and
teacher scorer consume:

  {"context", "reading", "target", "target_index",
   "candidates": [{"text", "search_cost", "kind", "whole_exact"}, ...]}

Candidates are generated at --generate-limit (512, the runtime session depth)
but stored truncated to --store-limit (128 > the teacher's head-97 scoring
window); rows whose target ranks below the stored window are dropped and
counted (recall). The train/dev split is a deterministic hash of the row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import orjson
from tqdm import tqdm

STORED_FIELDS = ("text", "search_cost", "kind", "whole_exact")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-file", type=Path, action="append", required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--ime-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generate-limit", type=int, default=512)
    parser.add_argument("--store-limit", type=int, default=128)
    parser.add_argument(
        "--dev-modulus", type=int, default=100, help="1/N of rows become the dev split"
    )
    return parser


def load_rows(files: list[Path]) -> list[dict]:
    rows = []
    for path in files:
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    row = orjson.loads(line)
                    row["source_file"] = path.name
                    rows.append(row)
    return rows


def dump_candidates(
    ime_cli: Path, pack: Path, readings: list[str], limit: int, cache: Path
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        readings_path = Path(tmp) / "readings.txt"
        readings_path.write_text("\n".join(readings) + "\n", encoding="utf-8", newline="\n")
        with cache.open("wb") as out:
            subprocess.run(
                [str(ime_cli), "convert-dump", str(pack), str(readings_path), str(limit)],
                stdout=out,
                check=True,
            )


def split_of(row: dict, dev_modulus: int) -> str:
    key = f"{row.get('context', '')} {row['reading']} {row['target']}"
    digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    return "dev" if digest % dev_modulus == 0 else "train"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(args.dataset_file)
    readings = sorted({str(row["reading"]) for row in rows})
    print(f"{len(rows)} rows, {len(readings)} unique readings")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "candidates.cache.jsonl"
    dump_candidates(args.ime_cli, args.pack, readings, args.generate_limit, cache)

    candidate_table: dict[str, list[list]] = {}
    with cache.open("rb") as handle:
        for line in handle:
            if line.strip():
                entry = orjson.loads(line)
                candidate_table[entry["reading"]] = entry["candidates"]

    counts = {"train": 0, "dev": 0, "target_missing": 0, "target_beyond_store": 0}
    outputs = {
        "train": (args.output_dir / "train.jsonl").open("wb"),
        "dev": (args.output_dir / "dev.jsonl").open("wb"),
    }
    try:
        for row in tqdm(rows, desc="materialize", unit="row"):
            candidates = candidate_table.get(str(row["reading"]), [])
            target = str(row["target"])
            target_index = next(
                (index for index, cand in enumerate(candidates) if cand[0] == target), None
            )
            if target_index is None:
                counts["target_missing"] += 1
                continue
            if target_index >= args.store_limit:
                counts["target_beyond_store"] += 1
                continue
            stored = [
                dict(zip(STORED_FIELDS, candidate[:4]))
                for candidate in candidates[: args.store_limit]
            ]
            split = split_of(row, args.dev_modulus)
            record = {
                "context": str(row.get("context", "")),
                "reading": str(row["reading"]),
                "target": target,
                "target_index": target_index,
                "source_file": row.get("source_file"),
                "candidates": stored,
            }
            outputs[split].write(orjson.dumps(record) + b"\n")
            counts[split] += 1
    finally:
        for handle in outputs.values():
            handle.close()
    cache.unlink()

    total = len(rows)
    report = {
        **counts,
        "total": total,
        "recall": (counts["train"] + counts["dev"]) / total if total else 0.0,
        "generate_limit": args.generate_limit,
        "store_limit": args.store_limit,
        "pack": str(args.pack),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
