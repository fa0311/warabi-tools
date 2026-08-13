from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .paths import ARTIFACTS_DIR, DEFAULT_DICTIONARY_PATH

from .assemble_calibrated_dictionary import (
    DEFAULT_SUPPLEMENTS,
    add_supplement,
    build_predictions,
)
from .benchmark_dictionary_sources import build_database
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
from .calibrate_dictionary_sources import load_overlap, merge_on_source_scale, quantile_mapping

ORIGINAL_REFERENCE_SOURCES = {
    "original:dict.bin",
    "original:kernel_dict.bin",
    "original:white_dict.bin",
}


def only_sources(entries: Iterable[Entry], allowed: set[str]) -> Iterable[Entry]:
    for entry in entries:
        if entry[3] in allowed:
            yield entry


def source_builds() -> dict[str, Iterable[Entry]]:
    return {
        "mozc.sqlite3": iter_mozc(),
        "original-extended.sqlite3": only_sources(
            iter_original(predictions=False), ORIGINAL_REFERENCE_SOURCES
        ),
        "skk.sqlite3": iter_skk(),
        "jmdict.sqlite3": iter_jmdict(
            DICT / "jmdict" / "xml" / "JMdict_e", "jmdict", 6_000
        ),
        "jmnedict.sqlite3": iter_jmdict(
            DICT / "jmdict" / "xml" / "JMnedict.xml", "jmnedict", 6_800
        ),
        "kanjidic2.sqlite3": iter_kanjidic(),
        "ipadic.sqlite3": iter_ipadic(),
        "sudachi.sqlite3": iter_sudachi(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the calibrated production IME lexicon")
    parser.add_argument(
        "--work-directory", type=Path, default=ARTIFACTS_DIR / "cache" / "dictionary-rebuild"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DICTIONARY_PATH)
    parser.add_argument("--reuse-sources", action="store_true")
    args = parser.parse_args()

    args.work_directory.mkdir(parents=True, exist_ok=True)
    source_reports: dict[str, int] = {}
    for filename, entries in source_builds().items():
        path = args.work_directory / filename
        if args.reuse_sources and path.is_file():
            connection = sqlite3.connect(path)
            source_reports[filename] = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
            connection.close()
        else:
            source_reports[filename] = build_database(path, entries)

    original = args.work_directory / "original-extended.sqlite3"
    mozc = args.work_directory / "mozc.sqlite3"
    base = args.work_directory / "original-mozc-calibrated.sqlite3"
    source_cost, reference_cost = load_overlap(original, mozc)
    transform, mapping = quantile_mapping(reference_cost, source_cost)
    metadata = {
        "method": "quantile",
        "output_scale": "source",
        "reference_order": "original_extended",
        "numeric_scale": "mozc",
        "overlap_anchors": len(source_cost),
    }
    merge_on_source_scale(original, mozc, base, transform, metadata, 0)

    temporary = args.output.with_name(args.output.name + ".building")
    if temporary.exists():
        temporary.unlink()
    import shutil

    shutil.copyfile(base, temporary)
    supplement_reports = [
        add_supplement(temporary, args.work_directory / filename, penalty)
        for filename, penalty in DEFAULT_SUPPLEMENTS
    ]
    predictions = build_predictions(temporary)
    connection = sqlite3.connect(temporary)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    entries = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('rebuild', ?)",
        (
            json.dumps(
                {
                    **metadata,
                    "mapping_points": len(mapping["x_thresholds"]),
                    "sources": source_reports,
                    "supplements": supplement_reports,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
    )
    connection.commit()
    connection.close()
    if integrity != "ok":
        raise RuntimeError(f"dictionary integrity check failed: {integrity}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "entries": entries,
                "predictions": predictions,
                "overlap_anchors": len(source_cost),
                "integrity": integrity,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
