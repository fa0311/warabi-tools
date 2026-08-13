#!/usr/bin/env python3
"""Build a normalized Japanese IME lexicon from third-party dictionary sources.

Outputs:
  artifacts/dictionaries/ime.sqlite3  searchable, deduplicated dictionary
  artifacts/dictionaries/merged.tsv   reading/candidate/cost/sources interchange file
  artifacts/dictionaries/predictions.tsv  predictive candidates kept separate
  artifacts/dictionaries/stats.json   accepted/rejected counts by source
"""

from __future__ import annotations

import csv
import re
import sqlite3
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from itertools import batched
from pathlib import Path

import jaconv
import orjson
from lxml import etree as ET
from tqdm import tqdm
from .paths import DICTIONARY_ARTIFACTS_DIR, DICTIONARY_SOURCES_DIR

DICT = DICTIONARY_SOURCES_DIR
OUT = DICTIONARY_ARTIFACTS_DIR
DB = OUT / "ime.sqlite3"
TSV = OUT / "merged.tsv"
PREDICTIONS_TSV = OUT / "predictions.tsv"
STATS = OUT / "stats.json"
BATCH_SIZE = 20_000

Entry = tuple[str, str, int, str]
UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
SKK_OKURI = re.compile(r"[a-zA-Z]$")
SKK_OCTAL_ESCAPE = re.compile(r"\\([0-7]{3})")
SENTINEL_CANDIDATES = {"NULL", "null", "None", "undefined"}
ORIGINAL_PREDICTION_FILES = {
    "hotpot_fourfivecost.bin.csv",
    "predict.bin.csv",
    "predict_model_dict.bin.csv",
    "relearn_dict.bin.csv",
    "search_dict.bin.csv",
    "strategy.bin.csv",
}
ORIGINAL_EMOJI_FILES = {
    "Emojiandroid.dat.csv",
    "combined_emoji_android.dat.csv",
    "emoji.bin.csv",
    "inputed_emoji_android.dat.csv",
}
ORIGINAL_AUXILIARY_FLOORS = {
    "correct.dat.csv": 9_000,
    "correct_reading.bin.csv": 9_000,
    "kana.dat.csv": 10_000,
    "kanji.bin.csv": 7_000,
    "moji_dict.bin.csv": 9_000,
    "moji_dict2.bin.csv": 9_000,
    "porn_dict.bin.csv": 9_000,
    "white_dict.bin.csv": 7_000,
}


def normalize_reading(text: str) -> str:
    text = UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    normalized = unicodedata.normalize("NFKC", text).strip()
    return jaconv.kata2hira(normalized)


def normalize_candidate(text: str) -> str:
    text = UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    return unicodedata.normalize("NFC", text).strip()


def strip_original_reading_metadata(text: str) -> str:
    """Remove the original dictionary's reading/context key suffix."""
    return text.partition("@reading")[0]


def valid_field(text: str) -> bool:
    return (
        bool(text)
        and len(text) <= 128
        and not any(unicodedata.category(char) in {"Cc", "Cs"} for char in text)
    )


def clean_entry(reading: str, candidate: str, cost: int, source: str) -> Entry | None:
    reading = normalize_reading(reading)
    candidate = normalize_candidate(candidate)
    if not valid_field(reading) or not valid_field(candidate) or candidate in SENTINEL_CANDIDATES:
        return None
    return reading, candidate, max(-32768, min(32767, int(cost))), source


def iter_mozc() -> Iterator[Entry]:
    directory = DICT / "mozc" / "dictionary_oss"
    for path in sorted(directory.glob("dictionary*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5:
                    continue
                reading, _left, _right, cost, candidate = fields
                entry = clean_entry(reading, candidate, int(cost), "mozc")
                if entry:
                    yield entry


def split_skk_candidates(text: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    escaped = False
    for char in text:
        if escaped:
            # Keep the marker until annotation parsing so escaped semicolons
            # remain distinguishable from annotation separators.
            current.extend(("\\", char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "/":
            if current:
                result.append("".join(current))
                current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    return result


def strip_skk_annotation(candidate: str) -> str:
    escaped = False
    result: list[str] = []
    for char in candidate:
        if escaped:
            if char in "01234567":
                result.extend(("\\", char))
            else:
                result.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ";":
            break
        else:
            result.append(char)
    text = "".join(result)
    return SKK_OCTAL_ESCAPE.sub(lambda m: chr(int(m.group(1), 8)), text)


def iter_skk() -> Iterator[Entry]:
    directory = DICT / "skk" / "extracted"
    for path in sorted(directory.rglob("SKK-JISYO.*")):
        if not path.is_file() or path.name.endswith((".md5", ".header", ".diff", ".orig")):
            continue
        source = f"skk:{path.relative_to(directory).as_posix()}"
        # euc_jis_2004 is a superset needed by the JIS3/4 and name files.
        with path.open(encoding="euc_jis_2004", errors="strict") as stream:
            for line in stream:
                if line.startswith(";") or " /" not in line:
                    continue
                reading, candidates = line.rstrip("\r\n").split(" ", 1)
                # SKK okuri-ari keys end in a Latin marker and contain stems,
                # not directly usable full-form IME candidates.
                if SKK_OKURI.search(reading):
                    continue
                for index, raw in enumerate(split_skk_candidates(candidates)):
                    candidate = strip_skk_annotation(raw)
                    entry = clean_entry(reading, candidate, 5000 + min(index, 999), source)
                    if entry:
                        yield entry


def priority_cost(tags: Iterable[str], default: int) -> int:
    tags = set(tags)
    if tags & {"news1", "ichi1", "spec1", "gai1"}:
        return 3200
    if tags & {"news2", "ichi2", "spec2", "gai2"}:
        return 4300
    nf = [int(tag[2:]) for tag in tags if re.fullmatch(r"nf\d\d", tag)]
    if nf:
        return 3500 + min(nf) * 45
    return default


def iter_jmdict(path: Path, source: str, default_cost: int) -> Iterator[Entry]:
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "entry":
            continue
        writings = [node.text or "" for node in elem.findall("./k_ele/keb")]
        writing_pri: dict[str, list[str]] = {
            node.findtext("keb", ""): [p.text or "" for p in node.findall("ke_pri")]
            for node in elem.findall("k_ele")
        }
        for reading_node in elem.findall("r_ele"):
            reading = reading_node.findtext("reb", "")
            restrictions = [n.text or "" for n in reading_node.findall("re_restr")]
            candidates = restrictions or writings or [reading]
            reading_pri = [p.text or "" for p in reading_node.findall("re_pri")]
            for candidate in candidates:
                cost = priority_cost(reading_pri + writing_pri.get(candidate, []), default_cost)
                entry = clean_entry(reading, candidate, cost, source)
                if entry:
                    yield entry
        elem.clear()


def iter_kanjidic() -> Iterator[Entry]:
    path = DICT / "jmdict" / "xml" / "kanjidic2.xml"
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "character":
            continue
        candidate = elem.findtext("literal", "")
        for node in elem.findall("./reading_meaning/rmgroup/reading"):
            if node.get("r_type") not in {"ja_on", "ja_kun"}:
                continue
            reading = (node.text or "").replace(".", "").replace("-", "")
            entry = clean_entry(reading, candidate, 6500, "kanjidic2")
            if entry:
                yield entry
        elem.clear()


def iter_ipadic() -> Iterator[Entry]:
    directory = DICT / "ipadic" / "source"
    for path in sorted(directory.rglob("*.csv")):
        with path.open(encoding="euc_jp", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) < 12:
                    continue
                cost = max(5000, min(15000, int(row[3])))
                entry = clean_entry(row[11], row[0], cost, "ipadic")
                if entry:
                    yield entry


def iter_sudachi() -> Iterator[Entry]:
    path = DICT / "sudachi" / "notcore_lex.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 13:
                continue
            # Sudachi's morphological connection cost is not on an IME cost
            # scale. Keep it useful as fallback vocabulary without allowing
            # zero-cost analysis entries to outrank native IME dictionaries.
            cost = max(6000, min(16000, 7000 + int(row[3]) // 4))
            entry = clean_entry(row[11], row[0], cost, "sudachi:notcore")
            if entry:
                yield entry


def calibrated_original_cost(filename: str, row: dict[str, str]) -> int:
    """Convert heterogeneous original-dictionary fields to the merged IME cost scale."""
    raw_cost = row.get("cost") or row.get("score") or row.get("cost_signed") or "6000"
    try:
        cost = int(raw_cost)
    except ValueError:
        cost = 6000
    if filename in ORIGINAL_EMOJI_FILES:
        return 18_000 + min(max(cost, 0), 10_000) // 2
    floor = ORIGINAL_AUXILIARY_FLOORS.get(filename, 4_500)
    return max(floor, min(cost, 30_000))


def iter_original(predictions: bool = False) -> Iterator[Entry]:
    excluded = {"black_dict.bin.csv"}
    for path in sorted((DICT / "original").glob("*.csv")):
        if path.name in excluded:
            continue
        if (path.name in ORIGINAL_PREDICTION_FILES) != predictions:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"reading", "surface"} <= set(reader.fieldnames):
                continue
            source = f"original:{path.stem}"
            for row in reader:
                # `cost_signed` wraps unsigned values above 32767 into negative
                # numbers. Prefer the original unsigned field. Emoji `score`
                # is not an IME word cost, so keep emoji candidates available
                # but on a separate, deliberately lower-priority scale.
                cost = calibrated_original_cost(path.name, row)
                reading = strip_original_reading_metadata(row["reading"])
                entry = clean_entry(reading, row["surface"], cost, source)
                if entry:
                    yield entry


def add_source(
    connection: sqlite3.Connection,
    label: str,
    entries: Iterable[Entry],
    stats: dict[str, dict[str, int]],
    table: str = "entries",
) -> None:
    if table not in {"entries", "predictions"}:
        raise ValueError(f"unsupported table: {table}")
    sql = f"""
        INSERT INTO {table}(reading, candidate, cost, sources)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(reading, candidate) DO UPDATE SET
            cost = min({table}.cost, excluded.cost),
            sources = CASE
                WHEN instr(',' || {table}.sources || ',', ',' || excluded.sources || ',') > 0
                    THEN {table}.sources
                ELSE {table}.sources || ',' || excluded.sources
            END
    """
    accepted = 0
    progress = tqdm(desc=label, unit=" entries", unit_scale=True)
    for batch in batched(entries, BATCH_SIZE):
        connection.executemany(sql, batch)
        accepted += len(batch)
        progress.update(len(batch))
    progress.close()
    connection.commit()
    stats[label] = {"accepted": accepted}
    total = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"{label}: {accepted:,} accepted; {total:,} unique total", flush=True)


def export_table(connection: sqlite3.Connection, table: str, target: Path) -> None:
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("reading", "candidate", "cost", "sources"))
        cursor = connection.execute(
            f"SELECT reading, candidate, cost, sources FROM {table} "
            "ORDER BY reading, cost, candidate"
        )
        writer.writerows(cursor)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    connection = sqlite3.connect(DB)
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
        CREATE TABLE predictions (
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            cost INTEGER NOT NULL,
            sources TEXT NOT NULL,
            PRIMARY KEY (reading, candidate)
        ) WITHOUT ROWID;
        """
    )
    stats: dict[str, dict[str, int]] = {}
    sources: list[tuple[str, Iterable[Entry]]] = [
        ("mozc", iter_mozc()),
        ("skk", iter_skk()),
        ("jmdict", iter_jmdict(DICT / "jmdict" / "xml" / "JMdict_e", "jmdict", 6000)),
        ("jmnedict", iter_jmdict(DICT / "jmdict" / "xml" / "JMnedict.xml", "jmnedict", 6800)),
        ("kanjidic2", iter_kanjidic()),
        ("ipadic", iter_ipadic()),
        ("sudachi", iter_sudachi()),
        ("original", iter_original(predictions=False)),
    ]
    try:
        for label, entries in sources:
            add_source(connection, label, entries, stats)
        add_source(
            connection,
            "original_predictions",
            iter_original(predictions=True),
            stats,
            table="predictions",
        )
        connection.execute("CREATE INDEX entries_reading_cost ON entries(reading, cost, candidate)")
        connection.execute(
            "CREATE INDEX predictions_reading_cost ON predictions(reading, cost, candidate)"
        )
        connection.commit()
        total = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
        prediction_total = connection.execute("SELECT count(*) FROM predictions").fetchone()[0]
        export_table(connection, "entries", TSV)
        export_table(connection, "predictions", PREDICTIONS_TSV)
        stats["total"] = {
            "unique_entries": total,
            "unique_predictions": prediction_total,
        }
        STATS.write_bytes(
            orjson.dumps(
                stats,
                option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
            )
        )
        print(
            f"done: {total:,} unique entries; {prediction_total:,} unique predictions",
            flush=True,
        )
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
