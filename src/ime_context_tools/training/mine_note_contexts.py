"""Mine natural (context, reading, target) rows from nakasyou/note-articles.

The dataset ships MeCab annotations (surface + katakana reading + POS per
token), which supplies the reading information our MeCab-free stack cannot
produce itself. For every eligible token occurrence:

  context = preceding text (rebuilt from surfaces, trailing chars)
  reading = token reading, katakana -> hiragana
  target  = token surface

Eligibility: content POS (noun/verb/adjective), surface contains kanji,
reading >= --min-reading-chars, non-empty context, and the dictionary knows
(reading -> target) with at least --min-entry-candidates competing entries
(ambiguity — unambiguous conversions teach the reranker nothing).

Output rows ({context, reading, target}) feed `materialize_candidates`
unchanged. Per-article and global caps keep the sample diverse.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import orjson
from tqdm import tqdm

CONTENT_POS = {"名詞", "動詞", "形容詞"}
KATAKANA_TO_HIRAGANA = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, required=True, help="snapshot with data/*.parquet")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=400_000)
    parser.add_argument("--max-per-article", type=int, default=3)
    parser.add_argument("--context-chars", type=int, default=160)
    parser.add_argument("--min-context-chars", type=int, default=4)
    parser.add_argument("--min-reading-chars", type=int, default=2)
    parser.add_argument("--min-entry-candidates", type=int, default=2)
    parser.add_argument(
        "--proper-nouns",
        action="store_true",
        help="proper-noun mode: mine 固有名詞 tokens only and accept unique "
        "readings (min-entry-candidates 1) — unique names still compete "
        "against the raw kana copy and compound splits at runtime",
    )
    return parser


def kata2hira(reading: str) -> str:
    return reading.translate(KATAKANA_TO_HIRAGANA)


def has_kanji(surface: str) -> bool:
    return any("一" <= char <= "鿿" for char in surface)


class DictionaryFilter:
    """Cached ambiguity/coverage checks against the pack."""

    def __init__(self, pack: Path, min_entry_candidates: int) -> None:
        self.connection = sqlite3.connect(pack)
        self.min_entry_candidates = min_entry_candidates
        self.reading_cache: dict[str, bool] = {}
        self.pair_cache: dict[tuple[str, str], bool] = {}

    def reading_is_ambiguous(self, reading: str) -> bool:
        cached = self.reading_cache.get(reading)
        if cached is None:
            count = self.connection.execute(
                "SELECT COUNT(*) FROM entries WHERE reading = ?", (reading,)
            ).fetchone()[0]
            cached = count >= self.min_entry_candidates
            self.reading_cache[reading] = cached
        return cached

    def pair_exists(self, reading: str, target: str) -> bool:
        key = (reading, target)
        cached = self.pair_cache.get(key)
        if cached is None:
            cached = (
                self.connection.execute(
                    "SELECT 1 FROM entries WHERE reading = ? AND candidate = ? LIMIT 1",
                    key,
                ).fetchone()
                is not None
            )
            self.pair_cache[key] = cached
        return cached


def mine_article(
    mecab: dict,
    dictionary: DictionaryFilter,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    surfaces = mecab["surface"]
    readings = mecab["reading"]
    pos = mecab["pos"]
    pos1 = mecab.get("pos1") or [None] * len(surfaces)
    rows: list[dict[str, str]] = []
    text_so_far: list[str] = []
    length = 0
    for surface, reading, part, sub in zip(surfaces, readings, pos, pos1):
        if surface is None:
            continue
        eligible = (
            reading is not None
            and (part in CONTENT_POS or sub in CONTENT_POS)
            and has_kanji(surface)
            and length >= args.min_context_chars
        )
        if eligible:
            hira = kata2hira(reading)
            if (
                len(hira) >= args.min_reading_chars
                and hira != surface
                and dictionary.reading_is_ambiguous(hira)
                and dictionary.pair_exists(hira, surface)
            ):
                context = "".join(text_so_far)[-args.context_chars :]
                rows.append({"context": context, "reading": hira, "target": surface})
                if len(rows) >= args.max_per_article:
                    text_so_far.append(surface)
                    length += len(surface)
                    break
        text_so_far.append(surface)
        length += len(surface)
    return rows


def main(argv: list[str] | None = None) -> int:
    import pyarrow.parquet as pq

    args = build_parser().parse_args(argv)
    if args.proper_nouns:
        global CONTENT_POS
        CONTENT_POS = {"固有名詞"}
        args.min_entry_candidates = 1
    dictionary = DictionaryFilter(args.pack, args.min_entry_candidates)
    parquet_files = sorted((args.dataset_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {args.dataset_dir / 'data'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    written = 0
    articles = 0
    with args.output.open("wb") as out:
        progress = tqdm(parquet_files, desc="mine", unit="file")
        for path in progress:
            for batch in pq.ParquetFile(path).iter_batches(columns=["mecab"], batch_size=256):
                for row in batch.to_pylist():
                    articles += 1
                    for mined in mine_article(row["mecab"], dictionary, args):
                        key = (mined["context"], mined["reading"], mined["target"])
                        if key in seen:
                            continue
                        seen.add(key)
                        out.write(orjson.dumps(mined) + b"\n")
                        written += 1
                if written >= args.max_rows:
                    break
            progress.set_postfix(rows=written)
            if written >= args.max_rows:
                break

    report = {
        "rows": written,
        "articles_scanned": articles,
        "max_per_article": args.max_per_article,
        "source": "nakasyou/note-articles",
    }
    (args.output.parent / "mine_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
