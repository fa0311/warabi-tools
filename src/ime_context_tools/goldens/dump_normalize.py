"""Dump golden rows for the Rust normalization/composer functions.

Per input, two references:
  folded       NFKC -> lower -> strip -> jaconv.kata2hira
               (engine dictionary-key hygiene; the engine contract is a
               hiragana reading, so romaji passes through untouched)
  with_romaji  folded + romkan2.to_hiragana + kata2hira
               (the desktop romaji-composer chain)

Inputs: every reading in the given eval JSONL files, their romaji/katakana
round-trips, and fixed edge cases. The Rust integration test
(`crates/ime-lexicon/tests/normalize_golden.rs`) replays the dump and demands
byte-identical output.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import jaconv
import orjson
import romkan2

EDGE_CASES = [
    "",
    " ",
    "　ＫＡＥＲＵ　",
    "kaeru",
    "KAERU",
    "konnichiha",
    "kin'en",
    "kin'iro",
    "honnori",
    "nya",
    "n",
    "nn",
    "nnn",
    "nnnn",
    "xtsu",
    "xtu",
    "ltu",
    "kitte",
    "sha",
    "sya",
    "chi",
    "tchi",
    "cchi",
    "ｶｴﾙ",
    "カエル",
    "ヴァイオリン",
    "らーめん",
    "ラーメン",
    "kaeru123",
    "12かえる34",
    "ｋａｅｒｕ",
    "かえる漢字",
    "ヶヵ",
    "ゐゑ",
    "wo",
    "we",
    "wi",
    "du",
    "dzu",
    "-",
    "--",
    "a-",
    "ー",
]


def reference_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    return jaconv.kata2hira(normalized)


def reference_with_romaji(value: str) -> str:
    folded = reference_fold(value)
    return jaconv.kata2hira(romkan2.to_hiragana(folded))


def collect_inputs(eval_files: list[Path]) -> list[str]:
    inputs = list(EDGE_CASES)
    seen = set(inputs)
    for path in eval_files:
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = orjson.loads(line)
                reading = str(row.get("reading", ""))
                if not reading:
                    continue
                for variant in (
                    reading,
                    romkan2.to_roma(reading),
                    jaconv.hira2kata(reading),
                ):
                    if variant not in seen:
                        seen.add(variant)
                        inputs.append(variant)
    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    inputs = collect_inputs(args.eval_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        for raw in inputs:
            out.write(
                orjson.dumps(
                    {
                        "raw": raw,
                        "folded": reference_fold(raw),
                        "with_romaji": reference_with_romaji(raw),
                    }
                )
                + b"\n"
            )
    print(f"wrote {len(inputs)} golden pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
