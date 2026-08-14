"""Build the character-level tokenizer shared by all distilled student rerankers.

The teacher (`ime-context-reranker-v1`) tokenizes with MeCab + SentencePiece,
which has no clean Rust port. Students instead use one character = one token so
the Rust runtime only needs a `HashMap<char, u32>` with `[UNK]` fallback.

Vocabulary layout (vocab.txt, line number - 1 == token id):
  0..4   [PAD] [UNK] [CLS] [SEP] [MASK]        (fixed ids, teacher-compatible)
  5..    every remaining character, in codepoint order — ids are arbitrary to
         the model, and codepoint order groups the file by Unicode block so a
         human can audit it. (Selection into the vocab is by corpus frequency;
         only the final ordering is codepoint.)

The guaranteed base set (always included, even at zero corpus frequency) is
every character encodable as strict Shift_JIS: ASCII printable, halfwidth
katakana, and the full JIS X 0208 repertoire (all level 1+2 kanji, kana,
punctuation). Remaining slots up to --vocab-size are filled by corpus
frequency.

Tokenization contract (identical in Python and Rust):
  1. NO normalization. NFKC would merge fullwidth/halfwidth variants
     (１ vs 1, Ａ vs A) that are distinct IME candidates. This is why the
     emitted artifact is a `tokenizers` fast tokenizer (WordLevel over single
     characters) instead of BertJapaneseTokenizer, whose CharacterTokenizer
     hard-codes NFKC.
  2. Characters with the Unicode White_Space property are dropped
     (`WhitespaceSplit` here == `char::is_whitespace()` in Rust).
  3. Every remaining character maps to its vocab id, or to [UNK].
  4. Model input is three segments assembled programmatically:
     `[CLS] context [SEP] reading [SEP] candidate [SEP]`
     (the emitted fast tokenizer carries no post-processor; callers encode
     each segment with add_special_tokens=False and splice the ids).

UNK statistics are split: `natural` (context/target/reading text — the gate)
vs `enumerated` (candidate lists and dictionary surface forms, where synthetic
data enumerates rare out-of-JIS kanji; UNK there merely blinds the reranker to
those candidates, whose ranking stays dictionary-cost-driven).

Outputs: vocab.txt (Rust side), tokenizer.json + tokenizer_config.json +
special_tokens_map.json (HF fast tokenizer, loadable via AutoTokenizer with no
remote code), build_report.json (coverage stats).
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm
from transformers import AutoTokenizer

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Unicode White_Space property, mirroring Rust's char::is_whitespace() and the
# `tokenizers` WhitespaceSplit pre-tokenizer. Python's str.isspace() differs
# (it also matches U+001C..U+001F), so the set is spelled out explicitly.
WHITESPACE_CHARS = frozenset(
    chr(codepoint)
    for codepoint in [
        *range(0x0009, 0x000E),
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    ]
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        action="append",
        default=[],
        help="directory containing train.jsonl/dev.jsonl candidate rows (repeatable)",
    )
    parser.add_argument(
        "--dictionary-tsv",
        type=Path,
        action="append",
        default=[],
        help="reading<TAB>candidate<TAB>... TSV whose first two columns feed the corpus",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--max-unk-rate",
        type=float,
        default=0.0005,
        help=(
            "fail if the NATURAL-text UNK character rate exceeds this fraction. "
            "The budget tolerates corpus noise (unassigned codepoints, bidi/ZW "
            "format chars, stray foreign scripts) that correctly maps to [UNK]; "
            "kana/JIS-kanji coverage is guaranteed by the base set regardless."
        ),
    )
    return parser


def is_token_char(char: str) -> bool:
    """True when `char` may occupy a vocabulary slot.

    White_Space characters are dropped by the pre-tokenizer and must never be
    tokens. Control/format characters (category C*) are excluded from the
    vocabulary too; at runtime they tokenize to [UNK] on both sides.
    """
    if char in WHITESPACE_CHARS:
        return False
    return not unicodedata.category(char).startswith("C")


def guaranteed_chars() -> set[str]:
    """Characters always granted a vocab slot.

    Strict Shift_JIS-encodable characters (ASCII + halfwidth kana + JIS X
    0208) plus the full hiragana/katakana blocks — the latter cover combining
    voicing marks (U+3099/309A, produced by NFD-decomposed input) and small
    kana (ゕゖ) that Shift_JIS lacks.
    """
    chars: set[str] = set()
    for codepoint in range(0x20, 0x10000):
        char = chr(codepoint)
        if not is_token_char(char):
            continue
        try:
            char.encode("shift_jis")
        except UnicodeEncodeError:
            continue
        chars.add(char)
    for codepoint in range(0x3041, 0x3100):  # hiragana + katakana blocks
        char = chr(codepoint)
        if is_token_char(char) and unicodedata.category(char) != "Cn":
            chars.add(char)
    return chars


def count_candidate_dir(directory: Path, natural: Counter[str], enumerated: Counter[str]) -> None:
    for name in ("train.jsonl", "dev.jsonl"):
        path = directory / name
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row: dict[str, Any] = orjson.loads(line)
                natural.update(str(row.get("context", "")))
                natural.update(str(row.get("reading", "")))
                natural.update(str(row.get("target", "")))
                for candidate in row.get("candidates", []):
                    enumerated.update(str(candidate.get("text", "")))


def count_dictionary_tsv(path: Path, natural: Counter[str], enumerated: Counter[str]) -> None:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if columns:
                natural.update(columns[0])  # readings are kana: natural input text
            if len(columns) > 1:
                enumerated.update(columns[1])


def build_vocab(
    natural: Counter[str],
    enumerated: Counter[str] | None = None,
    *,
    vocab_size: int,
    base_chars: set[str] | None = None,
) -> list[str]:
    """Return the full vocab (specials first) of at most `vocab_size` entries.

    Free slots are filled by natural-text frequency first (that is the UNK
    gate), falling back to enumerated (candidate-list/dictionary) frequency;
    dictionary-only rare kanji must not crowd out characters people type.
    """
    enumerated = enumerated or Counter()
    base = guaranteed_chars() if base_chars is None else base_chars
    base = {char for char in base if is_token_char(char)}
    seen = {char for char in natural | enumerated if is_token_char(char)}
    slots = vocab_size - len(SPECIAL_TOKENS)
    if len(base) > slots:
        raise ValueError(f"guaranteed base set ({len(base)}) exceeds available slots ({slots})")

    def sort_key(char: str) -> tuple[int, int, int]:
        return (-natural.get(char, 0), -enumerated.get(char, 0), ord(char))

    ordered = sorted(seen | base, key=sort_key)
    selected = set(base)
    for char in ordered:
        if len(selected) >= slots:
            break
        selected.add(char)
    # Codepoint order: token ids carry no meaning, and block-grouped output
    # keeps vocab.txt reviewable by eye. NOTE: any reorder is id-breaking for
    # existing weights — this only shapes future builds.
    return SPECIAL_TOKENS + sorted(selected, key=ord)


def unk_statistics(corpus_counts: Counter[str], vocab: list[str]) -> dict[str, Any]:
    known = set(vocab)
    total = 0
    unknown = 0
    unknown_examples: Counter[str] = Counter()
    for char, count in corpus_counts.items():
        if char in WHITESPACE_CHARS:
            continue
        total += count
        if char not in known:
            unknown += count
            unknown_examples[char] += count
    return {
        "total_chars": total,
        "unk_chars": unknown,
        "unk_rate": (unknown / total) if total else 0.0,
        "top_unk": [
            {"char": char, "codepoint": f"U+{ord(char):04X}", "count": count}
            for char, count in unknown_examples.most_common(20)
        ],
    }


def build_fast_tokenizer(vocab: list[str]):
    """Assemble the `tokenizers` object implementing the tokenization contract.

    No post-processor: segments are encoded with add_special_tokens=False and
    spliced into `[CLS] context [SEP] reading [SEP] candidate [SEP]` by the
    caller (see `encode_example` in the trainer).
    """
    from tokenizers import Regex, Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel

    vocab_map = {token: index for index, token in enumerate(vocab)}
    tokenizer = Tokenizer(WordLevel(vocab_map, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated"),
        ]
    )
    return tokenizer


def write_tokenizer(output_dir: Path, vocab: list[str]) -> None:
    from transformers import PreTrainedTokenizerFast

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8", newline="\n")
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=build_fast_tokenizer(vocab),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=256,
    )
    wrapped.save_pretrained(output_dir)


def verify_roundtrip(output_dir: Path) -> None:
    """Load through transformers and assert the tokenization contract holds."""

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    for index, token in enumerate(SPECIAL_TOKENS):
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != index:
            raise AssertionError(f"{token} id {actual} != {index}")
    probe = "Ｇｏ言語1位　です\nかえる"
    encoded = tokenizer(probe, add_special_tokens=False)
    decoded = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    expected = [char for char in probe if char not in WHITESPACE_CHARS]
    if len(decoded) != len(expected):
        raise AssertionError(f"length mismatch: {len(decoded)} tokens vs {len(expected)} chars")
    # NFKC must NOT happen: fullwidth Ｇ stays fullwidth.
    if "Ｇ" in tokenizer.get_vocab() and "Ｇ" not in decoded:
        raise AssertionError("fullwidth character was normalized away")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.candidates_dir and not args.dictionary_tsv:
        raise SystemExit("provide at least one --candidates-dir or --dictionary-tsv")

    natural: Counter[str] = Counter()
    enumerated: Counter[str] = Counter()
    for directory in tqdm(args.candidates_dir, desc="candidate dirs", unit="dir"):
        count_candidate_dir(directory, natural, enumerated)
    for path in tqdm(args.dictionary_tsv, desc="dictionary tsv", unit="file"):
        count_dictionary_tsv(path, natural, enumerated)

    combined = natural + enumerated
    vocab = build_vocab(natural, enumerated, vocab_size=args.vocab_size)
    natural_stats = unk_statistics(natural, vocab)
    enumerated_stats = unk_statistics(enumerated, vocab)
    write_tokenizer(args.output_dir, vocab)
    verify_roundtrip(args.output_dir)

    report = {
        "schema_version": "1.1.0",
        "tokenizer_type": "character",
        "vocab_size": len(vocab),
        "special_tokens": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        "normalization": "none",
        "whitespace_rule": "drop Unicode White_Space (Rust char::is_whitespace)",
        "unknown_rule": "all other unmapped chars (incl. control/format) -> [UNK]",
        "input_template": "[CLS] context [SEP] reading [SEP] candidate [SEP]",
        "guaranteed_base": "shift_jis-encodable (ASCII + halfwidth kana + JIS X 0208)",
        "sources": {
            "candidates_dirs": [str(path) for path in args.candidates_dir],
            "dictionary_tsv": [str(path) for path in args.dictionary_tsv],
        },
        "corpus_distinct_chars": len(combined),
        "natural": natural_stats,
        "enumerated": enumerated_stats,
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "vocab_size": len(vocab),
        "corpus_distinct_chars": len(combined),
        "natural_unk_rate": natural_stats["unk_rate"],
        "enumerated_unk_rate": enumerated_stats["unk_rate"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    if natural_stats["unk_rate"] > args.max_unk_rate:
        print(
            f"natural UNK rate {natural_stats['unk_rate']:.6f} exceeds "
            f"--max-unk-rate {args.max_unk_rate}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
