from __future__ import annotations

import json
import sqlite3

import pytest

from ime_context_tools.dictionary.build_ime_dictionary import (
    normalize_reading,
    split_skk_candidates,
    strip_original_reading_metadata,
    strip_skk_annotation,
)
from ime_context_tools.dictionary.kaomoji_filter import is_kaomoji_like
from ime_context_tools.dictionary.paths import DICTIONARY_ARTIFACTS_DIR

IME = DICTIONARY_ARTIFACTS_DIR


def test_kana_normalization() -> None:
    assert normalize_reading("ヴァイオリン") == "ゔぁいおりん"
    assert normalize_reading("ＡＩ") == "AI"


def test_common_katakana_is_normalized_to_hiragana() -> None:
    assert normalize_reading("\u30ad\u30c1\u30ac\u30a4") == "\u304d\u3061\u304c\u3044"


def test_original_reading_metadata_is_removed() -> None:
    assert strip_original_reading_metadata("おはよう@reading") == "おはよう"
    assert strip_original_reading_metadata("おはよう@reading\x02twitter") == "おはよう"
    assert strip_original_reading_metadata("通常の読み") == "通常の読み"


def test_kaomoji_detection() -> None:
    assert is_kaomoji_like("(´・ω・｀)")
    assert is_kaomoji_like("｡ﾟ(ﾟ⊃Д⊂ﾟ)ﾟ｡")
    assert is_kaomoji_like("orz")
    # Exotic-script faces: glyphs FACE_MARKS cannot enumerate.
    assert is_kaomoji_like("( ¨̮ )")
    assert is_kaomoji_like("٩(๑ơ ڡơ๑)۶♥")
    assert is_kaomoji_like("ʕ•ᴥ•ʔ")
    # Unicode emoji are legitimate conversions, not kaomoji.
    assert not is_kaomoji_like("🐸")
    assert not is_kaomoji_like("おはよう☀️")
    assert not is_kaomoji_like("帰る")
    assert not is_kaomoji_like("(株)")
    assert not is_kaomoji_like("→")


def test_skk_escaping_and_annotations() -> None:
    raw = r"/候補\;字;annotation/スラッシュ\/候補/八進\073候補/"
    candidates = split_skk_candidates(raw)
    assert [strip_skk_annotation(item) for item in candidates] == [
        "候補;字",
        "スラッシュ/候補",
        "八進;候補",
    ]


def test_generated_database() -> None:
    if not (IME / "stats.json").is_file():
        pytest.skip("dictionary artifacts not built in this checkout")
    stats = json.loads((IME / "stats.json").read_text(encoding="utf-8"))
    with sqlite3.connect(IME / "ime.sqlite3") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert (
            connection.execute("SELECT count(*) FROM entries").fetchone()[0]
            == stats["total"]["unique_entries"]
        )
        assert (
            connection.execute("SELECT count(*) FROM predictions").fetchone()[0]
            == stats["total"]["unique_predictions"]
        )


def test_exact_and_prediction_layers_are_separate() -> None:
    if not (IME / "ime.sqlite3").is_file():
        pytest.skip("dictionary artifacts not built in this checkout")
    with sqlite3.connect(IME / "ime.sqlite3") as connection:
        assert connection.execute(
            "SELECT 1 FROM entries WHERE reading = ? AND candidate = ?",
            ("にほん", "日本"),
        ).fetchone()
        assert connection.execute(
            "SELECT 1 FROM predictions WHERE reading = ? AND candidate = ?",
            ("にほん", "にほんご"),
        ).fetchone()
        assert not connection.execute(
            "SELECT 1 FROM entries WHERE candidate IN ('NULL', 'null', 'None', 'undefined')"
        ).fetchone()
