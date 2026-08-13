from collections import Counter

import pytest

from ime_context_tools.training.build_char_tokenizer import (
    SPECIAL_TOKENS,
    build_vocab,
    guaranteed_chars,
    is_token_char,
    unk_statistics,
    verify_roundtrip,
    write_tokenizer,
)


def test_special_tokens_have_fixed_ids() -> None:
    vocab = build_vocab(Counter("あいう"), vocab_size=64, base_chars=set("あいう"))
    assert vocab[:5] == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    assert SPECIAL_TOKENS == vocab[:5]


def test_whitespace_and_controls_are_never_tokens() -> None:
    for char in ("\n", "\t", "\r", " ", "　", "​", " "):
        assert not is_token_char(char)
    for char in ("あ", "亜", "ｱ", "Ａ", "a", "🍣", "、"):
        assert is_token_char(char)


def test_fullwidth_halfwidth_stay_distinct() -> None:
    corpus = Counter("１1ＡAｱア")
    vocab = build_vocab(corpus, vocab_size=64, base_chars=set())
    body = set(vocab[5:])
    assert {"１", "1", "Ａ", "A", "ｱ", "ア"} <= body


def test_frequency_fills_remaining_slots() -> None:
    corpus = Counter({"あ": 100, "い": 50, "🍣": 10, "🍺": 1})
    vocab = build_vocab(corpus, vocab_size=5 + 3, base_chars=set("あ"))
    body = vocab[5:]
    assert body == ["あ", "い", "🍣"]  # 🍺 loses the last slot


def test_guaranteed_base_includes_jis_kanji_and_halfwidth_kana() -> None:
    base = guaranteed_chars()
    assert {"亜", "腕", "熙", "ｱ", "ﾝ", "、", "。", "Ａ", "ー"} <= base
    # JIS X 0208 kanji levels 1+2 plus non-kanji rows: ~7,000 characters
    assert 6500 < len(base) < 7500


def test_unk_statistics_counts_out_of_vocab_chars() -> None:
    corpus = Counter({"あ": 9999, "🍣": 1})
    vocab = build_vocab(corpus, vocab_size=5 + 1, base_chars=set("あ"))
    stats = unk_statistics(corpus, vocab)
    assert stats["unk_chars"] == 1
    assert stats["unk_rate"] == pytest.approx(1 / 10000)


def test_unk_statistics_ignores_whitespace() -> None:
    corpus = Counter({"あ": 5, "\n": 100, " ": 100})
    vocab = build_vocab(corpus, vocab_size=5 + 1, base_chars=set("あ"))
    stats = unk_statistics(corpus, vocab)
    assert stats["total_chars"] == 5
    assert stats["unk_chars"] == 0


def test_astral_chars_do_not_crash() -> None:
    corpus = Counter({"あ": 5, "𠮷": 3})  # U+20BB7, outside the BMP
    vocab = build_vocab(corpus, vocab_size=64, base_chars=set())
    assert "𠮷" in vocab


def test_written_tokenizer_loads_via_transformers(tmp_path) -> None:
    corpus = Counter("家にかえる帰るテスト１1Ｇｏ言語位です")
    vocab = build_vocab(corpus, vocab_size=256, base_chars=set())
    write_tokenizer(tmp_path, vocab)
    verify_roundtrip(tmp_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    # Segments are encoded bare; the caller splices [CLS]/[SEP] ids.
    context = tokenizer("家に　", add_special_tokens=False)["input_ids"]
    reading = tokenizer("かえる", add_special_tokens=False)["input_ids"]
    candidate = tokenizer("帰る", add_special_tokens=False)["input_ids"]
    assembled = [2, *context, 3, *reading, 3, *candidate, 3]
    tokens = tokenizer.convert_ids_to_tokens(assembled)
    assert tokens == ["[CLS]", "家", "に", "[SEP]", "か", "え", "る", "[SEP]", "帰", "る", "[SEP]"]


def test_no_nfkc_normalization(tmp_path) -> None:
    corpus = Counter("１1ＡAｱア")
    vocab = build_vocab(corpus, vocab_size=64, base_chars=set())
    write_tokenizer(tmp_path, vocab)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    ids_full = tokenizer("１Ａｱ", add_special_tokens=False)["input_ids"]
    ids_half = tokenizer("1Aア", add_special_tokens=False)["input_ids"]
    assert set(ids_full).isdisjoint(set(ids_half))
