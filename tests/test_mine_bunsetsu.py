import argparse
import sqlite3

from ime_context_tools.training.mine_note_contexts import (
    DictionaryFilter,
    mine_article_bunsetsu,
)


def make_dictionary(tmp_path, pairs):
    path = tmp_path / "pack.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE entries (reading TEXT, candidate TEXT, cost INTEGER)")
    connection.executemany(
        "INSERT INTO entries VALUES (?, ?, 1000)", [(r, c) for r, c in pairs]
    )
    connection.commit()
    connection.close()
    return DictionaryFilter(path, min_entry_candidates=1)


def args(**overrides):
    defaults = dict(
        min_context_chars=4, min_reading_chars=2, context_chars=160, max_per_article=5
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def mecab(rows):
    surfaces, readings, pos, pos1 = zip(*rows)
    return {
        "surface": list(surfaces),
        "reading": list(readings),
        "pos": list(pos),
        "pos1": list(pos1),
    }


SENTENCE = [
    ("私", "ワタシ", "名詞", "代名詞"),
    ("の", "ノ", "助詞", None),
    ("将来", "ショウライ", "名詞", None),
    ("の", "ノ", "助詞", None),
    ("夢", "ユメ", "名詞", None),
    ("は", "ハ", "助詞", None),
    ("医者", "イシャ", "名詞", None),
    ("に", "ニ", "助詞", None),
    ("なる", "ナル", "動詞", None),
    ("こと", "コト", "名詞", "非自立"),
    ("です", "デス", "助動詞", None),
]


def test_joins_content_word_with_trailing_particles(tmp_path):
    dictionary = make_dictionary(tmp_path, [("ゆめ", "夢"), ("いしゃ", "医者")])
    rows = mine_article_bunsetsu(mecab(SENTENCE), dictionary, args())
    assert [(row["reading"], row["target"]) for row in rows] == [
        ("ゆめは", "夢は"),
        ("いしゃに", "医者に"),
    ]
    # context is the raw preceding text
    assert rows[1]["context"] == "私の将来の夢は"


def test_proper_noun_heads_are_skipped(tmp_path):
    dictionary = make_dictionary(tmp_path, [("とうきょう", "東京")])
    sentence = [
        ("昨日", "キノウ", "名詞", None),
        ("から", "カラ", "助詞", None),
        ("東京", "トウキョウ", "名詞", "固有名詞"),
        ("に", "ニ", "助詞", None),
    ]
    assert mine_article_bunsetsu(mecab(sentence), dictionary, args()) == []


def test_bare_content_words_are_not_emitted(tmp_path):
    dictionary = make_dictionary(tmp_path, [("ゆめ", "夢")])
    sentence = [
        ("私", "ワタシ", "名詞", "代名詞"),
        ("の", "ノ", "助詞", None),
        ("大きな", "オオキナ", "形容詞", None),
        ("夢", "ユメ", "名詞", None),
    ]
    # 夢 has no trailing particle -> token-mode territory, not a bunsetsu row
    assert mine_article_bunsetsu(mecab(sentence), dictionary, args()) == []


def test_katakana_heads_are_convertible(tmp_path):
    dictionary = make_dictionary(tmp_path, [("あめりかん", "アメリカン")])
    sentence = [
        ("とても", "トテモ", "副詞", None),
        ("素敵", "ステキ", "名詞", None),
        ("アメリカン", "アメリカン", "名詞", None),
        ("な", "ナ", "助動詞", None),
    ]
    rows = mine_article_bunsetsu(mecab(sentence), dictionary, args())
    assert [(row["reading"], row["target"]) for row in rows] == [("あめりかんな", "アメリカンな")]
