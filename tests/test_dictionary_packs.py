from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from ime_context_tools.dictionary.build_dictionary_packs import (
    PACK_CORE,
    PACK_EXTENDED,
    add_layer,
    attach_workspace,
    create_pack,
    create_workspace,
    ingest_layer,
    initialize_core_pack,
    source_policy,
)
from ime_context_tools.dictionary.paths import DICTIONARY_ARTIFACTS_DIR

PACK_DIRECTORY = DICTIONARY_ARTIFACTS_DIR / "packs"


def test_source_policy_keeps_core_permissive_and_extended_additive() -> None:
    assert source_policy("original:dict.bin").packs == {PACK_CORE, PACK_EXTENDED}
    assert source_policy("skk:SKK-JISYO.zipcode").packs == {PACK_EXTENDED}
    assert source_policy("skk:zipcode/zipcode/SKK-JISYO.zipcode").packs == {
        PACK_CORE,
        PACK_EXTENDED,
    }
    assert source_policy("jmdict").packs == {PACK_EXTENDED}
    assert source_policy("sudachi:notcore").packs == {PACK_CORE, PACK_EXTENDED}
    assert not source_policy("skk:SKK-JISYO.edict/SKK-JISYO.edict").packs


def test_workspace_keeps_all_sources_without_source_costs(tmp_path: Path) -> None:
    workspace = create_workspace(tmp_path / "workspace.sqlite3")
    ingest_layer(
        workspace,
        "original",
        [
            ("かな", "仮名", 300, "original:dict.bin"),
            ("かな", "仮名", 200, "original:white_dict.bin"),
        ],
    )
    assert workspace.execute(
        "SELECT cost FROM layer_terms WHERE layer='original'"
    ).fetchone() == (200,)
    assert workspace.execute(
        "SELECT source FROM layer_sources ORDER BY source"
    ).fetchall() == [("original:dict.bin",), ("original:white_dict.bin",)]
    columns = {
        row[1] for row in workspace.execute("PRAGMA table_info(layer_sources)").fetchall()
    }
    assert "cost" not in columns
    workspace.close()


def test_core_pack_merge_preserves_cost_and_unions_sources(tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace.sqlite3"
    workspace = create_workspace(workspace_path)
    original = [("かな", "仮名", 100, "original:dict.bin")]
    ingest_layer(workspace, "original", original)
    ingest_layer(workspace, "original-reference", original)
    ingest_layer(
        workspace,
        "mozc",
        [("かな", "仮名", 50, "mozc"), ("もじ", "文字", 60, "mozc")],
    )

    pack = create_pack(tmp_path / "pack.sqlite3")
    attach_workspace(pack, workspace_path)
    initialize_core_pack(pack, workspace)
    assert pack.execute(
        "SELECT source FROM entry_sources WHERE reading='かな' ORDER BY source"
    ).fetchall() == [("mozc",), ("original:dict.bin",)]
    assert pack.execute(
        "SELECT cost FROM entries WHERE reading='かな' AND candidate='仮名'"
    ).fetchone() == (50,)
    assert pack.execute(
        "SELECT candidate FROM entries WHERE reading='もじ'"
    ).fetchone() == ("文字",)
    pack.close()
    workspace.close()


def test_added_layer_is_a_true_superset(tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace.sqlite3"
    workspace = create_workspace(workspace_path)
    original = [("かな", "仮名", 100, "original:dict.bin")]
    ingest_layer(workspace, "original", original)
    ingest_layer(workspace, "original-reference", original)
    ingest_layer(workspace, "mozc", [("かな", "仮名", 50, "mozc")])
    ingest_layer(
        workspace,
        "jmdict",
        [("かな", "仮名", 500, "jmdict"), ("ついか", "追加", 600, "jmdict")],
    )

    pack = create_pack(tmp_path / "pack.sqlite3")
    attach_workspace(pack, workspace_path)
    initialize_core_pack(pack, workspace)
    before = pack.execute("SELECT cost FROM entries WHERE reading='かな'").fetchone()
    add_layer(pack, "jmdict", penalty=4_000)

    assert pack.execute("SELECT cost FROM entries WHERE reading='かな'").fetchone() == before
    assert pack.execute(
        "SELECT source FROM entry_sources WHERE reading='かな' ORDER BY source"
    ).fetchall() == [("jmdict",), ("mozc",), ("original:dict.bin",)]
    assert pack.execute(
        "SELECT candidate FROM entries WHERE reading='ついか'"
    ).fetchone() == ("追加",)
    pack.close()
    workspace.close()


@pytest.mark.skipif(
    not (PACK_DIRECTORY / PACK_EXTENDED / "dictionary.sqlite3").is_file(),
    reason="pack absent",
)
def test_generated_extended_is_a_cost_preserving_superset_of_core() -> None:
    connection = sqlite3.connect(PACK_DIRECTORY / PACK_EXTENDED / "dictionary.sqlite3")
    connection.execute(
        "ATTACH DATABASE ? AS core",
        (str(PACK_DIRECTORY / PACK_CORE / "dictionary.sqlite3"),),
    )
    assert connection.execute(
        """
        SELECT count(*) FROM core.entries AS core
        LEFT JOIN main.entries AS extended USING (reading, candidate)
        WHERE extended.reading IS NULL OR extended.cost != core.cost
        """
    ).fetchone() == (0,)
    assert connection.execute(
        """
        SELECT count(*) FROM core.entry_sources AS core
        LEFT JOIN main.entry_sources AS extended USING (reading, candidate, source)
        WHERE extended.reading IS NULL
        """
    ).fetchone() == (0,)
    connection.close()


@pytest.mark.skipif(
    not (PACK_DIRECTORY / PACK_CORE / "dictionary.sqlite3").is_file(),
    reason="pack absent",
)
def test_generated_core_has_no_copyleft_only_source() -> None:
    connection = sqlite3.connect(PACK_DIRECTORY / PACK_CORE / "dictionary.sqlite3")
    sources = {row[0] for row in connection.execute("SELECT DISTINCT source FROM entry_sources")}
    assert all(PACK_CORE in source_policy(source).packs for source in sources)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(entry_sources)")}
    assert columns == {"reading", "candidate", "source"}
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert tables == {"entries", "entry_sources", "predictions", "prediction_sources"}
    connection.close()
