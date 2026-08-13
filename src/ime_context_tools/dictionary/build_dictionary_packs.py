from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import batched, chain
from pathlib import Path

import numpy as np
from .paths import ARTIFACTS_DIR, DICTIONARY_ARTIFACTS_DIR, PROJECT_ROOT

from .build_ime_dictionary import (
    DICT,
    Entry,
    clean_entry,
    iter_ipadic,
    iter_jmdict,
    iter_kanjidic,
    iter_mozc,
    iter_original,
    iter_skk,
    iter_sudachi,
)
from .calibrate_dictionary_sources import isotonic_mapping
from .kaomoji_filter import is_kaomoji_like

PACK_CORE = "ime-dictionary-core"
PACK_EXTENDED = "ime-dictionary-extended-gpl"
PACKS = frozenset({PACK_CORE, PACK_EXTENDED})
BATCH_SIZE = 20_000

SKK_PUBLIC_DOMAIN = frozenset(
    {
        "skk:SKK-JISYO.itaiji",
        "skk:SKK-JISYO.okinawa",
        "skk:zipcode/zipcode/SKK-JISYO.office.zipcode",
        "skk:zipcode/zipcode/SKK-JISYO.zipcode",
    }
)
SKK_EXCLUDED = frozenset({"skk:SKK-JISYO.edict/SKK-JISYO.edict"})
ORIGINAL_REFERENCE_SOURCES = frozenset(
    {
        "original:dict.bin",
        "original:kernel_dict.bin",
        "original:white_dict.bin",
    }
)
# Kaomoji collection files; the Unicode-emoji dat files are NOT here —
# emoji are legitimate conversions and stay unflagged.
KAOMOJI_SOURCES = frozenset(
    {
        "original:moji_dict.bin",
        "original:moji_dict2.bin",
    }
)


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    license: str
    packs: frozenset[str]
    url: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CostMapping:
    x: tuple[float, ...]
    y: tuple[float, ...]

    def __call__(self, value: int) -> int:
        if not self.x:
            return value
        if value <= self.x[0]:
            return round(self.y[0])
        if value >= self.x[-1]:
            return round(self.y[-1])
        index = bisect_right(self.x, value)
        x0, x1 = self.x[index - 1], self.x[index]
        y0, y1 = self.y[index - 1], self.y[index]
        if x0 == x1:
            return round(y0)
        return round(y0 + (value - x0) * (y1 - y0) / (x1 - x0))

    @classmethod
    def identity(cls) -> CostMapping:
        return cls((), ())


@dataclass(frozen=True, slots=True)
class Layer:
    name: str
    entries: Iterable[Entry]


def source_policy(source: str) -> SourcePolicy:
    if source.startswith("original:"):
        return SourcePolicy(
            "GPL-3.0-only OR CC-BY-4.0",
            PACKS,
            "",
            "Project-authored dictionary data",
        )
    if source == "mozc":
        return SourcePolicy(
            "NAIST-2003 AND ICOT-terms AND Public-Domain",
            PACKS,
            "https://github.com/google/mozc/tree/master/src/data/dictionary_oss",
            "Retain the complete upstream dictionary notices",
        )
    if source == "ipadic":
        return SourcePolicy(
            "NAIST-2003 AND ICOT-terms",
            PACKS,
            "https://taku910.github.io/mecab/",
        )
    if source == "sudachi:notcore":
        return SourcePolicy(
            "Apache-2.0",
            PACKS,
            "https://github.com/WorksApplications/SudachiDict",
            (
                "notcore_lex.csv SHA-256 "
                "c1f6b64c7bae2d4d8411f2c523dbcfe7be9c30549a2ad48bccd31591ef446497; "
                "includes the upstream LEGAL notices for UniDic and NEologd-derived data"
            ),
        )
    if source in {"jmdict", "jmnedict", "kanjidic2"}:
        return SourcePolicy(
            "CC-BY-SA-4.0",
            frozenset({PACK_EXTENDED}),
            "https://www.edrdg.org/edrdg/licence.html",
            "Attribution and regular-update procedure required",
        )
    if source in SKK_EXCLUDED:
        return SourcePolicy(
            "CC-BY-SA-3.0",
            frozenset(),
            "https://skk-dev.github.io/dict/",
            "Excluded because current JMdict already supersedes it",
        )
    if source in SKK_PUBLIC_DOMAIN:
        return SourcePolicy(
            "Public-Domain",
            PACKS,
            "https://skk-dev.github.io/dict/",
        )
    if source.startswith("skk:"):
        return SourcePolicy(
            "GPL-2.0-or-later",
            frozenset({PACK_EXTENDED}),
            "https://skk-dev.github.io/dict/",
        )
    raise ValueError(f"unregistered dictionary source: {source}")


def _filtered_skk(pack: str, *, added_only: bool = False) -> Iterator[Entry]:
    for entry in iter_skk():
        packs = source_policy(entry[3]).packs
        if pack not in packs or (added_only and PACK_CORE in packs):
            continue
        yield entry


def iter_sensitive() -> Iterator[Entry]:
    path = DICT / "original" / "black_dict.bin.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            entry = clean_entry(
                row["reading"], row["surface"], int(row["cost"]), "original:sensitive"
            )
            if entry is not None:
                yield entry


def source_layers() -> list[Layer]:
    return [
        Layer("original", chain(iter_original(predictions=False), iter_sensitive())),
        Layer(
            "original-reference",
            (
                entry
                for entry in iter_original(predictions=False)
                if entry[3] in ORIGINAL_REFERENCE_SOURCES
            ),
        ),
        Layer("mozc", iter_mozc()),
        Layer("ipadic", iter_ipadic()),
        Layer("sudachi", iter_sudachi()),
        Layer("skk-core", _filtered_skk(PACK_CORE)),
        Layer(
            "jmdict",
            iter_jmdict(DICT / "jmdict" / "xml" / "JMdict_e", "jmdict", 6_000),
        ),
        Layer("skk-extended", _filtered_skk(PACK_EXTENDED, added_only=True)),
        Layer(
            "jmnedict",
            iter_jmdict(DICT / "jmdict" / "xml" / "JMnedict.xml", "jmnedict", 6_800),
        ),
        Layer("kanjidic2", iter_kanjidic()),
        Layer("predictions", iter_original(predictions=True)),
    ]


def create_workspace(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE layer_terms (
            layer TEXT NOT NULL,
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            cost INTEGER NOT NULL,
            PRIMARY KEY (layer, reading, candidate)
        ) WITHOUT ROWID;
        CREATE TABLE layer_sources (
            layer TEXT NOT NULL,
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (layer, reading, candidate, source)
        ) WITHOUT ROWID;
        """
    )
    return connection


def ingest_layer(connection: sqlite3.Connection, layer: str, entries: Iterable[Entry]) -> int:
    term_sql = """
        INSERT INTO layer_terms(layer, reading, candidate, cost) VALUES (?, ?, ?, ?)
        ON CONFLICT(layer, reading, candidate) DO UPDATE SET
            cost = min(layer_terms.cost, excluded.cost)
    """
    source_sql = """
        INSERT OR IGNORE INTO layer_sources(layer, reading, candidate, source)
        VALUES (?, ?, ?, ?)
    """
    count = 0
    for batch in batched(entries, BATCH_SIZE):
        terms = [(layer, reading, candidate, cost) for reading, candidate, cost, _source in batch]
        sources = [
            (layer, reading, candidate, source) for reading, candidate, _cost, source in batch
        ]
        connection.executemany(term_sql, terms)
        connection.executemany(source_sql, sources)
        count += len(batch)
    connection.commit()
    return count


def materialize_workspace(path: Path, *, reuse: bool) -> sqlite3.Connection:
    if reuse and path.is_file():
        connection = sqlite3.connect(path)
        for old, new in (("skk-b", "skk-core"), ("skk-a", "skk-extended")):
            connection.execute(
                "UPDATE layer_terms SET layer = ? WHERE layer = ?", (new, old)
            )
            connection.execute(
                "UPDATE layer_sources SET layer = ? WHERE layer = ?", (new, old)
            )
        connection.commit()
        expected = {layer.name for layer in source_layers()}
        present = {
            row[0] for row in connection.execute("SELECT DISTINCT layer FROM layer_terms")
        }
        has_sensitive = connection.execute(
            "SELECT 1 FROM layer_sources WHERE layer='original' AND source='original:sensitive'"
        ).fetchone()
        if expected <= present and has_sensitive:
            print(f"reuse workspace: {path}", flush=True)
            return connection
        connection.close()
    connection = create_workspace(path)
    for layer in source_layers():
        print(f"ingest layer: {layer.name}", flush=True)
        count = ingest_layer(connection, layer.name, layer.entries)
        print(f"ingested layer: {layer.name} ({count:,} observations)", flush=True)
    connection.execute("PRAGMA optimize")
    return connection


def _mapping_from_pairs(pairs: Iterable[tuple[int, int]], method: str) -> tuple[CostMapping, int]:
    values = np.asarray(list(pairs), dtype=np.float64)
    if not len(values):
        raise ValueError("cost calibration requires at least one overlap anchor")
    source_cost = values[:, 0]
    reference_cost = values[:, 1]
    if method == "quantile":
        quantiles = np.linspace(0.0, 1.0, 257)
        x_points = np.quantile(source_cost, quantiles)
        y_points = np.quantile(reference_cost, quantiles)
        unique_x, indices = np.unique(x_points, return_index=True)
        mapping = CostMapping(tuple(unique_x), tuple(y_points[indices]))
    elif method == "isotonic":
        _transform, metadata = isotonic_mapping(source_cost, reference_cost)
        mapping = CostMapping(tuple(metadata["x_thresholds"]), tuple(metadata["y_thresholds"]))
    else:
        raise ValueError(f"unknown calibration method: {method}")
    return mapping, len(values)


def learn_layer_mapping(
    workspace: sqlite3.Connection, reference: str, source: str, method: str
) -> tuple[CostMapping, int]:
    pairs = workspace.execute(
        """
        SELECT source.cost, reference.cost
        FROM layer_terms AS source
        JOIN layer_terms AS reference USING (reading, candidate)
        WHERE source.layer = ? AND reference.layer = ?
        """,
        (source, reference),
    )
    return _mapping_from_pairs(pairs, method)


def create_pack(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE entries (
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            cost INTEGER NOT NULL,
            kaomoji INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (reading, candidate)
        ) WITHOUT ROWID;
        CREATE TABLE entry_sources (
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (reading, candidate, source),
            FOREIGN KEY (reading, candidate) REFERENCES entries(reading, candidate)
        ) WITHOUT ROWID;
        CREATE TABLE predictions (
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            cost INTEGER NOT NULL,
            kaomoji INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (reading, candidate)
        ) WITHOUT ROWID;
        CREATE TABLE prediction_sources (
            reading TEXT NOT NULL,
            candidate TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (reading, candidate, source),
            FOREIGN KEY (reading, candidate) REFERENCES predictions(reading, candidate)
        ) WITHOUT ROWID;
        """
    )
    return connection


def attach_workspace(pack: sqlite3.Connection, workspace_path: Path) -> None:
    pack.execute("ATTACH DATABASE ? AS workspace", (str(workspace_path),))


def _insert_mapped_rows(
    pack: sqlite3.Connection,
    rows: Iterable[tuple[str, str, int]],
    mapping: CostMapping,
    penalty: int,
) -> int:
    count = 0
    cursor = iter(rows)
    for batch in batched(cursor, BATCH_SIZE):
        pack.executemany(
            "INSERT INTO entries(reading, candidate, cost) VALUES (?, ?, ?)",
            ((reading, candidate, mapping(cost) + penalty) for reading, candidate, cost in batch),
        )
        count += len(batch)
    return count


def initialize_core_pack(
    pack: sqlite3.Connection, workspace: sqlite3.Connection
) -> tuple[int, dict[str, int]]:
    original_mapping, anchors = learn_layer_mapping(
        workspace, "mozc", "original-reference", "quantile"
    )
    rows = workspace.execute(
        "SELECT reading, candidate, cost FROM layer_terms WHERE layer = 'original'"
    )
    inserted = _insert_mapped_rows(pack, rows, original_mapping, 0)
    pack.execute(
        """
        INSERT INTO entry_sources
        SELECT reading, candidate, source FROM workspace.layer_sources
        WHERE layer = 'original'
        """
    )
    add_layer(pack, "mozc", CostMapping.identity(), 0)
    return inserted, {"base_overlap_anchors": anchors}


def learn_pack_mapping(
    pack: sqlite3.Connection, layer: str
) -> tuple[CostMapping, int]:
    pairs = pack.execute(
        """
        SELECT source.cost, reference.cost
        FROM workspace.layer_terms AS source
        JOIN entries AS reference USING (reading, candidate)
        WHERE source.layer = ?
        """,
        (layer,),
    )
    return _mapping_from_pairs(pairs, "isotonic")


def add_layer(
    pack: sqlite3.Connection,
    layer: str,
    mapping: CostMapping | None = None,
    penalty: int = 0,
) -> dict[str, int]:
    if mapping is None:
        mapping, anchors = learn_pack_mapping(pack, layer)
    else:
        anchors = pack.execute(
            """
            SELECT count(*) FROM workspace.layer_terms AS source
            JOIN entries USING (reading, candidate)
            WHERE source.layer = ?
            """,
            (layer,),
        ).fetchone()[0]
    rows = pack.execute(
        """
        SELECT source.reading, source.candidate, source.cost
        FROM workspace.layer_terms AS source
        WHERE source.layer = ?
          AND NOT EXISTS (
              SELECT 1 FROM entries
              WHERE entries.reading = source.reading
                AND entries.candidate = source.candidate
          )
        """,
        (layer,),
    )
    added = _insert_mapped_rows(pack, rows, mapping, penalty)
    pack.execute(
        """
        INSERT OR IGNORE INTO entry_sources
        SELECT source.reading, source.candidate, source.source
        FROM workspace.layer_sources AS source
        JOIN entries USING (reading, candidate)
        WHERE source.layer = ?
        """,
        (layer,),
    )
    pack.commit()
    return {"added": added, "overlap_anchors": int(anchors)}


def add_predictions(pack: sqlite3.Connection) -> None:
    pack.executescript(
        """
        INSERT INTO predictions(reading, candidate, cost)
        SELECT reading, candidate, cost FROM workspace.layer_terms
        WHERE layer = 'predictions';
        INSERT INTO prediction_sources
        SELECT reading, candidate, source FROM workspace.layer_sources
        WHERE layer = 'predictions';
        """
    )


def finalize_pack(pack: sqlite3.Connection) -> None:
    add_predictions(pack)
    pack.commit()


def annotate_kaomoji(pack: sqlite3.Connection) -> None:
    """Stamp the build-time kaomoji decision onto every record."""
    pack.create_function("is_kaomoji_like", 1, is_kaomoji_like, deterministic=True)
    placeholders = ",".join("?" * len(KAOMOJI_SOURCES))
    for table, source_table in (
        ("entries", "entry_sources"),
        ("predictions", "prediction_sources"),
    ):
        pack.execute(f"UPDATE {table} SET kaomoji = is_kaomoji_like(candidate)")
        pack.execute(
            f"""
            UPDATE {table} SET kaomoji = 1
            WHERE NOT kaomoji AND NOT EXISTS (
                SELECT 1 FROM {source_table} AS s
                WHERE s.reading = {table}.reading
                  AND s.candidate = {table}.candidate
                  AND s.source NOT IN ({placeholders})
            )
            """,
            tuple(sorted(KAOMOJI_SOURCES)),
        )
    pack.commit()


def _export_query(connection: sqlite3.Connection, query: str, path: Path) -> None:
    cursor = connection.execute(query)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(tuple(column[0] for column in cursor.description))
        while rows := cursor.fetchmany(BATCH_SIZE):
            writer.writerows(rows)
    os.replace(temporary, path)


def export_pack(connection: sqlite3.Connection, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _export_query(
        connection,
        """
        WITH source_lists AS (
            SELECT reading, candidate, group_concat(source, ',') AS sources
            FROM (SELECT * FROM entry_sources ORDER BY reading, candidate, source)
            GROUP BY reading, candidate
        )
        SELECT entries.reading, entries.candidate, entries.cost, source_lists.sources
        FROM entries JOIN source_lists USING (reading, candidate)
        ORDER BY entries.reading, entries.candidate
        """,
        directory / "entries.tsv",
    )
    _export_query(
        connection,
        """
        WITH source_lists AS (
            SELECT reading, candidate, group_concat(source, ',') AS sources
            FROM (SELECT * FROM prediction_sources ORDER BY reading, candidate, source)
            GROUP BY reading, candidate
        )
        SELECT predictions.reading, predictions.candidate, predictions.cost, source_lists.sources
        FROM predictions JOIN source_lists USING (reading, candidate)
        ORDER BY predictions.reading, predictions.candidate
        """,
        directory / "predictions.tsv",
    )
    for obsolete in (
        "entries-no-emoji.tsv",
        "predictions-no-emoji.tsv",
        "blocked.tsv",
        "corrections.tsv",
        "romanization.tsv",
        "search_terms.tsv",
        "sources.tsv",
        "stats.tsv",
    ):
        path = directory / obsolete
        if path.exists():
            path.unlink()


def write_skk_notices(sources: Iterable[str], path: Path) -> None:
    sections = []
    for source in sources:
        if not source.startswith("skk:"):
            continue
        relative = source.removeprefix("skk:")
        dictionary = DICT / "skk" / "extracted" / relative
        header_lines = []
        for line in dictionary.read_bytes().splitlines():
            if line.startswith(b";;") or not line.strip():
                header_lines.append(line)
                continue
            break
        header_bytes = b"\n".join(header_lines)
        for encoding in (
            "euc_jp",
            "euc_jis_2004",
            "utf-8",
            "iso2022_jp",
            "iso2022_jp_2004",
            "shift_jis",
        ):
            try:
                header = header_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeError(f"cannot decode SKK notice header: {dictionary}")
        sections.append(f"===== {relative} =====\n{header.rstrip()}")
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def write_distribution_documents(
    connection: sqlite3.Connection, directory: Path, pack_name: str
) -> None:
    compilation_license = {
        PACK_CORE: "CC-BY-4.0",
        PACK_EXTENDED: "GPL-3.0-only",
    }[pack_name]
    display_name = {
        PACK_CORE: "IME Dictionary Core",
        PACK_EXTENDED: "IME Dictionary Extended GPL",
    }[pack_name]
    related_repository = {
        PACK_CORE: (
            "For the larger GPL-licensed vocabulary, see "
            "[ime-dictionary-extended-gpl](https://huggingface.co/datasets/"
            "fa0311/warabi-dictionary-extended-gpl)."
        ),
        PACK_EXTENDED: (
            "For the smaller core distribution, see "
            "[ime-dictionary-core](https://huggingface.co/datasets/"
            "fa0311/warabi-dictionary-core)."
        ),
    }[pack_name]
    merge_order = {
        PACK_CORE: "`ipadic` +5000, `sudachi` +6000, `skk` (core subset) +5000",
        PACK_EXTENDED: (
            "`ipadic` +5000, `sudachi` +6000, `skk` (core subset) +5000, "
            "`jmdict` +4000, `skk` (GPL remainder) +5000, `jmnedict` +4000, "
            "`kanjidic2` +6000"
        ),
    }[pack_name]
    license_name = f"{compilation_license.lower()}-compilation-with-upstream-terms"
    license_source = PROJECT_ROOT / "licenses" / f"{compilation_license}.txt"
    shutil.copyfile(license_source, directory / "LICENSE.txt")

    third_party = directory / "third-party-licenses"
    third_party.mkdir(exist_ok=True)
    copies = {
        "MOZC-LICENSE.txt": DICT / "mozc" / "LICENSE",
        "IPADIC-COPYING": (
            DICT
            / "ipadic"
            / "source"
            / "mecab-ipadic-2.7.0-20070801"
            / "COPYING"
        ),
        "SKK-UPSTREAM.md": DICT / "skk" / "UPSTREAM_README.md",
        "SUDACHI-LICENSE-2.0.txt": DICT / "sudachi" / "LICENSE-2.0.txt",
        "SUDACHI-LEGAL.txt": DICT / "sudachi" / "LEGAL",
    }
    if pack_name == PACK_EXTENDED:
        copies["EDRDG-LICENSE.html"] = DICT / "jmdict" / "LICENSE.html"
    for name, source in copies.items():
        target = third_party / name
        shutil.copyfile(source, target)
    expected_notices = copies.keys() | {"SKK-NOTICES.txt"}
    for stale in {path.name for path in third_party.iterdir()} - expected_notices:
        path = third_party / stale
        if path.is_file():
            path.unlink()

    sources = [
        row[0]
        for row in connection.execute(
            """
            SELECT source FROM entry_sources
            UNION
            SELECT source FROM prediction_sources
            ORDER BY source
            """
        )
    ]
    write_skk_notices(sources, third_party / "SKK-NOTICES.txt")
    notice = [
        f"# {pack_name} dictionary pack",
        "",
        f"Compilation license: `{compilation_license}`.",
        "",
        "The compilation license applies only to rights held by this project. ",
        "Each incorporated source remains subject to its own terms below. Keep ",
        "this notice and the `third-party-licenses/` directory when redistributing.",
        "",
        "The data was normalized, deduplicated, cost-calibrated, and merged for ",
        "Japanese IME conversion. Source-specific costs are not distributed.",
        "",
        "## Incorporated sources",
        "",
    ]
    for source in sources:
        if source.startswith("original:"):
            continue
        policy = source_policy(source)
        link = f"; {policy.url}" if policy.url else ""
        note = f" ({policy.notes})" if policy.notes else ""
        notice.append(f"- `{source}`: {policy.license}{link}{note}")
    notice.extend(
        [
            "",
            "Mozc dictionary notices are reproduced in `third-party-licenses/MOZC-LICENSE.txt`.",
            (
                "MeCab IPADIC terms are preserved byte-for-byte in "
                "`third-party-licenses/IPADIC-COPYING`."
            ),
            "SKK provenance is reproduced in `third-party-licenses/SKK-UPSTREAM.md`.",
            (
                "SKK source copyright and license headers are preserved in "
                "`third-party-licenses/SKK-NOTICES.txt`."
            ),
            (
                "SudachiDict's Apache license and incorporated-data notices are "
                "preserved in `third-party-licenses/SUDACHI-LICENSE-2.0.txt` and "
                "`third-party-licenses/SUDACHI-LEGAL.txt`."
            ),
        ]
    )
    if pack_name == PACK_EXTENDED:
        notice.append(
            "EDRDG terms and required attribution are reproduced in "
            "`third-party-licenses/EDRDG-LICENSE.html`."
        )
    (directory / "NOTICE.md").write_text("\n".join(notice) + "\n", encoding="utf-8")

    readme = f"""---
license: other
license_name: {license_name}
language:
- ja
size_categories:
- 1M<n<10M
tags:
- japanese
- ime
- dictionary
configs:
- config_name: conversion
  data_files:
  - split: train
    path: entries.tsv
- config_name: prediction
  data_files:
  - split: train
    path: predictions.tsv
---
# {display_name}

Japanese IME dictionary pack. The SQLite database and TSV files contain the
same conversion and prediction records. See `NOTICE.md` before redistribution.

- Compilation license: `{compilation_license}`
- SQLite tables: `entries`, `entry_sources`, `predictions`, `prediction_sources`
- TSV exports: `entries.tsv`, `entries-no-emoji.tsv`, `predictions.tsv`,
  `predictions-no-emoji.tsv`

## Emoji exclusion

The SQLite `emoji` column (0/1) drives the no-emoji TSVs (`WHERE NOT
emoji`). A candidate is flagged when it contains a Unicode emoji or
`\\p{{Extended_Pictographic}}`, a decorative `So` symbol (HEART / STAR /
FLOWER / SPARKLE), matches an ASCII emoticon (`:-)`, `orz`, `T_T`), looks
structurally like a kaomoji (bracket / face-mark / arm-mark heuristics,
<= 48 chars), or is sourced solely from the project's emoji and kaomoji
dictionaries.

## Cost calibration

Costs are on Mozc's word-cost scale; layers merged in order, the first
provider of a (reading, candidate) wins. Project-original dictionaries were
quantile-mapped onto Mozc via their shared entries, Mozc was added
unchanged, then each remaining layer was isotonic-calibrated on its overlap
with the pack and offset by a fixed penalty, adding only novel entries:
{merge_order}. Pre-normalization: sudachi `clamp(7000 + cost/4, 6000,
16000)`; original emoji files `18000 + clamp(cost, 0, 10000)/2`; other
original files clamped to per-file floors (default 4500) and a 30000
ceiling.

## Related repository

{related_repository}
"""
    (directory / "README.md").write_text(readme, encoding="utf-8")


def build_packs(output: Path, workspace_path: Path, *, reuse_sources: bool) -> None:
    workspace = materialize_workspace(workspace_path, reuse=reuse_sources)
    workspace.close()

    output.mkdir(parents=True, exist_ok=True)
    core_temporary = output / f"{PACK_CORE}.sqlite3.building"
    core_pack = create_pack(core_temporary)
    attach_workspace(core_pack, workspace_path)
    workspace_read = sqlite3.connect(workspace_path)
    initialize_core_pack(core_pack, workspace_read)
    workspace_read.close()
    for layer, penalty in (("ipadic", 5_000), ("sudachi", 6_000), ("skk-core", 5_000)):
        print(f"merge core layer: {layer}", flush=True)
        add_layer(core_pack, layer, penalty=penalty)
    finalize_pack(core_pack)
    core_pack.execute("DETACH DATABASE workspace")
    core_pack.close()

    extended_temporary = output / f"{PACK_EXTENDED}.sqlite3.building"
    if extended_temporary.exists():
        extended_temporary.unlink()
    shutil.copyfile(core_temporary, extended_temporary)
    extended_pack = sqlite3.connect(extended_temporary)
    attach_workspace(extended_pack, workspace_path)
    for layer, penalty in (
        ("jmdict", 4_000),
        ("skk-extended", 5_000),
        ("jmnedict", 4_000),
        ("kanjidic2", 6_000),
    ):
        print(f"merge extended-gpl layer: {layer}", flush=True)
        add_layer(extended_pack, layer, penalty=penalty)
    extended_pack.commit()
    extended_pack.execute("DETACH DATABASE workspace")
    extended_pack.close()

    for pack_name, temporary in (
        (PACK_EXTENDED, extended_temporary),
        (PACK_CORE, core_temporary),
    ):
        connection = sqlite3.connect(temporary)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"pack {pack_name} failed integrity check: {integrity}")
        print(f"annotate emoji flags: {pack_name}", flush=True)
        annotate_emoji(connection)
        directory = output / pack_name
        export_pack(connection, directory)
        write_distribution_documents(connection, directory, pack_name)
        connection.close()
        os.replace(temporary, directory / "dictionary.sqlite3")
    print(f"built packs: {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build provenance-complete core and extended GPL dictionary packs"
    )
    parser.add_argument("--output", type=Path, default=DICTIONARY_ARTIFACTS_DIR / "packs")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ARTIFACTS_DIR / "cache" / "dictionary-packs.sqlite3",
    )
    parser.add_argument("--reuse-sources", action="store_true")
    args = parser.parse_args()
    build_packs(args.output, args.workspace, reuse_sources=args.reuse_sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
