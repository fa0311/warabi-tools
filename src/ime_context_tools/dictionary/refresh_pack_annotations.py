"""Re-annotate an existing dictionary pack in place.

Migrates a published pack (downloaded from Hugging Face) to the current
annotation schema without rebuilding from sources: drops the legacy `emoji`
column if present, adds `kaomoji`, stamps it with `is_kaomoji_like` +
source rules, regenerates the TSV exports, and VACUUMs. The pack directory
must contain `dictionary.sqlite3`; sources tables are required (they drive
the source-based kaomoji rule and the TSV export joins).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from .build_dictionary_packs import annotate_kaomoji, export_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pack-dir",
        type=Path,
        action="append",
        required=True,
        help="directory holding dictionary.sqlite3; repeatable",
    )
    return parser


def migrate_columns(pack: sqlite3.Connection) -> None:
    for table in ("entries", "predictions"):
        columns = {row[1] for row in pack.execute(f"PRAGMA table_info({table})")}
        if "emoji" in columns:
            pack.execute(f"ALTER TABLE {table} DROP COLUMN emoji")
        if "kaomoji" not in columns:
            pack.execute(
                f"ALTER TABLE {table} ADD COLUMN kaomoji INTEGER NOT NULL DEFAULT 0"
            )
    pack.commit()


def refresh(directory: Path) -> None:
    pack = sqlite3.connect(directory / "dictionary.sqlite3")
    migrate_columns(pack)
    annotate_kaomoji(pack)
    flagged = {
        table: pack.execute(f"SELECT COUNT(*), SUM(kaomoji) FROM {table}").fetchone()
        for table in ("entries", "predictions")
    }
    export_pack(pack, directory)
    pack.execute("VACUUM")
    pack.close()
    for table, (total, kaomoji) in flagged.items():
        print(f"{directory.name}/{table}: {total} rows, {kaomoji} kaomoji")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for directory in args.pack_dir:
        refresh(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
