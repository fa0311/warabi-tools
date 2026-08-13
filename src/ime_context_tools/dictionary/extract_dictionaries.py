#!/usr/bin/env python3
"""Safely expand the dictionary archives into the repository layout."""

from __future__ import annotations

import gzip
import shutil
import tarfile
from pathlib import Path

from tqdm import tqdm
from .paths import DICTIONARY_SOURCES_DIR

DICT = DICTIONARY_SOURCES_DIR


def verify_member_path(base: Path, member_name: str) -> None:
    base = base.resolve()
    target = (base / member_name).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"archive member escapes target directory: {member_name}")


def extract_tar(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            verify_member_path(target, member.name)
        tar.extractall(target, filter="data")


def extract_gzip(archive: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive, "rb") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def main() -> None:
    jobs: list[tuple[Path, Path, str]] = []

    for archive in sorted((DICT / "skk" / "archives").glob("*.gz")):
        if archive.name.endswith(".tar.gz"):
            target = DICT / "skk" / "extracted" / archive.name.removesuffix(".tar.gz")
            jobs.append((archive, target, "tar"))
        else:
            target = DICT / "skk" / "extracted" / archive.name.removesuffix(".gz")
            jobs.append((archive, target, "gzip"))

    for archive in sorted((DICT / "jmdict" / "archives").glob("*.gz")):
        target = DICT / "jmdict" / "xml" / archive.name.removesuffix(".gz")
        jobs.append((archive, target, "gzip"))

    for archive in sorted((DICT / "jmdict-furigana" / "archives").glob("*.tar.gz")):
        jobs.append((archive, DICT / "jmdict-furigana" / "json", "tar"))

    for archive in sorted((DICT / "ipadic" / "archives").glob("*.tar.gz")):
        jobs.append((archive, DICT / "ipadic" / "source", "tar"))

    for archive, target, kind in tqdm(jobs, desc="extract", unit=" archive"):
        if kind == "tar":
            extract_tar(archive, target)
        else:
            extract_gzip(archive, target)


if __name__ == "__main__":
    main()
