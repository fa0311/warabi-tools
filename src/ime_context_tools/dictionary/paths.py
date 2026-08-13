"""Repository-relative locations for the dictionary build pipeline.

Everything lives inside ime-context-tools; the same IME_* environment
variables as the legacy tree override individual locations.

  dictionary-sources/  extracted upstream dictionaries (mozc, skk, sudachi,
                       jmdict, ipadic, original .bin dumps) — provided by
                       `extract_dictionaries`
  artifacts/dictionaries/  merged TSV/sqlite outputs and packs
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("IME_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()
DATASETS_DIR = Path(os.environ.get("IME_DATASETS_DIR", PROJECT_ROOT / "datasets")).resolve()
ARTIFACTS_DIR = Path(os.environ.get("IME_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")).resolve()
DICTIONARY_SOURCES_DIR = Path(
    os.environ.get("IME_DICTIONARY_SOURCES_DIR", ARTIFACTS_DIR / "dictionary-sources")
).resolve()
DICTIONARY_ARTIFACTS_DIR = ARTIFACTS_DIR / "dictionaries"

DEFAULT_DICTIONARY_PATH = (
    DICTIONARY_ARTIFACTS_DIR / "packs" / "ime-dictionary-extended-gpl" / "dictionary.sqlite3"
)

# Rust engine binary used for ranking benchmarks (parity-verified with the
# retired Python engine).
DEFAULT_IME_CLI = Path(
    os.environ.get(
        "IME_CLI",
        Path.home() / ".cargo" / "target" / "ime-context-core" / "release" / "ime-cli.exe",
    )
)
