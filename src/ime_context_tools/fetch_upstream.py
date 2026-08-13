"""Download every upstream asset the pipeline needs from Hugging Face.

Makes ime-context-tools self-contained: no path may point outside this
repository. Idempotent (snapshot_download caches revisions).

Layout under --output-dir (default artifacts/upstream/):
  teacher/            fa0311/ime-context-reranker-v1  (model)
  dictionary-core/    fa0311/ime-dictionary-core      (dataset: sqlite + tsv)
  dictionary-extended-gpl/  fa0311/ime-dictionary-extended-gpl (optional)
  context-dataset/    fa0311/ime-context-dataset      (raw context rows)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = [
    ("teacher", "fa0311/ime-context-reranker-v1", "model", True),
    ("dictionary-core", "fa0311/ime-dictionary-core", "dataset", True),
    ("dictionary-extended-gpl", "fa0311/ime-dictionary-extended-gpl", "dataset", False),
    ("context-dataset", "fa0311/ime-context-dataset", "dataset", True),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[2] / "artifacts" / "upstream",
    )
    parser.add_argument("--include-optional", action="store_true")
    return parser


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest: dict[str, dict[str, str]] = {}
    for name, repo, repo_type, required in REPOS:
        if not required and not args.include_optional:
            continue
        target = args.output_dir / name
        path = snapshot_download(repo, repo_type=repo_type, local_dir=target)
        entry = {"repo": repo, "repo_type": repo_type, "path": str(path)}
        weights = target / "model.safetensors"
        if weights.is_file():
            entry["model_safetensors_sha256"] = sha256_of(weights)
        manifest[name] = entry
        print(f"{name}: {repo} -> {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
