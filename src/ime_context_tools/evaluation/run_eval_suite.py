"""Run the versioned acceptance suite through the real engine.

The suite (data/context-dataset/eval/v1) is the post-reset evaluation
contract: runtime-level scoring (ime-cli convert = lattice + blend + model),
per-tag accounting, forbidden-candidate tracking, and optional gates.

Row schema: {context, reading, accepted?: [..], forbidden?: [..], tag}
  - accepted present: pass = top-1 in accepted
  - accepted absent (kaomoji-any rows): pass = any of the top 3 candidates
    looks like a kaomoji (structural detector; the reading has no canonical
    conversion, surfacing SOME kaomoji is the contract)
  - forbidden: top-1 landing here is counted as a hard failure class

Contamination: --contamination-check scans training jsonl files for rows
whose (context, reading) collide with suite cases — eval rows must never be
trained on (including via mining).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import orjson

from ..dictionary.kaomoji_filter import is_kaomoji_like

# Gates are on UNSEEN rows (seen_in_train pairs collide with the frozen v1
# training split and are reported separately). Floors = the certified v8
# fleet's worst passing score; a drop below is a regression alarm.
DEFAULT_GATES = {"homophone-39": 29, "attractor-14": 7, "qualitative-15": 9, "newchar-8": 6}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--ime-cli", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", required=True,
        metavar="NAME=DIR", help="model directory to evaluate; repeatable",
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--gates", default=None,
        help="comma-separated file:min-pass overrides; 'none' disables gating",
    )
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    parser.add_argument(
        "--contamination-check", type=Path, nargs="*", default=[],
        help="training jsonl files to scan for (context, reading) collisions",
    )
    return parser


def convert(cli: Path, pack: Path, model: str, context: str, reading: str, top: int) -> list[str]:
    result = subprocess.run(
        [str(cli), "convert", str(pack), model, context or "-", reading, str(top)],
        capture_output=True, text=True, encoding="utf-8",
    )
    return [
        line.strip().split(". ", 1)[1].split("  [")[0]
        for line in result.stdout.splitlines()
        if line.strip()[:1].isdigit()
    ]


def judge(row: dict, candidates: list[str]) -> tuple[bool, bool]:
    """(passed, forbidden_hit)"""
    top1 = candidates[0] if candidates else None
    forbidden_hit = top1 in set(row.get("forbidden") or [])
    accepted = row.get("accepted")
    if accepted is None:
        return any(is_kaomoji_like(candidate) for candidate in candidates), forbidden_hit
    return top1 in set(accepted), forbidden_hit


def check_contamination(suite_rows: list[dict], train_files: list[Path]) -> list[dict]:
    """New collisions only: rows already annotated seen_in_train are known."""
    keys = {
        (row["context"], row["reading"])
        for row in suite_rows
        if not row.get("seen_in_train")
    }
    hits = []
    for path in train_files:
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                train_row = orjson.loads(line)
                key = (train_row.get("context", ""), train_row.get("reading", ""))
                if key in keys:
                    hits.append({"file": str(path), "context": key[0], "reading": key[1]})
    return hits


def main(argv: list[str] | None = None) -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    models = dict(spec.split("=", 1) for spec in args.model)
    files = sorted(args.suite_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no suite files under {args.suite_dir}")

    gates = dict(DEFAULT_GATES)
    if args.gates == "none":
        gates = {}
    elif args.gates:
        gates.update(
            (name, int(value))
            for name, value in (part.split(":") for part in args.gates.split(","))
        )

    report: dict = {"suite": str(args.suite_dir), "models": {}, "files": {}}
    all_rows: list[dict] = []
    suite: dict[str, list[dict]] = {}
    for path in files:
        rows = [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
        suite[path.stem] = rows
        all_rows.extend(rows)
        report["files"][path.stem] = len(rows)

    gate_failures = []
    for name, model_dir in models.items():
        scores: dict[str, dict] = {}
        for stem, rows in suite.items():
            passed = seen_passed = seen_total = 0
            forbidden_hits = 0
            by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            failures = []
            for row in rows:
                candidates = convert(
                    args.ime_cli, args.pack, model_dir, row["context"], row["reading"], args.top
                )
                ok, forbidden = judge(row, candidates)
                if row.get("seen_in_train"):
                    seen_total += 1
                    seen_passed += ok
                else:
                    passed += ok
                forbidden_hits += forbidden
                by_tag[row["tag"]][0] += ok
                by_tag[row["tag"]][1] += 1
                if not ok:
                    failures.append({
                        "context": row["context"], "reading": row["reading"],
                        "top": candidates[:3], "tag": row["tag"],
                    })
            unseen_total = len(rows) - seen_total
            scores[stem] = {
                "unseen_passed": passed, "unseen_total": unseen_total,
                "seen_passed": seen_passed, "seen_total": seen_total,
                "forbidden_top1": forbidden_hits,
                "by_tag": {tag: {"passed": p, "total": t} for tag, (p, t) in by_tag.items()},
                "failures": failures,
            }
            seen_note = f"  (+seen {seen_passed}/{seen_total})" if seen_total else ""
            print(f"{name}  {stem}: {passed}/{unseen_total}{seen_note}"
                  + (f"  (forbidden top1: {forbidden_hits})" if forbidden_hits else ""))
            if stem in gates and passed < gates[stem]:
                gate_failures.append(f"{name}:{stem} {passed} < {gates[stem]}")
        report["models"][name] = scores

    if args.contamination_check:
        hits = check_contamination(all_rows, list(args.contamination_check))
        report["contamination"] = hits
        print(f"contamination: {len(hits)} collisions")
        if hits:
            gate_failures.append(f"contamination: {len(hits)} training collisions")

    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if gate_failures:
        print("GATE FAILURES: " + "; ".join(gate_failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
