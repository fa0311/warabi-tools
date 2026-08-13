"""Dump the Python engine's first-N ranked candidates per reading.

Reference implementation: `tui_ime.generator.CandidateGenerator.
iter_ranked_candidates` with the default `AlgorithmicScorer`, predictions
enabled — the exact stream the TUI engine ranks. Output shape matches
`ime-cli convert-dump` for byte-level diffing:

  {"reading": ..., "candidates": [[text, search_cost, kind, whole_exact,
   dictionary_cost, [[seg_reading, seg_text, seg_cost], ...]], ...]}

Requires the legacy `D:\\ime` checkout on PYTHONPATH (tui_ime package); this
dump is regenerated only while the Python engine remains the reference.
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import orjson
from tui_ime.generator import CandidateGenerator
from tui_ime.lexicon import Lexicon, normalize_reading
from tui_ime.scoring import AlgorithmicScorer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    lexicon = Lexicon(args.pack)
    generator = CandidateGenerator(lexicon, include_predictions=True)
    scorer = AlgorithmicScorer()
    readings = [
        line.strip()
        for line in args.readings.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        for raw in readings:
            reading = normalize_reading(raw)
            candidates = [
                [
                    candidate.text,
                    candidate.search_cost,
                    candidate.kind,
                    candidate.whole_exact,
                    candidate.dictionary_cost,
                    [[seg.reading, seg.text, seg.cost] for seg in candidate.segments],
                ]
                for candidate in islice(
                    generator.iter_ranked_candidates(reading, scorer), args.count
                )
            ]
            out.write(orjson.dumps({"reading": reading, "candidates": candidates}) + b"\n")
    print(f"dumped {len(readings)} readings x {args.count} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
