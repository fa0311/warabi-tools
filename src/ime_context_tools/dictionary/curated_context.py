from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CuratedContextCase:
    id: str
    context: str
    reading: str
    expected: tuple[str, ...]
    full_reading: str
    converted: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkippedInflectedTarget:
    id: str
    full_reading: str
    converted: str
    reading: str
    expected: str


@dataclass(frozen=True, slots=True)
class CuratedContextDataset:
    cases: tuple[CuratedContextCase, ...]
    source_rows: int
    skipped_inflected_targets: tuple[SkippedInflectedTarget, ...]
    excluded_invalid_targets: int = 0


def load_curated_context(
    path: Path | str,
    *,
    excluded_targets: frozenset[tuple[str, str]] = frozenset(),
) -> CuratedContextDataset:
    source = Path(path)
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    skipped: list[SkippedInflectedTarget] = []
    source_rows = 0
    excluded_invalid_targets = 0
    seen_ids: set[str] = set()

    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = raw_line.split("\t")
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: expected 5 tab-separated fields")
        identifier, full_reading, converted, target_reading, expected_field = fields
        if identifier in seen_ids:
            raise ValueError(f"line {line_number}: duplicate ID {identifier}")
        seen_ids.add(identifier)
        source_rows += 1

        for expected_index, expected in enumerate(expected_field.split("/"), start=1):
            if (identifier, expected) in excluded_targets:
                excluded_invalid_targets += 1
                continue
            position = converted.rfind(expected)
            if position < 0:
                skipped.append(
                    SkippedInflectedTarget(
                        identifier,
                        full_reading,
                        converted,
                        target_reading,
                        expected,
                    )
                )
                continue
            context = converted[:position]
            key = (context, target_reading)
            case_id = identifier if "/" not in expected_field else f"{identifier}-{expected_index}"
            if key not in grouped:
                grouped[key] = {
                    "id": case_id,
                    "context": context,
                    "reading": target_reading,
                    "expected": [],
                    "full_reading": full_reading,
                    "converted": converted,
                    "source_ids": [],
                }
            expected_values = grouped[key]["expected"]
            source_ids = grouped[key]["source_ids"]
            assert isinstance(expected_values, list)
            assert isinstance(source_ids, list)
            if expected not in expected_values:
                expected_values.append(expected)
            if identifier not in source_ids:
                source_ids.append(identifier)

    cases = tuple(
        CuratedContextCase(
            id=str(values["id"]),
            context=str(values["context"]),
            reading=str(values["reading"]),
            expected=tuple(values["expected"]),
            full_reading=str(values["full_reading"]),
            converted=str(values["converted"]),
            source_ids=tuple(values["source_ids"]),
        )
        for values in grouped.values()
    )
    return CuratedContextDataset(
        cases,
        source_rows,
        tuple(skipped),
        excluded_invalid_targets,
    )


def load_excluded_targets(path: Path | str) -> frozenset[tuple[str, str]]:
    source = Path(path)
    targets: set[tuple[str, str]] = set()
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8-sig").splitlines()):
        if line_number == 0 or not raw_line.strip() or raw_line.startswith("#"):
            continue
        fields = raw_line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"invalid excluded-target row at line {line_number + 1}")
        targets.add((fields[0], fields[1]))
    return frozenset(targets)
