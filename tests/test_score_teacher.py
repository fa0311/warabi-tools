import orjson

from ime_context_tools.training.score_teacher_cross_encoder import (
    select_indices,
    sidecar_path,
    verify_file,
)


def _row(candidate_count: int, target_index: int) -> dict:
    return {
        "id": f"row-{target_index}",
        "context": "家に",
        "reading": "かえる",
        "target_index": target_index,
        "candidates": [{"text": f"cand{i}"} for i in range(candidate_count)],
    }


def test_select_indices_head_contains_target() -> None:
    assert select_indices(_row(200, 3), head=97) == list(range(97))


def test_select_indices_appends_out_of_head_target() -> None:
    indices = select_indices(_row(200, 150), head=97)
    assert indices == [*range(97), 150]


def test_select_indices_short_candidate_list() -> None:
    assert select_indices(_row(5, 2), head=97) == [0, 1, 2, 3, 4]


def test_verify_file_reproduces_ranks(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    rows = [_row(4, 0), _row(4, 2), _row(4, 3)]
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))

    sidecar = sidecar_path(tmp_path, source)
    header = {
        "kind": "teacher_logits_header",
        "teacher": {"weights_sha256": "x"},
        "source": str(source),
    }
    # logits rank: row0 target first (rank 1), row1 target second (rank 2),
    # row2 target last (rank 4)
    records = [
        {"id": "a", "candidate_indices": [0, 1, 2, 3], "logits": [9.0, 1.0, 0.0, -1.0]},
        {"id": "b", "candidate_indices": [0, 1, 2, 3], "logits": [9.0, 1.0, 5.0, -1.0]},
        {"id": "c", "candidate_indices": [0, 1, 2, 3], "logits": [9.0, 5.0, 1.0, -1.0]},
    ]
    sidecar.write_bytes(
        orjson.dumps(header)
        + b"\n"
        + b"".join(orjson.dumps(record) + b"\n" for record in records)
    )

    metrics = verify_file(source, sidecar)
    assert metrics["rows"] == 3
    assert metrics["top1"] == 1 / 3
    assert metrics["top5"] == 1.0
    assert metrics["mrr"] == (1 + 1 / 2 + 1 / 4) / 3
