#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare local JSONL files for mlx_lm LoRA training."
    )
    parser.add_argument(
        "--source",
        default="data/train_original.jsonl",
        help="Source JSONL file containing chat samples.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where train/valid/test JSONL files will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        default=True,
        help="Drop near-duplicate assistant outputs before splitting.",
    )
    parser.add_argument(
        "--keep-problem-cluster",
        action="store_true",
        help="Keep the known repetitive '我不会对你说谎/你让我不爽' lyric cluster.",
    )
    parser.add_argument(
        "--drop-template-heavy",
        action="store_true",
        default=True,
        help="Drop chorus-heavy samples that overreuse high-frequency lines.",
    )
    parser.add_argument(
        "--drop-dense-fixed-style",
        action="store_true",
        default=True,
        help="Drop dense one-line fixed-style samples that look stitched together.",
    )
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_no} has invalid messages data")
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"{path}:{line_no} contains a non-object message")
                if "role" not in message or "content" not in message:
                    raise ValueError(
                        f"{path}:{line_no} message is missing role or content"
                    )
            rows.append(row)
    if len(rows) < 3:
        raise ValueError("Need at least 3 samples to create train/valid/test splits")
    return rows


def compute_split_sizes(total: int, train_ratio: float, valid_ratio: float):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1")

    train_count = max(1, int(total * train_ratio))
    valid_count = max(1, int(total * valid_ratio))
    test_count = total - train_count - valid_count

    while test_count < 1:
        if train_count >= valid_count and train_count > 1:
            train_count -= 1
        elif valid_count > 1:
            valid_count -= 1
        else:
            raise ValueError("Unable to reserve at least 1 sample for each split")
        test_count = total - train_count - valid_count

    return train_count, valid_count, test_count


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return "".join(text.split())


def is_problem_cluster(text: str) -> bool:
    normalized = normalize_text(text)
    markers = (
        "我不会对你说谎",
        "你让我不爽",
        "我们难道不都半斤八两",
        "你想都别想",
    )
    hits = sum(marker in normalized for marker in markers)
    return hits >= 2


def dedupe_rows(rows: Iterable[dict], keep_problem_cluster: bool):
    seen_assistant = set()
    deduped = []
    dropped = 0

    for row in rows:
        assistant_text = row["messages"][-1]["content"]
        normalized_assistant = normalize_text(assistant_text)

        if not keep_problem_cluster and is_problem_cluster(assistant_text):
            dropped += 1
            continue

        if normalized_assistant in seen_assistant:
            dropped += 1
            continue

        seen_assistant.add(normalized_assistant)
        deduped.append(row)

    return deduped, dropped


def should_drop_template_heavy_sample(text: str, line_counter: dict[str, int]) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return False

    unique_lines = len(set(lines))
    duplicate_ratio = 1 - unique_lines / len(lines)
    repeated_line_groups = sum(1 for line in set(lines) if lines.count(line) > 1)
    frequent_line_hits = sum(1 for line in lines if line_counter.get(line, 0) >= 5)

    # Filter samples that are mostly assembled from a small set of high-frequency
    # lines and contain repeated chorus fragments inside the same sample.
    return (
        duplicate_ratio >= 0.15
        and repeated_line_groups >= 1
        and frequent_line_hits >= 4
    )


def drop_template_heavy_rows(rows: Iterable[dict]):
    line_counter: dict[str, int] = {}
    for row in rows:
        for line in [x.strip() for x in row["messages"][-1]["content"].splitlines() if x.strip()]:
            line_counter[line] = line_counter.get(line, 0) + 1

    cleaned = []
    dropped = 0
    for row in rows:
        assistant_text = row["messages"][-1]["content"]
        if should_drop_template_heavy_sample(assistant_text, line_counter):
            dropped += 1
            continue
        cleaned.append(row)

    return cleaned, dropped


def should_drop_dense_fixed_style_row(row: dict) -> bool:
    user_text = row["messages"][0]["content"]
    assistant_text = row["messages"][-1]["content"]
    line_count = len([line for line in assistant_text.splitlines() if line.strip()])
    comma_count = assistant_text.count("，") + assistant_text.count(",")

    return (
        "写一段固定风格的歌词" in user_text
        and line_count <= 3
        and comma_count >= 8
    )


def drop_dense_fixed_style_rows(rows: Iterable[dict]):
    cleaned = []
    dropped = 0
    for row in rows:
        if should_drop_dense_fixed_style_row(row):
            dropped += 1
            continue
        cleaned.append(row)
    return cleaned, dropped


def main():
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)

    rows = load_rows(source)
    dropped = 0
    if args.dedupe:
        rows, dropped_now = dedupe_rows(rows, args.keep_problem_cluster)
        dropped += dropped_now
    if args.drop_template_heavy:
        rows, dropped_now = drop_template_heavy_rows(rows)
        dropped += dropped_now
    if args.drop_dense_fixed_style:
        rows, dropped_now = drop_dense_fixed_style_rows(rows)
        dropped += dropped_now
    random.Random(args.seed).shuffle(rows)
    train_count, valid_count, _ = compute_split_sizes(
        len(rows), args.train_ratio, args.valid_ratio
    )

    train_rows = rows[:train_count]
    valid_rows = rows[train_count : train_count + valid_count]
    test_rows = rows[train_count + valid_count :]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "train.jsonl", train_rows)
    write_rows(output_dir / "valid.jsonl", valid_rows)
    write_rows(output_dir / "test.jsonl", test_rows)

    print(
        "Prepared MLX dataset:",
        f"train={len(train_rows)}",
        f"valid={len(valid_rows)}",
        f"test={len(test_rows)}",
        f"dropped={dropped}",
    )


if __name__ == "__main__":
    main()
