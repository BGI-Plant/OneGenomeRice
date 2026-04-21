#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate sliding-window genomic datasets from FASTA files.

This script reads grouped FASTA files, trims leading/trailing Ns for each
sequence, creates fixed-length sliding windows, and exports records as JSONL.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FASTA_ROOT = SCRIPT_DIR.parent / "fasta_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "data"


FASTA_GROUPS = {
    "rice_introgression": {
        "train": {
            'jap': ['fasta_data/01.genome.fa'],
            'ind': ['fasta_data/02.genome.fa'],
        },
        "test": {
            'jap': ['fasta_data/03.genome.fa'],
            'ind': ['fasta_data/04.genome.fa'],
        },
    }
}


def read_fasta(file_path: str | Path) -> dict[str, str]:
    """Read a FASTA or gzipped FASTA and return {seq_id: sequence}."""
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    file_path = str(file_path)
    if file_path.endswith(".gz"):
        open_func = gzip.open
        mode = "rt"
    else:
        open_func = open
        mode = "r"

    with open_func(file_path, mode, encoding="utf-8") as fasta:
        for line in fasta:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            sequences[current_id] = "".join(current_seq)

    return sequences


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve absolute/relative path. Relative path is based on base_dir or current CWD."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        return (Path(base_dir) / path).resolve()
    return (Path.cwd() / path).resolve()


def resolve_fasta_file_paths(
    file_paths: list[str], fasta_root: str | Path
) -> list[Path]:
    """Resolve FASTA file paths from FASTA_GROUPS. Relative paths use fasta_root."""
    root = resolve_path(fasta_root)
    return [resolve_path(file_path, base_dir=root) for file_path in file_paths]


def trim_n(sequence: str) -> tuple[str, list[int]]:
    """Trim leading and trailing N bases, return trimmed seq and [start, end)."""
    sequence = sequence.upper()
    start = 0
    end = len(sequence)

    while start < end and sequence[start] == "N":
        start += 1
    while end > start and sequence[end - 1] == "N":
        end -= 1

    return sequence[start:end], [start, end]


def sliding_window(
    sequence: str, window_size: int = 8000, step: int = 4000
) -> list[dict[str, Any]]:
    """Create windows and return each window with [start, end) coordinates."""
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive integers")

    windows: list[dict[str, Any]] = []
    for start in range(0, len(sequence) - window_size + 1, step):
        end = start + window_size
        windows.append({"window": sequence[start:end], "seq_start_end": [start, end]})
    return windows


def collect_windows_from_files(
    file_paths: list[Path], window_size: int, step_size: int, unique: bool = False
) -> list[dict[str, Any]]:
    """Collect all windows from a list of FASTA files."""
    windows: list[dict[str, Any]] = []
    for fasta_file in file_paths:
        sequences = read_fasta(fasta_file)
        for seq_id, sequence in sequences.items():
            trimmed_seq, trimmed_seq_start_end = trim_n(sequence)
            seq_windows = sliding_window(
                trimmed_seq,
                window_size=window_size,
                step=step_size,
            )

            for sw in seq_windows:
                window_start, window_end = sw["seq_start_end"]
                windows.append(
                    {
                        "window": sw["window"],
                        "file": str(fasta_file),
                        "seq_id": seq_id,
                        "seq_start_end": [
                            trimmed_seq_start_end[0] + window_start,
                            trimmed_seq_start_end[0] + window_end,
                        ],
                    }
                )

    if not unique:
        return windows

    seen = set()
    unique_windows: list[dict[str, Any]] = []
    for record in windows:
        if record["window"] in seen:
            continue
        seen.add(record["window"])
        unique_windows.append(record)

    return unique_windows


def make_one_hot_labels(varieties: list[str]) -> dict[str, list[int]]:
    """Build one-hot labels from variety names."""
    labels: dict[str, list[int]] = {}
    for idx, variety in enumerate(varieties):
        one_hot = [0] * len(varieties)
        one_hot[idx] = 1
        labels[variety] = one_hot
    return labels


def build_dataset_with_selection(
    dataset_name: str,
    dataset_class: str,
    selected_varieties: list[str],
    window_size: int,
    step_size: int,
    fasta_root: str | Path,
    unique: bool = True,
) -> list[dict[str, Any]]:
    """Build training/testing records from selected varieties."""
    all_records: list[dict[str, Any]] = []
    variety_to_files = FASTA_GROUPS[dataset_name][dataset_class]
    variety_to_label = make_one_hot_labels(selected_varieties)

    for variety in selected_varieties:
        files = resolve_fasta_file_paths(variety_to_files[variety], fasta_root=fasta_root)
        window_data = collect_windows_from_files(
            files,
            window_size=window_size,
            step_size=step_size,
            unique=unique,
        )
        print(f"{dataset_class} | {variety} windows: {len(window_data)}")

        for window in window_data:
            all_records.append(
                {
                    "sequence": window["window"],
                    "label": variety_to_label[variety],
                    "variety": variety,
                    "source": window["file"],
                    "seq_id": window["seq_id"],
                    "seq_start_end": window["seq_start_end"],
                    "source_group": f"{dataset_name}_{dataset_class}",
                }
            )

    return all_records


def save_jsonl(data: list[dict[str, Any]], filename: str | Path) -> None:
    """Write records to JSONL."""
    with Path(filename).open("w", encoding="utf-8") as out_f:
        for item in data:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_config(config: dict[str, Any]) -> None:
    """Validate runtime configuration."""
    required_keys = {
        "dataset_name",
        "dataset_class",
        "selected_varieties",
        "window_size",
        "step_size",
        "unique_windows",
        "fasta_root",
        "output_dir",
    }
    missing = sorted(required_keys - set(config))
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")

    dataset_name = config["dataset_name"]
    dataset_class = config["dataset_class"]
    selected_varieties = config["selected_varieties"]

    if dataset_name not in FASTA_GROUPS:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    if dataset_class not in FASTA_GROUPS[dataset_name]:
        raise ValueError(
            f"Unknown dataset_class: {dataset_class} for dataset_name: {dataset_name}"
        )

    available_varieties = FASTA_GROUPS[dataset_name][dataset_class]
    for variety in selected_varieties:
        if variety not in available_varieties:
            raise ValueError(
                f"Variety '{variety}' not found in {dataset_name}/{dataset_class}"
            )


def run_one_config(config: dict[str, Any], random_seed: int = 42) -> Path:
    """Run one dataset build config and return output jsonl path."""
    validate_config(config)
    random.seed(random_seed)

    dataset_name = config["dataset_name"]
    dataset_class = config["dataset_class"]
    selected_varieties = config["selected_varieties"]
    window_size = config["window_size"]
    step_size = config["step_size"]
    fasta_root = config["fasta_root"]
    output_dir = resolve_path(config["output_dir"])

    run_dir = output_dir / (
        f"{dataset_name}_{'-'.join(selected_varieties)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    all_data = build_dataset_with_selection(
        dataset_name=dataset_name,
        dataset_class=dataset_class,
        selected_varieties=selected_varieties,
        window_size=window_size,
        step_size=step_size,
        fasta_root=fasta_root,
        unique=config["unique_windows"],
    )

    output_path = run_dir / f"{dataset_class}.jsonl"
    save_jsonl(all_data, output_path)
    print(f"Saved {len(all_data)} records to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate sliding-window genome datasets")
    parser.add_argument(
        "--dataset-name",
        default="rice_introgression",
        help="Dataset key in FASTA_GROUPS",
    )
    parser.add_argument(
        "--fasta-root",
        default=str(DEFAULT_FASTA_ROOT),
        help="Root directory for FASTA_GROUPS relative FASTA files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output root directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    configs: list[dict[str, Any]] = [
        {
            "dataset_name": args.dataset_name,
            "dataset_class": "train",
            "selected_varieties": ["jap", "ind"],
            "window_size": 8 * 1000,
            "step_size": 4 * 1000,
            "unique_windows": True,
            "fasta_root": str(args.fasta_root),
            "output_dir": str(args.output_dir),
        },
        {
            "dataset_name": args.dataset_name,
            "dataset_class": "test",
            "selected_varieties": ["jap", "ind"],
            "window_size": 8 * 1000,
            "step_size": 8 * 1000,
            "unique_windows": True,
            "fasta_root": str(args.fasta_root),
            "output_dir": str(args.output_dir),
        },
    ]

    for cfg in configs:
        run_one_config(cfg, random_seed=args.seed)


if __name__ == "__main__":
    main()
