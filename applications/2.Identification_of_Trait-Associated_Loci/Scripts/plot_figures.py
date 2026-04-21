#!/usr/bin/env python3
"""Generate differential-signal and gene-prioritization figures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def rel(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(config_path: Path) -> tuple[Path, dict]:
    root = config_path.resolve().parent
    return root, json.loads(config_path.read_text(encoding="utf-8"))


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def require_dirs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing pipeline outputs. Run 1.calc_attention.sh first:\n  - {joined}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root, cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    workflow = cfg["workflow"]
    scripts = root / "Scripts" / "lib"
    results = rel(root, paths["results_dir"])

    attention = results / "attention"
    bed_dir = attention / "region_beds"
    diff_dir = attention / "05_differential_sites"
    matrix_dir = attention / "04_attention_matrices"
    figures = results / "figures"
    tables = results / "tables"
    metrics = tables / "gene_metrics"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    region_dirs = [diff_dir / f"region_{i}" for i in range(1, 5)]
    matrix_dirs = [matrix_dir / f"region_{i}" for i in range(1, 5)]
    bed_files = [bed_dir / f"region_{i}.bed" for i in range(1, 5)]
    require_dirs(region_dirs + matrix_dirs + bed_files)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(results / ".mplconfig"))
    py = sys.executable

    common_signal_args = [
        py,
        str(scripts / "plot_signal_panel_annotated.py"),
        "--run-name",
        cfg["project"]["name"],
        "--gff",
        str(rel(root, paths["gene_annotation"])),
        "--chrom",
        str(workflow["chromosome"]),
        "--region-dir",
        *map(str, region_dirs),
        "--bed-file",
        *map(str, bed_files),
        "--matrix-dir",
        *map(str, matrix_dirs),
        "--display-title",
        *workflow["region_names"],
    ]
    run(
        [
            *common_signal_args,
            "--metric",
            "neglog10_padj",
            "--out-png",
            str(figures / "differential_attention_padj.png"),
        ],
        env,
    )
    run(
        [
            *common_signal_args,
            "--metric",
            "log2fc",
            "--out-png",
            str(figures / "differential_attention_log2fc.png"),
        ],
        env,
    )

    run(
        [
            py,
            str(scripts / "calculate_gene_metrics_from_blocks.py"),
            "--gff",
            str(rel(root, paths["gene_annotation"])),
            "--chrom",
            str(workflow["chromosome"]),
            "--matrix-dir",
            *map(str, matrix_dirs),
            "--matrix-files",
            ",".join(workflow["matrix_files"]),
            "--add-direction-sum",
            "--flank-length",
            "0",
            "--out-dir",
            str(metrics),
        ],
        env,
    )

    top_n = int(workflow.get("gene_metric_top_n", 8))
    run(
        [
            py,
            str(scripts / "plot_gene_metric_top10.py"),
            "--long-csv",
            str(metrics / "gene_level_17metrics_directional_pvalues.csv"),
            "--top-n",
            str(top_n),
            "--out-png",
            str(figures / f"gene_metric_sum_top{top_n}.png"),
            "--out-csv",
            str(tables / f"gene_metric_sum_top{top_n}.csv"),
        ],
        env,
    )

    print(f"Figures written to {figures}")
    print(f"Tables written to {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
