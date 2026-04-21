import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef


def _safe_macro_auc(labels: np.ndarray, logits: np.ndarray) -> float:
    """Compute macro AUC over columns with both positive and negative labels."""
    auc_scores = []
    for i in range(labels.shape[1]):
        unique_labels = np.unique(labels[:, i])
        if unique_labels.size < 2:
            continue
        auc_scores.append(roc_auc_score(labels[:, i], logits[:, i]))
    return float(np.mean(auc_scores)) if auc_scores else float("nan")


def compute_metrics(logits, labels, preds):
    """Compute multilabel metrics from probabilities/logits, labels and binary predictions."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    if not (logits.shape == labels.shape == preds.shape):
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, labels={labels.shape}, preds={preds.shape}"
        )

    def calculate_mcc(labels, probs):
        mcc_scores = []
        for i in range(labels.shape[1]):
            mcc_scores.append(matthews_corrcoef(labels[:, i], probs[:, i]))
        mcc = np.mean(mcc_scores)
        return mcc

    return {
        "accuracy": accuracy_score(labels, preds),
        "auc_roc": _safe_macro_auc(labels, logits),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "mcc": calculate_mcc(labels, preds)
    }


def eval_from_tsv(df, T=0.5, output_path=""):
    """Evaluate metrics from a TSV dataframe with columns: ground_truth, prob."""
    gt = df["ground_truth"].str.split(",", expand=True).astype(int)
    y_probs = df["prob"].str.split(",", expand=True).astype(float).values
    y_preds = (y_probs >= T).astype(int)

    result = compute_metrics(y_probs, gt.values, y_preds)

    filter_mask = ~((y_preds == 0).all(axis=1) | (y_preds == 1).all(axis=1))
    if filter_mask.any():
        result_filter = compute_metrics(
            y_probs[filter_mask], gt.values[filter_mask], y_preds[filter_mask]
        )
    else:
        result_filter = None

    result = {
        "all": result,
        "filtered": result_filter,
        "filtered_samples": int(filter_mask.sum()),
        "total_samples": len(df),
        "threshold": T
    }

    output_path = output_path.strip() if output_path is not None else ""
    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved eval_from_tsv result to: {out}")

    return result


def _collect_tsv_files(inputs: list[str]) -> list[Path]:
    """
    Resolve input paths to a sorted list of TSV files.
    Each input can be:
    - a .tsv file path
    - a directory path (all .tsv files under it, recursive)
    Supports both relative and absolute paths.
    """
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            if path.suffix.lower() != ".tsv":
                raise ValueError(f"Expected a .tsv file, got: {path}")
            files.append(path)
            continue

        if path.is_dir():
            dir_tsvs = sorted(p for p in path.rglob("*.tsv") if p.is_file())
            if not dir_tsvs:
                raise FileNotFoundError(f"No .tsv files found under directory: {path}")
            files.extend(dir_tsvs)
            continue

        raise ValueError(f"Unsupported input path type: {path}")

    # De-duplicate while preserving order
    deduped: list[Path] = []
    seen = set()
    for f in files:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics from one/many TSV files or directories containing TSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="TSV file path(s) and/or directory path(s). Relative and absolute paths are both supported."
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for converting probabilities to binary predictions. Default: 0.5"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Optional output JSON path. Empty means do not save."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tsv_files = _collect_tsv_files(args.inputs)

    print("Resolved TSV files:")
    for p in tsv_files:
        print(f"- {p}")

    dfs = [pd.read_csv(p, sep="\t") for p in tsv_files]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    res = eval_from_tsv(df, T=args.threshold, output_path=args.output)
    for split_name, metrics in res.items():
        print(f"{split_name}:")
        print(metrics)
