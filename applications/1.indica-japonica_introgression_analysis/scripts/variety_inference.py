#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import yaml
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.compute_metrics import eval_from_tsv
from utils.genomic_window_egmentation import collect_windows_from_files
from utils.utils import predict_with_rf_models


VARIETY_LABEL_MAPPING = {
    (1, 0): "Japonica",
    (0, 1): "Indica",
}


def map_label_to_group_jap_ind(label: np.ndarray | list[int]) -> str:
    return VARIETY_LABEL_MAPPING.get(tuple(int(x) for x in label), "Unknown")


def extract_sequences(
    fasta_file: str | Path,
    window_size: int = 8000,
    step_size: int = 8000,
) -> list[dict[str, Any]]:

    fasta_path = Path(fasta_file)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    window_data = collect_windows_from_files(
        file_paths=[fasta_path],
        window_size=window_size,
        step_size=step_size,
        unique=False,
    )

    records: list[dict[str, Any]] = []
    for item in window_data:
        chrom = item["seq_id"]
        start, end = item["seq_start_end"]
        records.append(
            {
                "chrom": chrom,
                "start": int(start),
                "end": int(end),
                "sequence": item["window"]
            }
        )
    print(f"Extracted {len(records)} windows from {fasta_file}")
    return records


def get_embeddings(
    sequences: list[dict[str, Any]],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    save_path: Path | None = None,
) -> np.ndarray:
    if not sequences:
        return np.empty((0, 0), dtype=np.float32)
    
    embeddings: list[np.ndarray] = []
    for seq_info in tqdm(sequences, desc="Extracting embeddings", unit="seq"):
        inputs = tokenizer(seq_info["sequence"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        emb = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        embeddings.append(emb)
    
    final_embeddings = (
        np.concatenate(embeddings, axis=0)
        if embeddings
        else np.empty((0, 0), dtype=np.float32)
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, final_embeddings)
        print(f"Saved embeddings to {save_path}")
    return final_embeddings


def generate_results_df(
    sequences: list[dict[str, Any]],
    labels: np.ndarray,
    probs: np.ndarray,
    ground_truth_label: list[int] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seq_info, label, prob in zip(sequences, labels, probs):
        rows.append(
            {
                "chrom": seq_info["chrom"],
                "start": seq_info["start"],
                "end": seq_info["end"],
                "ground_truth": ",".join(map(str, ground_truth_label)),
                "label": ",".join(map(str, label.tolist())),
                "prob": ",".join(f"{float(x):.6f}" for x in prob.tolist()),
                "group": map_label_to_group_jap_ind(label),
            }
        )

    return pd.DataFrame(rows)



def parse_config() -> object:
    if len(sys.argv) < 2 or not sys.argv[1].startswith("--config"):
        print("Usage: python script/variety_classification_xgb0.py --config config/classification_config.yaml")
        print("Configuration file (YAML format) is required.")
        sys.exit(1)
    
    config_path = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--config" and i + 2 < len(sys.argv):
            config_path = sys.argv[i + 2]
            break
    
    if not config_path:
        print("--config argument requires a file path")
        print("Usage: python script/variety_classification_xgb0.py --config config/classification_config.yaml")
        sys.exit(1)
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert dict to object with nested attribute access
        class Config:
            def __init__(self, data):
                for key, value in data.items():
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
        
        return Config(config_dict)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point for variety classification inference pipeline.

    Orchestrates the complete workflow:
    1. Load configuration from YAML file
    2. Load pre-trained model and tokenizer
    3. For each FASTA file:
       - Extract genomic windows
       - Generate embeddings (with caching)
       - Predict variety labels
       - Evaluate against ground truth if provided
       - Save results and metrics

    Configuration is provided via YAML file passed as: --config config/classification_config.yaml

    Exits with status code 1 if any critical error occurs.
    """
    try:
        config = parse_config()
    except SystemExit:
        return

    try:
        output_dir = Path(config.output.output_dir).expanduser().resolve()
        output_dir = output_dir / config.models.llm_name / f"{config.input.dataset_name}_ws{config.data_processing.window_size//1000}k_step{config.data_processing.step_size//1000}k"
        embedding_dir = Path(config.output.embedding_dir).expanduser().resolve()
        embedding_dir = embedding_dir / config.models.llm_name
        output_dir.mkdir(parents=True, exist_ok=True)
        embedding_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"Loading LLM model from {config.models.llm_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.models.llm_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            config.models.llm_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        if device == "cpu":
            model = model.to(device)
        print("Model loaded successfully")


        all_dfs: pd.DataFrame = pd.DataFrame()
        for index, fasta in enumerate(config.input.fasta_files.path):
            try:
                fasta_path = Path(fasta).expanduser().resolve()
                genome_name = fasta_path.name.split(".")[0]
                print(f"Processing genome: {genome_name}")
                
                # Extract sequences
                sequences = extract_sequences(
                    fasta_file=fasta_path,
                    window_size=config.data_processing.window_size,
                    step_size=config.data_processing.step_size,
                )
                if not sequences:
                    print(f"No windows generated for {fasta_path}, skipping")
                    continue

                # Load or compute embeddings
                embedding_path = embedding_dir / (
                    f"{genome_name}_ws{config.data_processing.window_size//1000}k_step{config.data_processing.step_size//1000}k_embeddings.pt.npy"
                )
                if embedding_path.exists():
                    embeddings = np.load(embedding_path)
                    print(f"Loaded cached embeddings: {embedding_path}")
                else:
                    print("Computing embeddings...")
                    embeddings = get_embeddings(
                        sequences=sequences,  # For first genome, only compute on last 100 windows for testing
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        save_path=embedding_path,
                    )
                
                print("Predicting variety labels...")

                model_path = Path(config.models.rf_model_path).expanduser().resolve()
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
    
                rf_models = joblib.load(model_path)
                labels, probs = predict_with_rf_models(embeddings, rf_models)

                df = generate_results_df(
                    sequences=sequences,
                    labels=labels,
                    probs=probs,
                    ground_truth_label=config.input.fasta_files.label[index],
                )
                all_dfs = pd.concat([all_dfs, df], axis=0, ignore_index=True)
                tsv_path = output_dir / f"{genome_name}_results.tsv"
                df.to_csv(tsv_path, sep="\t", header=True, index=False)

                print(f"Completed {genome_name}")

            except Exception as e:
                print(f"Error processing {fasta}: {e}")
                continue

        metrics_path = str(output_dir / f"result_metrics.json")
        eval_from_tsv(df=all_dfs, T=config.prediction.threshold, output_path=metrics_path)

    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
