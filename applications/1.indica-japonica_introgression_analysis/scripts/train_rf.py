#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import os
import yaml
import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from transformers import AutoModel, AutoTokenizer
torch.use_deterministic_algorithms(True)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.utils import catter, predict_with_rf_models
from utils.compute_metrics import compute_metrics
from benchmarks.embedding_extract import JSONLDataset, extract_embeddings, collate_fn


# -----------------------------------------------
# Random Forest Training and Prediction Functions
# -----------------------------------------------
def train_rf_classifier(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: dict,
    dataset_name: str,
    layer: int,
    n_estimators: int = 100,
    random_state: int = 42
) -> list:
    """Train Random Forest classifier on embeddings."""
    # Convert embeddings to numpy using catter function
    pooling_dim = config.get('evaluation', {}).get('pooled_embeddings_cat_dim', 0) if isinstance(config, dict) else config.evaluation.pooled_embeddings_cat_dim
    X_train_np = catter(X_train, pooling_dim).cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    n_labels = y_train_np.shape[1]
    rf_models = []

    for i in range(n_labels):
        print(f"  Training Random Forest classifier for label {i}...")
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train_np, y_train_np[:, i])
        rf_models.append(rf)

    # Save models
    os.makedirs(
        f"{config['output']['result_dir']}/{config['model']['name']}/last_epoch_model",
        exist_ok=True
    )
    model_save_path = f"{config['output']['result_dir']}/{config['model']['name']}/last_epoch_model/{dataset_name}-{layer}layer.rf.pkl"
    joblib.dump(rf_models, model_save_path)
    print(f"All {n_labels} models saved to {model_save_path}")
    
    return rf_models


def calculate_metrics(
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    y_test: torch.Tensor,
    dataset_name: str,
    layer: int,
    classifier_type: str
) -> dict:
    """Calculate evaluation metrics."""
    y_test_np = y_test.cpu().numpy()
    base_metrics = compute_metrics(y_probs, y_test_np, y_pred)
    return {
        'task': dataset_name,
        'layer': layer,
        'classifier': classifier_type,
        'accuracy': base_metrics['accuracy'],
        'roc_auc': base_metrics['auc_roc'],
        'precision': base_metrics['precision'],
        'recall': base_metrics['recall'],
        'f1': base_metrics['f1'],
        'mcc': base_metrics['mcc'],
    }


def train_and_evaluate_dataset_layer(
    dataset_name: str,
    layer: int,
    config: dict,
    device: str
) -> dict:
    """Train and evaluate RF classifier on a dataset-layer pair."""
    print(f"\n[TRAIN] Training {dataset_name} layer {layer}...")
    
    embedding_dir = f"{config['embedding']['output_dir']}/{config['model']['name']}"

    # Load embeddings and labels
    data_train_path = f"{embedding_dir}/{dataset_name}-{layer}layer_train.pt"
    data_test_path = f"{embedding_dir}/{dataset_name}-{layer}layer_test.pt"

    print(f"Loading training data from {data_train_path}...")
    X_train = torch.load(data_train_path, map_location=device)["embeddings"]
    y_train = torch.load(data_train_path, map_location=device)["labels"]
    
    print(f"Loading test data from {data_test_path}...")
    X_test = torch.load(data_test_path, map_location=device)["embeddings"]
    y_test = torch.load(data_test_path, map_location=device)["labels"]

    num_classes = len(y_train[0])
    print(f"Dataset: {dataset_name}, Layer: {layer}, Num classes: {num_classes}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Train RF classifier
    rf_n_estimators = config['classifiers']['random_forest'].get('n_estimators', 100) if hasattr(config, '__getitem__') else config.classifiers.random_forest.n_estimators
    random_state = config['environment'].get('seed', 42) if hasattr(config, '__getitem__') else config.environment.seed
    rf_models = train_rf_classifier(
        X_train, y_train, config, dataset_name, layer,
        n_estimators=rf_n_estimators,
        random_state=random_state
    )
    
    # Make predictions on test set
    pooling_dim = config['evaluation'].get('pooled_embeddings_cat_dim', 0) if hasattr(config, '__getitem__') else config.evaluation.pooled_embeddings_cat_dim
    X_test_np = catter(X_test, pooling_dim).cpu().numpy()
    y_pred, y_probs = predict_with_rf_models(X_test_np, rf_models)
    
    # Calculate metrics
    classifier_type = config.get('evaluation', {}).get('classifier_type', 'RF') if isinstance(config, dict) else config.evaluation.classifier_type
    metrics = calculate_metrics(
        y_pred, y_probs, y_test,
        dataset_name, layer, classifier_type
    )
    
    print(f"[TRAIN] Metrics for {dataset_name} layer {layer}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    
    return metrics


def parse_config() -> object:
    """Parse configuration from YAML file passed as --config argument."""
    config_path = None
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            break
    
    if not config_path:
        print("--config argument requires a file path")
        print("Usage: python script/train_rf.py --config config/train_rf_config.yaml")
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
            
            def __getitem__(self, key):
                return getattr(self, key)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config_obj = Config(config_dict)
        return config_obj
    
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)


def load_datasets_feature(config: object) -> dict:
    """Load dataset feature information from YAML file."""
    try:
        datasets_feature_path = config['dataset']['feature_path']
        with open(datasets_feature_path, "r", encoding="utf-8") as f:
            datasets_info = yaml.safe_load(f)
        
        # Build dataset info dict
        dataset_feature_dict = {}
        if 'dataset_feature' in datasets_info:
            for dataset_name, feature in datasets_info['dataset_feature'].items():
                dataset_feature_dict[dataset_name] = feature
        
        # Add metadata
        config.all_datasets_feature = {
            'long_sequence_dataset': datasets_info.get('long_sequence_dataset', []),
            'super_long_sequence_dataset': datasets_info.get('super_long_sequence_dataset', []),
        }
        config.dataset_info = dataset_feature_dict
        config.model_config = {'num_hidden_layers': config['model']['hidden_layers']}
        
        # Add backward compatibility: create flat config attributes for JSONLDataset
        config.dataset_path = config['dataset']['data_path']
        config.embedding_output_dir = config['embedding']['output_dir']
        config.batch_size = config['environment'].get('batch_size', 8)
        config.embedding_extract_split = config['embedding'].get('extract_split', 50000)
        config.layer_to_eval = config['evaluation']['layers']
        
        return config
    
    except Exception as e:
        print(f"Failed to load dataset features: {e}")
        sys.exit(1)


def extract_embeddings_for_dataset_split(
    dataset_name: str,
    split: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    config: object,
    gpu_id: int = 0
) -> tuple[dict, torch.Tensor]:
    """Extract embeddings for a dataset split."""
    print(f"\nExtracting embeddings for {dataset_name} {split}...")
    
    # Load dataset
    dataset = JSONLDataset(dataset_name, split, config)
    if dataset is None or len(dataset) == 0:
        print(f"Dataset {dataset_name} {split} is empty, skipping...")
        return None, None
    
    seq_number = dataset.get_seq_number()
    batch_size = config['environment'].get('batch_size', 8)
    
    # Adjust batch size for large sequences
    if dataset_name in config.all_datasets_feature.get('long_sequence_dataset', []):
        batch_size = batch_size // 2
    if dataset_name in config.all_datasets_feature.get('super_long_sequence_dataset', []):
        batch_size = 1
    
    # Further adjust for multiple sequences
    batch_size = batch_size // seq_number if batch_size // seq_number > 0 else 1
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )
    
    # Extract embeddings for all layers
    layer_to_eval = config['evaluation']['layers']
    tqdm_print = f"Extracting {dataset_name} {split}"
    
    embeddings_dict, labels = extract_embeddings(
        model, loader, device, gpu_id, dataset_name,
        layer_to_eval, seq_number, config, tqdm_print
    )
    
    return embeddings_dict, labels


def save_embeddings_to_disk(
    embeddings_dict: dict,
    labels: torch.Tensor,
    dataset_name: str,
    split: str,
    config: object
) -> bool:
    """Save extracted embeddings to disk."""
    output_dir = f"{config['embedding']['output_dir']}/{config['model']['name']}"
    os.makedirs(output_dir, exist_ok=True)
    
    for layer, embeddings in embeddings_dict.items():
        data = {"embeddings": embeddings, "labels": labels}
        save_path = f"{output_dir}/{dataset_name}-{layer}layer_{split}.pt"
        torch.save(data, save_path)
        print(f"Saved embeddings to {save_path}")
    
    return True


def main() -> None:
    """Main entry point for RF training pipeline.
    
    Orchestrates the complete workflow:
    1. Load configuration from YAML file
    2. Load pre-trained model and tokenizer
    3. For each training dataset:
       - Extract embeddings for train and test splits
       - Train Random Forest models for each layer
       - Evaluate on test set
       - Save results and models
    
    Configuration is provided via YAML file passed as: --config config/train_rf_config.yaml
    """
    try:
        print("[TRAIN] Starting RF training pipeline...")
        
        # Load configuration
        config = parse_config()
        print(f"Loaded configuration from {sys.argv[sys.argv.index('--config') + 1]}")
        
        # Load dataset features
        config = load_datasets_feature(config)
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create output directories
        os.makedirs(config['embedding']['output_dir'], exist_ok=True)
        os.makedirs(f"{config['output']['result_dir']}/{config['model']['name']}", exist_ok=True)
        
        # Load LLM model and tokenizer
        print(f"Loading model from {config['model']['path']}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config['model']['path'],
            trust_remote_code=True
        )
        
        model = AutoModel.from_pretrained(
            config['model']['path'],
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        ).eval()
        
        if device == "cpu":
            model = model.to(device)
        print("Model loaded successfully")
        try:
            config.model_config['hidden_size'] = int(model.config.hidden_size)
        except Exception:
            raise ValueError("Cannot infer hidden_size from model.config.hidden_size")

        # Get GPU ID from config or use 0
        gpu_list = config['environment'].get('gpu_list', [])
        gpu_id = gpu_list[0] if gpu_list else 0
        
        # Process each training dataset
        eval_datasets = config['dataset'].get('eval_datasets', [])
        if not eval_datasets:
            print("[TRAIN] No datasets configured for training")
            return
        
        all_results = []
        
        for dataset_name in eval_datasets:
            print(f"\n{'='*60}")
            print(f"[TRAIN] Processing dataset: {dataset_name}")
            print(f"{'='*60}")
            
            try:
                # Check if embeddings already exist
                embedding_dir = f"{config['embedding']['output_dir']}/{config['model']['name']}"
                layer_to_eval = config['evaluation']['layers']
                
                # Extract embeddings for train and test splits if needed
                split_failed = False
                for split in ['train', 'test']:
                    embeddings_exist = all(
                        os.path.exists(f"{embedding_dir}/{dataset_name}-{layer}layer_{split}.pt")
                        for layer in layer_to_eval
                    )
                    
                    if not embeddings_exist:
                        print(f"\n[TRAIN] Extracting embeddings for {split} split...")
                        embeddings_dict, labels = extract_embeddings_for_dataset_split(
                            dataset_name, split, model, tokenizer, device, config, gpu_id
                        )
                        
                        if embeddings_dict is not None and labels is not None:
                            save_embeddings_to_disk(embeddings_dict, labels, dataset_name, split, config)
                        else:
                            print(f"[TRAIN] Failed to extract embeddings for {split} split")
                            split_failed = True
                            break
                    else:
                        print(f"[TRAIN] Embeddings for {split} split already exist")
                if split_failed:
                    print(f"[TRAIN] Skipping dataset {dataset_name} because embeddings are incomplete")
                    continue
                
                # Train and evaluate for each layer
                for layer in layer_to_eval:
                    metrics = train_and_evaluate_dataset_layer(
                        dataset_name, layer, config, device
                    )
                    all_results.append(metrics)
            
            except Exception as e:
                print(f"[TRAIN] Error processing dataset {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save results to TSV file
        if all_results:
            results_df = pd.DataFrame(all_results)
            result_path = f"{config['output']['result_dir']}/{config['model']['name']}/training_results.tsv"
            results_df = results_df.sort_values(['task', 'layer'])
            results_df.to_csv(result_path, sep="\t", index=False)
            print(f"\n[TRAIN] Results saved to {result_path}")
            print(results_df.to_string())
            print("\n[TRAIN] ✅ Training pipeline completed successfully!")
        else:
            print("\n[TRAIN] ⚠️ Training pipeline completed but no results were generated!")
        
    except Exception as e:
        print(f"[TRAIN] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
