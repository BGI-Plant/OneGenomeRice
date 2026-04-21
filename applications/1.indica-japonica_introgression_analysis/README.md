# **ScenarioⅠ: *Indica*-*Japonica* Introgression Identification**

> Chinese version: [README_zh.md](README_zh.md)

## **1.Task Description**

This case aims to exploit the capacity of the OGR foundation model for fine-scale inference of subspecies origin across the rice genome, enabling the identification of introgression between *indica* (*Oryza sativa* subsp. *indica*) and *japonica* (*Oryza sativa* subsp. *japonica*). Unlike traditional approaches that rely on SNP-based statistics or local sequence alignment, this study starts directly from raw genomic sequences. High-dimensional embeddings are extracted using the OGR model, upon which downstream predictive models are built. This approach enables the capture of deep genetic structural differences at the sequence level, facilitating the identification of potential introgressed regions between subspecies.

## **2.Data Source and Processing**

This study utilizes high-quality, well-annotated assembled rice genomes with clearly defined origins, including:

Data foundation: a collection of high-quality assembled rice genomes

Subpopulation assignment: based on subpopulation labels from the RiceVarMap database

Sample selection: samples overlapping with the 3KRGP (3K rice genome project) were selected, followed by further filtering using whole-genome variation–based principal component analysis (PCA). Finally, each 10 representative samples were selected from both *indica* and temperate *japonica* groups, aiming to preserve within-subpopulation genetic diversity while minimizing potential interference from introgression introduced during breeding history. And one additional representative sample from each of *indica* and *japonica* was selected to construct an independent test set for evaluation.

## **3. Task Design**

### **(1) Overall Framework**

The model is built upon the OneGenome-Rice foundation model. Unlike conventional fine-tuning approaches, this study does not update the parameters of the foundation model. Instead, it directly extracts embeddings from each 8 kb sequence and builds a lightweight downstream predictive model based on these representations, with the core workflow as follows:

![Overall framework and workflow](images/Introgression_Framework.png)

**Data Construction and Partitioning**: Whole-genome sequences from *indica* and *japonica* rice are collected and divided into training and test sets at the individual level (10:1 ratio).

**Genomic Window Segmentation**: Each genome is partitioned into fixed-length sliding windows (8 kb), generating a set of sequence fragments that cover the entire genome.

**Sequence Representation Extraction**: Each 8 kb sequence fragment is encoded using the OGR foundation model to obtain high-dimensional embedding representations.

**Downstream Modeling**: A random forest model is trained on these embeddings to learn the mapping from sequence representations to subpopulation assignment probabilities ($P_{\mathrm{indica}}$: probability of belonging to *indica*; $P_{\mathrm{japonica}}$: probability of belonging to *japonica*).

**Introgression Map Construction and Performance Evaluation**: Predictions are generated on the test set, and the resulting probabilities are visualized along genomic coordinates to construct introgression landscapes. At the same time, indicators such as AUC and ACC are used to quantitatively evaluate the model performance.

### **(2) Model Evaluation**

Model performance is evaluated on the test set using the true subpopulation labels of each sample. Classification performance is assessed using AUC (Area Under the Curve) and ACC (Accuracy), which together reflect the model’s overall ability to distinguish subpopulation origins.


| **Test Set** | **Classifier** | **ACC** | **AUC** |
|:---:|:---:|:---:|:---:|
| 1 Temperate *japonica* + 1 *indica* | Random Forest (n_estimators=100) | 0.804 | 0.794 |


Based on the subpopulation probabilities ($P_{\mathrm{*indica*}}$ and $P_{\mathrm{*japonica*}}$), genomic segments are classified as follows:

- If either probability exceeds 0.8, the segment is assigned to the corresponding subpopulation.  
- If both probabilities are below 0.8, or both exceed 0.8, the segment is classified as a low-differentiation region, indicating weak or ambiguous subpopulation signals.

This strategy enables effective identification of: 1) regions with pure origin; 2) potential introgressed segments; 3) conserved shared regions between *japonica* and *indica*

### **(4) Case Study**

We tried to apply this framework to investigate *indica* introgression in the *japonica* cultivar Yanfeng 47 (YF47), a representative breeding line derived from historical inter-subspecific breeding behavior. These regions were consistently organized in extended blocks rather than isolated loci, indicating that introgression is captured at the segment level, reflecting the introgression of adjacent genomic fragments during breeding.

![Case study illustration (e.g. YF47)](images/Elite_Japonica_Cultivar_YF47_Introgression.png)

## **4. Project structure**

| Path | Role |
| --- | --- |
| `scripts/train_rf.py` | Loads the foundation model, extracts layer embeddings for `train` / `test` JSONL splits, trains multilabel random forests per layer, evaluates on the held-out split, writes `training_results.tsv` and `*.rf.pkl` checkpoints. **Requires** windowed JSONL under `data/<dataset>/` (typically produced first by `utils/genomic_window_egmentation.py`). |
| `scripts/variety_inference.py` | Sliding windows over input FASTA, embedding extraction, RF prediction, per-window TSV plus aggregate `result_metrics.json`. |
| `benchmarks/embedding_extract.py` | `JSONLDataset` and embedding extraction utilities used by training. |
| `utils/genomic_window_egmentation.py` | CLI to build `train.jsonl` / `test.jsonl` from grouped FASTA paths. |
| `utils/compute_metrics.py` | Shared metrics helpers; can aggregate existing TSVs from the command line. |
| `config/train_rf_config.yaml` | Training: model path, dataset list, embedding and results directories, RF hyperparameters, layers to evaluate. |
| `config/variety_inference_config.yaml` | Inference: FASTA list, labels for evaluation, paths to LLM and trained RF, window/step sizes, output dirs, probability threshold. |
| `data/datasets_info.yaml` | Declares supported dataset names and per-dataset keys (`seq_key`, `label_key`, splits). |

## **5. Usage**

Run all commands from the **repository root** so relative paths in the YAML files resolve correctly.

**Order of operations:** run `utils/genomic_window_egmentation.py` first to build `train.jsonl` and `test.jsonl`, then run `scripts/train_rf.py`. Training reads only prepared JSONL on disk; it does not split raw FASTA for you.

### **5.1 Environment**

- Python 3 with **PyTorch**, **Hugging Face Transformers**, **scikit-learn**, **pandas**, **NumPy**, **PyYAML**, **tqdm**, **joblib**, and **Biopython** (used when preparing windows from FASTA).
- A **CUDA** GPU is recommended for embedding extraction; the code falls back to CPU if CUDA is unavailable.
- The repository includes `requirements.txt` as a frozen export from one machine (many optional GPU stack pins). On a new machine you may prefer a clean environment and installing only the packages your workflow needs, or curating a shorter dependency list.

### **5.2 Foundation model and data layout**

1. **Foundation model**  
   Point `model.path` / `models.llm_path` in the YAML files to a local Hugging Face–style directory (default in configs: `model/rice_1B_stage2_8k_hf`). Inference uses `local_files_only=True`, so weights must be present on disk.

2. **Step 1 — Split genomes into windows (`utils/genomic_window_egmentation.py`)**  
   Before `scripts/train_rf.py`, you must prepare the training and test JSONL files. The usual path is to slice FASTA genomes into fixed windows with this script.

   Place FASTA files under `fasta_data/` (or edit `FASTA_GROUPS` / pass `--fasta-root`). Then run:

   ```bash
   python utils/genomic_window_egmentation.py --dataset-name rice_introgression --output-dir data
   ```

   This writes `data/rice_introgression_jap-ind/train.jsonl` and `data/rice_introgression_jap-ind/test.jsonl` (8 kb windows; train step 4 kb, test step 8 kb per script defaults). Paths must match `dataset.data_path` and `dataset.eval_datasets` in `config/train_rf_config.yaml`.

3. **Training JSONL format**  
   Each line must be a JSON object with at least the keys configured in `data/datasets_info.yaml` (`sequence` and `label` for `rice_introgression_jap-ind`). Labels are multilabel vectors used by `RandomForestClassifier` (one RF per output dimension). If you already have compatible JSONL from another pipeline, you may place them under `data/<dataset_name>/` instead of running the window script.

4. **Edit output paths in YAML**  
   In `config/train_rf_config.yaml`, set `embedding.output_dir` and `output.result_dir` (defaults `embedding_path`, `results_path`). The trained RF path used later is under `results_path/<model.name>/last_epoch_model/`.

### **5.3 Random forest training (`scripts/train_rf.py`)**

Run this **after** windowed `train.jsonl` / `test.jsonl` exist (Section 5.2, step 1).

```bash
python scripts/train_rf.py --config config/train_rf_config.yaml
```

Pipeline behavior:

- If `embedding_path/<model.name>/<dataset>-<L>layer_train.pt` (and `_test.pt`) are missing for each layer `L` in `evaluation.layers`, embeddings are computed from JSONL and saved as `.pt` tensors.
- If those files already exist, embedding extraction is skipped for that split.
- Random forests are saved as `.../last_epoch_model/<dataset>-<L>layer.rf.pkl` (a `joblib` list of one `RandomForestClassifier` per label dimension).
- Metrics are appended to `results_path/<model.name>/training_results.tsv`.

You can also run:

```bash
bash run_train_rf.sh
```

### **5.4 Genome-scale inference (`scripts/variety_inference.py`)**

1. Copy or symlink the trained checkpoint into `config/variety_inference_config.yaml` as `models.rf_model_path` (must match the `*-<layer>layer.rf.pkl` produced during training, e.g. `rice_introgression_jap-ind-12layer.rf.pkl` for layer 12).
2. Set `input.fasta_files.path` to one or more FASTA files and `input.fasta_files.label` to the same-length list of ground-truth multilabel vectors for metric computation (`[1, 0]` → *Japonica*, `[0, 1]` → *Indica* per `VARIETY_LABEL_MAPPING` in the script).
3. Adjust `data_processing.window_size` / `step_size` (defaults 8000 bp) and `prediction.threshold` for binarizing probabilities in `eval_from_tsv`.

```bash
python scripts/variety_inference.py --config config/variety_inference_config.yaml
```

Outputs:

- Per-genome TSV under `results_path/<llm_name>/<dataset_name>_ws…k_step…k/<genome>_results.tsv` (columns include `chrom`, `start`, `end`, `ground_truth`, `label`, `prob`, `group`).
- Cached embeddings as NumPy arrays under `embedding_path/<llm_name>/`.
- `result_metrics.json` in the same run output directory (overall and optional filtered subset).

Shell helper:

```bash
bash run_variety_inference.sh
```

### **5.5 Optional: aggregate metrics from TSV files**

```bash
python utils/compute_metrics.py path/to/results.tsv -t 0.5 -o metrics.json
```

Accepts multiple `.tsv` paths or directories (recursive); concatenates rows then runs the same logic as inference-time evaluation.
