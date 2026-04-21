# **ScenarioⅠ: Identification of *indica-japonica* Introgression**

## **1.Task Description**

This case aims to exploit the capacity of the OGR foundation model for fine-scale inference of subspecies origin across the rice genome, enabling the identification of introgression between *indica* (*Oryza sativa* subsp. *indica*) and *japonica* (*Oryza sativa* subsp. *japonica*). Unlike traditional approaches that rely on SNP-based statistics or local sequence alignment, this study starts directly from raw genomic sequences. High-dimensional embeddings are extracted using the OGR model, upon which downstream predictive models are built. This approach enables the capture of deep genetic structural differences at the sequence level, facilitating the identification of potential introgressed regions between subspecies.

## **2.Data Source and Processing**

This study utilizes high-quality, well-annotated assembled rice genomes with clearly defined origins, including:

Data foundation: a collection of high-quality assembled rice genomes

Subpopulation assignment: based on subpopulation labels from the RiceVarMap database

Sample selection: samples overlapping with the 3KRGP (3K rice genome project) were selected, followed by further filtering using whole-genome variation–based principal component analysis (PCA). Finally, each 10 representative samples were selected from both *indica* and temperate *japonica* groups, aiming to preserve within-subpopulation genetic diversity while minimizing potential interference from introgression introduced during breeding history. And one additional representative sample from each of *indica* and *japonica* was selected to construct an independent test set for evaluation.

| Sample Name     | ID_3K          | Region      | Subpop                | Set          | DOI                        |
| --------------- | -------------- | ----------- | --------------------- | ------------ | -------------------------- |
| Aimakang        | B115           | China       | *Indica* I          | Training Set | 10.1038/s41588-025-02365-1 |
| Lucaihao        | B208           | China       | *Indica* I          | Training Set | 10.1038/s41588-025-02365-1 |
| Nantehao        | B062           | China       | *Indica* I          | Training Set | 10.1038/s41588-025-02365-1 |
| Gang_46B        | CX10           | China       | *Indica* I          | Training Set | 10.1101/gr.276015.121      |
| Guangluai_4     | B061           | China       | *Indica* I          | Training Set | 10.1101/gr.276015.121      |
| TAICHUNGNATIVE1 | CX270          | China       | *Indica* I          | Training Set | 10.1101/gr.276015.121      |
| Gui_630         | B242           | China       | *Indica* II         | Training Set | 10.1016/j.cell.2021.04.046 |
| IR64-IL         | CX230          | China       | *Indica* II         | Training Set | 10.1016/j.cell.2021.04.046 |
| Laozaogu        | B246           | China       | *Indica* III        | Training Set | 10.1038/s41588-025-02365-1 |
| LUO_SI_ZHAN     | IRIS_313-11728 | China       | *Indica* III        | Training Set | 10.1101/gr.276015.121      |
| Jindao_1        | B236           | China       | Temperate*Japonica* | Training Set | 10.1038/s41588-025-02365-1 |
| Zhengdao_5      | B240           | China       | Temperate*Japonica* | Training Set | 10.1038/s41588-025-02365-1 |
| Heibiao         | B001           | China       | Temperate*Japonica* | Training Set | 10.1038/s41588-025-02365-1 |
| Annongwangeng_B | B250           | China       | Temperate*Japonica* | Training Set | 10.1101/gr.276015.121      |
| Linguo          | B171           | Italy       | Temperate*Japonica* | Training Set | 10.1038/s41588-025-02365-1 |
| Yueguang        | CX330          | Japan       | Temperate*Japonica* | Training Set | 10.1016/j.cell.2021.04.046 |
| Gongchengxiang  | B045           | Japan       | Temperate*Japonica* | Training Set | 10.1038/s41588-025-02365-1 |
| Qiutianxiaoting | B046           | Japan       | Temperate*Japonica* | Training Set | 10.1101/gr.276015.121      |
| Qingjinzaosheng | B167           | North Korea | Temperate*Japonica* | Training Set | 10.1101/gr.276015.121      |
| MAEKJO          | IRIS_313-10097 | South Korea | Temperate*Japonica* | Training Set | 10.1101/gr.276015.121      |
| Heidu 4         | B081           | China       | *Indica* I          | Test Set     | 10.1038/s41588-025-02365-1 |
| Dandongludao    | B069           | China       | Temperate*Japonica* | Test Set     | 10.1038/s41588-025-02365-1 |

## **3. Task Design**

### **3.1 Overall Framework**

The model is built upon the OneGenome-Rice foundation model. Unlike conventional fine-tuning approaches, this study does not update the parameters of the foundation model. Instead, it directly extracts embeddings from each 8 kb sequence and builds a lightweight downstream predictive model based on these representations, with the core workflow as follows:

![Overall framework and workflow](images/Introgression_Framework.png)

**Data Construction and Partitioning**: Whole-genome sequences from *indica* and *japonica* rice are collected and divided into training and test sets at the individual level (10:1 ratio).

**Genomic Window Segmentation**: Each genome is partitioned into fixed-length sliding windows (8 kb), generating a set of sequence fragments that cover the entire genome.

**Sequence Representation Extraction**: Each 8 kb sequence fragment is encoded using the OGR foundation model to obtain high-dimensional embedding representations.

**Downstream Modeling**: A random forest model is trained on these embeddings to learn the mapping from sequence representations to subpopulation assignment probabilities ($P_{\mathrm{indica}}$: probability of belonging to *indica*; $P_{\mathrm{japonica}}$: probability of belonging to *japonica*).

**Introgression Map Construction and Performance Evaluation**: Predictions are generated on the test set, and the resulting probabilities are visualized along genomic coordinates to construct introgression landscapes. At the same time, indicators such as AUC and ACC are used to quantitatively evaluate the model performance.

### **3.2 Model Evaluation**

Model performance is evaluated on the test set using the true subpopulation labels of each sample. Classification performance is assessed using AUC (Area Under the Curve) and ACC (Accuracy), which together reflect the model’s overall ability to distinguish subpopulation origins.

|           **Test Set**           |       **Classifier**       | **ACC** | **AUC** |
| :------------------------------------: | :------------------------------: | :-----------: | :-----------: |
| 1 Temperate*japonica* + 1 *indica* | Random Forest (n_estimators=100) |     0.804     |     0.794     |

Based on the subpopulation probabilities ($P_{\mathrm{*indica*}}$ and $P_{\mathrm{*japonica*}}$), genomic segments are classified as follows:

- If either probability exceeds 0.8, the segment is assigned to the corresponding subpopulation.
- If both probabilities are below 0.8, or both exceed 0.8, the segment is classified as a low-differentiation region, indicating weak or ambiguous subpopulation signals.

This strategy enables effective identification of: 1) regions with pure origin; 2) potential introgressed segments; 3) conserved shared regions between *japonica* and *indica*

### **3.3 Case Study**

We tried to apply this framework to investigate *indica* introgression in the *japonica* cultivar Yanfeng 47 (YF47), a representative breeding line derived from historical inter-subspecific breeding behavior. These regions were consistently organized in extended blocks rather than isolated loci, indicating that introgression is captured at the segment level, reflecting the introgression of adjacent genomic fragments during breeding.

![Case study illustration (e.g. YF47)](images/Elite_Japonica_Cultivar_YF47_Introgression.png)

## **4. Project structure**

**Directory Tree:**

```
Introgression_Analysis/
├── README.md                          # English documentation
├── README_zh.md                       # Chinese documentation
├── requirements.txt                   # Python dependencies
├── run_train_rf.sh                    # Script to run training pipeline
├── run_variety_inference.sh           # Script to run inference pipeline
│
├── scripts/                           # Main execution scripts
│   ├── train_rf.py                    # Train random forest on embeddings
│   └── variety_inference.py           # Genome-scale inference and metrics
│
├── benchmarks/                        # Embedding extraction utilities
│   ├── __init__.py
│   └── embedding_extract.py           # JSONLDataset and embedding utils
│
├── utils/                             # Data processing utilities
│   ├── __init__.py
│   ├── genomic_window_egmentation.py  # Convert FASTA to windowed JSONL
│   ├── compute_metrics.py             # Metrics computation helpers
│   └── utils.py                       # General utility functions
│
├── config/                            # Configuration files
│   ├── train_rf_config.yaml           # Training hyperparameters & paths
│   └── variety_inference_config.yaml  # Inference parameters
│  
│
├── data/                              # Input/output data directory
│   ├── datasets_info.yaml             # Declared dataset formats
│   └── rice_introgression_jap-ind/    # Dataset splits (after preprocessing)
│       ├── train.jsonl                # Training JSONL (generated)
│       └── test.jsonl                 # Testing JSONL (generated)
│
├── fasta_data/                        # Input FASTA files
│   ├── japonica.train01.genome.fa     # Japonica training genome
│   ├── ...               
│   ├── japonica.train10.genome.fa
│   │
│   ├── indica.train01.genome.fa       # Indica training genome
│   ├── ...               
│   ├── indica.train10.genome.fa
│   │
│   ├── japonica.test01.genome.fa      # Japonica test genome
│   └── indica.test01.genome.fa        # Indica test genome
│
├── model/                             # Foundation model weights
│   └── rice_1B_stage2_8k_hf/          # OneGenome-Rice model directory
│       ├── config.json
│       ├── model.safetensors          # Model weights
│       └── tokenizer.json
│
├── embedding_path/                    # Cached embeddings (generated)
│   └── rice_1B_stage2_8k_hf/          # Embeddings per model
│       ├── rice_introgression_jap-ind-12layer_train.pt
│       └── rice_introgression_jap-ind-12layer_test.pt
│
├── results_path/                      # Training & inference results (generated)
│   └── rice_1B_stage2_8k_hf/
│       ├── last_epoch_model/          # Trained RF models
│       │   ├── rice_introgression_jap-ind-12layer.rf.pkl
│       │   └── training_results.tsv
│       └── rice_introgression_jap_ind_ws8k_step8k/  # Inference outputs
│           ├── japonica.test01.genome_results.tsv   # Per-window predictions
│           ├── indica.test01.genome_results.tsv
│           └── result_metrics.json    # Overall metrics
│
└── images/                            # Documentation images
    ├── Introgression_Analysis-a.png   # Framework diagram
    └── Introgression_Analysis-b.png   # Case study illustration
```

**Key directories:**

- 📁 **data/**: Contains or will contain JSONL training/test splits
- 📁 **fasta_data/**: Place your FASTA files here before running `genomic_window_egmentation.py`
- 📁 **model/**: Download foundation model weights and place here
- 📁 **results_path/**: Auto-generated directory with trained models and inference outputs
- 📁 **embedding_path/**: Auto-generated directory with cached embeddings

**Detailed file roles:**

| Path                                     | Role                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/train_rf.py`                  | Loads the foundation model, extracts layer embeddings for `train` / `test` JSONL splits, trains multilabel random forests per layer, evaluates on the held-out split, writes `training_results.tsv` and `*.rf.pkl` checkpoints. **Requires** windowed JSONL under `data/<dataset>/` (typically produced first by `utils/genomic_window_egmentation.py`). |
| `scripts/variety_inference.py`         | Sliding windows over input FASTA, embedding extraction, RF prediction, per-window TSV plus aggregate `result_metrics.json`.                                                                                                                                                                                                                                              |
| `benchmarks/embedding_extract.py`      | `JSONLDataset` and embedding extraction utilities used by training.                                                                                                                                                                                                                                                                                                      |
| `utils/genomic_window_egmentation.py`  | CLI to build `train.jsonl` / `test.jsonl` from grouped FASTA paths.                                                                                                                                                                                                                                                                                                    |
| `utils/compute_metrics.py`             | Shared metrics helpers; can aggregate existing TSVs from the command line.                                                                                                                                                                                                                                                                                                 |
| `config/train_rf_config.yaml`          | Training: model path, dataset list, embedding and results directories, RF hyperparameters, layers to evaluate.                                                                                                                                                                                                                                                             |
| `config/variety_inference_config.yaml` | Inference: FASTA list, labels for evaluation, paths to LLM and trained RF, window/step sizes, output dirs, probability threshold.                                                                                                                                                                                                                                          |
| `data/datasets_info.yaml`              | Declares supported dataset names and per-dataset keys (`seq_key`, `label_key`, splits).                                                                                                                                                                                                                                                                                |

## **5. Quick Start**

### **Quick Environment Setup**

Run commands below from the **repository root**. Python 3.11 + conda is recommended.

#### **Option 1 (Recommended): Install from `requirements.txt`**

```bash
conda create -n env_introgression_analysis python=3.11 -y
conda activate env_introgression_analysis
pip install --upgrade pip
pip install -r requirements.txt
```

Best for: online environments where you want dependency versions consistent with the repository.

#### **Option 2: Use the setup script**

```bash
bash create_env.sh
```

The script will:

- create and activate `env_introgression_analysis`;
- install CUDA 12.8 PyTorch (`torch/torchvision/torchaudio`);
- install the remaining Python dependencies.

Best for: quick reproducible setup with fewer manual commands.

#### **Option 3: Offline PyTorch install (slow/no network servers)**

Best for: servers with slow or restricted internet, where **PyTorch installation is the main bottleneck**.Recommended approach: install only the PyTorch trio offline; install remaining packages via regular `pip`.

1) On a machine with internet, download PyTorch wheels:

- from the official CUDA 12.8 index:
  - `torch-2.10.0+cu128-cp311-cp311-<platform>.whl`
  - `torchvision-0.25.0+cu128-cp311-cp311-<platform>.whl`
  - `torchaudio-2.10.0+cu128-cp311-cp311-<platform>.whl`

2) On the target server, install PyTorch first (example):

```bash
pip install torch-2.10.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl \
            torchvision-0.25.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl \
            torchaudio-2.10.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
```

3) Install remaining dependencies online on the target server:

```bash
pip install -r requirements.txt
```

### **Minimal Example (5 Steps)**

```bash
# 1. Prepare FASTA files
#    Training: fasta_data/japonica.train01-10.genome.fa (japonica), fasta_data/indica.train01-10.genome.fa (indica)
#    Testing:  fasta_data/japonica.test01.genome.fa (japonica), fasta_data/indica.test01.genome.fa (indica)

# 2. Generate windowed training/test datasets
python utils/genomic_window_egmentation.py --dataset-name rice_introgression --output-dir data

# 3. Train the random forest model
python scripts/train_rf.py --config config/train_rf_config.yaml

# 4. Run genome-scale inference
python scripts/variety_inference.py --config config/variety_inference_config.yaml

# 5. Check results
ls results_path/rice_1B_stage2_8k_hf/rice_introgression_jap_ind_ws8k_step8k/
cat results_path/rice_1B_stage2_8k_hf/rice_introgression_jap_ind_ws8k_step8k/result_metrics.json
```

**Expected output structure:**

```
results_path/
├── rice_1B_stage2_8k_hf/
│   ├── last_epoch_model/              # Trained RF models
│   │   ├── rice_introgression_jap-ind-12layer.rf.pkl
│   │   └── training_results.tsv
│   └── rice_introgression_jap_ind_ws8k_step8k/
│       ├── japonica.test01.genome_results.tsv      # Per-window predictions
│       ├── indica.test01.genome_results.tsv
│       └── result_metrics.json         # Overall metrics (ACC, AUC)
```

## **6. Usage**

Run all commands from the **repository root** so relative paths in the YAML files resolve correctly.

**Order of operations:** run `utils/genomic_window_egmentation.py` first to build `train.jsonl` and `test.jsonl`, then run `scripts/train_rf.py`. Training reads only prepared JSONL on disk; it does not split raw FASTA for you.

### **6.1 Environment**

See **Section 5** for environment setup instructions. Quick summary:

- **Recommended**: Option 1, `pip install -r requirements.txt` (most stable and consistent)
- **Automated**: Option 2, run `bash create_env.sh` (quick reproducible setup)
- **Offline**: Option 3, pre-download `.whl` for PyTorch and install remaining deps with `pip`

### **6.2 Foundation model and data layout**

1. **Foundation model**Point `model.path` / `models.llm_path` in the YAML files to a local Hugging Face–style directory (default in configs: `model/rice_1B_stage2_8k_hf`). Inference uses `local_files_only=True`, so weights must be present on disk.
2. **Step 1 — Split genomes into windows (`utils/genomic_window_egmentation.py`)**Before `scripts/train_rf.py`, you must prepare the training and test JSONL files. The usual path is to slice FASTA genomes into fixed windows with this script.

   Place FASTA files under `fasta_data/` (or edit `FASTA_GROUPS` / pass `--fasta-root`). Then run:

   ```bash
   python utils/genomic_window_egmentation.py --dataset-name rice_introgression --output-dir data
   ```

   This writes `data/rice_introgression_jap-ind/train.jsonl` and `data/rice_introgression_jap-ind/test.jsonl` (8 kb windows; train step 4 kb, test step 8 kb per script defaults). Paths must match `dataset.data_path` and `dataset.eval_datasets` in `config/train_rf_config.yaml`.
3. **Training JSONL format**Each line must be a JSON object with at least the keys configured in `data/datasets_info.yaml` (`sequence` and `label` for `rice_introgression_jap-ind`). Labels are multilabel vectors used by `RandomForestClassifier` (one RF per output dimension). If you already have compatible JSONL from another pipeline, you may place them under `data/<dataset_name>/` instead of running the window script.

   **JSONL Format Specification:**

   **File location:** `data/rice_introgression_jap-ind/train.jsonl` or `test.jsonl`

   **Each line is a JSON object with this structure:**

   ```json
   {
     "sequence": "ATCGATCGATCG...",
     "label": [1, 0]
   }
   ```

   **Field details:**

   | Field        | Type            | Description                                            | Example                                                |
   | ------------ | --------------- | ------------------------------------------------------ | ------------------------------------------------------ |
   | `sequence` | string          | DNA sequence (case-insensitive, A/T/G/C/N only)        | `"ATCGATCG"`                                         |
   | `label`    | array of 2 ints | Multilabel vector:`[japonica_binary, indica_binary]` | `[1, 0]` (pure japonica) or `[0, 1]` (pure indica) |

   **Example JSONL file:**


   ```
   {"sequence": "ATCGATCGATCGATCGATCGATCGATCG", "label": [1, 0]}
   {"sequence": "TCGATCGATCGATCGATCGATCGATCGA", "label": [1, 0]}
   {"sequence": "CGATCGATCGATCGATCGATCGATCGAT", "label": [0, 1]}
   {"sequence": "CGATTACATGGTAGCTAGCTAAGCATTA", "label": [0, 1]}
   ```

   **Automatic generation via `genomic_window_egmentation.py`:**
   The script automatically converts FASTA files into this format:

   - Reads multi-sequence FASTA
   - Trims leading/trailing N bases
   - Creates fixed-length sliding windows (default: 8 kb for training, same for test)
   - Assigns labels based on the source file (japonica vs. indica)
   - Outputs JSONL with one JSON object per window
4. **Edit output paths in YAML**
   In `config/train_rf_config.yaml`, set `embedding.output_dir` and `output.result_dir` (defaults `embedding_path`, `results_path`). The trained RF path used later is under `results_path/<model.name>/last_epoch_model/`.

### **6.3 Train the random forest (`scripts/train_rf.py`)**

Run this **after** `train.jsonl` / `test.jsonl` are generated.

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

### **6.4 Genome-scale inference (`scripts/variety_inference.py`)**

1. Set `models.rf_model_path` in `config/variety_inference_config.yaml` to the trained checkpoint path (must match `*-<layer>layer.rf.pkl`, e.g. `rice_introgression_jap-ind-12layer.rf.pkl` for layer 12).
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

### **6.5 Output Data Formats**

**Per-genome TSV (`*_results.tsv`):**

```
chrom	start	end	ground_truth	label	prob_japonica	prob_indica	group
chr1	0	8000	[1,0]	japonica	0.92	0.08	japonica
chr1	8000	16000	[1,0]	japonica	0.85	0.15	japonica
chr1	16000	24000	[1,0]	introgressed	0.45	0.55	indica
```

**Column definitions:**

| Column            | Type   | Description                                                |
| ----------------- | ------ | ---------------------------------------------------------- |
| `chrom`         | string | Chromosome/sequence identifier                             |
| `start`         | int    | Window start position (0-based)                            |
| `end`           | int    | Window end position (exclusive)                            |
| `ground_truth`  | string | True label `[japonica_binary, indica_binary]`            |
| `label`         | string | Predicted class:`japonica`, `indica`, or `ambiguous` |
| `prob_japonica` | float  | Probability of*japonica* origin (0.0–1.0)               |
| `prob_indica`   | float  | Probability of*indica* origin (0.0–1.0)                 |
| `group`         | string | Classification result                                      |

### **6.6 Optional: aggregate metrics from TSV files**

```bash
python utils/compute_metrics.py path/to/results.tsv -t 0.5 -o metrics.json
```

Accepts multiple `.tsv` paths or directories (recursive); concatenates rows then runs the same logic as inference-time evaluation.
