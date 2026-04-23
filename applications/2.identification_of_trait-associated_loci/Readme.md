# Trait-Associated Loci Finding

## 1. Overview

This repository demonstrates a reproducible workflow for identifying rice candidate loci from bidirectional attention signals produced by OneGenome-Rice. The workflow reconstructs sample-specific sequences from variants, extracts forward and reverse-complement attention, performs position-level group comparisons, and summarizes gene-level differential signals in selected candidate regions.

## 2. Data

All required inputs are placed under `Data/`.

| File | Description |
|:--|:--|
| `samples.vcf.gz` | VCF subset for the selected rice samples and candidate regions |
| `samples.vcf.gz.tbi` | Tabix index for the VCF |
| `phenotype.tsv` | Two-column phenotype/group file: `SampleID` and `Trait` |
| `candidate_regions.bed` | Four selected 40 kb genomic intervals |
| `osa1_r7.asm.fa.gz` | OSA1/R7 reference genome FASTA used for pseudo-sequence reconstruction |
| `osa1_r7.asm.fa.gz.fai` / `osa1_r7.asm.fa.gz.gzi` | FASTA random-access indexes generated from `osa1_r7.asm.fa.gz` |
| `chr06.gff3.gz` | Chr6 MSU Rice Genome Annotation Project osa1r6 gene annotation used for gene-level scoring |
| `OneGenomeRice_model/` | Optional symlink or local directory for the pretrained OneGenome-Rice 8 kb Hugging Face model |

The reference FASTA is not required to be committed with the repository. If it is missing, `0.env_check.sh` prints the configured download URL and can download, BGZF-compress, and index it automatically:

```bash
bash 0.env_check.sh --download-reference
```

The model path is configured in `default_config.json` under `paths.model`. Users can either edit this value to an absolute model path or create a symlink at `Data/OneGenomeRice_model`.

## 3. Workflow

The root directory contains three entry-point scripts.

| Step | Command | Output |
|:--|:--|:--|
| 0 | `bash 0.env_check.sh` | Checks input files, model files, Python modules, reference indexes, and CUDA visibility |
| 1 | `bash 1.calc_attention.sh` | Generates pseudo-sequences, computes bidirectional attention, builds matrices, and runs base-pair differential tests |
| 2 | `bash 2.plot_figures.sh` | Generates position-level signal figures and gene-level ranking figures |

The main output directories are:

| Directory | Content |
|:--|:--|
| `Results/attention/` | Intermediate pseudo-sequence, attention, matrix, and differential-test outputs |
| `Results/figures/` | Final display figures |
| `Results/tables/` | Gene-level metric and ranking tables |

## 4. Methods

The four 40 kb regions are split into 8 kb windows with a 4 kb stride. During plotting and gene-level summarization, only the effective non-overlapping middle interval of each overlapping block is used, avoiding duplicated visualization or scoring of overlapping coordinates.

For each base position, forward and reverse-complement attention matrices are compared between phenotype groups using a Mann-Whitney U test, followed by Benjamini-Hochberg correction. The signal figures report `-log10(adjusted P)` and `log2FC` tracks for both attention directions.

For gene-level prioritization, the workflow uses gene bodies from `chr06.gff3.gz` without upstream/downstream extension. Forward, reverse-complement, and summed-direction matrices are evaluated with the ATLAS-style summary statistics implemented in this repository. The final display focuses on summed-direction Peak Density and Shannon Entropy rankings.

## 5. Environment

Run the workflow inside an environment that can execute OneGenome-Rice inference and the downstream scientific Python stack. The entry-point scripts use the currently active Python environment by default. If preferred, users can set `CONDA_ENV` or fill `environment.conda_env` and `environment.conda_sh` in `default_config.json` to run through `conda run`.

Required Python modules:

```text
Bio
cyvcf2
matplotlib
numpy
pandas
pysam
scipy
seaborn
statsmodels
torch
tqdm
transformers
```

## 6. Usage

Run the workflow from the repository root:

```bash
bash 0.env_check.sh
bash 1.calc_attention.sh --gpus 0,1
bash 2.plot_figures.sh
```

If the reference FASTA is not present after cloning the package, download and index it automatically:

```bash
bash 0.env_check.sh --download-reference
```

If `osa1_r7.asm.fa.gz` already exists but `.fai` or `.gzi` is missing, rebuild the indexes with:

```bash
bash 0.env_check.sh --repair-reference-index
```

If a user manually downloads `osa1_r7.asm.fa.gz` with `curl` or `wget`, the environment check will verify whether it supports random access. If needed, the repair step converts a plain gzip FASTA to BGZF before indexing.

If only one GPU is available:

```bash
bash 1.calc_attention.sh --gpus 0 --workers 1
```

The primary display figures are:

| Figure | Description |
|:--|:--|
| `Results/figures/differential_attention_padj.png` | Region-level adjusted-P differential attention signal |
| `Results/figures/differential_attention_log2fc.png` | Region-level log2FC directionality signal |
| `Results/figures/gene_metric_sum_top8.png` | Top gene rankings by summed-direction Peak Density and Shannon Entropy |
