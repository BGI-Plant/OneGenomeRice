# OneGenomeRice (OGR): A Genomic Foundation Model for Rice

<div align="center">
    <img src="figure/main.png" width="99%" alt="Genos" />
</div>

## 1. Introduction

OGR is a foundational AI infrastructure for the next generation of AI-driven precision breeding and functional genomics in rice.
OGR is a generative genomic foundation model designed to process DNA sequences up to 1 million base pairs in length. The model features 1.25 billion total parameters, utilizing a Mixture of Experts (MoE) architecture that allows for high representational capacity while maintaining computational efficiency during inference. OGR was pre-trained on a curated corpus of 422 rice genomes, representing a diverse array of genotypes from the rice genome group, which includes both modern high-yielding varieties and wild ancestral populations. We detail the architectural innovations, dataset composition, and application-specific findings that define OGR.

## 2. Model Information

OGR is a decoder-only MoE Transformer for long genomic context. The subsections below summarize **training data**, **model architecture**, and **training process** (full detail in the **Technical Report**, URL to be added).

### Training Data

The training corpus is a **QC-filtered pangenome of 422 rice genomes** spanning cultivated and wild *Oryza* diversity. For preprocessing and sampling details, see the [table](figure/422%20Curated%20Assembled%20Genome%20Collection.tsv)

- **Provenance:** assemblies come from **open datasets** published in the literature (public archives and associated papers).
- **Encoding:** raw DNA with a nucleotide-level tokenizer (A/T/C/G/N and special tokens).

### Model Architecture

OGR follows a Transformer decoder with **Mixture-of-Experts (MoE)** layers. Main technical highlights:

- **Ultra-long context:** **RoPE** with base **50M** supports up to **1M** tokens; multi-stage training scales the effective context window.
- **Efficient attention:** **GQA** with **16** heads and **8** KV groups, paired with **Flash Attention** kernels.
- **MoE routing:** **8** experts, **top-2** per token, **SwiGLU** experts, **RMSNorm**; objective is **next-token prediction (NTP)**.

The following table summarizes key specifications.

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center"><strong>Model Specification</strong></th>
      <th align="center"><strong>OneGenomeRice (OGR)</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" colspan="2"><strong>Model Scale</strong></td>
    </tr>
    <tr>
      <td align="center">Total Parameters</td>
      <td align="center">1.25B</td>
    </tr>
    <tr>
      <td align="center">Activated Parameters</td>
      <td align="center">0.33B</td>
    </tr>
    <tr>
      <td align="center" colspan="2"><strong>Architecture</strong></td>
    </tr>
    <tr>
      <td align="center">Architecture</td>
      <td align="center">MoE</td>
    </tr>
    <tr>
      <td align="center">Number of Experts</td>
      <td align="center">8</td>
    </tr>
    <tr>
      <td align="center">Selected Experts per Token</td>
      <td align="center">2</td>
    </tr>
    <tr>
      <td align="center">Number of Layers</td>
      <td align="center">12</td>
    </tr>
    <tr>
      <td align="center">Attention Hidden Dimension</td>
      <td align="center">1024</td>
    </tr>
    <tr>
      <td align="center">Number of Attention Heads</td>
      <td align="center">16 (GQA, 8 KV groups)</td>
    </tr>
    <tr>
      <td align="center">MoE Hidden Dimension (per Expert)</td>
      <td align="center">4096</td>
    </tr>
    <tr>
      <td align="center">Vocabulary Size</td>
      <td align="center">128 (padded)</td>
    </tr>
    <tr>
      <td align="center">Context Length</td>
      <td align="center">up to 1Mb</td>
    </tr>
  </tbody>
</table>

</div>

### Training Process

Genos pre-training is built on **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** with **5D parallelism** (**TP, PP, CP, DP, EP**).

- **Key Features**

  - **MoE:** 8 experts, Top-2 routing, sparse FFN execution
  - **GQA:** grouped-query attention for lower KV memory
  - **RoPE:** base **50M**, supports ultra-long context
  - **Modern stack:** RMSNorm, SwiGLU, Flash Attention
- **Pre-training Strategy**

  - **Objective:** self-supervised Next Token Prediction (**NTP**)
  - **Length curriculum:** **8K → 32K → 128K → 1M** tokens
  - **Orientation:** reverse-complement used across scales
  - **Data:** chromosome-scale de novo assemblies from public resources
  - **Tokenizer:** one-hot DNA encoding (A, T, C, G, N)
- **Infrastructure**

  - **Framework:** Megatron-LM on large A100 clusters (up to **256** GPUs)
  - **Parallelism:** 5D strategy (TP, PP, CP, DP, EP)
  - **Batch:** Global **1024**, Micro **1**
  - **Optimizer:** AdamW (distributed sharded)
  - **Learning rate:** peak **1e-4**, cosine decay, warm-up in **5-10%** range
  - **Precision:** **BF16** compute, **FP32** for softmax/gradients/routing
- **Key Optimizations**

  - **MoE load balancing:** auxiliary loss **1×10⁻³** + router **Z-loss 1×10⁻³**
  - **Communication / compute:** grouped **GEMM**, **AllToAll** dispatch, overlapped parameter aggregation and gradient reduction
  - **I/O:** **cyclic** data loader with **8** worker processes
  - **Memory / attention:** Flash Attention; GQA for KV efficiency

## 3. Performance Evaluation

  Across the 26 benchmark categories, OGR ranks first or second in 16 tasks, demonstrating strong overall performance and robust generalization across diverse genomic prediction tasks. The model performs particularly well in key regulatory and functional prediction tasks, including chromatin accessibility, histone modification, small RNA prediction, enhancer strength prediction, sweep region identification, and varieties classification, indicating its effectiveness in capturing genomic regulatory signals and functional patterns across multiple biological scales.

- **Short-sequence tasks:** OGR exhibits competitive overall performance, with strong results in chromatin accessibility, epigenetic mark prediction, and small RNA prediction, but relatively weaker performance in splice site recognition and variant detection.
- **Long-sequence tasks:** The model maintains stable performance across diverse tasks, showing advantages in variant detection over longer contexts, though it does not consistently lead in all categories.
- **Single-nucleotide tasks:** OGR shows a noticeable performance gap in high-resolution predictions, reflecting limited capacity for nucleotide-level modeling.
- **Sweep region identification:** The model demonstrates clear advantages in long-context settings (8kb–100kb), highlighting its ability to capture large-scale genomic signals.
- **Varieties classification:** OGR consistently outperforms other models across increasing sequence lengths, emphasizing its capability in population structure and evolutionary pattern recognition.
- **AgroNT benchmark tasks:** The model achieves strong performance in chromatin accessibility prediction but shows limitations in poly(A) site and gene expression prediction, reflecting weaknesses in fine-grained regulatory modeling.

<div align="center">
    <img src="figure/Performance Evaluation.png" width="60%" alt="Performance Evaluation" />
</div>

## 4. Quickstart

### Docker Deployment

We strongly recommend deploying OGR using Docker.

Pull the Docker Image

```
docker pull zjlabogr/onegenomerice:mega
```

Run the Container

```
docker run -it --gpus all --shm-size 32g zjlabogr/onegenomerice:mega /bin/bash
```

### Model Download

OGR models are available for download from [Hugging Face](https://huggingface.co/ZhejiangLab/OneGenomeRice) and [ModelScope](https://modelscope.cn/models/zhejianglab/OneGenomeRice). Each model employs a hybrid Mixture-of-Experts (MoE) architecture and supports analysis at single-nucleotide resolution.

<div align="center">

| **Model** | **Total Params** |                    **Hugging Face**                    |                       **ModelScope**                       |
| :-------------: | :--------------------: | :-----------------------------------------------------------: | :---------------------------------------------------------------: |
|    OGR-1.25B    |         1.25B         | [🤗 Hugging Face](https://huggingface.co/ZhejiangLab/OneGenomeRice) | [🤖 ModelScope](https://modelscope.cn/models/zhejianglab/OneGenomeRice) |

</div>

## 5. Application Scenarios

To further illustrate the practical value, extensibility, and potential of Genos, we present two representative application cases.

- **Case 1: <a href="applications/1.indica-japonica_introgression_analysis/README.md">Identification of <em>indica-japonica</em> Introgression</a>**
  This case aims to exploit the capacity of the OGR foundation model for fine-scale inference of subspecies origin across the rice genome, enabling the identification of introgression between indica (Oryza sativa subsp. indica) and japonica (Oryza sativa subsp. japonica). Unlike traditional approaches that rely on SNP-based statistics or local sequence alignment, this study starts directly from raw genomic sequences. High-dimensional embeddings are extracted using the OGR model, upon which downstream predictive models are built. This approach enables the capture of deep genetic structural differences at the sequence level, facilitating the identification of potential introgressed regions between subspecies.
- **Case 2: [Trait-Associated Loci Finding](applications/2.Identification_of_Trait-Associated_Loci/Readme.md)**
  This repository demonstrates a reproducible workflow for identifying rice candidate loci from bidirectional attention signals produced by OneGenomeRice. The workflow reconstructs sample-specific sequences from variants, extracts forward and reverse-complement attention, performs position-level group comparisons, and summarizes gene-level differential signals in selected candidate regions.
- **Case 3: [Gene Expression Prediction of DNA Sequence](applications/3.gene_expression_modeling_on_DNA/README.md)**
  This repository trains and runs models that predict nucleotide-resolution multi-modal data for DNA sequences up to 32 kb in length.
- **Case 4: [Gene Expression Prediction Based on Multi-modal Data](applications/4.gene_expression_modeling_on_DNA_and_ATAC/senario.md)**
  This repository trains and runs models that predict strand-specific RNA-seq coverage from a DNA sequence window and matching ATAC-seq (chromatin accessibility) in the same window. It targets plant and other genomics setups where BigWig tracks and a reference FASTA are available.


## 6. License and Uses

**License**：The OGR collection of models are licensed under the  [Apache License 2.0](LICENSE).

**Primary intended use**：The primary use of OGR models is to support rice genomics research, providing researchers with advanced analytical capabilities and long-context modeling tools powered by large-scale foundation models trained on rice genomes.

**Out-of-scope use**：OGR models are not intended for use in any manner that violates applicable laws or regulations, nor for any activities prohibited by the license agreement.

**Ethical Considerations and Limitations**: Like other foundation models, OGR models may exhibit behaviors that carry potential risks. They may generate inaccurate outputs when interpreting rice genomic sequences or making inferences. Therefore, users should conduct rigorous validation and apply appropriate safeguards before using OGR in downstream research. Developers deploying applications based on OGR must carefully assess risks specific to their use cases.

## 7. Citation and Acknowledgements(TODO!)

We acknowledge the Human Pangenome Reference Consortium (HRPC; BioProject ID: PRJNA730823) and its funding agency, the National Human Genome Research Institute (NHGRI), for providing publicly available data. We also thank the BGI AI team for technical assistance.

If you use this work in your research, please cite the following paper:

```
@article{10.1093/gigascience/giaf132,
    author = {Genos Team, Hangzhou, China},
    title = {Genos: A Human-Centric Genomic Foundation Model},
    journal = {GigaScience},
    pages = {giaf132},
    year = {2025},
    month = {10},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaf132},
    url = {https://doi.org/10.1093/gigascience/giaf132},
    eprint = {https://academic.oup.com/gigascience/advance-article-pdf/doi/10.1093/gigascience/giaf132/64848789/giaf132.pdf},
}
```

## 8. Contact(TODO!)

If you have any questions, please raise an issue or contact us at [OneGenomeRice@zhejianglab.org](mailto:OneGenomeRice@zhejianglab.org), [bgi-plant@genomics.cn](mailto:bgi-plant@genomics.cn).
