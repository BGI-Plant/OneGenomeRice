# OneGenome-Rice (OGR): A Genomic Foundation Model for Rice

<div align="center">
    <img src="figure/main.png" width="99%" alt="Genos" />
</div>


## 1. Introduction
OGR is a foundational AI infrastructure for the next generation of AI-driven precision breeding and functional genomics in rice.
OGR is a generative genomic foundation model designed to process DNA sequences up to 1 million base pairs in length. The model features 1.25 billion total parameters, utilizing a Mixture of Experts (MoE) architecture that allows for high representational capacity while maintaining computational efficiency during inference. OGR was pre-trained on a curated corpus of 422 rice genomes, representing a diverse array of genotypes from the rice genome group, which includes both modern high-yielding varieties and wild ancestral populations. We detail the architectural innovations, dataset composition, and application-specific findings that define OGR.
## 2. Model Information
OGR is a decoder-only MoE Transformer for long genomic context. The subsections below summarize **training data**, **model architecture**, and **training process** (full detail in the **Technical Report**, URL to be added).

### Training Data
The training corpus is a **QC-filtered pangenome of 422 rice genomes** spanning cultivated and wild *Oryza* diversity. For preprocessing and sampling details, see the **Technical Report**(**文章URL**).

- **Provenance:** assemblies come from **open datasets** published in the literature (public archives and associated papers).
- **Encoding:** raw DNA with a nucleotide-level tokenizer (A/T/C/G/N and special tokens).
- **Pre-training stages:** **multi-scale curriculum** up to **1M** bp windows, followed by **continued pre-training (CPT)**; exact token counts and mixing ratios are given under **Training Process** below and in the Technical Report.

### Model Architecture
OGR follows a Transformer decoder with **Mixture-of-Experts (MoE)** layers. Main technical highlights:

- **Ultra-long context:** **RoPE** with base **50M** supports up to **1M** tokens; multi-stage training scales the effective context window.
- **Efficient attention:** **GQA** with **16** heads and **8** KV groups, paired with **Flash Attention** kernels.
- **MoE routing:** **8** experts, **top-2** per token, **SwiGLU** experts, **RMSNorm**; objective is **next-token prediction (NTP)**.

The following table summarizes key specifications.

<div align="center">

| **Model Specification** | **OneGenome-Rice (OGR)** |
|:---:|:---:|
| **Model Scale** | |
| Total Parameters | 1.25B |
| Activated Parameters | 0.33B |
| **Architecture** | |
| Architecture | MoE |
| Number of Experts | 8 |
| Selected Experts per Token | 2 |
| Number of Layers | 12 |
| Attention Hidden Dimension | 1024 |
| Number of Attention Heads | 16 (GQA, 8 KV groups) |
| MoE Hidden Dimension (per Expert) | 4096 |
| Vocabulary Size | 128 (padded) |
| Context Length | up to 1M |

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
  - **MoE balancing:** aux loss **1e-3** + Z-loss **1e-3**
  - **Communication:** grouped GEMM + AllToAll + overlapped gradient reduction
  - **Memory:** Flash Attention + sequence parallel + distributed optimizer
  - **I/O:** cyclic dataloader with multi-worker prefetch

- **Representative Fine-tuning**
  - **Task:** RNA-seq coverage prediction
  - **Data:** ENCODE + GTEx (667 normalized samples)
  - **Architecture:** Genos backbone + 3-layer CNN head
  - **Hardware:** 64×H100, bf16
  - **Training:** 1 epoch, LR **5e-5**, cosine schedule

## 3. Performance Evaluation  
- **Short-sequence tasks:** Competitive overall performance with strong results in chromatin accessibility, epigenetic marks, and small RNA prediction, but weaker in splice sites and variant detection.  
- **Long-sequence tasks:** Stable performance across diverse tasks, with advantages in variant detection at longer contexts but not consistently leading in all categories.  
- **Single-nucleotide tasks:** Noticeable performance gap in high-resolution predictions, indicating limited nucleotide-level modeling capacity.  
- **Sweep region identification:** Clear advantage in long-context settings (8k–100k), demonstrating superior ability to capture large-scale genomic signals.  
- **Varieties classification:** Consistently outperforms other models across increasing sequence lengths, highlighting strong capability in population structure and evolutionary pattern recognition.  
- **AgroNT benchmark tasks:** Strong performance in chromatin accessibility but limited in poly(A) site and gene expression prediction, reflecting weaknesses in fine-grained regulatory modeling.  
<div align="center">
    <img src="figure/Performance Evaluation.png" width="60%" alt="Performance Evaluation" />
</div>


## 4. Quickstart
### Docker Deployment
We strongly recommend deploying Genos using Docker. 

Pull the Docker Image
```
docker pull zjlabogr/onegenomerice:mega
```

Run the Container
```
docker run -it --gpus all --shm-size 32g zjlabogr/onegenomerice:mega /bin/bash
```

### Model Download
Genos models are available for download from [Hugging Face](https://huggingface.co/ZhejiangLab/AgriGenome) and [ModelScope](https://modelscope.cn/models/zhejianglab/AgriGenome). Each model employs a hybrid Mixture-of-Experts (MoE) architecture and supports analysis at single-nucleotide resolution.

<div align="center">

| **Model** | **Total Params** | **Hugging Face** | **ModelScope** |
|:---------:|:----------------:|:----------------:|:--------------:|
| OGR-1.25B | 1.25B | [🤗 Hugging Face](https://huggingface.co/ZhejiangLab/AgriGenome) |[🤖 ModelScope](https://modelscope.cn/models/zhejianglab/AgriGenome) |

</div>

### Usage Guide(TODO!)
Please refer to the tutorial notebooks for common usage scenarios:
TODO
- [Biological sequence embedding extraction](Notebooks/01.embedding_en.ipynb)
- [Variant pathogenicity prediction](Notebooks\02.ClinVar_variant_predict_en.ipynb)
- [RNA coverage track prediction](Notebooks\03.RNASeqConvTrack_en.ipynb)

## 5. Application Scenarios(TODO!)
To further illustrate the practical value, extensibility, and potential of Genos, we present two representative application cases.

- **Case 1: [*Indica*-*Japonica* Introgression Identification](applications/1.indica-japonica_introgression_analysis/README.md)**  
  This case aims to exploit the capacity of the OGR foundation model for fine-scale inference of subspecies origin across the rice genome, enabling the identification of introgression between *indica* (*Oryza sativa* subsp. *indica*) and *japonica* (*Oryza sativa* subsp. *japonica*). Unlike traditional approaches that rely on SNP-based statistics or local sequence alignment, this study starts directly from raw genomic sequences. High-dimensional embeddings are extracted using the OGR model, upon which downstream predictive models are built. This approach enables the capture of deep genetic structural differences at the sequence level, facilitating the identification of potential introgressed regions between subspecies.
  
- **Case 2: [Identification of Trait-Associated Loci](applications/2.Identification_of_Trait-Associated_Loci/Readme.md)**  
  This repository demonstrates a reproducible workflow for identifying rice candidate loci from bidirectional attention signals produced by OneGenomeRice. The workflow reconstructs sample-specific sequences from variants, extracts forward and reverse-complement attention, performs position-level group comparisons, and summarizes gene-level differential signals in selected candidate regions.

- **Case 3: [Gene Expression Prediction based on DNA](applications/3.gene_expression_modeling_on_DNA/README.md)**


- **Case 4: [Gene Expression Prediction based on DNA and Other Modality](applications/4.gene_expression_modeling_on_DNA_and_ATAC/senario.md)**  
  A central challenge in predictive genomics is linking static DNA sequence to dynamic, context-specific gene expression and traits.
This scenario targets a concrete prediction task: given a genomic DNA sequence window and its aligned chromatin accessibility signal (ATAC-seq), predict the corresponding strand-specific RNA-seq signal at single-nucleotide resolution. By modeling DNA–ATAC interactions explicitly, the system aims to separate sequence-encoded potential from context-dependent activation, enabling base-level expression prediction that can support downstream analyses such as comparing regulatory conditions or simulating the effects of perturbations in silico.
## 6. Model Inference Optimization and Adaptation(待确认)
### vLLM Inference
We conduct inference optimization experiments on large language models using the vLLM framework. This initiative significantly enhances throughput and reduces inference latency. By leveraging vLLM's innovative ```PagedAttention``` algorithm and efficient memory management mechanisms, we achieve a throughput improvement of over 7× compared with conventional inference approaches.

- Pull the Docker Image

```
docker pull zjlabogr/vllm
```

- Run the Container

```
docker run -it --entrypoint /bin/bash --gpus all --shm-size 32g zjlabogr/vllm
```

- For detailed test results and practical usage examples of vLLM, please refer to the [vllm example](https://github.com/zhejianglab/Genos/blob/main/Notebooks/04.vllm_example.ipynb).

### Other GPU adaptations
We also conduct compatibility tests on the following hardware accelerators. For detailed adaptation and deployment instructions, please refer to the [Adaptation](https://github.com/zhejianglab/Genos/tree/main/Adaptation) for more information.
- Huawei Ascend NPU
- MUXI GPU

## 7. License and Uses
**License**：The OGR collection of models are licensed under the  [Apache License 2.0](LICENSE).

**Primary intended use**：The primary use of OGR models is to support rice genomics research, providing researchers with advanced analytical capabilities and long-context modeling tools powered by large-scale foundation models trained on rice genomes.

**Out-of-scope use**：OGR models are not intended for use in any manner that violates applicable laws or regulations, nor for any activities prohibited by the license agreement.

**Ethical Considerations and Limitations**: Like other foundation models, OGR models may exhibit behaviors that carry potential risks. They may generate inaccurate outputs when interpreting rice genomic sequences or making inferences. Therefore, users should conduct rigorous validation and apply appropriate safeguards before using OGR in downstream research. Developers deploying applications based on OGR must carefully assess risks specific to their use cases.

## 8. Citation and Acknowledgements(TODO!)
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

## 9. Contact(TODO!)
If you have any questions, please raise an issue or contact us at [genos@zhejianglab.org](mailto:genos@zhejianglab.org).
