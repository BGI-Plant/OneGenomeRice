<div align="center">
    <img src="figure/main.png" width="99%" alt="main" />
</div>

# OneGenomeRice (OGR): A Genomic Foundation Model for Rice
## 1. Introduction
OGR is a foundational AI infrastructure for the next generation of AI-driven precision breeding and functional genomics in rice.
OGR is a generative genomic foundation model designed to process DNA sequences up to 1 million base pairs in length. The model features 1.25 billion total parameters, utilizing a Mixture of Experts (MoE) architecture that allows for high representational capacity while maintaining computational efficiency during inference. OGR was pre-trained on a curated corpus of 422 rice genomes, representing a diverse array of genotypes from the rice genome group, which includes both modern high-yielding varieties and wild ancestral populations. We detail the architectural innovations, dataset composition, and application-specific findings that define OGR.
## 2. Model Information
The following figure illustrates the overall workflow of the model, including training data processing, model architecture, training process and downstream model inference and applications.

<div align="center">
  <img src="figure\model.png" width="90%" title="Architecture">
</div>

The subsections below summarize **training data**, **model architecture**, and **training process**

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
Pre-training is implemented with **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** on **128 NVIDIA A100-80GB GPUs** (**16 nodes × 8 GPUs**) on the National Artificial Intelligence Public Computing Power Open Innovation Platform (**Nanhu** scheduler; resource pool **>1,000** A100-80GB GPUs). End-to-end runtime is **~8 days**. Parallelism is **five-dimensional**: tensor, pipeline, context, data, and expert (**TP, PP, CP, DP, EP**).

- **Key Features**
  - **MoE:** 8 experts, Top-2 routing, load-balancing + Z-loss (see **Key Optimizations**)
  - **GQA:** 16 heads, 8 KV groups
  - **RoPE:** base **50M** for ultra-long context (up to **1M** tokens)
  - **Modern stack:** RMSNorm, SwiGLU, Flash Attention

- **Pre-training Strategy**
  - **Objective:** Next Token Prediction (**NTP**) under self-supervision
  - **Progressive context scaling:** multistage training with longer sequence windows, **learning-rate annealing** (mitigate catastrophic forgetting), and **RoPE-based** context window scaling up to **1M** tokens
  - **Sequence curriculum (initial phase):** **~490B** tokens; window sizes **8,192 / 32,768 / 131,072 / 1,024,000** bp at mix **5 : 2 : 2 : 1**; **8k** and **32k** combine **gene-dense** regions (gene bodies + **3 kb** flanking from **190** annotated genomes) with randomized whole-genome tiling; **128k** and **1M** use whole-genome tiling only; **reverse-complement** at all scales
  - **Continued pre-training (CPT):** **~104B** additional tokens (reshuffled lengths/orientations); **no** position-specific / intergenic-distance filtering
  - **Tokenizer / data:** one-hot–style nucleotide encoding (A, T, C, G, N); assemblies as in **Training Data**

- **Infrastructure**
  - **Framework:** Megatron-LM on **128** GPUs (**16×8** A100-80GB)
  - **Parallelism:** 5D strategy (TP, PP, CP, DP, EP)
  - **Batch:** Global **1024**, Micro **1** (gradient accumulation)
  - **Optimizer:** AdamW (distributed sharded)
  - **Learning Rate:** peak **1×10⁻⁴**, cosine decay, **5%** linear warm-up; **gradient clipping 1.0**; **weight decay 0.1**
  - **Precision:** **BF16** for most compute; **FP32** for softmax (attention), gradient accumulation / **All-Reduce**, and MoE routing; BF16 matmul disabled on those sensitive paths

- **Key Optimizations**
  - **MoE load balancing:** auxiliary loss **1×10⁻³** + router **Z-loss 1×10⁻³**
  - **Communication / compute:** grouped **GEMM**, **AllToAll** dispatch, overlapped parameter aggregation and gradient reduction
  - **I/O:** **cyclic** data loader with **8** worker processes
  - **Memory / attention:** Flash Attention; GQA for KV efficiency

## 3. Performance Evaluation(TODO!)


## 4. Quickstart
### Docker Deployment
We strongly recommend deploying Genos using Docker. 

Pull the Docker Image
```
docker pull zjlabgenos/mega:v1
```

Run the Container
```
docker run -it --gpus all --shm-size 32g zjlabgenos/mega:v1 /bin/bash
```

### Model Download
Genos models are available for download from [Hugging Face](https://huggingface.co/ZhejiangLab/OneGenomeRice) and [ModelScope](https://modelscope.cn/models/zhejianglab/OneGenomeRice). Each model employs a hybrid Mixture-of-Experts (MoE) architecture and supports analysis at single-nucleotide resolution.

<div align="center">

| **Model** | **Total Params** | **Hugging Face** | **ModelScope** |
|:---------:|:----------------:|:----------------:|:--------------:|
| OGR-1.25B | 1.25B | [🤗 Hugging Face](https://huggingface.co/ZhejiangLab/OneGenomeRice) |[🤖 ModelScope](https://modelscope.cn/models/zhejianglab/OneGenomeRice) |

</div>

### Usage Guide(TODO!)
Please refer to the tutorial notebooks for common usage scenarios:
TODO
- [Biological sequence embedding extraction](Notebooks/01.embedding_en.ipynb)
- [Variant pathogenicity prediction](Notebooks\02.ClinVar_variant_predict_en.ipynb)
- [RNA coverage track prediction](Notebooks\03.RNASeqConvTrack_en.ipynb)

## 5. Application Scenarios(TODO!)
To further illustrate the practical value, extensibility, and potential of Genos, we present two representative application cases.

- **Case 1: [*Indica*-*Japonica* Introgression Identification](indica-japonica_introgression_analysis/README.md)**  
  This case aims to leverage the representation learning capability of the OneGenome-Rice foundation model to perform fine-scale inference of subspecies origin across the rice genome, enabling the identification of introgression between indica (Oryza Sativa subsp. Indica) and japonica (Oryza Sativa subsp. Japonica). Unlike traditional approaches that rely on SNP-based statistics or local sequence alignment, this study starts directly from raw genomic sequences. High-dimensional embeddings are extracted using the OneGenome-Rice model, upon which downstream predictive models are built. This approach enables the capture of deep genetic structural differences at the sequence level, facilitating the identification of potential introgressed regions between subspecies.

- **Case 2: [Text-Genome Model Fusion](Text-genome_model_fusion/Case_2_Text_Genome_Model_Fusion.md)**  
  This case explores a multimodal framework that integrates genome-scale sequence encoders with large language models. It emphasizes the ability to jointly leverage biological prior knowledge, literature-based reasoning, and sequence-level representations, paving the way for more intelligent, interpretable, and knowledge-grounded bio-AI systems.

## 6. Model Inference Optimization and Adaptation(待确认)
### vLLM Inference
We conduct inference optimization experiments on large language models using the vLLM framework. This initiative significantly enhances throughput and reduces inference latency. By leveraging vLLM’s innovative ```PagedAttention``` algorithm and efficient memory management mechanisms, we achieve a throughput improvement of over 7× compared with conventional inference approaches.

- Pull the Docker Image

```
docker pull zjlabgenos/vllm:v1
```

- Run the Container

```
docker run -it --entrypoint /bin/bash --gpus all --shm-size 32g zjlabgenos/vllm:v1
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
