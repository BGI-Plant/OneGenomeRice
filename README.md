NOTE (Preparation Phase – to be removed before public release):
- All README files, code comments, inline explanations, and related documentation should be written in **English**. Chinese versions may be provided optionally.
- Before making the repository public, all commit history should be cleaned and organized to ensure clarity, consistency, and professionalism.
- Large model weight files and datasets are hosted on [Hugging Face](https://huggingface.co/ZhejiangLab/AgriGenome) and [ModelScope](https://modelscope.cn/models/zhejianglab/AgriGenome) due to size limitations of this repository. Links should be provided in the README to guide users for access and download.
- The research team is responsible for preparing all technical materials in the repository. The OSPO (Open Source Program Office) is responsible for conducting a compliance review and supplementing the repository with required governance documents, including CODE_OF_CONDUCT.md, CONTRIBUTING.md, LICENSE.md, and SECURITY.md.
- The README is recommended to follow the structure below. This serves as a guideline rather than a strict requirement, and may be adapted based on project-specific needs.


<img width="4144" height="4000" alt="OGR" src="https://github.com/user-attachments/assets/4daca7d1-4d38-41f2-8b95-00d1331f3db8" />

# OneGenome-Rice (OGR): A Genomic Foundation Model for Rice
## 1. Introduction
OGR is a foundational AI infrastructure for the next generation of AI-driven precision breeding and functional genomics in rice.
OGR is a generative genomic foundation model designed to process DNA sequences up to 1 million base pairs in length. The model features 1.25 billion total parameters, utilizing a Mixture of Experts (MoE) architecture that allows for high representational capacity while maintaining computational efficiency during inference. OGR was pre-trained on a curated corpus of 422 rice genomes, representing a diverse array of genotypes from the rice genome group, which includes both modern high-yielding varieties and wild ancestral populations. We detail the architectural innovations, dataset composition, and application-specific findings that define OGR.
## 2. Model Information
### Model Architecture
The architecture of AgriGenome consists of 12 MoE-Transformer layers. The model adopts a decoder-only configuration, utilizing a Next Token Prediction (NTP) objective which is naturally suited for modeling the sequential nature of DNA.
Mixture of Experts (MoE): Each layer contains 8 experts, with a router network that dynamically selects the top 2 experts for each nucleotide token. This allows the model to activate only a subset of its total parameters (0.33B out of 1.25B) for any given token, significantly reducing FLOPs per step.
Rotary Position Embedding (RoPE): Precision in positional awareness across long contexts is ensured via RoPE with a base frequency of 50,000,000. This mechanism allows the model to differentiate between motifs separated by hundreds of kilobases, which is essential for capturing distal enhancer-promoter interactions.
Grouped-Query Attention (GQA): AgriGenome implements GQA with 16 attention heads sharing 8 key-value groups. This design choice accelerates inference by reducing the memory bandwidth required for key-value caching, a critical factor for large-scale genomic screening.
SwiGLU Activation: Expert subnetworks utilize Swish-Gated Linear Units (SwiGLU), which have been shown to enhance expressive capacity and stabilize training in large-scale language models compared to standard ReLU or GELU activations.
### Training Data
### Training Process
## 3. Performance Evaluation
## 4. Quickstart
### Docker Deployment（if applicable）
### Model Download
### Script
## 5. Application Scenarios（if applicable）
## 6. License
## 7. Citation and Acknowledgements
## 8. Contact
