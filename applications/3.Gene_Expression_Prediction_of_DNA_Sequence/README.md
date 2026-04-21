# finetuned model for RNA prediction

This repository trains and runs models that predict nucleotide-resolution multi-modal data for DNA sequences up to 32 kb in length. The training data should be in BigWig format, and a reference genome must be included

## Technology stack
- **PyTorch** with **PyTorch Distributed (DDP)** and **NCCL** on CUDA for multi-GPU.
- **Hugging Face Transformers** (`Trainer`) for the training loop and integration with a frozen/unfrozen DNA LM.
- **Genomics I/O:** `pyBigWig`, `pyfaidx`; configs via **PyYAML**.

## Data prepartion
- The BigWig file needs to be renamed as tissue_species_1.bw.
- Run data_prepare.sh first to generate training and validation datasets.

## Training and prediction
- Run run_train.sh to train your fine-tuned model. Three days are needed for data from three rice species using 8 A100 GPUs.
- Run run_prediction.sh to perform batch prediction.
