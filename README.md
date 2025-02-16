# QLESS: A Quantized Approach for Data Valuation and Selection in Large Language Model Fine-Tuning


QLESS integrates gradient quantization with the **LESS framework** to enable memory-efficient data valuation and selection for fine-tuning large language models (LLMs). By combining **LoRA-based random projection** with **absmax quantization**, QLESS reduces gradient storage by up to **16x** while maintaining competitive performance on benchmarks like MMLU, BBH, and TyDiQA. This repository provides code for reproducing experiments and applying QLESS to custom datasets.


## Setup

### Docker Environment

We ran all experiments using the `nvcr.io/nvidia/pytorch:23.12-py3` Docker image. To get started, ensure you have Docker installed and pull the image.

### Install Dependencies
After configuring your Docker environment (or your native environment), install the required dependencies:

```bash
pip install -r requirements.txt
pip install evaluate
pip install traker[fast]
pip install hqq
pip install wandb
pip install -e .
```

## Download Data
The necessary datasets are available at https://huggingface.co/datasets/mosesananta/qless_data. Download and unzip the data into the root directory of this repository.

## Run Full Experiment
Execute the following script to run the complete experimental pipeline:
```bash
./full_run.sh <model_path> <output_model_name> <seed>
```
For instance, to run the experiment using the `meta-llama/Llama-3.2-3B` model:

```bash
./full_run.sh "meta-llama/Llama-3.2-3B" "meta-llama-3.2-3b" 3
```

This will create 3 folders 
* `out`: Contains all the models and evaluation results
* `grads_16bit`: Contains all the gradient data
* `selected_datas`: Contains the data selected through influence-based selection.







