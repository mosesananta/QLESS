#!/bin/bash

# Usage:
#   ./lora_train.sh <nproc_per_node> <nnodes> <train_files> <model_path> <job_name>
# Example:
#   ./lora_train.sh 8 2 /path/to/train.json meta-llama/Llama-2-7b-hf run-llama2-7b

# 1) nproc_per_node
# 2) nnodes
source less/scripts/train/base_training_args.sh "$1" "$2"

# 3) The training data file(s)
train_files=$3
# 4) The model path
model_path=$4
# 5) The name of the job
job_name=$5

# Create an output directory for logs/checkpoints
output_dir=./out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p "$output_dir"
fi

# For large models, enable FSDP (Fully Sharded Data Parallel)
if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
    
    elif [[ $model_path == "meta-llama/Llama-2-7b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_7b_finetune"
    
    elif [[ $model_path == "meta-llama/Llama-3.1-8B" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama3_1_8b_finetune"

    elif [[ $model_path == "meta-llama/Llama-3.2-3B" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama_finetune"
    
    elif [[ $model_path == "Qwen/Qwen2.5-7B" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config qwen2_finetune"
   
    elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune --tokenizer_name mistral_tokenizer" 
fi

# Construct the final training arguments
training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

# Print the combined command for clarity
echo "$header $training_args"

# Execute the training via torchrun
eval "$header" "$training_args"
