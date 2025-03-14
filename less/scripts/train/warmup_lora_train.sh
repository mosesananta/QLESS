#!/bin/bash

# Usage:
#   ./warmup_lora_train.sh <nproc_per_node> <nnodes> <data_dir> <model_path> <percentage> <data_seed> <job_name>

# 1) nproc_per_node
# 2) nnodes
source less/scripts/train/base_training_args.sh "$1" "$2"

# 3) data_dir
data_dir=$3
# 4) model_path
model_path=$4
# 5) percentage
percentage=$5
# 6) data_seed
data_seed=$6
# 7) job_name
job_name=$7

# Create the output directory
output_dir=./out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p "$output_dir"
fi

# Define train files
train_files=(
  "$data_dir/train/processed/flan_v2/flan_v2_data.jsonl"
  "$data_dir/train/processed/cot/cot_data.jsonl"
  "$data_dir/train/processed/dolly/dolly_data.jsonl"
  "$data_dir/train/processed/oasst1/oasst1_data.jsonl"
)

# Use FSDP for large models
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

# Assemble training arguments
training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

# Print the final command (optional for debugging)
echo "$header $training_args"

# Execute the training run
eval "$header" "$training_args"
