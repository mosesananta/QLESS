#!/bin/bash

nproc_per_node=$1
nnodes=$2
data_dir=$3
model_path=$4
percentage=$5
data_seed=$6
job_name=$7

# Generate a unique ID for rendezvous
ID=$RANDOM

header="torchrun \
  --nproc_per_node=$nproc_per_node \
  --nnodes=$nnodes \
  --rdzv-id=$ID \
  --rdzv_backend=c10d \
  -m less.train.train"

base_training_args="--do_train True \
--max_seq_length 2048 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--eval_strategy no \
--logging_steps 1 \
--num_train_epochs 1 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--report_to wandb \
--optim adamw_torch \
--seed 0 \
--percentage 1.0 \
--save_strategy no \
--lora True \
--lora_r 64 \
--lora_alpha 256 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32"


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
