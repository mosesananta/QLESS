#!/bin/bash

# Usage:
#    ./base_training_args.sh <nproc_per_node> <nnodes>
#
# Example:
#    ./base_training_args.sh 8 2
#    (This means 8 processes per node, 2 nodes total.)

# Input Argument 1: Number of processes per node
nproc_per_node=$1
# Input Argument 2: Number of nodes
nnodes=$2

# Assert that both arguments are provided
if [[ -z $nproc_per_node || -z $nnodes ]]; then
  echo "Usage: $0 <nproc_per_node> <nnodes>"
  echo "Example: $0 8 2  (8 processes/node, 2 nodes total)"
  exit 1
fi

# Calculate gradient accumulation steps to maintain total batch size = 128
gradient_accumulation_steps=$(( 128 / (nproc_per_node * nnodes) ))

# Assertion: Ensure the product matches 128
if (( gradient_accumulation_steps * nproc_per_node * nnodes != 128 )); then
  echo "Error: (gradient_accumulation_steps * nproc_per_node * nnodes) must be 128."
  echo "  Currently: $gradient_accumulation_steps * $nproc_per_node * $nnodes != 128"
  exit 1
fi

# Generate a unique ID for rendezvous
ID=$RANDOM

# Export the torchrun command header
# (If you need to set node_rank, you typically pass --node_rank=$SLURM_NODEID or similar.)
export header="torchrun \
  --nproc_per_node=$nproc_per_node \
  --nnodes=$nnodes \
  --rdzv-id=$ID \
  --rdzv_backend=c10d \
  -m less.train.train"

# Base training arguments
# Here we're assuming per_device_train_batch_size=1 so the total batch size
# across all processes is (nnodes * nproc_per_node * per_device_train_batch_size),
# times gradient_accumulation_steps = 128.
export base_training_args="--do_train True \
--max_seq_length 2048 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--eval_strategy no \
--logging_steps 1 \
--num_train_epochs 4 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--report_to wandb \
--optim adamw_torch \
--seed 0 \
--percentage 1.0 \
--save_strategy epoch \
--lora True \
--lora_r 64 \
--lora_alpha 256 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps $gradient_accumulation_steps"

# Output confirmation
echo "Successfully configured base_training_args.sh"
echo "  nproc_per_node         = $nproc_per_node"
echo "  nnodes                 = $nnodes"
echo "  gradient_accumulation_steps = $gradient_accumulation_steps"
echo
echo "header command:"
echo "  $header"
echo
echo "base_training_args:"
echo "  $base_training_args"
