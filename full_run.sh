#!/bin/bash

########################################
# Configuration Variables
########################################
NUM_GPUS=4
NNODES=1
DATA_DIR=./data
PERCENTAGE=0.05

MODEL_PATH_BASE=$1
PROJECT_NAME=$2
DATA_SEED=$3

# This JOB_NAME will be referenced throughout the script

JOB_NAME="${PROJECT_NAME}-p${PERCENTAGE}-lora-seed${DATA_SEED}"

########################################
# Derived Paths 
########################################
MODEL_OUTPUT_DIR="./out/${JOB_NAME}"
GRADS_OUTPUT_DIR="./grads_16bit/${JOB_NAME}"
SELECTED_DATA_BASE_PATH="./selected_datas/${JOB_NAME}"

########################################
# Initial Warmup Training
########################################
./less/scripts/train/warmup_lora_train.sh \
  "$NUM_GPUS" \
  "$NNODES" \
  "$DATA_DIR" \
  "$MODEL_PATH_BASE" \
  "$PERCENTAGE" \
  "$DATA_SEED" \
  "$JOB_NAME"

########################################
# Common Settings
########################################
GRADIENT_TYPE="adam"
DIMS="8192"

########################################
# Function: Dynamically retrieve checkpoint numbers
########################################
get_checkpoints() {
    local model_output_dir=$1
    find "$model_output_dir" -type d -name 'checkpoint-*' \
      | grep -oP 'checkpoint-\K\d+' \
      | sort -n
}

########################################
# Dynamically populate the CHECKPOINTS array
########################################
CHECKPOINTS=($(get_checkpoints "$MODEL_OUTPUT_DIR"))
echo "Discovered checkpoints: ${CHECKPOINTS[@]}"

########################################
# Data & Eval Task Names
########################################
TRAINING_DATA_NAMES=("flan_v2" "cot" "dolly" "oasst1")
EVAL_TASKS=("bbh" "mmlu" "tydiqa")

########################################
# Functions to Calculate Gradients
########################################
distributed_train_gradient_calculation() {
  local checkpoint=$1
  local data_name=$2
  local gpu=$3

  local training_data_file="$DATA_DIR/train/processed/${data_name}/${data_name}_data.jsonl"
  local model_path="${MODEL_OUTPUT_DIR}/checkpoint-${checkpoint}"
  local output_path="${GRADS_OUTPUT_DIR}/${data_name}-ckpt${checkpoint}-${GRADIENT_TYPE}"

  mkdir -p "$output_path"
  CUDA_VISIBLE_DEVICES=$gpu \
    ./less/scripts/get_info/grad/get_train_lora_grads.sh \
      "$training_data_file" \
      "$model_path" \
      "$output_path" \
      "$DIMS" \
      "$GRADIENT_TYPE" 2>&1 | tee "$output_path/run.txt"
}

distributed_eval_gradient_calculation() {
  local checkpoint=$1
  local task=$2
  local gpu=$3

  local model_path="${MODEL_OUTPUT_DIR}/checkpoint-${checkpoint}"
  local output_path="${GRADS_OUTPUT_DIR}/${task}-ckpt${checkpoint}-sgd"

  mkdir -p "$output_path"
  CUDA_VISIBLE_DEVICES=$gpu \
    ./less/scripts/get_info/grad/get_eval_lora_grads.sh \
      "$task" \
      "$DATA_DIR" \
      "$model_path" \
      "$output_path" \
      "$DIMS" 2>&1 | tee "$output_path/run.txt"
}

########################################
# Helper: Run Commands on GPUs in Parallel by Dataset
########################################
run_on_gpus_by_dataset() {
  local dataset=$1
  local num_gpus=$2
  local gpu=0

  for checkpoint in "${CHECKPOINTS[@]}"; do
    distributed_train_gradient_calculation "$checkpoint" "$dataset" "$gpu" &
    gpu=$(( (gpu + 1) % num_gpus ))
    if (( gpu == 0 )); then
      wait
    fi
  done
  wait
}

# Process Each Dataset (Training) Sequentially Across GPUs
for data_name in "${TRAINING_DATA_NAMES[@]}"; do
  run_on_gpus_by_dataset "$data_name" "$NUM_GPUS"
done

########################################
# Helper: Run Commands on GPUs in Parallel by Task
########################################
run_eval_on_gpus_by_task() {
  local task=$1
  local num_gpus=$2
  local gpu=0

  for checkpoint in "${CHECKPOINTS[@]}"; do
    distributed_eval_gradient_calculation "$checkpoint" "$task" "$gpu" &
    gpu=$(( (gpu + 1) % num_gpus ))
    if (( gpu == 0 )); then
      wait
    fi
  done
  wait
}

# Process Each Task (Evaluation) Sequentially Across GPUs
for task in "${EVAL_TASKS[@]}"; do
  run_eval_on_gpus_by_task "$task" "$NUM_GPUS"
done

########################################
# Quantize Gradients
########################################
quantize_gradients() {
  local nbits=$1
  local output_dir="${SELECTED_DATA_BASE_PATH}/${nbits}bit/"

  python3 quantize_gradients.py \
    --nbits "$nbits" \
    --absmeasure max \
    --base_grad_path "$GRADS_OUTPUT_DIR" \
    --output_path "$output_dir" \
    --target_tasks bbh mmlu tydiqa \
    --train_files flan_v2 cot dolly oasst1 \
    --ckpts ${CHECKPOINTS[@]} \
    --ckpt_weights 1.5626e-05 1.0417e-05 5.2088e-06 1.4742e-07 \
    --sim_type "cosine"
}

for nbits in 8 4 2 1; do
  quantize_gradients "$nbits" 2>&1 | tee "${SELECTED_DATA_BASE_PATH}/${nbits}bit/run.txt"
done

########################################
# Influential Data Selection
########################################
GRADIENT_PATH="${GRADS_OUTPUT_DIR}/{}-ckpt{}-adam/dim${DIMS}"
VALIDATION_GRADIENT_PATH="${GRADS_OUTPUT_DIR}/{}-ckpt{}-sgd/dim${DIMS}"
TRAIN_FILE_NAMES="${TRAINING_DATA_NAMES[*]}"
TARGET_TASK_NAMES="${EVAL_TASKS[*]}"
CHECKPOINT_WEIGHTS="1.5626e-05 1.0417e-05 5.2088e-06 1.4742e-07"
BITS=(16 8 4 2 1)

# Perform Matching
./less/scripts/data_selection/matching.sh \
  "$GRADIENT_PATH" \
  "$TRAIN_FILE_NAMES" \
  "$CHECKPOINTS" \
  "$CHECKPOINT_WEIGHTS" \
  "$VALIDATION_GRADIENT_PATH" \
  "$TARGET_TASK_NAMES" \
  "${SELECTED_DATA_BASE_PATH}/16bit/"

########################################
# Write Selected Data for All Bit Depths
########################################
write_selected_data() {
  local bit=$1
  local output_path="${SELECTED_DATA_BASE_PATH}/${bit}bit/"

  python3 -m less.data_selection.write_selected_data \
    --target_task_names $TARGET_TASK_NAMES \
    --train_file_names $TRAIN_FILE_NAMES \
    --train_files \
       ./data/train/processed/flan_v2/flan_v2_data.jsonl \
       ./data/train/processed/cot/cot_data.jsonl \
       ./data/train/processed/dolly/dolly_data.jsonl \
       ./data/train/processed/oasst1/oasst1_data.jsonl \
    --output_path "$output_path" \
    --percentage "$PERCENTAGE"
}

for bit in "${BITS[@]}"; do
  write_selected_data "$bit" 2>&1 | tee "${SELECTED_DATA_BASE_PATH}/${bit}bit/write.txt"
done

########################################
# Final Fine-Tune with Selected Data
########################################
fine_tune_model() {
  local target_task_name=$1
  local bitstore=$2

  # Build file path from bits
  local train_files="${SELECTED_DATA_BASE_PATH}/${bitstore}bit/${target_task_name}/top_p${PERCENTAGE}.jsonl"
  local job_name="${PROJECT_NAME}-less-${bitstore}bit-p${PERCENTAGE}-${target_task_name}-seed${DATA_SEED}"

  ./less/scripts/train/lora_train.sh \
    "$NUM_GPUS" \
    "$NNODES" \
    "$train_files" \
    "$MODEL_PATH_BASE" \
    "$job_name"
}

for target_task_name in "tydiqa" "mmlu" "bbh"; do
  for bitstore in 16 8 4 2 1; do
    fine_tune_model "$target_task_name" "$bitstore"
  done
done

########################################
# Evaluation (Newly Fine-Tuned Models)
########################################
cd evaluation

evaluate_model() {
  local target_task_name=$1
  local bitstore=$2
  local gpu=$3

  export CUDA_VISIBLE_DEVICES=$gpu
  local eval_model_path="../out/${PROJECT_NAME}-less-${bitstore}bit-p${PERCENTAGE}-${target_task_name}-seed${DATA_SEED}"

  case $target_task_name in
    tydiqa)
      source eval_tydiqa.sh && eval_tydiqa "$eval_model_path" &
      ;;
    mmlu)
      source eval_mmlu.sh && eval_mmlu "$eval_model_path" &
      ;;
    bbh)
      source eval_bbh.sh && eval_bbh "$eval_model_path" &
      ;;
    *)
      echo "Unknown task: $target_task_name"
      ;;
  esac
}

gpu=0
for target_task_name in "tydiqa" "mmlu" "bbh"; do
  for bitstore in 16 8 4 2 1; do
    evaluate_model "$target_task_name" "$bitstore" "$gpu"
    gpu=$(( (gpu + 1) % NUM_GPUS ))
    if (( gpu == 0 )); then
      wait
    fi
  done
done
wait

########################################
# Evaluation on Original Fine-Tuned Checkpoint
########################################
evaluate_model_pretrained() {
  local target_task_name=$1
  local job_name=$2
  local gpu=$3

  export CUDA_VISIBLE_DEVICES=$gpu
  case $target_task_name in
    tydiqa)
      source eval_tydiqa.sh && eval_tydiqa "../out/$job_name" &
      ;;
    mmlu)
      source eval_mmlu.sh && eval_mmlu "../out/$job_name" &
      ;;
    bbh)
      source eval_bbh.sh && eval_bbh "../out/$job_name" &
      ;;
    *)
      echo "Unknown task: $target_task_name"
      ;;
  esac
}

gpu=0
for target_task_name in "tydiqa" "mmlu" "bbh"; do
  evaluate_model_pretrained "$target_task_name" "$JOB_NAME" "$gpu"
  gpu=$(( (gpu + 1) % NUM_GPUS ))
  if (( gpu == 0 )); then
    wait
  fi
done
wait
