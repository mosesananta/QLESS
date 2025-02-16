import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import bitsandbytes as bnb
from hqq.core.bitpack import BitPack

def quantize(tensor, nbits=1, absmeasure="max", override_scale_factor=None):
    """
    override_scale_factor : if provided, we skip computing the scale factor from the tensor
                            and use override_scale_factor instead.
    """
    if override_scale_factor is not None:
        absmeasure_val = override_scale_factor
    else:
        if absmeasure == "mean":
            absmeasure_val = torch.mean(tensor.abs())
        elif absmeasure == "max":
            absmeasure_val = torch.max(tensor.abs())
        else:
            raise ValueError(f"Unknown absmeasure: {absmeasure}")

    if nbits == 1:
        scale_factor = absmeasure_val
        quantized_tensor = (tensor > 0).to(torch.uint8)

    else:
        min_val = -(2 ** (nbits - 1) - 1)
        max_val = (2 ** (nbits - 1) - 1)
        scale_factor = max_val / absmeasure_val if absmeasure_val > 1e-9 else 1.0
        scaled_tensor = tensor * scale_factor
        quantized_tensor = torch.clamp(scaled_tensor.round(), min_val, max_val)

    return quantized_tensor, scale_factor


def merge_and_quantize(
    output_dir: str, 
    nbits=1, 
    absmeasure="max", 
    override_scale_factor=None
):
    bit_to_packing = {
        8: "8bit_u8",
        4: "4bit_u8",
        2: "2bit_u8",
        1: "1bit_u8",
    }

    pack = {
        "8bit_u8": BitPack.pack_8bit_u8,
        "4bit_u8": BitPack.pack_4bit_u8,
        "2bit_u8": BitPack.pack_2bit_u8,
        "1bit_u8": BitPack.pack_1bit_u8,
    }

    merged_data = torch.load(os.path.join(output_dir, "all_orig.pt"))
    
    quantized_data, scale_factor = quantize(
        merged_data, 
        nbits=nbits, 
        absmeasure=absmeasure, 
        override_scale_factor=override_scale_factor
    )

    meta = {
        "nbits": nbits,
        "shape": merged_data.shape,
        "scale_factor": scale_factor,
        "offset": torch.max(quantized_data.abs()) if nbits > 1 else 0,
        "packing": bit_to_packing[nbits],
        "absmeasure": absmeasure,
    }

    flat_tensor = quantized_data.flatten()
    offset_tensor = flat_tensor + meta["offset"]
    bitpacked_tensor = pack[meta['packing']](offset_tensor)

    output = (bitpacked_tensor, meta)

    if absmeasure == "max":
        output_file = os.path.join(output_dir, f"all_quantized_{nbits}bits_absmax.pt")
    else:
        output_file = os.path.join(output_dir, f"all_quantized_{nbits}bits_absmean.pt")

    torch.save(output, output_file)
    print(f"Saving the quantized grads (Shape: {merged_data.shape}) to {output_file}.")

def calculate_influence_score(
    training_info: torch.Tensor,
    validation_info: torch.Tensor,
    sim_type: str = "cosine"
):
    if sim_type == "cosine":
        eps = 1e-9
        train_norm = training_info / (training_info.norm(dim=1, keepdim=True) + eps)
        valid_norm = validation_info / (validation_info.norm(dim=1, keepdim=True) + eps)
        influence_scores = train_norm @ valid_norm.T
    else:
        influence_scores = training_info @ validation_info.T

    return influence_scores

def compute_scale_factor(data_paths, absmeasure):
    """Compute scale factor from multiple data files"""
    total_abs = 0.0
    total_elements = 0
    max_abs = 0.0
    
    for path in data_paths:
        data = torch.load(os.path.join(path, "all_orig.pt"))
        if absmeasure == "max":
            current_max = data.abs().max().item()
            if current_max > max_abs:
                max_abs = current_max
        else:  # mean
            total_abs += data.abs().sum().item()
            total_elements += data.numel()
    
    if absmeasure == "max":
        return max_abs
    return total_abs / total_elements if total_elements > 0 else 1e-9


def main(args):
    TARGET_TASK_NAMES = args.target_tasks
    TRAIN_FILE_NAMES = args.train_files
    CKPTS = args.ckpts
    CHECKPOINT_WEIGHTS = args.ckpt_weights

    BASE_GRAD_PATH = args.base_grad_path
    GRADIENT_PATH = os.path.join(BASE_GRAD_PATH, "{}-ckpt{}-adam/dim8192")
    VALIDATION_GRADIENT_PATH = os.path.join(BASE_GRAD_PATH, "{}-ckpt{}-sgd/dim8192")
    SELECTED_DATA_OUTPUT_PATH = args.output_path

    unpack = {
        "8bit_u8": BitPack.unpack_8bit_u8,
        "4bit_u8": BitPack.unpack_4bit_u8,
        "2bit_u8": BitPack.unpack_2bit_u8,
        "1bit_u8": BitPack.unpack_1bit_u8,
    }

    # Normalize checkpoint weights if they don't sum to 1
    if sum(CHECKPOINT_WEIGHTS) != 1:
        s = sum(CHECKPOINT_WEIGHTS)
        CHECKPOINT_WEIGHTS = [i / s for i in CHECKPOINT_WEIGHTS]


    # Compute global training scale
    training_paths = []
    for train_file_name in TRAIN_FILE_NAMES:
        for ckpt in CKPTS:
            grad_path = GRADIENT_PATH.format(train_file_name, ckpt)  
            training_paths.append(grad_path)
    training_paths = list(set(training_paths))  

    global_scale_factor_training = compute_scale_factor(training_paths, args.absmeasure)
    print(f"Global training scale ({args.absmeasure}): {global_scale_factor_training:.6f}")
    
    # Compute validation scales per task
    task_validation_scales = {}
    for target_task in TARGET_TASK_NAMES:
        val_paths = [
            VALIDATION_GRADIENT_PATH.format(target_task, ckpt)
            for ckpt in CKPTS
        ]
        task_validation_scales[target_task] = compute_scale_factor(val_paths, args.absmeasure)
        print(f"{target_task} validation scale ({args.absmeasure}): {task_validation_scales[target_task]:.6f}")

    # Quantize all gradients
    all_grad_paths = []
    # Quantize training gradients with global training scale
    for train_file_name in TRAIN_FILE_NAMES:
        for ckpt in CKPTS:
            grad_path = GRADIENT_PATH.format(train_file_name, ckpt)
            if grad_path not in all_grad_paths:
                merge_and_quantize(
                    grad_path,
                    nbits=args.nbits,
                    absmeasure=args.absmeasure,
                    override_scale_factor=global_scale_factor_training
                )
                all_grad_paths.append(grad_path)
    
    # Quantize validation gradients with per-task scales
    for target_task in TARGET_TASK_NAMES:
        for ckpt in CKPTS:
            val_path = VALIDATION_GRADIENT_PATH.format(target_task, ckpt)
            if val_path not in all_grad_paths:
                merge_and_quantize(
                    val_path,
                    nbits=args.nbits,
                    absmeasure=args.absmeasure,
                    override_scale_factor=task_validation_scales[target_task]
                )
                all_grad_paths.append(val_path)
    
    # Influence Calculation
    N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for target_task_name in tqdm(TARGET_TASK_NAMES):
        for train_file_name in TRAIN_FILE_NAMES:
            influence_score = 0
            for i, ckpt in enumerate(CKPTS):
                # Load validation
                validation_path = VALIDATION_GRADIENT_PATH.format(target_task_name, ckpt)
                if args.absmeasure == "max":
                    val_quant_file = "all_quantized_{}bits_absmax.pt".format(args.nbits)
                else:
                    val_quant_file = "all_quantized_{}bits_absmean.pt".format(args.nbits)

                validation_path = os.path.join(validation_path, val_quant_file)
                validation_tensor, validation_meta = torch.load(validation_path)

                validation_info = unpack[validation_meta['packing']](
                    validation_tensor.unsqueeze(1)
                ).reshape(validation_meta['shape']).float()

                # Adjust sign if 1-bit
                if args.nbits == 1:
                    validation_info = torch.where(
                        validation_info == 0, 
                        -torch.ones_like(validation_info), 
                        validation_info
                    )
                else:
                    validation_info = validation_info - validation_meta["offset"]
                validation_info = validation_info.to(device)

                # Load training
                gradient_path = GRADIENT_PATH.format(train_file_name, ckpt)
                gradient_path = os.path.join(gradient_path, val_quant_file)
                training_tensor, training_meta = torch.load(gradient_path)

                training_info = unpack[training_meta['packing']](
                    training_tensor.unsqueeze(1)
                ).reshape(training_meta['shape']).float()

                if args.nbits == 1:
                    training_info = torch.where(
                        training_info == 0, 
                        -torch.ones_like(training_info), 
                        training_info
                    )
                else:
                    training_info = training_info - training_meta["offset"]
                training_info = training_info.to(device).float()

                # Calculate influence
                influence_score += CHECKPOINT_WEIGHTS[i] * calculate_influence_score(
                    training_info=training_info, 
                    validation_info=validation_info,
                    sim_type=args.sim_type
                )

            influence_score = influence_score.reshape(
                influence_score.shape[0],
                N_SUBTASKS[target_task_name],
                -1
            ).mean(-1).mean(-1)

            output_dir = os.path.join(SELECTED_DATA_OUTPUT_PATH, target_task_name)
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"{train_file_name}_influence_score.pt")
            torch.save(influence_score, output_file)
            print(f"Saved influence score to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize gradients and calculate influence scores.")
    parser.add_argument("--nbits", type=int, default=2,
                        help="Number of bits for quantization. E.g. 1, 2, 4, 8")
    parser.add_argument("--absmeasure", type=str, default="max",
                        choices=["max", "mean"], 
                        help="Measure for absolute value scaling (max or mean).")
    parser.add_argument("--base_grad_path", type=str, required=True, 
                        help="Base path for gradients.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save output.")
    parser.add_argument("--target_tasks", nargs='+', 
                        default=["bbh", "mmlu", "tydiqa"], 
                        help="Target tasks.")
    parser.add_argument("--train_files", nargs='+', 
                        default=["flan_v2", "cot", "dolly", "oasst1"], 
                        help="Training file names.")
    parser.add_argument("--ckpts", nargs='+', type=int, 
                        default=[105, 211, 317, 420], 
                        help="Checkpoint numbers.")
    parser.add_argument("--ckpt_weights", nargs='+', type=float, 
                        default=[1.5626e-05, 1.0417e-05, 5.2088e-06, 1.4742e-07],
                        help="Checkpoint weights.")
    parser.add_argument("--sim_type", type=str, default="cosine",
                        choices=["dot", "cosine"],
                        help="Similarity measure type: 'dot' or 'cosine'. Default='cosine'.")
    args = parser.parse_args()
    main(args)
