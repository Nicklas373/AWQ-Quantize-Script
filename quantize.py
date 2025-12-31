# AWQ Quantize Script using LLM-Compressor
# Reference: https://github.com/xhedit/quantkit/

import argparse
import torch

from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from pathlib import Path
from transformers import AutoModelForCausalLM

# --------------------------------------------------
# 1. Download & Prepare Model Directory
# --------------------------------------------------
def get_model_path(
        branch: str,
        force_download: bool,
        hf_cache: bool,
        model_id: str,
):
    path = Path(model_id)

    # If it's already a local directory with a config, return it
    if path.is_dir() and (path / "config.json").is_file():
        return path

    # Otherwise, download from Hugging Face
    # Standard repo_id format is "owner/repo"
    folder_name = model_id.split("/")[-1]
    local_path = Path(folder_name)

    print(f"Downloading {model_id} to {local_path}...")
    snapshot_download(
        repo_id=model_id,
        revision=branch,
        local_dir=local_path,
        local_dir_use_symlinks=hf_cache,
        force_download=force_download
    )
    return local_path

# --------------------------------------------------
# 2. Main Quantization Logic
# --------------------------------------------------
def run_awq_quantization(
        branch: str,
        dataset_id: str,
        dataset_split: str,
        hf_cache: bool,
        max_seq_length: int,
        num_samples: int,
        model_id: str,
        text_column: str,
):
    # Step 1: Download/Verify local model
    model_path = get_model_path(branch, False, hf_cache, model_id)

    # Step 2: Manually load the model with the trust flag
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="auto", # Use the model's native precision (usually BF16/FP16)
        device_map="auto"   # Efficiently balance across GPUs if available
    )

    # Disable KV cache (saves VRAM during calibration)
    model.config.use_cache = False

    # Step 3: Prepare Custom Dataset
    # Step 2: Optional dataset
    calibration_dataset = None

    if dataset_id is not None:
        if text_column is None:
            raise ValueError("--text_column is required when --dataset_id is provided")

        print(f"Loading dataset: {dataset_id}")
        ds = load_dataset(dataset_id, split=dataset_split)
        ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

        processed = []
        for row in ds:
            content = row[text_column]
            if isinstance(content, list):
                text = " ".join(
                    m["content"] for m in content if isinstance(m, dict) and "content" in m
                )
            else:
                text = str(content)

            processed.append({"text": text})

        calibration_dataset = Dataset.from_list(processed)
    else:
        print("Using default AWQ calibration dataset")

    # Step 4: Define Recipe (Standard 4-bit AWQ)
    # W4A16_ASYM is the standard format for vLLM performance
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": False,
                        "strategy": "group",
                        "group_size": 128,
                    },
                }
            }
        )
    ]

    # Step 5: Run Oneshot Quantization
    # llm-compressor handles loading the model from the local_path
    output_dir = f"{model_path.name}-AWQ"

    print(f"Starting AWQ quantization. Output: {output_dir}")
    oneshot(
        model=model, # Pass the local directory string
        dataset=calibration_dataset,
        recipe=recipe,
        num_calibration_samples=(len(calibration_dataset) if calibration_dataset else None),
        max_seq_length=max_seq_length,
        output_dir=output_dir,
    )

    print(f"Success! Quantized model saved to {output_dir}")

# --------------------------------------------------
# 3. Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AWQ 4‑bit quantization by using LLM-Compressor"
    )
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to download and quantize.")
    parser.add_argument("--dataset_id", type=str, default=None, help="The Dataset ID to use as dataset on calibration.")
    parser.add_argument("--dataset_split", type=str, default="train", help=("Which split of the dataset to use for AWQ calibration (e.g. 'train', 'validation'). Ignored if --dataset_id is not provided."))
    parser.add_argument("--text_column", type=str, default=None, help=("Name of the column containing text data in the dataset. Required when --dataset_id is provided. If the column contains a list of messages, their 'content' fields ""will be concatenated"))
    parser.add_argument("--num_samples", type=int, default=256, help=("Number of samples used for AWQ calibration. More samples can improve accuracy but require more VRAM and time. Typical values: 128 to 512."))
    parser.add_argument("--max_seq_length", type=int, default=1024, help=("Maximum sequence length (in tokens) used during calibration. Longer sequences improve weight calibration for long-context models but increase VRAM usage."))
    parser.add_argument("--hf_cache", type=bool, default=False, help=( "Whether to use Hugging Face cache symlinks when downloading the model. Enable if you want to reuse cached files; disable for fully local copies."))
    parser.add_argument("--branch", type=str, default="main", help=( "Model repository branch or revision to download from Hugging Face (e.g. 'main', 'fp16', 'bf16')."))
    args = parser.parse_args()

    run_awq_quantization(
        branch=args.branch,
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        hf_cache=args.hf_cache,
        max_seq_length=args.max_seq_length,
        model_id=args.model_id,
        num_samples=args.num_samples,
        text_column=args.text_column,
     )

# Example Usage
if __name__ == "__main__":
        main()