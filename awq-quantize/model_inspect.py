# LLM Model Inspection Script

import argparse
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM

def get_model_path(
        branch: str,
        force_download: bool,
        hf_cache: bool,
        model_id: str,
):
    path = Path(model_id)

    if path.is_dir() and (path / "config.json").is_file():
        return path

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

def get_vision_model_mapping(
    model_id: str,
    trust_remote_code: bool,
):
    model_path = get_model_path("main", False, False, model_id)

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    for name, _ in model.named_modules():
        # Print only Linear layers we might smooth
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]):
            print(name)

        # Norms (for ignore_modules)
        if "norm" in name.lower() or "input_layernorm" in name or "post_attention_layernorm" in name.lower():
            print("NORM:", name)

def get_llm_target_mapping(
    model_id: str,
    trust_remote_code: bool,
):
   model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
   
   for name, module in model.named_modules():
        # Norms (for ignore_modules)
        if "norm" in name.lower() or "input_layernorm" in name or "post_attention_layernorm" in name.lower():
            print("NORM:", name)

        # Attention projections
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            print("ATTN:", name, type(module))

        # MLP/FFN projections
        if any(x in name for x in ["up_proj", "down_proj", "gate_proj"]):
            print("MLP:", name, type(module))

def main():
    parser = argparse.ArgumentParser(
        description="LLM Model Inspection Script"
    )
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to download and quantize.")
    parser.add_argument("--inspect_type", type=str, required=True, choices=["vision", "llm"], help="Type of model to inspect: 'vision' or 'llm'.")
    parser.add_argument("--trust_remote_code", type=bool, default=False, help=("Whether to trust and execute custom model code from the Hugging Face repository. Required for many community models."))
    args = parser.parse_args()

    if args.inspect_type == "vision":
        get_vision_model_mapping(
            model_id=args.model_id,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.inspect_type == "llm":
        get_llm_target_mapping(
            model_id=args.model_id,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError("Invalid inspect-type. Choose 'vision' or 'llm'.")

# Example Usage
if __name__ == "__main__":
        main()