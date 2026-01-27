# LLM Compressor - AWQ Quantization (AWQModifier)

Quantize any LLM Models into AWQ 4 Bit format by using LLM Compressor package from [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/main).

## Current Recipe for LLM

```shell
SmoothQuantModifier(
    smoothing_strength=0.8
),
AWQModifier(
    ignore=[
        "re:.*embed_tokens",
        "re:.*model.norm",
        "re:.*input_layernorm$",
        "re:.*post_attention_layernorm$",
        "re:.*lm_head",
    ],
    targets=["Linear"],
    config_groups={
        "channel_sensitive": {
            "targets": [
                're:.*q_proj$',
                're:.*k_proj$',
                're:.*v_proj$',
                're:.*gate_proj$',
                're:.*up_proj$'
            ],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "channel",
                "observer": "minmax",
                "dynamic": False,
            },
        },
        "group_0": {
            "targets": ["re:.*down_proj$"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "observer": "mse",
                "dynamic": False,
            },
        }
    },
    mappings=[
        {
            "smooth_layer": r"re:.*up_proj$",
            "balance_layers": [r"re:.*down_proj$"],
        },
    ],
)
```

## Current Recipe for VLM

```shell
AWQModifier(
    ignore=[
        "re:.*embed_tokens",
        "re:.*model.norm",
        "re:.*input_layernorm$",
        "re:.*post_attention_layernorm$",
        "re:.*lm_head",
        "re:.*vision_tower.*",
        "re:.*vision_encoder.*",
        "re:.*multi_modal_projector.*",
        "re:model[.]visual.*",
    ],
    targets=["Linear"],
    config_groups={
        "channel_sensitive": {
            "targets": [
                're:.*q_proj$',
                're:.*k_proj$',
                're:.*v_proj$',
                're:.*gate_proj$',
                're:.*up_proj$'
            ],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "channel",
                "observer": "minmax",
                "dynamic": False,
            },
        },
        "group_0": {
            "targets": ["re:.*down_proj$"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "observer": "mse",
                "dynamic": False,
            },
        }
    },
    mappings=[
        {
            "smooth_layer": r"re:.*up_proj$",
            "balance_layers": [r"re:.*down_proj$"],
        },
    ],
)
```

## Recommended Settings

- **--num_samples** 256 or 512
- **--max_seq_length** 1024 or 2048 (More length, will took longer time)

## How to use

- Deploy this Docker then Access SSH or using VSCode on Runpod with port 8080
- Exec **quantize.py** with this parameters (Examples)

```shell
python model_quantize.py --model_id "HUGGINGFACE/HUGGINGFACE_MODEL" --dataset_id DATASET1/YOUR_DATASET_1,DATASET2/YOUR_DATASET_2 --dataset_mix 0.5,0.5 --dataset_split train --text_column messages --num_samples 256 --max_seq_length 1024 --hf_cache False --branch main --trust_remote_code False --trust_remote_code_model False
```

- After quantization complete, then run **upload.py** to upload to HF as repo model

```shell
python3 upload.py --hf_token XXXX --repo_id YOUR_REPO_NAME --local_dir YOUR_REPO_LOCAL_DIR --repo_type YOUR_REPO_TYPE
```

## How to use on runpod

- Go to this URL template [llm-quantize-awq](https://console.runpod.io/deploy?template=5ik1p956nd&ref=xv2vjyqp)
- Deploy this template into runpod then Access SSH or using VSCode with port 8080
- Exec **quantize.py** with this parameters (Examples)

```shell
python model_quantize.py --model_id "HUGGINGFACE/HUGGINGFACE_MODEL" --dataset_id DATASET1/YOUR_DATASET_1,DATASET2/YOUR_DATASET_2 --dataset_mix 0.5,0.5 --dataset_split train --text_column messages --num_samples 256 --max_seq_length 1024 --hf_cache False --branch main --trust_remote_code False --trust_remote_code_model False
```

- After quantization complete, then run **upload.py** to upload to HF as repo model

```shell
python3 upload.py --hf_token XXXX --repo_id YOUR_REPO_NAME --local_dir YOUR_REPO_LOCAL_DIR --repo_type YOUR_REPO_TYPE
```

## Access

- 8080: VS Code Server

## Directory Structure

- /workspace/model_inspect.py: Python based model tree inspect script
- /workspace/model_quantize.py: Python based quantization script
- /workspace/upload.py: Python based upload to HF script

## Python package requirements

- accelerate
- causal-conv1d
- datasets
- huggingface-hub
- hf-transfer
- llmcompressor
- mamba-ssm
- transformers

## Explanation for **quantize.py**

- **model_id**: HuggingFace Model Name (Required)
- **dataset_id**: HuggingFace Dataset ID (Required), can have multiple datasets with comma-separated ratios, e.g. xyz/xx1,xyz/xx2
- **dataset_config**: Dataset configuration name (if applicable).
- **dataset_mix**: Combine multiple datasets with comma-separated ratios, e.g. 0.7,0.3
- **dataset_split**: Split of the dataset to use for calibration (e.g. 'train', 'validation'). Ignored if --dataset_id is not provided
- **text_column**: Name of the column containing text data in the dataset. Required when --dataset_id is provided. If the column contains a list of messages, their 'content' fields ""will be concatenated"
- **num_samples**: Number of samples used for AWQ calibration. More samples can improve accuracy but require more VRAM and time. Typical values: 128 to 512
- **max_seq_length**: Maximum sequence length (in tokens) used during calibration. Longer sequences improve weight calibration for long-context models but increase VRAM usage
- **trust_remote_code**: Whether to trust and execute custom model code from the Hugging Face repository. Required for many community models.
- **trust_remote_code_model**: Whether to trust and execute custom model code when loading the model. Required for many community models.

## Explanation for **upload.py**

- **hf_token**: Hugging Face user token
- **repo_id**: Hugging Face repository target (eg: hello:my_llm)
- **local_dir**: Hugging Face local folder directory
- **repo_type**: Repository type (default: model)
