# LLM Compressor - AWQ Quantization (AWQModifier)

Quantize any LLM Models into AWQ 4 Bit format by using LLM Compressor package from [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/main).

For now only for LLM based, Vision Language (VL) will following later.

## Current Recipe

```shell
SmoothQuantModifier(smoothing_strength=0.8),
AWQModifier(
    ignore=[
        "model.embed_tokens",
        "model.norm",
        "lm_head",
    ],
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 64,
                "observer": "minmax",
            },
        }
    }
)
```

## Alternate Nemotron Recipe

```shell
recipe = [
    # SmoothQuant (critical for Nemotron)
    SmoothQuantModifier(
        smoothing_strength=0.8,
        mappings=[
            [
                "re:model\\.backbone\\.layers\\.\\d+\\.norm$",
                [
                    "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.q_proj$",
                    "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.k_proj$",
                    "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.v_proj$",
                ],
            ],
            [
                "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.v_proj$",
                [
                    "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.o_proj$",
                ],
            ],
            [
                "re:model\\.backbone\\.layers\\.\\d+\\.norm$",
                [
                    "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.up_proj$",
                ],
            ],
            [
                "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.up_proj$",
                [
                    "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.down_proj$",
                ],
            ],
        ],
    ),
    AWQModifier(
        ignore=[
            "model.embed_tokens",
            "model.norm",
            "lm_head",
        ],
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 64,
                    "observer": "minmax",
                },
            }
        },
    ),
]
```

## Recommended Settings

- **--num_samples** 512
- **--max_seq_length** 1024 or 2048 (More length, will took longer time)

## How to use

- Deploy this Docker then Access SSH or using VSCode on Runpod with port 8080
- Exec **quantize.py** with this parameters (Examples)

```shell
python3 quantize.py --model_id HUGGINGFACE/HUGGINGFACE_MODEL --dataset_id DATASET/YOUR_DATASET --dataset_split train --text_column messages --num_samples 512 --max_seq_length 1024 --hf_cache False --branch main --trust_remote_code False --trust_remote_code_model False
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
python3 quantize.py --model_id HUGGINGFACE/HUGGINGFACE_MODEL --dataset_id DATASET/YOUR_DATASET --dataset_split train --text_column messages --num_samples 512 --max_seq_length 1024 --hf_cache False --branch main --trust_remote_code False --trust_remote_code_model False
```

- After quantization complete, then run **upload.py** to upload to HF as repo model

```shell
python3 upload.py --hf_token XXXX --repo_id YOUR_REPO_NAME --local_dir YOUR_REPO_LOCAL_DIR --repo_type YOUR_REPO_TYPE
```

## Access

- 8080: VS Code Server

## Directory Structure

- /workspace/quantize.py: Python based quantization script
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
- **dataset_id**: HuggingFace Dataset ID (Required)
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
