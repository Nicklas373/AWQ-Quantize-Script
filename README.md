# LLM Quantization (AWQ) with LLM Compressor

## How to use

- Deploy this Docker then Access SSH
- Exec **quantize.py** with this parameters (Examples)

```shell
python3 quantize.py --model_id HUGGINGFACE/HUGGINGFACE_MODEL --dataset_id DATASET/YOUR_DATASET --dataset_split train --text_column messages --num_samples 512 --max_seq_length 2048 --hf_cache False --branch main
```

- After quantization complete, then run **upload.py** to upload to HF as repo model

```shell
python3 upload.py --hf_token XXXX --repo_id YOUR_REPO_NAME --local_dir YOUR_REPO_LOCAL_DIR --repo_type YOUR_REPO_TYPE
```

## Explanation for **quantize.py**

- **model_id**: HuggingFace Model Name (Required)
- **dataset_id**: HuggingFace Dataset ID (Required)
- **dataset_split**: Split of the dataset to use for calibration (e.g. 'train', 'validation'). Ignored if --dataset_id is not provided
- **text_column**: Name of the column containing text data in the dataset. Required when --dataset_id is provided. If the column contains a list of messages, their 'content' fields ""will be concatenated"
- **num_samples**: Number of samples used for AWQ calibration. More samples can improve accuracy but require more VRAM and time. Typical values: 128 to 512
- **max_seq_length**: Maximum sequence length (in tokens) used during calibration. Longer sequences improve weight calibration for long-context models but increase VRAM usage

## Explanation for **upload.py**

- **hf_token**: Hugging Face user token
- **repo_id**: Hugging Face repository target (eg: hello:my_llm)
- **local_dir**: Hugging Face local folder directory
- **repo_type**: Repository type (default: model)
