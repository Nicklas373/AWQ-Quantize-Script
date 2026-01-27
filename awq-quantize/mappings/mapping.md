## Model tree mapping inspector

Models tree mapping may be differ on each models, to get it. It can be check with model_inspect.py

Run this script

```shell
   python model_inspect.py --model_id "HUGGINGFACE/HUGGINGFACE_MODEL" --inspect_type "vision" --trust_remote_code True
```

It'll print models tree mapping for selected "vision" or "llm" variant

# For LLM Only

```shell
mappings=[] # Can be map from model_inspect.py, may be differ for each models
```

## Default AWQ Modifier Mappings

# For LLM Only

```shell
mappings=[
    {
        "smooth_layer": "re:.*up_proj$",
        "balance_layers": ["re:.*down_proj$"],
    },
],
```
