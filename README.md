# LLM-W-Pruning
A LLM Weight Pruning Tool For Profiling
## Introduction
This script can create a lightweight version of a language model by extracting only the weights of the specified N transformer layers (e.g., 4 out of 28). It copies the necessary components—including embeddings, selected hidden layers, and the output head—and updates the configuration accordingly. The resulting sub-model retains compatibility with the original model and can be used for inference or fine-tuning while significantly reducing memory footprint and computational cost, making it suitable for rapid prototyping, debugging, or resource-constrained environments.

## Usage
```
python3 main.py \
  --input-dir ./original_model/ \
  --output ./current_model/ \
  --layer 0 \
  --keep-base
```
This will generate:
- A new folder named `current_model` in the current directory, which contains the modified model files.
- The output model folder, with modifications only made to config.json and the weight files (.safetensors), while all other files are copied as-is.
## Arguments
- `--input-dir`: The input HuggingFace model directory."
- `--output`: The output HuggingFace model directory.
- `--layer`: The layer indices to extract, e.g. '0-3' or '0,1,2,3'.
- `--keep-base`: Keep base weights (embeddings, lm_head, etc.). Default: True.

