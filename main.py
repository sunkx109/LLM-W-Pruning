import argparse
import os
import torch
from safetensors.torch import safe_open, save_file
from typing import Dict, List


def load_safetensors(file_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not exits: {file_path}")
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            weights = {k: f.get_tensor(k) for k in f.keys()}
        return weights
    except Exception as e:
        raise RuntimeError(f"Load safetensors failed: {e}")



def filter_layer_weights(
    weights: Dict[str, torch.Tensor],
    layer_pattern: str,
    exclude_other: bool = True
) -> Dict[str, torch.Tensor]:
    layer_weights = {k: v for k, v in weights.items() if layer_pattern in k}
    
    if not exclude_other:
        # Retain the non-layer-related base weights (e.g., embedding, lm_head, etc.)
        non_layer_weights = {k: v for k, v in weights.items() if not any(
            # Match common layer naming patterns (compatible with different models)
            pat in k for pat in ["layers.", "h.", "transformer.h.", "encoder.layer."]
        )}
        layer_weights.update(non_layer_weights)
    
    if not layer_weights:
        raise ValueError(f"No weights found for pattern '{layer_pattern}', please check the layer name")
    
    return layer_weights


def save_layer_safetensors(
    weights: Dict[str, torch.Tensor],
    output_path: str,
    metadata: dict = None
) -> None:
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        save_file(weights, output_path, metadata=metadata or {})
        print(f"Saved {len(weights)} weights to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Save safetensors failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract specified layer weights from safetensors")
    parser.add_argument("--input", "-i", required=True, help="Input safetensors file path")
    parser.add_argument("--output", "-o", required=True, help="Output safetensors file path")
    parser.add_argument("--layer", "-l", required=True, help="Layer matching pattern (e.g., layers.0, h.3)")
    parser.add_argument("--keep-base", action="store_true", 
                        help="Whether to keep base weights (e.g., embedding, lm_head; default is to keep only specified layer)")
    args = parser.parse_args()
    
    try:
        print(f"Loading weights from: {args.input}")
        raw_weights = load_safetensors(args.input)
        print(f"Total weights: {len(raw_weights)}")

        print(f"Selecting weights for layer '{args.layer}'...")
        filtered_weights = filter_layer_weights(
            raw_weights, 
            layer_pattern=args.layer,
            exclude_other=not args.keep_base
        )
        print(f"Filtered weights count: {len(filtered_weights)}")

        save_layer_safetensors(filtered_weights, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()