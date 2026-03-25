#!/usr/bin/env python3
"""
inspect_peft_checkpoint.py — Show all weights stored in a PEFT checkpoint directory.

Usage:
  python inspect_peft_checkpoint.py <checkpoint_dir> [--shape] [--filter PATTERN]

Examples:
  python inspect_peft_checkpoint.py output/medical-embedding-custom/20250325_120000/epoch_1
  python inspect_peft_checkpoint.py output/.../ --shape
  python inspect_peft_checkpoint.py output/.../ --filter norm
  python inspect_peft_checkpoint.py output/.../ --filter lora_A --shape
"""

import argparse
import json
from pathlib import Path


def load_keys_safetensors(path: Path):
    """Return {key: shape} from a safetensors file without loading tensors into memory."""
    import struct
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return {
        k: v["shape"]
        for k, v in header.items()
        if k != "__metadata__"
    }


def load_keys_bin(path: Path):
    """Return {key: shape} from a pytorch .bin file."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: list(v.shape) for k, v in state.items()}


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint_dir", help="Path to PEFT checkpoint directory")
    p.add_argument("--shape",  action="store_true", help="Show tensor shapes")
    p.add_argument("--filter", default=None, help="Only show keys containing this substring")
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"ERROR: {ckpt_dir} does not exist")
        return

    # Find weight files
    weight_files = sorted(ckpt_dir.glob("*.safetensors")) + sorted(ckpt_dir.glob("*.bin"))
    if not weight_files:
        print(f"No .safetensors or .bin files found in {ckpt_dir}")
        return

    # Show adapter_config.json summary if present
    config_path = ckpt_dir / "adapter_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        print("=== adapter_config.json ===")
        for k in ("peft_type", "r", "lora_alpha", "target_modules", "modules_to_save"):
            if k in config:
                print(f"  {k}: {config[k]}")
        print()

    for wf in weight_files:
        print(f"=== {wf.name} ===")
        if wf.suffix == ".safetensors":
            keys = load_keys_safetensors(wf)
        else:
            keys = load_keys_bin(wf)

        filtered = {k: v for k, v in keys.items()
                    if args.filter is None or args.filter in k}

        if not filtered:
            print(f"  (no keys match filter '{args.filter}')")
        else:
            for k, shape in sorted(filtered.items()):
                if args.shape:
                    print(f"  {k}  {shape}")
                else:
                    print(f"  {k}")

        print(f"  ({len(filtered)} / {len(keys)} keys shown)\n")


if __name__ == "__main__":
    main()
