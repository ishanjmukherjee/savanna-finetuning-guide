#!/usr/bin/env python3
"""
hotfix_norm_scale.py  —  Rename `blocks.<LAST>.norm.scale` → `norm.scale`
Usage:
    python hotfix_norm_scale.py  \
        --ckpt  original_checkpoint.pt      \
        --last-block  32         \
        --out   fixed_checkpoint.pt
"""

import argparse, torch, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to bad checkpoint")
    parser.add_argument("--last-block", type=int, required=True,
                        help="index of the *final* block (e.g. 32)")
    parser.add_argument("--out", default=None, help="Path to save fixed ckpt")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    dst_path  = args.out or ckpt_path.replace(".pt", "_fixed.pt")
    last_idx  = args.last_block

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    src_key = f"blocks.{last_idx}.norm.scale"
    if src_key not in sd:
        raise KeyError(f"Couldn’t find {src_key} in {ckpt_path}")

    print(f"Moving  {src_key}  →  norm.scale")
    sd["norm.scale"] = sd.pop(src_key)

    # (optional) drop the super-fluous final_linear weight inside the block
    proj_key = f"blocks.{last_idx}.final_linear.weight"
    if proj_key in sd:
        print(f"Deleting orphan tensor {proj_key}")
        sd.pop(proj_key)

    torch.save(sd, dst_path)
    print(f"✅  Saved fixed checkpoint to {dst_path}")

if __name__ == "__main__":
    main()
