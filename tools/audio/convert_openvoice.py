#!/usr/bin/env python3
# Copyright (c) 2026 Ahmed Seada
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""Converts OpenVoice's tone colour converter (checkpoint.pth and config.json, as OpenVoice ships them) into a folder
Idrak's OpenVoice plug-in reads: config.json as it is and model.safetensors (the state dict; weight norm's weight_g and
weight_v kept as stored, the plug-in folds them). Idrak reads no PyTorch pickles, so this runs once, in Python, on a
checkpoint you trust (torch.load runs the pickle's code; weights_only=True is used to limit it).

    python tools/audio/convert_openvoice.py checkpoints_v2/converter -o openvoice-v2

Needs torch and safetensors (pip install torch safetensors)."""

import argparse
import json
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="OpenVoice's converter folder (checkpoint.pth and config.json)")
    parser.add_argument("-o", "--out", required=True, help="the folder to write (config.json and model.safetensors)")
    parser.add_argument("--checkpoint", default="checkpoint.pth", help="the checkpoint's file name in the folder")
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file

    config_path = os.path.join(args.folder, "config.json")
    checkpoint_path = os.path.join(args.folder, args.checkpoint)
    for path in (config_path, checkpoint_path):
        if not os.path.isfile(path):
            sys.exit(f"{path} is not there")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if "data" not in config or "model" not in config:
        sys.exit(f"{config_path} is not an OpenVoice config.json (no \"data\" and \"model\" sections)")

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = loaded.get("model", loaded) if isinstance(loaded, dict) else None
    if not isinstance(state, dict) or not state:
        sys.exit(f"{checkpoint_path} holds no state dict (OpenVoice's has one under \"model\")")

    tensors = {k: v.detach().to(torch.float32).contiguous() for k, v in state.items() if torch.is_tensor(v)}
    missing = [p for p in ("ref_enc.gru.weight_ih_l0", "enc_q.pre.weight", "dec.conv_pre.weight") if p not in tensors]
    if missing:
        sys.exit(f"{checkpoint_path} lacks {', '.join(missing)}: not an OpenVoice converter")

    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "model.safetensors"), metadata={"source": os.path.basename(checkpoint_path)})
    shutil.copyfile(config_path, os.path.join(args.out, "config.json"))
    size = sum(t.numel() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors ({size:,} values) to {os.path.join(args.out, 'model.safetensors')}, and config.json "
          f"(version {config.get('_version_', 'v1')}, {config['data'].get('sampling_rate')} Hz, embedding {config['model'].get('gin_channels')} wide)")


if __name__ == "__main__":
    main()
