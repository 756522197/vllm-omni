#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge Ascend msModelSlim MXFP8 HunyuanVideo-1.5 weights into Diffusers format.

The output directory is a normal HunyuanVideo-1.5 Diffusers checkpoint. Its
``transformer/config.json`` contains ``quantization_config`` so vllm-omni
selects the offline NPU MXFP8 path automatically.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Any

from safetensors.torch import load_file, save_file

MXFP8_QUANT_CONFIG: dict[str, Any] = {
    "quant_method": "mxfp8",
    "is_checkpoint_mxfp8_serialized": True,
}

RENAME_DICT: dict[str, str] = {}


def _load_safetensors(directory: pathlib.Path, glob_pattern: str = "*.safetensors") -> dict[str, Any]:
    files = sorted(directory.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No safetensors matching {glob_pattern!r} found in {directory}")

    tensors: dict[str, Any] = {}
    for file in files:
        tensors.update(load_file(str(file)))
    return tensors


def _load_quant_safetensors(directory: pathlib.Path) -> dict[str, Any]:
    try:
        return _load_safetensors(directory, "quant_model_weight*.safetensors")
    except FileNotFoundError:
        return _load_safetensors(directory)


def _rename_key(key: str) -> str:
    new_key = key
    for src, dst in RENAME_DICT.items():
        new_key = new_key.replace(src, dst)
    return new_key


def _inject_quant_config(config_path: pathlib.Path) -> None:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["quantization_config"] = MXFP8_QUANT_CONFIG
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def repack(
    original_model_path: pathlib.Path,
    quant_path: pathlib.Path,
    output_path: pathlib.Path,
    transformer_subfolder: str = "transformer",
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    shutil.copytree(str(original_model_path), str(output_path))

    original_transformer_dir = original_model_path / transformer_subfolder
    output_transformer_dir = output_path / transformer_subfolder
    if not original_transformer_dir.is_dir():
        raise FileNotFoundError(f"Missing original transformer dir: {original_transformer_dir}")
    if not quant_path.is_dir():
        raise FileNotFoundError(f"Missing quantized weights dir: {quant_path}")

    base_state = _load_safetensors(original_transformer_dir)
    quant_state_raw = _load_quant_safetensors(quant_path)
    quant_state = {_rename_key(key): tensor for key, tensor in quant_state_raw.items()}

    for stale in output_transformer_dir.glob("*.safetensors"):
        stale.unlink()
    for stale in output_transformer_dir.glob("*.index.json"):
        stale.unlink()
    for stale in output_transformer_dir.glob("*.bin"):
        stale.unlink()

    merged = {**base_state, **quant_state}
    save_file(merged, str(output_transformer_dir / "diffusion_pytorch_model.safetensors"))

    config_path = output_transformer_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing transformer config: {config_path}")
    _inject_quant_config(config_path)

    print(f"Merged {len(base_state)} BF16 tensors with {len(quant_state)} MXFP8 tensors.")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-model", required=True, help="Original HunyuanVideo-1.5 Diffusers directory")
    parser.add_argument("--quant-path", required=True, help="Directory containing msModelSlim MXFP8 safetensors")
    parser.add_argument("--output-path", required=True, help="New merged output directory")
    parser.add_argument("--transformer-subfolder", default="transformer")
    args = parser.parse_args()

    repack(
        original_model_path=pathlib.Path(args.original_model),
        quant_path=pathlib.Path(args.quant_path),
        output_path=pathlib.Path(args.output_path),
        transformer_subfolder=args.transformer_subfolder,
    )


if __name__ == "__main__":
    main()
