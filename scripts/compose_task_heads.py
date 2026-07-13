#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch


def load_payload(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("meta", {}), checkpoint
    return checkpoint, {}, {"state_dict": checkpoint, "meta": {}}


def parse_source_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Invalid --source value '{value}'. Expected TASK_ID=/path/to/checkpoint.pth")
    task_id, checkpoint_path = value.split("=", 1)
    task_id = task_id.strip()
    checkpoint_path = checkpoint_path.strip()
    if not task_id or not checkpoint_path:
        raise ValueError(f"Invalid --source value '{value}'. Expected TASK_ID=/path/to/checkpoint.pth")
    return task_id, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compose a checkpoint by keeping the base model and replacing one or more "
            "task-specific head blocks from source checkpoints."
        )
    )
    parser.add_argument("--base-checkpoint", required=True, help="Base checkpoint to start from.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Task head source in the form TASK_ID=/path/to/checkpoint.pth. Can be repeated.",
    )
    parser.add_argument("--output-path", required=True, help="Path to write the composed checkpoint.")
    args = parser.parse_args()

    base_state, base_meta, base_payload = load_payload(args.base_checkpoint)
    output_state = copy.deepcopy(base_state)
    composed_from = {}
    replaced_key_counts = {}

    for source_arg in args.source:
        task_id, checkpoint_path = parse_source_arg(source_arg)
        src_state, src_meta, _ = load_payload(checkpoint_path)
        prefix = f"heads.{task_id}."
        base_keys = sorted(key for key in output_state.keys() if key.startswith(prefix))
        src_keys = sorted(key for key in src_state.keys() if key.startswith(prefix))

        if not base_keys:
            raise KeyError(f"No base checkpoint keys found for task '{task_id}' with prefix '{prefix}'.")
        if not src_keys:
            raise KeyError(f"No source checkpoint keys found for task '{task_id}' with prefix '{prefix}'.")
        if set(base_keys) != set(src_keys):
            only_base = sorted(set(base_keys) - set(src_keys))
            only_src = sorted(set(src_keys) - set(base_keys))
            raise ValueError(
                f"Task head key mismatch for task '{task_id}'.\n"
                f"Only in base (first 10): {only_base[:10]}\n"
                f"Only in source (first 10): {only_src[:10]}"
            )

        for key in base_keys:
            if tuple(output_state[key].shape) != tuple(src_state[key].shape):
                raise ValueError(
                    f"Shape mismatch for key '{key}': "
                    f"{tuple(output_state[key].shape)} vs {tuple(src_state[key].shape)}"
                )
            output_state[key] = src_state[key].clone()

        composed_from[task_id] = {
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "task_decoder_profile": src_meta.get("task_decoder_profile"),
            "task_adapter_profile": src_meta.get("task_adapter_profile"),
        }
        replaced_key_counts[task_id] = len(base_keys)

    output_meta = copy.deepcopy(base_meta)
    output_meta["composed_from"] = {
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "task_sources": composed_from,
    }

    output_payload = copy.deepcopy(base_payload)
    output_payload["state_dict"] = output_state
    output_payload["meta"] = output_meta

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "base_checkpoint": os.path.abspath(args.base_checkpoint),
                "replaced_tasks": list(composed_from.keys()),
                "replaced_key_counts": replaced_key_counts,
                "task_decoder_profile": output_meta.get("task_decoder_profile"),
                "task_adapter_profile": output_meta.get("task_adapter_profile"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
