#!/usr/bin/env python
import argparse
import copy
import json
import os
import sys
from collections import defaultdict

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_DIR = os.path.join(REPO_ROOT, "baseline")
sys.path.insert(0, BASELINE_DIR)

from model import load_checkpoint_payload  # noqa: E402


def _parse_task_sources(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected TASK=checkpoint.pth, got: {value}")
        task_id, checkpoint = value.split("=", 1)
        task_id = task_id.strip()
        checkpoint = checkpoint.strip()
        if not task_id or not checkpoint:
            raise ValueError(f"Invalid task source: {value}")
        parsed[task_id] = checkpoint
    return parsed


def _task_prefixes(task_id: str) -> tuple[str, ...]:
    return (f"heads.{task_id}.", f"task_fpns.{task_id}.")


def _copy_task_modules(base_state, donor_state, task_id: str) -> dict[str, int]:
    stats = defaultdict(int)
    for prefix in _task_prefixes(task_id):
        for key, donor_tensor in donor_state.items():
            if not key.startswith(prefix):
                continue
            stats["candidate_keys"] += 1
            if key not in base_state:
                stats["missing_in_base"] += 1
                continue
            if tuple(base_state[key].shape) != tuple(donor_tensor.shape):
                stats["shape_mismatch"] += 1
                continue
            base_state[key] = donor_tensor.clone()
            stats["copied"] += 1
    return dict(stats)


def _architecture_signature(meta: dict) -> dict:
    keys = (
        "encoder_name",
        "use_fpn",
        "fpn_mode",
        "fpn_type",
        "head_type",
        "task_head_profile",
        "task_decoder_profile",
        "task_adapter_profile",
        "input_size",
        "heatmap_size",
    )
    return {key: meta.get(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create one checkpoint by transplanting task-specific heads/FPNs from compatible checkpoints."
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--task-source", action="append", default=[], help="TASK=donor_checkpoint.pth")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--allow-architecture-drift", action="store_true")
    args = parser.parse_args()

    task_sources = _parse_task_sources(list(args.task_source))
    base_state, base_meta = load_checkpoint_payload(args.base_checkpoint, torch.device("cpu"))
    output_state = copy.deepcopy(base_state)
    base_sig = _architecture_signature(base_meta)

    transplant_summary = {}
    donor_sigs = {}
    for task_id, checkpoint_path in task_sources.items():
        donor_state, donor_meta = load_checkpoint_payload(checkpoint_path, torch.device("cpu"))
        donor_sig = _architecture_signature(donor_meta)
        donor_sigs[task_id] = donor_sig
        if donor_sig != base_sig and not args.allow_architecture_drift:
            raise ValueError(
                "Donor architecture does not match base architecture for "
                f"{task_id}.\nBase: {base_sig}\nDonor: {donor_sig}"
            )
        transplant_summary[task_id] = {
            "checkpoint": os.path.abspath(checkpoint_path),
            **_copy_task_modules(output_state, donor_state, task_id),
        }

    output_meta = copy.deepcopy(base_meta)
    output_meta["task_module_transplant"] = {
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "task_sources": transplant_summary,
        "base_signature": base_sig,
        "donor_signatures": donor_sigs,
    }
    payload = {"state_dict": output_state, "meta": output_meta}

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": output_path,
                "num_state_keys": len(output_state),
                "transplant_summary": transplant_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
