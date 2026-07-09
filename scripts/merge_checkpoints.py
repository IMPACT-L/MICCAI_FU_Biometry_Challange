import argparse
import copy
import json
import os

import torch


def load_payload(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("meta", {}), checkpoint
    return checkpoint, {}, {"state_dict": checkpoint, "meta": {}}


def main():
    parser = argparse.ArgumentParser(description="Average two compatible model checkpoints into one merged checkpoint.")
    parser.add_argument("--checkpoint-a", required=True, help="Path to checkpoint A.")
    parser.add_argument("--checkpoint-b", required=True, help="Path to checkpoint B.")
    parser.add_argument("--output-path", required=True, help="Path to write merged checkpoint.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for checkpoint A; checkpoint B gets (1-alpha).")
    args = parser.parse_args()

    alpha = float(args.alpha)
    beta = 1.0 - alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1].")

    state_a, meta_a, payload_a = load_payload(args.checkpoint_a)
    state_b, meta_b, payload_b = load_payload(args.checkpoint_b)

    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        raise ValueError(
            "Checkpoint state_dict keys do not match.\n"
            f"Only in A (first 20): {only_a[:20]}\n"
            f"Only in B (first 20): {only_b[:20]}"
        )

    merged_state = {}
    for key in sorted(keys_a):
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"Shape mismatch for key '{key}': {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}")

        if torch.is_floating_point(tensor_a):
            merged_state[key] = tensor_a * alpha + tensor_b * beta
        else:
            # Keep non-floating tensors from the stronger baseline checkpoint A.
            merged_state[key] = tensor_a.clone()

    merged_meta = copy.deepcopy(meta_a)
    merged_meta.setdefault("encoder_name", meta_b.get("encoder_name"))
    merged_meta.setdefault("use_fpn", meta_b.get("use_fpn"))
    merged_meta.setdefault("fpn_mode", meta_b.get("fpn_mode"))
    merged_meta["fpn_type"] = meta_a.get("fpn_type") or meta_b.get("fpn_type") or "fpn"
    merged_meta.setdefault("head_type", meta_b.get("head_type"))
    merged_meta.setdefault("task_head_profile", meta_b.get("task_head_profile"))
    merged_meta.setdefault("task_decoder_profile", meta_b.get("task_decoder_profile"))
    merged_meta.setdefault("task_adapter_profile", meta_b.get("task_adapter_profile"))
    merged_meta.setdefault("input_size", meta_b.get("input_size"))
    merged_meta.setdefault("heatmap_size", meta_b.get("heatmap_size"))
    merged_meta["merged_from"] = {
        "checkpoint_a": os.path.abspath(args.checkpoint_a),
        "checkpoint_b": os.path.abspath(args.checkpoint_b),
        "alpha": alpha,
        "beta": beta,
    }

    output_payload = {
        "state_dict": merged_state,
        "meta": merged_meta,
    }

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_payload, output_path)

    summary = {
        "output_path": output_path,
        "num_keys": len(merged_state),
        "alpha": alpha,
        "beta": beta,
        "fpn_type": merged_meta.get("fpn_type"),
        "encoder_name": merged_meta.get("encoder_name"),
        "task_decoder_profile": merged_meta.get("task_decoder_profile"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
