# FU Biometry Workspace

This workspace prepares the MICCAI 2026 FU_Biometry dataset into the layout expected by the public baseline code.

## What this does

- Normalizes raw challenge CSV files into a single schema.
- Builds `data/csv/*.csv` and `data/images/*`.
- Prefers `FA_train_new.csv` over `FA_train.csv`.
- Falls back to `FA_train.csv` and swaps the 3rd/4th landmark columns as requested by the organizers.
- Converts task-specific landmark columns such as `LV_p1_xy`, `LV_p2_xy`, ... into generic `point_1_xy`, `point_2_xy`, ...

## Expected output layout

```text
data/
├── csv/
│   ├── A4C_train.csv
│   ├── AOP_train.csv
│   ├── FA_train.csv
│   ├── FUGC_train.csv
│   ├── HC_train.csv
│   ├── IVC_train.csv
│   ├── PLAX_train.csv
│   ├── PSAX_train.csv
│   └── Reg-Two_3.fetal_femur.csv
└── images/
    ├── A4C/
    ├── AOP/
    ├── FA/
    ├── FUGC/
    ├── HC/
    ├── IVC/
    ├── PLAX/
    ├── PSAX/
    └── fetal_femur/
```

## Usage

First download and extract the official challenge data somewhere locally. Then run:

```bash
python3 scripts/prepare_fu_biometry_dataset.py \
  --raw-root /path/to/extracted/challenge_data \
  --output-root data
```

Default behavior uses symlinks for images. To copy files instead:

```bash
python3 scripts/prepare_fu_biometry_dataset.py \
  --raw-root /path/to/extracted/challenge_data \
  --output-root data \
  --image-mode copy
```

## Notes

- The script assumes the raw archive already contains the official CSV files and image folders.
- If the raw dataset includes duplicate image basenames in the same task, the script preserves unique names with numeric suffixes.
- The public baseline repository has additional issues beyond dataset layout; this step only prepares the data.

## Output layout

All generated artifacts now live under `output/`:

```text
output/
├── runs/
│   ├── default/
│   ├── dinov3_vitb_fpn_deep/
│   ├── real_run/
│   ├── smoke_dinov3_vitb_fpn_deep/
│   ├── vit_base_dinov2_fpn_letterbox/
│   └── vitb_letterbox/
├── submissions/
│   ├── checked/
│   ├── dinov3_vitb_fpn_deep/
│   ├── legacy_submission_output/
│   └── vitb_letterbox/
└── misc/
    ├── log/
    └── predictions/
```

## Current workflow

Dedicated-head documentation:

- [README_dedicated_heads.md](README_dedicated_heads.md)

Current best public submission so far:

- Run: `dinov3_vitb_taskfpn`
- CodaBench rank: `21`
- Submission ID: `829167`
- Overall score: `29.2`

Best full training run:

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --fpn-mode task_specific \
  --measurement-loss-weight 0.0 \
  --output-dir output/runs/dinov3_vitb_taskfpn
```

Full local inference:

```bash
conda activate miccai_fu_biometry

python baseline/model.py \
  --data-root data \
  --checkpoint-path output/runs/dinov3_vitb_taskfpn/checkpoints/best_model.pth \
  --output-dir output/runs/dinov3_vitb_taskfpn/predictions
```

Separate local evaluation:

```bash
conda activate miccai_fu_biometry

python baseline/evaluate.py \
  --data-root data \
  --pred-root output/runs/dinov3_vitb_taskfpn/predictions \
  --output-file output/runs/dinov3_vitb_taskfpn/evaluation_results.json \
  --summary-file output/runs/dinov3_vitb_taskfpn/evaluation_summary.txt
```

Challenge submission package:

```bash
conda activate miccai_fu_biometry

python submit.py \
  --checkpoint-path output/runs/dinov3_vitb_taskfpn/checkpoints/best_model.pth \
  --output-dir output/submissions/dinov3_vitb_taskfpn \
  --batch-size 8 \
  --num-workers 4
```

Weak-task fine-tuning from the current best checkpoint:

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --epochs 200 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1e-5 \
  --init-checkpoint output/runs/dinov3_vitb_taskfpn/checkpoints/best_model.pth \
  --train-task-ids fetal_femur,IVC,A4C,PSAX,HC \
  --fpn-mode task_specific \
  --measurement-loss-weight 0.0 \
  --output-dir output/runs/dinov3_vitb_taskfpn_ft_weak5
```

Current weak-task order from local evaluation of `dinov3_vitb_taskfpn`:

1. `fetal_femur`: `20.172528`
2. `IVC`: `15.827585`
3. `A4C`: `13.290328`
4. `PSAX`: `12.665059`
5. `HC`: `12.132840`
6. `PLAX`: `11.697218`

## Improvement plan

The current baseline is improved and challenge-compliant, but it is still far from the top leaderboard cluster. We will follow this order for the next round of work:

1. Use dataset-dedicated decoders on top of the shared DINOv3 backbone instead of one generic head family for all tasks.
2. Strengthen the weakest tasks first, especially `fetal_femur`, `IVC`, and `FUGC`, without destabilizing already-strong tasks.
3. Track per-task validation metrics and select checkpoints based on multi-task generalization, not only average local MRE.
4. Add challenge-oriented inference improvements where allowed by the challenge packaging constraints.
5. Add stronger measurement-aware training and validation so checkpoint selection is closer to the official ranking metric.

## Why this plan

- Local `MRE` on the labeled training-side data is optimistic and does not predict CodaBench rank well.
- The challenge ranking uses both landmark accuracy and derived biometric measurement accuracy.
- Our current best single-checkpoint submission is already competitive enough to use as a fine-tuning base.
- The current biggest leaderboard weaknesses are `fetal_femur`, `IVC`, and `FUGC`, while broad all-head specialization has repeatedly hurt stronger tasks.

## Dedicated-head status

Current `dedicated_v1` decoder coverage:

- `A4C`: chamber-aware decoder
- `AOP`: arc-aware decoder
- `FA`: axis-aware decoder
- `FUGC`: local-refinement decoder with segment auxiliary branch
- `HC`: ring-aware decoder
- `IVC`: diameter-aware decoder with directional context branches
- `PLAX`: long-axis dense decoder
- `PSAX`: short-axis ring decoder
- `fetal_femur`: shaft-aware decoder with auxiliary shaft branch

Important:

- the intended dedicated-head run must include `--task-loss-family-profile dataset_v1`
- an earlier run named `dinov3_vitb_dedicated_head` used `dedicated_v1` but still had `task loss family profile: uniform`
- use a separate output directory for the real dataset-family-loss experiment

## Targeted weak-task status

Current `weak_tasks_v1` decoder coverage:

- `FUGC`: `fugc`
- `IVC`: `ivc`
- `fetal_femur`: `femur`

All other tasks fall back to the stronger proven `geometry_v1` / baseline path. This is the current recommended direction because it isolates the weak tasks without rewriting the heads that were already stable.

Recommended next full run:

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --fpn-mode task_specific \
  --task-head-profile challenge_v1 \
  --task-decoder-profile weak_tasks_v1 \
  --task-loss-family-profile dataset_v1 \
  --measurement-loss-weight 0.0 \
  --dataset-loss-weight 0.02 \
  --femur-shaft-loss-weight 0.15 \
  --fugc-segment-loss-weight 0.08 \
  --output-dir output/runs/dinov3_vitb_taskfpn_weaktasks_v1
```
