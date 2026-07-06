# Dedicated-Head FU_Biometry

This document describes the dedicated-head model family built for this repository after moving away from a single generic decoder design.

The goal of this branch is simple:

- keep one shared ultrasound foundation backbone
- keep one shared feature neck
- specialize the decoder per dataset/task anatomy
- train and validate with losses that better match each task geometry

## Architecture Figure

![Dedicated-head architecture](assets/fu_biometry_dedicated_heads_architecture.png)

The figure is conceptual. The exact implementation is defined by the code in [baseline/model_factory.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/model_factory.py), [baseline/train.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/train.py), [baseline/utils.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/utils.py), and [submit.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/submit.py).

## Why This Exists

The earlier model families used one decoder family for several tasks. That was practical, but not ideal for this challenge because the datasets are very different:

- `A4C` and `PLAX` are dense multi-landmark cardiac structure tasks.
- `HC` and `PSAX` are compact ring-like tasks.
- `AOP` and `FA` are compact but geometry-specific tasks.
- `IVC`, `FUGC`, and `fetal_femur` are 2-point tasks, but not the same kind of 2-point task.

That means a single shared decoder shape is too weak for some tasks and too generic for others.

## Current Model Design

The dedicated-head setup keeps:

- one shared `DINOv3 ViT-B` encoder
- one shared `FPN` neck
- one task-specific decoder head per challenge dataset

High-level flow:

1. input ultrasound image is letterboxed to a square input
2. shared `DINOv3` encoder extracts global features
3. shared `FPN` builds multi-scale features
4. task-specific decoder head predicts landmark heatmaps
5. some tasks also produce auxiliary outputs used only during training

## Dedicated Decoder Map

The `dedicated_v1` decoder profile currently maps tasks like this:

| Task | Decoder | Reason |
| --- | --- | --- |
| `A4C` | `a4c` | chamber-aware dense cardiac layout |
| `AOP` | `aop` | compact arc-like anatomy |
| `FA` | `fa` | axis-aware fetal abdomen geometry |
| `FUGC` | `fugc` | local 2-point segment with auxiliary short-segment branch |
| `HC` | `hc` | ring-aware head circumference structure |
| `IVC` | `ivc` | noisy short local vessel diameter |
| `PLAX` | `plax` | long-axis dense cardiac structure |
| `PSAX` | `psax` | short-axis compact ring with paired diameters |
| `fetal_femur` | `femur` | shaft-aware long bone endpoints with auxiliary shaft branch |

Code anchor:

- decoder profile preset: [baseline/model_factory.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/model_factory.py)

## Head Sizing Profile

The repository also supports task-specific head width scaling through `challenge_v1`.

Current sizing intent:

- `heavy`: `A4C`, `HC`, `PLAX`, `fetal_femur`
- `light`: `AOP`, `FUGC`, `IVC`
- `medium`: `FA`, `PSAX`

This is applied on top of the decoder family choice. So two tasks can both use custom decoders while still having different channel budgets.

## Loss Design

The training objective is not just one loss for all datasets.

Base losses:

- heatmap loss
- coordinate loss

Optional challenge-oriented losses:

- measurement loss
- dataset-family geometry loss

Task-specific auxiliary losses:

- `fetal_femur`: shaft mask loss
- `FUGC`: short segment mask loss

The dataset-family profile currently available is `dataset_v1`:

- `dense`: `A4C`, `PLAX`
- `compact`: `HC`, `AOP`, `FA`, `PSAX`
- `line`: `FUGC`, `IVC`, `fetal_femur`

This is implemented in [baseline/utils.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/utils.py).

## Challenge Metric Alignment

The official challenge ranking is based on:

- `50%` normalized landmark `MRE`
- `50%` normalized measurement error

This repo now supports validation reporting for:

- raw `MRE`
- raw measurement `MAE`
- normalized `MRE`
- normalized measurement `MAE`
- combined score

Important:

- the code is now closer to the challenge metric than the original baseline
- it is still a training approximation, not the organizer's hidden final evaluation code

## Main Training Arguments

The dedicated-head workflow depends mainly on these flags:

- `--fpn-mode`
- `--task-head-profile`
- `--task-decoder-profile`
- `--task-loss-family-profile`
- `--dataset-loss-weight`
- `--measurement-loss-weight`
- `--femur-shaft-loss-weight`
- `--fugc-segment-loss-weight`

See the parser in [baseline/train.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/train.py).

## Recommended Training Command

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --fpn-mode shared \
  --task-head-profile challenge_v1 \
  --task-decoder-profile dedicated_v1 \
  --task-loss-family-profile dataset_v1 \
  --measurement-loss-weight 0.0 \
  --dataset-loss-weight 0.02 \
  --femur-shaft-loss-weight 0.15 \
  --fugc-segment-loss-weight 0.08 \
  --output-dir output/runs/dinov3_vitb_dedicated_head
```

## Local Inference

```bash
conda activate miccai_fu_biometry

python baseline/model.py \
  --data-root data \
  --checkpoint-path output/runs/dinov3_vitb_dedicated_head/checkpoints/best_model.pth \
  --output-dir output/runs/dinov3_vitb_dedicated_head/predictions \
  --fpn-mode shared \
  --task-head-profile challenge_v1 \
  --task-decoder-profile dedicated_v1
```

## Local Evaluation

```bash
conda activate miccai_fu_biometry

python baseline/evaluate.py \
  --data-root data \
  --pred-root output/runs/dinov3_vitb_dedicated_head/predictions \
  --output-file output/runs/dinov3_vitb_dedicated_head/evaluation_results.json \
  --summary-file output/runs/dinov3_vitb_dedicated_head/evaluation_summary.txt
```

## Challenge Submission

`submit.py` can infer most model settings from checkpoint metadata. The important part is that the checkpoint must have been trained with the dedicated-head configuration.

```bash
conda activate miccai_fu_biometry

python submit.py \
  --checkpoint-path output/runs/dinov3_vitb_dedicated_head/checkpoints/best_model.pth \
  --output-dir output/submissions/dinov3_vitb_dedicated_head \
  --batch-size 8 \
  --num-workers 4
```

If you want to override manually, `submit.py` also supports:

- `--fpn-mode`
- `--task-head-profile`
- `--task-decoder-profile`
- `--encoder-name`
- `--head-type`
- `--input-size`

## What Changed Compared With Earlier Runs

Earlier strong runs in this repo relied on:

- shared decoder families
- shared or task-specific FPN experiments
- weaker challenge-metric alignment

The dedicated-head branch adds:

- dataset-specific decoder design
- more anatomy-specific inductive bias
- auxiliary supervision for `FUGC` and `fetal_femur`
- challenge-oriented normalized validation reporting

## What To Tune Next

If this model still underperforms on the leaderboard, the next practical levers are:

1. increase `--dataset-loss-weight` carefully
2. try `--measurement-loss-weight` above zero in small controlled runs
3. compare `shared` vs `task_specific` FPN again after dedicated heads
4. add more task-specific auxiliary branches only for the weakest public leaderboard tasks
5. compare `512` vs larger input size if memory allows

## Files To Read

- architecture factory: [baseline/model_factory.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/model_factory.py)
- training loop: [baseline/train.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/train.py)
- losses and evaluation utilities: [baseline/utils.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/utils.py)
- inference script: [baseline/model.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/baseline/model.py)
- submission builder: [submit.py](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/submit.py)

## Asset Used In This Document

- image: [assets/fu_biometry_dedicated_heads_architecture.png](/home/hamze/Documents/MICCAI_FU_Biometry_Challange/assets/fu_biometry_dedicated_heads_architecture.png)

