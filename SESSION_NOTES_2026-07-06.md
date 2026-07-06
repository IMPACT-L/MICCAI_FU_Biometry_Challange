# Session Notes 2026-07-06

This file records the main experiment conclusions and leaderboard outcomes from the current work session.

## Best current direction

- Base model family: `dinov3_vitb_taskfpn`
- Backbone: shared `DINOv3 ViT-B`
- FPN: `task_specific`
- Training focus: landmark losses with `--measurement-loss-weight 0.0`

## Main conclusions

- `task_specific` FPN consistently performed better than shared FPN.
- Adding measurement loss in the current implementation hurt validation MRE and did not improve the practical submission direction.
- Replacing all task heads at once with `dedicated_v1` was too aggressive and often regressed already-strong tasks.
- The most persistent weak tasks are:
  - `fetal_femur`
  - `IVC`
  - `FUGC`

## Public leaderboard snapshots saved in this repo

- Best public result so far:
  - run: `dinov3_vitb_taskfpn`
  - rank: `21`
  - submission id: `829167`
  - saved under:
    - `output/runs/dinov3_vitb_taskfpn/`
    - `output/submissions/dinov3_vitb_taskfpn/`

- Additional saved results:
  - `dinov3_vitb_taskfpn_geometryv1_datasetlossv1`
  - `dinov3_vitb_dedicated_head_datasetv1`
  - other older experiments under `output/runs/` and `output/submissions/`

## Why `weak_tasks_v1` was added

`weak_tasks_v1` keeps the stable proven path for stronger tasks and specializes only:

- `FUGC` -> `fugc`
- `IVC` -> `ivc`
- `fetal_femur` -> `femur`

This was chosen because:

- all-head specialization changed too many tasks at once
- weak-task-only specialization is the cleanest next ablation
- the goal is to improve leaderboard rank without destabilizing `AOP`, `FA`, `HC`, `PLAX`, and `PSAX`

## Recommended next run

```bash
conda activate miccai_fu_biometry

CUDA_VISIBLE_DEVICES=0 python baseline/train.py \
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
