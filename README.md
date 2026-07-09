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

Audit the prepared dataset against the organizer notes and current validation manifest:

```bash
python scripts/audit_challenge_dataset.py
```

This audit verifies:
- expected training CSV row counts and landmark counts
- validation manifest total (`619`) and per-task counts, including `FUGC=20`
- absence of duplicate validation entries
- FA point-order sanity after the organizer-requested fix path

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

- Run: `dinov3_vitb_taskfpn_grouped_serverproxy_v1`
- CodaBench rank: `17`
- Submission ID: `832817`
- Overall score: `27.36`

Current second-best public submission so far:

- Run: `dinov3_vitb_taskfpn_grouped_serverproxy_v2_a4c`
- CodaBench rank: `18`
- Submission ID: `834123`
- Overall score: `27.47`

Best local validation run so far:

- Run: `vitlarge_dinov3_taskfpn_v1`
- Backbone: `vit_large_patch16_dinov3`
- Local average `MRE`: `11.004783`
- Best known server-oriented run:
  - run: `dinov3_vitb_taskfpn_grouped_serverproxy_v1`
  - local average `MRE`: `11.349361`
  - CodaBench rank: `17`
  - overall score: `27.36`

Current read of the results:

- The best local model is not the best server model.
- Grouped validation and proxy-based checkpoint selection improved server behavior more than simply pushing local MRE lower.
- Pure local average `MRE` is not enough for model selection; the hidden server distribution is different enough that lower local error can still submit worse.
- The current evidence supports a server-oriented proxy score more than any single-task rule.
- The current best direction is `vit_base_patch16_dinov3` with task-specific FPN, grouped split, baseline augmentation, and server-oriented checkpoint selection.

Comparable saved runs:

| Run | Local avg MRE | CodaBench overall |
| --- | ---: | ---: |
| `dinov3_vitb_taskfpn_grouped_serverproxy_v1` | `11.349361` | `27.36` |
| `dinov3_vitb_taskfpn_grouped_serverproxy_v2_a4c` | `NA` | `27.47` |
| `dinov3_vitb_taskfpn_grouped_serverproxy_v1_balanced_ft` | `NA` | `28.31` |
| `dinov3_vitb_taskfpn_grouped_serverproxy_v1_recovery` | `NA` | `28.31` |
| `dinov3_vitb_taskfpn_localrefine_v1_robustdomain_v1` | `6.869928` | `28.86` |
| `dinov3_vitb_taskfpn_grouped_serverproxy_v1_ivc_refine` | `NA` | `29.56` |
| `dinov3_vitb_taskfpn_grouped_serverproxy_v1_fa_fixed_nosplits` | `24.389033` | `32.67` |
| `dinov3_vitb_taskfpn_pseudodomain_robust_uniform_v1` | `NA` | `34.69` |
| `vitlarge_dinov3_taskfpn_v1` | `11.004783` | `29.88` |
| `vitlarge_dinov3_taskfpn_grouped_strongaug_v1` | `11.019315` | `35.68` |

The current comparison shows the exact gap we care about: the best hidden-validation result still did not come from the lowest local `MRE`, and even the more aggressive pseudo-domain robustness branch regressed badly on the public server. Excluding detected split-screen cardiac rows was also strongly harmful, despite the test set containing no split-screen images. Future runs should therefore be judged with grouped validation and server-proxy style checkpointing rather than local mean `MRE` alone.

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

Current best server-oriented training run:

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --encoder-name vit_base_patch16_dinov3 \
  --fpn-mode task_specific \
  --task-head-profile challenge_v1 \
  --task-decoder-profile uniform \
  --task-adapter-profile uniform \
  --split-mode grouped \
  --augmentation-profile baseline \
  --checkpoint-score-mode server_proxy_v1 \
  --output-dir output/runs/dinov3_vitb_taskfpn_grouped_serverproxy_v1
```

Best large-backbone training run:

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --encoder-name vit_large_patch16_dinov3 \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 4 \
  --fpn-mode task_specific \
  --task-head-profile challenge_v1 \
  --task-decoder-profile uniform \
  --task-loss-family-profile uniform \
  --measurement-loss-weight 0.0 \
  --dataset-loss-weight 0.0 \
  --ema-decay 0.999 \
  --output-dir output/runs/vitlarge_dinov3_taskfpn_v1
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

Current best server submission package:

```bash
conda activate miccai_fu_biometry

python submit.py \
  --manifest data/manifests/validation_manifest.csv \
  --checkpoint-path output/runs/dinov3_vitb_taskfpn_grouped_serverproxy_v1/checkpoints/best_model.pth \
  --output-dir output/submissions/dinov3_vitb_taskfpn_grouped_serverproxy_v1 \
  --batch-size 8 \
  --num-workers 4 \
  --encoder-name vit_base_patch16_dinov3 \
  --head-type deep \
  --task-head-profile challenge_v1 \
  --task-decoder-profile uniform \
  --task-adapter-profile uniform \
  --fpn-mode task_specific
```

Large-backbone submission package:

```bash
conda activate miccai_fu_biometry

python submit.py \
  --checkpoint-path output/runs/vitlarge_dinov3_taskfpn_v1/checkpoints/best_model.pth \
  --output-dir output/submissions/vitlarge_dinov3_taskfpn_v1 \
  --batch-size 8 \
  --num-workers 4 \
  --encoder-name vit_large_patch16_dinov3 \
  --fpn-mode task_specific \
  --task-head-profile challenge_v1 \
  --task-decoder-profile uniform
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

1. Keep server-oriented checkpoint selection and grouped validation as the default path for new runs.
2. Prioritize the hard tasks that still show unstable hidden-set transfer, especially `A4C`, `IVC`, and `HC`.
3. Improve `IVC` and `AOP` without sacrificing current strengths in `FUGC`, `fetal_femur`, and `PLAX`.
4. Continue testing ViT-B variants before returning to larger backbones, since ViT-L improved local MRE but did not yet beat the best server result.
5. Add stronger local audit tooling so model selection is based on server-predictive task behavior, not mean local MRE alone.

## Why this plan

- Local `MRE` on the labeled training-side data is optimistic and does not predict CodaBench rank well.
- The best server result so far came from a model chosen with grouped validation and proxy-based checkpoint selection, not from the best local average MRE.
- The current saved comparisons are still too small to justify a strict single-task rule; the safer conclusion is that validation split design and checkpoint score policy matter more than raw local leaderboard position.
- Broad all-head specialization has repeatedly hurt server generalization, while cleaner shared decoding with task-specific FPN has been more reliable.

## Audit scripts

Use these scripts to compare local evaluation with saved CodaBench outcomes:

```bash
python scripts/audit_local_server_gap.py
python scripts/audit_local_server_tasks.py
```

They summarize which runs generalized better on the server and which local tasks were the clearest warning signs for poor hidden performance.

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
