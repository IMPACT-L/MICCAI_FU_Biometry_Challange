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

Train:

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --epochs 35 \
  --batch-size 4 \
  --num-workers 4 \
  --output-dir output/runs/vitb_letterbox
```

Full local inference:

```bash
conda activate miccai_fu_biometry

python baseline/model.py \
  --data-root data \
  --checkpoint-path output/runs/vitb_letterbox/checkpoints/best_model.pth \
  --output-dir output/runs/vitb_letterbox/predictions
```

Separate local evaluation:

```bash
conda activate miccai_fu_biometry

python baseline/evaluate.py \
  --data-root data \
  --pred-root output/runs/vitb_letterbox/predictions \
  --output-file output/runs/vitb_letterbox/evaluation_results.json \
  --summary-file output/runs/vitb_letterbox/evaluation_summary.txt
```

Challenge submission package:

```bash
conda activate miccai_fu_biometry

python submit.py \
  --checkpoint-path output/runs/vitb_letterbox/checkpoints/best_model.pth \
  --output-dir output/submissions/vitb_letterbox \
  --batch-size 8 \
  --num-workers 4
```

## Improvement plan

The current baseline is improved and challenge-compliant, but it is still far from the top leaderboard cluster. We will follow this order for the next round of work:

1. Add a real held-out validation protocol that is stricter than the current local sanity check.
2. Track per-task validation metrics and select checkpoints based on multi-task generalization, not only average local MRE.
3. Add challenge-oriented inference improvements:
   multi-checkpoint ensemble
   flip-TTA
4. Add measurement-aware validation so checkpoint selection is closer to the official ranking metric.

## Why this plan

- Local `MRE` on the labeled training-side data is optimistic and does not predict CodaBench rank well.
- The challenge ranking uses both landmark accuracy and derived biometric measurement accuracy.
- Our current single-checkpoint submission leaves performance on the table, especially on weaker tasks such as `AOP`, `PLAX`, `fetal_femur`, `FA`, and `A4C`.
