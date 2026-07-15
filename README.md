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
- fetal_femur orientation-anomaly note: the raw CSV may still contain the 25 organizer-listed flipped samples, but the training loader excludes them automatically

## Notes

- The script assumes the raw archive already contains the official CSV files and image folders.
- If the raw dataset includes duplicate image basenames in the same task, the script preserves unique names with numeric suffixes.
- Per the organizer note, 25 horizontally flipped `fetal_femur` training images are ignored by `KeypointDataset`; the hidden test set is reported to use standard orientation.
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

- Run: `dinov3_vitb_hidden_context_offset128_v1_femurclean_edgesnap_taskblend_v6_keepbestfemur_seed42`
- CodaBench rank: `6`
- Submission ID: `849816`
- Overall score: `24.22`

Previous best public submission:

- Run: `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v6_top2_seed42`
- CodaBench rank: `7`
- Submission ID: `848235`
- Overall score: `24.27`

Earlier best public submission:

- Run: `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v5_top1_or_top2_seed42`
- CodaBench rank: `8`
- Submission ID: `848186`
- Overall score: `24.29`

Earlier focused-blend public submission:

- Run: `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v4_top3_seed42`
- CodaBench rank: `8`
- Submission ID: `848078`
- Overall score: `24.32`

Earlier focused-blend public submission:

- Run: `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v3_top2_seed42`
- CodaBench rank: `8`
- Submission ID: `848010`
- Overall score: `24.38`

Earlier focused-blend public submission:

- Run: `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v3_top3_seed42`
- CodaBench rank: `8`
- Submission ID: `847976`
- Overall score: `24.45`

Earlier focused-blend public submission:

- Run: `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_top2_or_top3_seed42`
- CodaBench rank: `8`
- Submission ID: `847924`
- Overall score: `24.51`

Earlier best public submission:

- Run: `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_safe_v1_seed42`
- CodaBench rank: `8`
- Submission ID: `847891`
- Overall score: `24.64`

Previous architecture-anchor public submission:

- Run: `dinov3_vitb_hidden_context_offset128_v1_seed42`
- CodaBench rank: `8`
- Submission ID: `844351`
- Overall score: `24.77`

Current second-best public submission so far:

- Run: `dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local`
- CodaBench rank: `12`
- Submission ID: `843991`
- Overall score: `25.31`

Best local validation run so far:

- Run: `vitlarge_dinov3_taskfpn_v1`
- Backbone: `vit_large_patch16_dinov3`
- Local average `MRE`: `11.004783`

Best known server-oriented family so far:

- Backbone: `vit_base_patch16_dinov3`
- Neck: task-specific `FPN`
- Head sizing: `challenge_v1`
- Decoder: `hidden_a4c_hc_ivc_fugc_refine_v1`
- Adapter: `context_local_v1`
- Split: `grouped`
- Augmentation: `baseline`
- Checkpoint score: `server_proxy_v1`
- Best hidden-set variant so far:
  - run: `dinov3_vitb_hidden_context_offset128_v1_seed42`
  - overall score: `24.77`

Current read of the results:

- The best local model is not the best server model.
- Grouped validation and proxy-based checkpoint selection improved server behavior more than simply pushing local MRE lower.
- Pure local average `MRE` is not enough for model selection; the hidden server distribution is different enough that lower local error can still submit worse.
- The current evidence supports a server-oriented proxy score more than any single-task rule.
- The current best direction is the `hidden_context` family: `vit_base_patch16_dinov3` with task-specific FPN, grouped split, baseline augmentation, `context_local_v1` adapters, and the hidden `A4C/HC/IVC/FUGC` refinement decoder.
- The strongest result so far came from target-domain adaptation on the official validation distribution, not from broader decoder changes or stronger augmentation.
- The first clearly positive architecture change after the `25.31` plateau is `offset128_v1`: it keeps the hidden-context family but uses `128x128` heatmaps and learned subpixel offset refinement for dense point tasks.
- The best current public score is no longer from additional training. It comes from the `offset128_v1` anchor plus conservative image-edge snapping for safe boundary tasks and a focused audit-safe taskwise blend from prior specialist submissions.
- The `offset128_v3_hardtask_ft` branch strongly improved local validation but did not beat the hidden server score, so it should be treated as an over-adapted near-miss rather than a new anchor.
- The next high-leverage test is not another weight soup. It is cardiac split-screen normalization: detected split-screen cardiac training rows are kept but cropped to the landmark-containing panel, matching the organizer statement that the test set has no split-screen images.
- A small targeted measurement term can help slightly, but only when it is softened carefully. The stronger/default measurement variant regressed.

Comparable saved runs:

| Run | Local avg MRE | CodaBench overall |
| --- | ---: | ---: |
| `dinov3_vitb_hidden_context_offset128_v1_femurclean_edgesnap_taskblend_v6_keepbestfemur_seed42` | `NA` | `24.22` |
| `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v6_top2_seed42` | `NA` | `24.27` |
| `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v5_top1_or_top2_seed42` | `NA` | `24.29` |
| `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v4_top3_seed42` | `NA` | `24.32` |
| `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v3_top2_seed42` | `NA` | `24.38` |
| `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v3_top3_seed42` | `NA` | `24.45` |
| `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_top2_or_top3_seed42` | `NA` | `24.51` |
| `dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_safe_v1_seed42` | `NA` | `24.64` |
| `dinov3_vitb_hidden_context_offset128_v1_seed42` | `NA` | `24.77` |
| `dinov3_vitb_hidden_context_offset256_v1_seed42` | `5.525950` | `24.87` |
| `dinov3_vitb_hidden_context_offset128_v3_hardtask_ft_seed42` | `5.562717` | `24.87` |
| `dinov3_vitb_hidden_context_offset128_soup_v1_a85` | `5.604717` | `24.79` |
| `dinov3_vitb_hidden_context_offset128_soup_v1_a95` | `5.611312` | `24.79` |
| `dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local` | `NA` | `25.31` |
| `dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42` | `NA` | `25.32` |
| `dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1` | `NA` | `25.37` |
| `dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_headonly_v3_seed42` | `NA` | `25.41` |
| `dinov3_vitb_taskfpn_hidden_context_ft_seed7` | `NA` | `26.45` |
| `dinov3_vitb_taskfpn_hidden_context_ft_v1` | `NA` | `26.62` |
| `dinov3_vitb_taskfpn_hidden_context_ft_seed73` | `NA` | `26.86` |
| `dinov3_vitb_taskfpn_hidden_cluster_ft_seed7` | `8.436502` | `27.03` |
| `dinov3_vitb_taskfpn_hidden_context_ft_seed42` | `NA` | `27.53` |
| `dinov3_vitb_taskfpn_hidden_context_measure_v1` | `NA` | `28.32` |
| `dinov3_vitb_taskfpn_grouped_serverproxy_v1` | `11.349361` | `27.36` |
| `vitlarge_dinov3_taskfpn_v1` | `11.004783` | `29.88` |
| `vitlarge_dinov3_taskfpn_grouped_strongaug_v1` | `11.019315` | `35.68` |

The current comparison still shows the exact gap we care about: the best hidden-validation result still did not come from the lowest local `MRE`, and broad robustness or augmentation changes usually regressed on the public server. Excluding detected split-screen cardiac rows was also strongly harmful, despite the test set containing no split-screen images. The new best score also shows that target-domain pseudo adaptation of the harder hidden tasks can improve the leaderboard even when it does not look obviously dominant from local averages alone.

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

Current best server-oriented training run before hidden-set fine-tuning:

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

Current best hidden-set fine-tuning run:

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --model-profile hidden_context_ft_v1 \
  --seed 7 \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 5e-5 \
  --init-checkpoint output/runs/dinov3_vitb_taskfpn_grouped_serverproxy_v1/checkpoints/best_model.pth \
  --task-loss-weights A4C=1.25,AOP=1.00,FA=1.15,FUGC=1.00,HC=1.35,IVC=1.35,PLAX=1.10,PSAX=1.10,fetal_femur=1.30 \
  --sampler-task-weights AOP=1.20,FA=1.20,HC=1.40,IVC=1.40,fetal_femur=1.25 \
  --output-dir output/runs/dinov3_vitb_taskfpn_hidden_context_ft_seed7
```

Current best hidden-set measurement-tuned run:

```bash
conda activate miccai_fu_biometry

python baseline/train.py \
  --model-profile hidden_context_measure_v1 \
  --seed 42 \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 2e-5 \
  --init-checkpoint output/runs/dinov3_vitb_taskfpn_hidden_context_ft_seed7/checkpoints/best_model.pth \
  --measurement-loss-weight 0.008 \
  --measurement-task-weights FA=1.00,HC=1.15,IVC=1.30,PLAX=0.70,fetal_femur=0.70 \
  --task-loss-weights A4C=1.20,AOP=1.00,FA=1.15,FUGC=1.00,HC=1.40,IVC=1.40,PLAX=1.10,PSAX=1.10,fetal_femur=1.35 \
  --sampler-task-weights AOP=1.15,FA=1.20,HC=1.45,IVC=1.45,fetal_femur=1.30 \
  --output-dir output/runs/dinov3_vitb_taskfpn_hidden_context_measure_seed42_soft
```

Current best architecture run:

```bash
conda activate miccai_fu_biometry

bash scripts/train_hidden_context_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42 \
  output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local/checkpoints/best_model.pth \
  42
```

Next controlled architecture run:

```bash
conda activate miccai_fu_biometry

bash scripts/train_hidden_context_offset128_v2.sh \
  output/runs/dinov3_vitb_hidden_context_offset128_v2_hcivc_seed42 \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth \
  42
```

Next stronger hard-task adaptation run:

```bash
conda activate miccai_fu_biometry

bash scripts/train_hidden_context_offset128_v3_hardtask_ft.sh \
  output/runs/dinov3_vitb_hidden_context_offset128_v3_hardtask_ft_seed42 \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth \
  42
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

Current best server submission package before hidden-set fine-tuning:

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

Current best hidden-set submission package:

```bash
conda activate miccai_fu_biometry

bash scripts/submit_hidden_context_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42 \
  output/submissions/dinov3_vitb_hidden_context_offset128_v1_seed42
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
2. Keep the winning hidden-context family fixed: ViT-B DINOv3, task-specific FPN, `challenge_v1`, `context_local_v1`, and the hidden `A4C/HC/IVC/FUGC` refinement decoder.
3. Use `offset128_v1` as the new anchor architecture and only test nearby variants around it.
4. Prioritize the hard tasks that still show unstable hidden-set transfer, especially `A4C`, `IVC`, and `HC`.
5. Add stronger local audit tooling so model selection is based on server-predictive task behavior, not mean local MRE alone.

## Why this plan

- Local `MRE` on the labeled training-side data is optimistic and does not predict CodaBench rank well.
- The best server result so far came from a model chosen with grouped validation and proxy-based checkpoint selection, not from the best local average MRE.
- The current saved comparisons are still too small to justify a strict single-task rule; the safer conclusion is that validation split design and checkpoint score policy matter more than raw local leaderboard position.
- Broad all-head specialization has repeatedly hurt server generalization, while cleaner shared decoding with task-specific FPN has been more reliable.
- The `offset128_v1` change is the first architecture-side change that clearly improved hidden validation after the `25.31` plateau.
- The `offset128_v3_hardtask_ft` result shows that very low local MRE can still lose hidden performance if the model drifts from the best server-stable checkpoint.
- The `offset256_v1` result shows the same pattern: full local MRE improved to `5.525950`, but hidden score was `24.87`, so increasing heatmap resolution beyond `128x128` is not enough by itself.
- The `offset128_soup_v1_a85` and `offset128_soup_v1_a95` checkpoint soups both scored `24.79`, close but not better than the `24.77` anchor.
- The next non-incremental branch is `hidden_context_local_offset128_crop_panel_v1`, which keeps the `24.77` architecture but normalizes detected split-screen cardiac training/validation images into single-panel examples instead of excluding them.
- Measurement supervision is only useful when applied conservatively; the best hidden-server result still came from the offset-refined hidden-context branch rather than from stronger measurement tuning.

## Audit scripts

Use these scripts to compare local evaluation with saved CodaBench outcomes:

```bash
python scripts/audit_local_server_gap.py
python scripts/audit_local_server_tasks.py
```

They summarize which runs generalized better on the server and which local tasks were the clearest warning signs for poor hidden performance.

## Structure-Aware Branch

The next non-incremental model branch is `hidden_context_structure_v1`. It keeps the best stable `offset128_v1` recipe (`DINOv3 ViT-B`, task-specific FPN, `context_local_v1`, grouped split, `128x128` heatmaps) and adds an auxiliary structure-map output to the task heads. The structure map supervises anatomy support lines or compact shapes in addition to landmark heatmaps, so the model learns the relevant object geometry instead of only isolated points.

Run it from the current best `offset128_v1` checkpoint:

```bash
conda activate miccai_fu_biometry

CUDA_VISIBLE_DEVICES=0 bash scripts/train_hidden_context_structure_v1.sh \
  output/runs/dinov3_vitb_hidden_context_structure_v1_seed42 \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth \
  42
```

Predict/evaluate:

```bash
bash scripts/predict_eval_hidden_context_structure_v1.sh \
  output/runs/dinov3_vitb_hidden_context_structure_v1_seed42
```

Build submission:

```bash
bash scripts/submit_hidden_context_structure_v1.sh \
  output/runs/dinov3_vitb_hidden_context_structure_v1_seed42 \
  output/submissions/dinov3_vitb_hidden_context_structure_v1_seed42
```

## Texture-Context Branch

The larger architecture branch is `hidden_context_texture_offset128_v1`. It keeps the best `offset128_v1` head/heatmap design, but replaces the adapter with a dual-stream ultrasound texture adapter. The adapter fuses DINOv3/FPN semantic features with a raw-image stream built from RGB, grayscale, Sobel gradients, and gradient magnitude. This is intended to recover boundary and speckle evidence that patch-token features can miss.

Run:

```bash
conda activate miccai_fu_biometry

CUDA_VISIBLE_DEVICES=0 bash scripts/train_hidden_context_texture_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_texture_offset128_v1_seed42 \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth \
  42
```

Predict/evaluate:

```bash
bash scripts/predict_eval_hidden_context_texture_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_texture_offset128_v1_seed42
```

Build submission:

```bash
bash scripts/submit_hidden_context_texture_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_texture_offset128_v1_seed42 \
  output/submissions/dinov3_vitb_hidden_context_texture_offset128_v1_seed42
```

## Texture-Residual Branch

The direct `texture_context_v1` adapter overfit early because it replaced the old `context_local_v1` adapter, causing many adapter weights from the `24.77` checkpoint to be skipped during warm start. The safer branch is `hidden_context_texture_residual_offset128_v1`: it keeps the old context-local adapter layout load-compatible and adds only a zero-initialized Sobel/RGB texture residual. The model starts from the previous solution and can learn a small boundary correction without drifting immediately.

Train:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_hidden_context_texture_residual_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_texture_residual_offset128_v1_seed42 \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth \
  42
```

Predict/evaluate:

```bash
bash scripts/predict_eval_hidden_context_texture_residual_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_texture_residual_offset128_v1_seed42
```

Submit package:

```bash
bash scripts/submit_hidden_context_texture_residual_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_texture_residual_offset128_v1_seed42 \
  output/submissions/dinov3_vitb_hidden_context_texture_residual_offset128_v1_seed42
```

The next model-side branch is `hidden_context_texture_residual_offset128_v2`. It keeps the same `offset128_v1` anchor and the same load-compatible context-local pathway, but replaces the single global texture residual scale with conservative per-task gates. This is intended to let weak views use ultrasound edge/texture cues without forcing the same correction onto every task.

Train:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_hidden_context_texture_residual_offset128_v2.sh \
  output/runs/dinov3_vitb_hidden_context_texture_residual_offset128_v2_seed42 \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth \
  42
```

Predict/evaluate:

```bash
bash scripts/predict_eval_hidden_context_texture_residual_offset128_v2.sh \
  output/runs/dinov3_vitb_hidden_context_texture_residual_offset128_v2_seed42
```

Submit package:

```bash
bash scripts/submit_hidden_context_texture_residual_offset128_v2.sh \
  output/runs/dinov3_vitb_hidden_context_texture_residual_offset128_v2_seed42 \
  output/submissions/dinov3_vitb_hidden_context_texture_residual_offset128_v2_seed42
```

## FUGC Segment-Vector Branch

The `hidden_context_fugc_vector_offset128_v1` branch targets the largest remaining hidden-set gap: the two-point FUGC line task. It keeps the `24.77` `offset128_v1` anchor for all tasks and replaces only the FUGC endpoint refiner with a segment-vector refinement module. The old FUGC heatmap and endpoint-offset path is still warm-started, then a zero-started module corrects midpoint, angle, and length. Training freezes the encoder, FPN, adapters, and all non-FUGC heads, so only the FUGC head changes.

Train:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_hidden_context_fugc_vector_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_fugc_vector_offset128_v1_seed42 \
  output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth \
  42
```

Predict/evaluate:

```bash
bash scripts/predict_eval_hidden_context_fugc_vector_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_fugc_vector_offset128_v1_seed42
```

Submit package:

```bash
bash scripts/submit_hidden_context_fugc_vector_offset128_v1.sh \
  output/runs/dinov3_vitb_hidden_context_fugc_vector_offset128_v1_seed42 \
  output/submissions/dinov3_vitb_hidden_context_fugc_vector_offset128_v1_seed42
```

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

## Public Leaderboard Result Log

Lower overall score is better. The table below lists the saved CodaBench leaderboard results recorded in `output/submissions/manual_results`. Accounts `hmrasa`, `hmzrse`, `hrcacs`, and `saharch` are our submissions.

<details>
<summary>Saved leaderboard submissions</summary>

| Date | Rank | Submission | Account | Overall | Run / saved source |
|---|---:|---:|---|---:|---|
| 2026-07-07 14:08 | 21 | 832801 | hmrasa | 28.69 | `20260707_1408_832801` |
| 2026-07-07 17:52 | 20 | 832909 | hmzrse | 28.3 | `20260707_1752_832909` |
| 2026-07-07 17:55 | 28 | 832912 | hmrasa | 30.01 | `20260707_1755_832912` |
| 2026-07-07 18:54 | 19 | 832942 | hmzrse | 27.97 | `20260707_1854_832942` |
| 2026-07-07 20:45 | 34 | 832982 | hmrasa | 32.05 | `20260707_2045_832982` |
| 2026-07-07 23:11 | 21 | 833192 | hmrasa | 28.87 | `20260707_2311_833192` |
| 2026-07-07 23:55 | 22 | 833256 | hmzrse | 28.86 | `20260707_2355_833256` |
| 2026-07-08 07:52 | 18 | 834123 | hmzrse | 27.47 | `20260708_0752_834123` |
| 2026-07-08 08:00 | 41 | 834135 | hmzrse | 34.69 | `20260708_0800_834135` |
| 2026-07-08 09:38 | 22 | 834497 | hmzrse | 28.31 | `20260708_0938_834497` |
| 2026-07-08 10:35 | 29 | 834676 | hmzrse | 29.56 | `20260708_1035_834676` |
| 2026-07-08 12:30 | 36 | 834905 | hmrasa | 32.67 | `20260708_1230_834905` |
| 2026-07-08 14:47 | 23 | 835027 | hmrasa | 28.31 | `20260708_1447_835027` |
| 2026-07-08 21:20 | 33 | 835224 | hrcacs | 30.73 | `20260708_2120_835224` |
| 2026-07-08 23:42 | 32 | 835416 | hrcacs | 30.55 | `20260708_2342_835416` |
| 2026-07-09 07:59 | 32 | 836609 | hrcacs | 30.15 | `20260709_0759_836609` |
| 2026-07-09 09:31 | 30 | 836952 | hrcacs | 29.71 | `20260709_0931_836952` |
| 2026-07-09 09:36 | 41 | 836959 | hrcacs | 33.88 | `20260709_0936_836959` |
| 2026-07-09 11:49 | 26 | 837106 | hmrasa | 29.2 | `20260709_1149_837106` |
| 2026-07-09 11:52 | 50 | 837110 | hmrasa | 39.19 | `20260709_1152_837110` |
| 2026-07-09 12:57 | 19 | 837161 | hmrasa | 27.18 | `20260709_1257_837161` |
| 2026-07-09 13:02 | 20 | 837167 | hmrasa | 27.42 | `20260709_1302_837167` |
| 2026-07-09 13:31 | 25 | 837185 | hmrasa | 28.88 | `20260709_1331_837185` |
| 2026-07-09 13:48 | 24 | 837199 | hmzrse | 28.35 | `20260709_1348_837199` |
| 2026-07-09 14:08 | 19 | 837203 | hmzrse | 27.2 | `20260709_1408_837203` |
| 2026-07-09 14:49 | 31 | 837225 | hmzrse | 30.21 | `20260709_1449_837225` |
| 2026-07-09 15:02 | 18 | 837232 | hmzrse | 27.03 | `20260709_1502_837232` |
| 2026-07-09 15:09 | 24 | 837234 | hmzrse | 28.74 | `20260709_1509_837234` |
| 2026-07-09 21:21 | 25 | 837366 | hmzrse | 28.1 | `20260709_2121_837366` |
| 2026-07-09 23:04 | 18 | 837583 | hmzrse | 27.09 | `20260709_2304_837583` |
| 2026-07-09 23:06 | 16 | 837584 | hmzrse | 26.62 | `20260709_2306_837584` |
| 2026-07-10 00:15 | 23 | 837737 | hmzrse | 27.78 | `20260710_0015_837737` |
| 2026-07-10 00:20 | 16 | 837762 | hmzrse | 26.45 | `20260710_0020_837762` |
| 2026-07-10 00:48 | 21 | 837881 | hrcacs | 27.25 | `20260710_0048_837881` |
| 2026-07-10 07:50 | 16 | 838695 | hrcacs | 26.43 | `20260710_0750_838695` |
| 2026-07-10 07:55 | 21 | 838701 | hrcacs | 27.53 | `20260710_0755_838701` |
| 2026-07-10 08:04 | 18 | 838718 | hrcacs | 26.86 | `20260710_0804_838718` |
| 2026-07-10 08:08 | 26 | 838729 | hrcacs | 28.32 | `20260710_0808_838729` |
| 2026-07-10 08:44 | 17 | 838795 | hmrasa | 26.55 | `20260710_0844_838795` |
| 2026-07-10 09:24 | 16 | 838911 | hmrasa | 26.36 | `20260710_0924_838911` |
| 2026-07-10 10:03 | 15 | 839026 | hmrasa | 26.11 | `20260710_1003_839026` |
| 2026-07-10 11:38 | 13 | 839161 | hmrasa | 25.95 | `20260710_1138_839161` |
| 2026-07-10 14:05 | 16 | 839257 | hmrasa | 26.2 | `20260710_1405_839257` |
| 2026-07-10 15:05 | 13 | 839295 | saharch | 25.71 | `20260710_1505_839295` |
| 2026-07-10 15:38 | 13 | 839320 | saharch | 25.69 | `20260710_1538_839320` |
| 2026-07-10 16:03 | 13 | 839340 | saharch | 25.6 | `20260710_1603_839340` |
| 2026-07-10 16:15 | 13 | 839346 | saharch | 25.6 | `20260710_1615_839346` |
| 2026-07-10 18:11 | 13 | 839371 | saharch | 25.66 | `20260710_1811_839371` |
| 2026-07-10 20:47 | 13 | 839473 | saharch | 25.6 | `20260710_2047_839473` |
| 2026-07-10 20:58 | 13 | 839478 | saharch | 25.6 | `20260710_2058_839478` |
| 2026-07-10 21:58 | 13 | 839525 | saharch | 25.88 | `20260710_2158_839525` |
| 2026-07-10 22:46 | 13 | 839554 | saharch | 25.66 | `20260710_2246_839554` |
| 2026-07-10 23:59 | 13 | 839619 | saharch | 25.7 | `20260710_2359_839619` |
| 2026-07-11 01:37 | 34 | 839724 | hmzrse | 30.91 | `20260711_0137_839724` |
| 2026-07-12 01:30 | 13 | 842013 | hrcacs | 25.37 | `20260712_0130_842013` |
| 2026-07-12 01:34 | 13 | 842018 | hrcacs | 25.42 | `20260712_0134_842018` |
| 2026-07-12 02:06 | 13 | 842049 | hrcacs | 25.32 | `20260712_0206_842049` |
| 2026-07-12 17:11 | 12 | 843991 | hmzrse | 25.31 | `20260712_1711_843991` |
| 2026-07-12 18:52 | 15 | 844011 | hmzrse | 25.52 | `20260712_1852_844011` |
| 2026-07-12 20:11 | 14 | 844122 | hmzrse | 25.42 | `20260712_2011_844122` |
| 2026-07-12 22:14 | 14 | 844252 | hmzrse | 25.38 | `dinov3_vitb_hidden_context_two_stage_roi_zoom_jitter_safe_v2_seed42_real` |
| 2026-07-12 23:22 | 8 | 844351 | saharch | 24.77 | `20260712_2322_844351` |
| 2026-07-13 | - | - | - | 24.79 | `dinov3_vitb_hidden_context_offset128_soup_v1_a85` |
| 2026-07-13 | - | - | - | 24.87 | `dinov3_vitb_hidden_context_offset128_v3_hardtask_ft_seed42` |
| 2026-07-13 | - | unknown | - | 24.79 | `dinov3_vitb_hidden_context_offset128_soup_v1_a95` |
| 2026-07-13 09:15 | 9 | 845990 | saharch | 24.87 | `dinov3_vitb_hidden_context_offset256_v1_seed42` |
| 2026-07-14 20:47 | 6 | 850233 | hmzrse | 24.21 | `20260714_2047_850233` |
| 2026-07-15 | - | - | - | 24.22 | `dinov3_vitb_hidden_context_target_equiv_blend_safe_v2_mid_seed42` |
| 2026-07-15 | - | - | - | 24.23 | `dinov3_vitb_hidden_context_target_equiv_blend_safe_v2_fugc_hc_seed42` |
| 2026-07-15 | - | - | - | 24.88 | `dinov3_vitb_hidden_context_boundary_offset128_v1_seed42` |

</details>
