# Session Notes 2026-07-06

This file records the main experiment conclusions and leaderboard outcomes from the current work session.

## Best current direction

- Base model family: `dinov3_vitb_taskfpn`
- Backbone: shared `DINOv3 ViT-B`
- FPN: `task_specific`
- Training focus: landmark losses with `--measurement-loss-weight 0.0`

## Strongest local-only run

- Run: `vitlarge_dinov3_taskfpn_v1`
- Backbone: `vit_large_patch16_dinov3`
- Local evaluation:
  - Average `MRE`: `11.004783`
  - `A4C`: `14.043523`
  - `AOP`: `6.701760`
  - `FA`: `8.719284`
  - `FUGC`: `4.015719`
  - `HC`: `13.374666`
  - `IVC`: `16.709704`
  - `PLAX`: `11.209807`
  - `PSAX`: `13.906419`
  - `fetal_femur`: `10.362164`
- Public result:
  - rank: `24`
  - submission id: `831250`
  - overall score: `29.88`

This is important because it confirms the larger DINOv3 backbone is stronger on local validation, but public leaderboard generalization is still not better than the `rank 21` ViT-B submission.

## Main conclusions

- `task_specific` FPN consistently performed better than shared FPN.
- `vit_large_patch16_dinov3` is a valid stronger backbone path in this codebase.
- AMP plus gradient accumulation were added to make the larger backbone trainable with small GPU memory budgets.
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
  - `vitlarge_dinov3_taskfpn_v1`
  - `dinov3_vitb_taskfpn_geometryv1_datasetlossv1`
  - `dinov3_vitb_dedicated_head_datasetv1`
  - other older experiments under `output/runs/` and `output/submissions/`

- Latest robust-domain local-best submission:
  - run: `dinov3_vitb_taskfpn_localrefine_v1_robustdomain_v1`
  - rank: `22`
  - submission id: `833256`
  - overall score: `28.86`
  - local average `MRE`: `6.869928`
  - note: much stronger local validation still did not surpass the best hidden-server result

- New second-best public result:
  - run: `dinov3_vitb_taskfpn_grouped_serverproxy_v2_a4c`
  - rank: `18`
  - submission id: `834123`
  - overall score: `27.47`
  - note: stronger A4C-focused checkpoint weighting helped relative to most recent robust-domain runs, but still did not beat `dinov3_vitb_taskfpn_grouped_serverproxy_v1`

- Pseudo-domain robust branch result:
  - run: `dinov3_vitb_taskfpn_pseudodomain_robust_uniform_v1`
  - rank: `41`
  - submission id: `834135`
  - overall score: `34.69`
  - note: pseudo-domain grouping plus robust augmentation severely regressed hidden-server performance and should not be used as the main submission branch

- Balanced fine-tune result:
  - run: `dinov3_vitb_taskfpn_grouped_serverproxy_v1_balanced_ft`
  - rank: `22`
  - submission id: `834497`
  - overall score: `28.31`
  - note: mild balanced hard-task fine-tuning preserved reasonable performance but still underperformed both `grouped_serverproxy_v1` and `grouped_serverproxy_v2_a4c`

- IVC-refine result:
  - run: `dinov3_vitb_taskfpn_grouped_serverproxy_v1_ivc_refine`
  - rank: `29`
  - submission id: `834676`
  - overall score: `29.56`
  - note: replacing only the IVC head with a stronger coarse-to-fine decoder did not transfer to better hidden-server performance and regressed the public score

- FA-fixed no-split-screen result:
  - run: `dinov3_vitb_taskfpn_grouped_serverproxy_v1_fa_fixed_nosplits`
  - rank: `36`
  - submission id: `834905`
  - overall score: `32.67`
  - note: applying the FA correction but excluding detected split-screen cardiac rows caused a major hidden-server regression; split-screen exclusion should not be used

- Recovery result:
  - run: `dinov3_vitb_taskfpn_grouped_serverproxy_v1_recovery`
  - rank: `23`
  - submission id: `835027`
  - overall score: `28.31`
  - note: removing the FA auto-fix and keeping split-screen cardiac rows restored the stable behavior; this matches the earlier conclusion that the regression came from the training-pipeline changes

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

## 2026-07-12 update

- The `offset128_v1` architectural change produced the new best public result so far.
- run: `dinov3_vitb_hidden_context_offset128_v1_seed42`
- rank: `8`
- submission id: `844351`
- overall score: `24.77`
- note: this is the first architecture-level improvement after the `25.31` plateau. It keeps the hidden-context DINOv3 ViT-B/task-specific-FPN family, increases heatmap resolution to `128x128`, and adds learned subpixel offset refinement for dense point tasks.
- Next controlled branch: `hidden_context_local_offset128_v2` keeps the `offset128_v1` architecture and adds zero-started post-offset correction only for `HC` and `IVC`, trained from the `24.77` checkpoint with the encoder/FPN/adapters and non-target heads frozen.
- If `offset128_v2` is unchanged, the next branch is `offset128_v3_hardtask_ft`: train the task-specific FPN/adapters/heads for `A4C,AOP,HC,IVC,PLAX` from the `24.77` checkpoint, freeze only the DINOv3 encoder and non-target heads, and keep FA/FUGC/PSAX/femur behavior anchored.
- `offset128_v3_hardtask_ft` result: local average MRE improved to `5.562717`, but CodaBench overall was `24.87`, slightly worse than `offset128_v1`. This indicates local hard-task fitting exceeded the hidden-server optimum; use it for conservative checkpoint soup/backoff or as an analysis branch, not as the anchor.
- `offset128_soup_v1_a85` candidate was built by merging 85% `offset128_v2` and 15% `offset128_v3_hardtask_ft`. Local average MRE is `5.604717`; pre-submit audit versus `offset128_v1` passed with only `0.1601px` mean shift. Hidden score was `24.79`, close but not better than `offset128_v1`.
- `offset128_soup_v1_a95` candidate was built by merging 95% `offset128_v2` and 5% `offset128_v3_hardtask_ft`. Local average MRE is `5.611312`; pre-submit audit versus `offset128_v1` passed with only `0.1172px` mean shift. Hidden score was also `24.79`, so checkpoint soup is not the next route for improvement.
- `offset256_v1` result: local average MRE improved to `5.525950`, but CodaBench submission `845990` scored `24.87` (`rank=9` at submission time), not better than the `24.77` anchor. The run early-stopped after `14` epochs, with the best checkpoint at epoch `1`, so the extra resolution mostly preserved the warm-started model and did not generalize better.
- New high-leverage branch: `hidden_context_local_offset128_crop_panel_v1`. It keeps split-screen cardiac rows but crops them to the landmark-containing panel for train/internal-val, matching the organizer statement that the hidden test set has no split-screen images. Smoke run passed and detected `23` split-screen cardiac rows (`A4C=7`, `IVC=14`, `PSAX=2`).
- New structure-aware branch: `hidden_context_structure_v1`. It keeps the best `offset128_v1` backbone/FPN/context-local/128-heatmap recipe, but adds a supervised anatomy support map to each task head. The goal is to make the model learn the line, contour, vessel band, or shaft shape around the landmarks rather than fitting isolated points only. Use `scripts/train_hidden_context_structure_v1.sh`, then `scripts/predict_eval_hidden_context_structure_v1.sh`, then `scripts/submit_hidden_context_structure_v1.sh`.
- `hidden_context_structure_v1` hidden result was poor (`26.42`). Treat this as a failed branch: the structure-map auxiliary supervision appears to over-regularize or shift the landmark distribution away from the server optimum. Do not use it as the next anchor.
- `hidden_context_local_offset128_crop_panel_v1` also overfit quickly. Treat full fine-tuning away from `offset128_v1` as risky. The next safer branch is `hidden_context_offset128_microfit_v1`: same `offset128_v1` profile, encoder/FPN frozen, very low learning rate (`1e-6`), gradient accumulation (`4`), and short patience (`6`) to test whether heads/adapters can improve without drifting from the `24.77` anchor.
- Visual inspection sheets were generated under `assets/dataset_inspection/`. They show that many target landmarks are boundary endpoints on fine ultrasound edges and speckle structures, not only semantic regions. This motivates `hidden_context_texture_offset128_v1`: a one-model dual-stream branch that keeps the `offset128_v1` DINOv3/FPN/head design but replaces the adapter with an ultrasound texture-context adapter using RGB, grayscale, Sobel gradients, and gradient magnitude.
- `hidden_context_texture_offset128_v1` overfit early because replacing the adapter skipped the old context-local adapter weights. The safer follow-up is `hidden_context_texture_residual_offset128_v1`, which preserves the old adapter key layout (`skipped=0` in warm-start smoke) and adds only a zero-initialized Sobel/RGB texture residual.
- `hidden_context_texture_residual_offset128_v2` is the next controlled model branch. It keeps the same `offset128_v1` anchor and context-local warm start, but changes the texture residual from one global scale to conservative per-task gates. Smoke training passed: warm start matched `1463` keys, skipped `0`, and only the new texture/gate parameters were missing.
- Pseudo-hardtasks round-2 was the previous best.
- run: `dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local`
- rank: `12`
- submission id: `843991`
- overall score: `25.31`
- note: this remains the best pre-offset anchor checkpoint and is the warm-start source for `offset128_v1`.
- Pseudo-labeled target-domain adaptation on the official validation images produced the new best public result so far.
- run: `dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42`
- rank: `13`
- submission id: `842049`
- overall score: `25.32`
- note: adapting only the harder hidden-gap tasks on the unlabeled validation distribution beat the earlier BN-only recalibration branch.
- BN recalibration on the official unlabeled validation manifest produced the new best public result so far.
- run: `dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1`
- rank: `13`
- submission id: `842013`
- overall score: `25.37`
- note: this became the predecessor branch once pseudo-hardtasks adaptation improved it slightly.
- seed-73 BN recalibration follow-up was close but slightly worse.
- run: `dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_seed73_v1`
- rank: `13`
- submission id: `842018`
- overall score: `25.42`
- note: same adaptation idea works, but the seed-42 FUGC+FA base remains the better source checkpoint.
