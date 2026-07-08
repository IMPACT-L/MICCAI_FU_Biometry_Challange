# Leaderboard Diagnosis 2026-07-08

## Best public branch

- Best public run: `dinov3_vitb_taskfpn_grouped_serverproxy_v1`
- Rank: `17`
- Overall score: `27.36`

This remains the best submission branch even though several later models achieved much lower local validation `MRE`.

## Main conclusion

The project's current bottleneck is not raw local fitting. It is hidden-domain generalization.

Evidence:

- `dinov3_vitb_taskfpn_grouped_serverproxy_v1`
  - local avg `MRE`: `11.349361`
  - public overall: `27.36`
- `dinov3_vitb_taskfpn_grouped_serverproxy_v2_a4c`
  - local avg `MRE`: `10.487067`
  - public overall: `27.47`
- `dinov3_vitb_taskfpn_localrefine_v1_robustdomain_v1`
  - local avg `MRE`: `6.869928`
  - public overall: `28.86`
- `dinov3_vitb_taskfpn_pseudodomain_robust_uniform_v1`
  - public overall: `34.69`

Lower local `MRE` did not translate into a better leaderboard score.

## Current strengths

The best public branch is already relatively close to the local frontier on:

- `AOP`
- `FA`
- `HC`

Local gap to best observed task frontier for the best public branch:

- `AOP`: `7.322165` vs frontier `6.097190` (`1.201x`)
- `FA`: `9.891350` vs frontier `7.790410` (`1.270x`)
- `HC`: `11.610872` vs frontier `8.801749` (`1.319x`)

These tasks are not the main reason the public score is lagging.

## Current weaknesses

The best public branch is far from the best observed local task frontier on:

- `A4C`: `18.053491` vs frontier `7.384643` (`2.445x`)
- `fetal_femur`: `9.042503` vs frontier `3.739983` (`2.418x`)
- `FUGC`: `4.344666` vs frontier `1.855575` (`2.341x`)
- `IVC`: `16.339876` vs frontier `7.399098` (`2.208x`)
- `PSAX`: `13.921470` vs frontier `6.451145` (`2.158x`)
- `PLAX`: `11.617855` vs frontier `7.138314` (`1.628x`)

These are the tasks where there is still substantial room to improve the stable public branch.

## What clearly did not work

- Moving to a larger backbone (`vit_large_patch16_dinov3`) improved local metrics but not public rank.
- Heavy local refinement and coarse-to-fine branches improved visible validation strongly but did not improve hidden-server ranking.
- `bifpn` was clearly worse than the standard `fpn`.
- `pseudo_domain_grouped + ultrasound_robust_v1 + robust_domain_v1` regressed badly on the public server.
- Overfitting to one task family alone is not enough. `serverproxy_v2_a4c` improved local `A4C` and overall local average but still did not beat `serverproxy_v1`.

## What is most likely to improve the leaderboard

Stay on the stable public branch:

- `vit_base_patch16_dinov3`
- `task_specific` `fpn`
- `uniform` decoder
- grouped validation
- server-oriented checkpointing

Do not submit more complex architectures as the main branch unless they first prove themselves on grouped/server-proxy validation and preserve public behavior.

## Recommended next experiments

### 1. Conservative balanced hard-task weighting

Use the best public recipe and apply mild weights to the true weak tasks together, not only `A4C`:

- `A4C`
- `FUGC`
- `IVC`
- `PSAX`
- `fetal_femur`
- optionally `PLAX`

The key point is to keep the architecture simple and only bias sampling and checkpoint selection slightly.

### 2. Fine-tune from the best public checkpoint

Instead of training a new complex model, start from:

- `output/runs/dinov3_vitb_taskfpn_grouped_serverproxy_v1/checkpoints/best_model.pth`

Then fine-tune with:

- low learning rate
- grouped split
- baseline augmentation
- mild hard-task sampler weights

This is safer than switching architectures again.

### 3. Distill precision into the stable branch

The local-refine and coarse-refine models are useful as teachers, not as direct submission models.

Their role should be:

- provide stronger local targets
- improve weak-task localization in the simple public branch
- avoid changing the public-facing architecture

## Practical recommendation

The next submission branch should be:

- the `grouped_serverproxy_v1` family
- with balanced weak-task weighting
- with conservative fine-tuning from the current best public checkpoint

The project should stop spending submission quota on architecture-heavy branches until this safer branch is exhausted.
