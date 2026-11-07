# Seed-44 DEM surrogate baseline

Development evaluation, 2026-11-07

## Setup and reference checks

- Checkpoint `processor_data42_finetune_rollout4_v1_best.pt`, epoch 5. One processor round, interaction and early gates enabled, gravity correction with a 5 µs DEM reference timestep, and contact position fraction 1.0.
- Seed 44, 322 particles, drum speed 10 RPM. Initial DEM state at step 6000 (0.030 s), with history at step 5990. Each model prediction advances 50 µs, or 10 DEM steps. Autonomous runs use DEM only for initialization and evaluation.
- The dense and sparse seed-44 files agree exactly in IDs, groups, radii, and all nine state components at their 111 shared steps. The step-indexed dense evaluator reproduced the saved 50 ms result. The sparse evaluator matched all nine shared reports at the original cadence and skipped the two absent targets.
- The existing sparse file lacks DEM steps 6200 through 8000 on its 200-step grid, leaving a gap at elapsed times 1–10 ms. The writer rule was corrected in source. A regenerated 1 ms series remains pending.
- Bulk evaluation uses joint group-density total variation (TV) at 4 mm and 5 mm. The mixing score uses 4 mm cells with at least five particles and reports the fraction of particles included. Position, velocity, and spin RMSE pool components across matched particle IDs.

## Autonomous rollout results

| Elapsed time | Predictions completed | Final DEM step | Position RMSE (mm) | Velocity RMSE (m/s) | Spin RMSE (rad/s) | Joint TV, 4 / 5 mm | Mixing score, ML / DEM | Rotational energy, ML / DEM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 1,000 | 16000 | 0.241 | 0.02430 | 28.31 | 0.071 / 0.068 | — | 1.15× |
| 70 ms | 1,400 | 20000 | 0.387 | 0.01997 | 23.83 | 0.096 / 0.087 | 0.137 / 0.099 | 1.25× |
| 0.1 s | 2,000 | 26000 | 0.486 | 0.01010 | 13.40 | 0.124 / 0.071 | 0.146 / 0.145 | 2.22× |
| 0.5 s | 10,000 | 106000 | 1.445 | 0.01206 | 13.50 | 0.258 / 0.224 | 0.187 / 0.091 | 0.16× |
| 1.0 s | 20,000 | 206000 | 1.987 | 0.00711 | 10.99 | 0.220 / 0.180 | 0.171 / 0.262 | 0.50× |

The 50 ms row uses the original reproduction log, which did not report mixing. In the extended runs, model endpoint sampling found peak overlaps of 52.41 µm for particle pairs and 34.67 µm for walls. All runs completed under the fixed guard of 104.83 / 56.82 µm. That guard used twice the fine-step DEM peaks of 52.41 / 28.41 µm from steps 6000–16000. Fine-step DEM peaks for the longer windows are still pending.

Spin error was already 2.58 rad/s on the first prediction, while the binned group distributions still matched. Group-density TV was nonzero by 5 ms in the dense log. The first recorded mixing-score difference above floating-point noise was at 29 ms. Across the 1 s run, the largest recorded absolute mixing-score difference was 0.124 at 516 ms. At 1 s, the mixing score included 66.5% of ML particles and 68.0% of DEM particles. Position error grew between 0.5 and 1 s even as endpoint TV decreased at both bin widths.

## Paired clean-history diagnostic

For each reported target through DEM step 20000, initialize a separate prediction from the two dense DEM states 50 µs and 100 µs before the target. Compare it with the autonomous prediction at the same target. The separate prediction never feeds back into the autonomous state. The log contains 61 pairs, including the first prediction, where the two model inputs and outputs coincide. The medians below use the 60 later pairs at 11–70 ms.

| Metric | Clean-history median | Autonomous median |
|---|---:|---:|
| Position RMSE | 0.0327 µm | 174 µm |
| Velocity RMSE | 0.000469 m/s | 0.0200 m/s |
| Spin RMSE | 0.615 rad/s | 22.74 rad/s |
| Absolute rotational-energy error | 2.81 × 10⁻⁹ J | 9.93 × 10⁻⁸ J |

The clean-history prediction had lower spin RMSE at all 60 later reports and lower absolute rotational-energy error at 59 of 60. At 70 ms, spin RMSE was 0.467 rad/s with clean history and 23.83 rad/s in the autonomous rollout. The first prediction shows an immediate error even with clean inputs. The later paired results indicate that accumulated rollout error is the larger contributor over this measured window. The diagnostic does not isolate a particular contact mechanism or extend past 70 ms.

## Run records

The 50 ms result comes from `experiments/summaries/rollout_finetune4_seed44_50ms_bulk_v1.txt`. The 0.1 s, 0.5 s, 1 s, and 70 ms paired results come from the console logs supplied in the project chat.

Matched four- vs. eight-prediction fine-tuning results

Both runs initialized from `processor_data_42_best.pt` and used the same 997 seed-42 training examples, seed-43 validation indices, and 4,985 optimizer updates over five epochs. Three saved examples were excluded from both runs because the dense training trajectory ends before their eighth target. Both selected epoch 4 and completed all five 50-prediction validation windows.

| Seed-43 validation metric | Four predictions | Eight predictions |
|---|---:|---:|
| Selected rollout score ↓ | 7.2890 | 6.8376 |
| One-step loss at selected epoch ↓ | 0.3463 | 0.3754 |

The eight-prediction candidate lowered the selected rollout score by 6.2% and raised the one-step loss by 8.4%.

Both selected checkpoints completed the seed-44 70 ms diagnostic under the same fixed guard. Medians use the same 60 reports from 11–70 ms.

| Seed-44 metric | Matched four | Matched eight |
|---|---:|---:|
| Median clean-history spin RMSE (rad/s) | 0.610 | 0.631 |
| Median autonomous spin RMSE (rad/s) | 23.89 | 22.43 |
| Median autonomous position RMSE (µm) | 175.84 | 166.39 |
| Median autonomous absolute rotational-energy error (J) | 1.60 × 10⁻⁷ | 5.36 × 10⁻⁸ |
| Median joint group-density TV, 4 / 5 mm | 0.070 / 0.059 | 0.065 / 0.056 |
| Endpoint autonomous spin RMSE (rad/s) | 24.85 | 25.39 |
| Endpoint joint group-density TV, 4 / 5 mm | 0.102 / 0.081 | 0.121 / 0.093 |
| Model peak particle / wall overlap (µm) | 52.41 / 34.71 | 52.41 / 34.67 |

Eight-prediction training reduced median autonomous spin RMSE by 6.1% and median autonomous absolute rotational-energy error by 66.6%. The clean-history spin median rose by 3.4%, and the endpoint spin and group-density metrics were worse. Advance the eight-prediction model as a candidate for the matched 0.1 s and 0.5 s development rollouts. Keep the four-prediction checkpoint as the control. Use the longer curves to judge whether the median improvements persist and how often the candidate worsens the bulk metrics.

Training logs are `training_rollout4_matched997_v1.txt` and `training_rollout8_matched997_v1.txt`. Paired diagnostic logs are `seed44_paired_rollout4_matched997.txt` and `seed44_paired_rollout8_matched997.txt` in `experiments/summaries/`.
