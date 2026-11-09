# PharmaParticleSim accuracy–runtime comparison

Seed-44 CPU development benchmark · November 2026

## Historical workload, measured again

The workload used 322 particles at 10 RPM, from physical time 0.030 to 0.040 s. The Release C++ DEM ran 2,000 updates at 5 µs. Each ML checkpoint ran 200 predictions at 50 µs, from dense DEM history and state at steps 5990 and 6000. One warmup and five timed runs per method were executed serially on the same Darwin arm64 host. ML used one PyTorch CPU thread, early gating, and inference mode. Timings exclude loading, trajectory metrics, and CSV output during the timed window. The historical ML checkpoint used its original integration settings. The current checkpoints used their saved gravity correction and contact-position settings.

| Method | Median CPU time | Range | DEM / ML | Endpoint position RMSE | Velocity RMSE | Spin RMSE |
|---|---:|---:|---:|---:|---:|---:|
| C++ DEM, 5 µs | 203.94 ms | 200.09–214.46 ms | — | Reference | Reference | Reference |
| Historical gated ML | 160.84 ms | 157.73–178.68 ms | 1.268× | 68.92 µm | 0.01375 m/s | 9.72 rad/s |
| Clean swap-aware four | 163.97 ms | 160.38–181.86 ms | 1.244× | 66.55 µm | 0.01323 m/s | 10.34 rad/s |
| Clean swap-aware eight | 164.82 ms | 159.52–167.33 ms | **1.237×** | 61.49 µm | 0.01312 m/s | 10.20 rad/s |

The C++ executable came from the committed benchmark scenario on `ml-rollout-finetune`. The historical checkpoint reproduced the previously reported endpoint RMSE. The current eight-prediction checkpoint used 19.2% less compute time than 5 µs DEM in this 10 ms workload. This comparison matches simulated time. Mixing accuracy is evaluated separately below.

## Broader training and longer validation

Seed-42, 45, and 46 dense pilots are available through 0.1 s. They vary initial positional jitter at the same RPM, materials, and particle count. A previous 1,000-example experiment redistributed its fixed budget across these seeds and did worse under the old architecture. That result leaves open what happens when the original seed-42 examples are retained and new data are added.

The current clean checkpoints were re-evaluated on four seed-43 windows of 400 predictions (20 ms), starting at DEM steps 2000, 6000, 10000, and 14000. Both completed every window using per-window dense DEM overlap limits. Lower normalized rollout scores are better.

| Start step | Swap-aware four | Swap-aware eight |
|---|---:|---:|
| 2000 | 99.43 | 94.84 |
| 6000 | 310.86 | 264.98 |
| 10000 | 156.98 | 140.63 |
| 14000 | 209.80 | 196.10 |
| Mean | 194.27 | 174.14 |

Use these windows in both conditions of the next clean training experiment. Keep the swap-aware eight architecture and 50 µs inference interval fixed. Retain the 997 seed-42 examples and add 997 valid examples from each of seeds 45 and 46, sampled reproducibly within each trajectory before applying concatenation offsets. For an update-matched seed-42 control, repeat its existing 997 indices three times per epoch. Use the same initialization, optimizer schedule, shuffle seed, and number of updates in both conditions. These pilots broaden initial jitter coverage. Longer dense windows at later mixing stages would broaden the range of contact histories.

For the later learned-horizon sweep, train distinct 25, 50, and 100 µs checkpoints. Use history and targets at 5, 10, and 20 fine DEM steps, respectively, and set the model's physical timestep to match. Use the same physical training start times and a 0.4 ms training window (16, 8, or 4 predictions), plus the same 20 ms validation windows (800, 400, or 200 predictions). Keep wall and material settings fixed. Preserve fine DEM history at each required initial offset. These checkpoints still need to be trained and evaluated.

## 0.5 s mixing-error and runtime sweep

![CPU runtime versus mean absolute mixing-score error for the seed-44 pilot](accuracy-runtime-mixing.png)

All methods simulated the seed-44 drum over physical time 0.030–0.530 s. Each classical run started from the same seeded particle initialization at time zero and warmed up with its own timestep. The learned runs used the fine reference states at 0.02995 and 0.030 s as history and initial state. Timing excludes warmup, loading, output, and metric calculation. The C++ sweep used an isolated Release build with timestep and timing controls. The contact and integration equations were kept the same. The 5 µs sweep state at 0.030 s matched the prior dense pilot within 7.5 × 10⁻⁹ m/s in velocity and 1.91 × 10⁻⁶ rad/s in spin, with equal stored positions.

The agreed provisional tolerance is **mean absolute mixing-score error ≤ 0.05**. The comparison uses the existing 4 mm concentration-variance score, counting cells with at least five particles, over 500 shared reports spaced 1 ms apart. The 1.25 µs DEM curve is the comparison reference. Its mixing score ranged from approximately 0 to 0.239, with 69.7% mean coverage. Mean joint group-density TV includes all particles and is reported as a secondary metric at both bin widths.

| Method | CPU time (s) | Timed trials | Mixing MAE | Endpoint mixing error | Mean coverage | Mean group-density TV, 4 / 5 mm |
|---|---:|---:|---:|---:|---:|---:|
| DEM 20 µs | **3.282** | 3 | 0.0240 | 0.0525 | 69.8% | 0.112 / 0.077 |
| DEM 10 µs | 6.178 | 3 | 0.0247 | 0.0483 | 70.2% | 0.112 / 0.080 |
| DEM 8 µs | 7.623 | 1 | 0.0181 | 0.0286 | 71.0% | 0.117 / 0.082 |
| DEM 6.25 µs | 9.793 | 1 | 0.0196 | 0.0405 | 70.5% | 0.113 / 0.079 |
| DEM 5 µs | 12.379 | 3 | 0.0172 | 0.0147 | 71.0% | 0.110 / 0.088 |
| DEM 2.5 µs | 24.340 | 1 | 0.0184 | 0.0005 | 70.8% | 0.107 / 0.072 |
| DEM 1.25 µs reference | 48.663 | 1 | 0 by definition | 0 | 69.7% | 0 / 0 |
| ML swap-aware four, 50 µs | 11.994 | 2 | 0.0364 | 0.0719 | 72.2% | 0.170 / 0.125 |
| ML swap-aware eight, 50 µs | 11.936 | 2 | 0.0319 | 0.0488 | 71.2% | 0.165 / 0.121 |

Times are medians where repeated and single observations where indicated. The ML timings used one CPU thread and one short warmup per checkpoint. Both learned rollouts completed 10,000 predictions under full-window limits of 108.53 µm particle and 58.44 µm wall overlap, based on peaks measured at each fine DEM state. Learned peaks were checked at prediction endpoints.

- **Error under the same runtime budget.** Within the 11.94 s budget of swap-aware eight, 8 µs DEM had the lowest measured classical mixing MAE, 0.0181 in 7.62 s versus 0.0319 for ML. This uses measured points and a shared budget without interpolation.
- **Runtime at the agreed error tolerance.** Both learned runs meet the 0.05 mean tolerance. The fastest measured DEM configuration that meets it is 20 µs, at 3.28 s. That is 3.64× faster than the fastest measured ML configuration at this tolerance. The 20 µs endpoint error is 0.0525, so it would not pass a separate endpoint tolerance of 0.05.
- **Reference sensitivity.** The two finest DEM curves differ by 0.0184 mean mixing-score error. Using the 2.5 µs reference still makes 20 µs the fastest passing configuration, and a cheaper classical run still has lower error than either ML model. The best measured classical point under the ML budget changes to 6.25 µs. The sweep is a single-seed development result, with a provisional reference and mean tolerance.

This benchmark answers the 0.5 s comparison for the current 50 µs learned horizon. A final mixing comparison needs repeat timing near the frontier, a timestep refinement check over the intended mixing duration, and frozen acceptance criteria before evaluating untouched trajectories. The 25 and 100 µs learned points remain pending training and evaluation.
