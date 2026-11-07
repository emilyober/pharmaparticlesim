# Physics-GNN experiment log

Architecture specification: ml_physics_gnn.md

## Experiment 1 — Structured M1 training

### Setup

- Training trajectory: packing seed 42.
- Validation trajectory: packing seed 43.
- Fixed samples: 1,000 complete training frames; 200 validation frames.
- Both trajectories cover 0–0.1 s of startup.
- Model-training seed: 123.
- Horizon: 10 DEM steps = 50 µs.
- One internal learned update, M = 1.
- Adam, learning rate 1e-3.
- Ten epochs, one frame per optimizer update.
- Previous/current states are inputs; future state is the target.
- Analytical gravity, known masses/inertias, semi-implicit integration.
- Separate particle-pair and wall networks.
- No internal edge memory or rollout training.
- Checkpoint selected using isolated prediction validation loss.

Loss uses training-derived RMS changes, with one shared scalar
scale for each vector quantity: position, velocity, and spin.
Values are not directly comparable with earlier MLP losses.

### Training history

| Epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 0.707623 | 0.529550 |
| 2 | 0.425373 | 0.325658 |
| 3 | 0.306143 | 0.295877 |
| 4 | 0.291552 | 0.236304 |
| 5 | 0.279590 | 0.223128 |
| 6 | 0.256544 | 0.218282 |
| 7 | 0.252600 | 0.228187 |
| 8 | 0.232801 | 0.241307 |
| 9 | 0.226734 | 0.209395 |
| 10 | 0.221299 | 0.196748 |

Best checkpoint: structured_M1_best.pt, epoch 10.

### Best-checkpoint validation

| Predictor | Overall normalized MSE |
|---|---:|
| Structured model | 0.196748 |
| Constant velocity | 0.661476 |
| Gravity only | 0.668205 |

Component MSE, position / velocity / spin:

- Model: [0.000644378, 0.246578917, 0.343021005]
- Constant velocity: [0.000725638, 1.029842734, 0.953858852]
- Gravity only: [0.000739833, 1.050016761, 0.953858852]

Overall model error was 70.3% lower than constant velocity.
Position error worsened between epochs 6 and 10 despite improving
overall loss. The position term contributed much less to the loss
than velocity and spin.

## Experiment 2 — Autonomous rollout, original wall geometry

### Setup

- Same epoch-10 checkpoint.
- Seed-43 validation trajectory.
- Start frame: 6000, time 0.03 s.
- Initial previous state: frame 5990.
- Subsequent states/history are predictions.
- Neighbors and wall ghosts rebuilt each prediction.
- Requested 100 predictions = 5 ms.
- Metrics are physical-unit component RMSE across particles/axes.

### Selected results

| Prediction | Elapsed ms | Position RMSE, m | Velocity RMSE, m/s | Spin RMSE, rad/s | Max particle overlap, µm | Max wall penetration, µm |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 1.2509e-7 | 0.0028815 | 1.3755 | 48.51 | 20.88 |
| 10 | 0.5 | 9.7543e-6 | 0.037903 | 21.796 | 124.70 | 36.60 |
| 20 | 1.0 | 2.7647e-5 | 0.055717 | 69.932 | 154.07 | 92.96 |
| 30 | 1.5 | 9.8651e-5 | 0.37604 | 227.97 | 552.62 | 1263.30 |
| 50 | 2.5 | 8.4530e-4 | 2.0128 | 1855.2 | 1742.29 | 4167.91 |
| 80 | 4.0 | 0.0046328 | 6.4369 | 20723 | 1519.32 | 34823.64 |

Stopped at prediction 89, elapsed 4.45 ms:
nonfinite model state.

At prediction 10, constant-velocity RMSE was:
[9.2045e-6 m, 0.035323 m/s, 14.195 rad/s].
The model was already worse in all three quantities.

Conclusion:
Better isolated prediction loss did not produce stable autonomous
dynamics. Severe interpenetration, loss of confinement, and growing
motion errors developed.

## Experiment 3 — Wall-placement ablation

### Change

Replaced signed reflection with placement that keeps the ghost
outward of its particle after the particle center crosses a wall.

Evaluated the same epoch-10 checkpoint without retraining.
All rollout settings remained unchanged.

### Selected results

| Prediction | Elapsed ms | Position RMSE, m | Velocity RMSE, m/s | Spin RMSE, rad/s | Max particle overlap, µm | Max wall penetration, µm |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 1.2509e-7 | 0.0028815 | 1.3755 | 48.51 | 20.88 |
| 10 | 0.5 | 9.7543e-6 | 0.037903 | 21.796 | 124.70 | 36.60 |
| 20 | 1.0 | 2.7647e-5 | 0.055717 | 69.932 | 154.07 | 92.96 |
| 30 | 1.5 | 9.8879e-5 | 0.37982 | 227.33 | 552.62 | 1280.23 |
| 50 | 2.5 | 0.0010527 | 2.4039 | 1921.2 | 1746.27 | 9915.21 |
| 80 | 4.0 | 0.011386 | 47.314 | 279500 | 1249.27 | 147014.16 |

Stopped at prediction 88, elapsed 4.4 ms:
nonfinite model state.

Results matched through prediction 20.
The geometry change alone did not stabilize this checkpoint.
Significant error developed before the placement rules diverged.
This was not a retraining experiment.

## Interpretation and limitations

- The model is a simplified DYNAMI-CAL-inspired adaptation.
- Internal momentum balance does not guarantee dissipative or
  geometrically acceptable dynamics.
- The cause of instability has not been established.
- Possible contributors include local-frame degeneracy, learned
  impulse growth, missing internal processing, integration choices,
  and training only from true input states.
- All experiments use the same validation packing for development.
- No untouched test result, sustained mixing result, or computational
  speedup measurement exists yet.
- Printed results have limited precision.
- Git commit identifiers were not recorded with these runs.

## Next planned change

Add autonomous rollout evaluation during training:
- Fixed validation starting frames.
- Initially 50 predictions / 2.5 ms.
- Evaluate every five epochs.
- Report trajectory error and penetration throughout the rollout.
- Mark nonfinite rollouts as failed.
- Separate best numerical candidate from physically acceptable model.
- Establish physical acceptance tolerances using DEM statistics.