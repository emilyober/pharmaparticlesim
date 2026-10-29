# Multi-round processor experiment log

## Goal

Determine whether repeated graph interactions and internal edge
memory improve autonomous rollout stability while retaining the
potential to advance multiple DEM timesteps efficiently.

This remains a DYNAMI-CAL-inspired adaptation, not a reproduction.

## Architecture changes from M1

- Three internal interaction rounds, M = 3.
- Total prediction interval remains 50 µs:
  10 DEM timesteps of 5 µs each.
- Each internal round advances approximately 16.67 µs.
- Shared encoders and decoders across rounds.
- Updated particle states influence subsequent interaction rounds.
- Residual edge-memory updates with LayerNorm.
- Particle candidate graph fixed within each prediction interval.
- Wall geometry recalculated each round.
- Wall memory indexed by particle and wall identity.
- Memories reset at the beginning of each model prediction.
- Decoder impulses divided across the three internal rounds.
- Outward ghost placement retained.

The fixed candidate graph can miss newly relevant pairs.
Its margin must be assessed before increasing the prediction horizon.

## Training setup

- Training trajectory: packing seed 42.
- Validation trajectory: packing seed 43.
- Both cover 0–0.1 s of startup.
- Fixed samples: 1,000 training frames, 200 validation frames.
- Model-training seed: 123.
- Adam, learning rate 1e-3.
- Ten epochs.
- Two-step rollout training, without detaching predictions.
- Position, velocity, and spin losses use training-derived
  characteristic change scales.

## Rollout checkpoint screening

- Evaluated every five epochs.
- Starting frames: 2000, 6000, 10000, 14000, 18000.
- Requested length: 50 predictions = 2.5 ms.
- Initial current and previous states come from DEM.
- Subsequent states are model predictions.
- Nonfinite states/errors cause rejection.
- Particle and wall overlap limits are twice the maximum reference
  overlap in each window, with a 1 µm floor.
- These are provisional development limits, not validated physical
  acceptance criteria.
- Every window must pass for checkpoint eligibility.
- score=inf is a rejection marker, not a measured infinite error.

## Training results

| Epoch | Two-step training loss | One-step validation loss |
|---|---:|---:|
| 1 | 336.491782 | 1.053913 |
| 2 | 1.908457 | 0.782574 |
| 3 | 1.811201 | 0.769877 |
| 4 | 2.640982 | 0.787483 |
| 5 | 1712.228406 | 526.528748 |
| 6 | 4662.510774 | 214.233582 |
| 7 | 140.212035 | 28.902550 |
| 8 | 201.794757 | 19.444218 |
| 9 | 2324.693265 | 94.528969 |
| 10 | 64.250360 | 33.941788 |

One-step validation baselines:
- Constant velocity: 0.713220.
- Gravity only: 0.718741.

Lowest reported one-step validation loss: 0.769877 at epoch 3.
Even this was worse than both baselines.
Epochs 1–4 were not rollout-screened.

## Epoch 5 rollout screening

All windows failed by exceeding overlap limits.

| Start frame | Predictions completed / 50 | Peak particle overlap, µm | Peak wall penetration, µm |
|---|---:|---:|---:|
| 2000 | 13 | 0.00 | 36.38 |
| 6000 | 5 | 58.80 | 68.10 |
| 10000 | 3 | 25.76 | 33.78 |
| 14000 | 5 | 35.22 | 63.37 |
| 18000 | 4 | 37.37 | 48.11 |

eligible=False; score=inf.

## Epoch 10 rollout screening

All windows failed by exceeding overlap limits.

| Start frame | Predictions completed / 50 | Peak particle overlap, µm | Peak wall penetration, µm |
|---|---:|---:|---:|
| 2000 | 36 | 3.97 | 0.00 |
| 6000 | 12 | 114.45 | 30.40 |
| 10000 | 9 | 46.45 | 28.21 |
| 14000 | 13 | 41.81 | 48.41 |
| 18000 | 15 | 47.45 | 46.27 |

eligible=False; score=inf.

Peak overlaps above are measured only through early termination.
They are not results for completed 50-prediction rollouts.

No checkpoint qualified under the screening rule.

## Interpretation

- Training showed large spikes rather than a smooth overfitting trend.
- Isolated prediction quality was worse than the simple baselines.
- Three internal rounds and edge memory did not establish an
  improvement over M1.
- The free-fall window also failed screening at both evaluated epochs.
- These results concern this implementation and optimization setup;
  they do not establish failure of the published DYNAMI-CAL method.
- No computational speedup or acceptable mixing fidelity has been
  demonstrated.

## Unresolved causes

Possible contributors, not confirmed diagnoses:

- Large derivatives through near-degenerate local reference frames.
- Recurrent amplification across internal rounds and training steps.
- Changed optimization behavior from residual memory and LayerNorm.
- Unsuitable optimizer step size or occasional extreme gradients.
- Other implementation or scaling issues.

The experiment changed multiple architectural components together,
so it does not isolate which component caused the deterioration.

## Next experiment

Before another full training run:
1. Record loss and gradient norm for the first 100 optimizer updates.
2. Detect nonfinite gradients before applying updates.
3. Identify the first large spike and its input frame.
4. Test a targeted numerical or optimization change.
5. Resume longer training only after this behavior is understood.

Preserve M1 code, checkpoints, and logs for comparison.
Record Git commit and configuration with subsequent runs.

Confirmed first-update overshoot:
Both fixed examples had low initial loss. One Adam update at 1e-3
increased the training-example loss from 0.190695 to 163.800339.
Reducing the learning rate reduced the damage, but 1e-5 still
increased that loss to 0.201927. This identifies an optimization
step-size/sensitivity problem independent of multi-round recurrence.
The underlying source of that sensitivity remains unresolved.

First-update learning-rate sweep:
At 1e-6 and 1e-7, one Adam update reduced loss on both fixed examples.
At 1e-5 and above, it increased loss on the training example.
This supports first-update overshoot under the current scaling.
A learning rate of 1e-6 is a diagnostic starting point, not yet a
validated training setting.
