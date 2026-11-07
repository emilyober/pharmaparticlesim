# Surface-slip GNN: gated rollout evaluation

**Date:** 2026-10-28  |  **Checkpoint:** epoch 15  |  **Protocol:** full-window reference v2

## Purpose

Re-evaluate the unchanged `processor_surface_slip_gated_best.pt` using DEM overlap peaks from every recorded timestep over the complete evaluation window. Preserve the original screening-limit crossings separately. No retraining or checkpoint reselection was performed.

## Configuration

- Architecture: `processor_M1_surface_slip_v1`; hidden width 64; one processing round; third vector channel follows tangential surface slip, including rotation. Channel conditioning floor 0.1; length floor 1e-6 m.
- Interaction gate enabled during training, selection, and evaluation. Gate margin 1e-6 m; particle center-distance cutoff 2.5 mm; wall candidate gap margin 0.5 mm.
- DEM timestep 5e-6 s; prediction horizon 10 DEM steps = 5e-5 s. Two-step training; 1,000 selected training frames from seed 42; 200 validation frames from seed 43. Exact index arrays are in the JSON record.
- Adam: learning rate 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.
- Drum: radius 20 mm, length 20 mm, y-axis, 10 RPM; particle density 1,500 kg/m³; gravity (0, 0, -9.81) m/s². Semi-implicit integration; no persistent contact-history state.
- Input normalization: velocity 0.2981600 m/s; spin 212.9692 rad/s. Loss scales per component: position 2.425068e-6 m, velocity 0.002276635 m/s, spin 1.281399 rad/s.
- Particle and wall decoder scales: linear impulse 1.430453e-8 kg m/s; angular impulse 3.220507e-12 kg m²/s.

The checkpoint was originally selected with 50-prediction endpoint-sampled screening: one-step validation loss 0.313397; screened rollout score 7.908910. These are original-protocol metrics, not v2 metrics.

## Evaluation protocol

- Five validation starts: 2000, 6000, 10000, 14000, 18000; 200 autoregressive predictions per window = 10 ms = 2,000 DEM steps. Initialize current and previous states from DEM; no subsequent DEM corrections.
- Reference peaks include all 2,001 recorded DEM states per window. Separate particle/wall screening limits are `max(2 × full-window DEM peak, 1 µm)`. These remain provisional tolerances.
- Measure model overlap at prediction endpoints. Stop on nonfinite states/errors or a v2 limit crossing. Record the first original-limit crossing without stopping there.
- Save position/velocity/spin component RMSE against DEM for the model, constant velocity, and gravity-only baselines at every completed prediction.

## Results

**All five windows completed 200 predictions and passed v2 screening.** Endpoint errors:

| Start | Position RMSE (µm) | Velocity RMSE (m/s) | Spin RMSE (rad/s) | First original-limit crossing |
|---|---:|---:|---:|---|
| 2000 | 14.32 | 0.01375 | 13.77 | 55 |
| 6000 | 68.41 | 0.01351 | 9.46 | None within 200 |
| 10000 | 40.23 | 0.01176 | 11.78 | None within 200 |
| 14000 | 44.14 | 0.01317 | 13.91 | None within 200 |
| 18000 | 45.16 | 0.01103 | 11.87 | None within 200 |

Window 2000 explains the earlier rejection: the frozen original wall limit was 1 µm, but full-window DEM wall overlap reaches 20.64 µm. The model reaches 16.02 µm; particle peaks are 32.96 µm for DEM and 28.63 µm for the model. The original crossing at prediction 55 remains recorded.

At prediction 200, all five windows have lower position, velocity, and spin RMSE than both simple baselines. Passing revised screening is a different result from passing the original thresholds; it is not evidence of speedup or mixing fidelity.

## Provenance and artifacts

- Runtime: Python 3.13.15; PyTorch 2.14.1; NumPy 2.5.4; SciPy 1.18.1; macOS ARM64 CPU, two inference threads. State float32; overlap/error calculations float64.
- Captured Git HEAD: `6c69ff1443a2be1eb8ec979f1a50ea7df88e5b65`. The working tree included untracked/modified files; this commit alone does not identify the implementation.
- [Experiment record](experiment_record.json): configuration, optimizer settings, exact frame indices, dataset/source SHA-256 hashes, reference peaks, and protocol details.
- [Per-prediction results](evaluation_results_v2.json): all model/baseline errors, overlap traces, failures, and original-limit crossings.
- [Checkpoint and source snapshot](checkpoint_and_current_sources.zip): checkpoint plus eight current Python source files. The archived evaluation script retains the original reference implementation; v2 execution rules are documented in the JSON record.

SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| Epoch-15 checkpoint | `628743c21d5a44ad66ab14bdfc1aa0fc29293b38a9bea8c8f27b22cb01a6a1d2` |
| Seed-42 dense CSV | `71ea01135f968123e3e8f03b2706741a3ab18ae9d1f487f654d99c88f7e6dc79` |
| Seed-43 dense CSV | `6ac03da7b77cca0300e6716633252015b65c5990c260e59b8d317a7cda9fff62` |
| Captured `ml/processor_model.py` | `d293bdfb37e9daabb72e8a854554b6445980220733de3048156149d992916492` |

## Limits and next step

Sources were captured retrospectively, not at training time; exact historical training-code identity cannot be established. The five windows were used during development, not as an untouched test set. Gates are heuristic; persistent friction history is absent; intermediate model penetration between prediction endpoints is not measured. Mixing accuracy, longer-term stability, and wall-clock/GPU speedup remain untested.

Freeze this checkpoint and protocol, then evaluate an untouched seed before further architecture changes.
