# CPU benchmark: early gating and selected ghost construction

**Date:** 2026-10-28  
**Result:** 1.264x measured CPU speedup; 20.9% less elapsed compute time for the tested window.

## Setup

- Frozen checkpoint: `experiments/ml_runs/processor_surface_slip_gated_best.pt`, epoch 15, `processor_M1_surface_slip_v1`.
- Dataset: `pilot_seed44_10rpm_0p1s_dense.csv`; 322 particles, radius 1 mm; drum radius/length 20 mm, 10 RPM.
- Window: DEM frames 6000–8000, physical time 0.03–0.04 s; duration 0.01 s.
- Classical: Release C++ DEM, 2,000 steps at 5e-6 s; serial CPU implementation. Simulation before frame 6000 retains actual DEM contact history.
- ML: 200 predictions at 5e-5 s (horizon 10), one processing round, one PyTorch CPU thread, inference mode. Each repetition resets to recorded current/history states at frames 6000/5990.
- ML includes neighbor search, gates, ghost construction, neural calculations, aggregation, integration, and initial state clones. Dataset/checkpoint loading and diagnostic calculations are excluded.
- DEM uses an internal timer from immediately before the first update after frame 6000 through integration at frame 8000. CSV writes during this interval are suppressed. Process startup, earlier simulation steps, and later output are excluded.
- Both methods warmed up once; five serial pairs, DEM first then ML. Profiler runs occur after measurements and are excluded from the medians.

## Paired measurements

| Pair | DEM (s) | ML (s) |
|---|---:|---:|
| 1 | 0.195164 | 0.152162 |
| 2 | 0.192131 | 0.153575 |
| 3 | 0.193473 | 0.151389 |
| 4 | 0.192329 | 0.152817 |
| 5 | 0.191587 | 0.150131 |

Unrounded medians: DEM **0.192329458 s**; ML **0.1521623750013532 s**. Ratio `median_DEM / median_ML = 1.2639751318175048`. ML was faster in all five pairs. Printed pair values above are rounded to six decimal places.

## Optimizations

1. **Early gate:** restrict to one-round evaluation with gradients disabled. Filter inactive particle/wall edges before neural processing; retain existing late masks. Training and multiple-round execution use the original path.
2. **Selected ghost construction:** select nearby particle rows before calling `ghost_state` for current and previous states. Wall selection, ordering, geometry, and motion rules are unchanged.

Prior standalone ML medians were 230.34 ms before early gating, 159.20 ms with early gating, and 149.69 ms with both optimizations. These were separate sessions; use the paired result above for the primary CPU comparison.

## Accuracy checks

Previously reported seed-44 endpoint component RMSE after both optimizations: position **6.892214269e-5 m** (68.922 µm), velocity **0.01375369408 m/s**, spin **9.717950009 rad/s**. Selected ghost construction preserved those reported values. Early filtering produced small floating-point trajectory differences relative to late-only gating; it is not a bitwise-equivalence claim.

The paired timing output did not repeat the final RMSE, so those accuracy values refer to the preceding verified run of the same optimized path. Five-window in-memory early-gate checks retained full-window v2 screening passes; full-window reference thresholds are provisional physical screens.

## Interpretation and limits

This compares an approximate learned trajectory with classical DEM over equal simulated duration, not equal numerical accuracy. It establishes a modest speed advantage for this CPU implementation/window only. It does not establish GPU/CUDA speedup, particle-count scaling, long-term mixing accuracy, or general performance across hardware. Seed 44 was held out from training and checkpoint selection but is now being used for development measurements.

Next comparisons: verify a physics-matched GPU DEM baseline, then measure CPU/GPU implementations across particle counts and prediction horizons at documented accuracy tolerances. Include neighbor search and required transfers; synchronize GPU timing boundaries; exclude file I/O consistently. Preserve this horizon-10 model as the baseline before integration or contact-history changes.

## Provenance

Raw timing values were supplied in the chat; no standalone paired-output file was available to hash. Current source hashes below were captured retrospectively after the measurements, not automatically at execution. Git HEAD alone is insufficient because relevant changes include untracked/modified files.

- Checkpoint SHA-256: `628743c21d5a44ad66ab14bdfc1aa0fc29293b38a9bea8c8f27b22cb01a6a1d2`
- Git HEAD: `6c69ff1443a2be1eb8ec979f1a50ea7df88e5b65`

| Current file | SHA-256 |
|---|---|
| `ml/benchmark_processor.py` | `c9d9d9d67be6168adb5498ef10401054f280f555da47ee6e7c7bc6b905013a02` |
| `ml/processor_model.py` | `8b791e23f819b17b674d50ad307ed8e3f9a753f96819ffd48d95f0ad4ab6ece2` |
| `ml/graph.py` | `50b46be80c96f590f44f549366021e389fa506028b245d424c4b16b092686fe5` |
| `src/simulator.cpp` | `71c590fb05e39432851e193179b50742988bbdb9b18a29466d50bc8cf8a38132` |
| `src/main.cpp` | `616730c0c7e0b8b6c32c4176df3f24014d122a4f052790244f4effc9bc67db02` |
