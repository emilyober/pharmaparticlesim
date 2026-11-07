# Initial ML pilot experiments

## Setup

- Dataset: pilot_10rpm.csv
- Particles: 322
- Duration: 0–0.1 seconds
- DEM timestep: 5e-6 seconds
- Prediction horizon: 10 steps / 50 microseconds
- Split: chronological 80% training / 20% validation
- Training examples whose targets cross the split boundary excluded
- Batch size: 256
- Optimizer: Adam, learning rate 1e-3
- Training duration: 5 epochs
- Normalization statistics calculated from training data only
- Neighborhood radius: 0.0025 m
- Neighbor features: relative position and relative velocity
- Model: shared neighbor encoder, sum pooling, particle-state predictor
- No explicit wall features or contact-history inputs
- Full-training run seeds and Git commits: not recorded
- All losses below are normalized mean squared errors

## Data checks

Examples: 6437102
Inputs: [256, 9]
Neighbors: [256, M, 6]
Mask: [256, M]
Targets: [256, 9]
Contact flags: [256], bool
Predictions: [256, 9]
Initial input and target finiteness checks passed.

Neighbor counts:
Frame 0:     min=3, mean=5.02, max=6
Frame 10000: min=1, mean=5.15, max=8
Frame 20000: min=2, mean=5.71, max=8

## Implementation corrections

- Fixed future velocity target: it incorrectly used current velocity,
  making all velocity-change targets zero.
- Added missing neighbor normalization during validation.
- Changed horizon validation from `and` to `or`.

Results before the target correction are excluded from comparisons.

## Run 1: neighborhood model after corrections

Epoch 1: train=0.229965, val=0.015314
Epoch 2: train=0.132260, val=0.018840
Epoch 3: train=0.105484, val=0.017895
Epoch 4: train=0.092429, val=0.016030
Epoch 5: train=0.085589, val=0.017862

Constant-velocity validation loss: 0.011917

## Run 2: component-error diagnostics

Component order: position / velocity / spin

Epoch 1: train=0.221554, val=0.014586
Components: [0.0007, 0.0319, 0.0112]

Epoch 2: train=0.125711, val=0.015783
Components: [0.0011, 0.0322, 0.0140]

Epoch 3: train=0.100576, val=0.015977
Components: [0.0010, 0.0331, 0.0138]

Epoch 4: train=0.088615, val=0.017508
Components: [0.0011, 0.0336, 0.0178]

Epoch 5: train=0.081878, val=0.019422
Components: [0.0012, 0.0370, 0.0201]

Constant-velocity validation loss: 0.011917
Components: [3.3397e-05, 2.6852e-02, 8.8659e-03]

## Run 3: contact-group diagnostics

Contact means particle-particle center distance below 0.002 m
at the input frame. It excludes wall contacts and does not identify
all contacts occurring during the prediction interval.

Validation counts:
- No particle contact: 11347
- Particle contact: 1276331

Epoch 1: train=0.226552, val=0.017250
Components: [0.0012, 0.0395, 0.0111]
No particle contact: model=0.036557, baseline=0.026146
Particle contact:    model=0.017079, baseline=0.011791

Epoch 2: train=0.127628, val=0.025972
Components: [0.0011, 0.0608, 0.0160]
No particle contact: model=0.060024, baseline=0.026146
Particle contact:    model=0.025669, baseline=0.011791

Epoch 3: train=0.101327, val=0.033455
Components: [0.0010, 0.0814, 0.0180]
No particle contact: model=0.058172, baseline=0.026146
Particle contact:    model=0.033236, baseline=0.011791

Epoch 4: train=0.089800, val=0.042761
Components: [0.0012, 0.1042, 0.0230]
No particle contact: model=0.060852, baseline=0.026146
Particle contact:    model=0.042600, baseline=0.011791

Epoch 5: train=0.083078, val=0.038238
Components: [0.0014, 0.0945, 0.0188]
No particle contact: model=0.069535, baseline=0.026146
Particle contact:    model=0.037960, baseline=0.011791

Constant-velocity validation loss: 0.011917
Components: [3.3397e-05, 2.6852e-02, 8.8659e-03]

## Fixed-batch overfitting check

One fixed batch of 256 training examples.
Seed set to 42 before sampling the batch and constructing the model.
Same training-derived normalization as the full experiment.
Fresh NeighborhoodMLP; Adam, learning rate 1e-3.

Fixed-batch constant-velocity loss: 0.19643688201904297

Step 0:    loss=0.401507
Step 200:  loss=0.043422
Step 400:  loss=0.017676
Step 600:  loss=0.008954
Step 800:  loss=0.005677
Step 1000: loss=0.004491
Step 1200: loss=0.003431
Step 1400: loss=0.003286
Step 1600: loss=0.002932
Step 1800: loss=0.002320
Step 2000: loss=0.001980

## Findings and limitations

- The model can fit a small fixed sample.
- The current neighborhood model did not beat constant velocity
  on validation, including either contact group.
- About 99.1% of validation examples had particle contact.
- Contact presence does not measure collision intensity.
- Training loss decreased while validation generally deteriorated.
  The cause has not been established.
- The corrected particle-only MLP comparison remains outstanding.
- Validation comes from the same startup trajectory as training.
- No independent-run test, autonomous rollout, mixing-fidelity
  evaluation, or runtime/speedup benchmark has been completed.
- These results do not establish whether ML acceleration is viable.

Analytical-gravity residual experiment:
With unchanged datasets, subsets, architecture, training seed 123,
and a 20-epoch budget, best validation MSE was 0.637959 at epoch 18.
Constant-velocity baseline: 0.680956.
Gravity-only baseline: 0.683283.
Previous constant-velocity residual model: 0.636835.
Analytical gravity produced no clear aggregate prediction improvement.
Its effect on autonomous rollout penetration has not yet been tested.