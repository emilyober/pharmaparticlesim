# Physics-structured granular GNN

## Research question

Can a learned model of local particle neighborhoods accelerate
rotating-drum DEM simulations by advancing multiple timesteps
at once, while preserving mixing dynamics compared with
classical GPU DEM?

## Motivation

The previous neighborhood MLP predicted independent particle-state
changes. It improved some short-rollout errors but accumulated
unphysical interpenetration.

In the 5 ms rollout starting at 0.03 s:

- Constant-velocity residual model:
  maximum particle overlap 658.05 µm; wall penetration 150.26 µm.
- Gravity-residual model:
  maximum particle overlap 665.73 µm; wall penetration 216.66 µm.
- DEM reference at the endpoint:
  maximum particle overlap 34.10 µm; wall penetration 6.11 µm.

Low average trajectory error did not guarantee acceptable contact
geometry. Adding analytical gravity did not resolve the failure.

## Literature reference

Sharma and Fink, DYNAMI-CAL GraphNet:
“A physics-informed graph neural network conserving linear and
angular momentum for dynamical systems.”

Published paper:
https://www.nature.com/articles/s41467-025-67802-5

Preprint containing the equations consulted during implementation:
https://arxiv.org/html/2501.07373v2

This implementation is inspired by DYNAMI-CAL. It is not currently
a faithful reproduction. Differences must be documented explicitly.

## Main architectural change

Previous model:
particle features + pooled neighbors → independent state changes

New model:
pair geometry and motion
→ shared pair encoding
→ linear/angular interaction impulses
→ consistent application to both particles
→ state integration

Positions will be calculated from updated velocities rather than
independently predicted.

## Initial scope

- Identical spherical particles.
- Dry Hertz–Mindlin DEM reference.
- Particle radius: 0.001 m.
- Drum radius: 0.02 m.
- Drum length: 0.02 m.
- Drum axis: +y.
- Rotation: 10 RPM.
- Gravity: (0, 0, -9.81) m/s².
- Rolling resistance disabled.
- Masses and moments of inertia supplied from physical parameters.
- No energy-conservation constraint: contacts dissipate energy,
  and moving walls supply work.

## Time conventions

DEM timestep: dt = 5e-6 s.
Initial prediction horizon: K = 10 DEM steps.
Prediction interval: H = K*dt = 50 microseconds.

Planned internal learned steps: M.
Internal interval: h = H/M.

M must be included in accuracy and runtime comparisons.
Skipping K DEM steps does not imply K-fold speedup.

## Data representation — frame_data.py

One example contains every particle together:

- previous: state at t-H
- current: state at t
- target: state at t+H

Each state has shape [N, 9]:
[x, y, z, vx, vy, vz, omega_x, omega_y, omega_z].

For the current pilot:
N = 322.
20,001 recorded frames give 19,981 examples when K = 10.

Initial data split:
- Packing seed 42: training.
- Packing seed 43: validation.
- Separate untouched test runs still required.

Color/group labels are for mixing analysis, not physical inputs.

## Graph representation — graph.py

Candidate particle pairs are selected by center distance.

Initial cutoff: 0.0025 m.
Touching distance for identical particles: 0.002 m.

The extra distance includes nearby noncontacting particles.
The cutoff is experimental and must be reconsidered for larger horizons.

build_particle_edges() returns both edge directions:
first the unique pairs, then their reverses.

Impulse aggregation uses each unique pair only once.

For the sampled first current frame:
- 1,616 directed edges.
- 808 unique pairs.

## Pair geometry

For sender i and receiver j:

r_ij = x_j - x_i
n_ij = r_ij / |r_ij|
overlap = R_i + R_j - |r_ij|

Positive overlap means contact deformation.
Negative overlap means separation.

Relative center velocity:
v_j - v_i.

Facing surface velocities include rotation:

v_surface_i = v_i + omega_i × (R_i*n_ij)
v_surface_j = v_j + omega_j × (-R_j*n_ij)

Relative surface velocity:
v_surface_j - v_surface_i.

It is decomposed into normal speed and tangential velocity.
Negative normal speed means approaching.

Full radii are used for surface velocities, matching the verified
solver convention. These are not automatically the torque lever arms.

## Local frames and motion projection

build_pair_frames() constructs a pair-local basis using geometry,
velocities, and spins, following the consulted preprint equations.

project_pair_motion() projects:
- Current velocity.
- Current spin.
- Previous velocity.
- Previous spin.

Each endpoint produces 12 projected scalar components.
The other endpoint uses the reversed frame.

Important limitation:
Zero-safe normalization avoids division by zero but may discard
directions in degenerate configurations. Stationary, head-on, and
sliding cases need architecture-level verification.

## PairEncoder — proposed implementation

Shared endpoint motion encoder:
12 → 64 → 64, using SiLU.

Geometry encoder:
2 → 64 → 64, using SiLU.

Initial geometry inputs:
- Scaled center distance.
- Scaled signed overlap.

Endpoint embeddings and geometry embedding are summed, then
processed by an interaction MLP:
64 → 64 → 64.

Sharing endpoint weights and summing their embeddings makes the
encoding invariant to exchanging correctly constructed endpoint inputs.

Material properties and boundary-node types are not yet encoded.

## PairDecoder — proposed implementation

A linear layer maps the 64-dimensional embedding to six coefficients:

- Three for linear impulse.
- Three for total angular impulse.

Coefficients are reconstructed as vectors using the pair-local basis.

Fixed scalar scales convert outputs into:
- Linear impulse: kg·m/s.
- Angular impulse: kg·m²/s.

Scale values remain to be specified.
They should not differ by Cartesian axis.

Output weights and biases are initialized to zero.

Shared application point:
c_ij = (x_i + x_j)/2.

Using a fixed midpoint is an adaptation from the paper's learned
application-point mechanism. It is the initial choice for identical spheres.

## Momentum-consistent aggregation

Convention: J_ij and A_ij apply to particle i.

Apply:
J_ji = -J_ij
A_ji = -A_ij

Spin angular impulses:

S_i = A_ij - (x_i-c_ij) × J_ij
S_j = -A_ij - (x_j-c_ij) × (-J_ij)

Aggregate contributions from every unique pair onto its two particles.

Equal-and-opposite linear impulses cancel internally.
The spin/orbital decomposition balances total angular impulse.

This algebra does not guarantee accurate forces, energy behavior,
or acceptable particle overlap.

## State integration

For each internal interval h:

v_new = v + g*h + net_linear_impulse/m
omega_new = omega + net_spin_impulse/I
x_new = x + v_new*h

Impulses already represent integrated interactions.
Do not multiply them by h again.

Use I = (2/5)*m*R² for solid spheres.

Semi-implicit Euler is an intentional adaptation matching the
classical solver's integration rule. The consulted preprint uses
an averaged-endpoint-velocity position update.

## Walls — planned, not implemented

Use prescribed boundary ghost nodes based on reflected geometry.

For nearest wall point q:
x_ghost = 2*q - x_particle.

Wall surface velocity:
v_wall(q) = Omega × (q-axis_point).

Ghost state conventions must be checked against the published method.
Ghosts are prescribed boundaries, not freely integrated particles.

Particle-only momentum is not conserved when gravity or walls act.

## Contact history and edge memory

These are distinct:

1. Previous observed velocity/spin:
   provided by FrameDataset.

2. Latent edge memory across internal message-passing steps:
   planned, not implemented.

3. Explicit Hertz–Mindlin tangential displacement history:
   not present in current CSVs; possible later experimental variant.

Persistent memory between prediction intervals requires explicit rules
for identifying pairs and handling contact formation/separation.

Do not claim novelty merely from adding temporal memory:
DYNAMI-CAL already includes temporal information and edge embeddings.

## Checks completed

Graph geometry:
- Moving-frame tangential projection error: 2.8034e-8 m/s.
- Maximum tangential speed in that check: 0.3005 m/s.
- Normal reversal error: 0.
- Relative surface-velocity reversal error: 0.

Synthetic pair aggregation:
- Linear impulse balance error: 0.
- Angular impulse balance error: 0.

These checks do not validate the complete architecture or rollouts.

## Remaining implementation

- Wire encoder, decoder, aggregation, and integration together.
- Supply masses, inertias, and physical scales.
- Add boundary interactions.
- Add internal-step edge memory and define graph updates.
- Add normalization consistent with vector symmetries.
- Implement complete-frame training and checkpoint metadata.
- Verify full discrete momentum balance and rotation/permutation behavior.
- Test isolated collisions and prescribed wall motion.
- Train and repeat existing rollout comparisons.

No training or rollout results yet exist for the assembled new architecture.

## Evaluation plan

Compare against:
- Constant velocity.
- Gravity-only motion.
- Previous residual neighborhood MLP.
- Classical CPU DEM.
- Classical GPU DEM for acceleration claims.

Measure:
- Position, velocity, and spin errors.
- Particle overlap and wall penetration.
- Energy and momentum behavior with external forcing accounted for.
- Autonomous rollout stability.
- Mixing statistics over meaningful durations.
- End-to-end runtime including graph construction and internal steps.

Future controlled experiments:
- Prediction horizon K.
- Internal step count M.
- Contact-memory representation.
- Particle count and operating conditions.

Novelty remains provisional until compared with closely related work.

## First structured-model experiment

### Configuration

- Architecture: structured particle-pair and wall interactions.
- One learned internal step, M = 1.
- Prediction horizon: 10 DEM steps = 50 µs.
- Packing seed 42 for training; seed 43 for validation.
- Fixed samples: 1,000 training frames and 200 validation frames.
- Model-training seed: 123.
- Adam, learning rate 1e-3.
- Ten training epochs.
- Complete-frame examples include previous/current/future states.
- Known masses/inertias and analytical gravity.
- Semi-implicit Euler position integration.
- Separate particle and wall encoders/decoders.
- Fixed midpoint application points.
- No latent edge memory or multi-step rollout training.

### Validation results

Best checkpoint: epoch 10.

Normalized validation MSE:
- Model: 0.196748.
- Constant velocity: 0.661476.
- Gravity only: 0.668205.

Model component MSE, position / velocity / spin:
[0.000644378, 0.246578917, 0.343021005]

Constant-velocity component MSE:
[0.000725638, 1.029842734, 0.953858852]

Overall MSE was approximately 70.3% below constant velocity.

These metrics evaluate predictions from true DEM inputs.
They do not establish autonomous rollout stability.
Loss scaling and frame sampling differ from the earlier MLP runs.

### Autonomous rollout

Checkpoint: structured_M1_best.pt, epoch 10.
Reference: seed-43 trajectory.
Start: DEM step 6000, time 0.03 s.
Initialized with true current and previous states.
Subsequent inputs were model predictions.
Graph and wall ghosts rebuilt each prediction.
Requested duration: 100 predictions = 5 ms.

Prediction 1, elapsed 0.05 ms:
- Position RMSE: 1.2509e-7 m.
- Velocity RMSE: 2.8815e-3 m/s.
- Spin RMSE: 1.3755 rad/s.
- Maximum particle overlap: 48.51 µm.
- Maximum wall penetration: 20.88 µm.

Prediction 10, elapsed 0.5 ms:
- Position RMSE: 9.7543e-6 m.
- Velocity RMSE: 0.037903 m/s.
- Spin RMSE: 21.796 rad/s.
- Maximum particle overlap: 124.70 µm.
- Maximum wall penetration: 36.60 µm.
- DEM overlaps: particle 36.52 µm; wall 25.05 µm.

Prediction 30, elapsed 1.5 ms:
- Position RMSE: 9.8651e-5 m.
- Velocity RMSE: 0.37604 m/s.
- Spin RMSE: 227.97 rad/s.
- Maximum particle overlap: 552.62 µm.
- Maximum wall penetration: 1263.30 µm.

Prediction 50, elapsed 2.5 ms:
- Position RMSE: 8.4530e-4 m.
- Velocity RMSE: 2.0128 m/s.
- Spin RMSE: 1855.2 rad/s.
- Maximum particle overlap: 1742.29 µm.
- Maximum wall penetration: 4167.91 µm.

Prediction 80, elapsed 4 ms:
- Position RMSE: 0.0046328 m.
- Velocity RMSE: 6.4369 m/s.
- Spin RMSE: 20723 rad/s.
- Maximum particle overlap: 1519.32 µm.
- Maximum wall penetration: 34823.64 µm.

Stopped at prediction 89, elapsed 4.45 ms:
model state became nonfinite.

### Conclusion

The model substantially improved isolated prediction loss but became
unstable during autonomous rollout. It did not maintain confinement
or acceptable contact geometry.

Internal impulse balance is insufficient to guarantee accurate,
dissipative, or stable learned dynamics.

This experiment evaluates a simplified DYNAMI-CAL-inspired adaptation,
not a faithful reproduction of the published method.