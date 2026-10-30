# PharmaParticleSim

A research project exploring learned acceleration of granular particle mixing in a rotating drum.

**Research question:** Can a learned model of local particle neighborhoods advance multiple DEM timesteps at once while preserving mixing dynamics compared with classical GPU DEM?

## Implementations

- **CPU DEM:** C++ Hertz–Mindlin contact model with friction history, damping, particle rotation, and rotating-drum walls.
- **ML surrogate:** PyTorch graph-based interaction model with surface-slip features, pairwise momentum constraints, and interaction gating.
- **CUDA DEM:** In development, targeting the same contact physics as the CPU solver.

The current ML model advances 10 DEM timesteps per prediction. Short rollout and CPU timing results are recorded under `experiments/summaries/`. Long-term mixing fidelity and GPU speedup have not yet been implemented.

## Build and run CPU DEM

Requires CMake and a C++17 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/pharmaparticlesim
```

The current experiment is configured in `src/main.cpp`. Material and contact parameters are defined in `materials.csv` and `contacts.csv`.

## Optional CUDA verification

Requires an NVIDIA GPU, CUDA toolkit, and CMake 3.18 or newer.

```bash
cmake -S . -B build-cuda -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target cuda_integration_check
ctest --test-dir build-cuda --output-on-failure
```

This checks integration; it does not validate a complete GPU DEM solver.

## Repository layout

- `src/`, `include/`: classical solver and CUDA implementation.
- `ml/`: datasets, models, training, evaluation, and benchmarks.
- `verification/`: numerical and contact-model checks.
- `benchmarks/reference/lammps/`: reference collision cases.
- `experiments/summaries/`: experiment logs and limitations.

Generated datasets and model checkpoints are stored separately from source code.

## Validation scope

Selected CPU collision cases have been compared with LAMMPS. ML evaluation includes autoregressive trajectory errors and overlap screening. These checks do not yet establish experimental pharmaceutical mixing accuracy.