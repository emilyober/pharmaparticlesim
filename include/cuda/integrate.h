#pragma once
#include <cuda_runtime.h>
#include "cuda/particle_view.h"

void launchIntegrate(
    DeviceParticleView particles,
    double dt,
    cudaStream_t stream = nullptr
);