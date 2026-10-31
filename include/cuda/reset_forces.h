#pragma once
#include <cuda_runtime.h>
#include "cuda/particle_view.h"

void launchResetForces(DeviceParticleView particles, double gx, double gy, double gz, cudaStream_t stream = nullptr);
