#pragma once
#include <cuda_runtime.h>
#include "cuda/particle_view.h"
#include "contact_state.h"
#include "material.h"

void launchPairContact(
    DeviceParticleView particles,
    MaterialPairProperties properties,
    ContactState* deviceHistory,
    int* deviceInvalidGeometry,
    double dt,
    cudaStream_t stream = nullptr
);