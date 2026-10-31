#include "cuda/pair_contact.h"
#include "cuda/hertz_mindlin.cuh"
#include "cuda/cuda_check.h"
#include <cmath>
#include <stdexcept>

__global__ void pairContactKernel(
    DeviceParticleView p,
    MaterialPairProperties properties,
    ContactState* history,
    int* invalidGeometry,
    double dt
) {
    cuda_dem::ContactParticle first{
        {p.x[0], p.y[0], p.z[0]},
        {p.vx[0], p.vy[0], p.vz[0]},
        {p.wx[0], p.wy[0], p.wz[0]},
        p.radius[0],
        p.mass[0]
    };
    cuda_dem::ContactParticle second{
        {p.x[1], p.y[1], p.z[1]},
        {p.vx[1], p.vy[1], p.vz[1]},
        {p.wx[1], p.wy[1], p.wz[1]},
        p.radius[1],
        p.mass[1]
    };
    const auto result = cuda_dem::evaluateHertzMindlin(first, second, properties, *history, dt);
    *invalidGeometry = result.invalidGeometry ? 1 : 0;
    p.fx[0] -= result.forceOnSecond.x;
    p.fy[0] -= result.forceOnSecond.y;
    p.fz[0] -= result.forceOnSecond.z;
    p.fx[1] += result.forceOnSecond.x;
    p.fy[1] += result.forceOnSecond.y;
    p.fz[1] += result.forceOnSecond.z;
    p.tx[0] += result.torqueOnFirst.x;
    p.ty[0] += result.torqueOnFirst.y;
    p.tz[0] += result.torqueOnFirst.z;
    p.tx[1] += result.torqueOnSecond.x;
    p.ty[1] += result.torqueOnSecond.y;
    p.tz[1] += result.torqueOnSecond.z;
}

void launchPairContact(
    DeviceParticleView particles,
    MaterialPairProperties properties,
    ContactState* deviceHistory,
    int* deviceInvalidGeometry,
    double dt,
    cudaStream_t stream
) {
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::invalid_argument("Invalid timestep");
    }
    if (!deviceHistory || !deviceInvalidGeometry) {
        throw std::invalid_argument("Missing contact device storage");
    }
    if (particles.count != 2) {
        throw std::invalid_argument("Expected exactly two particles");
    }
    pairContactKernel<<<1, 1, 0, stream>>>(
        particles, properties, deviceHistory,
        deviceInvalidGeometry, dt
    );

    checkCuda(cudaGetLastError(), "launch pair contact kernel");
}