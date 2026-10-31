#include <cuda_runtime.h>
#include "cuda/particle_view.h"
#include <cmath>
#include <stdexcept>
#include "cuda/cuda_check.h"
#include "cuda/integrate.h"

__global__ void integrateKernel(
    DeviceParticleView particles,
    double dt
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= particles.count){
        return;
    }
    particles.vx[i] += (particles.fx[i] / particles.mass[i]) * dt;
    particles.vy[i] += (particles.fy[i] / particles.mass[i]) * dt;
    particles.vz[i] += (particles.fz[i] / particles.mass[i]) * dt;
    particles.wx[i] += (particles.tx[i] / particles.inertia[i]) * dt;
    particles.wy[i] += (particles.ty[i] / particles.inertia[i]) * dt;
    particles.wz[i] += (particles.tz[i] / particles.inertia[i]) * dt;
    particles.x[i] += particles.vx[i] * dt;
    particles.y[i] += particles.vy[i] * dt;
    particles.z[i] += particles.vz[i] * dt;
}

void launchIntegrate(
    DeviceParticleView particles,
    double dt,
    cudaStream_t stream
) {
    if (particles.count == 0){
        return;
    }
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((particles.count-1)/threads+1);
    integrateKernel<<<blocks,threads,0,stream>>>(particles,dt);
    checkCuda(cudaGetLastError(),"launch integration kernel");
}