#include "cuda/reset_forces.h"
#include "cuda/cuda_check.h"

__global__ void resetForcesKernel(DeviceParticleView p, double gx, double gy, double gz) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= p.count){
        return;
    }
    p.fx[i] = p.mass[i]*gx;
    p.fy[i] = p.mass[i] * gy;
    p.fz[i] = p.mass[i]*gz;
    p.tx[i] = 0.0;
    p.ty[i] = 0.0;
    p.tz[i] = 0.0;
}

void launchResetForces(DeviceParticleView particles, double gx, double gy, double gz, cudaStream_t stream){
    if (particles.count == 0){
        return;
    }
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((particles.count - 1)/threads +1);
    resetForcesKernel<<<blocks,threads,0,stream>>>(particles,gx,gy,gz);
    checkCuda(cudaGetLastError(),"launch force reset kernel");
}