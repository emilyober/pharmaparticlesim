#include "cuda/device_particles.h"
#include "cuda/cuda_check.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

DeviceParticles::DeviceParticles(const std::vector<Particle>& particles){
    constexpr std::size_t fields = 18;
    const std::size_t count = particles.size();
    view_.count = count;
    if (count == 0){
        return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / (fields * sizeof(double))){
        throw std::length_error("Particle allocation is too large");
    }
    std::vector<double> packed(fields * count);
    for (std::size_t i = 0; i < count; ++i){
        const Particle& p = particles[i];
        if (!std::isfinite(p.mass) || p.mass <= 0.0 || !std::isfinite(p.inertia) || p.inertia <= 0.0) {
            throw std::invalid_argument("Particles need positive finite mass and inertia");
        }
        const std::array<double, fields> values{
            p.position.x, p.position.y, p.position.z,
            p.velocity.x, p.velocity.y, p.velocity.z,
            p.angularVelocity.x, p.angularVelocity.y,
            p.angularVelocity.z,
            p.force.x, p.force.y, p.force.z,
            p.torque.x, p.torque.y, p.torque.z,
            p.radius, p.mass, p.inertia
        };
        for (std::size_t field = 0; field < fields; ++field) {
            packed[field * count + i] = values[field];
        }
    }
    const std::size_t bytes = packed.size() * sizeof(double);
    void* allocation = nullptr;
    checkCuda(cudaMalloc(&allocation, bytes), "Allocate particle arrays");
    storage_ = static_cast<double*>(allocation);
    try {
        checkCuda(cudaMemcpy(storage_, packed.data(), bytes, cudaMemcpyHostToDevice),"Upload particle arrays");
    } catch (...) {
        cudaFree(storage_);
        storage_ = nullptr;
        throw;
    }
    std::size_t offset = 0;
    auto nextArray = [&](){
        double* pointer = storage_+offset;
        offset += count;
        return pointer;
    }
    view_.x = nextArray();
    view_.y = nextArray();
    view_.z = nextArray();

    view_.vx = nextArray();
    view_.vy = nextArray();
    view_.vz = nextArray();

    view_.wx = nextArray();
    view_.wy = nextArray();
    view_.wz = nextArray();

    view_.fx = nextArray();
    view_.fy = nextArray();
    view_.fz = nextArray();

    view_.tx = nextArray();
    view_.ty = nextArray();
    view_.tz = nextArray();

    view_.radius = nextArray();
    view_.mass = nextArray();
    view_.inertia = nextArray();
}

DeviceParticles::~DeviceParticles() noexcept {
    if (storage_ != nullptr){
        cudaFree(storage_);
    }
}

void DeviceParticles::download(std::vector<Particle>& particles) const {
    const std::size_t count = view_.count;
    if (particles.size() != count){
        throw std::invalid_argument("Download particle count mismatch");
    }
    if (count == 0){
        return;
    }
    std::vector<double> packed(18*count);
    checkCuda(cudaMemcpy(packed.data(),storage_,packed.size()*sizeof(double),cudaMemcpyDeviceToHost),"download particle arrays");
    for (std::size_t i = 0; i < count; ++i){
        auto value = [&](std::size_t field){
            return packed[field*count+i];
        };
        Particle& p = particles[i];
        p.position = {value(0), value(1), value(2)};
        p.velocity = {value(3), value(4), value(5)};
        p.angularVelocity = {value(6), value(7), value(8)};
        p.force = {value(9), value(10), value(11)};
        p.torque = {value(12), value(13), value(14)};
        p.radius = value(15);
        p.mass = value(16);
        p.inertia = value(17);
    }
}