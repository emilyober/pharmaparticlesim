#include "cuda/device_particles.h"
#include "cuda/integrate.h"
#include "cuda/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void checkScalar(double actual, double expected){
    const double tolerance = 1e-12+1e-12*std::abs(expected);
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error("CPU/GPU integration mismatch");
    }
}

void checkVector(const Vec3D& actual, const Vec3D& expected){
    checkScalar(actual.x, expected.x);
    checkScalar(actual.y, expected.y);
    checkScalar(actual.z, expected.z);
}

int main() {
    try {
        constexpr double dt = 5e-6;
        constexpr double pi = 3.14159265358979323;
        for (std::size_t count : {0u,1u,257u,332u}){
            std::vector<Particle> initial(count);
            for (std::size_t i = 0; i < count; ++i){
                Particle& p = initial[i];
                p.radius = 0.001 + 1e-6 * (i % 5);
                p.mass = 1500.0 * (4.0 / 3.0) * pi * p.radius * p.radius * p.radius;
                p.inertia = 0.4 * p.mass * p.radius * p.radius;
                p.position = {0.003 * i, 0.001, -0.002};
                p.velocity = {0.01, -0.02, 0.03};
                p.angularVelocity = {1.0, -2.0, 3.0};
                p.force = {0.2 * p.mass, -0.1 * p.mass, -9.81 * p.mass};
                p.torque = {p.inertia, -2.0 * p.inertia, 0.5 * p.inertia};
            }
            auto expected = initial;
            for (Particle& p : expected){
                p.velocity += (p.force / p.mass) * dt;
                p.angularVelocity += (p.torque / p.inertia) * dt;
                p.position += p.velocity * dt;
            }
            DeviceParticles device(initial);
            launchIntegrate(device.view(),dt);
            checkCuda(cudaStreamSynchronize(nullptr), "wait for integration test");
            auto actual = initial;
            device.download(actual);
            for (std::size_t i = 0; i < count; ++i){
                checkVector(actual[i].position,expected[i].position);
                checkVector(actual[i].velocity,expected[i].velocity);
                checkVector(actual[i].angularVelocity, expected[i].angularVelocity);
                checkVector(actual[i].force, initial[i].force);
                checkVector(actual[i].torque, initial[i].torque);
                checkScalar(actual[i].radius, initial[i].radius);
                checkScalar(actual[i].mass, initial[i].mass);
                checkScalar(actual[i].inertia, initial[i].inertia);
            }
            std::cout << "passed: " << count << " particles\n";
        }
    } catch (const std::exception& error){
        std::cerr << error.what() << '\n';
        return 1;
    }
}