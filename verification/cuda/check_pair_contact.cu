#include "cuda/device_particles.h"
#include "cuda/pair_contact.h"
#include "cuda/reset_forces.h"
#include "cuda/cuda_check.h"
#include "contact_model.h"
#include "material_database.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void checkScalar(double actual, double expected) {
    const double tolerance = 1e-12 + 1e-10 * std::abs(expected);

    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error("CPU/GPU contact mismatch");
    }
}

void checkVector(const Vec3D& actual, const Vec3D& expected) {
    checkScalar(actual.x, expected.x);
    checkScalar(actual.y, expected.y);
    checkScalar(actual.z, expected.z);
}

int main(int argc, char** argv) {
    ContactState* deviceHistory = nullptr;
    int* deviceInvalid = nullptr;
    try {
        if (argc != 2) {
            throw std::invalid_argument("provide the project root path");
        }
        const std::string root = argv[1];
        MaterialDatabase materials;
        materials.loadMaterials(root + "/experiments/configs/hertz_mindlin/materials.csv");
        materials.loadContactProperties(root + "/experiments/configs/hertz_mindlin/contacts.csv");
        constexpr double dt = 5e-6;
        constexpr double pi = 3.14159265358979323;
        std::vector<Particle> expected(2);
        for (auto& p : expected) {
            p.material = "Powder";
            p.radius = 0.001;
            p.mass = materials.getMaterial("Powder").density * (4.0 / 3.0) * pi
                     * p.radius * p.radius * p.radius;
            p.inertia = 0.4 * p.mass * p.radius * p.radius;
        }
        expected[0].position = {0.0, 0.0, 0.0};
        expected[1].position = {0.00199, 0.0, 0.0};
        expected[0].velocity = {0.0, 0.01, 0.0};
        expected[1].velocity = {0.0, -0.01, 0.0};
        DeviceParticles device(expected);
        auto actual = expected;
        ContactState cpuHistory{};
        const ContactState emptyHistory{};
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceHistory),sizeof(ContactState)),"allocate contact history");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInvalid),sizeof(int)),"allocate geometry flag");
        checkCuda(cudaMemcpy(deviceHistory, &emptyHistory, sizeof(ContactState), cudaMemcpyHostToDevice), "initialize contact history");
        const auto properties = materials.getPairProperties("Powder", "Powder");
        for (int step = 0; step < 100; ++step) {
            for (auto& p : expected) {
                p.force = {};
                p.torque = {};
            }
            ContactModel::resolveContact(expected[0], expected[1], materials, cpuHistory, dt);
            launchResetForces(device.view(), 0.0, 0.0, 0.0);
            launchPairContact(device.view(), properties, deviceHistory, deviceInvalid, dt);
            checkCuda(cudaStreamSynchronize(nullptr),"wait for contact check");
            device.download(actual);
            ContactState gpuHistory{};
            int invalid = 0;
            checkCuda(cudaMemcpy(&gpuHistory, deviceHistory, sizeof(ContactState), cudaMemcpyDeviceToHost), "download contact history");
            checkCuda(cudaMemcpy(&invalid, deviceInvalid, sizeof(int), cudaMemcpyDeviceToHost), "download geometry flag");
            if (invalid) {
                throw std::runtime_error("invalid contact geometry");
            }
            for (std::size_t i = 0; i < 2; ++i) {
                checkVector(actual[i].force, expected[i].force);
                checkVector(actual[i].torque, expected[i].torque);
            }
            checkVector(gpuHistory.tangentialDisplacement, cpuHistory.tangentialDisplacement);
            checkVector(gpuHistory.previousNormal,cpuHistory.previousNormal);

            if (gpuHistory.sliding != cpuHistory.sliding) {
                throw std::runtime_error("Sliding-state mismatch");
            }
        }

        checkCuda(cudaFree(deviceInvalid), "free geometry flag");
        deviceInvalid = nullptr;

        checkCuda(cudaFree(deviceHistory), "free contact history");
        deviceHistory = nullptr;

        std::cout << "Passed 100 contact-history updates\n";
    } catch (const std::exception& error) {
        if (deviceInvalid) cudaFree(deviceInvalid);
        if (deviceHistory) cudaFree(deviceHistory);

        std::cerr << error.what() << '\n';
        return 1;
    }
}