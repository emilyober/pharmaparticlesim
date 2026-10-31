#pragma once
#include <vector>
#include "particle.h"
#include "cuda/particle_view.h"

class DeviceParticles {
    public:
        explicit DeviceParticles(const std::vector<Particle>& particles);
        ~DeviceParticles() noexcept;
        DeviceParticles(const DeviceParticles&) = delete;
        DeviceParticles& operator=(const DeviceParticles&) = delete;
        DeviceParticleView view() const {
            return view_;
        }
        void download(std::vector<Particle>& particles) const;
    private:
        double* storage_{nullptr};
        DeviceParticleView view_{};
};