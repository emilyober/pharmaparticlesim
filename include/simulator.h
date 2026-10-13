#pragma once


#include "particle.h"
#include <vector>

class Simulator {
    public:
        Simulator(double width, double height);
        void addParticle(const Particle& particle);
        void step(double dt);
        const std::vector<Particle>& getParticles() const;
    private:
        double width_;
        double height_;
        std::vector<Particle> particles_;
        void applyGravity(Particle& particle, double dt);
        void resolveWalls(Particle& particle);
        void resolveCollisions(const std::vector<Particle>& particles);
};