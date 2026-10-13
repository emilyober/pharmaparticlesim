#include "simulator.h"

Simulator::Simulator(double width, double height)
    : width_(width), height_(height) {}

void Simulator::addParticle(const Particle& particle) {
    particles_.push_back(particle);
}

const std::vector<Particle>& Simulator::getParticles() const {
    return particles_;
}

void Simulator::applyGravity(Particle& particle, double dt) {
    constexpr double gravity = -9.81;
    particle.velocity.y += gravity * dt;
}

void Simulator::step(double dt) {
    for (auto& particle : particles_) {
        applyGravity(particle, dt);
        particle.position.x += particle.velocity.x * dt;
        particle.position.y += particle.velocity.y * dt;
        resolveWalls(particle);
    }
}

void Simulator::resolveWalls(Particle& p){
    const double restitution = 0.9; //coefficient of restitution for wall collisions
    if (p.position.x - p.radius < 0) {
        p.position.x = p.radius;
        p.velocity.x *= -restitution;
    } else if (p.position.x + p.radius > width_) {
        p.position.x = width_ - p.radius;
        p.velocity.x *= -restitution;
    }
    if (p.position.y - p.radius < 0) {
        p.position.y = p.radius;
        p.velocity.y *= -restitution;
    } else if (p.position.y + p.radius > height_) {
        p.position.y = height_ - p.radius;
        p.velocity.y *= -restitution;
    }
}

void Simulator::resolveCollisions(const std::vector<Particle>& particles) {
    
}