#include <iostream>
#include "simulator.h"

int main(){
    Simulator simulator(10.0,10.0);
    Particle p{{5.0, 5.0}, {1.0, 1.0}, 1.0, 0.5};
    simulator.addParticle(p);
    constexpr double dt = 0.01;
    for (int i = 0; i < 1000; ++i) {
        simulator.step(dt);
        const auto& particles = simulator.getParticles();
        for (const auto& particle : particles) {
            std::cout << "Particle position: (" << particle.position.x << ", " << particle.position.y << ")\n";
        }
    }
    return 0;
}