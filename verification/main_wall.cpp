#include <iostream>
#include <exception>
#include "simulator.h"

int main(){
    try {
        Simulator simulator;
        simulator.loadMaterials(
            "materials.csv",
            "contacts.csv"
        );
        simulator.addWall(
            Vec3D{0.0, 0.0, 0.0},
            Vec3D{0.0, 0.0, 1.0},
            "Powder"
        );
        simulator.addParticle(
            Vec3D{0.0, 0.0, 0.003},
            Vec3D{0.02, 0.0, -0.05},
            0.001,
            "Powder"
        );

        simulator.run("wall_reference.csv");

        std::cout
            << "Output saved to wall_reference.csv\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}