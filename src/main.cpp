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
        constexpr double PI = 3.14159265358979323846;
        const double drumRadius = 0.02;
        const double particleRadius = 0.001;

        const double rpm = 10.0;
        const double angularSpeed =
            rpm * 2.0 * PI / 60.0;

        simulator.addCylinder(
            Vec3D{0.0, 0.0, 0.0},
            Vec3D{0.0, 1.0, 0.0},
            drumRadius,
            "Steel",
            angularSpeed
        );

        const double drumLength = 0.02;
        const double halfLength = 0.5 * drumLength;
        const Vec3D drumAngularVelocity{
            0.0, angularSpeed, 0.0
        };
        simulator.addWall(
            Vec3D{0.0, -halfLength, 0.0},
            Vec3D{0.0, 1.0, 0.0},
            "Steel",
            Vec3D{},
            drumAngularVelocity
        );

        simulator.addWall(
            Vec3D{0.0, halfLength, 0.0},
            Vec3D{0.0, -1.0, 0.0},
            "Steel",
            Vec3D{},
            drumAngularVelocity
        );


        simulator.addParticle(
            Vec3D{
                0.0,
                0.0,
                -(drumRadius - particleRadius)
            },
            Vec3D{},
            particleRadius,
            "Powder"
        );


        simulator.run("drum_test.csv");

        std::cout << "Output saved to drum_test.csv\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}