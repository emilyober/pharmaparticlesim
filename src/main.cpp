#include <iostream>
#include <exception>
#include "simulator.h"
#include <random>

int main(){
    try {
        Simulator simulator;
        simulator.loadMaterials(
            "materials.csv",
            "contacts.csv"
        );
        constexpr double PI = 3.14159265358979323846;
        const double drumRadius = 0.02;
        const double drumLength = 0.02;
        const double halfLength = drumLength / 2.0;
        const double particleRadius = 0.001;
        const double rpm = 10.0;
        const double angularSpeed =
            rpm * 2.0 * PI / 60.0;
        const Vec3D drumAngularVelocity{
            0.0, angularSpeed, 0.0
        };

        simulator.addCylinder(
            Vec3D{0.0, 0.0, 0.0},
            Vec3D{0.0, 1.0, 0.0},
            drumRadius,
            "Steel",
            angularSpeed
        );
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

        const double spacing = 0.0022;
        const double usableRadius =drumRadius - particleRadius - 0.0001;
        std::mt19937 generator(43);
        std::uniform_real_distribution<double> offset(-0.00002, 0.00002);
        for (int i = -6; i < 6; ++i) {
            for (int j = -3; j <= 3; ++j) {
                for (int k = -7; k <= -4; ++k) {
                    const double x = (i + 0.5) * spacing + offset(generator);
                    const double y = j * spacing + offset(generator);
                    const double z = k * spacing + offset(generator);

                    if (x * x + z * z >
                        usableRadius * usableRadius) {
                        continue;
                    }

                    const int group = (x < 0.0) ? 0 : 1;

                    simulator.addParticle(
                        Vec3D{x, y, z},
                        Vec3D{},
                        particleRadius,
                        "Powder",
                        Vec3D{},
                        group
                    );
                }
            }
        }

        simulator.setDuration(0.1);

        // Record every timestep during the final 1 ms
        // before 1, 3, and 6 seconds.
        simulator.addRecordingWindow(0, 20000);
        simulator.run("experiments/results/local_timestepping/pilot_seed43_10rpm_0p1s_dense.csv");
        std::cout
            << "Output saved to drum_mixing.csv\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}