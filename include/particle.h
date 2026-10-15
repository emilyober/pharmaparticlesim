#pragma once
#include <string>
#include "vec3d.h"
#include "material.h"

struct Particle {
    int id{-1};
    int group{0};
    Vec3D position{};
    Vec3D velocity{};
    Vec3D force{};
    Vec3D torque{};
    Vec3D angularVelocity{};

    double inertia{0.0};
    double radius{0.0};
    double mass{0.0};
    std::string material{};
};