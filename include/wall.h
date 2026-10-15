#pragma once
#include <string>
#include "vec3d.h"

struct PlaneWall {
    int id{-1};
    Vec3D point{};
    Vec3D velocity{0.0,0.0,0.0};
    Vec3D angularVelocity{};
    Vec3D inwardNormal{0.0,0.0,1.0};
    std::string material{};
};

struct CylindricalWall {
    int id{-1};
    Vec3D axisPoint{};
    Vec3D axisDirection{0.0,1.0,0.0};
    double radius{0.0};
    Vec3D velocity{};
    double angularSpeed{0.0};
    std::string material;
};