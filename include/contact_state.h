#pragma once
#include <cstddef>
#include "vec3d.h"

struct ContactState {
    Vec3D tangentialDisplacement{};
    Vec3D previousNormal{};
    bool sliding{false};
    std::size_t lastSeenStep{0};
};