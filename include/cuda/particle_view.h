#pragma once
#include <cstddef>

struct DeviceParticleView {
    std::size_t count{0};

    double* x{nullptr};
    double* y{nullptr};
    double* z{nullptr};

    double* vx{nullptr};
    double* vy{nullptr};
    double* vz{nullptr};

    double* wx{nullptr};
    double* wy{nullptr};
    double* wz{nullptr};

    double* fx{nullptr};
    double* fy{nullptr};
    double* fz{nullptr};

    double* tx{nullptr};
    double* ty{nullptr};
    double* tz{nullptr};

    double* radius{nullptr};
    double* mass{nullptr};
    double* inertia{nullptr};
};