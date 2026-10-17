#include "jkr_benchmark_helpers.h"

#include <algorithm>
#include <cmath>

double benchmarkContactRadius(
    double overlap,
    const JKRParameters& parameters
) {
    const double E = parameters.effectiveYoungModulus;
    const double R = parameters.effectiveRadius;
    const double gamma = parameters.surfaceEnergy;
    const double pi = std::acos(-1.0);
    const double R2 = R * R;
    const double dR = overlap * R;
    const double dR2 = dR * dR;
    const double t0 = gamma * gamma * R2 * R2 * E;
    const double t1 = 27.0 * pi * pi * t0;
    const double t2 = 8.0 * dR * dR2 * E * E * E;
    const double t3 = 4.0 * dR2 * E;
    const double sqrt1 = std::max(0.0, t0 * (t1 + 2.0 * t2));
    const double t4 = std::cbrt(t1 + t2 + 3.0 * std::sqrt(3.0) * pi * std::sqrt(sqrt1));
    const double t5 = t3 / t4 + t4 / E;
    const double t6 = std::sqrt(std::max(0.0, 2.0 * dR + t5));
    const double sqrt3 = std::max(0.0,4.0 * dR - t5 + 6.0 * std::sqrt(6.0) * gamma * pi * R2 /(E * t6));
    return (t6 + std::sqrt(sqrt3)) / std::sqrt(6.0);
}