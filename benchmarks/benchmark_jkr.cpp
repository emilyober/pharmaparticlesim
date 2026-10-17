#include "jkr_full.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
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

int main() {
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t sampleCount = 65536;
    constexpr int passes = 16;
    constexpr int trials = 501;
    const JKRParameters parameters{1.0e6,0.0005,0.05};
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(-1.0e-6, 5.0e-6);
    std::vector<double> overlaps(sampleCount);
    for (double& overlap : overlaps) {
        overlap = distribution(generator);
    }
    double maximumRelativeError = 0.0;

    for (double overlap : overlaps) {
        const auto reference = JKRFull::evaluate(
            overlap, true, parameters
        );
        const double radius = benchmarkContactRadius(
            overlap, parameters
        );
        if (!std::isfinite(radius) || radius <= 0.0) {
            std::cerr << "Invalid benchmark contact radius.\n";
            return 1;
        }
        const double relativeError = std::abs(radius - reference.contactRadius) /reference.contactRadius;
        maximumRelativeError = std::max(maximumRelativeError, relativeError);
    }

    std::cerr << "Maximum radius relative error: "
            << maximumRelativeError << '\n';

    if (maximumRelativeError > 1.0e-12) {
        std::cerr << "Radius comparison failed.\n";
        return 1;
    }
    auto evaluateBatch = [&]() {
        double checksum = 0.0;
        for (int pass = 0; pass < passes; ++pass) {
            for (double overlap : overlaps) {
                const auto result = JKRFull::evaluate(
                    overlap, true, parameters
                );
                checksum += result.elasticNormalForce;
            }
        }
        return checksum;
    };
    double checksum = evaluateBatch();
    std::vector<double> timings;
    for (int trial = 0; trial < trials; ++trial) {
        const auto start = Clock::now();
        const double batchChecksum = evaluateBatch();
        const auto finish = Clock::now();
        checksum += batchChecksum;
        const double nanoseconds =
            std::chrono::duration<double, std::nano>(finish - start).count();
        timings.push_back(nanoseconds /(static_cast<double>(sampleCount) * passes));
    }
    std::sort(timings.begin(), timings.end());
    std::cout << std::setprecision(10);
    std::cout << "measurement,ns_per_contact\n";
    std::cout << "minimum," << timings.front() << '\n';
    std::cout << "median,"
              << timings[timings.size() / 2] << '\n';
    std::cout << "maximum," << timings.back() << '\n';
    std::cout << "checksum," << checksum << '\n';
}