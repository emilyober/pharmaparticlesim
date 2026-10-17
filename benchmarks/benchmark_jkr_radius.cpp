#include "jkr_full.h"
#include "jkr_benchmark_helpers.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

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
                checksum += benchmarkContactRadius(overlap, parameters);
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