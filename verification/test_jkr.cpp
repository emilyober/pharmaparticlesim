#include "jkr_full.h"

#include <iomanip>
#include <iostream>
#include <cmath>

int main() {
    const JKRParameters parameters{
        1.0e6,
        0.0005,
        0.05
    };
    auto print = [&](const char* label, double overlap, bool previouslyActive){
        const auto result = JKRFull::evaluate(overlap,previouslyActive,parameters);
        std::cout << label << ','
                  << overlap << ','
                  << result.active << ','
                  << result.contactRadius << ','
                  << result.elasticNormalForce << '\n';
    };
    std::cout << std::setprecision(17);
    std::cout << "case,overlap,active,radius,force\n";
    print("separated_new", -1.0e-8, false);
    print("compressed_new", 1.0e-6, false);
    print("zero_overlap_existing", 0.0, true);
    print("small_gap_existing", -1.0e-8, true);
    print("detached_existing", -1.0e-5, true);

    const JKRParameters nonadhesive{
        1.0e6,
        0.0005,
        0.0
    };

    const auto hertz = JKRFull::evaluate(
        1.0e-6, false, nonadhesive
    );

    std::cout << "hertz_limit,"
            << 1.0e-6 << ','
            << hertz.active << ','
            << hertz.contactRadius << ','
            << hertz.elasticNormalForce << '\n';
    
    const double pi = std::acos(-1.0);
    const double E = parameters.effectiveYoungModulus;
    const double R = parameters.effectiveRadius;
    const double gamma = parameters.surfaceEnergy;

    const double equilibriumRadius = std::cbrt(9.0 * pi * gamma * R * R / E);

    const double equilibriumOverlap = equilibriumRadius * equilibriumRadius / R - 2.0 * std::sqrt(pi * gamma * equilibriumRadius / E);

    print("equilibrium", equilibriumOverlap, true);
    bool active = false;

    std::cout << "\nstep,overlap,active,radius,force\n";

    for (int step = 0; step <= 600; ++step) {
        const double overlap = 1.0e-6 - step * 5.0e-9;

        const auto result = JKRFull::evaluate(
            overlap, active, parameters
        );

        active = result.active;

        if (step % 100 == 0) {
            std::cout << step << ','
                    << overlap << ','
                    << result.active << ','
                    << result.contactRadius << ','
                    << result.elasticNormalForce << '\n';
        }
    }
}