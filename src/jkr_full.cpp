#include "jkr_full.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

JKRFullResult JKRFull::evaluate(
    double overlap,
    bool previouslyActive,
    const JKRParameters& parameters
) {
    const double E = parameters.effectiveYoungModulus;
    const double R = parameters.effectiveRadius;
    const double gamma = parameters.surfaceEnergy;
    if (!std::isfinite(overlap) || !std::isfinite(E) || !std::isfinite(R) || !std::isfinite(gamma)){
        throw std::invalid_argument("JKR inputs must be finite.");
    }
    if (E <= 0.0 || R <= 0.0 || gamma < 0.0) {
        throw std::invalid_argument("JKR requires positive modulus, radius, and nonnegative surface energy.");
    }
    JKRFullResult result{};

    //without adhesion, we have hertz.
    if (gamma == 0.0) {
        if (overlap <= 0.0) {
            return result;
        }
        result.active = true;
        result.contactRadius = std::sqrt(R*overlap);
        result.elasticNormalForce = (4.0/3.0)*E*result.contactRadius*overlap;
        return result;
    }
    const double PI = 3.1415926535897932;
    //existing adhesive contact
    const double pullOffDistance = std::cbrt(3.0*PI*PI/16.0) * std::cbrt(R*gamma*gamma/(E*E));
    if (!previouslyActive) {
        if (overlap <= 0.0){
            return result;
        }
    } else {
        if (overlap <= -pullOffDistance){
            return result;
        }
    }
    result.active = true;
    //radius and adhesive force calcs
    const double R2 = R*R;
    const double dR = overlap * R;
    const double dR2 = dR * dR;
    const double t0 = gamma * gamma * R2 * R2 * E;
    const double t1 = 27.0 * PI * PI *t0;
    const double t2 = 8.0*dR*dR2*E*E*E;
    const double t3 = 4.0*dR2*E;
    const double sqrt1 = std::max(0.0,t0*(t1+2.0*t2));
    const double t4 = std::cbrt(t1+t2+3.0*std::sqrt(3.0)*PI*std::sqrt(sqrt1));
    const double t5 = t3 / t4 + t4 / E;
    const double t6 = std::sqrt(std::max(0.0,2.0*dR+t5));
    const double sqrt3 = std::max(0.0,4.0*dR-t5+6.0*std::sqrt(6.0)*gamma*PI*R2/(E*t6));
    const double a = (t6+std::sqrt(sqrt3))/ std::sqrt(6.0);
    result.contactRadius = a;
    const double a2 = a*a;
    result.elasticNormalForce = (4.0/3.0)*E*a*a2/R-2.0*PI*a2*std::sqrt(4.0*gamma*E/(PI*a));
    return result;
}