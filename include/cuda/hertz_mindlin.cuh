#pragma once
#include "cuda/vector_math.cuh"
#include "contact_state.h"
#include "material.h"

namespace cuda_dem {
    struct ContactParticle {
        Vec3D position;
        Vec3D velocity;
        Vec3D angularVelocity;
        double radius;
        double mass;
    };
    struct PairContactResult {
        Vec3D forceOnSecond{};
        Vec3D torqueOnFirst{};
        Vec3D torqueOnSecond{};
        bool touching{false};
        bool invalidGeometry{false};
    };
    __device__ inline PairContactResult evaluateHertzMindlin(
        const ContactParticle& p1,
        const ContactParticle& p2,
        const MaterialPairProperties& pair,
        ContactState& history,
        double dt
    ) {
        PairContactResult result{};
        const Vec3D separation = vsub(p2.position,p1.position);
        const double distance = vnorm(separation);
        const double overlap = p1.radius + p2.radius - distance;
        if (overlap <= 0.0){
            history = ContactState{};
            return result;
        }
        if (distance <= 1e-12) {
            result.invalidGeometry = true;
            return result;
        }
        result.touching = true;
        const Vec3D normal = vdiv(separation, distance);
        const double effectiveR = (p1.radius * p2.radius) / (p1.radius + p2.radius);
        const double effectiveM = (p1.mass * p2.mass) / (p1.mass + p2.mass);
        const double contactRadius = sqrt(effectiveR * overlap);
        const double elasticForce = (4.0 / 3.0) * pair.effectiveYoungModulus * contactRadius * overlap;
        const double normalStiffness = 2.0 * pair.effectiveYoungModulus * contactRadius;
        const double tangentialStiffness = 8.0 * pair.effectiveShearModulus * contactRadius;
        const double damping = 2.0 * sqrt(5.0 / 6.0) * pair.dampingRatio * sqrt(effectiveM * normalStiffness);
        const Vec3D surface1 = vadd(p1.velocity,vcross(p1.angularVelocity, vmul(normal, p1.radius)));
        const Vec3D surface2 = vadd(p2.velocity,vcross(p2.angularVelocity, vmul(normal, -p2.radius)));
        const Vec3D relativeVelocity = vsub(surface2, surface1);
        const double normalVelocity = vdot(relativeVelocity, normal);
        const Vec3D tangentialVelocity = vsub(relativeVelocity, vmul(normal, normalVelocity));
        const double trialNormalForce =
        elasticForce - damping * normalVelocity;
        const double normalForce = trialNormalForce > 0.0 ? trialNormalForce : 0.0;
        Vec3D tangentialForce{};
        if (normalForce <= 0.0){
            history.tangentialDisplacement ={};
            history.sliding = false;
        } else {
            const double oldLength = vnorm(history.tangentialDisplacement);
            Vec3D displacement = vsub(history.tangentialDisplacement,vmul(normal, vdot(history.tangentialDisplacement, normal)));
            const double projectedLength = vnorm(displacement);
            if (projectedLength > 1e-14){
                displacement = vmul(displacement, oldLength / projectedLength);
            } else {
                displacement = {};
            }
            displacement = vadd(displacement,vmul(tangentialVelocity,dt));
            const double kt = tangentialStiffness;
            tangentialForce = vsub(vmul(displacement,-kt),vmul(tangentialVelocity,damping));
            const double trialMagnitude = vnorm(tangentialForce);
            const double frictionLimit = pair.contact.staticFriction * normalForce;
            history.sliding = trialMagnitude > frictionLimit;
            if (history.sliding) {
                tangentialForce = vmul(tangentialForce, frictionLimit / trialMagnitude);
                displacement = vdiv(vadd(tangentialForce,vmul(tangentialVelocity, damping)),-kt);
            }
            history.tangentialDisplacement = displacement;
        }
        history.previousNormal = normal;
        result.forceOnSecond = vadd(vmul(normal, normalForce), tangentialForce);
        const Vec3D arm1 = vmul(normal, p1.radius - overlap / 2.0);
        const Vec3D arm2 = vmul(normal, -(p2.radius - overlap / 2.0));
        result.torqueOnFirst = vcross(arm1, vmul(tangentialForce, -1.0));
        result.torqueOnSecond = vcross(arm2, tangentialForce);
        const Vec3D relativeOmega = vsub(p2.angularVelocity, p1.angularVelocity);
        const Vec3D rollingOmega = vsub(relativeOmega, vmul(normal, vdot(relativeOmega, normal)));
        const double rollingSpeed = vnorm(rollingOmega);
        if (rollingSpeed > 1e-8) {
            const double rollingMoment = pair.contact.rollingFriction * effectiveR * normalForce;
            const Vec3D rollingTorque = vmul(rollingOmega, -rollingMoment / rollingSpeed);
            result.torqueOnFirst = vsub(result.torqueOnFirst, rollingTorque);
            result.torqueOnSecond = vadd(result.torqueOnSecond, rollingTorque);
        }
        return result;
    }
}
