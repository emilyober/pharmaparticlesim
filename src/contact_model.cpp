#include "contact_model.h"
#include "wall.h"
#include <cmath>
#include <algorithm>

namespace {
    constexpr double PI = 3.1415926535897932;
    struct ContactParameters{
        double normalElasticForce;
        double normalDamping;
        double tangentialStiffness;
        double tangentialDamping;
    };
    double effectiveMass(double m1, double m2) {
        return (m1 * m2) / (m1 + m2);
    }
    double effectiveRadius(double r1, double r2) {
        return (r1 * r2) / (r1 + r2);
    }
    ContactParameters calculateParameters(
        const MaterialPairProperties& pair,
        double effectiveMass,
        double effectiveRadius,
        double overlap
    ) {
        const double contactRadius = std::sqrt(effectiveRadius * overlap);
        const double normalElasticForce = (4.0 / 3.0) * pair.effectiveYoungModulus * contactRadius * overlap;
        const double normalStiffness = 2.0 * pair.effectiveYoungModulus * contactRadius;
        const double tangentialStiffness = 8.0 * pair.effectiveShearModulus * contactRadius;
        const double normalDamping = 2.0 * std::sqrt(5.0/6.0) * pair.dampingRatio * std::sqrt(effectiveMass * normalStiffness);
        const double tangentialDamping = normalDamping; //assuming same damping for tangential
        return {normalElasticForce, normalDamping, tangentialStiffness, tangentialDamping};
    }
    Vec3D calculateTangentialForce(
        const Vec3D& normal,
        const Vec3D& tangentialVelocity,
        double normalForce,
        const ContactParameters& params,
        const ContactProperties& props,
        ContactState& contactState,
        double dt
    ) {
        constexpr double displacementTolerance = 1e-14;

        if (normalForce <= 0.0) {
            contactState.tangentialDisplacement = {};
            contactState.sliding = false;
            contactState.previousNormal = normal;
            return {};
        }
        const double oldLength = norm(contactState.tangentialDisplacement);
        Vec3D displacement = contactState.tangentialDisplacement - normal * dot(contactState.tangentialDisplacement,normal);
        const double projectedLength = norm(displacement);
        if (projectedLength > displacementTolerance) {
            displacement *= oldLength / projectedLength;
        } else {
            displacement = {};
        }
        displacement += tangentialVelocity*dt;
        const double kt = params.tangentialStiffness;
        const double ct = params.tangentialDamping;
        const Vec3D trialForce = -kt * displacement - ct * tangentialVelocity;
        const double trialMagnitude = norm(trialForce);
        const double frictionLimit = props.staticFriction * normalForce;
        Vec3D tangentialForce = trialForce;
        contactState.sliding = trialMagnitude > frictionLimit;
        if (contactState.sliding){
            tangentialForce *= frictionLimit / trialMagnitude;
            displacement = (tangentialForce + tangentialVelocity * ct) / (-kt);
        }
        contactState.tangentialDisplacement = displacement;
        contactState.previousNormal = normal;
        return tangentialForce;
    }
}

void ContactModel::resolveContact(
    Particle& p1, 
    Particle& p2, 
    const MaterialDatabase& materials,
    ContactState& contactState,
    double dt
) {
    const Vec3D separation = p2.position - p1.position;
    const double distance = norm(separation);
    const double overlap = p1.radius + p2.radius - distance;
    if (overlap <= 0.0) {
        return; //no contact
    }
    if (distance <= 1e-12) {
        return; //avoid singularity
    }
    const Vec3D normal = separation / distance;
    const MaterialPairProperties& pair = materials.getPairProperties(p1.material,p2.material);
    const ContactProperties& contactProps = pair.contact;
    const double effectiveR = effectiveRadius(p1.radius, p2.radius);
    const double effectiveM = effectiveMass(p1.mass, p2.mass);
    const ContactParameters params = calculateParameters(pair,effectiveM,effectiveR,overlap);
    const Vec3D r1 = normal*(p1.radius - overlap/2.0);
    const Vec3D r2 = -normal*(p2.radius - overlap/2.0);
    const Vec3D v1 = p1.velocity + cross(p1.angularVelocity, normal * p1.radius);
    const Vec3D v2 = p2.velocity + cross(p2.angularVelocity, normal * (-p2.radius));
    const Vec3D relativeVelocity = v2 - v1;
    const double normalVelocity = dot(relativeVelocity, normal);
    const Vec3D tangentialVelocity = relativeVelocity - normal * normalVelocity;
    const double normalForce = std::max(0.0, params.normalElasticForce - params.normalDamping * normalVelocity);
    const Vec3D tangentialForce = calculateTangentialForce(normal, tangentialVelocity, normalForce, params, contactProps, contactState, dt);
    const Vec3D totalForce = normal * normalForce + tangentialForce;
    p1.force -= totalForce;
    p2.force += totalForce;
    p1.torque += cross(r1,tangentialForce*-1.0);
    p2.torque += cross(r2,tangentialForce);
    const Vec3D relativeAngularVelocity = p2.angularVelocity - p1.angularVelocity;
    const Vec3D rollingAngularVelocity = relativeAngularVelocity - normal * dot(relativeAngularVelocity, normal);
    const double rollingSpeed = norm(rollingAngularVelocity);
    constexpr double rollingSpeedTolerance = 1e-8;
    if (rollingSpeed > rollingSpeedTolerance) {
        const double rollingMoment = contactProps.rollingFriction * effectiveR * normalForce;
        const Vec3D rollingTorque = rollingAngularVelocity * (-rollingMoment / rollingSpeed);
        p1.torque -= rollingTorque;
        p2.torque += rollingTorque;
    }
}

void ContactModel::particleWall(
    Particle& p,
    const PlaneWall& wall,
    const MaterialDatabase& materials,
    ContactState& contactState,
    double dt
) {
    const Vec3D normal = wall.inwardNormal;
    const double distance = dot(p.position - wall.point, normal);
    const double overlap = p.radius - distance;
    if (overlap <= 0.0) {
        return; //no contact
    }
    const MaterialPairProperties& pair = materials.getPairProperties(p.material, wall.material);
    const ContactProperties& contactProps = pair.contact;
    const double effectiveR = p.radius;
    const double effectiveM = p.mass;
    const ContactParameters params = calculateParameters(pair, effectiveM, effectiveR, overlap);
    const Vec3D r = normal * (-p.radius);
    const Vec3D surfaceV = p.velocity + cross(p.angularVelocity, r);
    const Vec3D wallContactPoint = p.position - normal*distance;
    const Vec3D wallSurfaceVelocity = wall.velocity + cross(wall.angularVelocity,wallContactPoint-wall.point);
    const Vec3D relativeV = surfaceV - wallSurfaceVelocity;
    const double normalVelocity = dot(relativeV, normal);
    const Vec3D tangentialVelocity = relativeV - normal * normalVelocity;
    const double normalForce = std::max(0.0, params.normalElasticForce - params.normalDamping * normalVelocity);
    const Vec3D tangentialForce = calculateTangentialForce(normal, tangentialVelocity, normalForce, params, contactProps, contactState, dt);
    p.force += normal * normalForce + tangentialForce;
    p.torque += cross(r, tangentialForce);
    const Vec3D relativeOmega = p.angularVelocity - wall.angularVelocity;
    const Vec3D rollingOmega = relativeOmega - normal * dot(relativeOmega,normal);
    const double rollingSpeed = norm(rollingOmega);
    constexpr double rollingSpeedTolerance = 1e-8;
    if (rollingSpeed > rollingSpeedTolerance) {
        const double rollingMoment = contactProps.rollingFriction * effectiveR * normalForce;
        p.torque += rollingOmega * (-rollingMoment / rollingSpeed);
    }
}

void ContactModel::particleCylinder(
    Particle& particle,
    const CylindricalWall& wall,
    const MaterialDatabase& materials,
    ContactState& state,
    double dt
) {
    const Vec3D axis = wall.axisDirection;
    const Vec3D offset = particle.position - wall.axisPoint;
    const Vec3D radialOffset = offset - axis * dot(offset, axis);
    const double radialDistance = norm(radialOffset);
    const double overlap = radialDistance + particle.radius - wall.radius;
    if (overlap <= 0.0){
        return;
    }
    if (radialDistance < 1e-12){
        return;
    }
    const Vec3D outward = radialOffset / radialDistance;
    const Vec3D normal = outward * -1.0;
    const MaterialPairProperties& pair =
    materials.getPairProperties(particle.material, wall.material);
    const ContactProperties& properties = pair.contact;
    const double effectiveR = particle.radius;
    const double effectiveM = particle.mass;
    const ContactParameters parameters = calculateParameters(pair,effectiveM,effectiveR,overlap);
    const Vec3D leverArm = outward*particle.radius;
    const Vec3D particleSurfaceVelocity = particle.velocity + cross(particle.angularVelocity, leverArm);
    const Vec3D wallOmega = axis * wall.angularSpeed;
    const Vec3D wallSurfaceVelocity = wall.velocity + cross(wallOmega, outward*wall.radius);
    const Vec3D relativeVelocity = particleSurfaceVelocity - wallSurfaceVelocity;
    const double normalVelocity = dot(relativeVelocity,normal);
    const Vec3D tangentialVelocity = relativeVelocity - normal*normalVelocity;
    const double normalForce = std::max(0.0,parameters.normalElasticForce-parameters.normalDamping*normalVelocity);
    const Vec3D tangentialForce = calculateTangentialForce(normal, tangentialVelocity, normalForce, parameters, properties, state, dt);
    particle.force += normal*normalForce + tangentialForce;
    particle.torque += cross(leverArm, tangentialForce);
    const Vec3D relativeOmega = particle.angularVelocity - wallOmega;
    const Vec3D rollingOmega = relativeOmega - normal*dot(relativeOmega,normal);
    const double rollingSpeed = norm(rollingOmega);
    constexpr double rollingSpeedTolerance = 1e-8;
    if (rollingSpeed > rollingSpeedTolerance){
        const double rollingMoment = properties.rollingFriction * effectiveR * normalForce;
        particle.torque += rollingOmega * (-rollingMoment/rollingSpeed);
    }
}   