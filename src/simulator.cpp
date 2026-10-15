#include "simulator.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <limits>

namespace {
    constexpr double PI = 3.1415926535897932;
    bool isFinite(const Vec3D& value){
        return std::isfinite(value.x) &&
               std::isfinite(value.y) &&
               std::isfinite(value.z);
    }
}

void Simulator::loadMaterials(
    const std::string& materialFile,
    const std::string& contactFile
) {
    materials_.loadMaterials(materialFile);
    materials_.loadContactProperties(contactFile);
}

double Simulator::calculateMass(
    double radius,
    const std::string& material
) const {
    const double density = materials_.getMaterial(material).density;
    const double volume = (4.0/3.0)*PI*radius*radius*radius;
    return density*volume;
}

int Simulator::addParticle(const Vec3D& position, const Vec3D& velocity, double radius, const std::string& material, const Vec3D& angularVelocity, int group
) {
    if (!std::isfinite(radius) || radius <= 0.0){
        throw std::runtime_error("Particle radius must be finite and positive");
    }
    if (!isFinite(position) || !isFinite(velocity) || !isFinite(angularVelocity)) {
        throw std::runtime_error("Particle position and velocity must be finite");
    }

    Particle particle;
    particle.id = nextParticleId_++;
    particle.group = group;
    particle.position = position;
    particle.velocity = velocity;
    particle.angularVelocity = angularVelocity;
    particle.radius = radius;
    particle.material = material;

    particle.mass = calculateMass(radius,material);
    particle.inertia = (2.0/5.0) * particle.mass*radius*radius;
    particles_.push_back(particle);
    return particle.id;
}

int Simulator::addWall(
    const Vec3D& point,
    const Vec3D& inwardNormal,
    const std::string& material,
    const Vec3D& velocity,
    const Vec3D& angularVelocity
) {
    if (!isFinite(point) ||
        !isFinite(inwardNormal) || 
        !isFinite(velocity) ||
        !isFinite(angularVelocity)) {
            throw std::runtime_error("Wall geometry and velocity must be finite");
    }
    const double normalLength = norm(inwardNormal);
    if (!std::isfinite(normalLength) || normalLength <= 1e-10){
        throw std::runtime_error(
            "Wall normal must have a nonzero finite length"
        );
    }
    materials_.getMaterial(material);
    PlaneWall wall;
    wall.id = nextWallId_++;
    wall.point = point;
    wall.inwardNormal = inwardNormal / normalLength;
    wall.velocity = velocity;
    wall.angularVelocity = angularVelocity;
    wall.material = material;
    walls_.push_back(wall);
    return wall.id;
}

int Simulator::addCylinder(const Vec3D& axisPoint, const Vec3D& axisDirection, double radius, const std::string& material, double angularSpeed, const Vec3D& velocity) {
    if (!isFinite(axisPoint) ||
        !isFinite(axisDirection) ||
        !isFinite(velocity) ||
        !std::isfinite(angularSpeed)) {
        throw std::runtime_error(
            "Cylinder geometry and velocities must be finite"
        );
    }
    if (!std::isfinite(radius) || radius <= 0.0) {
        throw std::runtime_error(
            "Cylinder radius must be finite and positive"
        );
    }
    const double axisLength = norm(axisDirection);
    if (!std::isfinite(axisLength) || axisLength <= 1e-12){
        throw std::runtime_error(
            "Cylinder axis must have a nonzero finite length"
        );
    }
    materials_.getMaterial(material);
    CylindricalWall cylinder;
    cylinder.id = nextWallId_++;
    cylinder.axisPoint = axisPoint;
    cylinder.axisDirection = axisDirection / axisLength;
    cylinder.radius = radius;
    cylinder.velocity = velocity;
    cylinder.angularSpeed = angularSpeed;
    cylinder.material = material;
    cylinders_.push_back(cylinder);
    return cylinder.id;
}

void Simulator::resetForces() {
    for (Particle& particle : particles_) {
        particle.force = gravity_ * particle.mass;
        particle.torque = {};
    }
}

void Simulator::buildCells() {
    cells_.clear();
    cellSize_ = 0.0;
    if (particles_.empty()) {
        return;
    }
    double largestRadius = 0.0;

    for (const Particle& particle : particles_) {
        largestRadius = std::max(largestRadius, particle.radius);
    }

    cellSize_ = 2.0 * largestRadius;

    if (!std::isfinite(cellSize_) || cellSize_ <= 0.0) {
        throw std::runtime_error("Invalid neighbor-search cell size");
    }

    const auto cellCoordinate = [this](double position) {
        const double coordinate = std::floor(position / cellSize_);
        const double minimum = static_cast<double>(std::numeric_limits<int>::min()) + 1.0;
        const double maximum = static_cast<double>(std::numeric_limits<int>::max()) - 1.0;
        if (!std::isfinite(coordinate) || coordinate < minimum || coordinate > maximum){
            throw std::runtime_error("particle position exceeds cell-grid range");
        }
        return static_cast<int>(coordinate);
    };
    for (std::size_t index = 0; index < particles_.size(); ++index){
        const Vec3D& position = particles_[index].position;
        const CellKey key {cellCoordinate(position.x), cellCoordinate(position.y), cellCoordinate(position.z)};
        cells_[key].push_back(index);
    }
}

void Simulator::computeContacts() {
    buildCells();
    for (const auto& [cellKey, indices] : cells_){
        const auto [cx,cy,cz] = cellKey;
        for (std::size_t i : indices) {
            Particle& p1 = particles_[i];
            for (int dx = -1; dx <= 1; ++dx){
                for (int dy = -1; dy<= 1; ++dy){
                    for (int dz = -1; dz <= 1; dz++){
                        const CellKey neighborKey{
                            cx+dx,
                            cy+dy,
                            cz+dz
                        };
                        const auto neighbor = cells_.find(neighborKey);
                        if (neighbor == cells_.end()){
                            continue;
                        }
                        for (std::size_t j : neighbor->second){
                            Particle& p2 = particles_[j];
                            if (p2.id <= p1.id){
                                continue;
                            }
                            const Vec3D separation = p2.position - p1.position;
                            const double distance = norm(separation);
                            const double overlap = p1.radius + p2.radius - distance;
                            if (overlap <= 0.0){
                                continue;
                            }
                            if (distance <= 1e-12){
                                throw std::runtime_error("Particle centers coincide.");
                            }
                            const ContactKey key{p1.id,p2.id};
                            auto [it,inserted] = particleContacts_.try_emplace(key);
                            ContactState& state = it->second;
                            if (inserted) {
                                state.previousNormal = separation / distance;
                            }
                            state.lastSeenStep = stepIndex_;
                            ContactModel::resolveContact(p1,p2,materials_,state,dt_);
                        }
                    }
                }
            }
        }
    }

    for (Particle& particle : particles_){
        for (const PlaneWall& wall : walls_){
            const double distanceToPlane = dot(particle.position - wall.point, wall.inwardNormal);
            const double overlap = particle.radius - distanceToPlane;
            if (overlap <= 0.0){
                continue;
            }
            const ContactKey key{particle.id,wall.id};
            auto [it, inserted] = wallContacts_.try_emplace(key);
            ContactState& state = it->second;
            if (inserted){
                state.previousNormal = wall.inwardNormal;
            }
            state.lastSeenStep = stepIndex_;
            ContactModel::particleWall(
                particle,
                wall,
                materials_,
                state,
                dt_
            );
        }
    }

    for (Particle& particle : particles_) {
        for (const CylindricalWall& cylinder : cylinders_) {
            const Vec3D offset = particle.position - cylinder.axisPoint;
            const Vec3D radialOffset = offset - cylinder.axisDirection * dot(offset,cylinder.axisDirection);
            const double radialDistance = norm(radialOffset);
            const double overlap = radialDistance + particle.radius - cylinder.radius;
            if (overlap <= 0.0){
                continue;
            }
            const ContactKey key{
                particle.id, cylinder.id
            };
            auto [it, inserted] = cylinderContacts_.try_emplace(key);
            ContactState& state = it->second;
            if (inserted && radialDistance > 1e-12){
                state.previousNormal = radialOffset * (-1.0/radialDistance);
            }
            state.lastSeenStep = stepIndex_;
            ContactModel::particleCylinder(particle,cylinder,materials_,state,dt_);
        }
    }

    const auto removeOldContacts = [this](auto& contacts){
        for (auto it = contacts.begin(); it != contacts.end();){
            if (it->second.lastSeenStep != stepIndex_){
                it = contacts.erase(it);
            } else {
                ++it;
            }
        }
    };
    removeOldContacts(particleContacts_);
    removeOldContacts(wallContacts_);
    removeOldContacts(cylinderContacts_);
}

void Simulator::integrate() {
    //semi-implicit Euler -- update velociy before position
    for (Particle& particle: particles_) {
        const Vec3D acceleration = particle.force / particle.mass;
        const Vec3D angularAcceleration = particle.torque / particle.inertia;
        particle.velocity += acceleration * dt_;
        particle.angularVelocity += angularAcceleration * dt_;
        particle.position += particle.velocity * dt_;
    }
    for (PlaneWall& wall : walls_){
        wall.point += wall.velocity * dt_;
    }
    for (CylindricalWall& cylinder : cylinders_){
        cylinder.axisPoint += cylinder.velocity*dt_;
    }
}

void Simulator::writeOutput(
    std::ofstream& file,
    std::size_t step,
    double time
) {
    file << std::setprecision(17);
    for (const Particle& particle : particles_) {
        file
            << step << ','
            << time << ','
            << particle.id << ','
            << particle.group << ','
            << particle.position.x << ','
            << particle.position.y << ','
            << particle.position.z << ','
            << particle.velocity.x << ','
            << particle.velocity.y << ','
            << particle.velocity.z << ','
            << particle.angularVelocity.x << ','
            << particle.angularVelocity.y << ','
            << particle.angularVelocity.z << ','
            << particle.radius
            << '\n';
    }
    if (!file) {
        throw std::runtime_error(
            "Failed to write simulation output"
        );
    }
}

void Simulator::run(
    const std::string& outputFilename
) {
    if (!std::isfinite(dt_) || dt_ <= 0.0){
        throw std::runtime_error(
            "Timestep must be finite and positive"
        );
    }
    if (!std::isfinite(endTime_) || endTime_ <= 0.0){
        throw std::runtime_error("Simulation duration must be finite and positive");
    }
    if (outputEvery_ <= 0){
        throw std::runtime_error("Output interval must be greater than zero");
    }
    std::ofstream file(outputFilename);
    if (!file) {
        throw std::runtime_error("Could not open output file: " + outputFilename);
    }
    file
        << "step,time,id,group,"
        << "x,y,z,"
        << "vx,vy,vz,"
        << "omega_x,omega_y,omega_z,"
        << "radius\n";
    writeOutput(
        file,
        stepIndex_,
        static_cast<double>(stepIndex_) * dt_
    );
    double elapsedTime = 0.0;
    while (elapsedTime < endTime_){
        ++stepIndex_;
        resetForces();
        computeContacts();
        integrate();

        elapsedTime += dt_;

        const double time = static_cast<double>(stepIndex_)*dt_;
        if (stepIndex_ % outputEvery_ == 0 || elapsedTime >= endTime_){
            writeOutput(file,stepIndex_,time);
       
        }
    }
    file.flush();
    if(!file){
        throw std::runtime_error(
            "Failed to save simulation output"
        );
    }
}