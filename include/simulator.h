#pragma once

#include <vector>
#include <map>
#include <tuple>
#include <utility>
#include <cstddef>
#include <string>
#include <fstream>
#include "particle.h"
#include "material_database.h"
#include "contact_model.h"
#include "contact_state.h"
#include "wall.h"


class Simulator {
    public:
        Simulator() = default;
        void loadMaterials(const std::string& materialFile, const std::string& contactFile);
        int addParticle(const Vec3D& position, const Vec3D& velocity, double radius, const std::string& material, const Vec3D& angularVelocity = Vec3D{}, int group = 0);
        int addWall(const Vec3D& point, const Vec3D& inwardNormal, const std::string& material, const Vec3D& velocity = Vec3D{0.0, 0.0, 0.0},const Vec3D& angularVelocity = Vec3D{});
        int addCylinder(const Vec3D& axisPoint, const Vec3D& axisDirection, double radius, const std::string& material, double angularSpeed=0.0, const Vec3D& velocity = Vec3D{});
        void run(const std::string& outputFilename = "output.csv");
        void addRecordingWindow(std::size_t firstStep,std::size_t lastStep);
        void setDuration(double duration);
    private:
        using ContactKey = std::pair<int,int>;
        using CellKey = std::tuple<int,int,int>;
        void buildCells();
        double calculateMass(double radius, const std::string& material) const;
        void resetForces();
        void computeContacts();
        void integrate();
        void writeOutput(std::ofstream& file, std::size_t step, double time);
        MaterialDatabase materials_;
        std::vector<Particle> particles_;
        std::vector<PlaneWall> walls_;
        std::vector<CylindricalWall> cylinders_;
        std::map<ContactKey, ContactState> particleContacts_;
        std::map<ContactKey, ContactState> wallContacts_;
        std::map<ContactKey, ContactState> cylinderContacts_;
        std::map<CellKey, std::vector<std::size_t>> cells_;
        std::vector<std::pair<std::size_t, std::size_t>> recordingWindows_;
        double cellSize_{0.0};
        int nextParticleId_{0};
        int nextWallId_{0};
        std::size_t stepIndex_{0};
        Vec3D gravity_{0.0, 0.0, -9.81};
        double dt_{5e-6};
        double endTime_{0.1};
        std::size_t outputEvery_{200};
};