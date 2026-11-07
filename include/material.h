#pragma once
#include <string>

struct Material {
    std::string name;
    double density{0.0};
    double youngModulus{0.0};
    double poissonRatio{0.0};
};

struct ContactProperties {
    double restitution{1.0};
    double staticFriction{0.0};
    double dynamicFriction{0.0};
    double rollingFriction{0.0};
};

struct MaterialPairProperties {
    double effectiveYoungModulus{0.0};
    double effectiveShearModulus{0.0};
    double dampingRatio{0.0};
    ContactProperties contact{};
};