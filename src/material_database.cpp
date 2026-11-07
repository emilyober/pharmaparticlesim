#include "material_database.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cmath>

namespace {
    std::vector<std::string> splitCSVLine(
        const std::string& line
    ) {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ',')) {
            fields.push_back(field);
        }
        return fields;
    }

    double readNumber(const std::string& text) {
        std::size_t used = 0;
        const double value = std::stod(text, &used);
        if (!std::isfinite(value) || text.find_first_not_of(" \t\r", used) != std::string::npos) {
            throw std::runtime_error("Invalid number: " + text);
        }
        return value;
    }
}

std::string MaterialDatabase::makePairKey(const std::string& m1, const std::string& m2) {
    return m1 < m2 ? m1 + "|" + m2 : m2 + "|" + m1;
}

void MaterialDatabase::loadMaterials(const std::string& filename) {
    std::ifstream file(filename);
    if (!file){
        throw std::runtime_error("Could not open material file: " + filename);
    }
    pair_properties_.clear();
    std::string line;
    std::getline(file,line);
    while(std::getline(file,line)){
        if (line.find_first_not_of(" \t\r") == std::string::npos){
            continue;
        }
        const auto fields = splitCSVLine(line);
        if (fields.size() != 4){
            throw std::runtime_error("Invalid material line: " + line);
        }
        Material material;
        material.name = fields[0];
        material.density = readNumber(fields[1]);
        material.youngModulus = readNumber(fields[2]);
        material.poissonRatio = readNumber(fields[3]);
        if (material.name.empty() || material.name.find('|') != std::string::npos || material.density <= 0.0 || material.youngModulus <= 0.0 || material.poissonRatio <= -1.0 || material.poissonRatio >= 0.5){
            throw std::runtime_error("Invalid material: " + material.name);
        }
        materials_[material.name] = material;
    }
}

void MaterialDatabase::loadContactProperties(const std::string& filename) {
    std::ifstream file(filename);
    if (!file){
        throw std::runtime_error("Could not open contact properties file: " + filename);
    }
    pair_properties_.clear();
    std::string line;
    std::getline(file,line);
    while(std::getline(file,line)){
        if (line.find_first_not_of(" \t\r") == std::string::npos){
            continue;
        }
        const auto fields = splitCSVLine(line);
        if (fields.size() != 6){
            throw std::runtime_error("Invalid contact properties line: " + line);
        }
        const std::string material1 = fields[0];
        const std::string material2 = fields[1];
        if (!hasMaterial(material1) || !hasMaterial(material2)){
            throw std::runtime_error("Unknown material in contact properties line: " + line);
        }
        ContactProperties properties;
        properties.restitution = readNumber(fields[2]);
        properties.staticFriction = readNumber(fields[3]);
        properties.dynamicFriction = readNumber(fields[4]);
        properties.rollingFriction = readNumber(fields[5]);
        if (properties.restitution <= 0.0 || properties.restitution > 1.0 || properties.staticFriction < 0.0 || properties.dynamicFriction < 0.0 || properties.rollingFriction < 0.0 || properties.dynamicFriction > properties.staticFriction){
            throw std::runtime_error("Invalid contact properties in line: " + line);
        }
        const Material& m1 = getMaterial(material1);
        const Material& m2 = getMaterial(material2);
        const double E1 = m1.youngModulus;
        const double E2 = m2.youngModulus;
        const double nu1 = m1.poissonRatio;
        const double nu2 = m2.poissonRatio;
        const double G1 = E1 / (2.0 * (1.0 + nu1));
        const double G2 = E2 / (2.0 * (1.0 + nu2));
        MaterialPairProperties pair;
        pair.effectiveYoungModulus = 1.0 / (
            (1.0 - nu1 * nu1) / E1 +
            (1.0 - nu2 * nu2) / E2
        );
        pair.effectiveShearModulus = 1.0 / (
            (2.0 - nu1) / G1 +
            (2.0 - nu2) / G2
        );
        constexpr double PI = 3.1415926535897932;
        const double ln = std::log(properties.restitution);
        pair.dampingRatio = -ln / std::sqrt(PI * PI + ln * ln);
        pair.contact = properties;
        const std::string key = makePairKey(material1, material2);
        contact_properties_[key] = properties;
        pair_properties_[key] = pair;
    }
}

const Material& MaterialDatabase::getMaterial(const std::string& name) const {
    const auto it = materials_.find(name);
    if (it == materials_.end()){
        throw std::runtime_error("Material not found: " + name);
    }
    return it->second;
}

const ContactProperties& MaterialDatabase::getContactProperties(const std::string& m1, const std::string& m2) const {
    const std::string key = makePairKey(m1, m2);
    const auto it = contact_properties_.find(key);
    if (it == contact_properties_.end()){
        throw std::runtime_error("Contact properties not found for materials: " + m1 + ", " + m2);
    }
    return it->second;
}

bool MaterialDatabase::hasMaterial(const std::string& name) const {
    return materials_.find(name) != materials_.end();
}

const MaterialPairProperties& MaterialDatabase::getPairProperties(const std::string& m1, const std::string& m2) const {
    const auto it = pair_properties_.find(makePairKey(m1,m2));
    if (it == pair_properties_.end()){
        throw std::runtime_error("Cached properties not found for: " + m1 + ", " + m2);
    }
    return it->second;
}
