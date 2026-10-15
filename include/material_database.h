#pragma once

#include <unordered_map>
#include <string>
#include "material.h"

class MaterialDatabase {
    public:
        void loadMaterials(const std::string& filename);
        void loadContactProperties(const std::string& filename);
        const Material& getMaterial(const std::string& m) const;
        const ContactProperties& getContactProperties(const std::string& m1, const std::string& m2) const;
        bool hasMaterial(const std::string& m) const;
    private:
        static std::string makePairKey(const std::string& m1, const std::string& m2);
        std::unordered_map<std::string, Material> materials_;
        std::unordered_map<std::string, ContactProperties> contact_properties_;
};