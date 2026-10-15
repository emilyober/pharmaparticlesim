#pragma once

#include "particle.h"
#include "material_database.h"
#include "contact_state.h"

struct PlaneWall;
struct CylindricalWall;

class ContactModel {
    public:
        static void resolveContact(
            Particle& p1, 
            Particle& p2, 
            const MaterialDatabase& materials,
            ContactState& contactState,
            double dt
        );
        static void particleWall(
            Particle& p, 
            const PlaneWall& wall,
            const MaterialDatabase& materials,
            ContactState& contactState,
            double dt
        );
        static void particleCylinder(
            Particle& particle,
            const CylindricalWall& wall,
            const MaterialDatabase& materials,
            ContactState& state,
            double dt
        );
};