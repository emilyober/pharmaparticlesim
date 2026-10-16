# Hertz–Mindlin drum configuration

Backend: classical CPU, single-threaded.
Build: CMake Release.

## Current configuration

- Drum inside radius: 0.02 m
- Drum length: 0.02 m
- Drum and end-cap rotation: 10 RPM
- Particle radius: 0.001 m
- Particle count: 322
- Initial spacing: 0.0022 m
- Two labeled groups with identical physical properties
- Initial linear and angular velocities: zero
- Gravity: (0, 0, -9.81) m/s²
- Timestep: 5e-6 s
- Duration: 1.0 s
- Output interval: 200 steps
- Rolling resistance: disabled
- Rotation starts immediately; no separate settling stage

Material and pair properties are in the accompanying CSV files.

## Implementation

- Hertz normal elasticity and restitution-based damping
- Mindlin tangential displacement history
- Single Coulomb friction coefficient
- Tangential damping equals normal damping
- Semi-implicit Euler integration
- Cell-based neighbor search

The dynamicFriction CSV field is currently unused.

Cylinder contact stiffness uses a local flat-wall approximation.

## Reproduction

The current program reads materials.csv and contacts.csv from its working directory. It does not yet read this folder automatically.

Geometry and initialization are defined in src/main.cpp.
Time settings are defined in include/simulator.h.
The baseline Git commit identifies the corresponding source version.