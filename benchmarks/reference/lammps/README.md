# Hertz–Mindlin reference benchmarks

Reference solver: LAMMPS 30 Sep 2026.

These tests compare the classical C++ CPU implementation
with the selected LAMMPS contact-model configuration.

## Shared settings

- SI units
- Particle radius: 0.001 m
- Particle density: 1500 kg/m³
- Young's modulus: 1e6 Pa
- Poisson's ratio: 0.25
- Restitution parameter: 0.7
- Coulomb friction coefficient: 0.4
- Tangential damping coefficient equals normal damping coefficient
- Attractive total normal force prevented by damping clamp
- Rolling and twisting resistance disabled
- Gravity disabled
- Timestep: 5e-6 s
- Duration: 0.1 s

Exact LAMMPS settings and initial conditions are in the input files.

## Final reference values

Values below refer to LAMMPS particle ID 1.
It corresponds to particle ID 0 in the C++ simulator.

| Test | vx (m/s) | vy (m/s) | vz (m/s) | Relevant angular velocity (rad/s) |
|---|---|---|---|---|
| Head-on | -0.035925358437145255 | 0 | 0 | All zero |
| Oblique | -0.017265750489369795 | 0.031762483468390286 | 0 | omega_z = -16.327743892697153 |
| Flat wall | 0.015928456216630197 | 0 | 0.035920170065727425 | omega_y = 10.178859458424533 |

The C++ implementation matched these final values to
floating-point precision in the recorded comparisons.

The wall test uses identical particle and wall elastic material
properties, with an infinite-mass flat wall.

## Scope and limitations

These are selected regression cases, not exhaustive validation.

They do not establish reference agreement for:
- Rolling resistance
- Curved or rotating walls
- Every contact loading history
- Experimental powder-mixing behavior

The friction implementation uses one Coulomb coefficient.
The dynamicFriction input field is currently unused.

For the head-on case, measured restitution is approximately
0.7185 despite an input restitution parameter of 0.7.
Both implementations produced this result.