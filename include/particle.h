#pragma once

struct Vec2D {
    double x;
    double y;
};

struct Particle {
    Vec2D position;
    Vec2D velocity;
    double mass;
    double radius;
};