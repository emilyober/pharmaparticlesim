#pragma once
#include <cmath>

struct Vec3D {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vec3D operator-() const {
        return {-x, -y, -z};
    }
    Vec3D operator+(const Vec3D& a) const {
        return {x + a.x, y + a.y, z + a.z};
    }
    Vec3D operator-(const Vec3D& a) const {
        return {x - a.x, y - a.y, z - a.z};
    }
    Vec3D operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
    Vec3D operator/(double scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    Vec3D& operator+=(const Vec3D& a) {
        x += a.x;
        y += a.y;
        z += a.z;
        return *this;
    }
    Vec3D& operator-=(const Vec3D& a) {
        x -= a.x;
        y -= a.y;
        z -= a.z;
        return *this;
    }
    Vec3D& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    Vec3D& operator/=(double scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
};

inline double dot(const Vec3D& a, const Vec3D& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double norm2(const Vec3D& a) {
    return dot(a, a);
}

inline double norm(const Vec3D& a) {
    return std::sqrt(norm2(a));
}

inline Vec3D operator*(double scalar, const Vec3D& a) {
    return a * scalar;
}

inline Vec3D normalize(const Vec3D& a) {
    double n = norm(a);
    return (n>0.0) ? a / n : Vec3D{0.0, 0.0, 0.0};
}

inline Vec3D cross(const Vec3D& a, const Vec3D& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}



