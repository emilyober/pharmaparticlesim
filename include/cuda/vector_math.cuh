#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "vec3d.h"

namespace cuda_dem{
    __device__ inline Vec3D vadd(Vec3D a, Vec3D b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
    __device__ inline Vec3D vsub(Vec3D a, Vec3D b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    __device__ inline Vec3D vmul(Vec3D a, double s) {
        return {a.x * s, a.y * s, a.z * s};
    }
    __device__ inline Vec3D vdiv(Vec3D a, double s) {
        return {a.x / s, a.y / s, a.z / s};
    }
    __device__ inline double vdot(Vec3D a, Vec3D b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    __device__ inline double vnorm(Vec3D a) {
        return sqrt(vdot(a, a));
    }
    __device__ inline Vec3D vcross(Vec3D a, Vec3D b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }
}