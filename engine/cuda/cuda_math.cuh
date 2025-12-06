/**
 * @file cuda_math.cuh
 * @brief GPU-compatible math primitives (vec3, ray, etc.)
 * 
 * These are device-side equivalents of the CPU math types.
 * Using __host__ __device__ for flexibility.
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

namespace cuda {

// ============================================================================
// Constants
// ============================================================================

__device__ __constant__ float PI = 3.14159265358979323846f;
__device__ __constant__ float INF = 1e20f;
__device__ __constant__ float EPSILON = 1e-4f;

// ============================================================================
// Vector3 (float precision for GPU efficiency)
// ============================================================================

struct Vec3 {
    float x, y, z;
    
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float v) : x(v), y(v), z(v) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    __host__ __device__ Vec3 operator/(float t) const { float inv = 1.0f / t; return Vec3(x * inv, y * inv, z * inv); }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    
    __host__ __device__ Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__ Vec3& operator*=(float t) { x *= t; y *= t; z *= t; return *this; }
    
    __host__ __device__ float length_squared() const { return x*x + y*y + z*z; }
    __host__ __device__ float length() const { return sqrtf(length_squared()); }
    
    __host__ __device__ bool near_zero() const {
        return (fabsf(x) < 1e-8f) && (fabsf(y) < 1e-8f) && (fabsf(z) < 1e-8f);
    }
};

// Free functions for Vec3
__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) { return v * t; }

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    float len = v.length();
    return len > 0 ? v / len : Vec3(0);
}

__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// Schlick's approximation for reflectance
__host__ __device__ inline float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

// Type aliases
using Point3 = Vec3;
using Color = Vec3;

// ============================================================================
// Ray
// ============================================================================

struct Ray {
    Point3 origin;
    Vec3 direction;
    
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point3& o, const Vec3& d) : origin(o), direction(d) {}
    
    __host__ __device__ Point3 at(float t) const { return origin + t * direction; }
};

// ============================================================================
// Random number generation (per-thread state)
// ============================================================================

struct RNG {
    curandState state;
    
    __device__ void init(unsigned long long seed, unsigned long long sequence) {
        curand_init(seed, sequence, 0, &state);
    }
    
    __device__ float uniform() {
        return curand_uniform(&state);
    }
    
    __device__ float uniform(float min, float max) {
        return min + (max - min) * uniform();
    }
    
    __device__ Vec3 in_unit_sphere() {
        while (true) {
            Vec3 p(uniform(-1, 1), uniform(-1, 1), uniform(-1, 1));
            if (p.length_squared() < 1.0f)
                return p;
        }
    }
    
    __device__ Vec3 unit_vector() {
        return normalize(in_unit_sphere());
    }
    
    __device__ Vec3 on_hemisphere(const Vec3& normal) {
        Vec3 on_sphere = unit_vector();
        return dot(on_sphere, normal) > 0.0f ? on_sphere : -on_sphere;
    }
    
    __device__ Vec3 in_unit_disk() {
        while (true) {
            Vec3 p(uniform(-1, 1), uniform(-1, 1), 0);
            if (p.length_squared() < 1.0f)
                return p;
        }
    }
    
    // Cosine-weighted hemisphere sampling (for diffuse)
    __device__ Vec3 cosine_direction() {
        float r1 = uniform();
        float r2 = uniform();
        float z = sqrtf(1.0f - r2);
        float phi = 2.0f * 3.14159265f * r1;
        float x = cosf(phi) * sqrtf(r2);
        float y = sinf(phi) * sqrtf(r2);
        return Vec3(x, y, z);
    }
};

// ============================================================================
// Orthonormal basis for local-to-world transforms
// ============================================================================

struct ONB {
    Vec3 u, v, w;
    
    __device__ void build_from_w(const Vec3& n) {
        w = normalize(n);
        Vec3 a = fabsf(w.x) > 0.9f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        v = normalize(cross(w, a));
        u = cross(w, v);
    }
    
    __device__ Vec3 local(const Vec3& a) const {
        return a.x * u + a.y * v + a.z * w;
    }
};

} // namespace cuda
