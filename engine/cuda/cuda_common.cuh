/**
 * @file cuda_common.cuh
 * @brief Common CUDA utilities and device-compatible math
 */

#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace raytracer {
namespace cuda {

// Device-compatible vec3
struct float3_t {
    float x, y, z;
    
    __host__ __device__ float3_t() : x(0), y(0), z(0) {}
    __host__ __device__ float3_t(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Basic vec3 operations (device-compatible)
__host__ __device__ inline float3_t operator+(const float3_t& a, const float3_t& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ inline float3_t operator-(const float3_t& a, const float3_t& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ inline float3_t operator*(const float3_t& a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__host__ __device__ inline float3_t operator*(float s, const float3_t& a) {
    return {a.x * s, a.y * s, a.z * s};
}

__host__ __device__ inline float3_t operator*(const float3_t& a, const float3_t& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ inline float dot(const float3_t& a, const float3_t& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3_t cross(const float3_t& a, const float3_t& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ inline float length(const float3_t& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline float3_t normalize(const float3_t& v) {
    float len = length(v);
    return len > 0 ? v * (1.0f / len) : float3_t{0, 0, 0};
}

// Ray structure for GPU
struct Ray {
    float3_t origin;
    float3_t direction;
    
    __host__ __device__ float3_t at(float t) const {
        return origin + direction * t;
    }
};

// Hit record for GPU
struct HitRecord {
    float3_t point;
    float3_t normal;
    float t;
    int material_id;
    bool front_face;
    
    __host__ __device__ void set_face_normal(const Ray& r, const float3_t& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : outward_normal * -1.0f;
    }
};

// Sphere for GPU
struct Sphere {
    float3_t center;
    float radius;
    int material_id;
    
    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float3_t oc = r.origin - center;
        float a = dot(r.direction, r.direction);
        float half_b = dot(oc, r.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = half_b * half_b - a * c;
        
        if (discriminant < 0) return false;
        
        float sqrtd = sqrtf(discriminant);
        float root = (-half_b - sqrtd) / a;
        
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max) {
                return false;
            }
        }
        
        rec.t = root;
        rec.point = r.at(root);
        float3_t outward_normal = (rec.point - center) * (1.0f / radius);
        rec.set_face_normal(r, outward_normal);
        rec.material_id = material_id;
        
        return true;
    }
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

} // namespace cuda
} // namespace raytracer
