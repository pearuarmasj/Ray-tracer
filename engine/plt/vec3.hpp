/**
 * @file vec3.hpp
 * @brief C++ Vec3 wrapper for PLT module
 * 
 * Provides a lightweight Vec3 class that interoperates with
 * the core C vec3 structure.
 */

#pragma once

extern "C" {
#include "../../core/vec3.h"
}

#include <cmath>

namespace plt {

/**
 * 3D vector for PLT calculations (uses float for efficiency)
 */
struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    // Construct from core vec3 (double to float)
    explicit Vec3(const vec3& v) : x(static_cast<float>(v.x)), 
                                    y(static_cast<float>(v.y)), 
                                    z(static_cast<float>(v.z)) {}

    // Convert to core vec3
    vec3 to_vec3() const {
        return vec3_create(static_cast<double>(x), 
                           static_cast<double>(y), 
                           static_cast<double>(z));
    }

    // Basic operations
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { float inv = 1.0f / s; return *this * inv; }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }

    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float length_squared() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(length_squared()); }

    Vec3 normalized() const {
        float len = length();
        if (len < 1e-10f) return Vec3(0, 0, 1);
        return *this / len;
    }

    // Array access
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

// Free functions
inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

inline Vec3 refract(const Vec3& v, const Vec3& n, float eta) {
    float cos_i = -dot(v, n);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t > 1.0f) return reflect(v, n);  // TIR
    float cos_t = std::sqrt(1.0f - sin2_t);
    return eta * v + (eta * cos_i - cos_t) * n;
}

// Convert core vec3 to plt::Vec3
inline Vec3 from_vec3(const vec3& v) {
    return Vec3(v);
}

} // namespace plt
