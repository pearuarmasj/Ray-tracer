/**
 * @file stokes.hpp
 * @brief Stokes vector representation for polarized light
 *
 * Stokes vectors represent the polarization state of light using 4 components:
 * - I (S0): Total intensity
 * - Q (S1): Linear polarization (horizontal - vertical)
 * - U (S2): Linear polarization (+45 - -45 degrees)
 * - V (S3): Circular polarization (right - left handed)
 *
 * For physically valid light: I >= sqrt(Q^2 + U^2 + V^2)
 * Equality holds for fully polarized light.
 */

#pragma once

#include <cmath>
#include <algorithm>

namespace plt {

/**
 * Stokes vector representing polarization state
 */
struct Stokes {
    float I = 0.0f;  // Total intensity (S0)
    float Q = 0.0f;  // Horizontal - vertical polarization (S1)
    float U = 0.0f;  // +45 - -45 degree polarization (S2)
    float V = 0.0f;  // Right - left circular polarization (S3)

    Stokes() = default;
    Stokes(float i, float q = 0.0f, float u = 0.0f, float v = 0.0f)
        : I(i), Q(q), U(u), V(v) {}

    // Create unpolarized light of given intensity
    static Stokes unpolarized(float intensity) {
        return Stokes(intensity, 0.0f, 0.0f, 0.0f);
    }

    // Create horizontally polarized light
    static Stokes horizontal(float intensity) {
        return Stokes(intensity, intensity, 0.0f, 0.0f);
    }

    // Create vertically polarized light
    static Stokes vertical(float intensity) {
        return Stokes(intensity, -intensity, 0.0f, 0.0f);
    }

    // Create +45 degree linearly polarized light
    static Stokes diagonal_plus(float intensity) {
        return Stokes(intensity, 0.0f, intensity, 0.0f);
    }

    // Create -45 degree linearly polarized light
    static Stokes diagonal_minus(float intensity) {
        return Stokes(intensity, 0.0f, -intensity, 0.0f);
    }

    // Create right-hand circularly polarized light
    static Stokes circular_right(float intensity) {
        return Stokes(intensity, 0.0f, 0.0f, intensity);
    }

    // Create left-hand circularly polarized light
    static Stokes circular_left(float intensity) {
        return Stokes(intensity, 0.0f, 0.0f, -intensity);
    }

    // Degree of polarization (0 = unpolarized, 1 = fully polarized)
    float degree_of_polarization() const {
        if (I <= 0.0f) return 0.0f;
        float pol = std::sqrt(Q * Q + U * U + V * V);
        return std::min(pol / I, 1.0f);
    }

    // Degree of linear polarization
    float degree_of_linear_polarization() const {
        if (I <= 0.0f) return 0.0f;
        return std::min(std::sqrt(Q * Q + U * U) / I, 1.0f);
    }

    // Degree of circular polarization
    float degree_of_circular_polarization() const {
        if (I <= 0.0f) return 0.0f;
        return std::min(std::abs(V) / I, 1.0f);
    }

    // Angle of linear polarization in radians (relative to reference frame)
    float polarization_angle() const {
        return 0.5f * std::atan2(U, Q);
    }

    // Ellipticity angle (0 = linear, +/-pi/4 = circular)
    float ellipticity_angle() const {
        float dop = degree_of_polarization();
        if (dop <= 0.0f) return 0.0f;
        return 0.5f * std::asin(std::clamp(V / (I * dop), -1.0f, 1.0f));
    }

    // Check if physically valid (I >= polarized component)
    bool is_valid() const {
        if (I < 0.0f) return false;
        float pol2 = Q * Q + U * U + V * V;
        return pol2 <= I * I * (1.0f + 1e-6f);  // Small epsilon for numerical stability
    }

    // Clamp to physically valid state
    Stokes clamped() const {
        if (I <= 0.0f) return Stokes();
        float pol = std::sqrt(Q * Q + U * U + V * V);
        if (pol <= I) return *this;
        float scale = I / pol;
        return Stokes(I, Q * scale, U * scale, V * scale);
    }

    // Arithmetic operations
    Stokes operator+(const Stokes& other) const {
        return Stokes(I + other.I, Q + other.Q, U + other.U, V + other.V);
    }

    Stokes operator-(const Stokes& other) const {
        return Stokes(I - other.I, Q - other.Q, U - other.U, V - other.V);
    }

    Stokes operator*(float s) const {
        return Stokes(I * s, Q * s, U * s, V * s);
    }

    Stokes operator/(float s) const {
        float inv = 1.0f / s;
        return *this * inv;
    }

    Stokes& operator+=(const Stokes& other) {
        I += other.I; Q += other.Q; U += other.U; V += other.V;
        return *this;
    }

    Stokes& operator*=(float s) {
        I *= s; Q *= s; U *= s; V *= s;
        return *this;
    }

    // Array access
    float& operator[](int i) {
        switch (i) {
            case 0: return I;
            case 1: return Q;
            case 2: return U;
            default: return V;
        }
    }

    float operator[](int i) const {
        switch (i) {
            case 0: return I;
            case 1: return Q;
            case 2: return U;
            default: return V;
        }
    }
};

inline Stokes operator*(float s, const Stokes& stokes) {
    return stokes * s;
}

} // namespace plt
