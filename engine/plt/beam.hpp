/**
 * @file beam.hpp
 * @brief Polarized light beam carrying Stokes vector and reference frame
 *
 * A PLT beam carries:
 * - Stokes vector (polarization state)
 * - Direction of propagation
 * - Reference frame tangent (defines s-polarization direction)
 * - Optional coherence information for interference
 */

#pragma once

#include "stokes.hpp"
#include "mueller.hpp"
#include "vec3.hpp"
#include <cmath>

namespace plt {

/**
 * Polarized light beam with reference frame
 */
struct Beam {
    Stokes stokes;      // Polarization state
    Vec3 direction;     // Direction of propagation
    Vec3 tangent;       // Reference frame tangent (s-polarization direction)

    Beam() = default;

    Beam(const Stokes& s, const Vec3& dir, const Vec3& tan)
        : stokes(s), direction(dir.normalized()), tangent(tan.normalized()) {}

    // Create unpolarized beam
    static Beam unpolarized(float intensity, const Vec3& dir) {
        Vec3 tan = perpendicular(dir);
        return Beam(Stokes::unpolarized(intensity), dir, tan);
    }

    // Create horizontally polarized beam (relative to tangent)
    static Beam horizontal(float intensity, const Vec3& dir, const Vec3& tan) {
        return Beam(Stokes::horizontal(intensity), dir, tan.normalized());
    }

    // Compute perpendicular vector to given direction
    static Vec3 perpendicular(const Vec3& v) {
        Vec3 vn = v.normalized();
        // Choose axis least aligned with v
        Vec3 up = (std::abs(vn.y) < 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        return cross(up, vn).normalized();
    }

    // Get the bitangent (p-polarization direction)
    Vec3 bitangent() const {
        return cross(tangent, direction).normalized();
    }

    // Scale intensity
    void scale(float s) {
        stokes *= s;
    }

    // Apply Mueller matrix (in current reference frame)
    void apply(const Mueller& m) {
        stokes = m * stokes;
    }

    /**
     * Rotate reference frame to align tangent with new direction
     * Also applies corresponding Mueller rotation to Stokes vector
     * @param new_tangent New tangent direction (will be normalized)
     */
    void rotate_frame(const Vec3& new_tangent) {
        Vec3 new_tan = new_tangent.normalized();
        
        // Project new_tangent onto plane perpendicular to direction
        float proj = dot(new_tan, direction);
        new_tan = (new_tan - proj * direction).normalized();
        
        // Compute rotation angle
        // cos(theta) = old_tangent . new_tangent
        // sin(theta) = (old_tangent x new_tangent) . direction
        float cos_theta = std::clamp(dot(tangent, new_tan), -1.0f, 1.0f);
        Vec3 cross_prod = cross(tangent, new_tan);
        float sin_theta = dot(cross_prod, direction);
        
        // Apply Mueller rotation (uses 2*theta)
        float c2 = cos_theta * cos_theta - sin_theta * sin_theta;
        float s2 = 2.0f * sin_theta * cos_theta;
        
        Mueller rot(
            1, 0, 0, 0,
            0, c2, s2, 0,
            0, -s2, c2, 0,
            0, 0, 0, 1
        );
        
        stokes = rot * stokes;
        tangent = new_tan;
    }

    /**
     * Rotate frame to align with plane of incidence at surface
     * @param normal Surface normal
     */
    void align_to_plane_of_incidence(const Vec3& normal) {
        // s-polarization direction = normal x direction (normalized)
        Vec3 s = cross(normal, direction);
        float s_len = s.length();
        
        if (s_len < 1e-8f) {
            // Normal incidence - frame is arbitrary, keep current
            return;
        }
        
        s = s / s_len;
        rotate_frame(s);
    }

    /**
     * Update direction after reflection/refraction
     * Maintains frame consistency
     * @param new_direction New propagation direction
     * @param normal Surface normal (for frame update)
     */
    void set_direction(const Vec3& new_direction, const Vec3& normal) {
        direction = new_direction.normalized();
        
        // s-direction stays perpendicular to plane of incidence
        // Recompute to ensure orthogonality
        Vec3 s = cross(normal, direction);
        float s_len = s.length();
        
        if (s_len > 1e-8f) {
            tangent = s / s_len;
        } else {
            // Along normal - pick arbitrary perpendicular
            tangent = perpendicular(direction);
        }
    }

    // Transform vector from world to beam-local coordinates
    Vec3 to_local(const Vec3& v) const {
        Vec3 b = bitangent();
        return Vec3(dot(v, tangent), dot(v, b), dot(v, direction));
    }

    // Transform vector from beam-local to world coordinates
    Vec3 from_local(const Vec3& v) const {
        Vec3 b = bitangent();
        return v.x * tangent + v.y * b + v.z * direction;
    }
};

} // namespace plt
