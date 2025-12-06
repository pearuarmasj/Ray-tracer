/**
 * @file plt_integrator.hpp
 * @brief Polarized Light Tracing integrator
 *
 * Extends path tracing to track polarization state using Stokes vectors
 * and Mueller matrices. This file provides:
 * - PLTMaterialProps: Polarization properties for materials
 * - PLTIntegrator: Core polarized path tracing logic (scene-independent)
 * 
 * Integration with the main renderer is done separately.
 */

#pragma once

// Define M_PI for MSVC compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "stokes.hpp"
#include "mueller.hpp"
#include "fresnel.hpp"
#include "beam.hpp"
#include "vec3.hpp"

extern "C" {
#include "../../core/vec3.h"
#include "../../core/ray.h"
#include "../../core/hit.h"
}

#include <cmath>
#include <random>
#include <algorithm>

namespace plt {

// ============================================================================
// PLT Material Extension
// ============================================================================

/**
 * Polarization properties for materials
 */
struct PLTMaterialProps {
    bool is_polarizer = false;
    float polarizer_angle = 0.0f;      // Radians
    float polarizer_efficiency = 1.0f; // 1 = ideal polarizer
    
    bool is_wave_plate = false;
    enum class WavePlateType { None, Quarter, Half } wave_plate = WavePlateType::None;
    float wave_plate_angle = 0.0f;
    
    // Metal extinction coefficient (for polarized conductor Fresnel)
    float extinction_k = 0.0f;
    
    // Thin film coating (future)
    bool has_thin_film = false;
    float film_thickness_nm = 0.0f;
    float film_ior = 1.5f;
    
    static PLTMaterialProps polarizer(float angle, float efficiency = 1.0f) {
        PLTMaterialProps p;
        p.is_polarizer = true;
        p.polarizer_angle = angle;
        p.polarizer_efficiency = efficiency;
        return p;
    }
    
    static PLTMaterialProps quarter_wave(float angle) {
        PLTMaterialProps p;
        p.is_wave_plate = true;
        p.wave_plate = WavePlateType::Quarter;
        p.wave_plate_angle = angle;
        return p;
    }
    
    static PLTMaterialProps half_wave(float angle) {
        PLTMaterialProps p;
        p.is_wave_plate = true;
        p.wave_plate = WavePlateType::Half;
        p.wave_plate_angle = angle;
        return p;
    }
    
    static PLTMaterialProps conductor(float k) {
        PLTMaterialProps p;
        p.extinction_k = k;
        return p;
    }
    
    static PLTMaterialProps thin_film(float thickness_nm, float ior = 1.5f) {
        PLTMaterialProps p;
        p.has_thin_film = true;
        p.film_thickness_nm = thickness_nm;
        p.film_ior = ior;
        return p;
    }
};

// ============================================================================
// PLT Scatter Functions (Scene-Independent)
// ============================================================================

/**
 * Material type for PLT scattering decisions
 */
enum class PLTMaterialType {
    Diffuse,
    Metal,
    Dielectric
};

/**
 * Apply linear polarizer Mueller matrix to beam
 * @param beam Beam to modify
 * @param angle Polarizer angle in radians
 * @param efficiency Polarizer efficiency (0-1)
 * @param normal Surface normal for frame alignment
 */
inline void apply_polarizer(Beam& beam, float angle, float efficiency, const Vec3& normal) {
    beam.align_to_plane_of_incidence(normal);
    Mueller pol = Mueller::linear_polarizer(angle, efficiency);
    beam.apply(pol);
}

/**
 * Apply wave plate Mueller matrix to beam
 * @param beam Beam to modify
 * @param type Wave plate type (quarter or half)
 * @param angle Fast axis angle in radians
 * @param normal Surface normal for frame alignment
 */
inline void apply_wave_plate(
    Beam& beam,
    PLTMaterialProps::WavePlateType type,
    float angle,
    const Vec3& normal
) {
    beam.align_to_plane_of_incidence(normal);
    Mueller wp;
    switch (type) {
        case PLTMaterialProps::WavePlateType::Quarter:
            wp = Mueller::quarter_wave_plate(angle);
            break;
        case PLTMaterialProps::WavePlateType::Half:
            wp = Mueller::half_wave_plate(angle);
            break;
        default:
            return;
    }
    beam.apply(wp);
}

/**
 * Scatter at diffuse surface (depolarizes)
 * @param beam Beam to modify
 * @param wo_local Output direction in local coords
 * @param albedo Material albedo (grayscale)
 * @param normal Surface normal
 * @param rand1, rand2 Random values [0,1]
 */
inline void scatter_diffuse(
    Beam& beam,
    Vec3& wo,
    float albedo,
    const Vec3& normal,
    float rand1,
    float rand2
) {
    // Cosine-weighted hemisphere sampling
    float phi = 2.0f * static_cast<float>(M_PI) * rand1;
    float cos_theta = std::sqrt(rand2);
    float sin_theta = std::sqrt(1.0f - rand2);
    
    // Build orthonormal basis
    Vec3 up = (std::abs(normal.y) < 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 u = cross(up, normal).normalized();
    Vec3 v = cross(normal, u);
    
    // Transform to world space
    wo = (std::cos(phi) * sin_theta * u + std::sin(phi) * sin_theta * v + cos_theta * normal).normalized();
    
    // Diffuse scattering depolarizes completely
    beam.stokes = Stokes::unpolarized(beam.stokes.I * albedo);
    beam.set_direction(wo, normal);
}

/**
 * Scatter at metal surface with polarized Fresnel
 * @param beam Beam to modify
 * @param wo Output direction
 * @param wi Incident direction (toward surface)
 * @param normal Surface normal
 * @param eta Complex refractive index real part
 * @param k Extinction coefficient
 * @param albedo Metal tint
 * @param fuzz Roughness (0 = perfect mirror)
 * @param rand_vec Random vector for fuzz
 */
inline bool scatter_metal(
    Beam& beam,
    Vec3& wo,
    const Vec3& wi,
    const Vec3& normal,
    float eta,
    float k,
    float albedo,
    float fuzz,
    const Vec3& rand_vec
) {
    beam.align_to_plane_of_incidence(normal);
    
    // Reflect
    float cos_i = std::abs(dot(wi, normal));
    Vec3 reflected = (wi - 2.0f * dot(wi, normal) * normal).normalized();
    
    // Add fuzz
    if (fuzz > 0.0f) {
        reflected = (reflected + fuzz * rand_vec).normalized();
    }
    
    // Compute polarized Fresnel
    FresnelResult fresnel = fresnel_conductor(cos_i, eta, k);
    
    // Apply Mueller matrix
    Mueller M = fresnel.reflection_mueller();
    beam.apply(M);
    beam.scale(albedo);
    
    // Update beam
    wo = reflected;
    beam.set_direction(wo, normal);
    
    return (dot(wo, normal) > 0);
}

/**
 * Scatter at dielectric surface with polarized Fresnel
 * @param beam Beam to modify
 * @param wo Output direction
 * @param wi Incident direction (toward surface)
 * @param normal Surface normal (always toward incident medium)
 * @param eta Refractive index
 * @param entering True if entering denser medium
 * @param rand Random value [0,1] for reflection/refraction choice
 * @return true if scattered, false if absorbed
 */
inline bool scatter_dielectric(
    Beam& beam,
    Vec3& wo,
    const Vec3& wi,
    const Vec3& normal,
    float eta,
    bool entering,
    float rand
) {
    beam.align_to_plane_of_incidence(normal);
    
    float cos_i = std::abs(dot(wi, normal));
    float eta_ratio = entering ? (1.0f / eta) : eta;
    float fresnel_eta = entering ? eta : (1.0f / eta);
    
    // Compute polarized Fresnel
    FresnelResult fresnel = fresnel_dielectric(cos_i, fresnel_eta);
    
    // Choose reflection or refraction
    float R = fresnel.R();
    bool will_reflect = (rand < R) || (fresnel.cos_t == 0.0f);
    
    if (will_reflect) {
        // Reflect
        Vec3 reflected = (wi - 2.0f * dot(wi, normal) * normal).normalized();
        wo = reflected;
        
        Mueller M = fresnel.reflection_mueller();
        beam.apply(M);
        
        // PDF correction
        if (fresnel.cos_t > 0.0f) {
            beam.scale(1.0f / R);
        }
    } else {
        // Refract using Snell's law
        float sin_i2 = 1.0f - cos_i * cos_i;
        float sin_t2 = eta_ratio * eta_ratio * sin_i2;
        float cos_t = std::sqrt(1.0f - sin_t2);
        
        // wi points toward surface, so negate for refraction formula
        Vec3 refracted = (eta_ratio * wi + (eta_ratio * cos_i - cos_t) * normal).normalized();
        wo = refracted;
        
        Mueller M = fresnel.transmission_mueller(eta_ratio, cos_i);
        beam.apply(M);
        
        // PDF correction
        beam.scale(1.0f / (1.0f - R));
    }
    
    beam.set_direction(wo, normal);
    return true;
}

// ============================================================================
// Thin-Film Scatter Functions
// ============================================================================

/**
 * Scatter at thin-film coated metal surface
 * Produces wavelength-dependent iridescent reflections
 * 
 * @param beam Beam to modify
 * @param wo Output direction
 * @param wi Incident direction (toward surface)
 * @param normal Surface normal
 * @param eta Metal refractive index (real part)
 * @param k Metal extinction coefficient
 * @param film_ior Refractive index of thin film coating
 * @param film_thickness_nm Film thickness in nanometers
 * @param wavelength_nm Wavelength of light in nanometers
 * @param albedo Metal tint/reflectivity
 * @param fuzz Surface roughness
 * @param rand_vec Random vector for fuzz
 */
inline bool scatter_thin_film_metal(
    Beam& beam,
    Vec3& wo,
    const Vec3& wi,
    const Vec3& normal,
    float eta,
    float k,
    float film_ior,
    float film_thickness_nm,
    float wavelength_nm,
    float albedo,
    float fuzz,
    const Vec3& rand_vec
) {
    beam.align_to_plane_of_incidence(normal);
    
    float cos_i = std::abs(dot(wi, normal));
    Vec3 reflected = (wi - 2.0f * dot(wi, normal) * normal).normalized();
    
    if (fuzz > 0.0f) {
        reflected = (reflected + fuzz * rand_vec).normalized();
    }
    
    // Apply thin-film Mueller matrix
    Mueller M = thin_film_on_metal(cos_i, film_ior, eta, k, film_thickness_nm, wavelength_nm);
    beam.apply(M);
    beam.scale(albedo);
    
    wo = reflected;
    beam.set_direction(wo, normal);
    
    return (dot(wo, normal) > 0);
}

/**
 * Scatter at thin-film coated dielectric surface (soap bubble, coated glass)
 * 
 * @param beam Beam to modify
 * @param wo Output direction
 * @param wi Incident direction
 * @param normal Surface normal
 * @param substrate_ior Refractive index of substrate
 * @param film_ior Refractive index of film
 * @param film_thickness_nm Film thickness in nanometers
 * @param wavelength_nm Wavelength in nanometers
 * @param rand Random value for reflection/transmission choice
 */
inline bool scatter_thin_film_dielectric(
    Beam& beam,
    Vec3& wo,
    const Vec3& wi,
    const Vec3& normal,
    float substrate_ior,
    float film_ior,
    float film_thickness_nm,
    float wavelength_nm,
    float rand
) {
    beam.align_to_plane_of_incidence(normal);
    
    float cos_i = std::abs(dot(wi, normal));
    
    // Get thin-film reflection Mueller matrix
    Mueller M_reflect = thin_film_reflection(cos_i, film_ior, substrate_ior, 
                                              film_thickness_nm, wavelength_nm);
    
    // Estimate total reflectance from Mueller matrix (m00 = average reflectance)
    float R = M_reflect(0, 0);
    
    // Check for TIR
    float eta_ratio = 1.0f / substrate_ior;  // Assuming entering from air
    float sin_i = std::sqrt(1.0f - cos_i * cos_i);
    bool tir = (eta_ratio * sin_i > 1.0f);
    
    bool will_reflect = tir || (rand < R);
    
    if (will_reflect) {
        Vec3 reflected = (wi - 2.0f * dot(wi, normal) * normal).normalized();
        wo = reflected;
        beam.apply(M_reflect);
        
        if (!tir && R > 0.0f) {
            beam.scale(1.0f / R);  // PDF correction
        }
    } else {
        // Transmission (simplified - ignores film for transmission)
        float sin_t2 = eta_ratio * eta_ratio * sin_i * sin_i;
        float cos_t = std::sqrt(1.0f - sin_t2);
        
        Vec3 refracted = (eta_ratio * wi + (eta_ratio * cos_i - cos_t) * normal).normalized();
        wo = refracted;
        
        // For transmission, subtract reflected energy
        float T = 1.0f - R;
        beam.scale(T / (1.0f - R));  // PDF correction cancels with transmission
    }
    
    beam.set_direction(wo, normal);
    return true;
}

// ============================================================================
// Utility: Convert Stokes to RGB
// ============================================================================

/**
 * Convert Stokes intensity to grayscale RGB
 */
inline vec3 stokes_to_rgb(const Stokes& s) {
    float i = std::max(0.0f, s.I);
    return vec3_create(i, i, i);
}

/**
 * Convert Stokes to RGB with polarization visualization
 * Uses degree of polarization to tint the color
 */
inline vec3 stokes_to_rgb_polarized(const Stokes& s) {
    float intensity = std::max(0.0f, s.I);
    float dop = s.degree_of_polarization();
    float angle = s.polarization_angle();
    
    // Map polarization angle to hue
    float hue = (angle + static_cast<float>(M_PI) / 2.0f) / static_cast<float>(M_PI);
    
    // Mix between grayscale (unpolarized) and hue-colored (polarized)
    float r = intensity * (1.0f - dop + dop * (0.5f + 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * hue)));
    float g = intensity * (1.0f - dop + dop * (0.5f + 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * (hue - 1.0f/3.0f))));
    float b = intensity * (1.0f - dop + dop * (0.5f + 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * (hue - 2.0f/3.0f))));
    
    return vec3_create(r, g, b);
}

} // namespace plt
