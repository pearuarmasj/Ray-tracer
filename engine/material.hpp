/**
 * @file material.hpp
 * @brief Material definitions for ray tracing
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/hit.h"
}

#include <cmath>

namespace raytracer {

/**
 * @brief Material types supported by the renderer
 */
enum class MaterialType {
    Lambertian,  // Diffuse material
    Metal,       // Reflective material
    Dielectric   // Transparent/refractive material
};

/**
 * @brief Material properties
 */
struct Material {
    MaterialType type = MaterialType::Lambertian;
    color albedo = {0.5, 0.5, 0.5};  // Base color
    double fuzz = 0.0;                // Metal roughness (0 = mirror)
    double refraction_index = 1.5;    // Index of refraction for dielectrics
    
    /**
     * @brief Create a Lambertian (diffuse) material
     */
    static Material lambertian(color c) {
        return Material{MaterialType::Lambertian, c, 0.0, 1.5};
    }
    
    /**
     * @brief Create a metal (reflective) material
     */
    static Material metal(color c, double fuzz_factor = 0.0) {
        return Material{MaterialType::Metal, c, fuzz_factor < 1.0 ? fuzz_factor : 1.0, 1.5};
    }
    
    /**
     * @brief Create a dielectric (glass-like) material
     */
    static Material dielectric(double ir) {
        return Material{MaterialType::Dielectric, {1.0, 1.0, 1.0}, 0.0, ir};
    }
    
    /**
     * @brief Schlick's approximation for reflectance
     */
    static double reflectance(double cosine, double ref_idx) {
        auto r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0 - r0) * std::pow((1.0 - cosine), 5.0);
    }
};

} // namespace raytracer
