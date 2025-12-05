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

#include "texture.hpp"
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
    color albedo = {0.5, 0.5, 0.5};  // Base color (used if no texture)
    Texture texture;                  // Texture for the material
    bool has_texture = false;         // Whether to use texture instead of albedo
    double fuzz = 0.0;                // Metal roughness (0 = mirror)
    double refraction_index = 1.5;    // Index of refraction for dielectrics
    
    /**
     * @brief Get the color at a given point
     */
    color get_albedo(point3 p, double u = 0.0, double v = 0.0) const {
        if (has_texture) {
            return texture.sample(p, u, v);
        }
        return albedo;
    }
    
    /**
     * @brief Create a Lambertian (diffuse) material
     */
    static Material lambertian(color c) {
        Material m;
        m.type = MaterialType::Lambertian;
        m.albedo = c;
        m.has_texture = false;
        return m;
    }
    
    /**
     * @brief Create a Lambertian material with a texture
     */
    static Material lambertian_textured(Texture tex) {
        Material m;
        m.type = MaterialType::Lambertian;
        m.texture = tex;
        m.has_texture = true;
        return m;
    }
    
    /**
     * @brief Create a metal (reflective) material
     */
    static Material metal(color c, double fuzz_factor = 0.0) {
        Material m;
        m.type = MaterialType::Metal;
        m.albedo = c;
        m.fuzz = fuzz_factor < 1.0 ? fuzz_factor : 1.0;
        m.has_texture = false;
        return m;
    }
    
    /**
     * @brief Create a metal material with a texture
     */
    static Material metal_textured(Texture tex, double fuzz_factor = 0.0) {
        Material m;
        m.type = MaterialType::Metal;
        m.texture = tex;
        m.fuzz = fuzz_factor < 1.0 ? fuzz_factor : 1.0;
        m.has_texture = true;
        return m;
    }
    
    /**
     * @brief Create a dielectric (glass-like) material
     */
    static Material dielectric(double ir) {
        Material m;
        m.type = MaterialType::Dielectric;
        m.albedo = {1.0, 1.0, 1.0};
        m.refraction_index = ir;
        m.has_texture = false;
        return m;
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
