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

// Forward declare random functions (defined in renderer.cpp)
double random_double();
vec3 random_unit_vector();
vec3 random_in_hemisphere(vec3 normal);

constexpr double INV_PI = 0.31830988618379067;

/**
 * @brief Material types supported by the renderer
 */
enum class MaterialType {
    Lambertian,  // Diffuse material
    Metal,       // Reflective material
    Dielectric   // Transparent/refractive material
};

/**
 * @brief Scatter record containing scatter direction and PDF
 */
struct ScatterRecord {
    ray scattered;
    color attenuation;
    double pdf;
    bool is_specular;  // If true, skip NEE (delta distribution)
};

/**
 * @brief Material properties
 */
struct Material {
    MaterialType type = MaterialType::Lambertian;
    color albedo = {0.5, 0.5, 0.5};  // Base color (used if no texture)
    color emission = {0.0, 0.0, 0.0}; // Emissive color (for path tracing lights)
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
     * @brief Check if material is emissive
     */
    bool is_emissive() const {
        return emission.x > 0.0 || emission.y > 0.0 || emission.z > 0.0;
    }
    
    /**
     * @brief Check if material has delta distribution (specular)
     */
    bool is_specular() const {
        return type == MaterialType::Metal && fuzz < 0.001;
    }
    
    /**
     * @brief Scatter a ray and return scatter record
     * @param r_in Incoming ray
     * @param rec Hit record
     * @param srec Output scatter record
     * @return true if ray scatters, false if absorbed
     */
    bool scatter(ray r_in, const hit_record& rec, ScatterRecord& srec) const {
        srec.attenuation = get_albedo(rec.point, rec.u, rec.v);
        
        switch (type) {
            case MaterialType::Lambertian: {
                // Cosine-weighted hemisphere sampling
                vec3 scatter_direction = vec3_add(rec.normal, random_unit_vector());
                if (vec3_length_squared(scatter_direction) < 1e-8) {
                    scatter_direction = rec.normal;
                }
                scatter_direction = vec3_normalize(scatter_direction);
                
                srec.scattered = ray_create(rec.point, scatter_direction);
                srec.pdf = vec3_dot(rec.normal, scatter_direction) * INV_PI;
                srec.is_specular = false;
                return true;
            }
            
            case MaterialType::Metal: {
                vec3 reflected = vec3_reflect(vec3_normalize(r_in.direction), rec.normal);
                
                if (fuzz > 0.0) {
                    vec3 fuzz_vec = vec3_scale(random_unit_vector(), fuzz);
                    reflected = vec3_normalize(vec3_add(reflected, fuzz_vec));
                }
                
                srec.scattered = ray_create(rec.point, reflected);
                srec.pdf = 1.0;  // Delta distribution for perfect mirror
                srec.is_specular = (fuzz < 0.001);
                return vec3_dot(reflected, rec.normal) > 0;
            }
            
            case MaterialType::Dielectric: {
                srec.attenuation = {1.0, 1.0, 1.0};
                srec.is_specular = true;
                srec.pdf = 1.0;
                
                double refraction_ratio = rec.front_face ? 
                    (1.0 / refraction_index) : refraction_index;
                
                vec3 unit_direction = vec3_normalize(r_in.direction);
                double cos_theta = std::fmin(vec3_dot(vec3_negate(unit_direction), rec.normal), 1.0);
                double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
                
                bool cannot_refract = refraction_ratio * sin_theta > 1.0;
                vec3 direction;
                
                if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) {
                    direction = vec3_reflect(unit_direction, rec.normal);
                } else {
                    direction = vec3_refract(unit_direction, rec.normal, refraction_ratio);
                }
                
                srec.scattered = ray_create(rec.point, direction);
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Evaluate BSDF PDF for a given scatter direction
     * @param r_in Incoming ray
     * @param rec Hit record  
     * @param scattered Scattered ray direction
     * @return PDF value for the given direction
     */
    double scattering_pdf(ray r_in, const hit_record& rec, ray scattered) const {
        switch (type) {
            case MaterialType::Lambertian: {
                double cosine = vec3_dot(rec.normal, vec3_normalize(scattered.direction));
                return cosine < 0 ? 0 : cosine * INV_PI;
            }
            case MaterialType::Metal:
            case MaterialType::Dielectric:
                // Delta distribution - PDF is technically infinite at the reflection direction
                return 0.0;
        }
        return 0.0;
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
     * @brief Create an emissive (light) material
     */
    static Material emissive(color emit, color c = {0.0, 0.0, 0.0}) {
        Material m;
        m.type = MaterialType::Lambertian;
        m.albedo = c;
        m.emission = emit;
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
