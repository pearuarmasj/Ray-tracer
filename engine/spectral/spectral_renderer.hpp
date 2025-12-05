/**
 * @file spectral_renderer.hpp
 * @brief Spectral path tracing integrator
 * 
 * Extends the path tracer to handle wavelength-dependent light transport.
 * Key features:
 * - Correlated multi-wavelength sampling (reduces chromatic noise)
 * - Wavelength-dependent refraction (dispersion/rainbows)
 * - Spectral accumulation and RGB conversion
 * 
 * Design: Opt-in spectral rendering that coexists with RGB mode.
 */

#pragma once

#include "spectrum.hpp"
#include "dispersion.hpp"
#include "spectral_data.hpp"
#include "../scene.hpp"
#include "../material.hpp"

extern "C" {
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/hit.h"
}

#include <cmath>
#include <random>

namespace raytracer {

// Forward declarations from renderer.cpp
double random_double();
double random_double(double min, double max);
vec3 random_unit_vector();
vec3 random_in_unit_sphere();

namespace spectral {

// ============================================================================
// Seeded Random Number Generator for Correlated Sampling
// ============================================================================

/**
 * @brief Seeded RNG for correlated multi-wavelength sampling
 * 
 * This is CRITICAL for reducing chromatic noise. By using the same seed
 * for all wavelengths in a pixel sample, we ensure they follow the same
 * random path - only wavelength-dependent properties (IOR, Fresnel) differ.
 */
class SeededRNG {
public:
    SeededRNG() : gen_(42), dist_(0.0, 1.0) {}
    
    void seed(uint64_t s) {
        gen_.seed(s);
    }
    
    double uniform() {
        return dist_(gen_);
    }
    
    double uniform(double min, double max) {
        return min + (max - min) * uniform();
    }
    
    vec3 unit_vector() {
        double z = uniform(-1.0, 1.0);
        double phi = uniform(0.0, 2.0 * 3.14159265358979323846);
        double r = std::sqrt(1.0 - z * z);
        return {r * std::cos(phi), r * std::sin(phi), z};
    }
    
    vec3 in_unit_sphere() {
        while (true) {
            vec3 p = {uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)};
            if (vec3_length_squared(p) < 1) return p;
        }
    }
    
    vec3 cosine_hemisphere(const vec3& normal) {
        // Cosine-weighted hemisphere sampling
        double r1 = uniform();
        double r2 = uniform();
        double phi = 2.0 * 3.14159265358979323846 * r1;
        double cos_theta = std::sqrt(r2);
        double sin_theta = std::sqrt(1.0 - r2);
        
        // Build orthonormal basis
        vec3 w = normal;
        vec3 a = (std::fabs(w.x) > 0.9) ? vec3{0, 1, 0} : vec3{1, 0, 0};
        vec3 u = vec3_normalize(vec3_cross(a, w));
        vec3 v = vec3_cross(w, u);
        
        // Transform to world space
        return vec3_normalize(vec3_add(
            vec3_add(vec3_scale(u, std::cos(phi) * sin_theta),
                     vec3_scale(v, std::sin(phi) * sin_theta)),
            vec3_scale(w, cos_theta)));
    }
    
private:
    std::mt19937_64 gen_;
    std::uniform_real_distribution<double> dist_;
};

// ============================================================================
// Spectral Material Properties
// ============================================================================

/**
 * @brief Spectral extension for materials
 * 
 * Stores wavelength-dependent properties that extend the base Material.
 */
struct SpectralMaterial {
    Dispersion dispersion;           // Wavelength-dependent IOR
    data::RGBSpectrum albedo_spectrum;  // Spectral reflectance
    
    // Metal properties
    bool is_spectral_metal = false;
    enum class MetalType { Custom, Gold, Silver, Copper, Aluminum } metal_type = MetalType::Custom;
    
    SpectralMaterial() = default;
    
    /**
     * @brief Create from existing Material with dispersion
     */
    static SpectralMaterial from_material(const Material& mat, const Dispersion& disp) {
        SpectralMaterial sm;
        sm.dispersion = disp;
        sm.albedo_spectrum = data::RGBSpectrum(mat.albedo.x, mat.albedo.y, mat.albedo.z);
        return sm;
    }
    
    /**
     * @brief Create dispersive glass material
     */
    static SpectralMaterial glass(const Dispersion& disp) {
        SpectralMaterial sm;
        sm.dispersion = disp;
        sm.albedo_spectrum = data::RGBSpectrum(1.0, 1.0, 1.0);
        return sm;
    }
    
    /**
     * @brief Create spectral metal (gold, silver, etc.)
     */
    static SpectralMaterial metal(MetalType type) {
        SpectralMaterial sm;
        sm.is_spectral_metal = true;
        sm.metal_type = type;
        return sm;
    }
    
    /**
     * @brief Get refractive index at specific wavelength
     */
    double ior_at(double lambda) const {
        return dispersion.n(lambda);
    }
    
    /**
     * @brief Get albedo at specific wavelength
     */
    double albedo_at(double lambda) const {
        return albedo_spectrum.evaluate(lambda);
    }
    
    /**
     * @brief Get metal Fresnel reflectance at wavelength
     */
    double metal_fresnel(double cos_theta, double lambda) const {
        if (!is_spectral_metal) return 1.0;
        
        data::ComplexIOR ior;
        switch (metal_type) {
            case MetalType::Gold:     ior = data::gold_ior(lambda); break;
            case MetalType::Silver:   ior = data::silver_ior(lambda); break;
            case MetalType::Copper:   ior = data::copper_ior(lambda); break;
            case MetalType::Aluminum: ior = data::aluminum_ior(lambda); break;
            default: return 1.0;
        }
        
        return data::fresnel_conductor(cos_theta, ior);
    }
};

// ============================================================================
// Spectral Ray State
// ============================================================================

/**
 * @brief Ray state for spectral path tracing
 * 
 * Tracks wavelength and accumulated spectral radiance along a path.
 */
struct SpectralRay {
    ray r;                          // Geometric ray
    double lambda;                  // Wavelength (nm)
    double throughput = 1.0;        // Accumulated throughput at this wavelength
    double radiance = 0.0;          // Accumulated radiance
    
    SpectralRay() : lambda(wavelengths::D_LINE) {
        r = ray_create({0,0,0}, {0,0,1});
    }
    
    SpectralRay(ray r_, double lambda_) : r(r_), lambda(lambda_) {}
};

// ============================================================================
// Spectral Path Tracer
// ============================================================================

/**
 * @brief Spectral path tracing integrator
 * 
 * Traces paths at specific wavelengths to capture dispersion effects.
 */
class SpectralIntegrator {
public:
    struct Settings {
        int max_depth = 50;
        bool use_hwss = false;      // Hero Wavelength Spectral Sampling
        double russian_roulette_depth = 5;
    };
    
    explicit SpectralIntegrator(const Settings& settings = Settings())
        : settings_(settings) {}
    
    /**
     * @brief Sample direct lighting with pre-drawn random numbers
     * 
     * Uses fixed random values for quad light sampling to maintain
     * wavelength correlation.
     */
    double sample_direct_light_fixed(const Scene& scene, const hit_record& rec, 
                                     const Material& mat, double lambda,
                                     double u1, double u2) const {
        double direct = 0.0;
        
        // Only diffuse materials receive direct lighting this way
        if (mat.type != MaterialType::Lambertian) {
            return 0.0;
        }
        
        // Sample point lights (deterministic - no RNG needed)
        for (const auto& light : scene.lights) {
            vec3 to_light = vec3_sub(light.position, rec.point);
            double dist_sq = vec3_length_squared(to_light);
            double dist = std::sqrt(dist_sq);
            vec3 light_dir = vec3_scale(to_light, 1.0 / dist);
            
            // Check visibility
            hit_record shadow_rec;
            ray shadow_ray = ray_create(rec.point, light_dir);
            if (scene.hit(shadow_ray, 0.001, dist - 0.001, shadow_rec)) {
                continue;  // Shadowed
            }
            
            // Lambertian BRDF
            double cos_theta = std::max(0.0, vec3_dot(rec.normal, light_dir));
            
            // Light intensity (treat as white/flat spectrum)
            double light_intensity = (light.intensity.x + light.intensity.y + light.intensity.z) / 3.0;
            
            // Inverse square falloff
            direct += light_intensity * cos_theta / dist_sq;
        }
        
        // Sample quad lights with pre-drawn random numbers
        for (const auto& qlight : scene.quad_lights) {
            vec3 sample_point = vec3_add(qlight.corner,
                vec3_add(vec3_scale(qlight.edge_u, u1), vec3_scale(qlight.edge_v, u2)));
            
            vec3 to_light = vec3_sub(sample_point, rec.point);
            double dist_sq = vec3_length_squared(to_light);
            double dist = std::sqrt(dist_sq);
            vec3 light_dir = vec3_scale(to_light, 1.0 / dist);
            
            // Check visibility
            hit_record shadow_rec;
            ray shadow_ray = ray_create(rec.point, light_dir);
            if (scene.hit(shadow_ray, 0.001, dist - 0.001, shadow_rec)) {
                continue;  // Shadowed
            }
            
            // Geometry terms
            double cos_theta_surface = std::max(0.0, vec3_dot(rec.normal, light_dir));
            double cos_theta_light = std::max(0.0, vec3_dot(qlight.normal, vec3_negate(light_dir)));
            
            // Light emission (flat spectrum)
            double emission = (qlight.emission.x + qlight.emission.y + qlight.emission.z) / 3.0;
            
            // Area of quad light
            double area = qlight.area;
            
            // Contribution with proper PDF
            direct += emission * cos_theta_surface * cos_theta_light * area / dist_sq;
        }
        
        return direct;
    }
    
    /**
     * @brief Sample direct lighting from point lights (NEE) with seeded RNG
     * @deprecated Use sample_direct_light_fixed for correlated sampling
     */
    double sample_direct_light(const Scene& scene, const hit_record& rec, 
                               const Material& mat, double lambda,
                               SeededRNG& rng) const {
        double direct = 0.0;
        
        // Only diffuse materials receive direct lighting this way
        if (mat.type != MaterialType::Lambertian) {
            return 0.0;
        }
        
        // Sample point lights (deterministic - no RNG needed for point lights)
        for (const auto& light : scene.lights) {
            vec3 to_light = vec3_sub(light.position, rec.point);
            double dist_sq = vec3_length_squared(to_light);
            double dist = std::sqrt(dist_sq);
            vec3 light_dir = vec3_scale(to_light, 1.0 / dist);
            
            // Check visibility
            hit_record shadow_rec;
            ray shadow_ray = ray_create(rec.point, light_dir);
            if (scene.hit(shadow_ray, 0.001, dist - 0.001, shadow_rec)) {
                continue;  // Shadowed
            }
            
            // Lambertian BRDF
            double cos_theta = std::max(0.0, vec3_dot(rec.normal, light_dir));
            
            // Light intensity (treat as white/flat spectrum)
            double light_intensity = (light.intensity.x + light.intensity.y + light.intensity.z) / 3.0;
            
            // Inverse square falloff
            direct += light_intensity * cos_theta / dist_sq;
        }
        
        // Sample quad lights with seeded RNG for correlation
        for (const auto& qlight : scene.quad_lights) {
            // Use seeded RNG so all wavelengths sample the same point
            double u1 = rng.uniform();
            double u2 = rng.uniform();
            vec3 sample_point = vec3_add(qlight.corner,
                vec3_add(vec3_scale(qlight.edge_u, u1), vec3_scale(qlight.edge_v, u2)));
            
            vec3 to_light = vec3_sub(sample_point, rec.point);
            double dist_sq = vec3_length_squared(to_light);
            double dist = std::sqrt(dist_sq);
            vec3 light_dir = vec3_scale(to_light, 1.0 / dist);
            
            // Check visibility
            hit_record shadow_rec;
            ray shadow_ray = ray_create(rec.point, light_dir);
            if (scene.hit(shadow_ray, 0.001, dist - 0.001, shadow_rec)) {
                continue;  // Shadowed
            }
            
            // Geometry terms
            double cos_theta_surface = std::max(0.0, vec3_dot(rec.normal, light_dir));
            double cos_theta_light = std::max(0.0, vec3_dot(qlight.normal, vec3_negate(light_dir)));
            
            // Light emission (flat spectrum)
            double emission = (qlight.emission.x + qlight.emission.y + qlight.emission.z) / 3.0;
            
            // Area of quad light
            double area = qlight.area;
            
            // Contribution with proper PDF
            direct += emission * cos_theta_surface * cos_theta_light * area / dist_sq;
        }
        
        return direct;
    }
    
    /**
     * @brief Compute radiance for a ray at a specific wavelength
     * 
     * Uses seeded RNG for correlated sampling across wavelengths.
     * CRITICAL: Each bounce consumes a FIXED number of random values to keep
     * wavelengths synchronized. This is the key to reducing chromatic noise.
     * 
     * Random budget per bounce:
     * - r[0], r[1]: NEE light sample (quad light UV)
     * - r[2], r[3]: Bounce direction (cosine hemisphere or similar)
     * - r[4], r[5], r[6]: Auxiliary (fuzz, reflect/refract decision, etc.)
     * - r[7]: Russian roulette
     * 
     * @param initial_ray Starting ray
     * @param scene The scene
     * @param lambda Wavelength in nm
     * @param spectral_materials Map of material IDs to spectral properties
     * @param rng Seeded random number generator (same seed = same path for all wavelengths)
     * @return Spectral radiance
     */
    double Li(ray initial_ray, const Scene& scene, double lambda,
              const std::vector<SpectralMaterial>& spectral_materials,
              SeededRNG& rng) const {
        
        constexpr double PI = 3.14159265358979323846;
        
        SpectralRay sray(initial_ray, lambda);
        sray.throughput = 1.0;
        sray.radiance = 0.0;
        
        for (int depth = 0; depth < settings_.max_depth; ++depth) {
            hit_record rec;
            
            if (!scene.hit(sray.r, 0.001, 1e9, rec)) {
                // Environment contribution
                double sky_intensity = 0.3;
                sray.radiance += sray.throughput * sky_intensity;
                break;
            }
            
            const Material& mat = scene.get_material(rec.material_id);
            
            // Check for emission (emissive surfaces like area lights)
            if (vec3_length_squared(mat.emission) > 0.001) {
                double emission_intensity = (mat.emission.x + mat.emission.y + mat.emission.z) / 3.0;
                sray.radiance += sray.throughput * emission_intensity;
                break;  // Pure emitters don't scatter
            }
            
            // ================================================================
            // FIXED RANDOM BUDGET PER BOUNCE - Critical for wavelength correlation
            // All material types consume the same random numbers to stay synchronized
            // ================================================================
            
            // Random values for this bounce (consumed regardless of material type)
            double r_nee_u = rng.uniform();      // r[0]: NEE sample U
            double r_nee_v = rng.uniform();      // r[1]: NEE sample V
            double r_bounce_u = rng.uniform();   // r[2]: Bounce direction U
            double r_bounce_v = rng.uniform();   // r[3]: Bounce direction V  
            double r_aux_0 = rng.uniform();      // r[4]: Auxiliary (fuzz x, reflect/refract)
            double r_aux_1 = rng.uniform();      // r[5]: Auxiliary (fuzz y)
            double r_aux_2 = rng.uniform();      // r[6]: Auxiliary (fuzz z)
            double r_rr = rng.uniform();         // r[7]: Russian roulette
            
            // Get spectral material properties (if available)
            const SpectralMaterial* smat = nullptr;
            if (rec.material_id < static_cast<int>(spectral_materials.size())) {
                smat = &spectral_materials[rec.material_id];
            }
            
            // Direct lighting (NEE) for diffuse surfaces only
            if (mat.type == MaterialType::Lambertian) {
                double direct = sample_direct_light_fixed(scene, rec, mat, lambda, 
                                                          r_nee_u, r_nee_v);
                double albedo = (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0;
                sray.radiance += sray.throughput * albedo * direct / PI;
            }
            
            // Sample BSDF based on material type
            vec3 scattered_dir;
            double bsdf_value;
            
            switch (mat.type) {
                case MaterialType::Lambertian: {
                    // Cosine-weighted hemisphere sampling using pre-drawn randoms
                    double phi = 2.0 * PI * r_bounce_u;
                    double cos_theta = std::sqrt(r_bounce_v);
                    double sin_theta = std::sqrt(1.0 - r_bounce_v);
                    
                    // Build orthonormal basis
                    vec3 w = rec.normal;
                    vec3 a = (std::fabs(w.x) > 0.9) ? vec3{0, 1, 0} : vec3{1, 0, 0};
                    vec3 u = vec3_normalize(vec3_cross(a, w));
                    vec3 v = vec3_cross(w, u);
                    
                    scattered_dir = vec3_normalize(vec3_add(
                        vec3_add(vec3_scale(u, std::cos(phi) * sin_theta),
                                 vec3_scale(v, std::sin(phi) * sin_theta)),
                        vec3_scale(w, cos_theta)));
                    
                    // Spectral albedo
                    bsdf_value = smat ? smat->albedo_at(lambda) : 
                                        (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0;
                    break;
                }
                    
                case MaterialType::Metal: {
                    vec3 reflected = vec3_reflect(sray.r.direction, rec.normal);
                    
                    // Add roughness (fuzz) using pre-drawn randoms
                    if (mat.fuzz > 0.001) {
                        // Box-Muller-ish rejection-free approximation using uniform randoms
                        vec3 fuzz_offset = {
                            (r_aux_0 * 2.0 - 1.0) * mat.fuzz,
                            (r_aux_1 * 2.0 - 1.0) * mat.fuzz,
                            (r_aux_2 * 2.0 - 1.0) * mat.fuzz
                        };
                        reflected = vec3_normalize(vec3_add(reflected, fuzz_offset));
                    }
                    
                    scattered_dir = reflected;
                    
                    // Spectral metal Fresnel
                    double cos_theta = std::abs(vec3_dot(vec3_negate(sray.r.direction), rec.normal));
                    if (smat && smat->is_spectral_metal) {
                        bsdf_value = smat->metal_fresnel(cos_theta, lambda);
                    } else {
                        bsdf_value = (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0;
                    }
                    break;
                }
                    
                case MaterialType::Dielectric: {
                    // DISPERSIVE REFRACTION - the key feature!
                    double ior = smat ? smat->ior_at(lambda) : mat.refraction_index;
                    
                    // Use rec.front_face which is properly set by hit detection
                    // rec.normal is already flipped to point against ray direction
                    double eta = rec.front_face ? (1.0 / ior) : ior;
                    
                    vec3 unit_dir = vec3_normalize(sray.r.direction);
                    double cos_theta = std::min(vec3_dot(vec3_negate(unit_dir), rec.normal), 1.0);
                    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
                    
                    bool cannot_refract = eta * sin_theta > 1.0;
                    
                    // Schlick approximation for Fresnel
                    double r0 = (1 - ior) / (1 + ior);
                    r0 = r0 * r0;
                    double reflectance = r0 + (1 - r0) * std::pow(1 - cos_theta, 5);
                    
                    // Use pre-drawn random for reflect/refract decision
                    // All wavelengths use same random value r_aux_0
                    if (cannot_refract || r_aux_0 < reflectance) {
                        // Reflect
                        scattered_dir = vec3_reflect(unit_dir, rec.normal);
                    } else {
                        // Refract using the standard formula
                        // rec.normal points against ray, which is correct for refraction
                        scattered_dir = vec3_refract(unit_dir, rec.normal, eta);
                    }
                    
                    bsdf_value = 1.0;  // Dielectric is energy-conserving
                    break;
                }
                    
                default:
                    return sray.radiance;
            }
            
            // Update throughput
            sray.throughput *= bsdf_value;
            
            // Russian roulette using pre-drawn random
            if (depth > settings_.russian_roulette_depth) {
                double q = std::max(0.05, 1.0 - sray.throughput);
                if (r_rr < q) break;
                sray.throughput /= (1.0 - q);
            }
            
            // Continue ray
            sray.r = ray_create(rec.point, scattered_dir);
        }
        
        return sray.radiance;
    }
    
    /**
     * @brief Render a pixel with CORRELATED spectral sampling
     * 
     * The key insight: use the SAME random seed for all wavelengths.
     * This ensures all wavelengths follow the same path (same bounce directions,
     * same light samples), only differing in wavelength-dependent properties
     * (IOR, Fresnel, spectral albedo). This dramatically reduces chromatic noise.
     */
    color render_pixel_spectral(ray cam_ray, const Scene& scene,
                                const std::vector<SpectralMaterial>& spectral_materials,
                                int wavelength_samples = 8) const {
        
        // Generate a random seed for this pixel sample
        // All wavelengths will use this same seed
        uint64_t base_seed = static_cast<uint64_t>(random_double() * 0xFFFFFFFFFFFFFFFFULL);
        
        double X = 0, Y = 0, Z = 0;
        
        for (int i = 0; i < wavelength_samples; ++i) {
            // Stratified wavelength sampling
            double u = (i + 0.5) / wavelength_samples;  // Deterministic stratification
            double lambda = sample_wavelength_uniform(u);
            
            // Create RNG with SAME seed for each wavelength - CRITICAL!
            SeededRNG rng;
            rng.seed(base_seed);
            
            // Trace at this wavelength - all wavelengths follow same random path
            double radiance = Li(cam_ray, scene, lambda, spectral_materials, rng);
            
            // Convert to XYZ (CIE color matching functions)
            double Xi, Yi, Zi;
            wavelength_to_xyz(lambda, radiance, Xi, Yi, Zi);
            X += Xi;
            Y += Yi;
            Z += Zi;
        }
        
        // Average over wavelength samples
        X /= wavelength_samples;
        Y /= wavelength_samples;
        Z /= wavelength_samples;
        
        double R, G, B;
        xyz_to_rgb(X, Y, Z, R, G, B);
        
        // Clamp negative values (out-of-gamut colors)
        R = std::max(0.0, R);
        G = std::max(0.0, G);
        B = std::max(0.0, B);
        
        return {R, G, B};
    }
    
private:
    Settings settings_;
};

} // namespace spectral
} // namespace raytracer
