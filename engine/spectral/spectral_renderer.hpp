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
        bool use_nee = true;        // Next Event Estimation
        bool use_mis = true;        // Multiple Importance Sampling  
        bool use_hwss = false;      // Hero Wavelength Spectral Sampling
        double russian_roulette_depth = 5;
    };
    
    explicit SpectralIntegrator(const Settings& settings = Settings())
        : settings_(settings) {}
    
private:
    /**
     * @brief Power heuristic for MIS (balance heuristic with power=2)
     */
    static double power_heuristic(double pdf_a, double pdf_b) {
        double a2 = pdf_a * pdf_a;
        double b2 = pdf_b * pdf_b;
        return a2 / (a2 + b2 + 1e-10);
    }
    
    /**
     * @brief Convert RGB emission to spectral intensity
     * Simple approximation: use luminance-weighted average
     */
    static double emission_to_spectral(const color& emission) {
        // Use luminance weights (approximate for D65)
        return 0.2126 * emission.x + 0.7152 * emission.y + 0.0722 * emission.z;
    }
    
public:
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
     * Full spectral path tracer with NEE + MIS.
     * Uses seeded RNG for correlated sampling across wavelengths.
     * 
     * CRITICAL: Each bounce consumes a FIXED number of random values to keep
     * wavelengths synchronized. This is the key to reducing chromatic noise.
     * 
     * Random budget per bounce (10 values):
     * - r[0]: Light source selection
     * - r[1], r[2]: NEE light sample point (UV on area light)
     * - r[3], r[4]: Bounce direction (cosine hemisphere or similar)
     * - r[5], r[6], r[7]: Auxiliary (fuzz, reflect/refract decision, etc.)
     * - r[8]: Russian roulette
     * - r[9]: Reserved for future use
     */
    double Li(ray initial_ray, const Scene& scene, double lambda,
              const std::vector<SpectralMaterial>& spectral_materials,
              SeededRNG& rng) const {
        
        constexpr double PI = 3.14159265358979323846;
        
        SpectralRay sray(initial_ray, lambda);
        sray.throughput = 1.0;
        sray.radiance = 0.0;
        
        bool was_specular = false;  // Track if last bounce was specular (for MIS)
        double prev_bsdf_pdf = 1.0; // PDF of previous bounce direction
        
        for (int depth = 0; depth < settings_.max_depth; ++depth) {
            hit_record rec;
            
            if (!scene.hit(sray.r, 0.001, 1e9, rec)) {
                // Environment contribution
                double sky_intensity = 0.3;
                // No MIS for environment in this simple version
                sray.radiance += sray.throughput * sky_intensity;
                break;
            }
            
            const Material& mat = scene.get_material(rec.material_id);
            
            // ================================================================
            // FIXED RANDOM BUDGET PER BOUNCE - Critical for wavelength correlation
            // ================================================================
            double r_light_sel = rng.uniform();   // r[0]: Light selection
            double r_nee_u = rng.uniform();       // r[1]: NEE sample U
            double r_nee_v = rng.uniform();       // r[2]: NEE sample V
            double r_bounce_u = rng.uniform();    // r[3]: Bounce direction U
            double r_bounce_v = rng.uniform();    // r[4]: Bounce direction V  
            double r_aux_0 = rng.uniform();       // r[5]: Auxiliary
            double r_aux_1 = rng.uniform();       // r[6]: Auxiliary
            double r_aux_2 = rng.uniform();       // r[7]: Auxiliary
            double r_rr = rng.uniform();          // r[8]: Russian roulette
            double r_reserved = rng.uniform();    // r[9]: Reserved
            (void)r_light_sel; (void)r_reserved;  // Suppress unused warnings for now
            
            // Get spectral material properties
            const SpectralMaterial* smat = nullptr;
            if (rec.material_id < static_cast<int>(spectral_materials.size())) {
                smat = &spectral_materials[rec.material_id];
            }
            
            // Check for emission (emissive surfaces like area lights)
            if (vec3_length_squared(mat.emission) > 0.001) {
                double emission_val = emission_to_spectral(mat.emission);
                
                if (!settings_.use_nee || !settings_.use_mis || was_specular || depth == 0) {
                    // Add full emission if:
                    // - NEE disabled
                    // - First hit (camera ray)
                    // - After specular bounce (NEE can't sample specular paths)
                    sray.radiance += sray.throughput * emission_val;
                } else {
                    // MIS: Weight BSDF sample against light sample
                    // Compute light PDF for this hit
                    double light_pdf_area = scene.light_pdf(
                        ray_at(sray.r, -0.001),  // Previous hit point (approximate)
                        rec.point);
                    
                    if (light_pdf_area > 0) {
                        double dist = rec.t;
                        double cos_light = std::abs(vec3_dot(rec.normal, 
                            vec3_negate(vec3_normalize(sray.r.direction))));
                        
                        if (cos_light > 0) {
                            double light_pdf_sa = light_pdf_area * dist * dist / cos_light;
                            double mis_weight = power_heuristic(prev_bsdf_pdf, light_pdf_sa);
                            sray.radiance += sray.throughput * emission_val * mis_weight;
                        }
                    } else {
                        // Light not in sampling distribution, add full contribution
                        sray.radiance += sray.throughput * emission_val;
                    }
                }
                break;  // Pure emitters don't scatter
            }
            
            // ================================================================
            // NEE: Sample lights directly for diffuse surfaces
            // ================================================================
            if (settings_.use_nee && mat.type == MaterialType::Lambertian) {
                // Sample a light using the scene's light sampling
                LightSample ls = sample_light_deterministic(scene, rec.point, 
                                                            r_light_sel, r_nee_u, r_nee_v);
                
                if (ls.valid) {
                    vec3 to_light = vec3_sub(ls.position, rec.point);
                    double light_dist = vec3_length(to_light);
                    vec3 light_dir = vec3_scale(to_light, 1.0 / light_dist);
                    
                    double cos_surf = vec3_dot(rec.normal, light_dir);
                    
                    if (cos_surf > 0) {
                        // Shadow test
                        hit_record shadow_rec;
                        bool occluded = scene.hit(ray_create(rec.point, light_dir), 
                                                  0.001, light_dist - 0.001, shadow_rec);
                        
                        if (!occluded) {
                            // Convert area PDF to solid angle PDF
                            double cos_light = std::abs(vec3_dot(ls.normal, vec3_negate(light_dir)));
                            double light_pdf_sa = ls.pdf * light_dist * light_dist / 
                                                  std::max(cos_light, 0.001);
                            
                            // BSDF: Lambertian = albedo / PI, PDF = cos_theta / PI
                            double bsdf_pdf = cos_surf / PI;
                            double albedo = smat ? smat->albedo_at(lambda) :
                                           (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0;
                            double bsdf_val = albedo / PI;
                            
                            // MIS weight
                            double mis_weight = settings_.use_mis ? 
                                power_heuristic(light_pdf_sa, bsdf_pdf) : 1.0;
                            
                            // Light emission as spectral
                            double light_emission = emission_to_spectral(ls.emission);
                            
                            // NEE contribution: Le * BSDF * cos / light_pdf * MIS
                            double nee_contrib = light_emission * bsdf_val * cos_surf / 
                                                 light_pdf_sa * mis_weight;
                            sray.radiance += sray.throughput * nee_contrib;
                        }
                    }
                }
            }
            
            // ================================================================
            // Sample BSDF for next direction
            // ================================================================
            vec3 scattered_dir;
            double bsdf_value;
            double bsdf_pdf = 1.0;
            bool is_specular = false;
            
            switch (mat.type) {
                case MaterialType::Lambertian: {
                    // Cosine-weighted hemisphere sampling
                    double phi = 2.0 * PI * r_bounce_u;
                    double cos_theta = std::sqrt(r_bounce_v);
                    double sin_theta = std::sqrt(1.0 - r_bounce_v);
                    
                    vec3 w = rec.normal;
                    vec3 a = (std::fabs(w.x) > 0.9) ? vec3{0, 1, 0} : vec3{1, 0, 0};
                    vec3 u = vec3_normalize(vec3_cross(a, w));
                    vec3 v = vec3_cross(w, u);
                    
                    scattered_dir = vec3_normalize(vec3_add(
                        vec3_add(vec3_scale(u, std::cos(phi) * sin_theta),
                                 vec3_scale(v, std::sin(phi) * sin_theta)),
                        vec3_scale(w, cos_theta)));
                    
                    bsdf_value = smat ? smat->albedo_at(lambda) : 
                                        (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0;
                    bsdf_pdf = cos_theta / PI;  // Cosine-weighted PDF
                    is_specular = false;
                    break;
                }
                    
                case MaterialType::Metal: {
                    vec3 reflected = vec3_reflect(sray.r.direction, rec.normal);
                    
                    if (mat.fuzz > 0.001) {
                        vec3 fuzz_offset = {
                            (r_aux_0 * 2.0 - 1.0) * mat.fuzz,
                            (r_aux_1 * 2.0 - 1.0) * mat.fuzz,
                            (r_aux_2 * 2.0 - 1.0) * mat.fuzz
                        };
                        reflected = vec3_normalize(vec3_add(reflected, fuzz_offset));
                        is_specular = (mat.fuzz < 0.1);  // Slightly rough = still mostly specular
                    } else {
                        is_specular = true;  // Perfect mirror
                    }
                    
                    scattered_dir = reflected;
                    
                    double cos_theta = std::abs(vec3_dot(vec3_negate(sray.r.direction), rec.normal));
                    if (smat && smat->is_spectral_metal) {
                        bsdf_value = smat->metal_fresnel(cos_theta, lambda);
                    } else {
                        bsdf_value = (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0;
                    }
                    bsdf_pdf = 1.0;  // Delta or near-delta
                    break;
                }
                    
                case MaterialType::Dielectric: {
                    double ior = smat ? smat->ior_at(lambda) : mat.refraction_index;
                    double eta = rec.front_face ? (1.0 / ior) : ior;
                    
                    vec3 unit_dir = vec3_normalize(sray.r.direction);
                    double cos_theta = std::min(vec3_dot(vec3_negate(unit_dir), rec.normal), 1.0);
                    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
                    
                    bool cannot_refract = eta * sin_theta > 1.0;
                    
                    double r0 = (1 - ior) / (1 + ior);
                    r0 = r0 * r0;
                    double reflectance = r0 + (1 - r0) * std::pow(1 - cos_theta, 5);
                    
                    if (cannot_refract || r_aux_0 < reflectance) {
                        scattered_dir = vec3_reflect(unit_dir, rec.normal);
                    } else {
                        scattered_dir = vec3_refract(unit_dir, rec.normal, eta);
                    }
                    
                    bsdf_value = 1.0;
                    bsdf_pdf = 1.0;  // Delta distribution
                    is_specular = true;
                    break;
                }
                    
                default:
                    return sray.radiance;
            }
            
            // Update state for next iteration
            sray.throughput *= bsdf_value;
            was_specular = is_specular;
            prev_bsdf_pdf = bsdf_pdf;
            
            // Russian roulette
            if (depth > settings_.russian_roulette_depth) {
                double q = std::max(0.05, 1.0 - sray.throughput);
                if (r_rr < q) break;
                sray.throughput /= (1.0 - q);
            }
            
            sray.r = ray_create(rec.point, scattered_dir);
        }
        
        return sray.radiance;
    }
    
    /**
     * @brief Sample light with deterministic random values
     * 
     * Uses pre-drawn random values to maintain wavelength correlation.
     */
    LightSample sample_light_deterministic(const Scene& scene, point3 from_point,
                                           double r_select, double r_u, double r_v) const {
        LightSample sample;
        sample.valid = false;
        
        // Count light sources
        size_t num_quad_lights = scene.quad_lights.size();
        size_t num_point_lights = scene.lights.size();
        size_t num_disk_lights = scene.disk_lights.size();
        
        // Get emissive spheres
        std::vector<size_t> emissive_spheres;
        for (size_t i = 0; i < scene.spheres.size(); ++i) {
            const Material& mat = scene.get_material(scene.spheres[i].material_id);
            if (mat.is_emissive()) {
                emissive_spheres.push_back(i);
            }
        }
        size_t num_emissive = emissive_spheres.size();
        
        size_t total = num_quad_lights + num_point_lights + num_disk_lights + num_emissive;
        if (total == 0) return sample;
        
        // Select a light
        size_t idx = static_cast<size_t>(r_select * total);
        if (idx >= total) idx = total - 1;
        
        size_t offset = 0;
        
        // Quad lights
        if (idx < num_quad_lights) {
            const auto& ql = scene.quad_lights[idx];
            sample.position = vec3_add(ql.corner,
                vec3_add(vec3_scale(ql.edge_u, r_u), vec3_scale(ql.edge_v, r_v)));
            sample.normal = ql.normal;
            sample.emission = ql.emission;
            sample.pdf = ql.pdf() / static_cast<double>(total);
            sample.distance = vec3_length(vec3_sub(sample.position, from_point));
            sample.valid = true;
            return sample;
        }
        offset += num_quad_lights;
        
        // Point lights
        if (idx < offset + num_point_lights) {
            const auto& pl = scene.lights[idx - offset];
            sample.position = pl.position;
            sample.normal = vec3{0, -1, 0};
            sample.emission = pl.intensity;
            sample.pdf = 1.0 / static_cast<double>(total);  // Delta
            sample.distance = vec3_length(vec3_sub(sample.position, from_point));
            sample.valid = true;
            return sample;
        }
        offset += num_point_lights;
        
        // Disk lights
        if (idx < offset + num_disk_lights) {
            const auto& dl = scene.disk_lights[idx - offset];
            // Sample disk using r_u, r_v
            double r = std::sqrt(r_u) * dl.radius;
            double theta = 2.0 * 3.14159265358979323846 * r_v;
            vec3 local = {r * std::cos(theta), 0, r * std::sin(theta)};
            // Transform to world (simplified - assumes disk in XZ plane)
            sample.position = vec3_add(dl.center, local);
            sample.normal = dl.normal;
            sample.emission = dl.emission;
            sample.pdf = dl.pdf() / static_cast<double>(total);
            sample.distance = vec3_length(vec3_sub(sample.position, from_point));
            sample.valid = true;
            return sample;
        }
        offset += num_disk_lights;
        
        // Emissive spheres
        if (idx < offset + num_emissive) {
            size_t sphere_idx = emissive_spheres[idx - offset];
            sample = scene.sample_sphere_light(sphere_idx, from_point);
            if (sample.valid) {
                sample.pdf /= static_cast<double>(total);
            }
            return sample;
        }
        
        return sample;
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
