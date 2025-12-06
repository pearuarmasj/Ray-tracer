#pragma once

#include "photon_map.hpp"
#include "scene.hpp"
#include <random>
#include <cmath>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace raytracer {

// Configuration for photon mapping
struct PhotonMappingSettings {
    // Photon emission
    size_t num_global_photons = 100000;
    size_t num_caustic_photons = 50000;
    int max_photon_depth = 10;
    
    // Gathering
    int gather_count = 100;          // Number of photons to gather
    float gather_radius = 0.5f;      // Maximum gather radius
    float caustic_radius = 0.1f;     // Smaller radius for caustics (sharper)
    
    // Final gather (optional second bounce for smoother results)
    bool use_final_gather = false;
    int final_gather_samples = 64;
    
    // Direct lighting
    bool compute_direct_separately = true;  // Use direct lighting instead of direct photons
};

// Photon integrator implementing two-pass photon mapping
class PhotonIntegrator {
public:
    PhotonMap global_map;
    PhotonMap caustic_map;
    PhotonMappingSettings settings;
    
private:
    std::mt19937 rng_;
    
public:
    PhotonIntegrator() : rng_(42) {}
    
    // -------------------------------------------------------------------------
    // Pass 1: Photon tracing from light sources
    // -------------------------------------------------------------------------
    void trace_photons(const Scene& scene) {
        trace_global_photons(scene);
        trace_caustic_photons(scene);
        
        global_map.build();
        caustic_map.build();
    }
    
private:
    void trace_global_photons(const Scene& scene) {
        // Find emissive spheres
        std::vector<size_t> emissive_spheres;
        double total_power = 0.0;
        
        for (size_t i = 0; i < scene.spheres.size(); ++i) {
            const auto& sphere = scene.spheres[i];
            const auto& mat = scene.materials[sphere.material_id];
            if (mat.emission.x > 0 || mat.emission.y > 0 || mat.emission.z > 0) {
                emissive_spheres.push_back(i);
                total_power += (mat.emission.x + mat.emission.y + mat.emission.z) / 3.0;
            }
        }
        
        if (emissive_spheres.empty()) return;
        
        for (size_t idx : emissive_spheres) {
            const auto& sphere = scene.spheres[idx];
            const auto& mat = scene.materials[sphere.material_id];
            double light_power = (mat.emission.x + mat.emission.y + mat.emission.z) / 3.0;
            double fraction = light_power / total_power;
            size_t photons_for_light = static_cast<size_t>(settings.num_global_photons * fraction);
            
            // Photon power = light power / num_photons
            double area = 4.0 * M_PI * sphere.radius * sphere.radius;
            color photon_power = vec3_scale(mat.emission, area / static_cast<double>(photons_for_light));
            
            for (size_t i = 0; i < photons_for_light; ++i) {
                emit_photon_from_sphere(scene, sphere, mat, photon_power, false);
            }
        }
    }
    
    void trace_caustic_photons(const Scene& scene) {
        // Find emissive spheres
        std::vector<size_t> emissive_spheres;
        double total_power = 0.0;
        
        for (size_t i = 0; i < scene.spheres.size(); ++i) {
            const auto& sphere = scene.spheres[i];
            const auto& mat = scene.materials[sphere.material_id];
            if (mat.emission.x > 0 || mat.emission.y > 0 || mat.emission.z > 0) {
                emissive_spheres.push_back(i);
                total_power += (mat.emission.x + mat.emission.y + mat.emission.z) / 3.0;
            }
        }
        
        if (emissive_spheres.empty()) return;
        
        // Find specular spheres for caustic targeting
        std::vector<size_t> specular_indices;
        for (size_t i = 0; i < scene.spheres.size(); ++i) {
            const auto& sphere = scene.spheres[i];
            const auto& mat = scene.materials[sphere.material_id];
            if (mat.type == MaterialType::Dielectric || 
                mat.type == MaterialType::Metal ||
                mat.fuzz < 0.1) {
                specular_indices.push_back(i);
            }
        }
        
        if (specular_indices.empty()) return;
        
        for (size_t idx : emissive_spheres) {
            const auto& sphere = scene.spheres[idx];
            const auto& mat = scene.materials[sphere.material_id];
            double light_power = (mat.emission.x + mat.emission.y + mat.emission.z) / 3.0;
            double fraction = light_power / total_power;
            size_t photons_for_light = static_cast<size_t>(settings.num_caustic_photons * fraction);
            
            double area = 4.0 * M_PI * sphere.radius * sphere.radius;
            color photon_power = vec3_scale(mat.emission, area / static_cast<double>(photons_for_light));
            
            for (size_t i = 0; i < photons_for_light; ++i) {
                emit_caustic_photon_from_sphere(scene, sphere, mat, photon_power, specular_indices);
            }
        }
    }
    
    void emit_photon_from_sphere(const Scene& scene, const Sphere& light_sphere,
                                  const Material& light_mat, const color& power, bool caustic_only) {
        // Sample point on sphere light
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(rng_);
        double v = dist(rng_);
        
        double theta = 2.0 * M_PI * u;
        double phi = std::acos(2.0 * v - 1.0);
        
        vec3 local_dir = {
            std::sin(phi) * std::cos(theta),
            std::sin(phi) * std::sin(theta),
            std::cos(phi)
        };
        
        point3 origin = vec3_add(light_sphere.center, vec3_scale(local_dir, light_sphere.radius * 1.001));
        
        // Random direction in hemisphere
        vec3 dir = sample_hemisphere(local_dir);
        
        ray r = {origin, dir};
        trace_photon(scene, r, power, 0, false, caustic_only);
    }
    
    void emit_caustic_photon_from_sphere(const Scene& scene, const Sphere& light_sphere,
                                          const Material& light_mat, const color& power, 
                                          const std::vector<size_t>& specular_indices) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        // Pick a random specular sphere to aim at
        size_t target_idx = specular_indices[static_cast<size_t>(dist(rng_) * specular_indices.size()) 
                                             % specular_indices.size()];
        const auto& target = scene.spheres[target_idx];
        
        // Sample point on light sphere
        double u = dist(rng_);
        double v = dist(rng_);
        double theta = 2.0 * M_PI * u;
        double phi = std::acos(2.0 * v - 1.0);
        vec3 local_dir = {
            std::sin(phi) * std::cos(theta),
            std::sin(phi) * std::sin(theta),
            std::cos(phi)
        };
        point3 origin = vec3_add(light_sphere.center, vec3_scale(local_dir, light_sphere.radius * 1.001));
        
        // Aim towards the specular object (with some randomness)
        vec3 sphere_sample = sample_sphere();
        point3 target_point = vec3_add(target.center, vec3_scale(sphere_sample, target.radius * 0.5));
        vec3 to_target = vec3_sub(target_point, origin);
        vec3 dir = vec3_normalize(to_target);
        
        // Adjust power for solid angle
        double dist_sq = vec3_length_squared(to_target);
        double solid_angle = M_PI * target.radius * target.radius / dist_sq;
        color adjusted_power = vec3_scale(power, 4.0 * M_PI / solid_angle);
        
        // Trace, but only store after hitting specular surface
        ray r = {origin, dir};
        trace_photon(scene, r, adjusted_power, 0, true, true);
    }
    
    void trace_photon(const Scene& scene, const ray& r, color power, 
                      int depth, bool hit_specular, bool caustic_only) {
        if (depth >= settings.max_photon_depth) return;
        
        // Russian roulette for termination
        double max_component = std::max({power.x, power.y, power.z});
        if (depth > 3) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double continue_prob = std::min(0.9, max_component);
            if (dist(rng_) > continue_prob) return;
            power = vec3_scale(power, 1.0 / continue_prob);
        }
        
        hit_record rec;
        if (!scene.hit(r, 0.001, 1e30, rec)) return;
        
        const Material& mat = scene.get_material(rec.material_id);
        
        if (mat.type == MaterialType::Dielectric) {
            // Refract/reflect photon
            vec3 outward_normal;
            double ni_over_nt;
            double cosine;
            
            if (vec3_dot(r.direction, rec.normal) > 0) {
                outward_normal = vec3_negate(rec.normal);
                ni_over_nt = mat.refraction_index;
                cosine = mat.refraction_index * vec3_dot(r.direction, rec.normal);
            } else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0 / mat.refraction_index;
                cosine = -vec3_dot(r.direction, rec.normal);
            }
            
            vec3 refracted;
            double reflect_prob = 1.0;
            if (refract_vec(r.direction, outward_normal, ni_over_nt, &refracted)) {
                reflect_prob = schlick(cosine, mat.refraction_index);
            }
            
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(rng_) < reflect_prob) {
                vec3 reflected = reflect_vec(r.direction, rec.normal);
                ray new_ray = {vec3_add(rec.point, vec3_scale(reflected, 0.001)), reflected};
                trace_photon(scene, new_ray, power, depth + 1, true, caustic_only);
            } else {
                color new_power = vec3_mul(power, mat.albedo);
                ray new_ray = {vec3_add(rec.point, vec3_scale(refracted, 0.001)), refracted};
                trace_photon(scene, new_ray, new_power, depth + 1, true, caustic_only);
            }
        }
        else if (mat.type == MaterialType::Metal) {
            vec3 reflected = reflect_vec(r.direction, rec.normal);
            color new_power = vec3_mul(power, mat.albedo);
            ray new_ray = {vec3_add(rec.point, vec3_scale(reflected, 0.001)), reflected};
            trace_photon(scene, new_ray, new_power, depth + 1, true, caustic_only);
        }
        else if (mat.type == MaterialType::Lambertian) {
            // Store photon at diffuse surface
            if (caustic_only) {
                // Only store if we've hit specular surface before
                if (hit_specular) {
                    Photon p;
                    p.position = rec.point;
                    p.direction = vec3_negate(r.direction);
                    p.power = power;
                    p.flags = Photon::CAUSTIC;
                    caustic_map.add_photon(p);
                }
            } else {
                // Global photon - store if not direct lighting
                if (depth > 0) {
                    Photon p;
                    p.position = rec.point;
                    p.direction = vec3_negate(r.direction);
                    p.power = power;
                    p.flags = hit_specular ? Photon::CAUSTIC : Photon::DIFFUSE;
                    global_map.add_photon(p);
                }
            }
            
            // Russian roulette for diffuse bounce
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double albedo_avg = (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0;
            if (dist(rng_) < albedo_avg && !caustic_only) {
                vec3 new_dir = sample_hemisphere(rec.normal);
                color new_power = vec3_scale(vec3_mul(power, mat.albedo), 1.0 / albedo_avg);
                ray new_ray = {vec3_add(rec.point, vec3_scale(new_dir, 0.001)), new_dir};
                trace_photon(scene, new_ray, new_power, depth + 1, false, false);
            }
        }
    }
    
public:
    // -------------------------------------------------------------------------
    // Pass 2: Radiance estimation using gathered photons
    // -------------------------------------------------------------------------
    color estimate_radiance(const Scene& scene, const ray& r, int depth = 0) const {
        if (depth > 10) return vec3_zero();
        
        hit_record rec;
        if (!scene.hit(r, 0.001, 1e30, rec)) {
            return vec3_zero();  // Background handled separately
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        color result = vec3_zero();
        
        // Emissive
        if (mat.emission.x > 0 || mat.emission.y > 0 || mat.emission.z > 0) {
            return mat.emission;
        }
        
        if (mat.type == MaterialType::Dielectric) {
            return trace_glass(scene, r, rec, mat, depth);
        }
        else if (mat.type == MaterialType::Metal) {
            vec3 reflected = reflect_vec(r.direction, rec.normal);
            ray new_ray = {vec3_add(rec.point, vec3_scale(reflected, 0.001)), reflected};
            return vec3_mul(mat.albedo, estimate_radiance(scene, new_ray, depth + 1));
        }
        else {
            // Diffuse surface - gather photons
            
            // Direct lighting (computed separately for better quality)
            if (settings.compute_direct_separately) {
                result = vec3_add(result, compute_direct_lighting(scene, rec));
            }
            
            // Caustic photons (always gathered, sharp features)
            if (caustic_map.size() > 0) {
                color caustic = caustic_map.estimate_irradiance(
                    rec.point, rec.normal, settings.gather_count, settings.caustic_radius);
                result = vec3_add(result, vec3_mul(mat.albedo, caustic));
            }
            
            // Global indirect illumination
            if (global_map.size() > 0) {
                color indirect = global_map.estimate_irradiance(
                    rec.point, rec.normal, settings.gather_count, settings.gather_radius);
                result = vec3_add(result, vec3_mul(mat.albedo, indirect));
            }
        }
        
        return result;
    }
    
private:
    color trace_glass(const Scene& scene, const ray& r, const hit_record& rec,
                      const Material& mat, int depth) const {
        vec3 outward_normal;
        double ni_over_nt;
        double cosine;
        
        if (vec3_dot(r.direction, rec.normal) > 0) {
            outward_normal = vec3_negate(rec.normal);
            ni_over_nt = mat.refraction_index;
            cosine = mat.refraction_index * vec3_dot(r.direction, rec.normal);
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / mat.refraction_index;
            cosine = -vec3_dot(r.direction, rec.normal);
        }
        
        vec3 refracted;
        double reflect_prob = 1.0;
        if (refract_vec(r.direction, outward_normal, ni_over_nt, &refracted)) {
            reflect_prob = schlick(cosine, mat.refraction_index);
        }
        
        // Deterministic for final render (importance sample both)
        vec3 reflected = reflect_vec(r.direction, rec.normal);
        ray reflect_ray = {vec3_add(rec.point, vec3_scale(reflected, 0.001)), reflected};
        color reflect_color = estimate_radiance(scene, reflect_ray, depth + 1);
        
        if (reflect_prob < 1.0) {
            ray refract_ray = {vec3_add(rec.point, vec3_scale(refracted, 0.001)), refracted};
            color refract_color = estimate_radiance(scene, refract_ray, depth + 1);
            return vec3_add(
                vec3_scale(reflect_color, reflect_prob),
                vec3_scale(vec3_mul(mat.albedo, refract_color), 1.0 - reflect_prob)
            );
        }
        
        return reflect_color;
    }
    
    color compute_direct_lighting(const Scene& scene, const hit_record& rec) const {
        color direct = vec3_zero();
        const Material& mat = scene.get_material(rec.material_id);
        
        // Sample from emissive spheres
        for (size_t i = 0; i < scene.spheres.size(); ++i) {
            const auto& sphere = scene.spheres[i];
            const auto& light_mat = scene.materials[sphere.material_id];
            
            // Skip non-emissive spheres
            if (light_mat.emission.x <= 0 && light_mat.emission.y <= 0 && light_mat.emission.z <= 0) {
                continue;
            }
            
            // Sample point on light sphere center (simplified)
            point3 light_sample = sphere.center;
            vec3 to_light = vec3_sub(light_sample, rec.point);
            double dist = vec3_length(to_light);
            vec3 L = vec3_scale(to_light, 1.0 / dist);
            
            double NdotL = vec3_dot(rec.normal, L);
            if (NdotL <= 0) continue;
            
            // Shadow ray
            ray shadow_ray = {vec3_add(rec.point, vec3_scale(rec.normal, 0.001)), L};
            hit_record shadow_rec;
            bool in_shadow = scene.hit(shadow_ray, 0.001, dist - sphere.radius, shadow_rec);
            
            if (!in_shadow) {
                double falloff = 1.0 / (dist * dist);
                double light_area = 4.0 * M_PI * sphere.radius * sphere.radius;
                color contrib = vec3_scale(
                    vec3_mul(mat.albedo, light_mat.emission),
                    NdotL * falloff * light_area / M_PI
                );
                direct = vec3_add(direct, contrib);
            }
        }
        
        return direct;
    }
    
    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------
    vec3 sample_hemisphere(const vec3& normal) const {
        thread_local std::mt19937 local_rng(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        double u1 = dist(local_rng);
        double u2 = dist(local_rng);
        double r = std::sqrt(u1);
        double theta = 2.0 * M_PI * u2;
        
        vec3 local = {r * std::cos(theta), r * std::sin(theta), std::sqrt(1.0 - u1)};
        
        // Transform to world space
        vec3 w = normal;
        vec3 a = (std::abs(w.x) > 0.1) ? vec3{0, 1, 0} : vec3{1, 0, 0};
        vec3 u_vec = vec3_normalize(vec3_cross(w, a));
        vec3 v_vec = vec3_cross(w, u_vec);
        
        return vec3_normalize(vec3_add(
            vec3_add(vec3_scale(u_vec, local.x), vec3_scale(v_vec, local.y)),
            vec3_scale(w, local.z)
        ));
    }
    
    vec3 sample_sphere() const {
        thread_local std::mt19937 local_rng(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        double u = dist(local_rng);
        double v = dist(local_rng);
        double theta = 2.0 * M_PI * u;
        double phi = std::acos(2.0 * v - 1.0);
        
        return vec3{
            std::sin(phi) * std::cos(theta),
            std::sin(phi) * std::sin(theta),
            std::cos(phi)
        };
    }
    
    static vec3 reflect_vec(const vec3& v, const vec3& n) {
        return vec3_sub(v, vec3_scale(n, 2.0 * vec3_dot(v, n)));
    }
    
    static bool refract_vec(const vec3& v, const vec3& n, double ni_over_nt, vec3* refracted) {
        vec3 uv = vec3_normalize(v);
        double dt = vec3_dot(uv, n);
        double discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
        if (discriminant > 0) {
            *refracted = vec3_sub(
                vec3_scale(vec3_sub(uv, vec3_scale(n, dt)), ni_over_nt),
                vec3_scale(n, std::sqrt(discriminant))
            );
            return true;
        }
        return false;
    }
    
    static double schlick(double cosine, double ior) {
        double r0 = (1.0 - ior) / (1.0 + ior);
        r0 = r0 * r0;
        return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
    }
};

} // namespace raytracer
