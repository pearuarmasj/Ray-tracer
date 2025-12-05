/**
 * @file renderer.cpp
 * @brief Implementation of Whitted-style ray tracer
 */

#include "renderer.hpp"
#include "stb_image_write.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <atomic>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace raytracer {

// Thread-local random number generator
thread_local std::mt19937 rng{std::random_device{}()};
thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

double random_double() {
    return dist(rng);
}

double random_double(double min, double max) {
    return min + (max - min) * random_double();
}

// Generate random point in unit sphere (rejection sampling)
vec3 random_in_unit_sphere() {
    while (true) {
        vec3 p = {random_double(-1, 1), random_double(-1, 1), random_double(-1, 1)};
        if (vec3_length_squared(p) < 1.0)
            return p;
    }
}

// Generate random unit vector (for true Lambertian)
vec3 random_unit_vector() {
    return vec3_normalize(random_in_unit_sphere());
}

// Generate random vector in hemisphere around normal
vec3 random_on_hemisphere(vec3 normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (vec3_dot(on_unit_sphere, normal) > 0.0)
        return on_unit_sphere;
    else
        return vec3_negate(on_unit_sphere);
}

bool Image::write_ppm(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "P3\n" << width << ' ' << height << "\n255\n";
    
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            color c = get_pixel(x, y);
            
            // Gamma correction (gamma = 2.0)
            double r = std::sqrt(c.x);
            double g = std::sqrt(c.y);
            double b = std::sqrt(c.z);
            
            // Clamp and convert to 0-255
            int ir = static_cast<int>(256 * std::clamp(r, 0.0, 0.999));
            int ig = static_cast<int>(256 * std::clamp(g, 0.0, 0.999));
            int ib = static_cast<int>(256 * std::clamp(b, 0.0, 0.999));
            
            file << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    
    return true;
}

bool Image::write_png(const std::string& filename) const {
    // Convert floating-point pixels to 8-bit RGB
    std::vector<unsigned char> data(width * height * 3);
    
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            color c = get_pixel(x, y);
            
            // Gamma correction (gamma = 2.0)
            double r = std::sqrt(c.x);
            double g = std::sqrt(c.y);
            double b = std::sqrt(c.z);
            
            // Clamp and convert to 0-255
            int ir = static_cast<int>(256 * std::clamp(r, 0.0, 0.999));
            int ig = static_cast<int>(256 * std::clamp(g, 0.0, 0.999));
            int ib = static_cast<int>(256 * std::clamp(b, 0.0, 0.999));
            
            // stb expects top-to-bottom, so flip Y
            int out_y = height - 1 - y;
            int idx = (out_y * width + x) * 3;
            data[idx + 0] = static_cast<unsigned char>(ir);
            data[idx + 1] = static_cast<unsigned char>(ig);
            data[idx + 2] = static_cast<unsigned char>(ib);
        }
    }
    
    return stbi_write_png(filename.c_str(), width, height, 3, data.data(), width * 3) != 0;
}

Image Renderer::render(const Scene& scene, const Camera& camera) const {
    Image image(settings_.width, settings_.height);
    
    const char* mode_name = (settings_.mode == RenderMode::PathTrace) ? "Path Tracing" : "Whitted";
    std::cout << "Rendering " << settings_.width << "x" << settings_.height 
              << " image (" << mode_name;
    if (settings_.mode == RenderMode::PathTrace) {
        std::cout << ", NEE=" << (settings_.use_nee ? "on" : "off")
                  << ", MIS=" << (settings_.use_mis ? "on" : "off");
    }
    std::cout << ")";
    
#ifdef _OPENMP
    std::cout << " using " << omp_get_max_threads() << " threads";
#endif
    std::cout << "..." << std::endl;
    
    // Build BVH before parallel rendering (must be single-threaded)
    scene.build_bvh();
    
    std::atomic<int> completed_lines{0};
    const int total_lines = settings_.height;
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < settings_.height; ++y) {
        for (int x = 0; x < settings_.width; ++x) {
            color pixel_color = vec3_zero();
            
            for (int s = 0; s < settings_.samples_per_pixel; ++s) {
                // Random jitter within pixel for anti-aliasing
                double u = (static_cast<double>(x) + random_double()) / settings_.width;
                double v = (static_cast<double>(y) + random_double()) / settings_.height;
                
                ray r = camera.get_ray(u, v);
                
                // Choose rendering mode
                if (settings_.mode == RenderMode::PathTrace) {
                    pixel_color = vec3_add(pixel_color, ray_color_path(r, scene, settings_.max_depth));
                } else {
                    pixel_color = vec3_add(pixel_color, ray_color_whitted(r, scene, settings_.max_depth));
                }
            }
            
            // Average samples
            pixel_color = vec3_scale(pixel_color, 1.0 / settings_.samples_per_pixel);
            image.set_pixel(x, y, pixel_color);
        }
        
        // Thread-safe progress update
        int done = ++completed_lines;
        if (done % 50 == 0 || done == total_lines) {
            #pragma omp critical
            {
                std::cout << "\rProgress: " << (100 * done / total_lines) << "% (" 
                          << done << "/" << total_lines << " lines)" << std::flush;
            }
        }
    }
    
    std::cout << "\nDone!" << std::endl;
    return image;
}

color Renderer::ray_color_whitted(ray r, const Scene& scene, int depth) const {
    // Exceeded recursion limit
    if (depth <= 0) {
        return vec3_zero();
    }
    
    hit_record rec;
    
    // Check for intersection (t_min = 0.001 to avoid shadow acne)
    if (scene.hit(r, 0.001, 1e9, rec)) {
        const Material& mat = scene.get_material(rec.material_id);
        
        // Get color from texture or albedo
        color surface_color = mat.get_albedo(rec.point, rec.u, rec.v);
        
        switch (mat.type) {
            case MaterialType::Lambertian: {
                // Calculate direct lighting from all lights
                color direct_light = vec3_zero();
                for (const auto& light : scene.lights) {
                    if (!scene.is_shadowed(rec.point, light.position)) {
                        vec3 light_dir = vec3_normalize(vec3_sub(light.position, rec.point));
                        double diffuse = std::fmax(0.0, vec3_dot(rec.normal, light_dir));
                        direct_light = vec3_add(direct_light, vec3_scale(light.intensity, diffuse));
                    }
                }
                
                // Indirect lighting via random scattering
                vec3 scatter_direction = vec3_add(rec.normal, random_unit_vector());
                if (vec3_length_squared(scatter_direction) < 1e-8) {
                    scatter_direction = rec.normal;
                }
                
                ray scattered = ray_create(rec.point, scatter_direction);
                color indirect = ray_color_whitted(scattered, scene, depth - 1);
                
                // Combine direct + indirect lighting
                color total_light = vec3_add(direct_light, indirect);
                return vec3_mul(surface_color, total_light);
            }
            
            case MaterialType::Metal: {
                // Specular reflection
                vec3 reflected = vec3_reflect(vec3_normalize(r.direction), rec.normal);
                ray scattered = ray_create(rec.point, reflected);
                
                // Only reflect if reflected ray goes away from surface
                if (vec3_dot(scattered.direction, rec.normal) > 0) {
                    return vec3_mul(surface_color, ray_color_whitted(scattered, scene, depth - 1));
                }
                return vec3_zero();
            }
            
            case MaterialType::Dielectric: {
                // Refraction (glass-like material)
                color attenuation = {1.0, 1.0, 1.0};
                double refraction_ratio = rec.front_face ? 
                    (1.0 / mat.refraction_index) : mat.refraction_index;
                
                vec3 unit_direction = vec3_normalize(r.direction);
                double cos_theta = std::fmin(vec3_dot(vec3_negate(unit_direction), rec.normal), 1.0);
                double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
                
                bool cannot_refract = refraction_ratio * sin_theta > 1.0;
                vec3 direction;
                
                // Use random number for probabilistic Fresnel reflection
                if (cannot_refract || Material::reflectance(cos_theta, refraction_ratio) > random_double()) {
                    // Reflect (total internal reflection or Fresnel)
                    direction = vec3_reflect(unit_direction, rec.normal);
                } else {
                    // Refract
                    direction = vec3_refract(unit_direction, rec.normal, refraction_ratio);
                }
                
                ray scattered = ray_create(rec.point, direction);
                return vec3_mul(attenuation, ray_color_whitted(scattered, scene, depth - 1));
            }
        }
    }
    
    // No hit - return background color
    return background_color(r, scene);
}

/**
 * @brief Power heuristic for MIS (beta = 2)
 */
inline double power_heuristic(double pdf_a, double pdf_b) {
    double a2 = pdf_a * pdf_a;
    double b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

color Renderer::ray_color_path(ray r, const Scene& scene, int depth) const {
    color throughput = {1.0, 1.0, 1.0};
    color accumulated = {0.0, 0.0, 0.0};
    ray current_ray = r;
    bool specular_bounce = false;
    bool was_diffuse_bounce = false;  // Track if last bounce was diffuse (for env MIS)
    
    for (int bounce = 0; bounce < depth; ++bounce) {
        hit_record rec;
        
        if (!scene.hit(current_ray, 0.001, 1e9, rec)) {
            // No hit - add background/environment contribution
            color bg = background_color(current_ray, scene);
            
            // If NEE+MIS is active and this is after a diffuse bounce, environment
            // was already handled by MIS in the previous iteration. Skip it here.
            bool env_handled_by_mis = settings_.use_nee && settings_.use_mis && 
                                      was_diffuse_bounce && 
                                      scene.environment && 
                                      scene.environment->has_importance_sampling();
            
            if (!env_handled_by_mis) {
                accumulated = vec3_add(accumulated, vec3_mul(throughput, bg));
            }
            break;
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        color surface_color = mat.get_albedo(rec.point, rec.u, rec.v);
        
        // Add emission on first hit, after specular bounce, or when NEE is disabled
        if (bounce == 0 || specular_bounce || !settings_.use_nee) {
            accumulated = vec3_add(accumulated, vec3_mul(throughput, mat.emission));
        }
        
        // Scatter based on material type
        vec3 scatter_dir;
        color attenuation = surface_color;
        specular_bounce = false;
        was_diffuse_bounce = false;
        double bsdf_pdf = 0.0;  // For MIS
        
        switch (mat.type) {
            case MaterialType::Lambertian: {
                // === NEE: Direct light sampling (geometry lights + environment) ===
                if (settings_.use_nee) {
                    LightSample ls = scene.sample_light(rec.point);
                    if (ls.valid) {
                        // Check if this is an environment sample (very large distance)
                        bool is_env_sample = ls.distance > 1e5;
                        
                        vec3 light_dir;
                        double light_dist;
                        
                        if (is_env_sample) {
                            // Environment sample: direction is what matters, not position
                            light_dir = vec3_normalize(vec3_sub(ls.position, rec.point));
                            light_dist = 1e9;  // Effectively infinite
                        } else {
                            // Geometry light sample
                            vec3 to_light = vec3_sub(ls.position, rec.point);
                            light_dist = vec3_length(to_light);
                            light_dir = vec3_scale(to_light, 1.0 / light_dist);
                        }
                        
                        double cos_surf = vec3_dot(rec.normal, light_dir);
                        
                        if (cos_surf > 0) {
                            // Shadow test
                            hit_record shadow_rec;
                            bool occluded = scene.hit(ray_create(rec.point, light_dir), 0.001, 
                                                      is_env_sample ? 1e9 : (light_dist - 0.001), shadow_rec);
                            
                            if (!occluded) {
                                double light_pdf_sa;
                                
                                if (is_env_sample) {
                                    // Environment: PDF is already in solid angle
                                    light_pdf_sa = ls.pdf;
                                } else {
                                    // Geometry light: convert area PDF to solid angle
                                    double cos_light = std::abs(vec3_dot(ls.normal, vec3_negate(light_dir)));
                                    if (cos_light <= 0) goto skip_nee;  // Light facing away
                                    light_pdf_sa = ls.pdf * light_dist * light_dist / cos_light;
                                }
                                
                                if (light_pdf_sa <= 0) goto skip_nee;
                                
                                // BSDF PDF for this direction (cosine-weighted)
                                double bsdf_pdf_light = cos_surf / 3.14159265358979323846;
                                
                                // MIS weight
                                double mis_w = 1.0;
                                if (settings_.use_mis) {
                                    mis_w = power_heuristic(light_pdf_sa, bsdf_pdf_light);
                                }
                                
                                // Direct: Le * (albedo/PI) * cos_surf / light_pdf_sa * mis_w
                                color direct = vec3_scale(
                                    vec3_mul(vec3_mul(ls.emission, surface_color), throughput),
                                    cos_surf / (3.14159265358979323846 * light_pdf_sa) * mis_w
                                );
                                accumulated = vec3_add(accumulated, direct);
                            }
                        }
                    }
                }
                skip_nee:
                
                // Cosine-weighted hemisphere sampling for indirect
                scatter_dir = vec3_add(rec.normal, random_unit_vector());
                if (vec3_length_squared(scatter_dir) < 1e-8) {
                    scatter_dir = rec.normal;
                }
                scatter_dir = vec3_normalize(scatter_dir);
                bsdf_pdf = vec3_dot(rec.normal, scatter_dir) / 3.14159265358979323846;
                was_diffuse_bounce = true;  // Mark for environment MIS handling
                break;
            }
            
            case MaterialType::Metal: {
                specular_bounce = true;
                vec3 reflected = vec3_reflect(vec3_normalize(current_ray.direction), rec.normal);
                if (mat.fuzz > 0.0) {
                    reflected = vec3_add(reflected, vec3_scale(random_unit_vector(), mat.fuzz));
                    specular_bounce = (mat.fuzz < 0.1);
                }
                scatter_dir = reflected;
                if (vec3_dot(scatter_dir, rec.normal) <= 0) {
                    break;
                }
                break;
            }
            
            case MaterialType::Dielectric: {
                specular_bounce = true;
                attenuation = {1.0, 1.0, 1.0};
                double refraction_ratio = rec.front_face ? 
                    (1.0 / mat.refraction_index) : mat.refraction_index;
                
                vec3 unit_dir = vec3_normalize(current_ray.direction);
                double cos_theta = std::fmin(vec3_dot(vec3_negate(unit_dir), rec.normal), 1.0);
                double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
                
                if (refraction_ratio * sin_theta > 1.0 || 
                    Material::reflectance(cos_theta, refraction_ratio) > random_double()) {
                    scatter_dir = vec3_reflect(unit_dir, rec.normal);
                } else {
                    scatter_dir = vec3_refract(unit_dir, rec.normal, refraction_ratio);
                }
                break;
            }
        }
        
        // Update throughput
        throughput = vec3_mul(throughput, attenuation);
        
        // Russian Roulette after bounce 3
        if (bounce > 3) {
            double p = std::fmax(throughput.x, std::fmax(throughput.y, throughput.z));
            if (p < 0.01) break;
            if (random_double() > p) break;
            throughput = vec3_scale(throughput, 1.0 / p);
        }
        
        // Create next ray
        ray next_ray = ray_create(rec.point, scatter_dir);
        
        // MIS: If BSDF sample hits emissive surface OR environment, add with MIS weight
        // Only when NEE is enabled (otherwise emission is added at hit / background)
        if (settings_.use_nee && settings_.use_mis && !specular_bounce && bsdf_pdf > 0) {
            hit_record next_rec;
            bool hit_geometry = scene.hit(next_ray, 0.001, 1e9, next_rec);
            
            if (hit_geometry) {
                // Hit geometry - check if emissive
                const Material& next_mat = scene.get_material(next_rec.material_id);
                if (next_mat.is_emissive()) {
                    // Compute light PDF for this hit
                    double hit_dist = next_rec.t;
                    double cos_light = std::abs(vec3_dot(next_rec.normal, vec3_negate(scatter_dir)));
                    double light_pdf_area = scene.light_pdf(rec.point, next_rec.point);
                    
                    if (light_pdf_area > 0 && cos_light > 0) {
                        double light_pdf_sa = light_pdf_area * hit_dist * hit_dist / cos_light;
                        
                        // MIS weight for BSDF sampling
                        double mis_w = power_heuristic(bsdf_pdf, light_pdf_sa);
                        
                        accumulated = vec3_add(accumulated, 
                            vec3_scale(vec3_mul(throughput, next_mat.emission), mis_w));
                    }
                }
            } else {
                // Missed geometry - hit environment
                if (scene.environment && scene.environment->has_importance_sampling()) {
                    // Get environment contribution with MIS weight
                    color env_emission = scene.environment->sample(scatter_dir);
                    double env_pdf = scene.env_light_pdf(scatter_dir);
                    
                    if (env_pdf > 0) {
                        double mis_w = power_heuristic(bsdf_pdf, env_pdf);
                        accumulated = vec3_add(accumulated, 
                            vec3_scale(vec3_mul(throughput, env_emission), mis_w));
                    }
                }
                // Note: If no importance sampling, background will be added normally
            }
        }
        
        current_ray = next_ray;
    }
    
    return accumulated;
}

color Renderer::background_color(ray r, const Scene& scene) const {
    // Sample environment map if present
    if (scene.environment && scene.environment->valid()) {
        return scene.environment->sample(r.direction);
    }
    
    // Fall back to sky gradient
    vec3 unit_direction = vec3_normalize(r.direction);
    double t = 0.5 * (unit_direction.y + 1.0);
    return vec3_lerp(settings_.background_bottom, settings_.background_top, t);
}

} // namespace raytracer
