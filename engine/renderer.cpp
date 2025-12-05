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

inline double random_double() {
    return dist(rng);
}

inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}

// Generate random point in unit sphere (rejection sampling)
inline vec3 random_in_unit_sphere() {
    while (true) {
        vec3 p = {random_double(-1, 1), random_double(-1, 1), random_double(-1, 1)};
        if (vec3_length_squared(p) < 1.0)
            return p;
    }
}

// Generate random unit vector (for true Lambertian)
inline vec3 random_unit_vector() {
    return vec3_normalize(random_in_unit_sphere());
}

// Generate random vector in hemisphere around normal
inline vec3 random_on_hemisphere(vec3 normal) {
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
              << " image (" << mode_name << ")";
    
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
    return background_color(r);
}

color Renderer::ray_color_path(ray r, const Scene& scene, int depth) const {
    // Exceeded recursion limit
    if (depth <= 0) {
        return vec3_zero();
    }
    
    hit_record rec;
    
    // Check for intersection (t_min = 0.001 to avoid shadow acne)
    if (scene.hit(r, 0.001, 1e9, rec)) {
        const Material& mat = scene.get_material(rec.material_id);
        
        // Get emission (for light sources)
        color emitted = mat.emission;
        
        // Get color from texture or albedo
        color surface_color = mat.get_albedo(rec.point, rec.u, rec.v);
        
        switch (mat.type) {
            case MaterialType::Lambertian: {
                // Random scatter in hemisphere
                vec3 scatter_direction = vec3_add(rec.normal, random_unit_vector());
                if (vec3_length_squared(scatter_direction) < 1e-8) {
                    scatter_direction = rec.normal;
                }
                
                ray scattered = ray_create(rec.point, scatter_direction);
                color incoming = ray_color_path(scattered, scene, depth - 1);
                
                // emission + albedo * incoming
                return vec3_add(emitted, vec3_mul(surface_color, incoming));
            }
            
            case MaterialType::Metal: {
                // Specular reflection with optional fuzz
                vec3 reflected = vec3_reflect(vec3_normalize(r.direction), rec.normal);
                
                // Add fuzz
                if (mat.fuzz > 0.0) {
                    vec3 fuzz_vec = vec3_scale(random_unit_vector(), mat.fuzz);
                    reflected = vec3_add(reflected, fuzz_vec);
                }
                
                ray scattered = ray_create(rec.point, reflected);
                
                // Only reflect if reflected ray goes away from surface
                if (vec3_dot(scattered.direction, rec.normal) > 0) {
                    color incoming = ray_color_path(scattered, scene, depth - 1);
                    return vec3_add(emitted, vec3_mul(surface_color, incoming));
                }
                return emitted;
            }
            
            case MaterialType::Dielectric: {
                // Refraction (glass-like material)
                double refraction_ratio = rec.front_face ? 
                    (1.0 / mat.refraction_index) : mat.refraction_index;
                
                vec3 unit_direction = vec3_normalize(r.direction);
                double cos_theta = std::fmin(vec3_dot(vec3_negate(unit_direction), rec.normal), 1.0);
                double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
                
                bool cannot_refract = refraction_ratio * sin_theta > 1.0;
                vec3 direction;
                
                // Use random number for probabilistic Fresnel reflection
                if (cannot_refract || Material::reflectance(cos_theta, refraction_ratio) > random_double()) {
                    direction = vec3_reflect(unit_direction, rec.normal);
                } else {
                    direction = vec3_refract(unit_direction, rec.normal, refraction_ratio);
                }
                
                ray scattered = ray_create(rec.point, direction);
                color incoming = ray_color_path(scattered, scene, depth - 1);
                return vec3_add(emitted, incoming);  // Glass doesn't absorb much
            }
        }
    }
    
    // No hit - return background color (acts as environment light)
    return background_color(r);
}

color Renderer::background_color(ray r) const {
    vec3 unit_direction = vec3_normalize(r.direction);
    double t = 0.5 * (unit_direction.y + 1.0);
    return vec3_lerp(settings_.background_bottom, settings_.background_top, t);
}

} // namespace raytracer
