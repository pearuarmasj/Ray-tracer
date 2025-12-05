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

namespace raytracer {

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
    
    std::cout << "Rendering " << settings_.width << "x" << settings_.height 
              << " image..." << std::endl;
    
    for (int y = 0; y < settings_.height; ++y) {
        // Progress indicator
        std::cout << "\rScanlines remaining: " << (settings_.height - y) << ' ' << std::flush;
        
        for (int x = 0; x < settings_.width; ++x) {
            color pixel_color = vec3_zero();
            
            for (int s = 0; s < settings_.samples_per_pixel; ++s) {
                // For single sample, use center of pixel
                // For multiple samples, could add random offset (not implemented for simplicity)
                double u = (static_cast<double>(x) + 0.5) / settings_.width;
                double v = (static_cast<double>(y) + 0.5) / settings_.height;
                
                ray r = camera.get_ray(u, v);
                pixel_color = vec3_add(pixel_color, ray_color(r, scene, settings_.max_depth));
            }
            
            // Average samples
            pixel_color = vec3_scale(pixel_color, 1.0 / settings_.samples_per_pixel);
            image.set_pixel(x, y, pixel_color);
        }
    }
    
    std::cout << "\nDone!" << std::endl;
    return image;
}

color Renderer::ray_color(ray r, const Scene& scene, int depth) const {
    // Exceeded recursion limit
    if (depth <= 0) {
        return vec3_zero();
    }
    
    hit_record rec;
    
    // Check for intersection (t_min = 0.001 to avoid shadow acne)
    if (scene.hit(r, 0.001, 1e9, rec)) {
        const Material& mat = scene.get_material(rec.material_id);
        
        switch (mat.type) {
            case MaterialType::Lambertian: {
                // Diffuse reflection - simple Lambertian approximation
                // For Whitted-style, we use the normal direction as scattered direction
                // This gives a simple diffuse look without Monte Carlo sampling
                vec3 target = vec3_add(rec.point, rec.normal);
                ray scattered = ray_create(rec.point, vec3_sub(target, rec.point));
                color attenuation = mat.albedo;
                color scattered_color = ray_color(scattered, scene, depth - 1);
                return vec3_mul(attenuation, vec3_scale(scattered_color, 0.5));
            }
            
            case MaterialType::Metal: {
                // Specular reflection
                vec3 reflected = vec3_reflect(vec3_normalize(r.direction), rec.normal);
                ray scattered = ray_create(rec.point, reflected);
                
                // Only reflect if reflected ray goes away from surface
                if (vec3_dot(scattered.direction, rec.normal) > 0) {
                    color attenuation = mat.albedo;
                    return vec3_mul(attenuation, ray_color(scattered, scene, depth - 1));
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
                
                if (cannot_refract || Material::reflectance(cos_theta, refraction_ratio) > 0.5) {
                    // Must reflect (total internal reflection or Fresnel)
                    direction = vec3_reflect(unit_direction, rec.normal);
                } else {
                    // Refract
                    direction = vec3_refract(unit_direction, rec.normal, refraction_ratio);
                }
                
                ray scattered = ray_create(rec.point, direction);
                return vec3_mul(attenuation, ray_color(scattered, scene, depth - 1));
            }
        }
    }
    
    // No hit - return background color
    return background_color(r);
}

color Renderer::background_color(ray r) const {
    vec3 unit_direction = vec3_normalize(r.direction);
    double t = 0.5 * (unit_direction.y + 1.0);
    return vec3_lerp(settings_.background_bottom, settings_.background_top, t);
}

} // namespace raytracer
