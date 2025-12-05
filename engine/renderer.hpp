/**
 * @file renderer.hpp
 * @brief Whitted-style ray tracer renderer
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/hit.h"
}

#include "scene.hpp"
#include "material.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace raytracer {

/**
 * @brief Image buffer for rendering
 */
struct Image {
    int width;
    int height;
    std::vector<color> pixels;
    
    Image(int w, int h) : width(w), height(h), pixels(w * h) {}
    
    /**
     * @brief Set pixel color
     */
    void set_pixel(int x, int y, color c) {
        pixels[y * width + x] = c;
    }
    
    /**
     * @brief Get pixel color
     */
    color get_pixel(int x, int y) const {
        return pixels[y * width + x];
    }
    
    /**
     * @brief Write image to PPM file
     * @param filename Output filename
     * @return true on success
     */
    bool write_ppm(const std::string& filename) const;
    
    /**
     * @brief Write image to PNG file
     * @param filename Output filename
     * @return true on success
     */
    bool write_png(const std::string& filename) const;
};

/**
 * @brief Whitted-style ray tracer renderer
 */
class Renderer {
public:
    /**
     * @brief Render settings
     */
    struct Settings {
        int width = 2560;
        int height = 1440;
        int max_depth = 10;         // Maximum recursion depth
        int samples_per_pixel = 1;  // Antialiasing samples (1 = no AA)
        color background_top = {0.5, 0.7, 1.0};    // Sky gradient top
        color background_bottom = {1.0, 1.0, 1.0}; // Sky gradient bottom
    };
    
    Renderer() = default;
    explicit Renderer(const Settings& settings) : settings_(settings) {}
    
    /**
     * @brief Render a scene
     * @param scene The scene to render
     * @param camera The camera to use
     * @return Rendered image
     */
    Image render(const Scene& scene, const Camera& camera) const;
    
    /**
     * @brief Get/set settings
     */
    Settings& settings() { return settings_; }
    const Settings& settings() const { return settings_; }
    
private:
    Settings settings_;
    
    /**
     * @brief Calculate color for a ray (recursive)
     */
    color ray_color(ray r, const Scene& scene, int depth) const;
    
    /**
     * @brief Calculate background color (sky gradient)
     */
    color background_color(ray r) const;
};

} // namespace raytracer
