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
#include "photon_map.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace raytracer {

/**
 * @brief Rendering mode
 */
enum class RenderMode {
    Whitted,    // Classic Whitted-style ray tracing (fast, explicit lights)
    PathTrace,  // Monte Carlo path tracing (slow, global illumination)
    BDPT,       // Bidirectional path tracing (better caustics, complex lighting)
    Spectral,   // Spectral path tracing (wavelength-dependent, dispersion)
    PLT,        // Polarized Light Tracing (tracks polarization state)
    PhotonMap,  // Two-pass photon mapping (global illumination + caustics)
    PathPhoton  // Hybrid: path tracing + caustic photons (best of both)
};

/**
 * @brief Tone mapping operators
 */
enum class ToneMapper {
    None,       // No tone mapping (just gamma correction)
    Reinhard,   // Simple Reinhard (L / (1 + L))
    ReinhardExtended,  // Extended Reinhard with white point
    ACES,       // ACES Filmic approximation
    Uncharted2  // Uncharted 2 filmic curve
};

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
     * @brief Apply tone mapping and exposure to all pixels (in-place)
     * @param mapper Tone mapping operator to use
     * @param exposure Exposure multiplier (applied before tone mapping)
     */
    void apply_tone_mapping(ToneMapper mapper, double exposure = 1.0);
    
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
        int max_depth = 50;         // Maximum recursion depth
        int samples_per_pixel = 16;  // Antialiasing samples (1 = no AA)
        
        // Background settings (from scene)
        color background_color = {0.0, 0.0, 0.0};  // Solid background
        color background_top = {0.5, 0.7, 1.0};    // Gradient top (sky)
        color background_bottom = {1.0, 1.0, 1.0}; // Gradient bottom (horizon)
        bool use_background_gradient = false;       // Use gradient vs solid
        
        RenderMode mode = RenderMode::Whitted;     // Rendering algorithm
        bool use_nee = true;         // Next Event Estimation (direct light sampling)
        bool use_mis = true;         // Multiple Importance Sampling
        bool use_mnee = true;        // Manifold NEE (caustics through glass)
        bool use_hwss = true;        // Hero Wavelength Spectral Sampling
        ToneMapper tone_mapper = ToneMapper::ACES;  // Tone mapping operator
        double exposure = 1.0;       // Exposure adjustment (multiplier before tone mapping)
        double clamp_max = 10.0;     // Firefly clamping for BDPT (0 = disabled)
        int wavelength_samples = 8;  // Wavelength samples per pixel (spectral mode)
        
        // PLT settings
        bool plt_visualize_polarization = false;  // Show polarization via hue
        
        // Photon mapping settings
        size_t photon_count = 100000;      // Global photons
        size_t caustic_photon_count = 50000; // Caustic photons
        int photon_gather_count = 100;      // Photons to gather per estimate
        float photon_gather_radius = 0.5f;  // Maximum gather radius
        float caustic_gather_radius = 0.1f; // Caustic gather radius (sharper)
        bool photon_final_gather = false;   // Use final gathering
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
     * @brief Calculate color for a ray - Whitted style (recursive)
     */
    color ray_color_whitted(ray r, const Scene& scene, int depth) const;
    
    /**
     * @brief Calculate color for a ray - Path tracing (recursive)
     */
    color ray_color_path(ray r, const Scene& scene, int depth) const;
    
    /**
     * @brief Calculate color for a ray - Path tracing with caustic photon lookup
     * @param caustic_map Caustic photon map for direct visualization of caustics
     */
    color ray_color_path_with_caustics(ray r, const Scene& scene, int depth,
                                       const PhotonMap& caustic_map, 
                                       int gather_count, float gather_radius) const;
    
    /**
     * @brief Calculate color for a ray - Polarized Light Tracing
     */
    color ray_color_plt(ray r, const Scene& scene, int depth) const;
    
    /**
     * @brief Calculate radiance for a single wavelength - Spectral PLT
     * @param wavelength_nm Wavelength in nanometers
     */
    double ray_radiance_plt_spectral(ray r, const Scene& scene, int depth, double wavelength_nm) const;
    
    /**
     * @brief Calculate background color (sky gradient or environment map)
     * @param r Ray direction for environment map sampling
     * @param scene Scene containing optional environment map
     */
    color background_color(ray r, const Scene& scene) const;
};

} // namespace raytracer
