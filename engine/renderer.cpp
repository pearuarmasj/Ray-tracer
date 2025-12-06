/**
 * @file renderer.cpp
 * @brief Implementation of Whitted-style ray tracer
 */

#include "renderer.hpp"
#include "bdpt.hpp"
#include "spectral/spectral_renderer.hpp"
#include "spectral/hwss.hpp"
#include "spectral/mnee.hpp"
#include "plt/plt.hpp"
#include "photon_integrator.hpp"
#include "profiler.hpp"
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

// Forward declaration
static color wavelength_to_rgb(double wavelength_nm);

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

/**
 * @brief Apply normal map perturbation to hit record
 * 
 * Modifies rec.normal in-place if the material has a normal map.
 */
inline void apply_normal_map(hit_record& rec, const Material& mat) {
    if (!mat.has_normal_map) {
        return;
    }
    
    // Sample the normal map (returns tangent-space normal)
    vec3 tangent_normal = mat.normal_map.sample(rec.u, rec.v);
    
    // Build TBN (Tangent, Bitangent, Normal) matrix
    vec3 N = rec.normal;
    vec3 T = rec.tangent;
    
    // Ensure tangent is orthogonal to normal (Gram-Schmidt)
    T = vec3_normalize(vec3_sub(T, vec3_scale(N, vec3_dot(N, T))));
    vec3 B = vec3_cross(N, T);
    
    // Transform tangent-space normal to world space
    rec.normal = vec3_normalize({
        T.x * tangent_normal.x + B.x * tangent_normal.y + N.x * tangent_normal.z,
        T.y * tangent_normal.x + B.y * tangent_normal.y + N.y * tangent_normal.z,
        T.z * tangent_normal.x + B.z * tangent_normal.y + N.z * tangent_normal.z
    });
}

// ==================== Tone Mapping Operators ====================

/**
 * @brief Simple Reinhard tone mapping
 * Maps [0, inf) to [0, 1)
 */
inline double reinhard(double x) {
    return x / (1.0 + x);
}

/**
 * @brief Extended Reinhard with white point
 * Allows white_point values to map to 1.0
 */
inline double reinhard_extended(double x, double white_point) {
    double wp2 = white_point * white_point;
    return x * (1.0 + x / wp2) / (1.0 + x);
}

/**
 * @brief ACES Filmic approximation (by Krzysztof Narkowicz)
 * Popular film-like tone curve
 */
inline double aces_filmic(double x) {
    double a = 2.51;
    double b = 0.03;
    double c = 2.43;
    double d = 0.59;
    double e = 0.14;
    return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

/**
 * @brief Uncharted 2 tone mapping helper
 */
inline double uncharted2_partial(double x) {
    double A = 0.15;
    double B = 0.50;
    double C = 0.10;
    double D = 0.20;
    double E = 0.02;
    double F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

/**
 * @brief Uncharted 2 filmic tone mapping
 */
inline double uncharted2(double x) {
    double exposure_bias = 2.0;
    double white = 11.2;
    double curr = uncharted2_partial(x * exposure_bias);
    double white_scale = 1.0 / uncharted2_partial(white);
    return curr * white_scale;
}

/**
 * @brief Apply tone mapping to a single color
 */
color apply_tone_mapper(color c, ToneMapper mapper) {
    switch (mapper) {
        case ToneMapper::Reinhard:
            return {reinhard(c.x), reinhard(c.y), reinhard(c.z)};
        
        case ToneMapper::ReinhardExtended: {
            double white = 4.0;  // Adjust as needed
            return {
                reinhard_extended(c.x, white),
                reinhard_extended(c.y, white),
                reinhard_extended(c.z, white)
            };
        }
        
        case ToneMapper::ACES:
            return {aces_filmic(c.x), aces_filmic(c.y), aces_filmic(c.z)};
        
        case ToneMapper::Uncharted2:
            return {uncharted2(c.x), uncharted2(c.y), uncharted2(c.z)};
        
        case ToneMapper::None:
        default:
            // Just clamp to [0, 1]
            return {
                std::clamp(c.x, 0.0, 1.0),
                std::clamp(c.y, 0.0, 1.0),
                std::clamp(c.z, 0.0, 1.0)
            };
    }
}

void Image::apply_tone_mapping(ToneMapper mapper, double exposure) {
    for (int i = 0; i < width * height; ++i) {
        // Apply exposure
        color c = {
            pixels[i].x * exposure,
            pixels[i].y * exposure,
            pixels[i].z * exposure
        };
        
        // Apply tone mapping
        pixels[i] = apply_tone_mapper(c, mapper);
    }
}

// ==================== Image Output ====================

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
            // Tone mapping already puts values in [0,1] range
            double r = std::sqrt(std::clamp(c.x, 0.0, 1.0));
            double g = std::sqrt(std::clamp(c.y, 0.0, 1.0));
            double b = std::sqrt(std::clamp(c.z, 0.0, 1.0));
            
            // Convert to 0-255
            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);
            
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
            // Tone mapping already puts values in [0,1] range
            double r = std::sqrt(std::clamp(c.x, 0.0, 1.0));
            double g = std::sqrt(std::clamp(c.y, 0.0, 1.0));
            double b = std::sqrt(std::clamp(c.z, 0.0, 1.0));
            
            // Convert to 0-255
            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);
            
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
    
    // Reset profiler for this render
    Profiler::instance().reset();
    Timer total_timer;
    
    const char* mode_name;
    switch (settings_.mode) {
        case RenderMode::PathTrace: mode_name = "Path Tracing"; break;
        case RenderMode::BDPT: mode_name = "Bidirectional Path Tracing"; break;
        case RenderMode::Spectral: mode_name = "Spectral Path Tracing"; break;
        case RenderMode::PLT: mode_name = "Polarized Light Tracing"; break;
        case RenderMode::PhotonMap: mode_name = "Photon Mapping"; break;
        case RenderMode::PathPhoton: mode_name = "Path Tracing + Caustic Photons"; break;
        default: mode_name = "Whitted"; break;
    }
    
    std::cout << "Rendering " << settings_.width << "x" << settings_.height 
              << " image (" << mode_name;
    if (settings_.mode == RenderMode::PathTrace) {
        std::cout << ", NEE=" << (settings_.use_nee ? "on" : "off")
                  << ", MIS=" << (settings_.use_mis ? "on" : "off");
    }
    if (settings_.mode == RenderMode::Spectral) {
        std::cout << ", " << settings_.wavelength_samples << " wavelength samples";
    }
    std::cout << ")";
    
#ifdef _OPENMP
    std::cout << " using " << omp_get_max_threads() << " threads";
#endif
    std::cout << "..." << std::endl;
    
    // Build BVH before parallel rendering (must be single-threaded)
    {
        Timer bvh_timer;
        scene.build_bvh();
        Profiler::instance().record("BVH Build", Profiler::Duration(bvh_timer.elapsed_ms()));
    };
    
    // Create BDPT integrator if needed
    BDPTIntegrator bdpt;
    if (settings_.mode == RenderMode::BDPT) {
        BDPTIntegrator::Settings bdpt_settings;
        bdpt_settings.max_eye_depth = settings_.max_depth;
        bdpt_settings.max_light_depth = settings_.max_depth;
        bdpt_settings.use_mis = settings_.use_mis;
        bdpt_settings.clamp_max = settings_.clamp_max;
        bdpt = BDPTIntegrator(bdpt_settings);
    }
    
    // Create spectral integrator and materials if needed
    spectral::SpectralIntegrator spectral_integrator;
    std::vector<spectral::SpectralMaterial> spectral_materials;
    if (settings_.mode == RenderMode::Spectral) {
        spectral::SpectralIntegrator::Settings spec_settings;
        spec_settings.max_depth = settings_.max_depth;
        spectral_integrator = spectral::SpectralIntegrator(spec_settings);
        
        // Create spectral materials from scene materials
        for (const auto& mat : scene.materials) {
            spectral::SpectralMaterial smat;
            
            // Set albedo spectrum
            smat.albedo_spectrum = spectral::data::RGBSpectrum(
                mat.albedo.x, mat.albedo.y, mat.albedo.z);
            
            // For dielectrics, use dispersive glass (SF11 for visible dispersion)
            if (mat.type == MaterialType::Dielectric) {
                // High-dispersion flint glass for nice rainbows
                smat.dispersion = spectral::Dispersion(spectral::materials::SF11());
            } else {
                // Non-dispersive
                smat.dispersion = spectral::Dispersion(mat.refraction_index);
            }
            
            spectral_materials.push_back(smat);
        }
    }
    
    // Create photon integrator if needed
    PhotonIntegrator photon_integrator;
    if (settings_.mode == RenderMode::PhotonMap || settings_.mode == RenderMode::PathPhoton) {
        // For hybrid mode, only trace caustic photons (path tracing handles the rest)
        bool caustic_only = (settings_.mode == RenderMode::PathPhoton);
        
        if (caustic_only) {
            photon_integrator.settings.num_global_photons = 0;  // No global photons needed
        } else {
            photon_integrator.settings.num_global_photons = settings_.photon_count;
        }
        photon_integrator.settings.num_caustic_photons = settings_.caustic_photon_count;
        photon_integrator.settings.gather_count = settings_.photon_gather_count;
        photon_integrator.settings.gather_radius = settings_.photon_gather_radius;
        photon_integrator.settings.caustic_radius = settings_.caustic_gather_radius;
        photon_integrator.settings.use_final_gather = settings_.photon_final_gather;
        
        std::cout << "\nTracing " << (caustic_only ? "caustic " : "") << "photons..." << std::flush;
        {
            Timer photon_timer;
            photon_integrator.trace_photons(scene);
            Profiler::instance().record("Photon Tracing", Profiler::Duration(photon_timer.elapsed_ms()));
        }
        std::cout << " Done.";
        if (!caustic_only) {
            std::cout << " Global: " << photon_integrator.global_map.size() << ",";
        }
        std::cout << " Caustic: " << photon_integrator.caustic_map.size() << std::endl;
    }
    
    Timer render_timer;  // Time just the pixel rendering loop
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
                switch (settings_.mode) {
                    case RenderMode::PathTrace:
                        pixel_color = vec3_add(pixel_color, ray_color_path(r, scene, settings_.max_depth));
                        break;
                    case RenderMode::BDPT:
                        pixel_color = vec3_add(pixel_color, bdpt.Li(r, scene));
                        break;
                    case RenderMode::Spectral: {
                        color spec_color = spectral_integrator.render_pixel_spectral(
                            r, scene, spectral_materials, settings_.wavelength_samples);
                        pixel_color = vec3_add(pixel_color, spec_color);
                        break;
                    }
                    case RenderMode::PLT: {
                        // Use HWSS (Hero Wavelength Spectral Sampling) for correlated wavelengths
                        // Wavelength count is configurable via settings
                        int num_wavelengths = std::max(4, settings_.wavelength_samples);
                        double xi = random_double();
                        auto lambdas = HWSS::sample_wavelengths_dynamic(num_wavelengths, xi);
                        std::vector<double> radiances(num_wavelengths);
                        
                        // Evaluate radiance for all correlated wavelengths
                        for (int w = 0; w < num_wavelengths; ++w) {
                            radiances[w] = ray_radiance_plt_spectral(r, scene, settings_.max_depth, lambdas[w]);
                        }
                        
                        // Convert to RGB using HWSS weighting (with optional spectral MIS)
                        auto rgb = HWSS::radiances_to_rgb_dynamic(radiances, lambdas, false);
                        color plt_color = {rgb[0], rgb[1], rgb[2]};
                        
                        pixel_color = vec3_add(pixel_color, plt_color);
                        break;
                    }
                    case RenderMode::PhotonMap: {
                        // Use photon mapping for radiance estimation
                        color pm_color = photon_integrator.estimate_radiance(scene, r);
                        pixel_color = vec3_add(pixel_color, pm_color);
                        break;
                    }
                    case RenderMode::PathPhoton: {
                        // Hybrid: path tracing + caustic photon lookup at diffuse surfaces
                        color hybrid_color = ray_color_path_with_caustics(
                            r, scene, settings_.max_depth,
                            photon_integrator.caustic_map,
                            settings_.photon_gather_count,
                            settings_.caustic_gather_radius);
                        pixel_color = vec3_add(pixel_color, hybrid_color);
                        break;
                    }
                    default:
                        pixel_color = vec3_add(pixel_color, ray_color_whitted(r, scene, settings_.max_depth));
                        break;
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
    
    // Record pixel rendering time
    Profiler::instance().record("Pixel Rendering", Profiler::Duration(render_timer.elapsed_ms()));
    
    std::cout << "\nApplying tone mapping..." << std::flush;
    {
        Timer tonemap_timer;
        image.apply_tone_mapping(settings_.tone_mapper, settings_.exposure);
        Profiler::instance().record("Tone Mapping", Profiler::Duration(tonemap_timer.elapsed_ms()));
    }
    
    // Record total time and print report
    Profiler::instance().record("Total Render", Profiler::Duration(total_timer.elapsed_ms()));
    
    std::cout << " Done!" << std::endl;
    
    // Print profiling report
    Profiler::instance().report();
    
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
        
        // Apply normal map if present
        apply_normal_map(rec, mat);
        
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
    double bsdf_pdf = 0.0;  // Track BSDF PDF from previous bounce for MIS
    
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
        
        // Check if we hit an area light
        if (scene.is_area_light(rec.material_id)) {
            color light_emission = scene.get_area_light_emission(rec.material_id);
            
            // Add emission on first hit, after specular bounce, or when NEE is disabled
            // When NEE+MIS is enabled on diffuse bounce, need MIS weight
            if (bounce == 0 || specular_bounce || !settings_.use_nee) {
                accumulated = vec3_add(accumulated, vec3_mul(throughput, light_emission));
            } else if (settings_.use_mis && was_diffuse_bounce && bsdf_pdf > 0) {
                // MIS weight for BSDF sampling hitting area light
                double hit_dist = rec.t;
                double cos_light = std::abs(vec3_dot(rec.normal, vec3_negate(vec3_normalize(current_ray.direction))));
                double light_pdf_area = scene.light_pdf(vec3_zero(), rec.point);  // from_point not used for area lights
                
                if (light_pdf_area > 0 && cos_light > 0) {
                    double light_pdf_sa = light_pdf_area * hit_dist * hit_dist / cos_light;
                    double mis_w = power_heuristic(bsdf_pdf, light_pdf_sa);
                    accumulated = vec3_add(accumulated, 
                        vec3_scale(vec3_mul(throughput, light_emission), mis_w));
                }
            }
            break;  // Area lights don't scatter
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        
        // Apply normal map if present
        apply_normal_map(rec, mat);
        
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
        bsdf_pdf = 0.0;  // Reset for MIS
        
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
                    
                    // === MNEE: Try to find paths through specular surfaces ===
                    // This handles caustics where regular NEE fails due to dielectric blockers
                    if (settings_.use_mnee) {
                        auto [mnee_contrib, mnee_used] = MNEE::evaluate(
                            scene, rec.point, rec.normal, 
                            vec3_negate(vec3_normalize(current_ray.direction)), mat);
                        
                        if (mnee_used) {
                            // Apply surface BSDF (Lambertian: albedo/PI)
                            color mnee_direct = vec3_mul(
                                vec3_mul(mnee_contrib, surface_color),
                                throughput
                            );
                            mnee_direct = vec3_scale(mnee_direct, 1.0 / 3.14159265358979323846);
                            accumulated = vec3_add(accumulated, mnee_direct);
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

// Path tracing with caustic photon lookup at diffuse surfaces
color Renderer::ray_color_path_with_caustics(ray r, const Scene& scene, int depth,
                                             const PhotonMap& caustic_map,
                                             int gather_count, float gather_radius) const {
    color throughput = {1.0, 1.0, 1.0};
    color accumulated = {0.0, 0.0, 0.0};
    ray current_ray = r;
    bool specular_bounce = false;
    
    for (int bounce = 0; bounce < depth; ++bounce) {
        hit_record rec;
        
        if (!scene.hit(current_ray, 0.001, 1e9, rec)) {
            accumulated = vec3_add(accumulated, vec3_mul(throughput, background_color(current_ray, scene)));
            break;
        }
        
        // Check if we hit an area light
        if (scene.is_area_light(rec.material_id)) {
            color light_emission = scene.get_area_light_emission(rec.material_id);
            if (bounce == 0 || specular_bounce || !settings_.use_nee) {
                accumulated = vec3_add(accumulated, vec3_mul(throughput, light_emission));
            }
            break;
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        apply_normal_map(rec, mat);
        color surface_color = mat.get_albedo(rec.point, rec.u, rec.v);
        
        // Emission on first hit or after specular
        if (bounce == 0 || specular_bounce || !settings_.use_nee) {
            accumulated = vec3_add(accumulated, vec3_mul(throughput, mat.emission));
        }
        
        vec3 scatter_dir;
        color attenuation = surface_color;
        specular_bounce = false;
        
        switch (mat.type) {
            case MaterialType::Lambertian: {
                // === CAUSTIC PHOTON LOOKUP ===
                // Add caustic contribution from photon map at diffuse surfaces
                if (caustic_map.size() > 0) {
                    color caustic = caustic_map.estimate_irradiance(
                        rec.point, rec.normal, gather_count, gather_radius);
                    // Modulate by surface albedo and throughput
                    accumulated = vec3_add(accumulated, 
                        vec3_mul(throughput, vec3_mul(surface_color, caustic)));
                }
                
                // NEE for direct lighting (same as ray_color_path)
                if (settings_.use_nee) {
                    LightSample ls = scene.sample_light(rec.point);
                    if (ls.valid) {
                        vec3 to_light = vec3_sub(ls.position, rec.point);
                        double light_dist = vec3_length(to_light);
                        vec3 light_dir = vec3_scale(to_light, 1.0 / light_dist);
                        double cos_surf = vec3_dot(rec.normal, light_dir);
                        
                        if (cos_surf > 0) {
                            hit_record shadow_rec;
                            bool is_env = ls.distance > 1e5;
                            double shadow_dist = is_env ? 1e9 : light_dist - 0.001;
                            bool occluded = scene.hit(ray_create(rec.point, light_dir), 
                                                     0.001, shadow_dist, shadow_rec);
                            
                            if (!occluded) {
                                double cos_light = std::abs(vec3_dot(ls.normal, vec3_negate(light_dir)));
                                double falloff = is_env ? 1.0 : 1.0 / (light_dist * light_dist);
                                color direct = vec3_scale(
                                    vec3_mul(surface_color, ls.emission),
                                    cos_surf * falloff * cos_light / (ls.pdf * M_PI));
                                accumulated = vec3_add(accumulated, vec3_mul(throughput, direct));
                            }
                        }
                    }
                }
                
                // Cosine-weighted hemisphere sampling
                scatter_dir = vec3_add(rec.normal, random_unit_vector());
                if (vec3_length_squared(scatter_dir) < 1e-8) scatter_dir = rec.normal;
                scatter_dir = vec3_normalize(scatter_dir);
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
                if (vec3_dot(scatter_dir, rec.normal) <= 0) break;
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
            
            default:
                return accumulated;
        }
        
        throughput = vec3_mul(throughput, attenuation);
        
        // Russian Roulette
        if (bounce > 3) {
            double p = std::fmax(throughput.x, std::fmax(throughput.y, throughput.z));
            if (p < 0.01 || random_double() > p) break;
            throughput = vec3_scale(throughput, 1.0 / p);
        }
        
        current_ray = ray_create(rec.point, scatter_dir);
    }
    
    return accumulated;
}

color Renderer::background_color(ray r, const Scene& scene) const {
    // Priority 1: Environment map (HDR lighting)
    if (scene.environment && scene.environment->valid()) {
        return scene.environment->sample(r.direction);
    }
    
    // Priority 2: Use scene-defined background
    if (settings_.use_background_gradient) {
        // Gradient from bottom to top based on ray Y direction
        vec3 unit_direction = vec3_normalize(r.direction);
        double t = 0.5 * (unit_direction.y + 1.0);
        return vec3_lerp(settings_.background_bottom, settings_.background_top, t);
    }
    
    // Solid background color
    return settings_.background_color;
}

// ==================== Polarized Light Tracing ====================

color Renderer::ray_color_plt(ray r, const Scene& scene, int depth) const {
    using namespace plt;
    
    // Start with unpolarized light (intensity 1)
    Vec3 direction(r.direction);
    Beam beam = Beam::unpolarized(1.0f, direction);
    
    color throughput = {1.0, 1.0, 1.0};
    color accumulated = {0.0, 0.0, 0.0};
    ray current_ray = r;
    bool specular_bounce = false;
    bool was_diffuse_bounce = false;
    double bsdf_pdf = 0.0;
    
    for (int bounce = 0; bounce < depth; ++bounce) {
        hit_record rec;
        
        if (!scene.hit(current_ray, 0.001, 1e9, rec)) {
            // No hit - add background contribution
            color bg = background_color(current_ray, scene);
            float I = std::max(0.0f, beam.stokes.I);
            
            // If NEE+MIS handled environment on diffuse bounce, skip
            bool env_handled_by_mis = settings_.use_nee && settings_.use_mis && 
                                      was_diffuse_bounce && 
                                      scene.environment && 
                                      scene.environment->has_importance_sampling();
            
            if (!env_handled_by_mis) {
                accumulated = vec3_add(accumulated, vec3_scale(vec3_mul(throughput, bg), I));
            }
            break;
        }
        
        // Check if we hit an area light
        if (scene.is_area_light(rec.material_id)) {
            color light_emission = scene.get_area_light_emission(rec.material_id);
            float I = std::max(0.0f, beam.stokes.I);
            
            if (bounce == 0 || specular_bounce || !settings_.use_nee) {
                accumulated = vec3_add(accumulated, vec3_scale(vec3_mul(throughput, light_emission), I));
            } else if (settings_.use_mis && was_diffuse_bounce && bsdf_pdf > 0) {
                // MIS weight for BSDF sampling hitting area light
                double hit_dist = rec.t;
                double cos_light = std::abs(vec3_dot(rec.normal, vec3_negate(vec3_normalize(current_ray.direction))));
                double light_pdf_area = scene.light_pdf(vec3_zero(), rec.point);
                
                if (light_pdf_area > 0 && cos_light > 0) {
                    double light_pdf_sa = light_pdf_area * hit_dist * hit_dist / cos_light;
                    double mis_w = power_heuristic(bsdf_pdf, light_pdf_sa);
                    accumulated = vec3_add(accumulated, 
                        vec3_scale(vec3_mul(throughput, light_emission), I * mis_w));
                }
            }
            break;
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        
        // Apply normal map if present
        apply_normal_map(rec, mat);
        
        // Check for emission
        if (mat.is_emissive()) {
            float I = std::max(0.0f, beam.stokes.I);
            if (bounce == 0 || specular_bounce || !settings_.use_nee) {
                accumulated = vec3_add(accumulated, vec3_scale(vec3_mul(throughput, mat.emission), I));
            }
        }
        
        Vec3 normal(rec.normal);
        Vec3 wi = Vec3(current_ray.direction).normalized();
        Vec3 wo;
        
        // Get surface color
        color surface_color = mat.get_albedo(rec.point, rec.u, rec.v);
        float albedo_avg = static_cast<float>((surface_color.x + surface_color.y + surface_color.z) / 3.0);
        
        specular_bounce = false;
        was_diffuse_bounce = false;
        bsdf_pdf = 0.0;
        
        switch (mat.type) {
            case MaterialType::Lambertian: {
                // === NEE: Direct light sampling ===
                if (settings_.use_nee) {
                    LightSample ls = scene.sample_light(rec.point);
                    if (ls.valid) {
                        bool is_env_sample = ls.distance > 1e5;
                        
                        vec3 light_dir;
                        double light_dist;
                        
                        if (is_env_sample) {
                            light_dir = vec3_normalize(vec3_sub(ls.position, rec.point));
                            light_dist = 1e9;
                        } else {
                            vec3 to_light = vec3_sub(ls.position, rec.point);
                            light_dist = vec3_length(to_light);
                            light_dir = vec3_scale(to_light, 1.0 / light_dist);
                        }
                        
                        double cos_surf = vec3_dot(rec.normal, light_dir);
                        
                        if (cos_surf > 0) {
                            hit_record shadow_rec;
                            bool occluded = scene.hit(ray_create(rec.point, light_dir), 0.001, 
                                                      is_env_sample ? 1e9 : (light_dist - 0.001), shadow_rec);
                            
                            if (!occluded) {
                                double light_pdf_sa;
                                
                                if (is_env_sample) {
                                    light_pdf_sa = ls.pdf;
                                } else {
                                    double cos_light = std::abs(vec3_dot(ls.normal, vec3_negate(light_dir)));
                                    if (cos_light <= 0) goto skip_nee_plt;
                                    light_pdf_sa = ls.pdf * light_dist * light_dist / cos_light;
                                }
                                
                                if (light_pdf_sa <= 0) goto skip_nee_plt;
                                
                                double bsdf_pdf_light = cos_surf / M_PI;
                                
                                double mis_w = 1.0;
                                if (settings_.use_mis) {
                                    mis_w = power_heuristic(light_pdf_sa, bsdf_pdf_light);
                                }
                                
                                // Direct light contribution (diffuse depolarizes, so just use I)
                                float I = std::max(0.0f, beam.stokes.I);
                                color direct = vec3_scale(
                                    vec3_mul(vec3_mul(ls.emission, surface_color), throughput),
                                    I * cos_surf / (M_PI * light_pdf_sa) * mis_w
                                );
                                accumulated = vec3_add(accumulated, direct);
                            }
                        }
                    }
                    
                    // === MNEE: Paths through specular surfaces ===
                    if (settings_.use_mnee) {
                        auto [mnee_contrib, mnee_used] = MNEE::evaluate(
                            scene, rec.point, rec.normal,
                            vec3_negate(vec3_normalize(current_ray.direction)), mat);
                        
                        if (mnee_used) {
                            float I = std::max(0.0f, beam.stokes.I);
                            color mnee_direct = vec3_mul(
                                vec3_mul(mnee_contrib, surface_color),
                                throughput
                            );
                            mnee_direct = vec3_scale(mnee_direct, I / M_PI);
                            accumulated = vec3_add(accumulated, mnee_direct);
                        }
                    }
                }
                skip_nee_plt:
                
                // Diffuse scattering depolarizes
                float r1 = static_cast<float>(random_double());
                float r2 = static_cast<float>(random_double());
                scatter_diffuse(beam, wo, albedo_avg, normal, r1, r2);
                throughput = vec3_mul(throughput, surface_color);
                bsdf_pdf = std::max(0.0, static_cast<double>(dot(wo, normal))) / M_PI;
                was_diffuse_bounce = true;
                break;
            }
            
            case MaterialType::Metal: {
                specular_bounce = true;
                float eta = static_cast<float>(mat.refraction_index);
                float k = 3.0f;
                float fuzz_val = static_cast<float>(mat.get_roughness(rec.u, rec.v));
                Vec3 rand_vec(random_in_unit_sphere());
                
                // Check for thin-film coating (note: requires spectral PLT for proper color)
                if (mat.has_thin_film) {
                    // For now, use middle wavelength (550nm green)
                    // Full spectral PLT would sample multiple wavelengths
                    float wavelength_nm = 550.0f;
                    if (!scatter_thin_film_metal(beam, wo, wi, normal, eta, k,
                                                  static_cast<float>(mat.thin_film_ior),
                                                  static_cast<float>(mat.thin_film_thickness),
                                                  wavelength_nm, albedo_avg, fuzz_val, rand_vec)) {
                        break;
                    }
                } else {
                    if (!scatter_metal(beam, wo, wi, normal, eta, k, albedo_avg, fuzz_val, rand_vec)) {
                        break;
                    }
                }
                throughput = vec3_mul(throughput, surface_color);
                if (fuzz_val >= 0.1f) specular_bounce = false;
                break;
            }
            
            case MaterialType::Dielectric: {
                specular_bounce = true;
                float eta = static_cast<float>(mat.refraction_index);
                bool entering = rec.front_face;
                float rand_val = static_cast<float>(random_double());
                
                // Check for thin-film coating
                if (mat.has_thin_film) {
                    float wavelength_nm = 550.0f;
                    if (!scatter_thin_film_dielectric(beam, wo, wi, normal, eta,
                                                       static_cast<float>(mat.thin_film_ior),
                                                       static_cast<float>(mat.thin_film_thickness),
                                                       wavelength_nm, rand_val)) {
                        break;
                    }
                } else {
                    if (!scatter_dielectric(beam, wo, wi, normal, eta, entering, rand_val)) {
                        break;
                    }
                }
                break;
            }
        }
        
        // Russian roulette after a few bounces
        if (bounce > 3) {
            double rr_prob = std::max(0.1, std::min(0.95, 
                (throughput.x + throughput.y + throughput.z) / 3.0 * beam.stokes.I));
            if (random_double() > rr_prob) {
                break;
            }
            throughput = vec3_scale(throughput, 1.0 / rr_prob);
        }
        
        // Update ray
        current_ray = ray_create(rec.point, wo.to_vec3());
    }
    
    return accumulated;
}

// ==================== Spectral PLT (for thin-film iridescence) ====================

// Wavelength to RGB conversion (CIE 1931 approximation)
static color wavelength_to_rgb(double wavelength_nm) {
    double r, g, b;
    
    if (wavelength_nm >= 380 && wavelength_nm < 440) {
        r = -(wavelength_nm - 440) / (440 - 380);
        g = 0.0;
        b = 1.0;
    } else if (wavelength_nm >= 440 && wavelength_nm < 490) {
        r = 0.0;
        g = (wavelength_nm - 440) / (490 - 440);
        b = 1.0;
    } else if (wavelength_nm >= 490 && wavelength_nm < 510) {
        r = 0.0;
        g = 1.0;
        b = -(wavelength_nm - 510) / (510 - 490);
    } else if (wavelength_nm >= 510 && wavelength_nm < 580) {
        r = (wavelength_nm - 510) / (580 - 510);
        g = 1.0;
        b = 0.0;
    } else if (wavelength_nm >= 580 && wavelength_nm < 645) {
        r = 1.0;
        g = -(wavelength_nm - 645) / (645 - 580);
        b = 0.0;
    } else if (wavelength_nm >= 645 && wavelength_nm <= 780) {
        r = 1.0;
        g = 0.0;
        b = 0.0;
    } else {
        r = g = b = 0.0;
    }
    
    // Intensity falloff at spectrum edges
    double factor;
    if (wavelength_nm >= 380 && wavelength_nm < 420) {
        factor = 0.3 + 0.7 * (wavelength_nm - 380) / (420 - 380);
    } else if (wavelength_nm >= 420 && wavelength_nm <= 700) {
        factor = 1.0;
    } else if (wavelength_nm > 700 && wavelength_nm <= 780) {
        factor = 0.3 + 0.7 * (780 - wavelength_nm) / (780 - 700);
    } else {
        factor = 0.0;
    }
    
    return {r * factor, g * factor, b * factor};
}

double Renderer::ray_radiance_plt_spectral(ray r, const Scene& scene, int depth, double wavelength_nm) const {
    using namespace plt;
    
    Vec3 direction(r.direction);
    Beam beam = Beam::unpolarized(1.0f, direction);
    
    double throughput = 1.0;
    double accumulated = 0.0;  // Accumulated radiance from NEE
    ray current_ray = r;
    bool specular_bounce = false;
    
    for (int bounce = 0; bounce < depth; ++bounce) {
        hit_record rec;
        
        if (!scene.hit(current_ray, 0.001, 1e9, rec)) {
            // Background - use scene background (luminance for spectral)
            color bg = background_color(current_ray, scene);
            double bg_luminance = (bg.x + bg.y + bg.z) / 3.0;
            return accumulated + throughput * beam.stokes.I * bg_luminance;
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        apply_normal_map(rec, mat);
        
        // Check if we hit an emissive surface
        if (mat.is_emissive() || scene.is_area_light(rec.material_id)) {
            // Only count emission on first bounce or after specular (NEE handles direct)
            if (bounce == 0 || specular_bounce || !settings_.use_nee) {
                double Le = (mat.emission.x + mat.emission.y + mat.emission.z) / 3.0;
                return accumulated + throughput * beam.stokes.I * Le;
            }
            // NEE already counted this, terminate
            return accumulated;
        }
        
        Vec3 normal(rec.normal);
        Vec3 wi = Vec3(current_ray.direction).normalized();
        Vec3 wo;
        
        color surface_color = mat.get_albedo(rec.point, rec.u, rec.v);
        float albedo_avg = static_cast<float>((surface_color.x + surface_color.y + surface_color.z) / 3.0);
        
        specular_bounce = false;
        
        switch (mat.type) {
            case MaterialType::Lambertian: {
                // === NEE for spectral PLT ===
                if (settings_.use_nee) {
                    LightSample ls = scene.sample_light(rec.point);
                    if (ls.valid && ls.distance < 1e5) {  // Skip environment for now
                        vec3 to_light = vec3_sub(ls.position, rec.point);
                        double light_dist = vec3_length(to_light);
                        vec3 light_dir = vec3_scale(to_light, 1.0 / light_dist);
                        
                        double cos_surf = vec3_dot(rec.normal, light_dir);
                        
                        if (cos_surf > 0) {
                            hit_record shadow_rec;
                            bool occluded = scene.hit(ray_create(rec.point, light_dir), 
                                                      0.001, light_dist - 0.001, shadow_rec);
                            
                            if (!occluded) {
                                double cos_light = std::abs(vec3_dot(ls.normal, vec3_negate(light_dir)));
                                if (cos_light > 0) {
                                    double light_pdf_sa = ls.pdf * light_dist * light_dist / cos_light;
                                    
                                    if (light_pdf_sa > 0) {
                                        // Lambertian BSDF = albedo/PI, PDF for cosine sampling = cos/PI
                                        double bsdf_pdf = cos_surf / M_PI;
                                        double mis_w = settings_.use_mis ? 
                                            power_heuristic(light_pdf_sa, bsdf_pdf) : 1.0;
                                        
                                        // Light emission (assume white light for spectral)
                                        double Le = (ls.emission.x + ls.emission.y + ls.emission.z) / 3.0;
                                        
                                        // Add NEE contribution
                                        accumulated += throughput * beam.stokes.I * 
                                            Le * albedo_avg * cos_surf / (M_PI * light_pdf_sa) * mis_w;
                                    }
                                }
                            }
                        }
                    }
                }
                
                float r1 = static_cast<float>(random_double());
                float r2 = static_cast<float>(random_double());
                scatter_diffuse(beam, wo, albedo_avg, normal, r1, r2);
                // Note: scatter_diffuse already applies albedo to beam.stokes.I
                break;
            }
            
            case MaterialType::Metal: {
                specular_bounce = true;
                float eta = static_cast<float>(mat.refraction_index);
                float k = 3.0f;
                float fuzz_val = static_cast<float>(mat.get_roughness(rec.u, rec.v));
                Vec3 rand_vec(random_in_unit_sphere());
                
                if (mat.has_thin_film) {
                    if (!scatter_thin_film_metal(beam, wo, wi, normal, eta, k,
                                                  static_cast<float>(mat.thin_film_ior),
                                                  static_cast<float>(mat.thin_film_thickness),
                                                  static_cast<float>(wavelength_nm),
                                                  albedo_avg, fuzz_val, rand_vec)) {
                        return 0.0;
                    }
                } else {
                    if (!scatter_metal(beam, wo, wi, normal, eta, k, albedo_avg, fuzz_val, rand_vec)) {
                        return 0.0;
                    }
                }
                // Note: scatter_metal already applies albedo to beam.stokes.I
                break;
            }
            
            case MaterialType::Dielectric: {
                specular_bounce = true;
                float eta = static_cast<float>(mat.refraction_index);
                bool entering = rec.front_face;
                float rand_val = static_cast<float>(random_double());
                
                if (mat.has_thin_film) {
                    if (!scatter_thin_film_dielectric(beam, wo, wi, normal, eta,
                                                       static_cast<float>(mat.thin_film_ior),
                                                       static_cast<float>(mat.thin_film_thickness),
                                                       static_cast<float>(wavelength_nm), rand_val)) {
                        return 0.0;
                    }
                } else {
                    if (!scatter_dielectric(beam, wo, wi, normal, eta, entering, rand_val)) {
                        return 0.0;
                    }
                }
                break;
            }
            
            default:
                // Unknown or emissive material - treat as diffuse
                {
                    float r1 = static_cast<float>(random_double());
                    float r2 = static_cast<float>(random_double());
                    scatter_diffuse(beam, wo, 0.5f, normal, r1, r2);
                    // Note: scatter_diffuse already applies albedo to beam.stokes.I
                }
                break;
        }
        
        // Russian roulette
        if (bounce > 3) {
            double rr_prob = std::max(0.1, std::min(0.95, throughput * beam.stokes.I));
            if (random_double() > rr_prob) {
                return accumulated;
            }
            throughput /= rr_prob;
        }
        
        current_ray = ray_create(rec.point, wo.to_vec3());
    }
    
    return accumulated;
}

} // namespace raytracer
