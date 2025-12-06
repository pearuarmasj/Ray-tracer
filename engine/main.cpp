/**
 * @file main.cpp
 * @brief Entry point for the ray tracer
 * 
 * Renders scenes from JSON files or uses a built-in demo scene.
 */

#include "renderer.hpp"
#include "scene.hpp"
#include "scene_loader.hpp"
#include "material.hpp"
#include "texture.hpp"

#include <iostream>
#include <string>

using namespace raytracer;

/**
 * @brief Create a fallback demo scene
 */
Scene create_demo_scene() {
    Scene scene;
    
    // Checker pattern ground
    int mat_ground = scene.add_material(Material::lambertian_textured(
        Texture::checker({0.9, 0.9, 0.9}, {0.2, 0.2, 0.2}, 1.0)
    ));
    int mat_center = scene.add_material(Material::lambertian({0.1, 0.2, 0.5}));
    int mat_left = scene.add_material(Material::dielectric(1.5));
    int mat_right = scene.add_material(Material::metal({0.8, 0.6, 0.2}, 0.0));
    int mat_box = scene.add_material(Material::lambertian({0.7, 0.3, 0.3}));
    
    scene.add_plane({0.0, -0.5, 0.0}, {0.0, 1.0, 0.0}, mat_ground);
    scene.add_sphere({0.0, 0.0, -1.0}, 0.5, mat_center);
    scene.add_sphere({-1.0, 0.0, -1.0}, 0.5, mat_left);
    scene.add_sphere({1.0, 0.0, -1.0}, 0.5, mat_right);
    scene.add_box_centered({0.0, 0.25, -2.5}, 1.0, 1.5, 1.0, mat_box);
    scene.add_light({-2.0, 3.0, 1.0}, {0.6, 0.6, 0.6});
    scene.add_light({2.0, 2.0, 2.0}, {0.3, 0.3, 0.3});
    
    return scene;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [scene.json] [output.png]" << std::endl;
    std::cout << "  scene.json  - Scene file to render (optional, uses demo scene if not provided)" << std::endl;
    std::cout << "  output.png  - Output file (optional, default from scene or 'output.png')" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Ray Tracer ===" << std::endl;
    
    Scene scene;
    std::string output_file = "output.png";
    int image_width = 2560;
    int image_height = 1440;
    int samples = 16;
    int max_depth = 50;
    RenderMode render_mode = RenderMode::Whitted;
    bool use_nee = true;
    bool use_mis = true;
    ToneMapper tone_mapper = ToneMapper::ACES;
    double exposure = 1.0;
    double clamp_max = 10.0;  // Firefly clamping for BDPT
    int wavelength_samples = 8;  // For spectral rendering
    
    // Background settings
    color background_color = {0.0, 0.0, 0.0};
    color background_top = {0.5, 0.7, 1.0};
    color background_bottom = {1.0, 1.0, 1.0};
    bool use_background_gradient = true;  // Default to gradient for backward compat
    
    // Photon mapping settings
    size_t photon_count = 100000;
    size_t caustic_photon_count = 50000;
    int photon_gather_count = 100;
    float photon_gather_radius = 0.5f;
    float caustic_gather_radius = 0.1f;
    bool photon_final_gather = false;
    
    point3 lookfrom = {3.0, 2.0, 4.0};
    point3 lookat = {0.0, 0.0, -1.0};
    vec3 vup = {0.0, 1.0, 0.0};
    double vfov = 50.0;
    
    // Check for scene file argument
    bool loaded_scene = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        
        if (arg == "--pathtrace" || arg == "-p") {
            render_mode = RenderMode::PathTrace;
            samples = 64;  // Path tracing needs more samples
            continue;
        }
        
        if (arg == "--bdpt" || arg == "-b") {
            render_mode = RenderMode::BDPT;
            samples = 64;  // BDPT also needs more samples
            continue;
        }
        
        if (arg == "--spectral" || arg == "-s") {
            render_mode = RenderMode::Spectral;
            samples = 64;  // Spectral needs more samples
            continue;
        }
        
        // --nee on/off flag
        if (arg == "--nee" && i + 1 < argc) {
            std::string val = argv[++i];
            use_nee = (val == "on" || val == "true" || val == "1");
            continue;
        }
        
        // --mis on/off flag
        if (arg == "--mis" && i + 1 < argc) {
            std::string val = argv[++i];
            use_mis = (val == "on" || val == "true" || val == "1");
            continue;
        }
        
        // --samples N flag
        if (arg == "--samples" && i + 1 < argc) {
            samples = std::stoi(argv[++i]);
            continue;
        }
        
        // --output filename flag
        if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
            continue;
        }
        
        // --tonemapper name flag (none, reinhard, aces, uncharted2)
        if (arg == "--tonemapper" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "none") tone_mapper = ToneMapper::None;
            else if (val == "reinhard") tone_mapper = ToneMapper::Reinhard;
            else if (val == "aces") tone_mapper = ToneMapper::ACES;
            else if (val == "uncharted2") tone_mapper = ToneMapper::Uncharted2;
            continue;
        }
        
        // --exposure N flag
        if (arg == "--exposure" && i + 1 < argc) {
            exposure = std::stod(argv[++i]);
            continue;
        }
        
        // --clamp N flag (firefly clamping, 0 to disable)
        if (arg == "--clamp" && i + 1 < argc) {
            clamp_max = std::stod(argv[++i]);
            continue;
        }
        
        // --wavelengths N flag (wavelength samples for spectral rendering)
        if (arg == "--wavelengths" && i + 1 < argc) {
            wavelength_samples = std::stoi(argv[++i]);
            continue;
        }
        
        // Check if it's a JSON file
        if (arg.size() > 5 && arg.substr(arg.size() - 5) == ".json") {
            auto data = SceneLoader::load(arg);
            if (!data.scene.materials.empty() || !data.scene.spheres.empty() || 
                !data.scene.planes.empty() || !data.scene.boxes.empty()) {
                scene = std::move(data.scene);
                lookfrom = data.camera_position;
                lookat = data.camera_target;
                vup = data.camera_up;
                vfov = data.camera_fov;
                image_width = data.width;
                image_height = data.height;
                samples = data.samples;
                max_depth = data.max_depth;
                output_file = data.output_file;
                render_mode = data.mode;
                use_nee = data.use_nee;
                use_mis = data.use_mis;
                tone_mapper = data.tone_mapper;
                exposure = data.exposure;
                clamp_max = data.clamp_max;
                wavelength_samples = data.wavelength_samples;
                
                // Background settings
                background_color = data.background_color;
                background_top = data.background_top;
                background_bottom = data.background_bottom;
                use_background_gradient = data.use_background_gradient;
                
                // Photon mapping settings
                photon_count = data.photon_count;
                caustic_photon_count = data.caustic_photon_count;
                photon_gather_count = data.photon_gather_count;
                photon_gather_radius = data.photon_gather_radius;
                caustic_gather_radius = data.caustic_gather_radius;
                photon_final_gather = data.photon_final_gather;
                
                loaded_scene = true;
            }
        } else if (arg[0] != '-') {
            // Assume it's an output filename
            output_file = arg;
        }
    }
    
    // Use demo scene if no scene was loaded
    if (!loaded_scene) {
        std::cout << "Using built-in demo scene" << std::endl;
        scene = create_demo_scene();
    }
    
    double aspect_ratio = static_cast<double>(image_width) / image_height;
    Camera camera(lookfrom, lookat, vup, vfov, aspect_ratio);
    
    // Set up renderer
    Renderer::Settings settings;
    settings.width = image_width;
    settings.height = image_height;
    settings.max_depth = max_depth;
    settings.samples_per_pixel = samples;
    settings.mode = render_mode;
    settings.use_nee = use_nee;
    settings.use_mis = use_mis;
    settings.tone_mapper = tone_mapper;
    settings.exposure = exposure;
    settings.clamp_max = clamp_max;
    settings.wavelength_samples = wavelength_samples;
    
    // Background settings
    settings.background_color = background_color;
    settings.background_top = background_top;
    settings.background_bottom = background_bottom;
    settings.use_background_gradient = use_background_gradient;
    
    // Photon mapping settings
    settings.photon_count = photon_count;
    settings.caustic_photon_count = caustic_photon_count;
    settings.photon_gather_count = photon_gather_count;
    settings.photon_gather_radius = photon_gather_radius;
    settings.caustic_gather_radius = caustic_gather_radius;
    settings.photon_final_gather = photon_final_gather;
    
    Renderer renderer(settings);
    
    // Render
    Image image = renderer.render(scene, camera);
    
    // Save output
    bool success = false;
    if (output_file.size() >= 4 && output_file.substr(output_file.size() - 4) == ".png") {
        success = image.write_png(output_file);
    } else {
        success = image.write_ppm(output_file);
    }
    
    if (success) {
        std::cout << "Image saved to: " << output_file << std::endl;
    } else {
        std::cerr << "Error: Failed to save image to " << output_file << std::endl;
        return 1;
    }
    
    return 0;
}
