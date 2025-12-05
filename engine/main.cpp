/**
 * @file main.cpp
 * @brief Entry point for the ray tracer
 * 
 * This creates a simple demo scene and renders it to a PPM image.
 */

#include "renderer.hpp"
#include "scene.hpp"
#include "material.hpp"

#include <iostream>
#include <string>

using namespace raytracer;

/**
 * @brief Create a demo scene with several spheres
 */
Scene create_demo_scene() {
    Scene scene;
    
    // Add materials
    int mat_ground = scene.add_material(Material::lambertian({0.3, 0.3, 0.3}));  // Dark gray ground
    int mat_center = scene.add_material(Material::lambertian({0.1, 0.2, 0.5}));
    int mat_left = scene.add_material(Material::dielectric(1.5));
    int mat_right = scene.add_material(Material::metal({0.8, 0.6, 0.2}, 0.0));
    int mat_box = scene.add_material(Material::lambertian({0.7, 0.3, 0.3}));     // Red box
    
    // Ground plane (infinite, much cleaner than huge sphere)
    scene.add_plane({0.0, -0.5, 0.0}, {0.0, 1.0, 0.0}, mat_ground);
    
    // Center sphere (diffuse blue)
    scene.add_sphere({0.0, 0.0, -1.0}, 0.5, mat_center);
    
    // Left sphere (glass)
    scene.add_sphere({-1.0, 0.0, -1.0}, 0.5, mat_left);
    
    // Right sphere (metal gold)
    scene.add_sphere({1.0, 0.0, -1.0}, 0.5, mat_right);
    
    // Add a box behind the spheres
    scene.add_box_centered({0.0, 0.25, -2.5}, 1.0, 1.5, 1.0, mat_box);
    
    // Add lights
    scene.add_light({-2.0, 3.0, 1.0}, {0.6, 0.6, 0.6});  // Main light (upper left)
    scene.add_light({2.0, 2.0, 2.0}, {0.3, 0.3, 0.3});   // Fill light (upper right)
    
    return scene;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Whitted-Style Ray Tracer ===" << std::endl;
    
    // Parse command line for output filename
    std::string output_file = "output.png";
    if (argc > 1) {
        output_file = argv[1];
    }
    
    // Image settings
    const int image_width = 2560;
    const int image_height = 1440;
    const double aspect_ratio = static_cast<double>(image_width) / image_height;
    
    // Create camera
    point3 lookfrom = {3.0, 2.0, 4.0};   // Pulled back and up
    point3 lookat = {0.0, 0.0, -1.0};
    vec3 vup = {0.0, 1.0, 0.0};
    double vfov = 50.0;  // Narrower FOV for less distortion
    
    Camera camera(lookfrom, lookat, vup, vfov, aspect_ratio);
    
    // Create scene
    Scene scene = create_demo_scene();
    
    // Set up renderer
    Renderer::Settings settings;
    settings.width = image_width;
    settings.height = image_height;
    settings.max_depth = 100;
    settings.samples_per_pixel = 16;
    
    Renderer renderer(settings);
    
    // Render!
    Image image = renderer.render(scene, camera);
    
    // Save output (use PNG if filename ends with .png, otherwise PPM)
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
