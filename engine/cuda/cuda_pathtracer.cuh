/**
 * @file cuda_pathtracer.cuh
 * @brief GPU path tracer interface
 */

#pragma once

#include "cuda_math.cuh"
#include "cuda_scene.cuh"
#include <vector>

namespace cuda {

// ============================================================================
// Host-side data structures for uploading scene
// ============================================================================

struct GPUSphere {
    float cx, cy, cz;
    float radius;
    int material_id;
};

struct GPUPlane {
    float px, py, pz;
    float nx, ny, nz;
    int material_id;
};

struct GPUMaterial {
    int type;  // 0=Lambertian, 1=Metal, 2=Dielectric, 3=Emissive
    float albedo_r, albedo_g, albedo_b;
    float fuzz;
    float ior;
    float emission_r, emission_g, emission_b;
    float emission_strength;
};

// ============================================================================
// Path tracer class (manages GPU resources)
// ============================================================================

class PathTracerGPU {
public:
    PathTracerGPU();
    ~PathTracerGPU();
    
    // Upload scene data to GPU
    void upload_scene(
        const std::vector<GPUSphere>& spheres,
        const std::vector<GPUPlane>& planes,
        const std::vector<GPUMaterial>& materials,
        const std::vector<int>& emissive_indices,
        float bg_r, float bg_g, float bg_b,
        bool use_gradient = false,
        float bg_top_r = 0.5f, float bg_top_g = 0.7f, float bg_top_b = 1.0f,
        float bg_bot_r = 1.0f, float bg_bot_g = 1.0f, float bg_bot_b = 1.0f
    );
    
    // Set camera parameters
    void set_camera(
        float pos_x, float pos_y, float pos_z,
        float target_x, float target_y, float target_z,
        float up_x, float up_y, float up_z,
        float fov, float aspect_ratio,
        float aperture = 0.0f, float focus_dist = 1.0f
    );
    
    // Render to host buffer (RGB floats, width*height*3)
    void render(
        float* h_output,
        int width, int height,
        int samples_per_pixel,
        int max_depth,
        bool use_nee = true
    );
    
private:
    void free_scene();
    
    // Device memory
    Sphere* d_spheres;
    Plane* d_planes;
    Material* d_materials;
    int* d_emissive_indices;
    float* d_output;
    
    // Host-side counts
    int h_num_spheres;
    int h_num_planes;
    int h_num_materials;
    int h_num_emissive;
    
    // Background
    Color h_background;
    bool h_use_gradient;
    Color h_background_top;
    Color h_background_bottom;
    
    // Camera
    GPUCamera h_camera;
    
    // Allocation tracking
    int allocated_pixels;
};

} // namespace cuda
