/**
 * @file cuda_api.hpp
 * @brief C++ API for CUDA functionality (callable from non-CUDA code)
 */

#pragma once

#include <vector>

namespace raytracer {
namespace cuda {

/**
 * @brief Test CUDA availability and basic execution
 * @return true if CUDA is working
 */
bool test_cuda();

// ============================================================================
// GPU Path Tracer API (POD structs for C++ interop)
// ============================================================================

struct CudaSphere {
    float cx, cy, cz;
    float radius;
    int material_id;
};

struct CudaPlane {
    float px, py, pz;
    float nx, ny, nz;
    int material_id;
};

struct CudaMaterial {
    int type;  // 0=Lambertian, 1=Metal, 2=Dielectric, 3=Emissive
    float albedo_r, albedo_g, albedo_b;
    float fuzz;
    float ior;
    float emission_r, emission_g, emission_b;
    float emission_strength;
};

struct CudaCamera {
    float pos_x, pos_y, pos_z;
    float target_x, target_y, target_z;
    float up_x, up_y, up_z;
    float fov;
    float aperture;
    float focus_dist;
};

struct CudaRenderSettings {
    int width;
    int height;
    int samples_per_pixel;
    int max_depth;
    bool use_nee;
    
    // Background
    float bg_r, bg_g, bg_b;
    bool use_gradient;
    float bg_top_r, bg_top_g, bg_top_b;
    float bg_bot_r, bg_bot_g, bg_bot_b;
};

/**
 * @brief GPU Path Tracer
 * 
 * Manages GPU resources and provides simple render interface.
 */
class GPUPathTracer {
public:
    GPUPathTracer();
    ~GPUPathTracer();
    
    // Non-copyable
    GPUPathTracer(const GPUPathTracer&) = delete;
    GPUPathTracer& operator=(const GPUPathTracer&) = delete;
    
    /**
     * @brief Upload scene geometry and materials to GPU
     */
    void upload_scene(
        const std::vector<CudaSphere>& spheres,
        const std::vector<CudaPlane>& planes,
        const std::vector<CudaMaterial>& materials,
        const std::vector<int>& emissive_sphere_indices
    );
    
    /**
     * @brief Set render settings (background, etc.)
     */
    void set_settings(const CudaRenderSettings& settings);
    
    /**
     * @brief Set camera
     */
    void set_camera(const CudaCamera& camera);
    
    /**
     * @brief Render scene to output buffer
     * @param output RGB float buffer (width * height * 3), allocated by caller
     */
    void render(float* output);
    
private:
    void* impl_;  // Opaque pointer to implementation
    CudaRenderSettings settings_;
};

} // namespace cuda
} // namespace raytracer
