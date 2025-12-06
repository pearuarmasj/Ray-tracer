/**
 * @file cuda_test.cu
 * @brief Minimal CUDA test to verify GPU execution works
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

namespace raytracer {
namespace cuda {

// Simple test kernel - fills array with thread indices
__global__ void test_kernel(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<float>(idx) * 0.001f;
    }
}

/**
 * @brief Test CUDA availability and basic execution
 * @return true if CUDA is working
 */
bool test_cuda() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA Error: No CUDA devices found\n";
        return false;
    }
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "CUDA Device: " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "  Global memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "  Max threads/block: " << prop.maxThreadsPerBlock << "\n";
    
    // Run a simple test kernel
    const int n = 1024;
    float* d_output = nullptr;
    float h_output[n];
    
    err = cudaMalloc(&d_output, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    test_kernel<<<blocks, threads_per_block>>>(d_output, n);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_output);
        return false;
    }
    
    // Copy back and verify
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    
    // Quick sanity check
    bool ok = (h_output[0] == 0.0f && h_output[100] == 0.1f && h_output[500] == 0.5f);
    
    if (ok) {
        std::cout << "CUDA test kernel: PASSED\n";
    } else {
        std::cerr << "CUDA test kernel: FAILED (output mismatch)\n";
    }
    
    return ok;
}

} // namespace cuda
} // namespace raytracer

// ============================================================================
// GPUPathTracer implementation (wrapper around cuda::PathTracerGPU)
// ============================================================================

#include "cuda_api.hpp"
#include "cuda_pathtracer.cuh"

namespace raytracer {
namespace cuda {

GPUPathTracer::GPUPathTracer() {
    impl_ = new ::cuda::PathTracerGPU();
}

GPUPathTracer::~GPUPathTracer() {
    delete static_cast<::cuda::PathTracerGPU*>(impl_);
}

void GPUPathTracer::upload_scene(
    const std::vector<CudaSphere>& spheres,
    const std::vector<CudaPlane>& planes,
    const std::vector<CudaMaterial>& materials,
    const std::vector<int>& emissive_sphere_indices
) {
    auto* pt = static_cast<::cuda::PathTracerGPU*>(impl_);
    
    // Convert to internal format
    std::vector<::cuda::GPUSphere> gpu_spheres(spheres.size());
    for (size_t i = 0; i < spheres.size(); i++) {
        gpu_spheres[i].cx = spheres[i].cx;
        gpu_spheres[i].cy = spheres[i].cy;
        gpu_spheres[i].cz = spheres[i].cz;
        gpu_spheres[i].radius = spheres[i].radius;
        gpu_spheres[i].material_id = spheres[i].material_id;
    }
    
    std::vector<::cuda::GPUPlane> gpu_planes(planes.size());
    for (size_t i = 0; i < planes.size(); i++) {
        gpu_planes[i].px = planes[i].px;
        gpu_planes[i].py = planes[i].py;
        gpu_planes[i].pz = planes[i].pz;
        gpu_planes[i].nx = planes[i].nx;
        gpu_planes[i].ny = planes[i].ny;
        gpu_planes[i].nz = planes[i].nz;
        gpu_planes[i].material_id = planes[i].material_id;
    }
    
    std::vector<::cuda::GPUMaterial> gpu_materials(materials.size());
    for (size_t i = 0; i < materials.size(); i++) {
        gpu_materials[i].type = materials[i].type;
        gpu_materials[i].albedo_r = materials[i].albedo_r;
        gpu_materials[i].albedo_g = materials[i].albedo_g;
        gpu_materials[i].albedo_b = materials[i].albedo_b;
        gpu_materials[i].fuzz = materials[i].fuzz;
        gpu_materials[i].ior = materials[i].ior;
        gpu_materials[i].emission_r = materials[i].emission_r;
        gpu_materials[i].emission_g = materials[i].emission_g;
        gpu_materials[i].emission_b = materials[i].emission_b;
        gpu_materials[i].emission_strength = materials[i].emission_strength;
    }
    
    pt->upload_scene(gpu_spheres, gpu_planes, gpu_materials, emissive_sphere_indices,
                     settings_.bg_r, settings_.bg_g, settings_.bg_b,
                     settings_.use_gradient,
                     settings_.bg_top_r, settings_.bg_top_g, settings_.bg_top_b,
                     settings_.bg_bot_r, settings_.bg_bot_g, settings_.bg_bot_b);
}

void GPUPathTracer::set_settings(const CudaRenderSettings& settings) {
    settings_ = settings;
}

void GPUPathTracer::set_camera(const CudaCamera& camera) {
    auto* pt = static_cast<::cuda::PathTracerGPU*>(impl_);
    float aspect = (float)settings_.width / (float)settings_.height;
    pt->set_camera(
        camera.pos_x, camera.pos_y, camera.pos_z,
        camera.target_x, camera.target_y, camera.target_z,
        camera.up_x, camera.up_y, camera.up_z,
        camera.fov, aspect, camera.aperture, camera.focus_dist
    );
}

void GPUPathTracer::render(float* output) {
    auto* pt = static_cast<::cuda::PathTracerGPU*>(impl_);
    pt->render(output, settings_.width, settings_.height, 
               settings_.samples_per_pixel, settings_.max_depth, settings_.use_nee);
}

} // namespace cuda
} // namespace raytracer
