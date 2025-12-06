/**
 * @file cuda_pathtracer.cu
 * @brief GPU path tracing kernel implementation
 */

#include "cuda_pathtracer.cuh"
#include <stdio.h>

namespace cuda {

// ============================================================================
// Material scattering (device functions)
// ============================================================================

__device__ bool scatter_lambertian(const Material& mat, const Ray& r_in, const HitRecord& rec,
                                   Color& attenuation, Ray& scattered, RNG& rng) {
    // Cosine-weighted hemisphere sampling
    ONB uvw;
    uvw.build_from_w(rec.normal);
    Vec3 scatter_dir = uvw.local(rng.cosine_direction());
    
    if (scatter_dir.near_zero())
        scatter_dir = rec.normal;
    
    scattered = Ray(rec.point, normalize(scatter_dir));
    attenuation = mat.albedo;
    return true;
}

__device__ bool scatter_metal(const Material& mat, const Ray& r_in, const HitRecord& rec,
                              Color& attenuation, Ray& scattered, RNG& rng) {
    Vec3 reflected = reflect(normalize(r_in.direction), rec.normal);
    scattered = Ray(rec.point, normalize(reflected + mat.fuzz * rng.in_unit_sphere()));
    attenuation = mat.albedo;
    return dot(scattered.direction, rec.normal) > 0;
}

__device__ bool scatter_dielectric(const Material& mat, const Ray& r_in, const HitRecord& rec,
                                   Color& attenuation, Ray& scattered, RNG& rng) {
    attenuation = Color(1.0f, 1.0f, 1.0f);
    float refraction_ratio = rec.front_face ? (1.0f / mat.ior) : mat.ior;
    
    Vec3 unit_direction = normalize(r_in.direction);
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    
    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    Vec3 direction;
    
    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rng.uniform())
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    
    scattered = Ray(rec.point, direction);
    return true;
}

__device__ bool scatter(const Material& mat, const Ray& r_in, const HitRecord& rec,
                        Color& attenuation, Ray& scattered, RNG& rng) {
    switch (mat.type) {
        case MaterialType::Lambertian:
            return scatter_lambertian(mat, r_in, rec, attenuation, scattered, rng);
        case MaterialType::Metal:
            return scatter_metal(mat, r_in, rec, attenuation, scattered, rng);
        case MaterialType::Dielectric:
            return scatter_dielectric(mat, r_in, rec, attenuation, scattered, rng);
        case MaterialType::Emissive:
            return false;  // Emissive materials don't scatter
        default:
            return false;
    }
}

// ============================================================================
// Path tracing kernel
// ============================================================================

__device__ Color trace_path(Ray r, const GPUScene& scene, RNG& rng, int max_depth, bool use_nee) {
    Color throughput(1.0f, 1.0f, 1.0f);
    Color accumulated(0.0f, 0.0f, 0.0f);
    bool specular_bounce = false;
    
    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;
        
        if (!scene.hit(r, 0.001f, 1e20f, rec)) {
            // No hit - add background
            accumulated += throughput * scene.get_background(r.direction);
            break;
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        
        // Add emission only on first hit or after specular bounce (avoid double counting with NEE)
        if (mat.type == MaterialType::Emissive) {
            if (depth == 0 || !use_nee || specular_bounce) {
                accumulated += throughput * mat.emission * mat.emission_strength;
            }
            break;
        }
        
        // Track if this was a specular bounce for next iteration
        specular_bounce = (mat.type == MaterialType::Metal || mat.type == MaterialType::Dielectric);
        
        // NEE: Direct light sampling for diffuse surfaces
        if (use_nee && mat.type == MaterialType::Lambertian && scene.num_emissive > 0) {
            Point3 light_pos;
            Color light_emission;
            float light_pdf;
            
            if (scene.sample_light(rng, light_pos, light_emission, light_pdf)) {
                Vec3 to_light = light_pos - rec.point;
                float dist_to_light = to_light.length();
                Vec3 light_dir = to_light / dist_to_light;
                
                // Shadow ray
                HitRecord shadow_rec;
                Ray shadow_ray(rec.point, light_dir);
                
                if (!scene.hit(shadow_ray, 0.001f, dist_to_light - 0.001f, shadow_rec)) {
                    // Not occluded - add direct light contribution
                    float cos_theta = fmaxf(0.0f, dot(rec.normal, light_dir));
                    // Proper NEE: BRDF * Li * cos / pdf, with inv_pi for Lambertian BRDF
                    float inv_pi = 0.31830988618f;
                    Color direct = throughput * mat.albedo * inv_pi * light_emission * cos_theta / (dist_to_light * dist_to_light * light_pdf);
                    accumulated += direct;
                }
            }
        }
        
        // Scatter
        Color attenuation;
        Ray scattered;
        
        if (!scatter(mat, r, rec, attenuation, scattered, rng)) {
            break;
        }
        
        throughput = throughput * attenuation;
        r = scattered;
        
        // Russian roulette termination
        if (depth > 3) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rng.uniform() > p)
                break;
            throughput = throughput / p;
        }
    }
    
    return accumulated;
}

// ============================================================================
// Main render kernel
// ============================================================================

__global__ void render_kernel(
    float* output,
    int width, int height,
    int samples_per_pixel,
    int max_depth,
    bool use_nee,
    GPUScene scene,
    GPUCamera camera,
    unsigned long long seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = y * width + x;
    
    // Initialize RNG for this pixel
    RNG rng;
    rng.init(seed, pixel_idx);
    
    Color pixel_color(0.0f, 0.0f, 0.0f);
    
    for (int s = 0; s < samples_per_pixel; s++) {
        float u = (x + rng.uniform()) / (float)width;
        float v = (y + rng.uniform()) / (float)height;
        
        Ray r = camera.get_ray(u, v, rng);
        pixel_color += trace_path(r, scene, rng, max_depth, use_nee);
    }
    
    // Average and store (RGB, no gamma - done on CPU)
    pixel_color = pixel_color / (float)samples_per_pixel;
    
    int out_idx = pixel_idx * 3;
    output[out_idx + 0] = pixel_color.x;
    output[out_idx + 1] = pixel_color.y;
    output[out_idx + 2] = pixel_color.z;
}

// ============================================================================
// RNG initialization kernel
// ============================================================================

__global__ void init_rng_kernel(curandState* states, int width, int height, unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    curand_init(seed, idx, 0, &states[idx]);
}

// ============================================================================
// Host-side launcher
// ============================================================================

PathTracerGPU::PathTracerGPU() 
    : d_spheres(nullptr), d_planes(nullptr), d_materials(nullptr),
      d_emissive_indices(nullptr), d_output(nullptr),
      allocated_pixels(0) {}

PathTracerGPU::~PathTracerGPU() {
    free_scene();
    if (d_output) cudaFree(d_output);
}

void PathTracerGPU::free_scene() {
    if (d_spheres) { cudaFree(d_spheres); d_spheres = nullptr; }
    if (d_planes) { cudaFree(d_planes); d_planes = nullptr; }
    if (d_materials) { cudaFree(d_materials); d_materials = nullptr; }
    if (d_emissive_indices) { cudaFree(d_emissive_indices); d_emissive_indices = nullptr; }
}

void PathTracerGPU::upload_scene(
    const std::vector<GPUSphere>& spheres,
    const std::vector<GPUPlane>& planes,
    const std::vector<GPUMaterial>& materials,
    const std::vector<int>& emissive_indices,
    float bg_r, float bg_g, float bg_b,
    bool use_gradient,
    float bg_top_r, float bg_top_g, float bg_top_b,
    float bg_bot_r, float bg_bot_g, float bg_bot_b
) {
    free_scene();
    
    // Allocate and copy spheres
    if (!spheres.empty()) {
        cudaMalloc(&d_spheres, spheres.size() * sizeof(Sphere));
        
        // Convert from host struct to device struct
        std::vector<Sphere> device_spheres(spheres.size());
        for (size_t i = 0; i < spheres.size(); i++) {
            device_spheres[i].center = Point3(spheres[i].cx, spheres[i].cy, spheres[i].cz);
            device_spheres[i].radius = spheres[i].radius;
            device_spheres[i].material_id = spheres[i].material_id;
        }
        cudaMemcpy(d_spheres, device_spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
    }
    h_num_spheres = (int)spheres.size();
    
    // Allocate and copy planes
    if (!planes.empty()) {
        cudaMalloc(&d_planes, planes.size() * sizeof(Plane));
        
        std::vector<Plane> device_planes(planes.size());
        for (size_t i = 0; i < planes.size(); i++) {
            device_planes[i].point = Point3(planes[i].px, planes[i].py, planes[i].pz);
            device_planes[i].normal = Vec3(planes[i].nx, planes[i].ny, planes[i].nz);
            device_planes[i].material_id = planes[i].material_id;
        }
        cudaMemcpy(d_planes, device_planes.data(), planes.size() * sizeof(Plane), cudaMemcpyHostToDevice);
    }
    h_num_planes = (int)planes.size();
    
    // Allocate and copy materials
    if (!materials.empty()) {
        cudaMalloc(&d_materials, materials.size() * sizeof(Material));
        
        std::vector<Material> device_materials(materials.size());
        for (size_t i = 0; i < materials.size(); i++) {
            device_materials[i].type = static_cast<MaterialType>(materials[i].type);
            device_materials[i].albedo = Color(materials[i].albedo_r, materials[i].albedo_g, materials[i].albedo_b);
            device_materials[i].fuzz = materials[i].fuzz;
            device_materials[i].ior = materials[i].ior;
            device_materials[i].emission = Color(materials[i].emission_r, materials[i].emission_g, materials[i].emission_b);
            device_materials[i].emission_strength = materials[i].emission_strength;
        }
        cudaMemcpy(d_materials, device_materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    }
    h_num_materials = (int)materials.size();
    
    // Emissive indices
    if (!emissive_indices.empty()) {
        cudaMalloc(&d_emissive_indices, emissive_indices.size() * sizeof(int));
        cudaMemcpy(d_emissive_indices, emissive_indices.data(), emissive_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    h_num_emissive = (int)emissive_indices.size();
    
    // Background
    h_background = Color(bg_r, bg_g, bg_b);
    h_use_gradient = use_gradient;
    h_background_top = Color(bg_top_r, bg_top_g, bg_top_b);
    h_background_bottom = Color(bg_bot_r, bg_bot_g, bg_bot_b);
}

void PathTracerGPU::set_camera(
    float pos_x, float pos_y, float pos_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float fov, float aspect_ratio, float aperture, float focus_dist
) {
    h_camera.setup(
        Point3(pos_x, pos_y, pos_z),
        Point3(target_x, target_y, target_z),
        Vec3(up_x, up_y, up_z),
        fov, aspect_ratio, aperture, focus_dist
    );
}

void PathTracerGPU::render(
    float* h_output,
    int width, int height,
    int samples_per_pixel,
    int max_depth,
    bool use_nee
) {
    // Allocate output buffer if needed
    int num_pixels = width * height;
    if (num_pixels > allocated_pixels) {
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_output, num_pixels * 3 * sizeof(float));
        allocated_pixels = num_pixels;
    }
    
    // Build scene struct for kernel
    GPUScene scene;
    scene.spheres = d_spheres;
    scene.num_spheres = h_num_spheres;
    scene.planes = d_planes;
    scene.num_planes = h_num_planes;
    scene.materials = d_materials;
    scene.num_materials = h_num_materials;
    scene.emissive_spheres = d_emissive_indices;
    scene.num_emissive = h_num_emissive;
    scene.background = h_background;
    scene.use_gradient = h_use_gradient;
    scene.background_top = h_background_top;
    scene.background_bottom = h_background_bottom;
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    unsigned long long seed = 1234ULL;  // Could randomize
    
    render_kernel<<<grid, block>>>(
        d_output,
        width, height,
        samples_per_pixel,
        max_depth,
        use_nee,
        scene,
        h_camera,
        seed
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, num_pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace cuda
