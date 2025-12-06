/**
 * @file cuda_scene.cuh
 * @brief GPU scene representation (materials, primitives, hit records)
 * 
 * Flat arrays for coalesced memory access on GPU.
 */

#pragma once

#include "cuda_math.cuh"

namespace cuda {

// ============================================================================
// Material types
// ============================================================================

enum class MaterialType : int {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    Emissive = 3
};

struct Material {
    MaterialType type;
    Color albedo;
    float fuzz;           // For metal
    float ior;            // For dielectric (index of refraction)
    Color emission;       // For emissive materials
    float emission_strength;
    
    __host__ __device__ Material() 
        : type(MaterialType::Lambertian), albedo(0.5f), fuzz(0), ior(1.5f), 
          emission(0), emission_strength(0) {}
};

// ============================================================================
// Hit record
// ============================================================================

struct HitRecord {
    Point3 point;
    Vec3 normal;
    float t;
    int material_id;
    bool front_face;
    
    __device__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// ============================================================================
// Primitives
// ============================================================================

struct Sphere {
    Point3 center;
    float radius;
    int material_id;
    
    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Vec3 oc = r.origin - center;
        float a = r.direction.length_squared();
        float half_b = dot(oc, r.direction);
        float c = oc.length_squared() - radius * radius;
        float discriminant = half_b * half_b - a * c;
        
        if (discriminant < 0) return false;
        
        float sqrtd = sqrtf(discriminant);
        float root = (-half_b - sqrtd) / a;
        
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max)
                return false;
        }
        
        rec.t = root;
        rec.point = r.at(root);
        Vec3 outward_normal = (rec.point - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.material_id = material_id;
        
        return true;
    }
    
    // Sample a point on the sphere (for light sampling)
    __device__ Point3 sample(RNG& rng) const {
        Vec3 dir = rng.unit_vector();
        return center + radius * dir;
    }
    
    // PDF for sampling this sphere from a point
    __device__ float pdf_value(const Point3& origin, const Vec3& direction) const {
        HitRecord rec;
        Ray r(origin, direction);
        if (!hit(r, 0.001f, 1e20f, rec))
            return 0;
        
        float cos_theta_max = sqrtf(1.0f - radius * radius / (center - origin).length_squared());
        float solid_angle = 2.0f * 3.14159265f * (1.0f - cos_theta_max);
        return 1.0f / solid_angle;
    }
};

struct Plane {
    Point3 point;
    Vec3 normal;
    int material_id;
    
    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float denom = dot(normal, r.direction);
        if (fabsf(denom) < 1e-8f) return false;
        
        float t = dot(point - r.origin, normal) / denom;
        if (t < t_min || t > t_max) return false;
        
        rec.t = t;
        rec.point = r.at(t);
        rec.set_face_normal(r, normal);
        rec.material_id = material_id;
        
        return true;
    }
};

// ============================================================================
// GPU Scene (flat arrays for device memory)
// ============================================================================

struct GPUScene {
    // Device pointers (allocated with cudaMalloc)
    Sphere* spheres;
    int num_spheres;
    
    Plane* planes;
    int num_planes;
    
    Material* materials;
    int num_materials;
    
    // Emissive sphere indices for light sampling
    int* emissive_spheres;
    int num_emissive;
    
    // Background color
    Color background;
    bool use_gradient;
    Color background_top;
    Color background_bottom;
    
    __device__ Color get_background(const Vec3& direction) const {
        if (use_gradient) {
            float t = 0.5f * (normalize(direction).y + 1.0f);
            return (1.0f - t) * background_bottom + t * background_top;
        }
        return background;
    }
    
    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest = t_max;
        
        // Test spheres
        for (int i = 0; i < num_spheres; i++) {
            if (spheres[i].hit(r, t_min, closest, temp_rec)) {
                hit_anything = true;
                closest = temp_rec.t;
                rec = temp_rec;
            }
        }
        
        // Test planes
        for (int i = 0; i < num_planes; i++) {
            if (planes[i].hit(r, t_min, closest, temp_rec)) {
                hit_anything = true;
                closest = temp_rec.t;
                rec = temp_rec;
            }
        }
        
        return hit_anything;
    }
    
    __device__ const Material& get_material(int id) const {
        return materials[id];
    }
    
    // Sample a random light
    __device__ bool sample_light(RNG& rng, Point3& light_pos, Color& light_emission, float& pdf) const {
        if (num_emissive == 0) return false;
        
        // Pick random emissive sphere
        int idx = (int)(rng.uniform() * num_emissive);
        if (idx >= num_emissive) idx = num_emissive - 1;
        
        int sphere_idx = emissive_spheres[idx];
        const Sphere& light_sphere = spheres[sphere_idx];
        const Material& mat = materials[light_sphere.material_id];
        
        light_pos = light_sphere.sample(rng);
        light_emission = mat.emission * mat.emission_strength;
        pdf = 1.0f / (float)num_emissive;  // Simplified - could weight by power
        
        return true;
    }
};

// ============================================================================
// Camera
// ============================================================================

struct GPUCamera {
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius;
    
    __host__ void setup(Point3 lookfrom, Point3 lookat, Vec3 vup, 
                        float vfov, float aspect_ratio, float aperture, float focus_dist) {
        float theta = vfov * 3.14159265f / 180.0f;
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;
        
        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);
        
        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - w;
        lens_radius = aperture / 2.0f;
    }
    
    __device__ Ray get_ray(float s, float t, RNG& rng) const {
        Vec3 rd = lens_radius * rng.in_unit_disk();
        Vec3 offset = u * rd.x + v * rd.y;
        return Ray(
            origin + offset,
            normalize(lower_left_corner + s * horizontal + t * vertical - origin - offset)
        );
    }
};

} // namespace cuda
