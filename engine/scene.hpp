/**
 * @file scene.hpp
 * @brief Scene management for ray tracing
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/hit.h"
}

#include "sphere.hpp"
#include "primitives.hpp"
#include "material.hpp"
#include "bvh.hpp"
#include <vector>
#include <limits>
#include <cmath>
#include <iostream>

namespace raytracer {

/**
 * @brief Point light source
 */
struct PointLight {
    point3 position;
    color intensity;  // Light color and brightness
    
    PointLight(point3 pos, color c) : position(pos), intensity(c) {}
};

/**
 * @brief Camera for generating rays
 */
struct Camera {
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    
    /**
     * @brief Create a camera with the given parameters
     * @param lookfrom Camera position
     * @param lookat Point to look at
     * @param vup View up vector
     * @param vfov Vertical field of view in degrees
     * @param aspect_ratio Width/height ratio
     */
    Camera(point3 lookfrom, point3 lookat, vec3 vup, double vfov, double aspect_ratio) {
        double theta = vfov * 3.14159265358979323846 / 180.0;
        double h = std::tan(theta / 2.0);
        double viewport_height = 2.0 * h;
        double viewport_width = aspect_ratio * viewport_height;
        
        vec3 w = vec3_normalize(vec3_sub(lookfrom, lookat));
        vec3 u = vec3_normalize(vec3_cross(vup, w));
        vec3 v = vec3_cross(w, u);
        
        origin = lookfrom;
        horizontal = vec3_scale(u, viewport_width);
        vertical = vec3_scale(v, viewport_height);
        
        // lower_left = origin - horizontal/2 - vertical/2 - w
        lower_left_corner = vec3_sub(
            vec3_sub(
                vec3_sub(origin, vec3_scale(horizontal, 0.5)),
                vec3_scale(vertical, 0.5)
            ),
            w
        );
    }
    
    /**
     * @brief Generate a ray for given screen coordinates
     * @param s Horizontal coordinate [0, 1]
     * @param t Vertical coordinate [0, 1]
     * @return Ray from camera through screen point
     */
    ray get_ray(double s, double t) const {
        vec3 direction = vec3_sub(
            vec3_add(
                vec3_add(lower_left_corner, vec3_scale(horizontal, s)),
                vec3_scale(vertical, t)
            ),
            origin
        );
        return ray_create(origin, direction);
    }
};

// Primitive type constants for BVH
constexpr int PRIM_SPHERE = 0;
constexpr int PRIM_PLANE = 1;
constexpr int PRIM_TRIANGLE = 2;
constexpr int PRIM_BOX = 3;

/**
 * @brief Scene containing objects and materials
 */
class Scene {
public:
    std::vector<Sphere> spheres;
    std::vector<Plane> planes;
    std::vector<Triangle> triangles;
    std::vector<Box> boxes;
    std::vector<Material> materials;
    std::vector<PointLight> lights;
    
private:
    mutable BVH bvh_;
    mutable bool bvh_dirty_ = true;
    
public:
    /**
     * @brief Add a material and return its ID
     */
    int add_material(const Material& mat) {
        int id = static_cast<int>(materials.size());
        materials.push_back(mat);
        return id;
    }
    
    /**
     * @brief Add a sphere to the scene
     */
    void add_sphere(point3 center, double radius, int material_id) {
        spheres.emplace_back(center, radius, material_id);
        bvh_dirty_ = true;
    }
    
    /**
     * @brief Add a plane to the scene
     */
    void add_plane(point3 point, vec3 normal, int material_id) {
        planes.emplace_back(point, normal, material_id);
        // Planes are infinite - handled separately, not in BVH
    }
    
    /**
     * @brief Add a triangle to the scene
     */
    void add_triangle(point3 v0, point3 v1, point3 v2, int material_id) {
        triangles.emplace_back(v0, v1, v2, material_id);
        bvh_dirty_ = true;
    }
    
    /**
     * @brief Add an axis-aligned box to the scene
     */
    void add_box(point3 min_pt, point3 max_pt, int material_id) {
        boxes.emplace_back(min_pt, max_pt, material_id);
        bvh_dirty_ = true;
    }
    
    /**
     * @brief Add a centered box to the scene
     */
    void add_box_centered(point3 center, double w, double h, double d, int material_id) {
        boxes.push_back(Box::centered(center, w, h, d, material_id));
        bvh_dirty_ = true;
    }
    
    /**
     * @brief Add a point light to the scene
     */
    void add_light(point3 position, color intensity) {
        lights.emplace_back(position, intensity);
    }
    
    /**
     * @brief Build or rebuild the BVH
     */
    void build_bvh() const {
        if (!bvh_dirty_) return;
        
        std::vector<BVHPrimitive> prims;
        prims.reserve(spheres.size() + triangles.size() + boxes.size());
        
        // Add spheres
        for (size_t i = 0; i < spheres.size(); ++i) {
            const auto& s = spheres[i];
            BVHPrimitive p;
            p.type = PRIM_SPHERE;
            p.index = static_cast<int>(i);
            // Sphere bounding box
            vec3 rad = {s.radius, s.radius, s.radius};
            p.bounds.min_pt = vec3_sub(s.center, rad);
            p.bounds.max_pt = vec3_add(s.center, rad);
            prims.push_back(p);
        }
        
        // Add triangles
        for (size_t i = 0; i < triangles.size(); ++i) {
            const auto& t = triangles[i];
            BVHPrimitive p;
            p.type = PRIM_TRIANGLE;
            p.index = static_cast<int>(i);
            p.bounds.expand(t.v0);
            p.bounds.expand(t.v1);
            p.bounds.expand(t.v2);
            prims.push_back(p);
        }
        
        // Add boxes
        for (size_t i = 0; i < boxes.size(); ++i) {
            const auto& b = boxes[i];
            BVHPrimitive p;
            p.type = PRIM_BOX;
            p.index = static_cast<int>(i);
            p.bounds.min_pt = b.box_min;
            p.bounds.max_pt = b.box_max;
            prims.push_back(p);
        }
        
        if (prims.empty()) {
            std::cout << "BVH: No bounded primitives to build" << std::endl;
            bvh_dirty_ = false;
            return;
        }
        
        bvh_.build(std::move(prims));
        bvh_dirty_ = false;
        
        std::cout << "BVH built: " << bvh_.nodes.size() << " nodes for " 
                  << bvh_.primitives.size() << " primitives" << std::endl;
    }
    
    /**
     * @brief Check if a point is in shadow from a light
     * @param point The surface point
     * @param light_pos The light position
     * @return true if point is in shadow
     */
    bool is_shadowed(point3 point, point3 light_pos) const {
        vec3 to_light = vec3_sub(light_pos, point);
        double distance = vec3_length(to_light);
        ray shadow_ray = ray_create(point, to_light);
        
        hit_record rec;
        if (hit(shadow_ray, 0.001, distance, rec)) {
            return true;
        }
        return false;
    }
    
    /**
     * @brief Test ray against all objects in scene (uses BVH)
     * @param r The ray
     * @param t_min Minimum t value
     * @param t_max Maximum t value
     * @param rec Hit record to populate
     * @return true if any intersection found
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        hit_record temp_rec = {};
        bool hit_anything = false;
        double closest_so_far = t_max;
        
        // Test BVH (spheres, triangles, boxes) using iterative stack-based traversal
        if (!bvh_.nodes.empty()) {
            // Stack for iterative traversal (avoid deep recursion issues)
            int stack[64];
            int stack_ptr = 0;
            stack[stack_ptr++] = 0;  // Start with root
            
            while (stack_ptr > 0 && stack_ptr < 64) {
                int node_idx = stack[--stack_ptr];
                
                // Bounds check
                if (node_idx < 0 || node_idx >= static_cast<int>(bvh_.nodes.size())) {
                    continue;
                }
                
                const BVHNode& node = bvh_.nodes[node_idx];
                
                // Test against node bounding box
                if (!node.bounds.hit(r, t_min, closest_so_far)) {
                    continue;
                }
                
                // Leaf node - test actual primitives
                if (node.prim_count > 0) {
                    for (int i = 0; i < node.prim_count; ++i) {
                        int prim_idx = node.prim_offset + i;
                        if (prim_idx < 0 || prim_idx >= static_cast<int>(bvh_.primitives.size())) {
                            continue;
                        }
                        const BVHPrimitive& prim = bvh_.primitives[prim_idx];
                        bool did_hit = false;
                        
                        switch (prim.type) {
                            case PRIM_SPHERE:
                                if (prim.index >= 0 && prim.index < static_cast<int>(spheres.size()))
                                    did_hit = spheres[prim.index].hit(r, t_min, closest_so_far, temp_rec);
                                break;
                            case PRIM_TRIANGLE:
                                if (prim.index >= 0 && prim.index < static_cast<int>(triangles.size()))
                                    did_hit = triangles[prim.index].hit(r, t_min, closest_so_far, temp_rec);
                                break;
                            case PRIM_BOX:
                                if (prim.index >= 0 && prim.index < static_cast<int>(boxes.size()))
                                    did_hit = boxes[prim.index].hit(r, t_min, closest_so_far, temp_rec);
                                break;
                        }
                        
                        if (did_hit) {
                            hit_anything = true;
                            closest_so_far = temp_rec.t;
                            rec = temp_rec;
                        }
                    }
                } else {
                    // Internal node - push children onto stack
                    if (node.right >= 0 && stack_ptr < 63) stack[stack_ptr++] = node.right;
                    if (node.left >= 0 && stack_ptr < 63) stack[stack_ptr++] = node.left;
                }
            }
        }
        
        // Test planes separately (infinite, can't be bounded)
        for (const auto& plane : planes) {
            if (plane.hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        
        return hit_anything;
    }
    
    /**
     * @brief Get material by ID
     */
    const Material& get_material(int id) const {
        return materials[id];
    }
};

} // namespace raytracer
