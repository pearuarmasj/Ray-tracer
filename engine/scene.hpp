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
#include "material.hpp"
#include <vector>
#include <limits>
#include <cmath>

namespace raytracer {

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

/**
 * @brief Scene containing objects and materials
 */
class Scene {
public:
    std::vector<Sphere> spheres;
    std::vector<Material> materials;
    
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
    }
    
    /**
     * @brief Test ray against all objects in scene
     * @param r The ray
     * @param t_min Minimum t value
     * @param t_max Maximum t value
     * @param rec Hit record to populate
     * @return true if any intersection found
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;
        
        for (const auto& sphere : spheres) {
            if (sphere.hit(r, t_min, closest_so_far, temp_rec)) {
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
