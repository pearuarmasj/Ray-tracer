/**
 * @file sphere.hpp
 * @brief Sphere geometry for ray tracing
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/hit.h"
}

#include <cmath>

namespace raytracer {

constexpr double PI = 3.14159265358979323846;

/**
 * @brief Calculate UV coordinates for a point on a unit sphere
 * @param p Point on unit sphere (normalized direction from center)
 * @param u Output U coordinate [0, 1]
 * @param v Output V coordinate [0, 1]
 */
inline void get_sphere_uv(const vec3& p, double& u, double& v) {
    double theta = std::acos(-p.y);
    double phi = std::atan2(-p.z, p.x) + PI;
    
    u = phi / (2.0 * PI);
    v = theta / PI;
}

/**
 * @brief Calculate tangent vector for a point on a unit sphere
 * @param p Point on unit sphere (normalized direction from center)
 * @return Tangent vector in direction of increasing U (longitude)
 */
inline vec3 get_sphere_tangent(const vec3& p) {
    // Tangent is perpendicular to normal (p) and points in direction of increasing phi
    // For spherical coordinates: tangent = d(point)/d(phi) = (-sin(phi), 0, cos(phi))
    // But we work with the normal p = (x, y, z) where x = sin(theta)*cos(phi), z = -sin(theta)*sin(phi)
    // The tangent in phi direction is proportional to (-z, 0, x) = cross(p, (0,1,0)) normalized
    // Handle pole case where p is nearly parallel to Y axis
    if (std::fabs(p.y) > 0.999) {
        return {1.0, 0.0, 0.0};  // At poles, pick arbitrary tangent
    }
    vec3 tangent = {-p.z, 0.0, p.x};  // Cross product of p with Y-up
    return vec3_normalize(tangent);
}

/**
 * @brief Sphere primitive
 */
struct Sphere {
    point3 center;
    double radius;
    int material_id;
    
    Sphere(point3 c, double r, int mat_id = 0) 
        : center(c), radius(r), material_id(mat_id) {}
    
    /**
     * @brief Test ray-sphere intersection
     * @param r The ray
     * @param t_min Minimum t value
     * @param t_max Maximum t value
     * @param rec Hit record to populate
     * @return true if intersection found
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        vec3 oc = vec3_sub(r.origin, center);
        
        double a = vec3_length_squared(r.direction);
        double half_b = vec3_dot(oc, r.direction);
        double c = vec3_length_squared(oc) - radius * radius;
        
        double discriminant = half_b * half_b - a * c;
        if (discriminant < 0) {
            return false;
        }
        
        double sqrtd = std::sqrt(discriminant);
        
        // Find the nearest root in the acceptable range
        double root = (-half_b - sqrtd) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max) {
                return false;
            }
        }
        
        rec.t = root;
        rec.point = ray_at(r, rec.t);
        vec3 outward_normal = vec3_scale(vec3_sub(rec.point, center), 1.0 / radius);
        hit_record_set_face_normal(&rec, r, outward_normal);
        rec.material_id = material_id;
        
        // Calculate UV coordinates and tangent
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.tangent = get_sphere_tangent(outward_normal);
        
        return true;
    }
};

} // namespace raytracer
