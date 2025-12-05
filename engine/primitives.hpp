/**
 * @file primitives.hpp
 * @brief Additional geometric primitives for ray tracing
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/hit.h"
}

#include <cmath>
#include <algorithm>

namespace raytracer {

/**
 * @brief Infinite plane primitive
 * 
 * Defined by a point on the plane and a normal vector.
 */
struct Plane {
    point3 point;      // A point on the plane
    vec3 normal;       // Normal vector (should be normalized)
    int material_id;
    
    Plane(point3 p, vec3 n, int mat_id = 0) 
        : point(p), normal(vec3_normalize(n)), material_id(mat_id) {}
    
    /**
     * @brief Test ray-plane intersection
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        double denom = vec3_dot(normal, r.direction);
        
        // Check if ray is parallel to plane (or nearly so)
        if (std::fabs(denom) < 1e-8) {
            return false;
        }
        
        vec3 p0_to_origin = vec3_sub(point, r.origin);
        double t = vec3_dot(p0_to_origin, normal) / denom;
        
        if (t < t_min || t > t_max) {
            return false;
        }
        
        rec.t = t;
        rec.point = ray_at(r, t);
        hit_record_set_face_normal(&rec, r, normal);
        rec.material_id = material_id;
        
        return true;
    }
};

/**
 * @brief Triangle primitive
 * 
 * Defined by three vertices.
 */
struct Triangle {
    point3 v0, v1, v2;
    int material_id;
    
    Triangle(point3 a, point3 b, point3 c, int mat_id = 0) 
        : v0(a), v1(b), v2(c), material_id(mat_id) {}
    
    /**
     * @brief Test ray-triangle intersection using Möller–Trumbore algorithm
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        const double EPSILON = 1e-8;
        
        vec3 edge1 = vec3_sub(v1, v0);
        vec3 edge2 = vec3_sub(v2, v0);
        vec3 h = vec3_cross(r.direction, edge2);
        double a = vec3_dot(edge1, h);
        
        // Ray is parallel to triangle
        if (std::fabs(a) < EPSILON) {
            return false;
        }
        
        double f = 1.0 / a;
        vec3 s = vec3_sub(r.origin, v0);
        double u = f * vec3_dot(s, h);
        
        if (u < 0.0 || u > 1.0) {
            return false;
        }
        
        vec3 q = vec3_cross(s, edge1);
        double v = f * vec3_dot(r.direction, q);
        
        if (v < 0.0 || u + v > 1.0) {
            return false;
        }
        
        double t = f * vec3_dot(edge2, q);
        
        if (t < t_min || t > t_max) {
            return false;
        }
        
        rec.t = t;
        rec.point = ray_at(r, t);
        vec3 outward_normal = vec3_normalize(vec3_cross(edge1, edge2));
        hit_record_set_face_normal(&rec, r, outward_normal);
        rec.material_id = material_id;
        
        return true;
    }
};

/**
 * @brief Axis-aligned box primitive
 * 
 * Defined by minimum and maximum corners.
 */
struct Box {
    point3 box_min;
    point3 box_max;
    int material_id;
    
    Box(point3 min_pt, point3 max_pt, int mat_id = 0)
        : box_min(min_pt), box_max(max_pt), material_id(mat_id) {}
    
    /**
     * @brief Create a box centered at a point with given dimensions
     */
    static Box centered(point3 center, double width, double height, double depth, int mat_id = 0) {
        vec3 half = {width / 2.0, height / 2.0, depth / 2.0};
        return Box(vec3_sub(center, half), vec3_add(center, half), mat_id);
    }
    
    /**
     * @brief Test ray-box intersection using slab method
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        double tmin = t_min;
        double tmax = t_max;
        
        // For each axis
        for (int axis = 0; axis < 3; ++axis) {
            double origin = (axis == 0) ? r.origin.x : (axis == 1) ? r.origin.y : r.origin.z;
            double dir = (axis == 0) ? r.direction.x : (axis == 1) ? r.direction.y : r.direction.z;
            double bmin = (axis == 0) ? box_min.x : (axis == 1) ? box_min.y : box_min.z;
            double bmax = (axis == 0) ? box_max.x : (axis == 1) ? box_max.y : box_max.z;
            
            double invD = 1.0 / dir;
            double t0 = (bmin - origin) * invD;
            double t1 = (bmax - origin) * invD;
            
            if (invD < 0.0) {
                std::swap(t0, t1);
            }
            
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            
            if (tmax <= tmin) {
                return false;
            }
        }
        
        rec.t = tmin;
        rec.point = ray_at(r, tmin);
        
        // Calculate normal based on which face was hit
        vec3 outward_normal = {0, 0, 0};
        const double EPSILON = 1e-4;
        
        if (std::fabs(rec.point.x - box_min.x) < EPSILON) outward_normal = {-1, 0, 0};
        else if (std::fabs(rec.point.x - box_max.x) < EPSILON) outward_normal = {1, 0, 0};
        else if (std::fabs(rec.point.y - box_min.y) < EPSILON) outward_normal = {0, -1, 0};
        else if (std::fabs(rec.point.y - box_max.y) < EPSILON) outward_normal = {0, 1, 0};
        else if (std::fabs(rec.point.z - box_min.z) < EPSILON) outward_normal = {0, 0, -1};
        else outward_normal = {0, 0, 1};
        
        hit_record_set_face_normal(&rec, r, outward_normal);
        rec.material_id = material_id;
        
        return true;
    }
};

} // namespace raytracer
