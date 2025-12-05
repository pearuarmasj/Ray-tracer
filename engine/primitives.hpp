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
        
        // Compute tangent (arbitrary direction perpendicular to normal)
        vec3 up = (std::fabs(normal.y) < 0.999) ? vec3{0, 1, 0} : vec3{1, 0, 0};
        rec.tangent = vec3_normalize(vec3_cross(up, normal));
        
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
        
        // Tangent is along the first edge
        rec.tangent = vec3_normalize(edge1);
        
        // Store barycentric as UV (u already computed above)
        rec.u = u;
        rec.v = v;
        
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
        
        // Compute tangent based on face orientation
        if (std::fabs(outward_normal.x) > 0.5) {
            rec.tangent = {0, 0, 1};  // X-facing: tangent along Z
        } else if (std::fabs(outward_normal.y) > 0.5) {
            rec.tangent = {1, 0, 0};  // Y-facing: tangent along X
        } else {
            rec.tangent = {1, 0, 0};  // Z-facing: tangent along X
        }
        
        return true;
    }
};

// Forward declare random function
double random_double();

/**
 * @brief Quad (rectangular) area light
 * 
 * Defined by a corner point and two edge vectors.
 * The light emits from one side (determined by the cross product of edges).
 */
struct QuadLight {
    point3 corner;      // Corner position
    vec3 edge_u;        // First edge vector (from corner)
    vec3 edge_v;        // Second edge vector (from corner)
    vec3 normal;        // Normal (computed from edges, points toward emission)
    color emission;     // Emitted radiance
    double area;        // Surface area
    int material_id;    // Optional material for visibility (emissive)
    
    QuadLight(point3 corner_, vec3 u, vec3 v, color emit, int mat_id = -1)
        : corner(corner_), edge_u(u), edge_v(v), emission(emit), material_id(mat_id) 
    {
        vec3 cross = vec3_cross(edge_u, edge_v);
        area = vec3_length(cross);
        normal = vec3_scale(cross, 1.0 / area);  // Normalize
    }
    
    /**
     * @brief Create a centered quad light
     * @param center Center position of the quad
     * @param u_dir Direction of first edge (will be normalized and scaled by width)
     * @param v_dir Direction of second edge (will be normalized and scaled by height)
     * @param width Width along u direction
     * @param height Height along v direction
     * @param emit Emission color/intensity
     */
    static QuadLight centered(point3 center, vec3 u_dir, vec3 v_dir, 
                               double width, double height, color emit, int mat_id = -1) {
        vec3 u = vec3_scale(vec3_normalize(u_dir), width);
        vec3 v = vec3_scale(vec3_normalize(v_dir), height);
        // Corner = center - u/2 - v/2
        point3 corner = vec3_sub(vec3_sub(center, vec3_scale(u, 0.5)), vec3_scale(v, 0.5));
        return QuadLight(corner, u, v, emit, mat_id);
    }
    
    /**
     * @brief Sample a random point on the quad
     * @return Point on the quad surface
     */
    point3 sample_point() const {
        double u = random_double();
        double v = random_double();
        return vec3_add(corner, vec3_add(vec3_scale(edge_u, u), vec3_scale(edge_v, v)));
    }
    
    /**
     * @brief Get the PDF for uniform sampling (1/area)
     */
    double pdf() const {
        return 1.0 / area;
    }
    
    /**
     * @brief Test ray-quad intersection
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        // Plane intersection first
        double denom = vec3_dot(normal, r.direction);
        if (std::fabs(denom) < 1e-8) {
            return false;
        }
        
        vec3 corner_to_origin = vec3_sub(corner, r.origin);
        double t = vec3_dot(corner_to_origin, normal) / denom;
        
        if (t < t_min || t > t_max) {
            return false;
        }
        
        // Check if hit point is within quad bounds
        point3 p = ray_at(r, t);
        vec3 d = vec3_sub(p, corner);
        
        // Project onto edge vectors
        double u_len_sq = vec3_length_squared(edge_u);
        double v_len_sq = vec3_length_squared(edge_v);
        double u_proj = vec3_dot(d, edge_u) / u_len_sq;
        double v_proj = vec3_dot(d, edge_v) / v_len_sq;
        
        if (u_proj < 0.0 || u_proj > 1.0 || v_proj < 0.0 || v_proj > 1.0) {
            return false;
        }
        
        rec.t = t;
        rec.point = p;
        hit_record_set_face_normal(&rec, r, normal);
        rec.material_id = material_id;
        rec.u = u_proj;
        rec.v = v_proj;
        
        // Tangent is along edge_u
        rec.tangent = vec3_normalize(edge_u);
        
        return true;
    }
};

/**
 * @brief Disk area light
 * 
 * Circular light defined by center, normal, and radius.
 */
struct DiskLight {
    point3 center;      // Center position
    vec3 normal;        // Normal direction (emission direction)
    double radius;      // Disk radius
    color emission;     // Emitted radiance
    double area;        // Surface area (PI * r^2)
    int material_id;    // Optional material for visibility
    
    // Local coordinate frame for sampling
    vec3 tangent;       // U direction on disk
    vec3 bitangent;     // V direction on disk
    
    DiskLight(point3 center_, vec3 normal_, double radius_, color emit, int mat_id = -1)
        : center(center_), normal(vec3_normalize(normal_)), radius(radius_), 
          emission(emit), material_id(mat_id)
    {
        constexpr double PI = 3.14159265358979323846;
        area = PI * radius * radius;
        
        // Build local coordinate frame
        vec3 up = (std::fabs(normal.y) < 0.999) ? vec3{0, 1, 0} : vec3{1, 0, 0};
        tangent = vec3_normalize(vec3_cross(up, normal));
        bitangent = vec3_cross(normal, tangent);
    }
    
    /**
     * @brief Sample a random point on the disk (uniform by area)
     * @return Point on the disk surface
     */
    point3 sample_point() const {
        // Uniform disk sampling using concentric mapping
        double u = random_double();
        double v = random_double();
        
        // Map to [-1, 1]
        double sx = 2.0 * u - 1.0;
        double sy = 2.0 * v - 1.0;
        
        double r_sample, theta;
        if (sx == 0 && sy == 0) {
            return center;
        }
        
        // Concentric disk mapping
        if (std::fabs(sx) > std::fabs(sy)) {
            r_sample = sx;
            theta = (3.14159265358979323846 / 4.0) * (sy / sx);
        } else {
            r_sample = sy;
            theta = (3.14159265358979323846 / 2.0) - (3.14159265358979323846 / 4.0) * (sx / sy);
        }
        
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        double actual_r = radius * r_sample;
        
        // Convert to world coordinates
        vec3 offset = vec3_add(
            vec3_scale(tangent, actual_r * cos_theta),
            vec3_scale(bitangent, actual_r * sin_theta)
        );
        
        return vec3_add(center, offset);
    }
    
    /**
     * @brief Get the PDF for uniform sampling (1/area)
     */
    double pdf() const {
        return 1.0 / area;
    }
    
    /**
     * @brief Test ray-disk intersection
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        // Plane intersection first
        double denom = vec3_dot(normal, r.direction);
        if (std::fabs(denom) < 1e-8) {
            return false;
        }
        
        vec3 center_to_origin = vec3_sub(center, r.origin);
        double t = vec3_dot(center_to_origin, normal) / denom;
        
        if (t < t_min || t > t_max) {
            return false;
        }
        
        // Check if hit point is within disk radius
        point3 p = ray_at(r, t);
        vec3 d = vec3_sub(p, center);
        double dist_sq = vec3_length_squared(d);
        
        if (dist_sq > radius * radius) {
            return false;
        }
        
        rec.t = t;
        rec.point = p;
        hit_record_set_face_normal(&rec, r, normal);
        rec.material_id = material_id;
        
        // Tangent is the disk's tangent vector
        rec.tangent = tangent;
        
        // Compute UV (polar coordinates on disk)
        double dist = std::sqrt(dist_sq);
        rec.u = (dist > 1e-8) ? (vec3_dot(d, tangent) / dist + 1.0) * 0.5 : 0.5;
        rec.v = (dist > 1e-8) ? (vec3_dot(d, bitangent) / dist + 1.0) * 0.5 : 0.5;
        
        return true;
    }
};

} // namespace raytracer
