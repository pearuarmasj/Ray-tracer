/**
 * @file bvh.hpp
 * @brief Bounding Volume Hierarchy for ray tracing acceleration
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/hit.h"
}

#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <functional>

namespace raytracer {

/**
 * @brief Axis-Aligned Bounding Box
 */
struct AABB {
    point3 min_pt = {1e30, 1e30, 1e30};
    point3 max_pt = {-1e30, -1e30, -1e30};
    
    AABB() = default;
    AABB(point3 a, point3 b) : min_pt(a), max_pt(b) {}
    
    /**
     * @brief Test ray-AABB intersection using slab method
     */
    bool hit(ray r, double t_min, double t_max) const {
        // X slab
        double inv_d = 1.0 / r.direction.x;
        double t0 = (min_pt.x - r.origin.x) * inv_d;
        double t1 = (max_pt.x - r.origin.x) * inv_d;
        if (inv_d < 0.0) std::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;
        
        // Y slab
        inv_d = 1.0 / r.direction.y;
        t0 = (min_pt.y - r.origin.y) * inv_d;
        t1 = (max_pt.y - r.origin.y) * inv_d;
        if (inv_d < 0.0) std::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;
        
        // Z slab
        inv_d = 1.0 / r.direction.z;
        t0 = (min_pt.z - r.origin.z) * inv_d;
        t1 = (max_pt.z - r.origin.z) * inv_d;
        if (inv_d < 0.0) std::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;
        
        return true;
    }
    
    /**
     * @brief Expand this AABB to include a point
     */
    void expand(point3 p) {
        min_pt.x = std::fmin(min_pt.x, p.x);
        min_pt.y = std::fmin(min_pt.y, p.y);
        min_pt.z = std::fmin(min_pt.z, p.z);
        max_pt.x = std::fmax(max_pt.x, p.x);
        max_pt.y = std::fmax(max_pt.y, p.y);
        max_pt.z = std::fmax(max_pt.z, p.z);
    }
    
    /**
     * @brief Expand this AABB to include another AABB
     */
    void expand(const AABB& other) {
        expand(other.min_pt);
        expand(other.max_pt);
    }
    
    /**
     * @brief Get centroid of the AABB
     */
    point3 centroid() const {
        return {
            (min_pt.x + max_pt.x) * 0.5,
            (min_pt.y + max_pt.y) * 0.5,
            (min_pt.z + max_pt.z) * 0.5
        };
    }
    
    /**
     * @brief Get the longest axis (0=x, 1=y, 2=z)
     */
    int longest_axis() const {
        double dx = max_pt.x - min_pt.x;
        double dy = max_pt.y - min_pt.y;
        double dz = max_pt.z - min_pt.z;
        if (dx > dy && dx > dz) return 0;
        if (dy > dz) return 1;
        return 2;
    }
};

/**
 * @brief Primitive reference for BVH building
 */
struct BVHPrimitive {
    AABB bounds;
    int index;      // Index into the original array
    int type;       // 0=sphere, 1=plane, 2=triangle, 3=box
    
    point3 centroid() const { return bounds.centroid(); }
};

/**
 * @brief BVH Node
 */
struct BVHNode {
    AABB bounds;
    int left = -1;      // Index of left child (-1 if leaf)
    int right = -1;     // Index of right child
    int prim_offset = 0; // Start index in sorted primitive array
    int prim_count = 0;  // Number of primitives (>0 means leaf)
};

/**
 * @brief BVH acceleration structure
 */
class BVH {
public:
    std::vector<BVHNode> nodes;
    std::vector<BVHPrimitive> primitives;
    
    // Hit function callback type
    using HitFunc = std::function<bool(int type, int index, ray r, double t_min, double t_max, hit_record& rec)>;
    
    BVH() = default;
    
    /**
     * @brief Build BVH from primitives
     */
    void build(std::vector<BVHPrimitive> prims) {
        if (prims.empty()) return;
        
        primitives = std::move(prims);
        nodes.clear();
        nodes.reserve(primitives.size() * 2);
        
        build_recursive(0, static_cast<int>(primitives.size()));
    }
    
    /**
     * @brief Test ray against BVH
     */
    bool hit(ray r, double t_min, double t_max, hit_record& rec, const HitFunc& hit_func) const {
        if (nodes.empty()) return false;
        return hit_recursive(0, r, t_min, t_max, rec, hit_func);
    }
    
private:
    int build_recursive(int start, int end) {
        int node_idx = static_cast<int>(nodes.size());
        nodes.emplace_back();
        BVHNode& node = nodes[node_idx];
        
        // Compute bounds for this node
        for (int i = start; i < end; ++i) {
            node.bounds.expand(primitives[i].bounds);
        }
        
        int count = end - start;
        
        // Leaf node if few primitives
        if (count <= 2) {
            node.prim_offset = start;
            node.prim_count = count;
            return node_idx;
        }
        
        // Find best axis to split on
        AABB centroid_bounds;
        for (int i = start; i < end; ++i) {
            centroid_bounds.expand(primitives[i].centroid());
        }
        int axis = centroid_bounds.longest_axis();
        
        // Sort primitives along axis
        int mid = start + count / 2;
        std::nth_element(
            primitives.begin() + start,
            primitives.begin() + mid,
            primitives.begin() + end,
            [axis](const BVHPrimitive& a, const BVHPrimitive& b) {
                point3 ca = a.centroid();
                point3 cb = b.centroid();
                if (axis == 0) return ca.x < cb.x;
                if (axis == 1) return ca.y < cb.y;
                return ca.z < cb.z;
            }
        );
        
        // Build children (need to update node reference after recursive calls
        // since vector may reallocate)
        int left_idx = build_recursive(start, mid);
        int right_idx = build_recursive(mid, end);
        
        nodes[node_idx].left = left_idx;
        nodes[node_idx].right = right_idx;
        
        return node_idx;
    }
    
    bool hit_recursive(int node_idx, ray r, double t_min, double t_max, 
                       hit_record& rec, const HitFunc& hit_func) const {
        const BVHNode& node = nodes[node_idx];
        
        // Test against node bounding box
        if (!node.bounds.hit(r, t_min, t_max)) {
            return false;
        }
        
        // Leaf node - test actual primitives
        if (node.prim_count > 0) {
            bool hit_anything = false;
            for (int i = 0; i < node.prim_count; ++i) {
                const BVHPrimitive& prim = primitives[node.prim_offset + i];
                if (hit_func(prim.type, prim.index, r, t_min, t_max, rec)) {
                    hit_anything = true;
                    t_max = rec.t;  // Shrink range for closer hits
                }
            }
            return hit_anything;
        }
        
        // Internal node - recurse into children
        bool hit_left = hit_recursive(node.left, r, t_min, t_max, rec, hit_func);
        bool hit_right = hit_recursive(node.right, r, t_min, hit_left ? rec.t : t_max, rec, hit_func);
        
        return hit_left || hit_right;
    }
};

} // namespace raytracer
