/**
 * @file photon_map.hpp
 * @brief Photon mapping data structures and kd-tree
 * 
 * Implements photon storage with a kd-tree for efficient nearest-neighbor
 * queries used in density estimation.
 * 
 * Reference: "Realistic Image Synthesis Using Photon Mapping" Henrik Wann Jensen
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
}

#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace raytracer {

/**
 * @brief A single photon stored in the photon map
 */
struct Photon {
    point3 position;      // World position where photon hit
    vec3 direction;       // Incoming direction (toward surface)
    color power;          // Photon power (flux)
    uint8_t flags;        // Bit flags: 0=diffuse, 1=caustic, 2=volume
    uint8_t axis;         // Split axis for kd-tree (0=x, 1=y, 2=z)
    
    enum Flag : uint8_t {
        DIFFUSE = 0,
        CAUSTIC = 1,
        VOLUME = 2
    };
    
    bool is_caustic() const { return flags == CAUSTIC; }
    bool is_diffuse() const { return flags == DIFFUSE; }
};

/**
 * @brief Result of a nearest-neighbor query
 */
struct PhotonQueryResult {
    const Photon* photon;
    double distance_squared;
    
    bool operator<(const PhotonQueryResult& other) const {
        return distance_squared < other.distance_squared;
    }
    
    bool operator>(const PhotonQueryResult& other) const {
        return distance_squared > other.distance_squared;
    }
};

/**
 * @brief Photon map with kd-tree for spatial queries
 * 
 * The kd-tree is built in-place using a heap-like layout for cache efficiency.
 * Queries use a max-heap to find k nearest neighbors.
 */
class PhotonMap {
public:
    PhotonMap() = default;
    
    /**
     * @brief Reserve space for expected number of photons
     */
    void reserve(size_t count) {
        photons_.reserve(count);
    }
    
    /**
     * @brief Add a photon to the map (before building)
     */
    void add_photon(const Photon& photon) {
        photons_.push_back(photon);
    }
    
    /**
     * @brief Add a photon with explicit parameters
     */
    void add_photon(point3 pos, vec3 dir, color pow, Photon::Flag flag) {
        Photon p;
        p.position = pos;
        p.direction = dir;
        p.power = pow;
        p.flags = static_cast<uint8_t>(flag);
        p.axis = 0;
        photons_.push_back(p);
    }
    
    /**
     * @brief Build the kd-tree (call after adding all photons)
     */
    void build() {
        if (photons_.empty()) return;
        build_tree(0, photons_.size(), 0);
        built_ = true;
    }
    
    /**
     * @brief Find k nearest photons to a query point
     * @param pos Query position
     * @param k Maximum number of photons to find
     * @param max_dist_sq Maximum squared distance to search
     * @return Vector of query results sorted by distance
     */
    std::vector<PhotonQueryResult> find_nearest(
        point3 pos, 
        int k, 
        double max_dist_sq
    ) const {
        if (!built_ || photons_.empty()) return {};
        
        // Max-heap to track k nearest (furthest at top for easy replacement)
        std::priority_queue<PhotonQueryResult> heap;
        
        // Search the tree
        search_tree(0, photons_.size(), pos, k, max_dist_sq, heap);
        
        // Convert heap to sorted vector
        std::vector<PhotonQueryResult> results;
        results.reserve(heap.size());
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }
        std::reverse(results.begin(), results.end());
        
        return results;
    }
    
    /**
     * @brief Estimate irradiance at a point using nearby photons
     * @param pos Surface position
     * @param normal Surface normal
     * @param k Number of photons to gather
     * @param max_radius Maximum search radius
     * @return Estimated irradiance (power per area)
     */
    color estimate_irradiance(
        point3 pos,
        vec3 normal,
        int k,
        double max_radius
    ) const {
        double max_dist_sq = max_radius * max_radius;
        auto nearest = find_nearest(pos, k, max_dist_sq);
        
        if (nearest.empty()) return vec3_zero();
        
        // Find actual radius (distance to furthest photon found)
        double r_sq = nearest.back().distance_squared;
        if (r_sq < 1e-10) r_sq = 1e-10;  // Avoid division by zero
        
        // Sum photon contributions
        color flux = vec3_zero();
        for (const auto& result : nearest) {
            const Photon* p = result.photon;
            
            // Filter by normal (only count photons from correct hemisphere)
            double cos_theta = -vec3_dot(p->direction, normal);
            if (cos_theta <= 0) continue;
            
            // Cone filter for smoother results (weight by distance)
            double w = 1.0 - std::sqrt(result.distance_squared / r_sq);
            
            flux = vec3_add(flux, vec3_scale(p->power, w));
        }
        
        // Density estimation: divide by area of disk
        // Using cone filter normalization factor
        double area = M_PI * r_sq;
        double cone_filter_norm = 3.0;  // Normalization for cone filter
        
        return vec3_scale(flux, cone_filter_norm / area);
    }
    
    /**
     * @brief Get total number of stored photons
     */
    size_t size() const { return photons_.size(); }
    
    /**
     * @brief Check if the tree has been built
     */
    bool is_built() const { return built_; }
    
    /**
     * @brief Clear all photons
     */
    void clear() {
        photons_.clear();
        built_ = false;
    }
    
    /**
     * @brief Scale all photon powers by a factor
     */
    void scale_power(double factor) {
        for (auto& p : photons_) {
            p.power = vec3_scale(p.power, factor);
        }
    }

private:
    std::vector<Photon> photons_;
    bool built_ = false;
    
    /**
     * @brief Recursively build kd-tree
     */
    void build_tree(size_t start, size_t end, int depth) {
        if (end - start <= 1) return;
        
        // Choose split axis based on depth (cycle through x, y, z)
        int axis = depth % 3;
        
        // Find median and partition
        size_t mid = start + (end - start) / 2;
        std::nth_element(
            photons_.begin() + start,
            photons_.begin() + mid,
            photons_.begin() + end,
            [axis](const Photon& a, const Photon& b) {
                return get_axis(a.position, axis) < get_axis(b.position, axis);
            }
        );
        
        // Store split axis in median photon
        photons_[mid].axis = static_cast<uint8_t>(axis);
        
        // Recursively build subtrees
        build_tree(start, mid, depth + 1);
        build_tree(mid + 1, end, depth + 1);
    }
    
    /**
     * @brief Recursively search kd-tree for nearest neighbors
     */
    void search_tree(
        size_t start,
        size_t end,
        point3 pos,
        int k,
        double max_dist_sq,
        std::priority_queue<PhotonQueryResult>& heap
    ) const {
        if (start >= end) return;
        
        size_t mid = start + (end - start) / 2;
        const Photon& photon = photons_[mid];
        
        // Compute distance to this photon
        vec3 diff = vec3_sub(photon.position, pos);
        double dist_sq = vec3_length_squared(diff);
        
        // Current search radius (either max_dist_sq or furthest in heap)
        double search_radius_sq = max_dist_sq;
        if (static_cast<int>(heap.size()) >= k) {
            search_radius_sq = std::min(search_radius_sq, heap.top().distance_squared);
        }
        
        // Add this photon if close enough
        if (dist_sq < search_radius_sq) {
            PhotonQueryResult result{&photon, dist_sq};
            
            if (static_cast<int>(heap.size()) < k) {
                heap.push(result);
            } else if (dist_sq < heap.top().distance_squared) {
                heap.pop();
                heap.push(result);
            }
        }
        
        // Determine which subtree to search first
        int axis = photon.axis;
        double delta = get_axis(pos, axis) - get_axis(photon.position, axis);
        double delta_sq = delta * delta;
        
        // Search near subtree first
        if (delta < 0) {
            search_tree(start, mid, pos, k, max_dist_sq, heap);
            
            // Only search far subtree if it could contain closer photons
            if (static_cast<int>(heap.size()) < k || delta_sq < heap.top().distance_squared) {
                search_tree(mid + 1, end, pos, k, max_dist_sq, heap);
            }
        } else {
            search_tree(mid + 1, end, pos, k, max_dist_sq, heap);
            
            if (static_cast<int>(heap.size()) < k || delta_sq < heap.top().distance_squared) {
                search_tree(start, mid, pos, k, max_dist_sq, heap);
            }
        }
    }
    
    /**
     * @brief Get component of position by axis index
     */
    static double get_axis(const point3& p, int axis) {
        switch (axis) {
            case 0: return p.x;
            case 1: return p.y;
            default: return p.z;
        }
    }
};

} // namespace raytracer
