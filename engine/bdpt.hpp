/**
 * @file bdpt.hpp
 * @brief Bidirectional Path Tracing (BDPT) implementation
 * 
 * BDPT traces paths from both the camera and light sources, then connects
 * them to form complete light transport paths. This is more efficient than
 * unidirectional path tracing for:
 * - Caustics (light focused through glass onto diffuse surfaces)
 * - Small or indirect light sources
 * - Interior scenes with complex light bounces
 * 
 * The algorithm:
 * 1. Trace a path from the camera (eye subpath)
 * 2. Trace a path from a light source (light subpath)  
 * 3. Connect vertices from both subpaths in all valid combinations
 * 4. Weight each connection using Multiple Importance Sampling (MIS)
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/hit.h"
}

#include "scene.hpp"
#include "material.hpp"
#include <vector>
#include <cmath>

namespace raytracer {

// Forward declarations
double random_double();
double random_double(double min, double max);
vec3 random_unit_vector();

/**
 * @brief Vertex type in a light transport path
 */
enum class VertexType {
    Camera,     // Camera/eye vertex (t=0 in eye subpath)
    Light,      // Light source vertex (s=0 in light subpath)
    Surface,    // Surface interaction vertex
    Medium      // Medium/volume interaction (future: volumetrics)
};

/**
 * @brief A vertex in a light transport path
 * 
 * Stores all information needed to evaluate and connect path vertices.
 */
struct PathVertex {
    VertexType type = VertexType::Surface;
    
    // Geometry
    point3 position = {0, 0, 0};
    vec3 normal = {0, 1, 0};        // Shading normal
    vec3 geo_normal = {0, 1, 0};    // Geometric normal
    
    // Material info
    int material_id = -1;
    color albedo = {1, 1, 1};
    MaterialType mat_type = MaterialType::Lambertian;
    double roughness = 0.0;             // For metals
    double refraction_index = 1.5;      // For dielectrics
    
    // Path tracing state
    color throughput = {1, 1, 1};   // Accumulated throughput to this vertex
    double pdf_fwd = 0.0;           // PDF for sampling this vertex (forward)
    double pdf_rev = 0.0;           // PDF for sampling this vertex (reverse)
    
    // For light vertices
    color emission = {0, 0, 0};     // Light emission (if light vertex)
    double light_pdf = 0.0;         // PDF of choosing this light
    
    // Connection info
    vec3 wi = {0, 0, 0};            // Incoming direction (toward previous vertex)
    vec3 wo = {0, 0, 0};            // Outgoing direction (toward next vertex)
    
    // Texture coordinates
    double u = 0.0, v = 0.0;
    
    /**
     * @brief Check if vertex is on a delta distribution (perfect specular)
     */
    bool is_delta() const {
        return mat_type == MaterialType::Metal && roughness < 0.001;
        // Note: Dielectric is also delta but we handle it specially
    }
    
    /**
     * @brief Check if this is a connectible vertex
     * Delta distributions cannot be connected (infinite PDF)
     */
    bool is_connectible() const {
        if (type == VertexType::Light) return true;
        if (type == VertexType::Camera) return true;
        return !is_delta() && mat_type != MaterialType::Dielectric;
    }
    
    /**
     * @brief Check if vertex is on a light source
     */
    bool is_light() const {
        return type == VertexType::Light || 
               (vec3_length_squared(emission) > 0.001);
    }
};

/**
 * @brief Bidirectional Path Tracer
 */
class BDPTIntegrator {
public:
    /**
     * @brief BDPT settings
     */
    struct Settings {
        int max_eye_depth = 8;      // Maximum eye subpath length
        int max_light_depth = 8;    // Maximum light subpath length
        bool use_mis = true;        // Use MIS weighting
        double clamp_max = 10.0;    // Maximum contribution per sample (firefly reduction)
        bool debug_strategy = false; // Debug: only use specific (s,t) strategy
        int debug_s = -1;           // Debug: light subpath length
        int debug_t = -1;           // Debug: eye subpath length
    };
    
    explicit BDPTIntegrator(const Settings& settings = Settings())
        : settings_(settings) {}
    
    /**
     * @brief Compute radiance for a camera ray using BDPT
     * @param initial_ray Ray from camera
     * @param scene The scene
     * @return Computed radiance
     */
    color Li(ray initial_ray, const Scene& scene) const;
    
    Settings& settings() { return settings_; }
    const Settings& settings() const { return settings_; }
    
private:
    Settings settings_;
    
    /**
     * @brief Trace eye subpath from camera
     * @param ray Initial camera ray
     * @param scene The scene
     * @param path Output path vertices
     * @return Number of vertices in path (including camera)
     */
    int trace_eye_path(ray r, const Scene& scene, 
                       std::vector<PathVertex>& path) const;
    
    /**
     * @brief Trace light subpath from a light source
     * @param scene The scene
     * @param path Output path vertices
     * @return Number of vertices in path (including light)
     */
    int trace_light_path(const Scene& scene,
                         std::vector<PathVertex>& path) const;
    
    /**
     * @brief Sample a starting point on a light source
     * @param scene The scene
     * @param vertex Output light vertex
     * @param ray_dir Output: initial ray direction from light
     * @param pdf_pos Output: PDF of position
     * @param pdf_dir Output: PDF of direction
     * @return true if valid light sample
     */
    bool sample_light_origin(const Scene& scene, PathVertex& vertex,
                             vec3& ray_dir, double& pdf_pos, double& pdf_dir) const;
    
    /**
     * @brief Connect two path vertices and compute contribution
     * @param eye_path Eye subpath vertices
     * @param light_path Light subpath vertices  
     * @param s Number of light subpath vertices to use (0 = hit light directly)
     * @param t Number of eye subpath vertices to use (1 = direct camera connection)
     * @param scene The scene
     * @return Contribution from this connection strategy
     */
    color connect_paths(const std::vector<PathVertex>& eye_path,
                        const std::vector<PathVertex>& light_path,
                        int s, int t, const Scene& scene) const;
    
    /**
     * @brief Evaluate BSDF at a vertex
     * @param vertex The surface vertex
     * @param wi Incoming direction (toward light/previous)
     * @param wo Outgoing direction (toward camera/next)
     * @return BSDF value
     */
    color evaluate_bsdf(const PathVertex& vertex, vec3 wi, vec3 wo) const;
    
    /**
     * @brief PDF for sampling direction wo given wi at vertex
     */
    double bsdf_pdf(const PathVertex& vertex, vec3 wi, vec3 wo) const;
    
    /**
     * @brief Sample BSDF at vertex, returning new direction
     * @param vertex The surface vertex
     * @param wi Incoming direction
     * @param wo Output: sampled outgoing direction
     * @param pdf Output: PDF of sampled direction
     * @param bsdf_val Output: BSDF value for this sample
     * @return true if valid sample (not absorbed)
     */
    bool sample_bsdf(const PathVertex& vertex, vec3 wi,
                     vec3& wo, double& pdf, color& bsdf_val) const;
    
    /**
     * @brief Compute MIS weight for a path connection strategy
     * @param eye_path Eye subpath
     * @param light_path Light subpath
     * @param s Light subpath vertices used
     * @param t Eye subpath vertices used
     * @return MIS weight using power heuristic
     */
    double mis_weight(const std::vector<PathVertex>& eye_path,
                      const std::vector<PathVertex>& light_path,
                      int s, int t) const;
    
    /**
     * @brief Check visibility between two points
     */
    bool visible(const Scene& scene, point3 p1, point3 p2) const;
    
    /**
     * @brief Geometry term between two vertices
     * G(p1 <-> p2) = |cos(theta1)| * |cos(theta2)| / distance^2
     */
    double geometry_term(const PathVertex& v1, const PathVertex& v2) const;
};

} // namespace raytracer
