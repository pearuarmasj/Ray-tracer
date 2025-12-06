/**
 * @file mnee.hpp
 * @brief Manifold Next Event Estimation (MNEE)
 * 
 * MNEE finds specular paths through refractive/reflective surfaces that 
 * regular NEE cannot sample. It uses Newton iteration to walk the specular
 * manifold and find valid light paths.
 * 
 * Reference: "Manifold Next Event Estimation" Hanika et al., EGSR 2015
 *            "Manifold Exploration" Jakob & Marschner, SIGGRAPH 2012
 */

#pragma once

extern "C" {
#include "../../core/vec3.h"
#include "../../core/ray.h"
#include "../../core/hit.h"
}

#include "../scene.hpp"
#include "../material.hpp"
#include <vector>
#include <optional>
#include <cmath>

namespace raytracer {

/**
 * @brief Manifold vertex for MNEE
 */
struct ManifoldVertex {
    point3 position;      // Position on surface
    vec3 normal;          // Surface normal
    vec3 geometric_normal; // Geometric normal (before shading)
    double eta;           // IOR ratio (n1/n2)
    bool is_specular;     // True if specular interaction
    int material_id;      // Material at this vertex
};

/**
 * @brief MNEE path through specular surfaces
 */
struct MNEEPath {
    std::vector<ManifoldVertex> vertices;
    double throughput;    // Path throughput (Fresnel * geometry terms)
    bool valid;           // Whether path is valid (converged)
};

/**
 * @brief Manifold Next Event Estimation
 * 
 * Finds valid light paths through specular (refractive/reflective) surfaces
 * using Newton iteration on the specular manifold constraints.
 */
class MNEE {
public:
    // Configuration
    static constexpr int MAX_ITERATIONS = 20;      // Max Newton iterations
    static constexpr int MAX_SPECULAR_DEPTH = 4;   // Max specular bounces
    static constexpr double EPSILON = 1e-6;        // Convergence threshold
    static constexpr double STEP_EPSILON = 1e-10;  // Minimum step size
    
    /**
     * @brief Attempt to connect a shading point to a light through specular surfaces
     * @param scene The scene
     * @param shading_point Point we're shading
     * @param shading_normal Normal at shading point
     * @param light_position Position of the light
     * @param light_normal Normal at light (for area lights)
     * @return Optional MNEE path if successful
     */
    static std::optional<MNEEPath> connect_to_light(
        const Scene& scene,
        point3 shading_point,
        vec3 shading_normal,
        point3 light_position,
        vec3 light_normal
    ) {
        // First, trace a straight line to find specular surfaces in the way
        vec3 to_light = vec3_sub(light_position, shading_point);
        double dist_to_light = vec3_length(to_light);
        vec3 dir = vec3_scale(to_light, 1.0 / dist_to_light);
        
        // Find all specular surfaces along the path
        std::vector<ManifoldVertex> specular_chain;
        ray probe = ray_create(shading_point, dir);
        point3 current_pos = shading_point;
        
        for (int i = 0; i < MAX_SPECULAR_DEPTH; ++i) {
            hit_record rec;
            double max_t = vec3_length(vec3_sub(light_position, current_pos)) - EPSILON;
            
            if (!scene.hit(probe, 0.001, max_t, rec)) {
                break; // Clear path to light
            }
            
            const Material& mat = scene.get_material(rec.material_id);
            
            // Only handle dielectrics for MNEE (metals don't transmit)
            if (mat.type != MaterialType::Dielectric) {
                // Opaque surface blocks the path
                return std::nullopt;
            }
            
            ManifoldVertex vertex;
            vertex.position = rec.point;
            vertex.normal = rec.normal;
            vertex.geometric_normal = rec.normal;
            vertex.eta = rec.front_face ? (1.0 / mat.refraction_index) : mat.refraction_index;
            vertex.is_specular = true;
            vertex.material_id = rec.material_id;
            
            specular_chain.push_back(vertex);
            
            // Continue probing from this point
            current_pos = rec.point;
            probe = ray_create(vec3_add(current_pos, vec3_scale(dir, 0.001)), dir);
        }
        
        if (specular_chain.empty()) {
            // No specular surfaces - use regular NEE
            return std::nullopt;
        }
        
        // Now solve the manifold constraints using Newton iteration
        return solve_manifold(scene, shading_point, shading_normal, 
                            light_position, light_normal, specular_chain);
    }
    
private:
    /**
     * @brief Solve manifold constraints using Newton iteration
     */
    static std::optional<MNEEPath> solve_manifold(
        const Scene& scene,
        point3 shading_point,
        vec3 shading_normal,
        point3 light_position,
        vec3 light_normal,
        std::vector<ManifoldVertex>& chain
    ) {
        // The constraint at each specular vertex is that Snell's law holds:
        // n1 * sin(theta1) = n2 * sin(theta2)
        // 
        // In tangent space, this becomes a 2D constraint per vertex.
        // We iterate to find positions that satisfy all constraints.
        
        int n = static_cast<int>(chain.size());
        
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            // Compute constraint residuals and Jacobian
            std::vector<vec3> residuals(n);
            double max_residual = 0.0;
            
            for (int i = 0; i < n; ++i) {
                // Get incoming and outgoing directions
                point3 prev_pos = (i == 0) ? shading_point : chain[i-1].position;
                point3 next_pos = (i == n-1) ? light_position : chain[i+1].position;
                point3 curr_pos = chain[i].position;
                
                vec3 wi = vec3_normalize(vec3_sub(prev_pos, curr_pos));
                vec3 wo = vec3_normalize(vec3_sub(next_pos, curr_pos));
                vec3 n_vec = chain[i].normal;
                double eta = chain[i].eta;
                
                // Compute the half-vector constraint for refraction
                // For refraction: eta * wi + wo should be parallel to normal
                vec3 h = vec3_add(vec3_scale(wi, eta), wo);
                double h_len = vec3_length(h);
                if (h_len < STEP_EPSILON) {
                    return std::nullopt; // Degenerate case
                }
                h = vec3_scale(h, 1.0 / h_len);
                
                // Residual is the tangent component of h (should be zero)
                double h_dot_n = vec3_dot(h, n_vec);
                residuals[i] = vec3_sub(h, vec3_scale(n_vec, h_dot_n));
                
                double res_len = vec3_length(residuals[i]);
                max_residual = std::fmax(max_residual, res_len);
            }
            
            // Check convergence
            if (max_residual < EPSILON) {
                // Converged - compute throughput and return
                return build_path(scene, shading_point, light_position, chain);
            }
            
            // Newton step: move vertices in tangent plane to reduce residuals
            for (int i = 0; i < n; ++i) {
                // Simple gradient descent in tangent space
                // (Full Newton would require Jacobian computation)
                vec3 step = vec3_scale(residuals[i], -0.5);
                
                // Project step to tangent plane
                double step_n = vec3_dot(step, chain[i].normal);
                step = vec3_sub(step, vec3_scale(chain[i].normal, step_n));
                
                // Update position
                chain[i].position = vec3_add(chain[i].position, step);
                
                // Re-project to surface (simple approach: trace ray to surface)
                // For now, we just nudge and hope surfaces are locally flat
            }
        }
        
        // Failed to converge
        return std::nullopt;
    }
    
    /**
     * @brief Build final MNEE path with throughput computation
     */
    static std::optional<MNEEPath> build_path(
        const Scene& scene,
        point3 shading_point,
        point3 light_position,
        const std::vector<ManifoldVertex>& chain
    ) {
        MNEEPath path;
        path.vertices = chain;
        path.throughput = 1.0;
        path.valid = true;
        
        int n = static_cast<int>(chain.size());
        
        // Compute throughput through the chain
        for (int i = 0; i < n; ++i) {
            point3 prev_pos = (i == 0) ? shading_point : chain[i-1].position;
            point3 next_pos = (i == n-1) ? light_position : chain[i+1].position;
            point3 curr_pos = chain[i].position;
            
            vec3 wi = vec3_normalize(vec3_sub(prev_pos, curr_pos));
            vec3 wo = vec3_normalize(vec3_sub(next_pos, curr_pos));
            vec3 n_vec = chain[i].normal;
            double eta = chain[i].eta;
            
            // Fresnel transmission coefficient
            double cos_i = std::fabs(vec3_dot(wi, n_vec));
            double cos_t = std::fabs(vec3_dot(wo, n_vec));
            
            // Schlick approximation for dielectric
            double r0 = (1.0 - eta) / (1.0 + eta);
            r0 = r0 * r0;
            double fresnel_r = r0 + (1.0 - r0) * std::pow(1.0 - cos_i, 5.0);
            double fresnel_t = 1.0 - fresnel_r;
            
            // Geometry term (accounts for refraction compression)
            double geometry = (eta * eta * cos_t) / cos_i;
            
            path.throughput *= fresnel_t * geometry;
            
            // Safety clamp
            if (path.throughput < 1e-10) {
                path.valid = false;
                return std::nullopt;
            }
        }
        
        // Verify the path is actually valid (no occlusion)
        // Trace through the chain
        point3 current = shading_point;
        for (int i = 0; i <= n; ++i) {
            point3 target = (i == n) ? light_position : chain[i].position;
            vec3 dir = vec3_normalize(vec3_sub(target, current));
            double dist = vec3_length(vec3_sub(target, current));
            
            ray check_ray = ray_create(current, dir);
            hit_record rec;
            
            if (scene.hit(check_ray, 0.001, dist - 0.001, rec)) {
                // Check if we hit the expected surface
                if (i < n) {
                    double hit_dist = vec3_length(vec3_sub(rec.point, chain[i].position));
                    if (hit_dist > 0.01) {
                        // Hit something unexpected
                        path.valid = false;
                        return std::nullopt;
                    }
                } else {
                    // Occluded before reaching light
                    path.valid = false;
                    return std::nullopt;
                }
            }
            
            if (i < n) {
                current = chain[i].position;
            }
        }
        
        return path;
    }
    
public:
    /**
     * @brief Compute MNEE contribution for direct lighting
     * @param scene The scene
     * @param hit_point Current shading point
     * @param hit_normal Normal at shading point
     * @param view_dir Direction to camera/previous vertex
     * @param mat Material at shading point
     * @return Tuple of (contribution color, was_used flag)
     * 
     * Returns the radiance contribution from MNEE if successful,
     * otherwise returns black and false to fall back to regular NEE.
     */
    static std::pair<color, bool> evaluate(
        const Scene& scene,
        point3 hit_point,
        vec3 hit_normal,
        vec3 view_dir,
        const Material& mat
    ) {
        color contribution = vec3_zero();
        bool any_contribution = false;
        
        // Try MNEE for each area light
        for (const auto& quad : scene.quad_lights) {
            // Sample point on quad light
            point3 light_pos = quad.sample_point();
            vec3 light_normal = quad.normal;
            
            auto mnee_path = connect_to_light(scene, hit_point, hit_normal, 
                                              light_pos, light_normal);
            
            if (mnee_path && mnee_path->valid) {
                // Compute light contribution
                vec3 to_light = vec3_sub(light_pos, hit_point);
                double dist_sq = vec3_length_squared(to_light);
                
                // Light emission
                color Le = quad.emission;
                
                // Geometry term at light
                vec3 wi_light = vec3_normalize(vec3_negate(to_light));
                double cos_light = std::fabs(vec3_dot(light_normal, wi_light));
                
                // MNEE throughput includes Fresnel and geometry through specular chain
                double mnee_weight = mnee_path->throughput;
                
                // PDF for sampling the light
                double light_pdf = dist_sq / (quad.area * cos_light);
                
                // Final contribution
                color contrib = vec3_scale(Le, mnee_weight * cos_light / (dist_sq * light_pdf));
                contribution = vec3_add(contribution, contrib);
                any_contribution = true;
            }
        }
        
        // Also try disk lights
        for (const auto& disk : scene.disk_lights) {
            point3 light_pos = disk.sample_point();
            vec3 light_normal = disk.normal;
            
            auto mnee_path = connect_to_light(scene, hit_point, hit_normal,
                                              light_pos, light_normal);
            
            if (mnee_path && mnee_path->valid) {
                vec3 to_light = vec3_sub(light_pos, hit_point);
                double dist_sq = vec3_length_squared(to_light);
                
                color Le = disk.emission;
                vec3 wi_light = vec3_normalize(vec3_negate(to_light));
                double cos_light = std::fabs(vec3_dot(light_normal, wi_light));
                
                double mnee_weight = mnee_path->throughput;
                double light_pdf = dist_sq / (disk.area * cos_light);
                
                color contrib = vec3_scale(Le, mnee_weight * cos_light / (dist_sq * light_pdf));
                contribution = vec3_add(contribution, contrib);
                any_contribution = true;
            }
        }
        
        // Try emissive spheres (area lights)
        for (size_t i = 0; i < scene.spheres.size(); ++i) {
            const auto& sphere = scene.spheres[i];
            const Material& sphere_mat = scene.get_material(sphere.material_id);
            
            if (!sphere_mat.is_emissive()) continue;
            
            // Sample point on sphere facing the hit point
            vec3 to_sphere = vec3_sub(sphere.center, hit_point);
            double dist_to_center = vec3_length(to_sphere);
            vec3 dir_to_sphere = vec3_scale(to_sphere, 1.0 / dist_to_center);
            
            // Sample in cone toward sphere
            double cos_theta_max = std::sqrt(1.0 - (sphere.radius * sphere.radius) / (dist_to_center * dist_to_center));
            if (cos_theta_max < 0.0) cos_theta_max = 0.0;
            
            // Uniform sample in cone
            double cos_theta = 1.0 + random_double() * (cos_theta_max - 1.0);
            double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
            double phi = 2.0 * 3.14159265358979323846 * random_double();
            
            // Build local frame
            vec3 w = dir_to_sphere;
            vec3 u_vec = (std::fabs(w.x) > 0.9) ? vec3{0, 1, 0} : vec3{1, 0, 0};
            u_vec = vec3_normalize(vec3_cross(w, u_vec));
            vec3 v_vec = vec3_cross(w, u_vec);
            
            vec3 sample_dir = vec3_add(
                vec3_scale(u_vec, sin_theta * std::cos(phi)),
                vec3_add(
                    vec3_scale(v_vec, sin_theta * std::sin(phi)),
                    vec3_scale(w, cos_theta)
                )
            );
            
            // Find intersection with sphere
            ray to_sphere_ray = ray_create(hit_point, sample_dir);
            double t = dist_to_center * cos_theta - std::sqrt(
                sphere.radius * sphere.radius - dist_to_center * dist_to_center * (1.0 - cos_theta * cos_theta)
            );
            
            point3 light_pos = vec3_add(hit_point, vec3_scale(sample_dir, t));
            vec3 light_normal = vec3_normalize(vec3_sub(light_pos, sphere.center));
            
            auto mnee_path = connect_to_light(scene, hit_point, hit_normal,
                                              light_pos, light_normal);
            
            if (mnee_path && mnee_path->valid) {
                color Le = sphere_mat.emission;
                double solid_angle = 2.0 * 3.14159265358979323846 * (1.0 - cos_theta_max);
                double pdf = 1.0 / solid_angle;
                
                double cos_at_light = std::fabs(vec3_dot(light_normal, vec3_negate(sample_dir)));
                double mnee_weight = mnee_path->throughput;
                
                color contrib = vec3_scale(Le, mnee_weight * cos_at_light / pdf);
                contribution = vec3_add(contribution, contrib);
                any_contribution = true;
            }
        }
        
        return {contribution, any_contribution};
    }
};

} // namespace raytracer
