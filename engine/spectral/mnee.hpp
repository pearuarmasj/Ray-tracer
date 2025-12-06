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
        // Strategy 1: Straight-line probe (original approach)
        auto result = connect_to_light_straight(scene, shading_point, shading_normal, 
                                                 light_position, light_normal);
        if (result) return result;
        
        // Strategy 2: Seed path through nearby dielectric spheres
        return connect_to_light_seeded(scene, shading_point, shading_normal,
                                        light_position, light_normal);
    }
    
private:
    /**
     * @brief Original straight-line probe approach
     */
    static std::optional<MNEEPath> connect_to_light_straight(
        const Scene& scene,
        point3 shading_point,
        vec3 shading_normal,
        point3 light_position,
        vec3 light_normal
    ) {
        vec3 to_light = vec3_sub(light_position, shading_point);
        double dist_to_light = vec3_length(to_light);
        vec3 dir = vec3_scale(to_light, 1.0 / dist_to_light);
        
        std::vector<ManifoldVertex> specular_chain;
        ray probe = ray_create(shading_point, dir);
        point3 current_pos = shading_point;
        
        for (int i = 0; i < MAX_SPECULAR_DEPTH; ++i) {
            hit_record rec;
            double max_t = vec3_length(vec3_sub(light_position, current_pos)) - EPSILON;
            
            if (!scene.hit(probe, 0.001, max_t, rec)) {
                break;
            }
            
            const Material& mat = scene.get_material(rec.material_id);
            
            if (mat.type != MaterialType::Dielectric) {
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
            
            current_pos = rec.point;
            probe = ray_create(vec3_add(current_pos, vec3_scale(dir, 0.001)), dir);
        }
        
        if (specular_chain.empty()) {
            return std::nullopt;
        }
        
        return solve_manifold(scene, shading_point, shading_normal, 
                            light_position, light_normal, specular_chain);
    }
    
    /**
     * @brief Seed path generation - find nearby dielectric spheres and build paths through them
     * 
     * For caustics, the straight-line probe often misses because the actual light path
     * refracts at a different angle. This method explicitly samples dielectric spheres
     * and creates seed vertices for the manifold solver.
     */
    static std::optional<MNEEPath> connect_to_light_seeded(
        const Scene& scene,
        point3 shading_point,
        vec3 shading_normal,
        point3 light_position,
        vec3 light_normal
    ) {
        // Find dielectric spheres that could create caustics
        for (const auto& sphere : scene.spheres) {
            const Material& mat = scene.get_material(sphere.material_id);
            if (mat.type != MaterialType::Dielectric) continue;
            
            // Check if sphere is between shading point and light (roughly)
            vec3 to_sphere = vec3_sub(sphere.center, shading_point);
            vec3 to_light = vec3_sub(light_position, shading_point);
            double dist_to_sphere = vec3_length(to_sphere);
            double dist_to_light = vec3_length(to_light);
            
            // Skip if sphere is behind us or beyond the light
            if (dist_to_sphere > dist_to_light + sphere.radius) continue;
            if (dist_to_sphere < sphere.radius) continue;  // We're inside the sphere
            
            // Check if sphere is roughly in the direction of the light
            vec3 dir_to_sphere = vec3_scale(to_sphere, 1.0 / dist_to_sphere);
            vec3 dir_to_light = vec3_scale(to_light, 1.0 / dist_to_light);
            double alignment = vec3_dot(dir_to_sphere, dir_to_light);
            
            // Allow fairly wide cone (caustics can come from oblique angles)
            if (alignment < 0.0) continue;
            
            // Try to build a seed path through this sphere
            auto result = try_sphere_seed_path(scene, sphere, mat, shading_point, 
                                                shading_normal, light_position, light_normal);
            if (result) return result;
        }
        
        return std::nullopt;
    }
    
    /**
     * @brief Try to build an MNEE path through a specific dielectric sphere
     */
    static std::optional<MNEEPath> try_sphere_seed_path(
        const Scene& scene,
        const Sphere& sphere,
        const Material& mat,
        point3 shading_point,
        vec3 shading_normal,
        point3 light_position,
        vec3 light_normal
    ) {
        // Compute entry and exit points on the sphere
        // Entry: point on sphere closest to a line from shading_point toward light
        // Exit: point on sphere closest to a line from light toward shading_point
        
        vec3 to_light = vec3_normalize(vec3_sub(light_position, shading_point));
        vec3 to_shade = vec3_normalize(vec3_sub(shading_point, light_position));
        
        // Find where a ray from shading point toward light would hit the sphere
        // (or the closest point on sphere surface if it misses)
        point3 entry_point, exit_point;
        vec3 entry_normal, exit_normal;
        
        // Ray-sphere intersection for entry
        ray entry_ray = ray_create(shading_point, to_light);
        hit_record entry_rec;
        if (!sphere.hit(entry_ray, 0.001, 1e9, entry_rec)) {
            // Ray misses - find closest point on sphere to ray
            vec3 to_center = vec3_sub(sphere.center, shading_point);
            double proj = vec3_dot(to_center, to_light);
            if (proj < 0) return std::nullopt;  // Sphere is behind
            
            point3 closest_on_ray = vec3_add(shading_point, vec3_scale(to_light, proj));
            vec3 to_closest = vec3_sub(sphere.center, closest_on_ray);
            double dist_to_axis = vec3_length(to_closest);
            
            if (dist_to_axis >= sphere.radius) {
                // Project to sphere surface
                vec3 dir_to_surface = vec3_normalize(vec3_sub(closest_on_ray, sphere.center));
                entry_point = vec3_add(sphere.center, vec3_scale(dir_to_surface, sphere.radius));
            } else {
                // Ray would hit, use the intersection
                double offset = std::sqrt(sphere.radius * sphere.radius - dist_to_axis * dist_to_axis);
                entry_point = vec3_add(shading_point, vec3_scale(to_light, proj - offset));
            }
            entry_normal = vec3_normalize(vec3_sub(entry_point, sphere.center));
        } else {
            entry_point = entry_rec.point;
            entry_normal = entry_rec.normal;
        }
        
        // Ray-sphere intersection for exit (from light toward shading point)
        ray exit_ray = ray_create(light_position, to_shade);
        hit_record exit_rec;
        if (!sphere.hit(exit_ray, 0.001, 1e9, exit_rec)) {
            // Similar fallback
            vec3 to_center = vec3_sub(sphere.center, light_position);
            double proj = vec3_dot(to_center, to_shade);
            if (proj < 0) return std::nullopt;
            
            point3 closest_on_ray = vec3_add(light_position, vec3_scale(to_shade, proj));
            vec3 to_closest = vec3_sub(sphere.center, closest_on_ray);
            double dist_to_axis = vec3_length(to_closest);
            
            if (dist_to_axis >= sphere.radius) {
                vec3 dir_to_surface = vec3_normalize(vec3_sub(closest_on_ray, sphere.center));
                exit_point = vec3_add(sphere.center, vec3_scale(dir_to_surface, sphere.radius));
            } else {
                double offset = std::sqrt(sphere.radius * sphere.radius - dist_to_axis * dist_to_axis);
                exit_point = vec3_add(light_position, vec3_scale(to_shade, proj - offset));
            }
            exit_normal = vec3_normalize(vec3_sub(exit_point, sphere.center));
        } else {
            exit_point = exit_rec.point;
            exit_normal = exit_rec.normal;
        }
        
        // Build seed chain with entry and exit vertices
        std::vector<ManifoldVertex> specular_chain;
        
        // Entry vertex (outside to inside)
        ManifoldVertex entry_vertex;
        entry_vertex.position = entry_point;
        entry_vertex.normal = entry_normal;
        entry_vertex.geometric_normal = entry_normal;
        entry_vertex.eta = 1.0 / mat.refraction_index;  // Air to glass
        entry_vertex.is_specular = true;
        entry_vertex.material_id = sphere.material_id;
        specular_chain.push_back(entry_vertex);
        
        // Exit vertex (inside to outside)
        ManifoldVertex exit_vertex;
        exit_vertex.position = exit_point;
        exit_vertex.normal = vec3_negate(exit_normal);  // Inward-facing normal for exit
        exit_vertex.geometric_normal = exit_normal;
        exit_vertex.eta = mat.refraction_index;  // Glass to air
        exit_vertex.is_specular = true;
        exit_vertex.material_id = sphere.material_id;
        specular_chain.push_back(exit_vertex);
        
        // Solve manifold constraints
        return solve_manifold(scene, shading_point, shading_normal,
                            light_position, light_normal, specular_chain);
    }
    /**
     * @brief Build tangent-space basis vectors for a surface point
     */
    static void build_tangent_frame(const vec3& normal, vec3& tangent, vec3& bitangent) {
        // Choose a vector not parallel to normal
        vec3 up = (std::fabs(normal.y) < 0.9) ? vec3{0, 1, 0} : vec3{1, 0, 0};
        tangent = vec3_normalize(vec3_cross(up, normal));
        bitangent = vec3_cross(normal, tangent);
    }
    
    /**
     * @brief Project vector to tangent plane, return 2D coordinates
     */
    static std::pair<double, double> to_tangent_space(const vec3& v, const vec3& tangent, const vec3& bitangent) {
        return {vec3_dot(v, tangent), vec3_dot(v, bitangent)};
    }
    
    /**
     * @brief Convert 2D tangent coordinates to 3D vector
     */
    static vec3 from_tangent_space(double u, double v, const vec3& tangent, const vec3& bitangent) {
        return vec3_add(vec3_scale(tangent, u), vec3_scale(bitangent, v));
    }
    
    /**
     * @brief Compute the refraction constraint at a vertex
     * 
     * The constraint is that eta * wi_t + wo_t = 0, where wi_t and wo_t are
     * the tangential components of the incoming and outgoing directions.
     * 
     * Returns the 2D constraint value in tangent space.
     */
    static std::pair<double, double> compute_constraint(
        const point3& prev_pos,
        const point3& curr_pos, 
        const point3& next_pos,
        const vec3& normal,
        double eta,
        const vec3& tangent,
        const vec3& bitangent
    ) {
        vec3 wi = vec3_normalize(vec3_sub(prev_pos, curr_pos));
        vec3 wo = vec3_normalize(vec3_sub(next_pos, curr_pos));
        
        // Tangential components
        auto [wi_u, wi_v] = to_tangent_space(wi, tangent, bitangent);
        auto [wo_u, wo_v] = to_tangent_space(wo, tangent, bitangent);
        
        // Snell's law constraint: eta * sin(theta_i) = sin(theta_o)
        // In tangent space: eta * wi_tangent + wo_tangent should be zero
        return {eta * wi_u + wo_u, eta * wi_v + wo_v};
    }
    
    /**
     * @brief Compute Jacobian of constraint w.r.t. vertex position (2x2 for single vertex)
     * 
     * Uses finite differences for robustness on curved surfaces.
     */
    static void compute_jacobian_block(
        const point3& prev_pos,
        const point3& curr_pos,
        const point3& next_pos,
        const vec3& normal,
        double eta,
        const vec3& tangent,
        const vec3& bitangent,
        double J[2][2]
    ) {
        constexpr double h = 1e-5;  // Finite difference step
        
        // Base constraint
        auto [c0_u, c0_v] = compute_constraint(prev_pos, curr_pos, next_pos, normal, eta, tangent, bitangent);
        
        // Perturb in tangent direction
        point3 curr_u = vec3_add(curr_pos, vec3_scale(tangent, h));
        auto [cu_u, cu_v] = compute_constraint(prev_pos, curr_u, next_pos, normal, eta, tangent, bitangent);
        
        // Perturb in bitangent direction
        point3 curr_v = vec3_add(curr_pos, vec3_scale(bitangent, h));
        auto [cv_u, cv_v] = compute_constraint(prev_pos, curr_v, next_pos, normal, eta, tangent, bitangent);
        
        // Jacobian: dc/dx
        J[0][0] = (cu_u - c0_u) / h;  // dc_u/dx_u
        J[0][1] = (cv_u - c0_u) / h;  // dc_u/dx_v
        J[1][0] = (cu_v - c0_v) / h;  // dc_v/dx_u
        J[1][1] = (cv_v - c0_v) / h;  // dc_v/dx_v
    }
    
    /**
     * @brief Solve 2x2 linear system Jx = b
     */
    static bool solve_2x2(const double J[2][2], const double b[2], double x[2]) {
        double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        if (std::fabs(det) < STEP_EPSILON) {
            return false;  // Singular
        }
        double inv_det = 1.0 / det;
        x[0] = inv_det * (J[1][1] * b[0] - J[0][1] * b[1]);
        x[1] = inv_det * (-J[1][0] * b[0] + J[0][0] * b[1]);
        return true;
    }
    
    /**
     * @brief Re-project a point back onto the nearest surface
     */
    static bool reproject_to_surface(
        const Scene& scene,
        point3& position,
        vec3& normal,
        int expected_material_id
    ) {
        // Cast rays in normal direction (both ways) to find surface
        vec3 search_dirs[2] = {normal, vec3_negate(normal)};
        
        for (int d = 0; d < 2; ++d) {
            ray search_ray = ray_create(position, search_dirs[d]);
            hit_record rec;
            
            if (scene.hit(search_ray, 0.0, 0.1, rec)) {  // Search within 0.1 units
                if (rec.material_id == expected_material_id) {
                    position = rec.point;
                    normal = rec.normal;
                    return true;
                }
            }
        }
        
        // Try from slightly offset position
        point3 offset_pos = vec3_add(position, vec3_scale(normal, 0.05));
        ray down_ray = ray_create(offset_pos, vec3_negate(normal));
        hit_record rec;
        
        if (scene.hit(down_ray, 0.0, 0.2, rec)) {
            if (rec.material_id == expected_material_id) {
                position = rec.point;
                normal = rec.normal;
                return true;
            }
        }
        
        return false;  // Failed to find surface
    }
    
    /**
     * @brief Solve manifold constraints using proper Newton iteration
     */
    static std::optional<MNEEPath> solve_manifold(
        const Scene& scene,
        point3 shading_point,
        vec3 shading_normal,
        point3 light_position,
        vec3 light_normal,
        std::vector<ManifoldVertex>& chain
    ) {
        int n = static_cast<int>(chain.size());
        
        // Build tangent frames for each vertex
        std::vector<vec3> tangents(n), bitangents(n);
        for (int i = 0; i < n; ++i) {
            build_tangent_frame(chain[i].normal, tangents[i], bitangents[i]);
        }
        
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            // Compute constraints and check convergence
            double max_constraint = 0.0;
            std::vector<std::pair<double, double>> constraints(n);
            
            for (int i = 0; i < n; ++i) {
                point3 prev_pos = (i == 0) ? shading_point : chain[i-1].position;
                point3 next_pos = (i == n-1) ? light_position : chain[i+1].position;
                
                constraints[i] = compute_constraint(
                    prev_pos, chain[i].position, next_pos,
                    chain[i].normal, chain[i].eta,
                    tangents[i], bitangents[i]
                );
                
                double c_mag = std::sqrt(constraints[i].first * constraints[i].first + 
                                        constraints[i].second * constraints[i].second);
                max_constraint = std::fmax(max_constraint, c_mag);
            }
            
            // Check convergence
            if (max_constraint < EPSILON) {
                return build_path(scene, shading_point, light_position, chain);
            }
            
            // Newton step for each vertex
            // For simplicity, we use a Gauss-Seidel approach (update vertices sequentially)
            // Full Newton would solve the coupled system, but this is more stable
            for (int i = 0; i < n; ++i) {
                point3 prev_pos = (i == 0) ? shading_point : chain[i-1].position;
                point3 next_pos = (i == n-1) ? light_position : chain[i+1].position;
                
                // Compute local Jacobian
                double J[2][2];
                compute_jacobian_block(
                    prev_pos, chain[i].position, next_pos,
                    chain[i].normal, chain[i].eta,
                    tangents[i], bitangents[i], J
                );
                
                // Solve for step: J * step = -constraint
                double b[2] = {-constraints[i].first, -constraints[i].second};
                double step[2];
                
                if (!solve_2x2(J, b, step)) {
                    // Jacobian singular, fall back to gradient descent
                    step[0] = -0.5 * constraints[i].first;
                    step[1] = -0.5 * constraints[i].second;
                }
                
                // Limit step size for stability
                double step_mag = std::sqrt(step[0] * step[0] + step[1] * step[1]);
                constexpr double MAX_STEP = 0.1;
                if (step_mag > MAX_STEP) {
                    double scale = MAX_STEP / step_mag;
                    step[0] *= scale;
                    step[1] *= scale;
                }
                
                // Apply step in tangent plane
                vec3 delta = from_tangent_space(step[0], step[1], tangents[i], bitangents[i]);
                chain[i].position = vec3_add(chain[i].position, delta);
                
                // Re-project to surface
                if (!reproject_to_surface(scene, chain[i].position, chain[i].normal, chain[i].material_id)) {
                    // Failed to find surface - try to continue anyway
                }
                
                // Update tangent frame after normal may have changed
                build_tangent_frame(chain[i].normal, tangents[i], bitangents[i]);
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
