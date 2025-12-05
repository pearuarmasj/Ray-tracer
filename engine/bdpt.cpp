/**
 * @file bdpt.cpp
 * @brief Bidirectional Path Tracing implementation
 */

#include "bdpt.hpp"
#include <algorithm>
#include <cmath>

namespace raytracer {

// Use the PI defined in sphere.hpp and INV_PI from material.hpp
// Define only BDPT_EPSILON locally
constexpr double BDPT_EPSILON = 1e-6;

// ============================================================================
// Helper functions
// ============================================================================

/**
 * @brief Power heuristic for MIS (beta = 2)
 */
inline double power_heuristic(double pdf_a, double pdf_b) {
    double a2 = pdf_a * pdf_a;
    double b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + BDPT_EPSILON);
}

/**
 * @brief Cosine-weighted hemisphere PDF
 */
inline double cosine_hemisphere_pdf(double cos_theta) {
    return cos_theta * INV_PI;
}

/**
 * @brief Sample cosine-weighted hemisphere direction
 */
inline vec3 sample_cosine_hemisphere(vec3 normal) {
    // Generate random point on disk
    double r1 = random_double();
    double r2 = random_double();
    
    double phi = 2.0 * PI * r1;
    double cos_theta = std::sqrt(r2);
    double sin_theta = std::sqrt(1.0 - r2);
    
    // Local coordinates
    double x = std::cos(phi) * sin_theta;
    double y = std::sin(phi) * sin_theta;
    double z = cos_theta;
    
    // Build orthonormal basis around normal
    vec3 w = normal;
    vec3 a = (std::fabs(w.x) > 0.9) ? vec3{0, 1, 0} : vec3{1, 0, 0};
    vec3 u = vec3_normalize(vec3_cross(a, w));
    vec3 v = vec3_cross(w, u);
    
    // Transform to world space
    return vec3_normalize({
        u.x * x + v.x * y + w.x * z,
        u.y * x + v.y * y + w.y * z,
        u.z * x + v.z * y + w.z * z
    });
}

// ============================================================================
// BDPTIntegrator implementation
// ============================================================================

color BDPTIntegrator::Li(ray initial_ray, const Scene& scene) const {
    // Trace subpaths
    std::vector<PathVertex> eye_path;
    std::vector<PathVertex> light_path;
    
    eye_path.reserve(settings_.max_eye_depth + 1);
    light_path.reserve(settings_.max_light_depth + 1);
    
    int num_eye = trace_eye_path(initial_ray, scene, eye_path);
    int num_light = trace_light_path(scene, light_path);
    
    color L = {0, 0, 0};
    
    // Connect paths with all valid (s, t) strategies
    // s = number of light vertices, t = number of eye vertices
    // Total path length = s + t
    for (int t = 1; t <= num_eye; ++t) {
        for (int s = 0; s <= num_light; ++s) {
            // Skip invalid strategies
            int path_length = s + t;
            if (path_length < 2) continue;  // Need at least camera + light
            
            // Debug mode: only use specific strategy
            if (settings_.debug_strategy) {
                if (settings_.debug_s >= 0 && s != settings_.debug_s) continue;
                if (settings_.debug_t >= 0 && t != settings_.debug_t) continue;
            }
            
            // Skip strategies involving delta distributions at connection
            // (t=1 connects to camera, always valid)
            // (s=0 is direct light hit, no connection needed)
            if (t > 1 && !eye_path[t-1].is_connectible()) continue;
            if (s > 0 && !light_path[s-1].is_connectible()) continue;
            
            // Compute contribution from this strategy
            color Lpath = connect_paths(eye_path, light_path, s, t, scene);
            
            // MIS weight
            double mis_w = settings_.use_mis ? 
                           mis_weight(eye_path, light_path, s, t) : 1.0;
            
            color weighted = vec3_scale(Lpath, mis_w);
            
            // Firefly clamping - reduce bright outliers
            if (settings_.clamp_max > 0.0) {
                double luminance = 0.2126 * weighted.x + 0.7152 * weighted.y + 0.0722 * weighted.z;
                if (luminance > settings_.clamp_max) {
                    double scale = settings_.clamp_max / luminance;
                    weighted = vec3_scale(weighted, scale);
                }
            }
            
            L = vec3_add(L, weighted);
        }
    }
    
    return L;
}

int BDPTIntegrator::trace_eye_path(ray r, const Scene& scene,
                                    std::vector<PathVertex>& path) const {
    path.clear();
    
    // Camera vertex (t=0)
    PathVertex cam_vertex;
    cam_vertex.type = VertexType::Camera;
    cam_vertex.position = r.origin;
    cam_vertex.throughput = {1, 1, 1};
    cam_vertex.pdf_fwd = 1.0;  // Delta from camera
    path.push_back(cam_vertex);
    
    color throughput = {1, 1, 1};
    ray current_ray = r;
    
    for (int bounce = 0; bounce < settings_.max_eye_depth; ++bounce) {
        hit_record rec;
        
        if (!scene.hit(current_ray, 0.001, 1e9, rec)) {
            // Could add environment map vertex here
            break;
        }
        
        // Check for area light hit
        if (scene.is_area_light(rec.material_id)) {
            // Create light vertex at hit point
            PathVertex light_v;
            light_v.type = VertexType::Light;
            light_v.position = rec.point;
            light_v.normal = rec.normal;
            light_v.geo_normal = rec.normal;
            light_v.emission = scene.get_area_light_emission(rec.material_id);
            light_v.throughput = throughput;
            light_v.wi = vec3_negate(vec3_normalize(current_ray.direction));
            path.push_back(light_v);
            break;
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        
        // Create surface vertex
        PathVertex vertex;
        vertex.type = VertexType::Surface;
        vertex.position = rec.point;
        vertex.normal = rec.normal;
        vertex.geo_normal = rec.normal;
        vertex.material_id = rec.material_id;
        vertex.mat_type = mat.type;
        vertex.albedo = mat.get_albedo(rec.point, rec.u, rec.v);
        vertex.emission = mat.emission;
        vertex.roughness = mat.fuzz;
        vertex.refraction_index = mat.refraction_index;
        vertex.u = rec.u;
        vertex.v = rec.v;
        vertex.throughput = throughput;
        vertex.wi = vec3_negate(vec3_normalize(current_ray.direction));
        
        path.push_back(vertex);
        
        // Sample BSDF for next direction
        vec3 wo;
        double pdf;
        color bsdf_val;
        
        if (!sample_bsdf(vertex, vertex.wi, wo, pdf, bsdf_val)) {
            break;  // Absorbed
        }
        
        if (pdf < BDPT_EPSILON) break;
        
        // Update throughput
        double cos_theta = std::abs(vec3_dot(vertex.normal, wo));
        throughput = vec3_mul(throughput, vec3_scale(bsdf_val, cos_theta / pdf));
        
        // Russian roulette after a few bounces
        if (bounce > 3) {
            double q = std::max(0.05, 1.0 - std::max({throughput.x, throughput.y, throughput.z}));
            if (random_double() < q) break;
            throughput = vec3_scale(throughput, 1.0 / (1.0 - q));
        }
        
        // Store outgoing direction in vertex
        path.back().wo = wo;
        path.back().pdf_fwd = pdf;
        
        // Next ray
        current_ray = ray_create(rec.point, wo);
    }
    
    return static_cast<int>(path.size());
}

int BDPTIntegrator::trace_light_path(const Scene& scene,
                                      std::vector<PathVertex>& path) const {
    path.clear();
    
    // Sample light origin
    PathVertex light_vertex;
    vec3 ray_dir;
    double pdf_pos, pdf_dir;
    
    if (!sample_light_origin(scene, light_vertex, ray_dir, pdf_pos, pdf_dir)) {
        return 0;
    }
    
    light_vertex.pdf_fwd = pdf_pos;
    path.push_back(light_vertex);
    
    if (pdf_dir < BDPT_EPSILON) {
        return 1;  // Point light, no direction sampling
    }
    
    // Initial throughput from light
    double cos_theta = std::abs(vec3_dot(light_vertex.normal, ray_dir));
    color throughput = vec3_scale(light_vertex.emission, cos_theta / (pdf_pos * pdf_dir));
    
    ray current_ray = ray_create(light_vertex.position, ray_dir);
    
    for (int bounce = 0; bounce < settings_.max_light_depth; ++bounce) {
        hit_record rec;
        
        if (!scene.hit(current_ray, 0.001, 1e9, rec)) {
            break;  // Escaped scene
        }
        
        // Skip area light hits from light path
        if (scene.is_area_light(rec.material_id)) {
            break;
        }
        
        const Material& mat = scene.get_material(rec.material_id);
        
        // Create surface vertex
        PathVertex vertex;
        vertex.type = VertexType::Surface;
        vertex.position = rec.point;
        vertex.normal = rec.normal;
        vertex.geo_normal = rec.normal;
        vertex.material_id = rec.material_id;
        vertex.mat_type = mat.type;
        vertex.albedo = mat.get_albedo(rec.point, rec.u, rec.v);
        vertex.roughness = mat.fuzz;
        vertex.refraction_index = mat.refraction_index;
        vertex.u = rec.u;
        vertex.v = rec.v;
        vertex.throughput = throughput;
        vertex.wi = vec3_negate(vec3_normalize(current_ray.direction));
        
        path.push_back(vertex);
        
        // Sample BSDF for next direction
        vec3 wo;
        double pdf;
        color bsdf_val;
        
        if (!sample_bsdf(vertex, vertex.wi, wo, pdf, bsdf_val)) {
            break;
        }
        
        if (pdf < BDPT_EPSILON) break;
        
        // Update throughput
        cos_theta = std::abs(vec3_dot(vertex.normal, wo));
        throughput = vec3_mul(throughput, vec3_scale(bsdf_val, cos_theta / pdf));
        
        // Russian roulette
        if (bounce > 3) {
            double q = std::max(0.05, 1.0 - std::max({throughput.x, throughput.y, throughput.z}));
            if (random_double() < q) break;
            throughput = vec3_scale(throughput, 1.0 / (1.0 - q));
        }
        
        path.back().wo = wo;
        path.back().pdf_fwd = pdf;
        
        current_ray = ray_create(rec.point, wo);
    }
    
    return static_cast<int>(path.size());
}

bool BDPTIntegrator::sample_light_origin(const Scene& scene, PathVertex& vertex,
                                          vec3& ray_dir, double& pdf_pos, 
                                          double& pdf_dir) const {
    // Count total lights
    size_t num_point = scene.lights.size();
    size_t num_sphere_lights = 0;
    for (const auto& s : scene.spheres) {
        if (vec3_length_squared(scene.get_material(s.material_id).emission) > 0.001) {
            num_sphere_lights++;
        }
    }
    size_t num_quad = scene.quad_lights.size();
    size_t num_disk = scene.disk_lights.size();
    size_t total_lights = num_point + num_sphere_lights + num_quad + num_disk;
    
    if (total_lights == 0) {
        return false;
    }
    
    double light_choice_pdf = 1.0 / total_lights;
    size_t light_idx = static_cast<size_t>(random_double() * total_lights);
    light_idx = std::min(light_idx, total_lights - 1);
    
    vertex.type = VertexType::Light;
    
    // Point lights
    if (light_idx < num_point) {
        const auto& light = scene.lights[light_idx];
        vertex.position = light.position;
        vertex.normal = {0, 1, 0};  // Point lights have no normal
        vertex.emission = light.intensity;
        vertex.light_pdf = light_choice_pdf;
        pdf_pos = light_choice_pdf;  // Delta position
        
        // Sample direction uniformly on sphere
        ray_dir = random_unit_vector();
        pdf_dir = 1.0 / (4.0 * PI);
        return true;
    }
    light_idx -= num_point;
    
    // Emissive spheres
    if (light_idx < num_sphere_lights) {
        size_t sphere_light_count = 0;
        for (const auto& s : scene.spheres) {
            const auto& mat = scene.get_material(s.material_id);
            if (vec3_length_squared(mat.emission) > 0.001) {
                if (sphere_light_count == light_idx) {
                    // Sample point on sphere
                    vec3 dir = random_unit_vector();
                    vertex.position = vec3_add(s.center, vec3_scale(dir, s.radius));
                    vertex.normal = dir;
                    vertex.emission = mat.emission;
                    vertex.light_pdf = light_choice_pdf;
                    
                    double area = 4.0 * PI * s.radius * s.radius;
                    pdf_pos = light_choice_pdf / area;
                    
                    // Sample cosine-weighted direction from surface
                    ray_dir = sample_cosine_hemisphere(dir);
                    pdf_dir = cosine_hemisphere_pdf(vec3_dot(dir, ray_dir));
                    return true;
                }
                sphere_light_count++;
            }
        }
    }
    light_idx -= num_sphere_lights;
    
    // Quad lights
    if (light_idx < num_quad) {
        const auto& quad = scene.quad_lights[light_idx];
        vertex.position = quad.sample_point();
        vertex.normal = quad.normal;
        vertex.emission = quad.emission;
        vertex.light_pdf = light_choice_pdf;
        
        pdf_pos = light_choice_pdf * quad.pdf();
        
        // Sample cosine-weighted direction
        ray_dir = sample_cosine_hemisphere(quad.normal);
        pdf_dir = cosine_hemisphere_pdf(vec3_dot(quad.normal, ray_dir));
        return true;
    }
    light_idx -= num_quad;
    
    // Disk lights
    if (light_idx < num_disk) {
        const auto& disk = scene.disk_lights[light_idx];
        vertex.position = disk.sample_point();
        vertex.normal = disk.normal;
        vertex.emission = disk.emission;
        vertex.light_pdf = light_choice_pdf;
        
        pdf_pos = light_choice_pdf * disk.pdf();
        
        ray_dir = sample_cosine_hemisphere(disk.normal);
        pdf_dir = cosine_hemisphere_pdf(vec3_dot(disk.normal, ray_dir));
        return true;
    }
    
    return false;
}

color BDPTIntegrator::connect_paths(const std::vector<PathVertex>& eye_path,
                                     const std::vector<PathVertex>& light_path,
                                     int s, int t, const Scene& scene) const {
    color L = {0, 0, 0};
    
    if (t == 0) {
        // No eye vertices - not valid for standard rendering
        return L;
    }
    
    if (s == 0) {
        // t vertices from eye, hit light directly
        // This is the standard path tracing contribution
        const PathVertex& eye_v = eye_path[t - 1];
        if (eye_v.is_light()) {
            // Return emission weighted by eye path throughput
            L = vec3_mul(eye_v.throughput, eye_v.emission);
        }
        return L;
    }
    
    if (t == 1) {
        // s vertices from light, connect to camera
        // This is "light tracing" - contributes to specific pixel
        // For now, skip this as it requires splatting to image
        // TODO: Implement light tracing with image splatting
        return L;
    }
    
    // General case: connect eye vertex (t-1) to light vertex (s-1)
    const PathVertex& eye_v = eye_path[t - 1];
    const PathVertex& light_v = light_path[s - 1];
    
    // Check visibility
    if (!visible(scene, eye_v.position, light_v.position)) {
        return L;
    }
    
    // Connection direction
    vec3 d = vec3_sub(light_v.position, eye_v.position);
    double dist_sq = vec3_length_squared(d);
    double dist = std::sqrt(dist_sq);
    vec3 wo_eye = vec3_scale(d, 1.0 / dist);    // From eye vertex toward light
    vec3 wi_light = vec3_negate(wo_eye);         // From light vertex toward eye
    
    // Evaluate BSDF at eye vertex
    color f_eye = evaluate_bsdf(eye_v, eye_v.wi, wo_eye);
    if (vec3_length_squared(f_eye) < BDPT_EPSILON) return L;
    
    // Evaluate BSDF at light vertex (or emission if s=1)
    color f_light;
    if (s == 1) {
        // Light vertex is the light source itself
        // No BSDF, just emission with cosine falloff
        double cos_light = std::abs(vec3_dot(light_v.normal, wi_light));
        f_light = vec3_scale(light_v.emission, cos_light);
    } else {
        // Regular surface vertex on light path
        f_light = evaluate_bsdf(light_v, light_v.wi, wi_light);
    }
    if (vec3_length_squared(f_light) < BDPT_EPSILON) return L;
    
    // Geometry term
    double G = geometry_term(eye_v, light_v);
    
    // Combine throughputs and connection
    color eye_contrib = vec3_mul(eye_v.throughput, f_eye);
    color light_contrib = (s == 1) ? f_light : vec3_mul(light_v.throughput, f_light);
    
    L = vec3_scale(vec3_mul(eye_contrib, light_contrib), G);
    
    return L;
}

color BDPTIntegrator::evaluate_bsdf(const PathVertex& vertex, vec3 wi, vec3 wo) const {
    double cos_theta_o = vec3_dot(vertex.normal, wo);
    
    switch (vertex.mat_type) {
        case MaterialType::Lambertian: {
            // Lambertian: f = albedo / PI
            if (cos_theta_o <= 0) return {0, 0, 0};  // Below surface
            return vec3_scale(vertex.albedo, INV_PI);
        }
        
        case MaterialType::Metal: {
            // Simplified metal: mirror reflection with roughness
            if (cos_theta_o <= 0) return {0, 0, 0};
            
            vec3 reflected = vec3_reflect(vec3_negate(wi), vertex.normal);
            double cos_r = vec3_dot(reflected, wo);
            
            if (vertex.roughness < 0.001) {
                // Perfect mirror - delta, shouldn't be evaluated
                return {0, 0, 0};
            }
            
            // Rough metal: approximate with cosine lobe
            double n = 2.0 / (vertex.roughness * vertex.roughness) - 2.0;
            if (cos_r > 0) {
                double D = (n + 1.0) * INV_PI * 0.5 * std::pow(cos_r, n);
                return vec3_scale(vertex.albedo, D);
            }
            return {0, 0, 0};
        }
        
        case MaterialType::Dielectric:
            // Dielectric is delta - can't evaluate
            return {0, 0, 0};
        
        default:
            return {0, 0, 0};
    }
}

double BDPTIntegrator::bsdf_pdf(const PathVertex& vertex, vec3 wi, vec3 wo) const {
    double cos_theta_o = vec3_dot(vertex.normal, wo);
    
    switch (vertex.mat_type) {
        case MaterialType::Lambertian: {
            if (cos_theta_o <= 0) return 0;
            return cosine_hemisphere_pdf(cos_theta_o);
        }
        
        case MaterialType::Metal: {
            if (cos_theta_o <= 0) return 0;
            
            if (vertex.roughness < 0.001) {
                return 0;  // Delta distribution
            }
            
            vec3 reflected = vec3_reflect(vec3_negate(wi), vertex.normal);
            double cos_r = vec3_dot(reflected, wo);
            if (cos_r > 0) {
                double n = 2.0 / (vertex.roughness * vertex.roughness) - 2.0;
                return (n + 1.0) * INV_PI * 0.5 * std::pow(cos_r, n);
            }
            return 0;
        }
        
        case MaterialType::Dielectric:
            return 0;  // Delta
        
        default:
            return 0;
    }
}

bool BDPTIntegrator::sample_bsdf(const PathVertex& vertex, vec3 wi,
                                  vec3& wo, double& pdf, color& bsdf_val) const {
    switch (vertex.mat_type) {
        case MaterialType::Lambertian: {
            // Cosine-weighted hemisphere sampling
            wo = sample_cosine_hemisphere(vertex.normal);
            double cos_theta = vec3_dot(vertex.normal, wo);
            pdf = cosine_hemisphere_pdf(cos_theta);
            bsdf_val = vec3_scale(vertex.albedo, INV_PI);
            return true;
        }
        
        case MaterialType::Metal: {
            // Reflect with optional roughness
            vec3 reflected = vec3_reflect(vec3_negate(wi), vertex.normal);
            
            if (vertex.roughness < 0.001) {
                // Perfect mirror
                wo = reflected;
                pdf = 1.0;  // Delta
                bsdf_val = vertex.albedo;
            } else {
                // Add roughness perturbation
                vec3 random_in_sphere = vec3_scale(random_unit_vector(), vertex.roughness);
                wo = vec3_normalize(vec3_add(reflected, random_in_sphere));
                
                if (vec3_dot(wo, vertex.normal) <= 0) {
                    return false;  // Below surface
                }
                
                double cos_r = vec3_dot(reflected, wo);
                double n = 2.0 / (vertex.roughness * vertex.roughness) - 2.0;
                pdf = (n + 1.0) * INV_PI * 0.5 * std::pow(std::max(0.0, cos_r), n);
                bsdf_val = vertex.albedo;
            }
            return true;
        }
        
        case MaterialType::Dielectric: {
            // Simplified: just refract/reflect based on Fresnel
            double cos_theta_i = vec3_dot(wi, vertex.normal);
            double etai_over_etat = (cos_theta_i > 0) ? (1.0 / vertex.refraction_index) : vertex.refraction_index;
            
            vec3 n = (cos_theta_i > 0) ? vertex.normal : vec3_negate(vertex.normal);
            cos_theta_i = std::abs(cos_theta_i);
            
            double sin_theta_t_sq = etai_over_etat * etai_over_etat * (1.0 - cos_theta_i * cos_theta_i);
            
            // Fresnel (Schlick approximation)
            double r0 = (1.0 - vertex.refraction_index) / (1.0 + vertex.refraction_index);
            r0 = r0 * r0;
            double fresnel = r0 + (1.0 - r0) * std::pow(1.0 - cos_theta_i, 5.0);
            
            if (sin_theta_t_sq > 1.0 || random_double() < fresnel) {
                // Reflect
                wo = vec3_reflect(vec3_negate(wi), n);
            } else {
                // Refract
                wo = vec3_refract(vec3_negate(wi), n, etai_over_etat);
            }
            
            pdf = 1.0;  // Delta
            bsdf_val = {1, 1, 1};  // Perfect transmission/reflection
            return true;
        }
        
        default:
            return false;
    }
}

double BDPTIntegrator::mis_weight(const std::vector<PathVertex>& eye_path,
                                   const std::vector<PathVertex>& light_path,
                                   int s, int t) const {
    if (!settings_.use_mis) {
        return 1.0;
    }
    
    // Proper MIS weight using the balance heuristic
    // Weight = p_s / sum(p_i) for all valid strategies i
    // We compute ratios relative to the current strategy's PDF
    
    // For BDPT, the key insight is that we can express other strategy PDFs
    // as ratios to the current strategy by "moving" vertices between subpaths
    
    // Special cases
    if (s == 0 && t == 1) return 1.0;  // Direct camera hit - only strategy
    
    // Get the vertices at the connection point
    const PathVertex* eye_v = (t > 0 && t <= static_cast<int>(eye_path.size())) ? 
                               &eye_path[t-1] : nullptr;
    const PathVertex* light_v = (s > 0 && s <= static_cast<int>(light_path.size())) ? 
                                 &light_path[s-1] : nullptr;
    
    // Compute sum of PDF ratios using the power heuristic
    // sum_i (p_i / p_current)^beta where beta = 2 for power heuristic
    double sum_ri = 1.0;  // Start with current strategy (ratio = 1)
    
    // We use a simplified but effective approach:
    // - s=0 (direct light hit): weight by light emission PDF vs BSDF sampling
    // - s=1 (NEE): weight by direct light sampling PDF  
    // - s>1: weight by path PDFs
    
    // Walk through vertices and accumulate PDF ratios
    // This is a simplified version - full BDPT would track all intermediate PDFs
    
    double ri = 1.0;  // Running PDF ratio
    
    // Consider moving connection point toward the light (increase s, decrease t)
    if (t > 1 && eye_v && eye_v->is_connectible()) {
        // Could have sampled this vertex from the light path instead
        double pdf_fwd = eye_v->pdf_fwd;
        double pdf_rev = bsdf_pdf(*eye_v, eye_v->wo, eye_v->wi);
        
        if (pdf_fwd > BDPT_EPSILON && pdf_rev > BDPT_EPSILON) {
            ri *= pdf_rev / pdf_fwd;
            double ri2 = ri * ri;
            sum_ri += ri2;
        }
    }
    
    // Consider moving connection point toward the camera (decrease s, increase t)
    ri = 1.0;
    if (s > 0 && light_v && light_v->is_connectible()) {
        double pdf_fwd = light_v->pdf_fwd;
        double pdf_rev = bsdf_pdf(*light_v, light_v->wo, light_v->wi);
        
        if (pdf_fwd > BDPT_EPSILON && pdf_rev > BDPT_EPSILON) {
            ri *= pdf_rev / pdf_fwd;
            double ri2 = ri * ri;
            sum_ri += ri2;
        }
    }
    
    // For s=0 (direct light hit), consider the PDF of having sampled the light directly
    if (s == 0 && eye_v && eye_v->is_light()) {
        // This path hit a light - compare to NEE sampling
        // Simplified: assume light sampling would have similar PDF
        sum_ri += 1.0;  // Add weight for s=1 strategy
    }
    
    // For s=1 (NEE), compare to direct hit probability
    if (s == 1 && light_v) {
        // Weight against the probability of hitting this light via BSDF sampling
        // This is approximated by the solid angle of the light
        sum_ri += 0.5;  // Heuristic weight for direct hit strategy
    }
    
    return 1.0 / sum_ri;
}

bool BDPTIntegrator::visible(const Scene& scene, point3 p1, point3 p2) const {
    vec3 d = vec3_sub(p2, p1);
    double dist = vec3_length(d);
    
    if (dist < BDPT_EPSILON) return true;
    
    ray shadow_ray = ray_create(p1, vec3_scale(d, 1.0 / dist));
    hit_record rec;
    
    return !scene.hit(shadow_ray, 0.001, dist - 0.001, rec);
}

double BDPTIntegrator::geometry_term(const PathVertex& v1, const PathVertex& v2) const {
    vec3 d = vec3_sub(v2.position, v1.position);
    double dist_sq = vec3_length_squared(d);
    
    if (dist_sq < BDPT_EPSILON) return 0;
    
    double dist = std::sqrt(dist_sq);
    vec3 dir = vec3_scale(d, 1.0 / dist);
    
    double cos1 = std::abs(vec3_dot(v1.normal, dir));
    double cos2 = std::abs(vec3_dot(v2.normal, vec3_negate(dir)));
    
    return cos1 * cos2 / dist_sq;
}

} // namespace raytracer
