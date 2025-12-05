# Rendering Features

This document details advanced rendering algorithms and optimizations to improve image quality and performance.

## Table of Contents

- [Current Rendering Model](#current-rendering-model)
- [Path Tracing](#path-tracing)
- [Sampling Strategies](#sampling-strategies)
- [Global Illumination](#global-illumination)
- [Denoising](#denoising)
- [Tone Mapping & Color](#tone-mapping--color)
- [Performance Optimizations](#performance-optimizations)

---

## Current Rendering Model

The ray tracer currently implements **Whitted-style ray tracing**:

```
For each pixel:
    Cast ray from camera
    If hit:
        Based on material:
            Lambertian: Simple diffuse (normal direction)
            Metal: Perfect specular reflection
            Dielectric: Refraction with Fresnel
        Recursively trace scattered ray
    Else:
        Return sky gradient
```

### Limitations

- No soft shadows
- No indirect diffuse illumination
- Limited to perfect reflections/refractions
- No caustics
- Deterministic (same seed → same image)

---

## Path Tracing

### Monte Carlo Path Tracing

Replace Whitted-style with unbiased path tracing for physically accurate results:

```cpp
color path_trace(ray r, const Scene& scene, int depth, RNG& rng) {
    if (depth <= 0) return vec3_zero();
    
    hit_record rec;
    if (!scene.hit(r, 0.001, INFINITY, rec)) {
        return environment_color(r);
    }
    
    const Material& mat = scene.get_material(rec.material_id);
    
    // Sample scattered direction from BSDF
    ScatterRecord srec;
    if (!mat.scatter(r, rec, rng, srec)) {
        return mat.emitted(rec);  // Light source
    }
    
    // Recursive path continuation
    color incoming = path_trace(srec.scattered, scene, depth - 1, rng);
    
    // Light transport equation: L = Le + f * Li * cos(θ) / pdf
    return vec3_add(
        mat.emitted(rec),
        vec3_scale(
            vec3_mul(srec.attenuation, incoming),
            vec3_dot(rec.normal, srec.scattered.direction) / srec.pdf
        )
    );
}
```

### Russian Roulette

Unbiased path termination to reduce computation:

```cpp
color path_trace_rr(ray r, const Scene& scene, int depth, RNG& rng) {
    // After minimum depth, probabilistically terminate
    if (depth > 3) {
        double q = std::max(0.05, 1.0 - max_component(throughput));
        if (rng.random_double() < q) {
            return vec3_zero();
        }
        throughput = vec3_scale(throughput, 1.0 / (1.0 - q));
    }
    // ... continue tracing
}
```

---

## Sampling Strategies

### Random Number Generation

Add proper random sampling to the core layer:

```c
// core/random.h
typedef struct rng_state {
    uint64_t state;
    uint64_t inc;
} rng_state;

void rng_seed(rng_state* rng, uint64_t seed);
double rng_double(rng_state* rng);           // [0, 1)
vec3 rng_in_unit_sphere(rng_state* rng);     // Random point in unit sphere
vec3 rng_unit_vector(rng_state* rng);        // Random unit vector
vec3 rng_in_hemisphere(rng_state* rng, vec3 normal);
vec3 rng_cosine_direction(rng_state* rng);   // Cosine-weighted hemisphere
```

### Importance Sampling

Sample directions proportional to their contribution:

```cpp
// Cosine-weighted hemisphere sampling (for diffuse)
vec3 sample_cosine_hemisphere(vec3 normal, RNG& rng) {
    double r1 = rng.random_double();
    double r2 = rng.random_double();
    
    double phi = 2.0 * PI * r1;
    double cos_theta = std::sqrt(r2);
    double sin_theta = std::sqrt(1.0 - r2);
    
    vec3 local = vec3_create(
        std::cos(phi) * sin_theta,
        std::sin(phi) * sin_theta,
        cos_theta
    );
    
    return to_world(local, normal);  // Transform to world space
}
```

### Multiple Importance Sampling (MIS)

Combine multiple sampling strategies optimally:

```cpp
color mis_path_trace(ray r, const Scene& scene, int depth, RNG& rng) {
    // ... hit detection
    
    // Sample BSDF
    ScatterRecord bsdf_sample;
    mat.scatter(r, rec, rng, bsdf_sample);
    double bsdf_pdf = bsdf_sample.pdf;
    
    // Sample light directly
    LightSample light_sample = scene.sample_light(rec.point, rng);
    double light_pdf = light_sample.pdf;
    
    // Power heuristic weight
    double weight_bsdf = power_heuristic(bsdf_pdf, light_pdf);
    double weight_light = power_heuristic(light_pdf, bsdf_pdf);
    
    // Combine contributions
    return vec3_add(
        vec3_scale(bsdf_contribution, weight_bsdf),
        vec3_scale(light_contribution, weight_light)
    );
}

double power_heuristic(double pdf1, double pdf2, int beta = 2) {
    double f = std::pow(pdf1, beta);
    double g = std::pow(pdf2, beta);
    return f / (f + g);
}
```

### Stratified Sampling

Divide sample space for better coverage:

```cpp
// engine/sampler.hpp
class StratifiedSampler {
public:
    StratifiedSampler(int x_samples, int y_samples);
    
    // Get next 2D sample in [0,1)^2
    std::pair<double, double> get_2d();
    
    void start_pixel(int x, int y);
    void start_sample(int sample_index);
    
private:
    int x_samples_, y_samples_;
    int current_stratum_;
    RNG rng_;
};

void render_pixel(int px, int py, StratifiedSampler& sampler) {
    sampler.start_pixel(px, py);
    
    for (int s = 0; s < samples_per_pixel; ++s) {
        sampler.start_sample(s);
        auto [jitter_x, jitter_y] = sampler.get_2d();
        
        double u = (px + jitter_x) / width;
        double v = (py + jitter_y) / height;
        
        ray r = camera.get_ray(u, v, sampler);
        // ... trace
    }
}
```

---

## Global Illumination

### Next Event Estimation (NEE)

Directly sample light sources at each bounce:

```cpp
color nee_path_trace(ray r, const Scene& scene, int depth, RNG& rng) {
    // ... hit surface
    
    color result = mat.emitted(rec);  // Direct emission
    
    // Direct lighting (sample light explicitly)
    for (const auto& light : scene.lights()) {
        LightSample ls = light->sample(rec.point, rng);
        
        // Shadow ray
        ray shadow_ray = ray_create(rec.point, ls.direction);
        if (!scene.hit(shadow_ray, 0.001, ls.distance - 0.001, temp_rec)) {
            // Not occluded
            double cos_theta = std::max(0.0, vec3_dot(rec.normal, ls.direction));
            color bsdf = mat.eval(r.direction, ls.direction, rec);
            result = vec3_add(result, 
                vec3_scale(vec3_mul(ls.radiance, bsdf), cos_theta / ls.pdf));
        }
    }
    
    // Indirect lighting (sample BSDF)
    ScatterRecord srec;
    if (mat.scatter(r, rec, rng, srec)) {
        color indirect = nee_path_trace(srec.scattered, scene, depth - 1, rng);
        result = vec3_add(result, vec3_mul(srec.attenuation, indirect));
    }
    
    return result;
}
```

### Bidirectional Path Tracing

Connect paths from both camera and lights:

```cpp
class BidirectionalPathTracer {
public:
    color render_pixel(int x, int y, RNG& rng) {
        // Generate camera subpath
        std::vector<PathVertex> camera_path = trace_camera_path(x, y, rng);
        
        // Generate light subpath
        std::vector<PathVertex> light_path = trace_light_path(rng);
        
        // Connect all valid combinations
        color result = vec3_zero();
        for (int t = 1; t <= camera_path.size(); ++t) {
            for (int s = 0; s <= light_path.size(); ++s) {
                double weight = mis_weight(s, t);
                color contribution = connect_paths(camera_path, t, light_path, s);
                result = vec3_add(result, vec3_scale(contribution, weight));
            }
        }
        return result;
    }
};
```

### Photon Mapping

For caustics and complex light transport:

```cpp
// Phase 1: Trace photons from lights
class PhotonMap {
public:
    void trace_photons(const Scene& scene, int num_photons, RNG& rng);
    void build_kd_tree();
    
    // Phase 2: Gather photons at hit points
    color gather(point3 point, vec3 normal, double radius) const;
    
private:
    struct Photon {
        point3 position;
        vec3 direction;
        color power;
    };
    
    std::vector<Photon> photons_;
    KDTree<Photon> kd_tree_;
};
```

---

## Denoising

### Accumulation Buffer

Progressive rendering with sample accumulation:

```cpp
class AccumulationBuffer {
public:
    AccumulationBuffer(int width, int height);
    
    void add_sample(int x, int y, color c);
    color get_averaged(int x, int y) const;
    int sample_count() const { return sample_count_; }
    
private:
    std::vector<color> accumulated_;
    int sample_count_ = 0;
};
```

### Auxiliary Feature Buffers (AOVs)

Generate data for denoising:

```cpp
struct AOVs {
    std::vector<color> albedo;
    std::vector<vec3> normal;
    std::vector<float> depth;
    std::vector<vec3> motion;  // For temporal denoising
};

void render_with_aovs(const Scene& scene, const Camera& camera, 
                      Image& beauty, AOVs& aovs);
```

### Intel Open Image Denoise Integration

```cpp
#include <OpenImageDenoise/oidn.hpp>

Image denoise(const Image& noisy, const AOVs& aovs) {
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();
    
    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", noisy.data(), oidn::Format::Float3, width, height);
    filter.setImage("albedo", aovs.albedo.data(), oidn::Format::Float3, width, height);
    filter.setImage("normal", aovs.normal.data(), oidn::Format::Float3, width, height);
    filter.setImage("output", result.data(), oidn::Format::Float3, width, height);
    filter.commit();
    filter.execute();
    
    return result;
}
```

---

## Tone Mapping & Color

### HDR Pipeline

```cpp
// Reinhard tone mapping
color tone_map_reinhard(color hdr) {
    return vec3_create(
        hdr.x / (1.0 + hdr.x),
        hdr.y / (1.0 + hdr.y),
        hdr.z / (1.0 + hdr.z)
    );
}

// ACES filmic tone mapping
color tone_map_aces(color hdr) {
    const double a = 2.51;
    const double b = 0.03;
    const double c = 2.43;
    const double d = 0.59;
    const double e = 0.14;
    
    color x = hdr;
    return vec3_clamp(
        vec3_mul(
            vec3_add(vec3_scale(x, a), vec3_create(b, b, b)),
            x
        ) / vec3_mul(
            vec3_add(vec3_scale(x, c), vec3_create(d, d, d)),
            vec3_add(x, vec3_create(e, e, e))
        ),
        0.0, 1.0
    );
}
```

### Color Space Management

```cpp
enum class ColorSpace {
    sRGB,
    LinearRGB,
    ACEScg,
    DisplayP3
};

class ColorPipeline {
public:
    // Working space is linear
    color to_working_space(color c, ColorSpace from);
    color to_output_space(color c, ColorSpace to);
    
    // Gamma correction
    color apply_gamma(color linear, double gamma = 2.2);
    color remove_gamma(color encoded, double gamma = 2.2);
};
```

---

## Performance Optimizations

### Adaptive Sampling

Concentrate samples where needed:

```cpp
class AdaptiveSampler {
public:
    AdaptiveSampler(int min_samples, int max_samples, double variance_threshold);
    
    bool needs_more_samples(int x, int y) const {
        return sample_count(x, y) < min_samples_ ||
               (variance(x, y) > threshold_ && sample_count(x, y) < max_samples_);
    }
    
private:
    double variance(int x, int y) const;  // Compute pixel variance
    
    int min_samples_, max_samples_;
    double threshold_;
    std::vector<color> sum_;
    std::vector<color> sum_squared_;
    std::vector<int> counts_;
};
```

### Progressive Rendering

Show results as they accumulate:

```cpp
class ProgressiveRenderer {
public:
    void start(Scene& scene, Camera& camera);
    void render_pass();           // Render one sample per pixel
    const Image& current() const; // Get current state
    void stop();
    
private:
    std::atomic<bool> running_{false};
    AccumulationBuffer buffer_;
    std::thread render_thread_;
};
```

### Ray Packet Tracing

Process multiple rays together:

```cpp
// Trace 4 rays at once using SIMD
struct RayPacket4 {
    __m128 origin_x, origin_y, origin_z;
    __m128 dir_x, dir_y, dir_z;
    __m128 t_min, t_max;
    __m128i active;  // Mask for active rays
};

struct HitPacket4 {
    __m128 t;
    __m128 normal_x, normal_y, normal_z;
    __m128i material_id;
    __m128i hit_mask;
};

void intersect_bvh_packet(const BVH& bvh, RayPacket4& rays, HitPacket4& hits);
```

---

## Implementation Roadmap

### Phase 1: Foundation (Required)
1. ✅ Add random number generation
2. Add anti-aliasing with random jitter
3. Implement cosine-weighted hemisphere sampling

### Phase 2: Basic Path Tracing
1. Convert to proper path tracing
2. Add Russian roulette
3. Implement NEE for direct lighting

### Phase 3: Quality Improvements
1. Multiple importance sampling
2. Stratified sampling
3. Adaptive sampling

### Phase 4: Advanced
1. Bidirectional path tracing
2. Denoising integration
3. Photon mapping for caustics

---

## Comparison: Whitted vs Path Tracing

| Feature | Whitted | Path Tracing |
|---------|---------|--------------|
| Soft shadows | ❌ | ✅ |
| Color bleeding | ❌ | ✅ |
| Caustics | ❌ | ✅ |
| Physically correct | ❌ | ✅ |
| Noise-free | ✅ | Requires samples |
| Performance | Fast | Slower |

---

*See also: [Architecture Improvements](ARCHITECTURE_IMPROVEMENTS.md) for structural changes*
