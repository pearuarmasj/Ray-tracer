# Architecture Improvements

This document details proposed architectural enhancements to make the ray tracer more modular, performant, and maintainable.

## Table of Contents

- [Current Architecture Overview](#current-architecture-overview)
- [Acceleration Structures](#acceleration-structures)
- [Memory Management](#memory-management)
- [SIMD Optimizations](#simd-optimizations)
- [Threading Model](#threading-model)
- [Plugin System](#plugin-system)
- [Error Handling](#error-handling)

---

## Current Architecture Overview

The current architecture consists of two layers:

```
┌─────────────────────────────────────────────┐
│            C++ Engine Layer                 │
│  (renderer, scene, materials, camera)       │
├─────────────────────────────────────────────┤
│              C Core Layer                   │
│      (vec3, ray, hit_record)                │
└─────────────────────────────────────────────┘
```

### Strengths
- Clean separation between math primitives (C) and scene management (C++)
- Simple, readable code
- OpenMP parallelization

### Areas for Improvement
- No acceleration structure (linear scene traversal)
- Limited extensibility
- No caching or memory pooling

---

## Acceleration Structures

### Bounding Volume Hierarchy (BVH)

The most impactful improvement would be implementing a BVH for O(log n) intersection tests.

#### Proposed Design

```cpp
// core/aabb.h - Axis-Aligned Bounding Box
typedef struct aabb {
    point3 min;
    point3 max;
} aabb;

aabb aabb_surrounding_box(aabb box0, aabb box1);
bool aabb_hit(aabb box, ray r, double t_min, double t_max);

// engine/bvh.hpp - BVH Node
struct BVHNode {
    aabb bbox;
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    int primitive_index;  // -1 for internal nodes
    
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const;
};
```

#### Construction Strategies

| Strategy | Build Time | Quality | Best For |
|----------|-----------|---------|----------|
| **Midpoint** | Fast | Low | Dynamic scenes |
| **SAH (Surface Area Heuristic)** | Slow | High | Static scenes |
| **HLBVH** | Medium | Medium | Large scenes |

#### Implementation Priority

1. **Phase 1**: Basic midpoint BVH
2. **Phase 2**: SAH-based construction
3. **Phase 3**: Parallel BVH building

### Future Acceleration Structures

- **k-d Trees** for scenes with varying primitive density
- **Octrees** for spatial queries and instancing
- **Two-Level BVH** for instance support (TLAS/BLAS)

---

## Memory Management

### Custom Allocators

For better cache utilization, implement arena allocators for render-time allocations:

```cpp
// engine/memory.hpp
class RenderArena {
public:
    explicit RenderArena(size_t block_size = 1024 * 1024);
    
    template<typename T, typename... Args>
    T* alloc(Args&&... args);
    
    void reset();  // Clear for next frame
    
private:
    std::vector<std::unique_ptr<char[]>> blocks_;
    size_t current_offset_ = 0;
};
```

### Benefits

| Aspect | Current | With Allocators |
|--------|---------|-----------------|
| Allocation overhead | `malloc`/`new` per object | Near-zero |
| Cache locality | Poor (scattered) | Excellent (contiguous) |
| Memory fragmentation | Yes | No |
| Thread safety | Global locks | Per-thread arenas |

### Memory Layout Optimization

```cpp
// Structure of Arrays (SoA) for SIMD-friendly access
struct SphereData {
    std::vector<float> center_x;
    std::vector<float> center_y;
    std::vector<float> center_z;
    std::vector<float> radius;
    std::vector<int> material_id;
};
```

---

## SIMD Optimizations

### Current vec3 Operations

All vec3 operations currently use scalar code. SIMD could provide 2-4x speedup.

### Proposed SIMD vec3

```c
// core/vec3_simd.h
#ifdef __AVX__
#include <immintrin.h>

typedef struct vec3_simd {
    __m256d data;  // [x, y, z, w] - w unused or used for batch processing
} vec3_simd;

// Process 4 vec3 dot products simultaneously
__m256d vec3_dot4(vec3_simd a[4], vec3_simd b[4]);
#endif
```

### Ray-Sphere Intersection (4 spheres at once)

```cpp
// Test ray against 4 spheres using AVX
bool hit_spheres_simd(ray r, 
                      const SphereData& spheres,
                      int start_idx,
                      double t_min, double t_max,
                      hit_record& rec);
```

### Fallback Strategy

```cpp
#if defined(__AVX512F__)
    #define SIMD_WIDTH 8
#elif defined(__AVX__)
    #define SIMD_WIDTH 4
#elif defined(__SSE__)
    #define SIMD_WIDTH 2
#else
    #define SIMD_WIDTH 1  // Scalar fallback
#endif
```

---

## Threading Model

### Current Model

- OpenMP parallel for with dynamic scheduling
- Row-based work distribution

### Proposed Improvements

#### Tile-Based Rendering

```cpp
struct RenderTile {
    int x, y;           // Tile position
    int width, height;  // Tile dimensions
    int samples;        // Samples completed
};

class TileScheduler {
public:
    std::optional<RenderTile> get_next_tile();
    void tile_completed(RenderTile tile, const std::vector<color>& pixels);
    
private:
    std::queue<RenderTile> pending_tiles_;
    std::mutex mutex_;
};
```

#### Work-Stealing

```cpp
class WorkStealingScheduler {
    std::vector<std::deque<RenderTile>> per_thread_queues_;
    
public:
    void push(int thread_id, RenderTile tile);
    std::optional<RenderTile> pop(int thread_id);  // Own queue first, then steal
};
```

#### Benefits

| Feature | Row-Based | Tile-Based | Work-Stealing |
|---------|-----------|------------|---------------|
| Load balancing | Poor | Good | Excellent |
| Cache locality | Poor | Good | Good |
| Complexity | Low | Medium | High |
| Progress reporting | Line-based | Tile-based | Fine-grained |

---

## Plugin System

### Plugin Types

```cpp
// engine/plugin.hpp
class Plugin {
public:
    virtual ~Plugin() = default;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
};

class MaterialPlugin : public Plugin {
public:
    virtual std::unique_ptr<Material> create(const PropertyMap& props) = 0;
};

class ShapePlugin : public Plugin {
public:
    virtual std::unique_ptr<Shape> create(const PropertyMap& props) = 0;
};

class IntegratorPlugin : public Plugin {
public:
    virtual std::unique_ptr<Integrator> create(const Settings& settings) = 0;
};
```

### Plugin Registry

```cpp
class PluginManager {
public:
    static PluginManager& instance();
    
    void register_material(const std::string& name, MaterialPlugin* plugin);
    void register_shape(const std::string& name, ShapePlugin* plugin);
    void register_integrator(const std::string& name, IntegratorPlugin* plugin);
    
    std::unique_ptr<Material> create_material(const std::string& name, const PropertyMap& props);
    // ... etc
    
private:
    std::unordered_map<std::string, MaterialPlugin*> materials_;
    std::unordered_map<std::string, ShapePlugin*> shapes_;
    std::unordered_map<std::string, IntegratorPlugin*> integrators_;
};

// Usage in material plugins
class GlassMaterialPlugin : public MaterialPlugin {
public:
    std::string name() const override { return "glass"; }
    std::unique_ptr<Material> create(const PropertyMap& props) override {
        double ior = props.get<double>("ior", 1.5);
        return std::make_unique<DielectricMaterial>(ior);
    }
};
```

### Dynamic Loading (Future)

```cpp
// Load plugins from shared libraries
void load_plugins_from_directory(const std::filesystem::path& plugin_dir);
```

---

## Error Handling

### Current State

Limited error handling with basic return codes.

### Proposed Error System

```cpp
// engine/error.hpp
enum class ErrorCode {
    Success = 0,
    FileNotFound,
    InvalidFormat,
    OutOfMemory,
    InvalidParameter,
    UnsupportedFeature
};

class Result {
public:
    bool success() const { return code_ == ErrorCode::Success; }
    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }
    
    static Result ok() { return Result{ErrorCode::Success, ""}; }
    static Result error(ErrorCode code, std::string msg) { return Result{code, std::move(msg)}; }
    
private:
    ErrorCode code_;
    std::string message_;
};

// Expected type for values that might fail
template<typename T>
class Expected {
public:
    bool has_value() const;
    T& value();
    const Result& error() const;
    
    // Monadic operations
    template<typename F>
    auto and_then(F&& f) -> Expected<decltype(f(std::declval<T>()))>;
};
```

### Usage Example

```cpp
Expected<Scene> load_scene(const std::string& filename) {
    auto file = open_file(filename);
    if (!file.has_value()) {
        return Expected<Scene>::error(file.error());
    }
    
    return parse_scene(file.value())
        .and_then([](auto& data) { return validate_scene(data); })
        .and_then([](auto& data) { return build_scene(data); });
}
```

---

## Implementation Priority

| Priority | Feature | Impact | Effort |
|----------|---------|--------|--------|
| 🔴 High | BVH | Performance: 10-100x | Medium |
| 🔴 High | Error handling | Reliability | Low |
| 🟡 Medium | Tile-based rendering | Better parallelism | Low |
| 🟡 Medium | Memory pooling | Performance: 1.5x | Medium |
| 🟢 Low | Plugin system | Extensibility | High |
| 🟢 Low | SIMD | Performance: 2-4x | High |

---

## Next Steps

1. Implement AABB primitives in the C core layer
2. Create BVH node structure
3. Add scene building with automatic BVH construction
4. Benchmark before/after with complex scenes

---

*See also: [Rendering Features](RENDERING_FEATURES.md) for rendering algorithm improvements*
