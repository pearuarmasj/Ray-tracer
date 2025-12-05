# Geometry & Materials

This document covers proposed additions for geometry primitives, mesh support, and advanced material systems.

## Table of Contents

- [Current Geometry Support](#current-geometry-support)
- [Additional Primitives](#additional-primitives)
- [Triangle Meshes](#triangle-meshes)
- [Constructive Solid Geometry](#constructive-solid-geometry)
- [Advanced Materials](#advanced-materials)
- [Texture System](#texture-system)
- [Procedural Generation](#procedural-generation)

---

## Current Geometry Support

### Sphere Only

The ray tracer currently supports only spheres:

```cpp
struct Sphere {
    point3 center;
    double radius;
    int material_id;
    
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const;
};
```

### Current Materials

| Material | Description |
|----------|-------------|
| Lambertian | Diffuse surface |
| Metal | Perfect specular reflection |
| Dielectric | Glass-like refraction |

---

## Additional Primitives

### Plane

Infinite plane defined by point and normal:

```cpp
// engine/plane.hpp
struct Plane {
    point3 point;
    vec3 normal;
    int material_id;
    
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        double denom = vec3_dot(normal, r.direction);
        if (std::abs(denom) < 1e-6) return false;  // Parallel
        
        double t = vec3_dot(vec3_sub(point, r.origin), normal) / denom;
        if (t < t_min || t > t_max) return false;
        
        rec.t = t;
        rec.point = ray_at(r, t);
        hit_record_set_face_normal(&rec, r, normal);
        rec.material_id = material_id;
        return true;
    }
};
```

### Axis-Aligned Rectangle

For area lights and simple geometry:

```cpp
// engine/rectangle.hpp
struct Rectangle {
    double x0, x1, y0, y1, k;  // z = k plane
    vec3 normal;
    int material_id;
    
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const;
    
    // For area light sampling
    point3 random_point(RNG& rng) const {
        return vec3_create(
            x0 + rng.random_double() * (x1 - x0),
            y0 + rng.random_double() * (y1 - y0),
            k
        );
    }
    
    double area() const { return (x1 - x0) * (y1 - y0); }
};
```

### Box (Axis-Aligned)

```cpp
// engine/box.hpp
struct Box {
    point3 min_point;
    point3 max_point;
    int material_id;
    
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        // Slab method intersection
        double t_enter = t_min;
        double t_exit = t_max;
        int hit_axis = -1;
        
        for (int axis = 0; axis < 3; ++axis) {
            double inv_d = 1.0 / get_component(r.direction, axis);
            double t0 = (get_component(min_point, axis) - get_component(r.origin, axis)) * inv_d;
            double t1 = (get_component(max_point, axis) - get_component(r.origin, axis)) * inv_d;
            
            if (inv_d < 0.0) std::swap(t0, t1);
            
            if (t0 > t_enter) { t_enter = t0; hit_axis = axis; }
            if (t1 < t_exit) t_exit = t1;
            
            if (t_exit < t_enter) return false;
        }
        
        rec.t = t_enter;
        rec.point = ray_at(r, t_enter);
        // Compute normal from hit_axis...
        return true;
    }
};
```

### Cylinder

```cpp
// engine/cylinder.hpp
struct Cylinder {
    point3 center;
    double radius;
    double height;
    vec3 axis;       // Usually (0, 1, 0)
    int material_id;
    bool capped;     // Include end caps
    
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const;
};
```

### Disk

```cpp
// engine/disk.hpp
struct Disk {
    point3 center;
    vec3 normal;
    double radius;
    int material_id;
    
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const {
        double denom = vec3_dot(normal, r.direction);
        if (std::abs(denom) < 1e-6) return false;
        
        double t = vec3_dot(vec3_sub(center, r.origin), normal) / denom;
        if (t < t_min || t > t_max) return false;
        
        point3 p = ray_at(r, t);
        double dist_sq = vec3_length_squared(vec3_sub(p, center));
        if (dist_sq > radius * radius) return false;
        
        rec.t = t;
        rec.point = p;
        hit_record_set_face_normal(&rec, r, normal);
        rec.material_id = material_id;
        return true;
    }
};
```

### Torus

```cpp
// engine/torus.hpp
struct Torus {
    point3 center;
    double major_radius;  // Distance from center to tube center
    double minor_radius;  // Tube radius
    vec3 axis;
    int material_id;
    
    // Requires solving quartic equation
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const;
};
```

---

## Triangle Meshes

### Triangle Primitive

```c
// core/triangle.h
typedef struct triangle {
    point3 v0, v1, v2;           // Vertices
    vec3 n0, n1, n2;             // Vertex normals (for smooth shading)
    double u0, v0_uv, u1, v1_uv, u2, v2_uv;  // UV coordinates
} triangle;

// Möller–Trumbore intersection algorithm
bool triangle_hit(triangle tri, ray r, double t_min, double t_max, hit_record* rec);
```

### Mesh Structure

```cpp
// engine/mesh.hpp
struct Mesh {
    std::vector<point3> vertices;
    std::vector<vec3> normals;
    std::vector<std::pair<double, double>> uvs;
    std::vector<std::array<int, 3>> indices;  // Triangle indices
    int material_id;
    
    // BVH for this mesh
    std::unique_ptr<BVH> bvh;
    
    void build_bvh();
    bool hit(ray r, double t_min, double t_max, hit_record& rec) const;
    
    // Compute smooth normals if not provided
    void compute_normals();
};
```

### OBJ File Loader

```cpp
// engine/obj_loader.hpp
class OBJLoader {
public:
    static std::optional<Mesh> load(const std::string& filename);
    
private:
    static void parse_vertex(const std::string& line, std::vector<point3>& vertices);
    static void parse_normal(const std::string& line, std::vector<vec3>& normals);
    static void parse_texcoord(const std::string& line, std::vector<std::pair<double, double>>& uvs);
    static void parse_face(const std::string& line, 
                          std::vector<std::array<int, 3>>& vertex_indices,
                          std::vector<std::array<int, 3>>& normal_indices,
                          std::vector<std::array<int, 3>>& uv_indices);
};

// Usage
auto mesh = OBJLoader::load("bunny.obj");
if (mesh) {
    mesh->material_id = scene.add_material(Material::lambertian({0.8, 0.6, 0.4}));
    scene.add_mesh(*mesh);
}
```

### PLY File Loader

```cpp
// engine/ply_loader.hpp
class PLYLoader {
public:
    static std::optional<Mesh> load(const std::string& filename);
    // Supports ASCII and binary PLY formats
};
```

### Mesh Instancing

Reuse mesh geometry with different transforms:

```cpp
// engine/instance.hpp
struct Transform {
    mat4 matrix;
    mat4 inverse;
    mat4 inverse_transpose;  // For normals
    
    static Transform translate(vec3 t);
    static Transform rotate(vec3 axis, double angle);
    static Transform scale(vec3 s);
    static Transform compose(const Transform& a, const Transform& b);
};

struct Instance {
    int mesh_index;
    Transform transform;
    int material_id;
    
    bool hit(ray r, const Scene& scene, double t_min, double t_max, hit_record& rec) const {
        // Transform ray to object space
        ray local_ray = transform_ray(r, transform.inverse);
        
        // Intersect with mesh
        if (!scene.meshes[mesh_index].hit(local_ray, t_min, t_max, rec)) {
            return false;
        }
        
        // Transform hit back to world space
        rec.point = transform_point(rec.point, transform.matrix);
        rec.normal = transform_normal(rec.normal, transform.inverse_transpose);
        return true;
    }
};
```

---

## Constructive Solid Geometry

### CSG Operations

```cpp
// engine/csg.hpp
enum class CSGOperation {
    Union,
    Intersection,
    Difference
};

struct CSGNode {
    CSGOperation operation;
    std::variant<int, std::unique_ptr<CSGNode>> left;   // Primitive index or sub-CSG
    std::variant<int, std::unique_ptr<CSGNode>> right;
    
    bool hit(ray r, const Scene& scene, double t_min, double t_max, hit_record& rec) const;
};
```

### CSG Example: Hollow Sphere

```cpp
// Create hollow sphere (difference of two spheres)
int outer = scene.add_sphere({0, 0, 0}, 1.0, mat_glass);
int inner = scene.add_sphere({0, 0, 0}, 0.9, mat_glass);

CSGNode hollow;
hollow.operation = CSGOperation::Difference;
hollow.left = outer;
hollow.right = inner;
```

---

## Advanced Materials

### BSDF Interface

Generalize materials to Bidirectional Scattering Distribution Functions:

```cpp
// engine/bsdf.hpp
struct BSDFSample {
    vec3 wi;           // Incident direction
    color f;           // BSDF value f(wo, wi)
    double pdf;        // Probability density
    bool is_specular;  // Perfect specular (no multiple sampling needed)
};

class BSDF {
public:
    virtual ~BSDF() = default;
    
    // Evaluate BSDF for given directions
    virtual color eval(vec3 wo, vec3 wi, const SurfaceInteraction& si) const = 0;
    
    // Sample incoming direction given outgoing
    virtual BSDFSample sample(vec3 wo, const SurfaceInteraction& si, RNG& rng) const = 0;
    
    // Probability density for sampling wi given wo
    virtual double pdf(vec3 wo, vec3 wi, const SurfaceInteraction& si) const = 0;
};
```

### Disney Principled BSDF

Industry-standard artist-friendly material:

```cpp
// engine/disney.hpp
struct DisneyMaterial {
    color base_color = {0.8, 0.8, 0.8};
    double metallic = 0.0;
    double roughness = 0.5;
    double specular = 0.5;
    double specular_tint = 0.0;
    double anisotropic = 0.0;
    double sheen = 0.0;
    double sheen_tint = 0.0;
    double clearcoat = 0.0;
    double clearcoat_gloss = 1.0;
    double ior = 1.5;
    double transmission = 0.0;
    double transmission_roughness = 0.0;
    
    std::unique_ptr<BSDF> create_bsdf(const SurfaceInteraction& si) const;
};
```

### Subsurface Scattering

For realistic skin, wax, marble:

```cpp
// engine/subsurface.hpp
struct SubsurfaceMaterial {
    color scatter_color;
    double scatter_distance;
    double scale;
    
    color sample_subsurface(const SurfaceInteraction& si, RNG& rng) const;
};
```

### Emission (Area Lights)

```cpp
// engine/emissive.hpp
struct EmissiveMaterial {
    color emission;
    double intensity;
    
    color emitted() const { return vec3_scale(emission, intensity); }
};
```

### Thin Film Interference

For soap bubbles, oil slicks:

```cpp
// engine/thin_film.hpp
struct ThinFilmMaterial {
    double thickness;  // Film thickness in nanometers
    double ior;       // Film refractive index
    
    color interference_color(double wavelength, double angle) const;
};
```

---

## Texture System

### Texture Base Class

```cpp
// engine/texture.hpp
template<typename T>
class Texture {
public:
    virtual ~Texture() = default;
    virtual T sample(double u, double v, const point3& p) const = 0;
};

using ColorTexture = Texture<color>;
using FloatTexture = Texture<double>;
```

### Constant Texture

```cpp
template<typename T>
class ConstantTexture : public Texture<T> {
public:
    explicit ConstantTexture(T value) : value_(value) {}
    T sample(double u, double v, const point3& p) const override { return value_; }
private:
    T value_;
};
```

### Image Texture

```cpp
class ImageTexture : public ColorTexture {
public:
    ImageTexture(const std::string& filename);
    
    color sample(double u, double v, const point3& p) const override {
        // Clamp UV coordinates
        u = std::clamp(u, 0.0, 1.0);
        v = 1.0 - std::clamp(v, 0.0, 1.0);  // Flip V
        
        int i = static_cast<int>(u * width_);
        int j = static_cast<int>(v * height_);
        
        return get_pixel(i, j);
    }
    
private:
    int width_, height_;
    std::vector<unsigned char> data_;
    
    color get_pixel(int x, int y) const;
};
```

### Procedural Textures

#### Checkerboard

```cpp
class CheckerTexture : public ColorTexture {
public:
    CheckerTexture(color c1, color c2, double scale)
        : color1_(c1), color2_(c2), scale_(scale) {}
    
    color sample(double u, double v, const point3& p) const override {
        double sines = std::sin(scale_ * p.x) * 
                       std::sin(scale_ * p.y) * 
                       std::sin(scale_ * p.z);
        return sines < 0 ? color1_ : color2_;
    }
    
private:
    color color1_, color2_;
    double scale_;
};
```

#### Noise Textures

```cpp
// Perlin noise
class PerlinTexture : public ColorTexture {
public:
    PerlinTexture(double scale = 1.0) : scale_(scale) {}
    
    color sample(double u, double v, const point3& p) const override {
        double noise = perlin_.turbulence(vec3_scale(p, scale_));
        return vec3_scale({1.0, 1.0, 1.0}, 0.5 * (1.0 + noise));
    }
    
private:
    PerlinNoise perlin_;
    double scale_;
};
```

### Normal Mapping

```cpp
class NormalMap : public Texture<vec3> {
public:
    NormalMap(const std::string& filename);
    
    vec3 sample(double u, double v, const point3& p) const override {
        color rgb = sample_image(u, v);
        // Convert from [0,1] to [-1,1]
        return vec3_normalize({
            rgb.x * 2.0 - 1.0,
            rgb.y * 2.0 - 1.0,
            rgb.z * 2.0 - 1.0
        });
    }
};
```

---

## Procedural Generation

### Fractal Terrain

```cpp
// engine/terrain.hpp
class TerrainGenerator {
public:
    TerrainGenerator(int resolution, double scale, int seed);
    
    Mesh generate_heightmap() const;
    
private:
    double height_at(double x, double z) const;  // Uses fractal noise
    
    int resolution_;
    double scale_;
    PerlinNoise noise_;
};
```

### Procedural Geometry

```cpp
// Generate sphere with subdivisions
Mesh generate_uv_sphere(int u_segments, int v_segments, double radius);

// Generate cube
Mesh generate_cube(double size);

// Generate torus mesh
Mesh generate_torus(double major_radius, double minor_radius, int u_segments, int v_segments);
```

---

## Implementation Priority

| Priority | Feature | Impact | Effort |
|----------|---------|--------|--------|
| 🔴 High | Triangle primitive | Essential for meshes | Low |
| 🔴 High | OBJ loader | Import 3D models | Medium |
| 🔴 High | Basic textures | Visual variety | Medium |
| 🟡 Medium | Additional primitives | Scene variety | Low each |
| 🟡 Medium | Normal mapping | Surface detail | Medium |
| 🟢 Low | Disney BSDF | Artistic control | High |
| 🟢 Low | CSG | Complex shapes | Medium |
| 🟢 Low | Subsurface | Realism | High |

---

## UV Coordinate Examples

### Sphere UV Mapping

```cpp
void sphere_uv(const point3& p, double& u, double& v) {
    // p is a point on the unit sphere centered at origin
    double theta = std::acos(-p.y);
    double phi = std::atan2(-p.z, p.x) + PI;
    
    u = phi / (2 * PI);
    v = theta / PI;
}
```

### Planar UV Mapping

```cpp
void planar_uv(const point3& p, double scale, double& u, double& v) {
    u = p.x * scale;
    v = p.z * scale;
}
```

---

*See also: [Rendering Features](RENDERING_FEATURES.md) for material shading algorithms*
