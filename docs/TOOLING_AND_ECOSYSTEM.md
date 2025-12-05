# Tooling & Ecosystem

This document covers build system improvements, testing infrastructure, debugging tools, and integration with the broader graphics ecosystem.

## Table of Contents

- [Build System](#build-system)
- [Testing Infrastructure](#testing-infrastructure)
- [Debugging & Profiling](#debugging--profiling)
- [Scene Description](#scene-description)
- [Command Line Interface](#command-line-interface)
- [Interactive Preview](#interactive-preview)
- [External Integrations](#external-integrations)
- [Documentation](#documentation)

---

## Build System

### Current Setup

- CMake 3.21+
- C23 / C++23 standards
- OpenMP for parallelization
- Manual build process

### Proposed Improvements

#### Package Management with vcpkg/Conan

```cmake
# CMakeLists.txt with vcpkg
find_package(OpenImageIO CONFIG REQUIRED)
find_package(OpenImageDenoise CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

target_link_libraries(raytracer PRIVATE 
    OpenImageIO::OpenImageIO
    OpenImageDenoise
    nlohmann_json::nlohmann_json
    fmt::fmt
)
```

#### CMake Presets Expansion

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "base",
      "hidden": true,
      "binaryDir": "${sourceDir}/build/${presetName}",
      "installDir": "${sourceDir}/install/${presetName}"
    },
    {
      "name": "debug",
      "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "ENABLE_SANITIZERS": "ON"
      }
    },
    {
      "name": "release",
      "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "ENABLE_LTO": "ON"
      }
    },
    {
      "name": "profile",
      "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "ENABLE_PROFILING": "ON"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "debug",
      "configurePreset": "debug"
    },
    {
      "name": "release",
      "configurePreset": "release",
      "configuration": "Release"
    }
  ],
  "testPresets": [
    {
      "name": "default",
      "configurePreset": "debug",
      "output": {"outputOnFailure": true}
    }
  ]
}
```

#### Build Options

```cmake
# New CMake options
option(ENABLE_SIMD "Enable SIMD optimizations" ON)
option(ENABLE_GPU "Enable GPU rendering (CUDA/OptiX)" OFF)
option(ENABLE_SANITIZERS "Enable address/undefined behavior sanitizers" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_LTO "Enable link-time optimization" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_DOCS "Build documentation" OFF)

# Sanitizer configuration
if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
```

---

## Testing Infrastructure

### Unit Testing Framework

Use Google Test for C++ testing:

```cmake
# CMakeLists.txt
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    
    add_executable(raytracer_tests
        tests/vec3_test.cpp
        tests/ray_test.cpp
        tests/sphere_test.cpp
        tests/material_test.cpp
        tests/renderer_test.cpp
    )
    
    target_link_libraries(raytracer_tests 
        PRIVATE raytracer_core raytracer_engine GTest::gtest_main)
    
    include(GoogleTest)
    gtest_discover_tests(raytracer_tests)
endif()
```

### Test Examples

```cpp
// tests/vec3_test.cpp
#include <gtest/gtest.h>

extern "C" {
#include "../core/vec3.h"
}

TEST(Vec3Test, CreateZero) {
    vec3 v = vec3_zero();
    EXPECT_DOUBLE_EQ(v.x, 0.0);
    EXPECT_DOUBLE_EQ(v.y, 0.0);
    EXPECT_DOUBLE_EQ(v.z, 0.0);
}

TEST(Vec3Test, Addition) {
    vec3 a = vec3_create(1.0, 2.0, 3.0);
    vec3 b = vec3_create(4.0, 5.0, 6.0);
    vec3 c = vec3_add(a, b);
    
    EXPECT_DOUBLE_EQ(c.x, 5.0);
    EXPECT_DOUBLE_EQ(c.y, 7.0);
    EXPECT_DOUBLE_EQ(c.z, 9.0);
}

TEST(Vec3Test, Normalization) {
    vec3 v = vec3_create(3.0, 0.0, 4.0);
    vec3 n = vec3_normalize(v);
    
    EXPECT_DOUBLE_EQ(vec3_length(n), 1.0);
    EXPECT_DOUBLE_EQ(n.x, 0.6);
    EXPECT_DOUBLE_EQ(n.z, 0.8);
}

TEST(Vec3Test, DotProduct) {
    vec3 a = vec3_create(1.0, 0.0, 0.0);
    vec3 b = vec3_create(0.0, 1.0, 0.0);
    
    EXPECT_DOUBLE_EQ(vec3_dot(a, b), 0.0);  // Orthogonal
    EXPECT_DOUBLE_EQ(vec3_dot(a, a), 1.0);  // Same direction
}
```

```cpp
// tests/sphere_test.cpp
#include <gtest/gtest.h>
#include "../engine/sphere.hpp"

using namespace raytracer;

TEST(SphereTest, RayHitsSphere) {
    Sphere sphere({0, 0, -1}, 0.5, 0);
    ray r = ray_create({0, 0, 0}, {0, 0, -1});
    hit_record rec;
    
    EXPECT_TRUE(sphere.hit(r, 0.001, 1000, rec));
    EXPECT_NEAR(rec.t, 0.5, 1e-6);
}

TEST(SphereTest, RayMissesSphere) {
    Sphere sphere({0, 0, -1}, 0.5, 0);
    ray r = ray_create({10, 0, 0}, {0, 0, -1});
    hit_record rec;
    
    EXPECT_FALSE(sphere.hit(r, 0.001, 1000, rec));
}

TEST(SphereTest, RayInsideSphere) {
    Sphere sphere({0, 0, 0}, 1.0, 0);
    ray r = ray_create({0, 0, 0}, {1, 0, 0});
    hit_record rec;
    
    EXPECT_TRUE(sphere.hit(r, 0.001, 1000, rec));
    EXPECT_FALSE(rec.front_face);  // Hit from inside
}
```

### Render Comparison Tests

```cpp
// tests/render_test.cpp
TEST(RenderTest, GroundTruthComparison) {
    Scene scene = create_test_scene();
    Camera camera = create_test_camera();
    
    Renderer::Settings settings;
    settings.width = 100;
    settings.height = 100;
    settings.samples_per_pixel = 1;
    
    Renderer renderer(settings);
    Image result = renderer.render(scene, camera);
    
    // Compare against reference image
    Image reference = load_reference("test_scene_ref.png");
    
    double mse = compute_mse(result, reference);
    EXPECT_LT(mse, 0.01);  // Allow small numerical differences
}
```

### Benchmarking

```cpp
// benchmarks/render_benchmark.cpp
#include <benchmark/benchmark.h>

static void BM_RenderSimpleScene(benchmark::State& state) {
    Scene scene = create_simple_scene();
    Camera camera = create_camera();
    
    Renderer::Settings settings;
    settings.width = state.range(0);
    settings.height = state.range(0);
    settings.samples_per_pixel = 1;
    
    Renderer renderer(settings);
    
    for (auto _ : state) {
        Image image = renderer.render(scene, camera);
        benchmark::DoNotOptimize(image);
    }
    
    state.SetItemsProcessed(state.iterations() * settings.width * settings.height);
}

BENCHMARK(BM_RenderSimpleScene)->Range(64, 512);

static void BM_Vec3Dot(benchmark::State& state) {
    vec3 a = vec3_create(1.0, 2.0, 3.0);
    vec3 b = vec3_create(4.0, 5.0, 6.0);
    
    for (auto _ : state) {
        double result = vec3_dot(a, b);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BM_Vec3Dot);
```

---

## Debugging & Profiling

### Debug Visualization

```cpp
// engine/debug.hpp
namespace debug {

// Render surface normals as RGB
void enable_normal_view(Renderer& renderer);

// Render depth buffer
void enable_depth_view(Renderer& renderer, double near, double far);

// Render BVH bounding boxes
void render_bvh_boxes(const Scene& scene, Image& output);

// Render sample heat map (samples per pixel)
void render_sample_heatmap(const AdaptiveSampler& sampler, Image& output);

}
```

### Performance Profiling

```cpp
// engine/profiler.hpp
class ScopedTimer {
public:
    ScopedTimer(const std::string& name);
    ~ScopedTimer();
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

#define PROFILE_SCOPE(name) ScopedTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

// Usage
color ray_color(ray r, const Scene& scene, int depth) {
    PROFILE_FUNCTION();
    // ...
}
```

### Statistics Collection

```cpp
// engine/stats.hpp
struct RenderStats {
    std::atomic<uint64_t> rays_traced{0};
    std::atomic<uint64_t> primary_rays{0};
    std::atomic<uint64_t> shadow_rays{0};
    std::atomic<uint64_t> bvh_traversals{0};
    std::atomic<uint64_t> intersection_tests{0};
    
    double rays_per_second() const;
    void print_summary() const;
    void reset();
};
```

---

## Scene Description

### JSON Scene Format

```json
{
  "settings": {
    "width": 1920,
    "height": 1080,
    "samples_per_pixel": 64,
    "max_depth": 50,
    "output": "render.png"
  },
  "camera": {
    "position": [0, 1, 3],
    "look_at": [0, 0, 0],
    "up": [0, 1, 0],
    "fov": 45,
    "aperture": 0.1,
    "focus_distance": 3.0
  },
  "materials": [
    {
      "name": "ground",
      "type": "lambertian",
      "albedo": [0.8, 0.8, 0.0]
    },
    {
      "name": "glass",
      "type": "dielectric",
      "ior": 1.5
    },
    {
      "name": "metal_gold",
      "type": "metal",
      "albedo": [0.8, 0.6, 0.2],
      "roughness": 0.0
    }
  ],
  "objects": [
    {
      "type": "sphere",
      "center": [0, -100.5, -1],
      "radius": 100,
      "material": "ground"
    },
    {
      "type": "sphere",
      "center": [0, 0, -1],
      "radius": 0.5,
      "material": "glass"
    },
    {
      "type": "mesh",
      "file": "models/bunny.obj",
      "transform": {
        "translate": [1, 0, -1],
        "scale": [0.5, 0.5, 0.5]
      },
      "material": "metal_gold"
    }
  ],
  "environment": {
    "type": "gradient",
    "top": [0.5, 0.7, 1.0],
    "bottom": [1.0, 1.0, 1.0]
  }
}
```

### Scene Parser

```cpp
// engine/scene_parser.hpp
class SceneParser {
public:
    static std::optional<SceneData> parse(const std::string& filename);
    
    struct SceneData {
        Renderer::Settings settings;
        Camera camera;
        Scene scene;
    };
    
private:
    static Material parse_material(const json& j);
    static Sphere parse_sphere(const json& j, const MaterialMap& materials);
    static Mesh parse_mesh(const json& j, const MaterialMap& materials);
    static Camera parse_camera(const json& j, double aspect_ratio);
};
```

---

## Command Line Interface

### CLI Options

```cpp
// engine/cli.hpp
struct CLIOptions {
    std::string scene_file;
    std::string output_file = "output.png";
    int width = 0;           // 0 = use scene default
    int height = 0;
    int samples = 0;
    int max_depth = 0;
    int threads = 0;         // 0 = auto
    bool quiet = false;
    bool verbose = false;
    bool help = false;
    bool version = false;
    
    static CLIOptions parse(int argc, char* argv[]);
    static void print_help();
};
```

### Usage Examples

```bash
# Basic render
./raytracer scene.json

# Custom output
./raytracer scene.json -o render.png

# Override settings
./raytracer scene.json --width 1920 --height 1080 --samples 128

# Quick preview
./raytracer scene.json --samples 4 --width 640 --height 360

# Verbose output
./raytracer scene.json -v
```

### Implementation with argparse or CLI11

```cpp
#include <CLI/CLI.hpp>

int main(int argc, char* argv[]) {
    CLI::App app{"Whitted-style Ray Tracer"};
    
    CLIOptions opts;
    
    app.add_option("scene", opts.scene_file, "Scene file (JSON)")
        ->required()
        ->check(CLI::ExistingFile);
    
    app.add_option("-o,--output", opts.output_file, "Output file");
    app.add_option("-w,--width", opts.width, "Image width");
    app.add_option("-h,--height", opts.height, "Image height");
    app.add_option("-s,--samples", opts.samples, "Samples per pixel");
    app.add_option("-d,--depth", opts.max_depth, "Max ray depth");
    app.add_option("-t,--threads", opts.threads, "Number of threads");
    app.add_flag("-q,--quiet", opts.quiet, "Suppress output");
    app.add_flag("-v,--verbose", opts.verbose, "Verbose output");
    
    CLI11_PARSE(app, argc, argv);
    
    // Run render with options...
}
```

---

## Interactive Preview

### Real-Time Preview Window

```cpp
// engine/preview.hpp
class PreviewWindow {
public:
    PreviewWindow(int width, int height, const std::string& title);
    ~PreviewWindow();
    
    void update(const Image& image);
    bool should_close() const;
    void poll_events();
    
    // Callbacks
    void on_key(std::function<void(int key)> callback);
    void on_mouse(std::function<void(double x, double y)> callback);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Usage
int main() {
    PreviewWindow window(800, 600, "Ray Tracer Preview");
    ProgressiveRenderer renderer;
    
    renderer.start(scene, camera);
    
    while (!window.should_close()) {
        window.poll_events();
        renderer.render_pass();
        window.update(renderer.current());
    }
}
```

### GUI Library Options

| Library | Pros | Cons |
|---------|------|------|
| **GLFW + OpenGL** | Lightweight, cross-platform | Requires GL setup |
| **SDL2** | Simple, well-documented | Larger dependency |
| **Dear ImGui** | Great for debug UI | Needs backend |
| **Qt** | Full-featured | Heavy dependency |

---

## External Integrations

### Blender Exporter

Python add-on to export Blender scenes:

```python
# blender_addon/export_raytracer.py
bl_info = {
    "name": "Ray Tracer Exporter",
    "blender": (3, 0, 0),
    "category": "Import-Export",
}

class ExportRaytracer(bpy.types.Operator):
    bl_idname = "export_scene.raytracer"
    bl_label = "Export to Ray Tracer"
    
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    
    def execute(self, context):
        scene_data = {
            "camera": export_camera(context.scene.camera),
            "materials": export_materials(bpy.data.materials),
            "objects": export_objects(context.scene.objects),
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(scene_data, f, indent=2)
        
        return {'FINISHED'}
```

### USD (Universal Scene Description) Support

```cpp
// engine/usd_reader.hpp
#include <pxr/usd/usd/stage.h>

class USDReader {
public:
    static std::optional<Scene> load(const std::string& filename);
    
private:
    static void import_mesh(const pxr::UsdGeomMesh& usdMesh, Scene& scene);
    static void import_sphere(const pxr::UsdGeomSphere& usdSphere, Scene& scene);
    static Material import_material(const pxr::UsdShadeMaterial& usdMat);
};
```

### MaterialX Support

```cpp
// engine/materialx_reader.hpp
#include <MaterialXCore/Document.h>

class MaterialXReader {
public:
    static Material load(const std::string& filename);
    
private:
    static Material convert_standard_surface(const mx::Node& node);
};
```

---

## Documentation

### API Documentation (Doxygen)

```cmake
# CMakeLists.txt
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_MAN NO)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
    
    doxygen_add_docs(docs
        ${CMAKE_SOURCE_DIR}/core
        ${CMAKE_SOURCE_DIR}/engine
        COMMENT "Generate API documentation"
    )
endif()
```

### Doxyfile Configuration

```ini
# Doxyfile
PROJECT_NAME           = "Ray Tracer"
PROJECT_BRIEF          = "Whitted-style Ray Tracer Engine"
INPUT                  = core engine
RECURSIVE              = YES
EXTRACT_ALL            = YES
GENERATE_TREEVIEW      = YES
HAVE_DOT               = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
```

### User Guide Structure

```
docs/
├── index.md                    # Overview
├── getting-started.md          # Quick start guide
├── building.md                 # Build instructions
├── scene-format.md             # Scene file format reference
├── materials.md                # Material reference
├── cli-reference.md            # Command line options
├── api/                        # Generated API docs
└── tutorials/
    ├── first-render.md
    ├── custom-materials.md
    └── importing-meshes.md
```

---

## Implementation Priority

| Priority | Feature | Impact | Effort |
|----------|---------|--------|--------|
| 🔴 High | Unit tests | Reliability | Medium |
| 🔴 High | JSON scene format | Usability | Medium |
| 🔴 High | CLI options | Usability | Low |
| 🟡 Medium | Benchmarks | Optimization | Low |
| 🟡 Medium | Debug visualization | Development | Low |
| 🟡 Medium | Doxygen docs | Maintainability | Low |
| 🟢 Low | Preview window | UX | Medium |
| 🟢 Low | Blender exporter | Workflow | Medium |
| 🟢 Low | USD support | Interoperability | High |

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        build_type: [Debug, Release]
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure CMake
        run: cmake -B build -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} -DBUILD_TESTS=ON
      
      - name: Build
        run: cmake --build build --config ${{ matrix.build_type }}
      
      - name: Test
        run: ctest --test-dir build -C ${{ matrix.build_type }} --output-on-failure
      
      - name: Render test scene
        run: ./build/raytracer --samples 4 --width 320 --height 180

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure with coverage
        run: cmake -B build -DENABLE_COVERAGE=ON -DBUILD_TESTS=ON
      - name: Build and test
        run: cmake --build build && ctest --test-dir build
      - name: Generate coverage report
        run: lcov --capture --directory build --output-file coverage.info
      - uses: codecov/codecov-action@v3
```

---

*See also: [Architecture Improvements](ARCHITECTURE_IMPROVEMENTS.md) for structural changes needed to support these features*
