# Ray Tracer

A **Whitted-style ray tracer** built with a C core layer (C23) and C++ engine layer (C++23).

## Features

- **C Core Layer**: Low-level math primitives (vec3, ray, hit records) written in C23
- **C++ Engine Layer**: Scene management, materials, and rendering in C++23
- **Whitted-style Rendering**: Recursive ray tracing with reflection and refraction
- **Materials**: Lambertian (diffuse), Metal (reflective), and Dielectric (glass) materials
- **PPM Image Output**: Portable image format for immediate viewing

## Project Structure

```
Ray-tracer/
├── core/           # C core layer (C23)
│   ├── vec3.h/c    # 3D vector mathematics
│   ├── ray.h/c     # Ray structure and operations
│   └── hit.h/c     # Hit record for intersections
├── engine/         # C++ engine layer (C++23)
│   ├── material.hpp    # Material definitions
│   ├── sphere.hpp      # Sphere geometry
│   ├── scene.hpp       # Scene and camera management
│   ├── renderer.hpp/cpp # Whitted-style renderer
│   └── main.cpp        # Entry point
├── CMakeLists.txt  # Build configuration
└── CMakePresets.json # Build presets
```

## Building

### Prerequisites

- CMake 3.21 or higher
- C compiler with C23 support (GCC 13+, Clang 18+, MSVC 2022+)
- C++ compiler with C++23 support

### Linux/macOS

```bash
# Create build directory
mkdir -p build && cd build

# Configure
cmake ..

# Build
cmake --build .

# Run
./raytracer output.ppm
```

### Windows (Visual Studio)

```bash
# Using CMake presets
cmake --preset x64-debug
cmake --build out/build/x64-debug

# Run
.\out\build\x64-debug\raytracer.exe output.ppm
```

Or open the folder in Visual Studio and use the built-in CMake support.

## Usage

```bash
# Render to default output.ppm
./raytracer

# Render to custom filename
./raytracer my_render.ppm
```

The rendered image will be in PPM format. You can view PPM files with:
- **Linux**: `feh`, `eog`, `gimp`
- **macOS**: Preview.app, `open output.ppm`
- **Windows**: IrfanView, GIMP
- **Online**: [PPM Viewer](https://www.cs.rhodes.edu/welshc/COMP141_F16/ppmReader.html)

## Demo Scene

The default demo scene includes:
- A large yellow ground sphere
- A blue diffuse sphere (center)
- A glass sphere (left)
- A gold metallic sphere (right)

## Technical Details

### Ray Tracing Algorithm

This implements Whitted-style ray tracing:
1. Cast primary rays from camera through each pixel
2. Find closest intersection with scene objects
3. Based on material:
   - **Lambertian**: Scatter in hemisphere, attenuate by albedo
   - **Metal**: Perfect specular reflection
   - **Dielectric**: Refraction with Fresnel (Schlick approximation)
4. Recursively trace scattered rays up to max depth
5. Accumulate colors and apply gamma correction

### Color Pipeline

- Colors are computed in linear space
- Gamma correction (γ=2.0) applied on output
- HDR values clamped to [0, 1] range

## License

MIT License