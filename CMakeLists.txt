# CMakeList.txt : CMake project for Ray tracer
# Mixed C23/C++23 Ray tracer with optional CUDA support
cmake_minimum_required(VERSION 3.21)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# CUDA support option
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

if(ENABLE_CUDA)
    # Set CUDA paths for Windows - CUDA Toolkit 13.1
    set(CUDA_TOOLKIT_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
    set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_PATH}")
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_PATH}/bin/nvcc.exe")
    
    # Force environment variables for MSBuild
    set(ENV{CUDA_PATH} "${CUDA_TOOLKIT_PATH}")
    set(ENV{CudaToolkitDir} "${CUDA_TOOLKIT_PATH}")
    
    # CUDA architecture - adjust for your GPU (e.g., 86 for RTX 30xx, 89 for RTX 40xx)
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")
    
    # Allow unsupported host compiler (VS 2026 not officially supported by CUDA 13.1 yet)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    
    # Pass CudaToolkitDir to MSBuild via CMAKE_CUDA_HOST_FLAGS
    set(CMAKE_CUDA_HOST_FLAGS "${CMAKE_CUDA_HOST_FLAGS} /p:CudaToolkitDir=\"${CUDA_TOOLKIT_PATH}\"")
endif()

project("Ray tracer" LANGUAGES C CXX)

if(ENABLE_CUDA)
    # Enable CUDA language after project() to use detected host compiler
    enable_language(CUDA)
    
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    
    message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA disabled - CPU-only build")
endif()

# Language standards - set globally AND enforce on targets
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Profiling option
option(ENABLE_PROFILING "Enable performance profiling instrumentation" ON)

# C Core library (vec3, ray, hit)
set(CORE_SOURCES
    core/vec3.c
    core/ray.c
    core/hit.c
)

set(CORE_HEADERS
    core/vec3.h
    core/ray.h
    core/hit.h
)

add_library(raytracer_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(raytracer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Explicitly set C23 on core library (MSVC needs this per-target)
set_target_properties(raytracer_core PROPERTIES
    C_STANDARD 23
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)

# C++ Engine (renderer, scene, materials)
set(ENGINE_SOURCES
    engine/renderer.cpp
    engine/bdpt.cpp
    engine/stb_impl.cpp
    engine/main.cpp
)

set(ENGINE_HEADERS
    engine/renderer.hpp
    engine/bdpt.hpp
    engine/scene.hpp
    engine/sphere.hpp
    engine/material.hpp
)

add_executable(raytracer ${ENGINE_SOURCES} ${ENGINE_HEADERS})
target_link_libraries(raytracer PRIVATE raytracer_core)
target_include_directories(raytracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Profiling instrumentation
if(ENABLE_PROFILING)
    target_compile_definitions(raytracer PRIVATE ENABLE_PROFILING=1)
    message(STATUS "Profiling enabled")
endif()

# Explicitly set C++23 on raytracer (MSVC needs this per-target)
set_target_properties(raytracer PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# OpenMP for parallel rendering (only need C/CXX, not CUDA)
find_package(OpenMP COMPONENTS C CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(raytracer PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for parallel rendering")
endif()

# Enable math library on Unix
if(UNIX)
    target_link_libraries(raytracer_core PRIVATE m)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(raytracer_core PRIVATE /W4)
    target_compile_options(raytracer PRIVATE /W4)
else()
    target_compile_options(raytracer_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(raytracer PRIVATE -Wall -Wextra -Wpedantic)
endif()

# PLT Module Tests (optional)
option(BUILD_PLT_TESTS "Build PLT module tests" OFF)
if(BUILD_PLT_TESTS)
    add_executable(plt_test engine/plt/tests/plt_test.cpp)
    target_include_directories(plt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(plt_test PRIVATE raytracer_core)
    set_target_properties(plt_test PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

# CUDA acceleration library (when enabled)
if(ENABLE_CUDA)
    set(CUDA_SOURCES
        engine/cuda/cuda_test.cu
        engine/cuda/cuda_pathtracer.cu
    )
    
    add_library(raytracer_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(raytracer_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(raytracer_cuda PRIVATE CUDA::cudart CUDA::curand)
    set_target_properties(raytracer_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Link CUDA library to main raytracer
    target_compile_definitions(raytracer PRIVATE CUDA_ENABLED=1)
    target_link_libraries(raytracer PRIVATE raytracer_cuda CUDA::cudart CUDA::curand)
    
    message(STATUS "CUDA path tracer configured")
endif()
