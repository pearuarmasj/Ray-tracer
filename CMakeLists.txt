# CMakeList.txt : CMake project for Ray tracer
# Mixed C23/C++23 Ray tracer
cmake_minimum_required(VERSION 3.21)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("Ray tracer" LANGUAGES C CXX)

# Language standards
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# OpenMP for parallel rendering
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(raytracer PRIVATE OpenMP::OpenMP_CXX)
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
