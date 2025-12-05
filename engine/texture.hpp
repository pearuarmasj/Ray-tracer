/**
 * @file texture.hpp
 * @brief Texture system for procedural and image-based textures
 */

#pragma once

extern "C" {
#include "../core/vec3.h"
}

#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>

// Forward declare stb_image functions (implemented in stb_impl.cpp)
extern "C" {
    unsigned char* stbi_load(const char* filename, int* x, int* y, int* comp, int req_comp);
    void stbi_image_free(void* retval_from_stbi_load);
}

namespace raytracer {

/**
 * @brief Texture types
 */
enum class TextureType {
    Solid,      // Constant color
    Checker,    // 3D checker pattern
    Image       // Image texture (requires UV coordinates)
};

/**
 * @brief Texture class supporting solid, checker, and image textures
 */
struct Texture {
    TextureType type = TextureType::Solid;
    color color1 = {1.0, 1.0, 1.0};  // Primary color (or solid color)
    color color2 = {0.0, 0.0, 0.0};  // Secondary color for patterns
    double scale = 1.0;               // Pattern scale
    
    // Image texture data
    std::shared_ptr<std::vector<unsigned char>> image_data;
    int image_width = 0;
    int image_height = 0;
    int image_channels = 0;
    
    /**
     * @brief Create a solid color texture
     */
    static Texture solid(color c) {
        Texture tex;
        tex.type = TextureType::Solid;
        tex.color1 = c;
        return tex;
    }
    
    /**
     * @brief Create a checker pattern texture
     * @param c1 First color
     * @param c2 Second color
     * @param scale Pattern scale (smaller = larger squares)
     */
    static Texture checker(color c1, color c2, double scale = 10.0) {
        Texture tex;
        tex.type = TextureType::Checker;
        tex.color1 = c1;
        tex.color2 = c2;
        tex.scale = scale;
        return tex;
    }
    
    /**
     * @brief Create an image texture from raw data
     */
    static Texture image(const unsigned char* data, int width, int height, int channels) {
        Texture tex;
        tex.type = TextureType::Image;
        tex.image_width = width;
        tex.image_height = height;
        tex.image_channels = channels;
        tex.image_data = std::make_shared<std::vector<unsigned char>>(
            data, data + width * height * channels
        );
        return tex;
    }
    
    /**
     * @brief Load an image texture from file (PNG, JPG, etc.)
     * @param filename Path to image file
     * @return Texture, or magenta solid if loading fails
     */
    static Texture load_image(const std::string& filename) {
        int width, height, channels;
        unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
        
        if (!data) {
            std::cerr << "Error: Could not load texture: " << filename << std::endl;
            return solid({1.0, 0.0, 1.0});  // Magenta for error
        }
        
        Texture tex = image(data, width, height, 3);
        stbi_image_free(data);
        
        std::cout << "Loaded texture: " << filename << " (" << width << "x" << height << ")" << std::endl;
        return tex;
    }
    
    /**
     * @brief Sample the texture at a given 3D point
     * @param p World position
     * @param u Optional U coordinate for image textures
     * @param v Optional V coordinate for image textures
     */
    color sample(point3 p, double u = 0.0, double v = 0.0) const {
        switch (type) {
            case TextureType::Solid:
                return color1;
                
            case TextureType::Checker: {
                // 3D checker pattern based on world coordinates
                double inv_scale = 1.0 / scale;
                int x = static_cast<int>(std::floor(p.x * inv_scale));
                int y = static_cast<int>(std::floor(p.y * inv_scale));
                int z = static_cast<int>(std::floor(p.z * inv_scale));
                
                bool is_even = ((x + y + z) % 2) == 0;
                return is_even ? color1 : color2;
            }
                
            case TextureType::Image: {
                if (!image_data || image_width == 0 || image_height == 0) {
                    return {1.0, 0.0, 1.0};  // Magenta for missing texture
                }
                
                // Clamp UV coordinates
                u = std::clamp(u, 0.0, 1.0);
                v = 1.0 - std::clamp(v, 0.0, 1.0);  // Flip V for image coordinates
                
                int i = static_cast<int>(u * image_width);
                int j = static_cast<int>(v * image_height);
                
                // Clamp to valid range
                i = std::min(i, image_width - 1);
                j = std::min(j, image_height - 1);
                
                int idx = (j * image_width + i) * image_channels;
                double r = (*image_data)[idx] / 255.0;
                double g = (*image_data)[idx + 1] / 255.0;
                double b = (*image_data)[idx + 2] / 255.0;
                
                return {r, g, b};
            }
        }
        
        return color1;
    }
};

/**
 * @brief Calculate UV coordinates for a sphere
 * @param p Point on unit sphere (normalized direction from center)
 * @param u Output U coordinate [0, 1]
 * @param v Output V coordinate [0, 1]
 */
inline void get_sphere_uv(point3 p, double& u, double& v) {
    // p is assumed to be a unit vector
    double theta = std::acos(-p.y);
    double phi = std::atan2(-p.z, p.x) + 3.14159265358979323846;
    
    u = phi / (2.0 * 3.14159265358979323846);
    v = theta / 3.14159265358979323846;
}

} // namespace raytracer
