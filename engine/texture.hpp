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
    float* stbi_loadf(const char* filename, int* x, int* y, int* comp, int req_comp);
    int stbi_is_hdr(const char* filename);
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

// Forward declare random function
double random_double();

/**
 * @brief Environment sample result for importance sampling
 */
struct EnvSample {
    vec3 direction;      // Sampled direction (world space)
    color emission;      // Environment radiance in that direction
    double pdf;          // Probability density of this sample
    bool valid;          // Whether sample is valid
};

/**
 * @brief Environment map for HDR image-based lighting with importance sampling
 * 
 * Supports equirectangular HDR images (.hdr format) for realistic
 * sky lighting in path tracing. The environment map replaces the
 * default gradient background when present.
 * 
 * Importance sampling uses a 2D CDF built from luminance values:
 * - marginal_cdf: 1D CDF over rows (Y), size = height+1
 * - conditional_cdf: 2D CDF over columns (X) per row, size = height * (width+1)
 */
struct EnvironmentMap {
    std::shared_ptr<std::vector<float>> hdr_data;
    int width = 0;
    int height = 0;
    double intensity = 1.0;      // Brightness multiplier
    double rotation = 0.0;       // Y-axis rotation in radians
    
    // Importance sampling CDFs
    std::vector<double> marginal_cdf;       // CDF over rows (Y), size = height+1
    std::vector<double> conditional_cdf;    // CDF over columns per row, size = height * (width+1)
    double total_luminance = 0.0;           // Sum of all luminance * sin(theta) weights
    
    /**
     * @brief Load an HDR environment map from file
     * @param filename Path to .hdr file
     * @param intensity Brightness multiplier (default 1.0)
     * @param rotation_degrees Y-axis rotation in degrees (default 0.0)
     * @return EnvironmentMap, or empty if loading fails
     */
    static EnvironmentMap load(const std::string& filename, 
                               double intensity = 1.0, 
                               double rotation_degrees = 0.0) {
        EnvironmentMap env;
        
        int w, h, channels;
        float* data = stbi_loadf(filename.c_str(), &w, &h, &channels, 3);
        
        if (!data) {
            std::cerr << "Error: Could not load environment map: " << filename << std::endl;
            return env;
        }
        
        env.width = w;
        env.height = h;
        env.intensity = intensity;
        env.rotation = rotation_degrees * 3.14159265358979323846 / 180.0;
        env.hdr_data = std::make_shared<std::vector<float>>(data, data + w * h * 3);
        
        stbi_image_free(data);
        
        // Build importance sampling CDFs
        env.build_sampling_cdf();
        
        std::cout << "Loaded environment map: " << filename 
                  << " (" << w << "x" << h << ", total luminance: " << env.total_luminance << ")" << std::endl;
        return env;
    }
    
    /**
     * @brief Build CDFs for importance sampling from luminance values
     */
    void build_sampling_cdf() {
        if (!hdr_data || width <= 0 || height <= 0) return;
        
        // Using PI from raytracer namespace (defined in sphere.hpp)
        
        // Allocate CDF arrays
        marginal_cdf.resize(height + 1);
        conditional_cdf.resize(height * (width + 1));
        
        // Build conditional CDFs (per row) and accumulate row totals for marginal
        std::vector<double> row_integrals(height);
        
        for (int y = 0; y < height; ++y) {
            // sin(theta) weight for equirectangular projection
            // theta = PI * (y + 0.5) / height, from top (theta=0) to bottom (theta=PI)
            double theta = PI * (y + 0.5) / height;
            double sin_theta = std::sin(theta);
            
            double row_sum = 0.0;
            int cdf_row_offset = y * (width + 1);
            conditional_cdf[cdf_row_offset] = 0.0;  // CDF starts at 0
            
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                float r = (*hdr_data)[idx];
                float g = (*hdr_data)[idx + 1];
                float b = (*hdr_data)[idx + 2];
                
                // Luminance with sin(theta) weighting for solid angle
                double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                double weighted_lum = luminance * sin_theta;
                
                row_sum += weighted_lum;
                conditional_cdf[cdf_row_offset + x + 1] = row_sum;
            }
            
            // Normalize conditional CDF for this row
            if (row_sum > 0.0) {
                for (int x = 1; x <= width; ++x) {
                    conditional_cdf[cdf_row_offset + x] /= row_sum;
                }
            } else {
                // Uniform distribution if row has no luminance
                for (int x = 1; x <= width; ++x) {
                    conditional_cdf[cdf_row_offset + x] = static_cast<double>(x) / width;
                }
            }
            
            row_integrals[y] = row_sum;
        }
        
        // Build marginal CDF from row integrals
        marginal_cdf[0] = 0.0;
        for (int y = 0; y < height; ++y) {
            marginal_cdf[y + 1] = marginal_cdf[y] + row_integrals[y];
        }
        
        total_luminance = marginal_cdf[height];
        
        // Normalize marginal CDF
        if (total_luminance > 0.0) {
            for (int y = 1; y <= height; ++y) {
                marginal_cdf[y] /= total_luminance;
            }
        } else {
            // Uniform distribution if no luminance
            for (int y = 1; y <= height; ++y) {
                marginal_cdf[y] = static_cast<double>(y) / height;
            }
        }
    }
    
    /**
     * @brief Check if environment map is valid/loaded
     */
    bool valid() const {
        return hdr_data && width > 0 && height > 0;
    }
    
    /**
     * @brief Sample the environment map for a given ray direction
     * @param direction Ray direction (will be normalized)
     * @return HDR color value scaled by intensity
     */
    color sample(vec3 direction) const {
        if (!valid()) {
            return {0.0, 0.0, 0.0};
        }
        
        vec3 d = vec3_normalize(direction);
        
        // Apply Y-axis rotation
        if (rotation != 0.0) {
            double cos_r = std::cos(rotation);
            double sin_r = std::sin(rotation);
            double new_x = d.x * cos_r - d.z * sin_r;
            double new_z = d.x * sin_r + d.z * cos_r;
            d.x = new_x;
            d.z = new_z;
        }
        
        // Equirectangular mapping: direction -> UV
        // phi: azimuth angle around Y axis [-PI, PI] -> U [0, 1]
        // theta: elevation from -Y to +Y [-PI/2, PI/2] -> V [0, 1]
        double phi = std::atan2(d.z, d.x);           // [-PI, PI]
        double theta = std::asin(std::clamp(d.y, -1.0, 1.0));  // [-PI/2, PI/2]
        
        double u = 0.5 + phi / (2.0 * PI);           // [0, 1]
        double v = 0.5 + theta / PI;                  // [0, 1]
        
        // Bilinear interpolation for smoother sampling
        double fx = u * width - 0.5;
        double fy = (1.0 - v) * height - 0.5;  // Flip V for image coords
        
        int x0 = static_cast<int>(std::floor(fx));
        int y0 = static_cast<int>(std::floor(fy));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        double tx = fx - x0;
        double ty = fy - y0;
        
        // Wrap horizontally, clamp vertically
        x0 = ((x0 % width) + width) % width;
        x1 = ((x1 % width) + width) % width;
        y0 = std::clamp(y0, 0, height - 1);
        y1 = std::clamp(y1, 0, height - 1);
        
        auto get_pixel = [this](int x, int y) -> color {
            int idx = (y * width + x) * 3;
            return {
                (*hdr_data)[idx] * intensity,
                (*hdr_data)[idx + 1] * intensity,
                (*hdr_data)[idx + 2] * intensity
            };
        };
        
        color c00 = get_pixel(x0, y0);
        color c10 = get_pixel(x1, y0);
        color c01 = get_pixel(x0, y1);
        color c11 = get_pixel(x1, y1);
        
        // Bilinear interpolation
        color c0 = vec3_lerp(c00, c10, tx);
        color c1 = vec3_lerp(c01, c11, tx);
        return vec3_lerp(c0, c1, ty);
    }
    
    /**
     * @brief Sample a direction from the environment map using importance sampling
     * @return EnvSample with direction, emission, and PDF
     */
    EnvSample sample_direction() const {
        EnvSample result;
        result.valid = false;
        
        if (!valid() || marginal_cdf.empty() || conditional_cdf.empty()) {
            return result;
        }
        
        // Two random numbers for 2D sampling
        double u1 = random_double();
        double u2 = random_double();
        
        // Sample row (Y) from marginal CDF using binary search
        auto row_it = std::upper_bound(marginal_cdf.begin(), marginal_cdf.end(), u1);
        int y = static_cast<int>(row_it - marginal_cdf.begin()) - 1;
        y = std::clamp(y, 0, height - 1);
        
        // Interpolate within the selected row for continuous sampling
        double marginal_below = marginal_cdf[y];
        double marginal_above = marginal_cdf[y + 1];
        double row_t = (marginal_above > marginal_below) 
            ? (u1 - marginal_below) / (marginal_above - marginal_below) 
            : 0.0;
        
        // Sample column (X) from conditional CDF for this row
        int cdf_row_offset = y * (width + 1);
        auto col_begin = conditional_cdf.begin() + cdf_row_offset;
        auto col_end = col_begin + width + 1;
        auto col_it = std::upper_bound(col_begin, col_end, u2);
        int x = static_cast<int>(col_it - col_begin) - 1;
        x = std::clamp(x, 0, width - 1);
        
        // Interpolate within pixel for continuous sampling
        double cond_below = conditional_cdf[cdf_row_offset + x];
        double cond_above = conditional_cdf[cdf_row_offset + x + 1];
        double col_t = (cond_above > cond_below) 
            ? (u2 - cond_below) / (cond_above - cond_below) 
            : 0.0;
        
        // Convert pixel coordinates to UV, then to direction
        double u = (x + col_t + 0.5) / width;
        double v = (y + row_t + 0.5) / height;
        
        // UV to spherical angles
        double phi = (u - 0.5) * 2.0 * PI;    // [-PI, PI]
        double theta = v * PI;                 // [0, PI] from top to bottom
        
        // Spherical to Cartesian
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        vec3 dir = {
            sin_theta * std::cos(phi),
            cos_theta,
            sin_theta * std::sin(phi)
        };
        
        // Apply inverse rotation (sample is in image space, need world space)
        if (rotation != 0.0) {
            double cos_r = std::cos(-rotation);
            double sin_r = std::sin(-rotation);
            double new_x = dir.x * cos_r - dir.z * sin_r;
            double new_z = dir.x * sin_r + dir.z * cos_r;
            dir.x = new_x;
            dir.z = new_z;
        }
        
        result.direction = dir;
        result.emission = sample(dir);  // Get actual color (handles rotation internally)
        result.pdf = pdf(dir);
        result.valid = result.pdf > 0.0;
        
        return result;
    }
    
    /**
     * @brief Compute the PDF for sampling a given direction
     * @param direction World-space direction
     * @return Probability density per solid angle
     */
    double pdf(vec3 direction) const {
        if (!valid() || marginal_cdf.empty() || conditional_cdf.empty() || total_luminance <= 0.0) {
            return 0.0;
        }
        
        vec3 d = vec3_normalize(direction);
        
        // Apply rotation to get image-space direction
        if (rotation != 0.0) {
            double cos_r = std::cos(rotation);
            double sin_r = std::sin(rotation);
            double new_x = d.x * cos_r - d.z * sin_r;
            double new_z = d.x * sin_r + d.z * cos_r;
            d.x = new_x;
            d.z = new_z;
        }
        
        // Direction to UV (same as in sample())
        double phi = std::atan2(d.z, d.x);                          // [-PI, PI]
        double theta = std::acos(std::clamp(d.y, -1.0, 1.0));       // [0, PI]
        
        double u = 0.5 + phi / (2.0 * PI);    // [0, 1]
        double v = theta / PI;                 // [0, 1]
        
        // UV to pixel coordinates
        int x = static_cast<int>(u * width);
        int y = static_cast<int>(v * height);
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        
        // Get PDF from CDF derivatives
        // Marginal PDF for row y
        double marginal_pdf = (marginal_cdf[y + 1] - marginal_cdf[y]) * height;
        
        // Conditional PDF for column x in row y
        int cdf_row_offset = y * (width + 1);
        double conditional_pdf = (conditional_cdf[cdf_row_offset + x + 1] - conditional_cdf[cdf_row_offset + x]) * width;
        
        // Combined PDF in UV space
        double pdf_uv = marginal_pdf * conditional_pdf;
        
        // Convert from UV space to solid angle
        // Jacobian: d(solid_angle) = sin(theta) * d(theta) * d(phi)
        // UV covers [0,1]x[0,1] mapping to theta=[0,PI], phi=[-PI,PI]
        // So: pdf_solid_angle = pdf_uv / (2 * PI * PI * sin(theta))
        double sin_theta = std::sin(theta);
        if (sin_theta < 1e-10) {
            return 0.0;  // Avoid division by zero at poles
        }
        
        double pdf_solid_angle = pdf_uv / (2.0 * PI * PI * sin_theta);
        
        return pdf_solid_angle;
    }
    
    /**
     * @brief Check if importance sampling is available
     */
    bool has_importance_sampling() const {
        return valid() && !marginal_cdf.empty() && total_luminance > 0.0;
    }
};

} // namespace raytracer
