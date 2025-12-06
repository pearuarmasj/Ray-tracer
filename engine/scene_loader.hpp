/**
 * @file scene_loader.hpp
 * @brief JSON scene file loading
 */

#pragma once

#include "json.hpp"
#include "scene.hpp"
#include "material.hpp"
#include "texture.hpp"
#include "renderer.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace raytracer {

using json = nlohmann::json;

/**
 * @brief Load a scene from a JSON file
 */
class SceneLoader {
public:
    struct SceneData {
        Scene scene;
        
        // Camera settings
        point3 camera_position = {0, 1, 5};
        point3 camera_target = {0, 0, 0};
        vec3 camera_up = {0, 1, 0};
        double camera_fov = 60.0;
        
        // Render settings
        int width = 1920;
        int height = 1080;
        int samples = 16;
        int max_depth = 50;
        RenderMode mode = RenderMode::Whitted;
        bool use_nee = true;
        bool use_mis = true;
        ToneMapper tone_mapper = ToneMapper::ACES;
        double exposure = 1.0;
        double clamp_max = 10.0;  // Firefly clamping
        int wavelength_samples = 8;  // For spectral rendering
        
        std::string output_file = "output.png";
    };
    
    /**
     * @brief Parse a texture from JSON
     */
    static Texture parse_texture(const json& tex) {
        std::string type = tex.value("type", "solid");
        
        if (type == "checker") {
            auto c1 = tex.value("color1", std::vector<double>{1.0, 1.0, 1.0});
            auto c2 = tex.value("color2", std::vector<double>{0.0, 0.0, 0.0});
            double scale = tex.value("scale", 10.0);
            return Texture::checker({c1[0], c1[1], c1[2]}, {c2[0], c2[1], c2[2]}, scale);
        }
        else if (type == "image") {
            std::string filename = tex.value("file", "");
            if (filename.empty()) {
                std::cerr << "Error: Image texture missing 'file' property" << std::endl;
                return Texture::solid({1.0, 0.0, 1.0});  // Magenta for error
            }
            return Texture::load_image(filename);
        }
        else {
            // Solid color
            auto c = tex.value("color", std::vector<double>{0.5, 0.5, 0.5});
            return Texture::solid({c[0], c[1], c[2]});
        }
    }
    
    /**
     * @brief Load scene from JSON file
     * @param filename Path to JSON file
     * @return SceneData with scene and settings, or empty on failure
     */
    static SceneData load(const std::string& filename) {
        SceneData data;
        
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open scene file: " << filename << std::endl;
            return data;
        }
        
        json j;
        try {
            file >> j;
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
            return data;
        }
        
        // Material name -> ID mapping
        std::unordered_map<std::string, int> material_ids;
        
        // Load materials
        if (j.contains("materials")) {
            for (auto& [name, mat] : j["materials"].items()) {
                Material m;
                std::string type = mat.value("type", "lambertian");
                
                // Check for texture
                bool has_texture = mat.contains("texture");
                
                if (type == "lambertian") {
                    if (has_texture) {
                        m = Material::lambertian_textured(parse_texture(mat["texture"]));
                    } else {
                        auto c = mat.value("color", std::vector<double>{0.5, 0.5, 0.5});
                        m = Material::lambertian({c[0], c[1], c[2]});
                    }
                }
                else if (type == "metal") {
                    double fuzz = mat.value("fuzz", 0.0);
                    if (has_texture) {
                        m = Material::metal_textured(parse_texture(mat["texture"]), fuzz);
                    } else {
                        auto c = mat.value("color", std::vector<double>{0.8, 0.8, 0.8});
                        m = Material::metal({c[0], c[1], c[2]}, fuzz);
                    }
                }
                else if (type == "dielectric" || type == "glass") {
                    double ior = mat.value("ior", 1.5);
                    m = Material::dielectric(ior);
                }
                else if (type == "emissive") {
                    auto c = mat.value("color", std::vector<double>{1.0, 1.0, 1.0});
                    auto e = mat.value("emission", std::vector<double>{1.0, 1.0, 1.0});
                    // Also support "emit" as alias for "emission"
                    if (mat.contains("emit")) {
                        auto emit_val = mat["emit"];
                        if (emit_val.is_array() && emit_val.size() >= 3) {
                            e = emit_val.get<std::vector<double>>();
                        }
                    }
                    double intensity = mat.value("intensity", 1.0);
                    m = Material::emissive(
                        {e[0] * intensity, e[1] * intensity, e[2] * intensity},
                        {c[0], c[1], c[2]}
                    );
                }
                
                // Check for normal map
                if (mat.contains("normal_map")) {
                    std::string nmap_file = mat["normal_map"].value("file", "");
                    double strength = mat["normal_map"].value("strength", 1.0);
                    if (!nmap_file.empty()) {
                        m.with_normal_map(NormalMap::load(nmap_file, strength));
                    }
                }
                
                // Check for roughness map
                if (mat.contains("roughness_map")) {
                    std::string rmap_file = mat["roughness_map"].value("file", "");
                    if (!rmap_file.empty()) {
                        m.with_roughness_map(RoughnessMap::load(rmap_file));
                    }
                }
                
                // Check for thin-film coating
                if (mat.contains("thin_film")) {
                    auto film = mat["thin_film"];
                    double thickness = film.value("thickness", 300.0);  // nm
                    double film_ior = film.value("ior", 1.4);           // MgF2 default
                    m.has_thin_film = true;
                    m.thin_film_thickness = thickness;
                    m.thin_film_ior = film_ior;
                }
                
                int id = data.scene.add_material(m);
                material_ids[name] = id;
            }
        }
        
        // Helper to get material ID by name
        auto get_material = [&](const json& obj) -> int {
            if (obj.contains("material")) {
                std::string name = obj["material"];
                if (material_ids.count(name)) {
                    return material_ids[name];
                }
                std::cerr << "Warning: Unknown material '" << name << "'" << std::endl;
            }
            return 0;
        };
        
        // Load objects
        if (j.contains("objects")) {
            for (auto& obj : j["objects"]) {
                std::string type = obj.value("type", "");
                int mat_id = get_material(obj);
                
                if (type == "sphere") {
                    auto pos = obj.value("position", std::vector<double>{0, 0, 0});
                    double radius = obj.value("radius", 1.0);
                    data.scene.add_sphere({pos[0], pos[1], pos[2]}, radius, mat_id);
                }
                else if (type == "plane") {
                    auto pos = obj.value("position", std::vector<double>{0, 0, 0});
                    auto norm = obj.value("normal", std::vector<double>{0, 1, 0});
                    data.scene.add_plane({pos[0], pos[1], pos[2]}, {norm[0], norm[1], norm[2]}, mat_id);
                }
                else if (type == "box") {
                    if (obj.contains("min") && obj.contains("max")) {
                        auto min_pt = obj["min"].get<std::vector<double>>();
                        auto max_pt = obj["max"].get<std::vector<double>>();
                        data.scene.add_box({min_pt[0], min_pt[1], min_pt[2]}, 
                                          {max_pt[0], max_pt[1], max_pt[2]}, mat_id);
                    } else {
                        auto pos = obj.value("position", std::vector<double>{0, 0, 0});
                        double w = obj.value("width", 1.0);
                        double h = obj.value("height", 1.0);
                        double d = obj.value("depth", 1.0);
                        data.scene.add_box_centered({pos[0], pos[1], pos[2]}, w, h, d, mat_id);
                    }
                }
                else if (type == "triangle") {
                    auto v0 = obj["v0"].get<std::vector<double>>();
                    auto v1 = obj["v1"].get<std::vector<double>>();
                    auto v2 = obj["v2"].get<std::vector<double>>();
                    data.scene.add_triangle({v0[0], v0[1], v0[2]}, 
                                           {v1[0], v1[1], v1[2]}, 
                                           {v2[0], v2[1], v2[2]}, mat_id);
                }
            }
        }
        
        // Load lights (point, quad, disk)
        if (j.contains("lights")) {
            for (auto& light : j["lights"]) {
                std::string type = light.value("type", "point");
                auto col = light.value("color", std::vector<double>{1, 1, 1});
                double intensity = light.value("intensity", 1.0);
                color emission = {col[0] * intensity, col[1] * intensity, col[2] * intensity};
                
                if (type == "point") {
                    auto pos = light.value("position", std::vector<double>{0, 5, 0});
                    data.scene.add_light({pos[0], pos[1], pos[2]}, emission);
                }
                else if (type == "quad" || type == "rect" || type == "rectangle") {
                    // Quad light: can be defined by corner + edges, or center + size
                    if (light.contains("corner") && light.contains("edge_u") && light.contains("edge_v")) {
                        // Corner-based definition
                        auto corner = light["corner"].get<std::vector<double>>();
                        auto u = light["edge_u"].get<std::vector<double>>();
                        auto v = light["edge_v"].get<std::vector<double>>();
                        data.scene.add_quad_light(
                            {corner[0], corner[1], corner[2]},
                            {u[0], u[1], u[2]},
                            {v[0], v[1], v[2]},
                            emission
                        );
                    } else {
                        // Center-based definition (easier for users)
                        auto pos = light.value("position", std::vector<double>{0, 3, 0});
                        double width = light.value("width", 1.0);
                        double height = light.value("height", 1.0);
                        auto normal = light.value("normal", std::vector<double>{0, -1, 0});
                        
                        // Build local coordinate frame from normal
                        vec3 n = vec3_normalize({normal[0], normal[1], normal[2]});
                        vec3 up = (std::fabs(n.y) < 0.999) ? vec3{0, 1, 0} : vec3{1, 0, 0};
                        vec3 u_dir = vec3_normalize(vec3_cross(up, n));
                        vec3 v_dir = vec3_cross(n, u_dir);
                        
                        data.scene.add_quad_light_centered(
                            {pos[0], pos[1], pos[2]},
                            u_dir, v_dir,
                            width, height,
                            emission
                        );
                    }
                }
                else if (type == "disk" || type == "disc") {
                    auto pos = light.value("position", std::vector<double>{0, 3, 0});
                    double radius = light.value("radius", 0.5);
                    auto normal = light.value("normal", std::vector<double>{0, -1, 0});
                    
                    data.scene.add_disk_light(
                        {pos[0], pos[1], pos[2]},
                        {normal[0], normal[1], normal[2]},
                        radius,
                        emission
                    );
                }
            }
        }
        
        // Load camera settings
        if (j.contains("camera")) {
            auto& cam = j["camera"];
            if (cam.contains("position")) {
                auto p = cam["position"].get<std::vector<double>>();
                data.camera_position = {p[0], p[1], p[2]};
            }
            if (cam.contains("target")) {
                auto t = cam["target"].get<std::vector<double>>();
                data.camera_target = {t[0], t[1], t[2]};
            }
            if (cam.contains("up")) {
                auto u = cam["up"].get<std::vector<double>>();
                data.camera_up = {u[0], u[1], u[2]};
            }
            data.camera_fov = cam.value("fov", 60.0);
        }
        
        // Load render settings
        if (j.contains("render")) {
            auto& r = j["render"];
            data.width = r.value("width", 2560);
            data.height = r.value("height", 1440);
            data.samples = r.value("samples", 16);
            data.max_depth = r.value("max_depth", 50);
            data.output_file = r.value("output", "output.png");
            data.use_nee = r.value("nee", true);
            data.use_mis = r.value("mis", true);
            data.exposure = r.value("exposure", 1.0);
            data.clamp_max = r.value("clamp_max", 10.0);
            data.wavelength_samples = r.value("wavelength_samples", 8);
            
            // Parse render mode
            std::string mode_str = r.value("mode", "whitted");
            if (mode_str == "pathtrace" || mode_str == "pathtracing" || mode_str == "path") {
                data.mode = RenderMode::PathTrace;
            } else if (mode_str == "bdpt" || mode_str == "bidirectional") {
                data.mode = RenderMode::BDPT;
            } else if (mode_str == "spectral" || mode_str == "wavelength") {
                data.mode = RenderMode::Spectral;
            } else if (mode_str == "plt" || mode_str == "polarized") {
                data.mode = RenderMode::PLT;
            } else {
                data.mode = RenderMode::Whitted;
            }
            
            // Parse tone mapper
            std::string tm_str = r.value("tonemapper", "aces");
            if (tm_str == "none") data.tone_mapper = ToneMapper::None;
            else if (tm_str == "reinhard") data.tone_mapper = ToneMapper::Reinhard;
            else if (tm_str == "uncharted2") data.tone_mapper = ToneMapper::Uncharted2;
            else data.tone_mapper = ToneMapper::ACES;
        }
        
        // Load environment map (HDR sky lighting)
        if (j.contains("environment")) {
            auto& env = j["environment"];
            std::string env_file = env.value("file", "");
            
            if (!env_file.empty()) {
                double env_intensity = env.value("intensity", 1.0);
                double env_rotation = env.value("rotation", 0.0);
                
                auto envmap = std::make_shared<EnvironmentMap>(
                    EnvironmentMap::load(env_file, env_intensity, env_rotation)
                );
                
                if (envmap->valid()) {
                    data.scene.environment = envmap;
                }
            }
        }
        
        // Support top-level render_mode key (alternative to render.mode)
        auto parse_render_mode = [](const std::string& mode_str) -> RenderMode {
            if (mode_str == "pathtrace" || mode_str == "pathtracing" || mode_str == "path") {
                return RenderMode::PathTrace;
            } else if (mode_str == "bdpt" || mode_str == "bidirectional") {
                return RenderMode::BDPT;
            } else if (mode_str == "spectral" || mode_str == "wavelength") {
                return RenderMode::Spectral;
            } else if (mode_str == "plt" || mode_str == "polarized") {
                return RenderMode::PLT;
            }
            return RenderMode::Whitted;
        };
        
        if (j.contains("render_mode")) {
            data.mode = parse_render_mode(j["render_mode"].get<std::string>());
        }
        
        // Also support top-level image block (common JSON format)
        if (j.contains("image")) {
            auto& img = j["image"];
            data.width = img.value("width", data.width);
            data.height = img.value("height", data.height);
            data.samples = img.value("samples_per_pixel", img.value("samples", data.samples));
            data.max_depth = img.value("max_depth", data.max_depth);
        }
        
        std::cout << "Loaded scene: " << data.scene.spheres.size() << " spheres, "
                  << data.scene.planes.size() << " planes, "
                  << data.scene.boxes.size() << " boxes, "
                  << data.scene.triangles.size() << " triangles, "
                  << data.scene.lights.size() << " point lights, "
                  << data.scene.quad_lights.size() << " quad lights, "
                  << data.scene.disk_lights.size() << " disk lights";
        if (data.scene.environment) {
            std::cout << ", environment map";
        }
        std::cout << std::endl;
        
        return data;
    }
};

} // namespace raytracer
