/**
 * @file scene_loader.hpp
 * @brief JSON scene file loading
 */

#pragma once

#include "json.hpp"
#include "scene.hpp"
#include "material.hpp"
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
        
        std::string output_file = "output.png";
    };
    
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
                
                if (type == "lambertian") {
                    auto c = mat.value("color", std::vector<double>{0.5, 0.5, 0.5});
                    m = Material::lambertian({c[0], c[1], c[2]});
                }
                else if (type == "metal") {
                    auto c = mat.value("color", std::vector<double>{0.8, 0.8, 0.8});
                    double fuzz = mat.value("fuzz", 0.0);
                    m = Material::metal({c[0], c[1], c[2]}, fuzz);
                }
                else if (type == "dielectric" || type == "glass") {
                    double ior = mat.value("ior", 1.5);
                    m = Material::dielectric(ior);
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
        
        // Load lights
        if (j.contains("lights")) {
            for (auto& light : j["lights"]) {
                auto pos = light.value("position", std::vector<double>{0, 5, 0});
                auto col = light.value("color", std::vector<double>{1, 1, 1});
                double intensity = light.value("intensity", 1.0);
                data.scene.add_light({pos[0], pos[1], pos[2]}, 
                                    {col[0] * intensity, col[1] * intensity, col[2] * intensity});
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
            data.width = r.value("width", 1920);
            data.height = r.value("height", 1080);
            data.samples = r.value("samples", 16);
            data.max_depth = r.value("max_depth", 50);
            data.output_file = r.value("output", "output.png");
        }
        
        std::cout << "Loaded scene: " << data.scene.spheres.size() << " spheres, "
                  << data.scene.planes.size() << " planes, "
                  << data.scene.boxes.size() << " boxes, "
                  << data.scene.triangles.size() << " triangles, "
                  << data.scene.lights.size() << " lights" << std::endl;
        
        return data;
    }
};

} // namespace raytracer
