/**
 * @file dispersion.hpp
 * @brief Wavelength-dependent refractive index models
 * 
 * Implements standard dispersion models for optical materials:
 * - Sellmeier equation (most accurate for glasses)
 * - Cauchy equation (simpler, good approximation)
 * - Tabulated data for metals and special materials
 * 
 * These enable realistic rainbow/prism effects and chromatic aberration.
 */

#pragma once

#include "spectrum.hpp"
#include <cmath>
#include <array>

namespace raytracer {
namespace spectral {

// ============================================================================
// Dispersion Models
// ============================================================================

/**
 * @brief Sellmeier dispersion model
 * 
 * The Sellmeier equation accurately models refractive index for transparent
 * materials like glass, water, and crystals. It's based on the material's
 * resonance wavelengths.
 * 
 * n²(λ) = 1 + Σ (Bᵢ·λ²) / (λ² - Cᵢ)
 * 
 * where Bᵢ are oscillator strengths and Cᵢ are resonance wavelengths squared.
 */
struct SellmeierDispersion {
    // Up to 3 terms (sufficient for most materials)
    std::array<double, 3> B = {0, 0, 0};  // Oscillator strengths
    std::array<double, 3> C = {0, 0, 0};  // Resonance wavelengths² (μm²)
    
    /**
     * @brief Compute refractive index at given wavelength
     * @param lambda_nm Wavelength in nanometers
     * @return Refractive index n
     */
    double n(double lambda_nm) const {
        // Convert nm to μm (Sellmeier coefficients use μm)
        double lambda_um = lambda_nm / 1000.0;
        double lambda_sq = lambda_um * lambda_um;
        
        double n_sq = 1.0;
        for (int i = 0; i < 3; ++i) {
            if (B[i] != 0.0) {
                n_sq += (B[i] * lambda_sq) / (lambda_sq - C[i]);
            }
        }
        
        return std::sqrt(std::max(1.0, n_sq));
    }
    
    /**
     * @brief Compute Abbe number (dispersion strength)
     * 
     * V = (nD - 1) / (nF - nC)
     * 
     * Higher Abbe number = less dispersion
     * Crown glass: ~60, Flint glass: ~30-40, Diamond: ~44
     */
    double abbe_number() const {
        double nD = n(wavelengths::D_LINE);
        double nF = n(wavelengths::F_LINE);
        double nC = n(wavelengths::C_LINE);
        
        if (std::abs(nF - nC) < 1e-6) return 1000.0;  // No dispersion
        return (nD - 1.0) / (nF - nC);
    }
};

/**
 * @brief Cauchy dispersion model
 * 
 * Simpler approximation: n(λ) = A + B/λ² + C/λ⁴
 * 
 * Less accurate than Sellmeier but easier to fit and faster to compute.
 * Good for quick approximations and materials with weak dispersion.
 */
struct CauchyDispersion {
    double A = 1.5;    // Base refractive index
    double B = 0.004;  // First-order dispersion (μm²)
    double C = 0.0;    // Second-order dispersion (μm⁴)
    
    /**
     * @brief Compute refractive index at given wavelength
     * @param lambda_nm Wavelength in nanometers
     * @return Refractive index n
     */
    double n(double lambda_nm) const {
        double lambda_um = lambda_nm / 1000.0;
        double lambda_sq = lambda_um * lambda_um;
        
        return A + B / lambda_sq + C / (lambda_sq * lambda_sq);
    }
    
    /**
     * @brief Create from refractive index at D-line and Abbe number
     * 
     * Convenient constructor for when you know nD and V (common specs).
     */
    static CauchyDispersion from_abbe(double nD, double V) {
        // Approximate B from Abbe number
        // V ≈ (nD - 1) / (B/λF² - B/λC²)
        double denom = 1.0 / (wavelengths::F_LINE * wavelengths::F_LINE / 1e6) 
                     - 1.0 / (wavelengths::C_LINE * wavelengths::C_LINE / 1e6);
        double B = (nD - 1.0) / (V * denom);
        
        // Solve for A: nD = A + B/λD²
        double A = nD - B / (wavelengths::D_LINE * wavelengths::D_LINE / 1e6);
        
        return {A, B, 0.0};
    }
};

// ============================================================================
// Predefined Materials
// ============================================================================

namespace materials {

/**
 * @brief BK7 borosilicate crown glass (standard optical glass)
 * 
 * Very common in lenses and prisms. nD ≈ 1.5168, V ≈ 64
 */
inline SellmeierDispersion BK7() {
    SellmeierDispersion s;
    s.B = {1.03961212, 0.231792344, 1.01046945};
    s.C = {0.00600069867, 0.0200179144, 103.560653};
    return s;
}

/**
 * @brief SF10 dense flint glass (high dispersion)
 * 
 * Used in achromatic doublets. nD ≈ 1.7283, V ≈ 28.4
 */
inline SellmeierDispersion SF10() {
    SellmeierDispersion s;
    s.B = {1.62153902, 0.256287842, 1.64447552};
    s.C = {0.0122241457, 0.0595736775, 147.468793};
    return s;
}

/**
 * @brief SF11 extra-dense flint glass (very high dispersion)
 * 
 * nD ≈ 1.7847, V ≈ 25.7 - great for rainbows!
 */
inline SellmeierDispersion SF11() {
    SellmeierDispersion s;
    s.B = {1.73759695, 0.313747346, 1.89878101};
    s.C = {0.013188707, 0.0623068142, 155.23629};
    return s;
}

/**
 * @brief Fused silica (quartz glass)
 * 
 * Very pure, low dispersion. nD ≈ 1.4585, V ≈ 67.8
 */
inline SellmeierDispersion FusedSilica() {
    SellmeierDispersion s;
    s.B = {0.6961663, 0.4079426, 0.8974794};
    s.C = {0.0684043 * 0.0684043, 0.1162414 * 0.1162414, 9.896161 * 9.896161};
    return s;
}

/**
 * @brief Diamond
 * 
 * Very high refractive index, strong dispersion → fire!
 * nD ≈ 2.417, V ≈ 44
 */
inline SellmeierDispersion Diamond() {
    SellmeierDispersion s;
    s.B = {0.3306, 4.3356, 0.0};
    s.C = {0.0 * 0.0, 0.1060 * 0.1060, 0.0};
    return s;
}

/**
 * @brief Water at 20°C
 * 
 * nD ≈ 1.333, V ≈ 55
 */
inline SellmeierDispersion Water() {
    SellmeierDispersion s;
    s.B = {0.5684027565, 0.1726177391, 0.02086189578};
    s.C = {0.005101829712, 0.01821153936, 0.02620722293};
    return s;
}

/**
 * @brief Sapphire (Al2O3)
 * 
 * nD ≈ 1.77, V ≈ 72.2
 */
inline SellmeierDispersion Sapphire() {
    SellmeierDispersion s;
    s.B = {1.4313493, 0.65054713, 5.3414021};
    s.C = {0.0052799261, 0.0142382647, 325.01783};
    return s;
}

// Simple Cauchy approximations for common materials

/**
 * @brief Generic crown glass (simple approximation)
 */
inline CauchyDispersion CrownGlass() {
    return CauchyDispersion::from_abbe(1.52, 59.0);
}

/**
 * @brief Generic flint glass (simple approximation)
 */
inline CauchyDispersion FlintGlass() {
    return CauchyDispersion::from_abbe(1.62, 36.0);
}

/**
 * @brief Acrylic/PMMA (plastic)
 */
inline CauchyDispersion Acrylic() {
    return CauchyDispersion::from_abbe(1.49, 57.0);
}

/**
 * @brief Polycarbonate (plastic, high dispersion)
 */
inline CauchyDispersion Polycarbonate() {
    return CauchyDispersion::from_abbe(1.585, 30.0);
}

} // namespace materials

// ============================================================================
// Dispersion Interface
// ============================================================================

/**
 * @brief Unified dispersion model that can use Sellmeier or Cauchy
 * 
 * Allows materials to specify dispersion in either form.
 */
class Dispersion {
public:
    enum class Model { None, Sellmeier, Cauchy };
    
    Dispersion() : model_(Model::None), base_n_(1.5) {}
    
    explicit Dispersion(double fixed_n) 
        : model_(Model::None), base_n_(fixed_n) {}
    
    explicit Dispersion(const SellmeierDispersion& s) 
        : model_(Model::Sellmeier), sellmeier_(s), base_n_(s.n(wavelengths::D_LINE)) {}
    
    explicit Dispersion(const CauchyDispersion& c) 
        : model_(Model::Cauchy), cauchy_(c), base_n_(c.n(wavelengths::D_LINE)) {}
    
    /**
     * @brief Get refractive index at specific wavelength
     */
    double n(double lambda_nm) const {
        switch (model_) {
            case Model::Sellmeier: return sellmeier_.n(lambda_nm);
            case Model::Cauchy:    return cauchy_.n(lambda_nm);
            default:               return base_n_;
        }
    }
    
    /**
     * @brief Get refractive index at D-line (589nm reference)
     */
    double nD() const { return base_n_; }
    
    /**
     * @brief Check if this material has wavelength-dependent dispersion
     */
    bool has_dispersion() const { return model_ != Model::None; }
    
    /**
     * @brief Get Abbe number (dispersion strength)
     */
    double abbe_number() const {
        if (model_ == Model::Sellmeier) return sellmeier_.abbe_number();
        // Approximate for Cauchy
        double nF = n(wavelengths::F_LINE);
        double nC = n(wavelengths::C_LINE);
        if (std::abs(nF - nC) < 1e-6) return 1000.0;
        return (base_n_ - 1.0) / (nF - nC);
    }
    
private:
    Model model_;
    SellmeierDispersion sellmeier_;
    CauchyDispersion cauchy_;
    double base_n_;  // Reference IOR at D-line
};

// ============================================================================
// Snell's Law with Dispersion
// ============================================================================

/**
 * @brief Compute refraction direction with wavelength-dependent IOR
 * 
 * @param incident Incident direction (pointing toward surface)
 * @param normal Surface normal (pointing outward)
 * @param n1 Refractive index of incident medium
 * @param n2 Refractive index of transmitted medium
 * @param refracted Output refracted direction
 * @return true if refraction occurs, false if total internal reflection
 */
template<typename Vec3>
bool refract_dispersive(const Vec3& incident, const Vec3& normal, 
                        double n1, double n2, Vec3& refracted) {
    double eta = n1 / n2;
    double cos_i = -dot(incident, normal);
    
    // Handle rays from inside the material
    Vec3 n = normal;
    if (cos_i < 0) {
        n = -normal;
        cos_i = -cos_i;
        eta = n2 / n1;
    }
    
    double sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    
    // Total internal reflection
    if (sin2_t > 1.0) {
        return false;
    }
    
    double cos_t = std::sqrt(1.0 - sin2_t);
    refracted = eta * incident + (eta * cos_i - cos_t) * n;
    
    return true;
}

} // namespace spectral
} // namespace raytracer
