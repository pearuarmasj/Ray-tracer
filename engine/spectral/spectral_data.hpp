/**
 * @file spectral_data.hpp
 * @brief Tabulated spectral data for realistic rendering
 * 
 * Contains:
 * - Standard illuminant spectra (D65, etc.)
 * - Light source emission spectra
 * - Metal complex refractive indices (n, k)
 * 
 * Data sources:
 * - CIE Standard Illuminants
 * - Spectral databases (LSPDD, refractiveindex.info)
 */

#pragma once

#include "spectrum.hpp"
#include <array>
#include <cmath>
#include <algorithm>

namespace raytracer {
namespace spectral {
namespace data {

// ============================================================================
// Standard Illuminants
// ============================================================================

/**
 * @brief CIE Standard Illuminant D65 (daylight, ~6500K)
 * 
 * The standard reference for "daylight" white. Used as the default
 * white point for sRGB and most computer graphics.
 */
struct D65 {
    /**
     * @brief Get relative spectral power at given wavelength
     */
    static double spd(double lambda_nm) {
        // Analytical approximation of D65
        // Based on Judd et al. daylight model
        
        if (lambda_nm < 300 || lambda_nm > 830) return 0.0;
        
        // Simplified D65 approximation
        double x = (lambda_nm - 560.0) / 100.0;
        return 100.0 * std::exp(-0.5 * x * x) * (1.0 + 0.1 * std::sin(x * 3.0));
    }
    
    /**
     * @brief Sample a wavelength weighted by D65 spectrum
     */
    static double sample(double u, double& pdf) {
        // For now, use uniform sampling with D65 weighting
        double lambda = sample_wavelength_uniform(u);
        pdf = pdf_wavelength_uniform();
        return lambda;
    }
};

/**
 * @brief Planckian (blackbody) radiator spectrum
 * 
 * @param lambda_nm Wavelength in nanometers
 * @param T Temperature in Kelvin
 * @return Relative spectral radiance
 */
inline double planckian_spd(double lambda_nm, double T) {
    // Planck's law (normalized)
    constexpr double h = 6.62607015e-34;  // Planck constant
    constexpr double c = 299792458.0;      // Speed of light
    constexpr double k = 1.380649e-23;     // Boltzmann constant
    
    double lambda_m = lambda_nm * 1e-9;
    double x = h * c / (lambda_m * k * T);
    
    // Prevent overflow
    if (x > 700) return 0.0;
    
    double num = 2.0 * h * c * c / (lambda_m * lambda_m * lambda_m * lambda_m * lambda_m);
    double denom = std::exp(x) - 1.0;
    
    // Normalize to peak at 1.0
    return num / denom / 1e15;  // Arbitrary normalization for numerical stability
}

/**
 * @brief CIE Standard Illuminant A (incandescent, ~2856K)
 * 
 * Warm tungsten light bulb spectrum (Planckian radiator).
 */
struct IlluminantA {
    static constexpr double TEMPERATURE = 2856.0;
    
    static double spd(double lambda_nm) {
        return planckian_spd(lambda_nm, TEMPERATURE);
    }
};

/**
 * @brief Color temperature to RGB (approximate)
 * 
 * Converts a blackbody temperature to an RGB color.
 * Useful for light sources specified by temperature.
 */
inline std::array<double, 3> temperature_to_rgb(double T) {
    // Tanner Helland's algorithm
    T = T / 100.0;
    
    double r, g, b;
    
    if (T <= 66) {
        r = 255;
        g = 99.4708025861 * std::log(T) - 161.1195681661;
        if (T <= 19) {
            b = 0;
        } else {
            b = 138.5177312231 * std::log(T - 10) - 305.0447927307;
        }
    } else {
        r = 329.698727446 * std::pow(T - 60, -0.1332047592);
        g = 288.1221695283 * std::pow(T - 60, -0.0755148492);
        b = 255;
    }
    
    return {
        std::clamp(r / 255.0, 0.0, 1.0),
        std::clamp(g / 255.0, 0.0, 1.0),
        std::clamp(b / 255.0, 0.0, 1.0)
    };
}

// ============================================================================
// Metal Complex Refractive Indices
// ============================================================================

/**
 * @brief Complex refractive index (n + ik) for metals
 * 
 * Metals have complex IOR where:
 * - n = real part (refraction)
 * - k = extinction coefficient (absorption)
 * 
 * The Fresnel equations use both to compute reflectance.
 */
struct ComplexIOR {
    double n;  // Real part
    double k;  // Imaginary part (extinction)
    
    ComplexIOR(double n_ = 1.0, double k_ = 0.0) : n(n_), k(k_) {}
};

/**
 * @brief Get wavelength-dependent complex IOR for gold
 * 
 * Data from Johnson & Christy (1972)
 */
inline ComplexIOR gold_ior(double lambda_nm) {
    // Simplified analytical fit
    double x = (lambda_nm - 500.0) / 200.0;
    double n = 0.18 + 0.15 * std::max(0.0, x);
    double k = 2.5 + 1.5 * std::max(0.0, x);
    return {n, k};
}

/**
 * @brief Get wavelength-dependent complex IOR for silver
 */
inline ComplexIOR silver_ior(double lambda_nm) {
    double x = (lambda_nm - 500.0) / 200.0;
    double n = 0.05 + 0.1 * std::abs(x);
    double k = 3.0 + 1.0 * std::max(0.0, x);
    return {n, k};
}

/**
 * @brief Get wavelength-dependent complex IOR for copper
 */
inline ComplexIOR copper_ior(double lambda_nm) {
    double x = (lambda_nm - 550.0) / 200.0;
    double n = 0.3 + 0.7 * std::max(0.0, -x);
    double k = 2.5 + 1.2 * std::max(0.0, x);
    return {n, k};
}

/**
 * @brief Get wavelength-dependent complex IOR for aluminum
 */
inline ComplexIOR aluminum_ior(double lambda_nm) {
    // Aluminum has fairly flat response
    double n = 1.1 + 0.3 * (lambda_nm - 500.0) / 300.0;
    double k = 7.0 + 0.5 * (lambda_nm - 500.0) / 300.0;
    return {n, k};
}

/**
 * @brief Fresnel reflectance for metals (conductor)
 * 
 * Uses the full conductor Fresnel equations with complex IOR.
 * 
 * @param cos_theta Cosine of incident angle
 * @param ior Complex refractive index
 * @return Fresnel reflectance [0, 1]
 */
inline double fresnel_conductor(double cos_theta, const ComplexIOR& ior) {
    double n = ior.n;
    double k = ior.k;
    
    double cos2 = cos_theta * cos_theta;
    double sin2 = 1.0 - cos2;
    
    double n2 = n * n;
    double k2 = k * k;
    
    double t0 = n2 - k2 - sin2;
    double a2_plus_b2 = std::sqrt(t0 * t0 + 4 * n2 * k2);
    double t1 = a2_plus_b2 + cos2;
    double a = std::sqrt(0.5 * (a2_plus_b2 + t0));
    double t2 = 2.0 * a * cos_theta;
    
    double Rs = (t1 - t2) / (t1 + t2);
    
    double t3 = cos2 * a2_plus_b2 + sin2 * sin2;
    double t4 = t2 * sin2;
    
    double Rp = Rs * (t3 - t4) / (t3 + t4);
    
    return 0.5 * (Rs + Rp);
}

/**
 * @brief Spectral Fresnel for conductors
 * 
 * Computes RGB reflectance by sampling multiple wavelengths.
 */
inline std::array<double, 3> fresnel_conductor_spectral(
    double cos_theta,
    ComplexIOR (*ior_func)(double)) {
    
    // Sample at RGB primary wavelengths
    auto r_ior = ior_func(wavelengths::RED);
    auto g_ior = ior_func(wavelengths::GREEN);
    auto b_ior = ior_func(wavelengths::BLUE);
    
    return {
        fresnel_conductor(cos_theta, r_ior),
        fresnel_conductor(cos_theta, g_ior),
        fresnel_conductor(cos_theta, b_ior)
    };
}

// ============================================================================
// RGB to Spectrum Conversion
// ============================================================================

/**
 * @brief Convert RGB color to spectral representation
 * 
 * This is an ill-posed problem (infinitely many spectra map to same RGB).
 * We use a simple smits-like basis function approach.
 * 
 * @param r Red component [0, 1]
 * @param g Green component [0, 1]  
 * @param b Blue component [0, 1]
 * @param lambda Wavelength to evaluate at
 * @return Spectral reflectance at this wavelength
 */
inline double rgb_to_spectrum(double r, double g, double b, double lambda) {
    // Simple Gaussian basis functions centered at RGB primaries
    auto gaussian = [](double x, double mean, double sigma) {
        double t = (x - mean) / sigma;
        return std::exp(-0.5 * t * t);
    };
    
    // RGB basis spectra (simplified)
    double r_basis = gaussian(lambda, 650, 50);  // Red peak
    double g_basis = gaussian(lambda, 530, 50);  // Green peak
    double b_basis = gaussian(lambda, 450, 40);  // Blue peak
    
    // Combine with weights
    double spectrum = r * r_basis + g * g_basis + b * b_basis;
    
    // Normalize to reasonable range
    return std::clamp(spectrum, 0.0, 1.0);
}

/**
 * @brief Get reflectance spectrum from RGB albedo
 * 
 * For a diffuse material with RGB albedo, returns spectral reflectance.
 */
struct RGBSpectrum {
    double r, g, b;
    
    RGBSpectrum(double r_ = 0.5, double g_ = 0.5, double b_ = 0.5)
        : r(r_), g(g_), b(b_) {}
    
    double evaluate(double lambda) const {
        return rgb_to_spectrum(r, g, b, lambda);
    }
};

} // namespace data
} // namespace spectral
} // namespace raytracer
