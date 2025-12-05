/**
 * @file spectrum.hpp
 * @brief Core spectral representation for wavelength-dependent rendering
 * 
 * This module provides the foundation for spectral rendering:
 * - Wavelength sampling and representation
 * - Spectral power distributions (SPD)
 * - Conversion between spectral and RGB color spaces
 * 
 * Design: Extensible for future wave optics (polarization, coherence)
 */

#pragma once

#include <cmath>
#include <algorithm>
#include <array>
#include <vector>

namespace raytracer {
namespace spectral {

// ============================================================================
// Constants
// ============================================================================

/// Visible spectrum bounds (nanometers)
constexpr double LAMBDA_MIN = 380.0;  // Violet
constexpr double LAMBDA_MAX = 780.0;  // Deep red

/// Common reference wavelengths (nm)
namespace wavelengths {
    constexpr double VIOLET = 400.0;
    constexpr double BLUE   = 470.0;
    constexpr double CYAN   = 500.0;
    constexpr double GREEN  = 530.0;
    constexpr double YELLOW = 580.0;
    constexpr double ORANGE = 600.0;
    constexpr double RED    = 650.0;
    
    // Fraunhofer lines (standard for optical glass characterization)
    constexpr double F_LINE = 486.1;  // Hydrogen blue
    constexpr double D_LINE = 589.3;  // Sodium yellow (standard reference)
    constexpr double C_LINE = 656.3;  // Hydrogen red
}

// ============================================================================
// Spectral Sample
// ============================================================================

/**
 * @brief A single wavelength sample with its radiance value
 * 
 * This is the fundamental unit for spectral path tracing.
 * Each ray carries a specific wavelength and accumulates spectral radiance.
 */
struct SpectralSample {
    double lambda = wavelengths::D_LINE;  // Wavelength in nm
    double value = 0.0;                    // Spectral radiance at this wavelength
    
    SpectralSample() = default;
    SpectralSample(double l, double v) : lambda(l), value(v) {}
    
    /// Scale the radiance
    SpectralSample operator*(double s) const { return {lambda, value * s}; }
    SpectralSample& operator*=(double s) { value *= s; return *this; }
    
    /// Add radiance (wavelengths must match!)
    SpectralSample operator+(const SpectralSample& other) const {
        return {lambda, value + other.value};
    }
    SpectralSample& operator+=(const SpectralSample& other) {
        value += other.value;
        return *this;
    }
};

// ============================================================================
// Wavelength Sampling
// ============================================================================

/**
 * @brief Sample a wavelength uniformly from the visible spectrum
 * @param u Random number in [0, 1)
 * @return Wavelength in nm
 */
inline double sample_wavelength_uniform(double u) {
    return LAMBDA_MIN + u * (LAMBDA_MAX - LAMBDA_MIN);
}

/**
 * @brief PDF for uniform wavelength sampling
 */
inline double pdf_wavelength_uniform() {
    return 1.0 / (LAMBDA_MAX - LAMBDA_MIN);
}

/**
 * @brief Sample wavelength with importance sampling toward visible sensitivity
 * 
 * Uses a tent function peaked around green (where human vision is most sensitive)
 * This reduces variance for scenes with natural lighting.
 * 
 * @param u Random number in [0, 1)
 * @return Wavelength in nm
 */
inline double sample_wavelength_visible(double u) {
    // Tent distribution peaked at 550nm (green)
    // Better importance sampling for human-perceived luminance
    constexpr double peak = 550.0;
    constexpr double range = LAMBDA_MAX - LAMBDA_MIN;
    
    if (u < 0.5) {
        // Left side of tent
        double t = u * 2.0;
        return LAMBDA_MIN + (peak - LAMBDA_MIN) * std::sqrt(t);
    } else {
        // Right side of tent
        double t = (u - 0.5) * 2.0;
        return peak + (LAMBDA_MAX - peak) * (1.0 - std::sqrt(1.0 - t));
    }
}

/**
 * @brief PDF for visibility-weighted wavelength sampling
 */
inline double pdf_wavelength_visible(double lambda) {
    constexpr double peak = 550.0;
    
    if (lambda < LAMBDA_MIN || lambda > LAMBDA_MAX) return 0.0;
    
    if (lambda < peak) {
        double t = (lambda - LAMBDA_MIN) / (peak - LAMBDA_MIN);
        return 2.0 * t / (LAMBDA_MAX - LAMBDA_MIN);
    } else {
        double t = (LAMBDA_MAX - lambda) / (LAMBDA_MAX - peak);
        return 2.0 * t / (LAMBDA_MAX - LAMBDA_MIN);
    }
}

// ============================================================================
// Spectral to RGB Conversion (CIE XYZ color matching)
// ============================================================================

/**
 * @brief CIE 1931 2-degree color matching functions
 * 
 * These convert a single wavelength to XYZ tristimulus values.
 * Uses the analytical approximation by Wyman et al. (2013)
 */
namespace cie {

/**
 * @brief CIE X color matching function (red-ish response)
 */
inline double x_bar(double lambda) {
    double t1 = (lambda - 442.0) * ((lambda < 442.0) ? 0.0624 : 0.0374);
    double t2 = (lambda - 599.8) * ((lambda < 599.8) ? 0.0264 : 0.0323);
    double t3 = (lambda - 501.1) * ((lambda < 501.1) ? 0.0490 : 0.0382);
    return 0.362 * std::exp(-0.5 * t1 * t1) 
         + 1.056 * std::exp(-0.5 * t2 * t2)
         - 0.065 * std::exp(-0.5 * t3 * t3);
}

/**
 * @brief CIE Y color matching function (luminance/green response)
 */
inline double y_bar(double lambda) {
    double t1 = (lambda - 568.8) * ((lambda < 568.8) ? 0.0213 : 0.0247);
    double t2 = (lambda - 530.9) * ((lambda < 530.9) ? 0.0613 : 0.0322);
    return 0.821 * std::exp(-0.5 * t1 * t1)
         + 0.286 * std::exp(-0.5 * t2 * t2);
}

/**
 * @brief CIE Z color matching function (blue response)
 */
inline double z_bar(double lambda) {
    double t1 = (lambda - 437.0) * ((lambda < 437.0) ? 0.0845 : 0.0278);
    double t2 = (lambda - 459.0) * ((lambda < 459.0) ? 0.0385 : 0.0725);
    return 1.217 * std::exp(-0.5 * t1 * t1)
         + 0.681 * std::exp(-0.5 * t2 * t2);
}

} // namespace cie

/**
 * @brief Convert a spectral sample to XYZ tristimulus values
 */
inline void wavelength_to_xyz(double lambda, double radiance, 
                               double& X, double& Y, double& Z) {
    X = radiance * cie::x_bar(lambda);
    Y = radiance * cie::y_bar(lambda);
    Z = radiance * cie::z_bar(lambda);
}

/**
 * @brief Convert XYZ to linear sRGB
 * 
 * Uses the standard sRGB transformation matrix
 */
inline void xyz_to_rgb(double X, double Y, double Z,
                        double& R, double& G, double& B) {
    // sRGB transformation matrix (D65 white point)
    R =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
    B =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
}

/**
 * @brief Convert a single wavelength to RGB color
 * 
 * Returns the RGB color that this wavelength appears as.
 * Useful for visualization and single-wavelength rendering.
 * 
 * @param lambda Wavelength in nm
 * @param intensity Radiance value (default 1.0)
 * @return RGB tuple (may contain values outside [0,1] for saturated colors)
 */
inline std::array<double, 3> wavelength_to_rgb(double lambda, double intensity = 1.0) {
    double X, Y, Z;
    wavelength_to_xyz(lambda, intensity, X, Y, Z);
    
    double R, G, B;
    xyz_to_rgb(X, Y, Z, R, G, B);
    
    return {R, G, B};
}

// ============================================================================
// Hero Wavelength Sampling (for spectral MIS)
// ============================================================================

/**
 * @brief Number of wavelengths to track per ray (Hero Wavelength Spectral Sampling)
 * 
 * HWSS traces multiple wavelengths per ray for better efficiency.
 * 4 is a good balance between quality and performance.
 */
constexpr int HWSS_SAMPLES = 4;

/**
 * @brief Generate stratified wavelength samples for HWSS
 * 
 * Divides the spectrum into N strata and samples one wavelength per stratum.
 * This reduces variance compared to independent sampling.
 * 
 * @param u Base random number in [0, 1)
 * @param wavelengths Output array of wavelengths
 */
inline void sample_hwss_wavelengths(double u, std::array<double, HWSS_SAMPLES>& wavelengths) {
    double stratum_size = (LAMBDA_MAX - LAMBDA_MIN) / HWSS_SAMPLES;
    
    for (int i = 0; i < HWSS_SAMPLES; ++i) {
        // Stratified jitter within each stratum
        double jitter = std::fmod(u + static_cast<double>(i) / HWSS_SAMPLES, 1.0);
        wavelengths[i] = LAMBDA_MIN + (i + jitter) * stratum_size;
    }
}

/**
 * @brief Hero Wavelength Spectral Sample
 * 
 * Tracks multiple wavelengths per ray for efficient spectral rendering.
 * The "hero" wavelength determines path decisions (reflection vs refraction),
 * while all wavelengths accumulate radiance.
 */
struct HWSSample {
    std::array<double, HWSS_SAMPLES> lambda;   // Wavelengths (nm)
    std::array<double, HWSS_SAMPLES> value;    // Radiance per wavelength
    int hero_index = 0;                         // Index of hero wavelength
    
    HWSSample() {
        lambda.fill(wavelengths::D_LINE);
        value.fill(0.0);
    }
    
    /// Get the hero wavelength (used for path decisions)
    double hero_lambda() const { return lambda[hero_index]; }
    
    /// Get hero radiance
    double hero_value() const { return value[hero_index]; }
    
    /// Scale all wavelengths
    HWSSample operator*(double s) const {
        HWSSample result = *this;
        for (int i = 0; i < HWSS_SAMPLES; ++i) {
            result.value[i] *= s;
        }
        return result;
    }
    
    /// Convert accumulated spectral radiance to RGB
    std::array<double, 3> to_rgb() const {
        double X = 0, Y = 0, Z = 0;
        
        // Integrate over all wavelength samples
        for (int i = 0; i < HWSS_SAMPLES; ++i) {
            double Xi, Yi, Zi;
            wavelength_to_xyz(lambda[i], value[i], Xi, Yi, Zi);
            X += Xi;
            Y += Yi;
            Z += Zi;
        }
        
        // Normalize by number of samples and spectrum width
        double norm = (LAMBDA_MAX - LAMBDA_MIN) / HWSS_SAMPLES;
        X *= norm;
        Y *= norm;
        Z *= norm;
        
        double R, G, B;
        xyz_to_rgb(X, Y, Z, R, G, B);
        
        return {R, G, B};
    }
};

} // namespace spectral
} // namespace raytracer
