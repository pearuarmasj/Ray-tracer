/**
 * @file hwss.hpp
 * @brief Hero Wavelength Spectral Sampling (HWSS)
 * 
 * Implements correlated wavelength sampling to reduce spectral noise.
 * Instead of sampling wavelengths independently, we pick a "hero" wavelength
 * and derive N-1 stratified wavelengths from it, all sharing the same path.
 * 
 * Reference: "A Low-Dimensional Function Space for Efficient Spectral Upsampling"
 *            Mallett & Yuksel, 2019
 */

#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

namespace raytracer {

/**
 * @brief Hero Wavelength Spectral Sampling
 * 
 * Generates stratified wavelength samples from a single random value.
 * All samples are equidistant in wavelength space, reducing variance
 * compared to independent sampling.
 */
class HWSS {
public:
    // Visible spectrum bounds (nm)
    static constexpr double LAMBDA_MIN = 380.0;
    static constexpr double LAMBDA_MAX = 780.0;
    static constexpr double LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;
    
    /**
     * @brief Generate N stratified wavelengths from a single random value
     * @tparam N Number of wavelength samples (typically 4)
     * @param xi Random value in [0, 1)
     * @return Array of N wavelengths in nm
     * 
     * The wavelengths are evenly distributed across the visible spectrum,
     * with the hero wavelength at position determined by xi.
     */
    template<size_t N>
    static std::array<double, N> sample_wavelengths(double xi) {
        std::array<double, N> lambdas;
        
        // Hero wavelength
        double hero = LAMBDA_MIN + xi * LAMBDA_RANGE;
        lambdas[0] = hero;
        
        // Generate stratified samples by rotating through wavelength space
        double step = LAMBDA_RANGE / N;
        for (size_t i = 1; i < N; ++i) {
            double lambda = hero + i * step;
            // Wrap around if we exceed max
            if (lambda > LAMBDA_MAX) {
                lambda -= LAMBDA_RANGE;
            }
            lambdas[i] = lambda;
        }
        
        return lambdas;
    }
    
    /**
     * @brief Generate 4 stratified wavelengths (common case)
     * @param xi Random value in [0, 1)
     * @return Array of 4 wavelengths in nm
     */
    static std::array<double, 4> sample_wavelengths_4(double xi) {
        return sample_wavelengths<4>(xi);
    }
    
    /**
     * @brief CIE XYZ color matching functions (approximation)
     * Based on Wyman, Sloan, Shirley "Simple Analytic Approximations to the CIE XYZ Color Matching Functions"
     */
    struct XYZ {
        double X, Y, Z;
        
        XYZ operator+(const XYZ& other) const {
            return {X + other.X, Y + other.Y, Z + other.Z};
        }
        
        XYZ operator*(double s) const {
            return {X * s, Y * s, Z * s};
        }
        
        XYZ& operator+=(const XYZ& other) {
            X += other.X; Y += other.Y; Z += other.Z;
            return *this;
        }
    };
    
    /**
     * @brief Evaluate CIE XYZ for a wavelength
     * @param lambda Wavelength in nm
     * @return XYZ tristimulus values
     */
    static XYZ wavelength_to_xyz(double lambda) {
        // Gaussian fits from Wyman et al.
        auto gaussian = [](double x, double mu, double s1, double s2) {
            double t = (x - mu) / (x < mu ? s1 : s2);
            return std::exp(-0.5 * t * t);
        };
        
        XYZ xyz;
        
        // X
        xyz.X = 1.056 * gaussian(lambda, 599.8, 37.9, 31.0)
              + 0.362 * gaussian(lambda, 442.0, 16.0, 26.7)
              - 0.065 * gaussian(lambda, 501.1, 20.4, 26.2);
        
        // Y
        xyz.Y = 0.821 * gaussian(lambda, 568.8, 46.9, 40.5)
              + 0.286 * gaussian(lambda, 530.9, 16.3, 31.1);
        
        // Z
        xyz.Z = 1.217 * gaussian(lambda, 437.0, 11.8, 36.0)
              + 0.681 * gaussian(lambda, 459.0, 26.0, 13.8);
        
        return xyz;
    }
    
    /**
     * @brief Convert XYZ to linear sRGB
     * @param xyz XYZ tristimulus values
     * @return RGB in [0, 1] range (may exceed for saturated colors)
     */
    static std::array<double, 3> xyz_to_rgb(const XYZ& xyz) {
        // sRGB D65 matrix
        double r =  3.2406 * xyz.X - 1.5372 * xyz.Y - 0.4986 * xyz.Z;
        double g = -0.9689 * xyz.X + 1.8758 * xyz.Y + 0.0415 * xyz.Z;
        double b =  0.0557 * xyz.X - 0.2040 * xyz.Y + 1.0570 * xyz.Z;
        
        return {r, g, b};
    }
    
    /**
     * @brief Get RGB weight for a wavelength (for Monte Carlo integration)
     * @param lambda Wavelength in nm
     * @return RGB contribution weights
     * 
     * This is the XYZ->RGB transformed value, used to weight spectral radiance
     * contributions to the final RGB pixel value.
     */
    static std::array<double, 3> wavelength_to_rgb_weight(double lambda) {
        XYZ xyz = wavelength_to_xyz(lambda);
        return xyz_to_rgb(xyz);
    }
    
    /**
     * @brief Get luminance weight for a wavelength (for spectral MIS)
     * @param lambda Wavelength in nm
     * @return Luminance (Y component of XYZ)
     */
    static double wavelength_luminance(double lambda) {
        return wavelength_to_xyz(lambda).Y;
    }
    
    /**
     * @brief Generate N stratified wavelengths dynamically
     * @param n Number of wavelength samples
     * @param xi Random value in [0, 1)
     * @return Vector of wavelengths in nm
     */
    static std::vector<double> sample_wavelengths_dynamic(int n, double xi) {
        std::vector<double> lambdas(n);
        
        double hero = LAMBDA_MIN + xi * LAMBDA_RANGE;
        lambdas[0] = hero;
        
        double step = LAMBDA_RANGE / n;
        for (int i = 1; i < n; ++i) {
            double lambda = hero + i * step;
            if (lambda > LAMBDA_MAX) {
                lambda -= LAMBDA_RANGE;
            }
            lambdas[i] = lambda;
        }
        
        return lambdas;
    }
    
    /**
     * @brief Compute RGB from wavelength samples with optional MIS weighting
     * @param radiances Vector of spectral radiance values
     * @param lambdas Vector of wavelengths in nm
     * @param use_mis If true, weight samples by luminance for variance reduction
     * @return RGB color
     */
    static std::array<double, 3> radiances_to_rgb_dynamic(
        const std::vector<double>& radiances,
        const std::vector<double>& lambdas,
        bool use_mis = false
    ) {
        static constexpr double EE_WHITE_R = 0.3209;
        static constexpr double EE_WHITE_G = 0.2539;
        static constexpr double EE_WHITE_B = 0.2426;
        
        size_t n = radiances.size();
        if (n == 0) return {0, 0, 0};
        
        XYZ accumulated{0, 0, 0};
        
        if (use_mis) {
            // MIS weighting: weight samples by their luminance contribution
            // This reduces variance for colored lights/surfaces
            double total_weight = 0.0;
            std::vector<double> weights(n);
            
            for (size_t i = 0; i < n; ++i) {
                weights[i] = wavelength_luminance(lambdas[i]);
                total_weight += weights[i];
            }
            
            if (total_weight > 0) {
                for (size_t i = 0; i < n; ++i) {
                    XYZ xyz = wavelength_to_xyz(lambdas[i]);
                    double w = weights[i] / total_weight * n;  // Normalize to average 1
                    accumulated += xyz * (radiances[i] / std::max(w, 0.01));
                }
            }
        } else {
            // Simple averaging
            for (size_t i = 0; i < n; ++i) {
                XYZ xyz = wavelength_to_xyz(lambdas[i]);
                accumulated += xyz * radiances[i];
            }
        }
        
        double scale = 1.0 / n;
        accumulated.X *= scale;
        accumulated.Y *= scale;
        accumulated.Z *= scale;
        
        auto rgb = xyz_to_rgb(accumulated);
        return {rgb[0] / EE_WHITE_R, rgb[1] / EE_WHITE_G, rgb[2] / EE_WHITE_B};
    }
    
    /**
     * @brief Compute RGB from multiple wavelength samples using HWSS
     * @tparam N Number of samples
     * @param radiances Array of spectral radiance values for each wavelength
     * @param lambdas Array of wavelengths in nm
     * @return RGB color
     * 
     * For Monte Carlo spectral rendering:
     * RGB = integral(L(λ) * CMF(λ) dλ) ≈ (1/N) * sum(L(λi) * CMF(λi))
     * 
     * Energy normalization: Equal-energy white through XYZ->sRGB gives
     * RGB ≈ (0.32, 0.25, 0.24) due to D65 assumption. We normalize each
     * channel so flat spectrum -> equal RGB, matching path tracer behavior.
     */
    template<size_t N>
    static std::array<double, 3> radiances_to_rgb(
        const std::array<double, N>& radiances,
        const std::array<double, N>& lambdas
    ) {
        // RGB values for equal-energy white (flat spectrum L=1)
        // after XYZ->sRGB conversion, computed from CMF integrals
        static constexpr double EE_WHITE_R = 0.3209;
        static constexpr double EE_WHITE_G = 0.2539;
        static constexpr double EE_WHITE_B = 0.2426;
        
        XYZ accumulated{0, 0, 0};
        
        for (size_t i = 0; i < N; ++i) {
            XYZ xyz = wavelength_to_xyz(lambdas[i]);
            accumulated += xyz * radiances[i];
        }
        
        // Average over samples
        double scale = 1.0 / N;
        accumulated.X *= scale;
        accumulated.Y *= scale;
        accumulated.Z *= scale;
        
        auto rgb = xyz_to_rgb(accumulated);
        
        // Normalize so flat spectrum gives [1,1,1] (path tracer compatible)
        return {rgb[0] / EE_WHITE_R, rgb[1] / EE_WHITE_G, rgb[2] / EE_WHITE_B};
    }
};

} // namespace raytracer
