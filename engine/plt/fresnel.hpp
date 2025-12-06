/**
 * @file fresnel.hpp
 * @brief Polarized Fresnel equations for dielectric and conductor interfaces
 *
 * Computes separate s- and p-polarization Fresnel coefficients
 * and returns Mueller matrices for polarization-aware path tracing.
 */

#pragma once

#include "mueller.hpp"
#include <cmath>
#include <complex>
#include <algorithm>

namespace plt {

/**
 * Result of polarized Fresnel calculation
 */
struct FresnelResult {
    float Rs = 0.0f;   // Reflectance for s-polarization
    float Rp = 0.0f;   // Reflectance for p-polarization
    float Ts = 0.0f;   // Transmittance for s-polarization
    float Tp = 0.0f;   // Transmittance for p-polarization
    
    // Complex Fresnel coefficients (amplitude)
    float rs_real = 0.0f, rs_imag = 0.0f;  // r_s coefficient
    float rp_real = 0.0f, rp_imag = 0.0f;  // r_p coefficient
    float ts_real = 0.0f, ts_imag = 0.0f;  // t_s coefficient
    float tp_real = 0.0f, tp_imag = 0.0f;  // t_p coefficient
    
    float cos_t = 1.0f;  // Cosine of transmitted angle

    // Unpolarized reflectance/transmittance
    float R() const { return 0.5f * (Rs + Rp); }
    float T() const { return 0.5f * (Ts + Tp); }

    // Build Mueller matrix for reflection
    Mueller reflection_mueller() const {
        return Mueller::fresnel_reflection_complex(
            rs_real, rs_imag, rp_real, rp_imag);
    }

    // Build Mueller matrix for transmission
    Mueller transmission_mueller(float eta, float cos_i) const {
        return Mueller::fresnel_transmission(
            std::sqrt(Ts), std::sqrt(Tp), eta, cos_i, cos_t);
    }
};

/**
 * Compute polarized Fresnel for dielectric interface
 * @param cos_i Cosine of incident angle (positive)
 * @param eta Ratio of refractive indices (n_t / n_i)
 * @return FresnelResult with s and p coefficients
 */
inline FresnelResult fresnel_dielectric(float cos_i, float eta) {
    FresnelResult result;
    
    cos_i = std::abs(cos_i);
    
    // Check for total internal reflection
    float sin_i2 = 1.0f - cos_i * cos_i;
    float sin_t2 = sin_i2 / (eta * eta);
    
    if (sin_t2 >= 1.0f) {
        // Total internal reflection
        result.Rs = 1.0f;
        result.Rp = 1.0f;
        result.Ts = 0.0f;
        result.Tp = 0.0f;
        
        // Complex coefficients for TIR (phase shifts)
        float cos_t_imag = std::sqrt(sin_t2 - 1.0f);
        
        // r_s = (cos_i - i*|cos_t|) / (cos_i + i*|cos_t|)
        // |r_s|^2 = 1, but with phase shift
        float denom_s = cos_i * cos_i + cos_t_imag * cos_t_imag;
        result.rs_real = (cos_i * cos_i - cos_t_imag * cos_t_imag) / denom_s;
        result.rs_imag = -2.0f * cos_i * cos_t_imag / denom_s;
        
        // r_p = (eta^2*cos_i - i*|cos_t|) / (eta^2*cos_i + i*|cos_t|)
        float eta2_cos_i = eta * eta * cos_i;
        float denom_p = eta2_cos_i * eta2_cos_i + cos_t_imag * cos_t_imag;
        result.rp_real = (eta2_cos_i * eta2_cos_i - cos_t_imag * cos_t_imag) / denom_p;
        result.rp_imag = -2.0f * eta2_cos_i * cos_t_imag / denom_p;
        
        result.cos_t = 0.0f;
        return result;
    }
    
    float cos_t = std::sqrt(1.0f - sin_t2);
    result.cos_t = cos_t;
    
    // Fresnel amplitude coefficients
    // s-polarization: perpendicular to plane of incidence
    // r_s = (n_i*cos_i - n_t*cos_t) / (n_i*cos_i + n_t*cos_t)
    float rs_num = cos_i - eta * cos_t;
    float rs_den = cos_i + eta * cos_t;
    float rs = rs_num / rs_den;
    
    // p-polarization: parallel to plane of incidence
    // r_p = (n_t*cos_i - n_i*cos_t) / (n_t*cos_i + n_i*cos_t)
    float rp_num = eta * cos_i - cos_t;
    float rp_den = eta * cos_i + cos_t;
    float rp = rp_num / rp_den;
    
    // Transmission coefficients
    float ts = 2.0f * cos_i / rs_den;
    float tp = 2.0f * cos_i / rp_den;
    
    result.rs_real = rs;
    result.rp_real = rp;
    result.ts_real = ts;
    result.tp_real = tp;
    
    // Power coefficients
    result.Rs = rs * rs;
    result.Rp = rp * rp;
    
    // Transmittance accounts for beam compression
    float factor = (eta * cos_t) / cos_i;
    result.Ts = ts * ts * factor;
    result.Tp = tp * tp * factor;
    
    return result;
}

/**
 * Compute polarized Fresnel for conductor interface
 * @param cos_i Cosine of incident angle (positive)
 * @param eta Real part of complex refractive index
 * @param k Extinction coefficient (imaginary part)
 * @return FresnelResult with s and p coefficients
 */
inline FresnelResult fresnel_conductor(float cos_i, float eta, float k) {
    FresnelResult result;
    
    cos_i = std::abs(cos_i);
    float sin_i2 = 1.0f - cos_i * cos_i;
    
    // Complex arithmetic for conductor
    // n_t = eta + i*k
    // n_t^2 = eta^2 - k^2 + 2*i*eta*k
    float eta2_minus_k2 = eta * eta - k * k;
    float two_eta_k = 2.0f * eta * k;
    
    // sin_t^2 = sin_i^2 / n_t^2 (complex)
    // We need sqrt(n_t^2 - sin_i^2) = a + i*b
    float a2_minus_b2 = eta2_minus_k2 - sin_i2;
    float two_ab = two_eta_k;
    
    // |sqrt(x + iy)| = sqrt((|x+iy| + x) / 2) for real part
    float mag = std::sqrt(a2_minus_b2 * a2_minus_b2 + two_ab * two_ab);
    float a = std::sqrt(0.5f * (mag + a2_minus_b2));
    float b = std::sqrt(0.5f * (mag - a2_minus_b2));
    if (two_ab < 0) b = -b;
    
    // r_s = (cos_i - (a + ib)) / (cos_i + (a + ib))
    float rs_num_r = cos_i - a;
    float rs_num_i = -b;
    float rs_den_r = cos_i + a;
    float rs_den_i = b;
    float rs_den_mag2 = rs_den_r * rs_den_r + rs_den_i * rs_den_i;
    
    result.rs_real = (rs_num_r * rs_den_r + rs_num_i * rs_den_i) / rs_den_mag2;
    result.rs_imag = (rs_num_i * rs_den_r - rs_num_r * rs_den_i) / rs_den_mag2;
    result.Rs = result.rs_real * result.rs_real + result.rs_imag * result.rs_imag;
    
    // r_p = ((a + ib)*cos_i - 1) / ((a + ib)*cos_i + 1)
    // But we need n_t^2 * cos_i vs cos_t relation
    // Actually: r_p = (n_t^2*cos_i - (a+ib)) / (n_t^2*cos_i + (a+ib))
    float n2_cos_i_r = (eta2_minus_k2) * cos_i;
    float n2_cos_i_i = two_eta_k * cos_i;
    
    float rp_num_r = n2_cos_i_r - a;
    float rp_num_i = n2_cos_i_i - b;
    float rp_den_r = n2_cos_i_r + a;
    float rp_den_i = n2_cos_i_i + b;
    float rp_den_mag2 = rp_den_r * rp_den_r + rp_den_i * rp_den_i;
    
    result.rp_real = (rp_num_r * rp_den_r + rp_num_i * rp_den_i) / rp_den_mag2;
    result.rp_imag = (rp_num_i * rp_den_r - rp_num_r * rp_den_i) / rp_den_mag2;
    result.Rp = result.rp_real * result.rp_real + result.rp_imag * result.rp_imag;
    
    // Conductors don't transmit
    result.Ts = 0.0f;
    result.Tp = 0.0f;
    result.cos_t = 0.0f;
    
    return result;
}

/**
 * Compute frame rotation Mueller matrix
 * When light hits a surface, we need to rotate from the incident frame
 * to align with the plane of incidence
 * @param wi Incident direction (pointing toward surface)
 * @param normal Surface normal
 * @param tangent Current frame tangent (direction of s-polarization)
 * @param sin_rotation Output: sine of rotation angle
 * @param cos_rotation Output: cosine of rotation angle
 * @return Mueller rotation matrix
 */
inline Mueller frame_rotation_mueller(
    const float* wi,      // 3D vector
    const float* normal,  // 3D vector  
    const float* tangent, // 3D vector
    float& sin_rotation,
    float& cos_rotation
) {
    // Plane of incidence is defined by wi and normal
    // s-polarization is perpendicular to this plane
    
    // s direction = normalize(normal x wi)
    float s[3];
    s[0] = normal[1] * wi[2] - normal[2] * wi[1];
    s[1] = normal[2] * wi[0] - normal[0] * wi[2];
    s[2] = normal[0] * wi[1] - normal[1] * wi[0];
    
    float s_len = std::sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    
    if (s_len < 1e-8f) {
        // Normal incidence - no rotation needed
        sin_rotation = 0.0f;
        cos_rotation = 1.0f;
        return Mueller::identity();
    }
    
    float inv_s_len = 1.0f / s_len;
    s[0] *= inv_s_len;
    s[1] *= inv_s_len;
    s[2] *= inv_s_len;
    
    // cos(rotation) = tangent . s
    cos_rotation = tangent[0]*s[0] + tangent[1]*s[1] + tangent[2]*s[2];
    
    // sin(rotation) = (tangent x s) . wi
    float cross[3];
    cross[0] = tangent[1] * s[2] - tangent[2] * s[1];
    cross[1] = tangent[2] * s[0] - tangent[0] * s[2];
    cross[2] = tangent[0] * s[1] - tangent[1] * s[0];
    
    sin_rotation = cross[0]*wi[0] + cross[1]*wi[1] + cross[2]*wi[2];
    
    // Use double-angle identities: cos(2θ) = cos²θ - sin²θ, sin(2θ) = 2sinθcosθ
    float c2 = cos_rotation * cos_rotation - sin_rotation * sin_rotation;
    float s2 = 2.0f * sin_rotation * cos_rotation;
    
    return Mueller(
        1, 0, 0, 0,
        0, c2, s2, 0,
        0, -s2, c2, 0,
        0, 0, 0, 1
    );
}

} // namespace plt
