/**
 * @file mueller.hpp
 * @brief Mueller matrix operations for polarized light transforms
 *
 * Mueller matrices are 4x4 real matrices that transform Stokes vectors.
 * S_out = M * S_in
 *
 * Key Mueller matrices:
 * - Identity: passes light unchanged
 * - Rotation: rotates polarization reference frame
 * - Linear polarizer: filters to specific linear polarization
 * - Fresnel reflection/transmission: polarization-dependent interface interaction
 */

#pragma once

#include "stokes.hpp"
#include <cmath>
#include <array>

namespace plt {

/**
 * 4x4 Mueller matrix for polarization transforms
 * Stored row-major: m[row][col]
 */
struct Mueller {
    std::array<std::array<float, 4>, 4> m = {};

    Mueller() = default;

    // Construct from explicit values (row-major)
    Mueller(float m00, float m01, float m02, float m03,
            float m10, float m11, float m12, float m13,
            float m20, float m21, float m22, float m23,
            float m30, float m31, float m32, float m33) {
        m[0] = {m00, m01, m02, m03};
        m[1] = {m10, m11, m12, m13};
        m[2] = {m20, m21, m22, m23};
        m[3] = {m30, m31, m32, m33};
    }

    // Identity matrix
    static Mueller identity() {
        return Mueller(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );
    }

    // Zero matrix
    static Mueller zero() {
        return Mueller();
    }

    /**
     * Rotation matrix for reference frame rotation
     * Rotates the polarization reference plane by angle theta
     * @param theta Rotation angle in radians
     */
    static Mueller rotation(float theta) {
        float c = std::cos(2.0f * theta);
        float s = std::sin(2.0f * theta);
        return Mueller(
            1, 0, 0, 0,
            0, c, s, 0,
            0, -s, c, 0,
            0, 0, 0, 1
        );
    }

    /**
     * Linear polarizer at angle theta from horizontal
     * @param theta Polarizer axis angle in radians
     * @param extinction Extinction ratio (1 = ideal, 0 = transparent)
     */
    static Mueller linear_polarizer(float theta = 0.0f, float extinction = 1.0f) {
        float c2 = std::cos(2.0f * theta);
        float s2 = std::sin(2.0f * theta);
        float k = extinction;
        float h = 0.5f * k;
        return Mueller(
            h,       h * c2,       h * s2, 0,
            h * c2,  h * c2 * c2,  h * c2 * s2, 0,
            h * s2,  h * c2 * s2,  h * s2 * s2, 0,
            0, 0, 0, 0
        );
    }

    /**
     * Quarter-wave plate at angle theta
     * Converts linear to circular polarization and vice versa
     * @param theta Fast axis angle in radians
     */
    static Mueller quarter_wave_plate(float theta = 0.0f) {
        float c2 = std::cos(2.0f * theta);
        float s2 = std::sin(2.0f * theta);
        float c4 = std::cos(4.0f * theta);
        float s4 = std::sin(4.0f * theta);
        return Mueller(
            1, 0, 0, 0,
            0, 0.5f * (1 + c4), 0.5f * s4, s2,
            0, 0.5f * s4, 0.5f * (1 - c4), -c2,
            0, -s2, c2, 0
        );
    }

    /**
     * Half-wave plate at angle theta
     * Rotates linear polarization by 2*theta
     * @param theta Fast axis angle in radians
     */
    static Mueller half_wave_plate(float theta = 0.0f) {
        float c4 = std::cos(4.0f * theta);
        float s4 = std::sin(4.0f * theta);
        return Mueller(
            1, 0, 0, 0,
            0, c4, s4, 0,
            0, s4, -c4, 0,
            0, 0, 0, -1
        );
    }

    /**
     * Depolarizer (partial or complete)
     * @param factor Depolarization factor (1 = full depolarization, 0 = no effect)
     */
    static Mueller depolarizer(float factor = 1.0f) {
        float k = 1.0f - factor;
        return Mueller(
            1, 0, 0, 0,
            0, k, 0, 0,
            0, 0, k, 0,
            0, 0, 0, k
        );
    }

    /**
     * Mueller matrix from Fresnel coefficients
     * For reflection at dielectric interface
     * @param rs Fresnel coefficient for s-polarization
     * @param rp Fresnel coefficient for p-polarization
     */
    static Mueller fresnel_reflection(float rs, float rp) {
        float rs2 = rs * rs;
        float rp2 = rp * rp;
        float h = 0.5f;
        return Mueller(
            h * (rs2 + rp2), h * (rs2 - rp2), 0, 0,
            h * (rs2 - rp2), h * (rs2 + rp2), 0, 0,
            0, 0, rs * rp, 0,
            0, 0, 0, rs * rp
        );
    }

    /**
     * Mueller matrix from complex Fresnel coefficients
     * Handles phase differences between s and p
     * @param rs_real Real part of s-polarization coefficient
     * @param rs_imag Imaginary part of s-polarization coefficient
     * @param rp_real Real part of p-polarization coefficient
     * @param rp_imag Imaginary part of p-polarization coefficient
     */
    static Mueller fresnel_reflection_complex(
        float rs_real, float rs_imag,
        float rp_real, float rp_imag
    ) {
        // Magnitudes
        float rs2 = rs_real * rs_real + rs_imag * rs_imag;
        float rp2 = rp_real * rp_real + rp_imag * rp_imag;
        
        // Phase difference: delta = arg(rs) - arg(rp)
        // cos(delta) = Re(rs * conj(rp)) / (|rs| |rp|)
        // sin(delta) = Im(rs * conj(rp)) / (|rs| |rp|)
        float rs_conj_rp_real = rs_real * rp_real + rs_imag * rp_imag;
        float rs_conj_rp_imag = rs_imag * rp_real - rs_real * rp_imag;
        
        float rs_mag = std::sqrt(rs2);
        float rp_mag = std::sqrt(rp2);
        float mag_prod = rs_mag * rp_mag;
        
        float cos_delta = (mag_prod > 0.0f) ? rs_conj_rp_real / mag_prod : 1.0f;
        float sin_delta = (mag_prod > 0.0f) ? rs_conj_rp_imag / mag_prod : 0.0f;
        
        float h = 0.5f;
        return Mueller(
            h * (rs2 + rp2), h * (rs2 - rp2), 0, 0,
            h * (rs2 - rp2), h * (rs2 + rp2), 0, 0,
            0, 0, mag_prod * cos_delta, mag_prod * sin_delta,
            0, 0, -mag_prod * sin_delta, mag_prod * cos_delta
        );
    }

    /**
     * Mueller matrix from Fresnel transmission coefficients
     * @param ts Fresnel transmission coefficient for s-polarization
     * @param tp Fresnel transmission coefficient for p-polarization
     * @param eta_ratio Ratio of refractive indices (n2/n1)
     * @param cos_i Cosine of incident angle
     * @param cos_t Cosine of transmitted angle
     */
    static Mueller fresnel_transmission(float ts, float tp, float eta_ratio, float cos_i, float cos_t) {
        float ts2 = ts * ts;
        float tp2 = tp * tp;
        // Account for beam compression/expansion at interface
        float factor = (eta_ratio * cos_t) / cos_i;
        float h = 0.5f * factor;
        return Mueller(
            h * (ts2 + tp2), h * (ts2 - tp2), 0, 0,
            h * (ts2 - tp2), h * (ts2 + tp2), 0, 0,
            0, 0, factor * ts * tp, 0,
            0, 0, 0, factor * ts * tp
        );
    }

    // Apply Mueller matrix to Stokes vector
    Stokes operator*(const Stokes& s) const {
        return Stokes(
            m[0][0] * s.I + m[0][1] * s.Q + m[0][2] * s.U + m[0][3] * s.V,
            m[1][0] * s.I + m[1][1] * s.Q + m[1][2] * s.U + m[1][3] * s.V,
            m[2][0] * s.I + m[2][1] * s.Q + m[2][2] * s.U + m[2][3] * s.V,
            m[3][0] * s.I + m[3][1] * s.Q + m[3][2] * s.U + m[3][3] * s.V
        );
    }

    // Matrix multiplication
    Mueller operator*(const Mueller& other) const {
        Mueller result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.m[i][j] = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }

    Mueller operator*(float s) const {
        Mueller result = *this;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.m[i][j] *= s;
        return result;
    }

    Mueller operator+(const Mueller& other) const {
        Mueller result;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.m[i][j] = m[i][j] + other.m[i][j];
        return result;
    }

    // Element access
    float& operator()(int row, int col) { return m[row][col]; }
    float operator()(int row, int col) const { return m[row][col]; }
};

inline Mueller operator*(float s, const Mueller& mat) {
    return mat * s;
}

// ============================================================================
// Thin-Film Interference
// ============================================================================

/**
 * Compute thin-film interference Mueller matrix
 * 
 * Models a thin dielectric film on a substrate. Light reflects from both
 * the top surface (air-film) and bottom surface (film-substrate), creating
 * interference patterns that depend on wavelength and angle.
 * 
 * @param cos_i Cosine of incident angle
 * @param n_film Refractive index of thin film
 * @param n_substrate Refractive index of substrate (can be complex for metals)
 * @param thickness_nm Film thickness in nanometers
 * @param wavelength_nm Wavelength of light in nanometers
 * @return Mueller matrix for the thin-film reflection
 */
inline Mueller thin_film_reflection(
    float cos_i,
    float n_film,
    float n_substrate,
    float thickness_nm,
    float wavelength_nm
) {
    // Snell's law: sin(theta_i) = n_film * sin(theta_film)
    float sin_i = std::sqrt(1.0f - cos_i * cos_i);
    float sin_film = sin_i / n_film;
    
    // Check for TIR at air-film interface (shouldn't happen for n_film > 1)
    if (sin_film > 1.0f) {
        return Mueller::fresnel_reflection(1.0f, 1.0f); // TIR
    }
    
    float cos_film = std::sqrt(1.0f - sin_film * sin_film);
    
    // Fresnel coefficients at air-film interface (external)
    // r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    // r_p = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
    float n1 = 1.0f;  // air
    float n2 = n_film;
    
    float rs_12 = (n1 * cos_i - n2 * cos_film) / (n1 * cos_i + n2 * cos_film);
    float rp_12 = (n2 * cos_i - n1 * cos_film) / (n2 * cos_i + n1 * cos_film);
    
    // Transmission coefficients (for amplitude, not intensity)
    float ts_12 = 2.0f * n1 * cos_i / (n1 * cos_i + n2 * cos_film);
    float tp_12 = 2.0f * n1 * cos_i / (n2 * cos_i + n1 * cos_film);
    
    // Snell's law at film-substrate interface
    float sin_sub = sin_film * n_film / n_substrate;
    float cos_sub;
    bool tir_sub = (sin_sub > 1.0f);
    
    float rs_23, rp_23;
    if (tir_sub) {
        // TIR at film-substrate interface
        rs_23 = 1.0f;
        rp_23 = 1.0f;
        cos_sub = 0.0f;
    } else {
        cos_sub = std::sqrt(1.0f - sin_sub * sin_sub);
        rs_23 = (n_film * cos_film - n_substrate * cos_sub) / (n_film * cos_film + n_substrate * cos_sub);
        rp_23 = (n_substrate * cos_film - n_film * cos_sub) / (n_substrate * cos_film + n_film * cos_sub);
    }
    
    // Transmission back from film to air
    float ts_21 = 2.0f * n2 * cos_film / (n2 * cos_film + n1 * cos_i);
    float tp_21 = 2.0f * n2 * cos_film / (n1 * cos_film + n2 * cos_i);
    
    // Phase shift from traveling through film twice
    // path = 2 * thickness * n_film * cos(theta_film)
    // phase = 2 * pi * path / wavelength
    constexpr float PI = 3.14159265358979323846f;
    float optical_path = 2.0f * thickness_nm * n_film * cos_film;
    float phase = 2.0f * PI * optical_path / wavelength_nm;
    
    // Total reflection amplitude: r = r12 + t12 * r23 * t21 * exp(i*phase) / (1 - r21*r23*exp(i*phase))
    // For first-order approximation (single reflection in film):
    // r ≈ r12 + t12 * r23 * t21 * exp(i*phase)
    
    // Amplitude of film-reflected ray (travels through film twice)
    float film_amp_s = ts_12 * rs_23 * ts_21;
    float film_amp_p = tp_12 * rp_23 * tp_21;
    
    // Complex addition: r_total = r12 + film_amp * exp(i*phase)
    // Using: a + b*exp(i*phi) = sqrt(a^2 + b^2 + 2ab*cos(phi)) * exp(i*theta)
    float cos_phase = std::cos(phase);
    float sin_phase = std::sin(phase);
    
    // For s-polarization
    float rs_total_real = rs_12 + film_amp_s * cos_phase;
    float rs_total_imag = film_amp_s * sin_phase;
    
    // For p-polarization  
    float rp_total_real = rp_12 + film_amp_p * cos_phase;
    float rp_total_imag = film_amp_p * sin_phase;
    
    return Mueller::fresnel_reflection_complex(
        rs_total_real, rs_total_imag,
        rp_total_real, rp_total_imag
    );
}

/**
 * Compute thin-film interference for a coated conductor (metal substrate)
 * 
 * @param cos_i Cosine of incident angle
 * @param n_film Refractive index of thin film coating
 * @param n_metal Real part of metal refractive index
 * @param k_metal Extinction coefficient of metal
 * @param thickness_nm Film thickness in nanometers
 * @param wavelength_nm Wavelength of light in nanometers
 */
inline Mueller thin_film_on_metal(
    float cos_i,
    float n_film,
    float n_metal,
    float k_metal,
    float thickness_nm,
    float wavelength_nm
) {
    constexpr float PI = 3.14159265358979323846f;
    
    // Snell's law at air-film interface
    float sin_i = std::sqrt(1.0f - cos_i * cos_i);
    float sin_film = sin_i / n_film;
    
    if (sin_film > 1.0f) {
        return Mueller::fresnel_reflection(1.0f, 1.0f);
    }
    
    float cos_film = std::sqrt(1.0f - sin_film * sin_film);
    
    // Air-film Fresnel coefficients
    float n1 = 1.0f;
    float n2 = n_film;
    
    float rs_12 = (n1 * cos_i - n2 * cos_film) / (n1 * cos_i + n2 * cos_film);
    float rp_12 = (n2 * cos_i - n1 * cos_film) / (n2 * cos_i + n1 * cos_film);
    
    float ts_12 = 2.0f * n1 * cos_i / (n1 * cos_i + n2 * cos_film);
    float tp_12 = 2.0f * n1 * cos_i / (n2 * cos_i + n1 * cos_film);
    
    float ts_21 = 2.0f * n2 * cos_film / (n2 * cos_film + n1 * cos_i);
    float tp_21 = 2.0f * n2 * cos_film / (n1 * cos_film + n2 * cos_i);
    
    // Film-metal interface: complex Fresnel for conductor
    // For metal: n_complex = n_metal + i*k_metal
    // Simplified conductor Fresnel at film-metal interface
    float n_ratio = n_metal / n_film;
    float k_ratio = k_metal / n_film;
    
    // Conductor Fresnel (simplified for normal-ish incidence in film)
    float denom_s = (cos_film + n_ratio) * (cos_film + n_ratio) + k_ratio * k_ratio;
    float denom_p = (n_ratio * cos_film + 1.0f) * (n_ratio * cos_film + 1.0f) + (k_ratio * cos_film) * (k_ratio * cos_film);
    
    // |r_s|^2 and |r_p|^2 for metal
    float Rs_23 = ((cos_film - n_ratio) * (cos_film - n_ratio) + k_ratio * k_ratio) / denom_s;
    float Rp_23 = ((n_ratio * cos_film - 1.0f) * (n_ratio * cos_film - 1.0f) + (k_ratio * cos_film) * (k_ratio * cos_film)) / denom_p;
    
    float rs_23 = std::sqrt(Rs_23);
    float rp_23 = std::sqrt(Rp_23);
    
    // Phase from film travel
    float optical_path = 2.0f * thickness_nm * n_film * cos_film;
    float phase = 2.0f * PI * optical_path / wavelength_nm;
    
    // Film contribution amplitude
    float film_amp_s = ts_12 * rs_23 * ts_21;
    float film_amp_p = tp_12 * rp_23 * tp_21;
    
    float cos_phase = std::cos(phase);
    float sin_phase = std::sin(phase);
    
    float rs_total_real = rs_12 + film_amp_s * cos_phase;
    float rs_total_imag = film_amp_s * sin_phase;
    
    float rp_total_real = rp_12 + film_amp_p * cos_phase;
    float rp_total_imag = film_amp_p * sin_phase;
    
    return Mueller::fresnel_reflection_complex(
        rs_total_real, rs_total_imag,
        rp_total_real, rp_total_imag
    );
}

} // namespace plt
