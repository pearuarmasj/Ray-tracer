/**
 * @file plt_test.cpp
 * @brief Simple tests for PLT module
 * 
 * Verifies:
 * - Stokes vector operations
 * - Mueller matrix multiplication
 * - Polarized Fresnel equations
 * - Beam frame rotation
 */

// Define M_PI for MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "engine/plt/stokes.hpp"
#include "engine/plt/mueller.hpp"
#include "engine/plt/fresnel.hpp"
#include "engine/plt/beam.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace plt;

constexpr float EPSILON = 1e-5f;

bool approx_equal(float a, float b, float eps = EPSILON) {
    return std::abs(a - b) < eps;
}

void test_stokes_creation() {
    std::cout << "Testing Stokes creation...\n";
    
    // Unpolarized light
    Stokes s1 = Stokes::unpolarized(1.0f);
    assert(approx_equal(s1.I, 1.0f));
    assert(approx_equal(s1.Q, 0.0f));
    assert(approx_equal(s1.U, 0.0f));
    assert(approx_equal(s1.V, 0.0f));
    assert(approx_equal(s1.degree_of_polarization(), 0.0f));
    
    // Horizontal polarization
    Stokes s2 = Stokes::horizontal(1.0f);
    assert(approx_equal(s2.I, 1.0f));
    assert(approx_equal(s2.Q, 1.0f));
    assert(approx_equal(s2.degree_of_polarization(), 1.0f));
    
    // Vertical polarization
    Stokes s3 = Stokes::vertical(1.0f);
    assert(approx_equal(s3.Q, -1.0f));
    
    // Circular polarization
    Stokes s4 = Stokes::circular_right(1.0f);
    assert(approx_equal(s4.V, 1.0f));
    assert(approx_equal(s4.degree_of_circular_polarization(), 1.0f));
    
    std::cout << "  PASSED\n";
}

void test_mueller_polarizer() {
    std::cout << "Testing Mueller polarizer...\n";
    
    // Ideal horizontal polarizer
    Mueller pol = Mueller::linear_polarizer(0.0f, 1.0f);
    
    // Unpolarized light through polarizer -> half intensity, horizontal
    Stokes unpol = Stokes::unpolarized(2.0f);
    Stokes result = pol * unpol;
    
    assert(approx_equal(result.I, 1.0f));
    assert(approx_equal(result.Q, 1.0f));
    assert(approx_equal(result.degree_of_polarization(), 1.0f));
    
    // Horizontal through horizontal -> passes
    Stokes horiz = Stokes::horizontal(1.0f);
    Stokes r2 = pol * horiz;
    assert(approx_equal(r2.I, 1.0f, 0.01f));
    
    // Vertical through horizontal -> blocked
    Stokes vert = Stokes::vertical(1.0f);
    Stokes r3 = pol * vert;
    assert(approx_equal(r3.I, 0.0f, 0.01f));
    
    std::cout << "  PASSED\n";
}

void test_mueller_rotation() {
    std::cout << "Testing Mueller rotation...\n";
    
    // Rotate horizontal by 45 degrees -> +45 degree polarization
    Mueller rot = Mueller::rotation(static_cast<float>(M_PI) / 4.0f);
    Stokes horiz = Stokes::horizontal(1.0f);
    Stokes result = rot * horiz;
    
    // After 45 deg rotation: Q->0, U->1
    assert(approx_equal(result.I, 1.0f));
    assert(approx_equal(result.Q, 0.0f, 0.01f));
    assert(approx_equal(std::abs(result.U), 1.0f, 0.01f));
    
    // Rotate by 90 degrees -> vertical
    Mueller rot90 = Mueller::rotation(static_cast<float>(M_PI) / 2.0f);
    Stokes r2 = rot90 * horiz;
    assert(approx_equal(r2.Q, -1.0f, 0.01f));
    
    std::cout << "  PASSED\n";
}

void test_fresnel_normal_incidence() {
    std::cout << "Testing Fresnel at normal incidence...\n";
    
    // Glass (n=1.5) at normal incidence
    FresnelResult f = fresnel_dielectric(1.0f, 1.5f);
    
    // At normal incidence, Rs = Rp
    assert(approx_equal(f.Rs, f.Rp, 0.001f));
    
    // R = ((n-1)/(n+1))^2 = 0.04
    float expected_R = 0.04f;
    assert(approx_equal(f.R(), expected_R, 0.01f));
    
    // Energy conservation
    assert(approx_equal(f.R() + f.T(), 1.0f, 0.01f));
    
    std::cout << "  PASSED\n";
}

void test_fresnel_brewster() {
    std::cout << "Testing Fresnel at Brewster angle...\n";
    
    // Brewster angle for glass (n=1.5): tan(theta_B) = n
    float theta_B = std::atan(1.5f);
    float cos_B = std::cos(theta_B);
    
    FresnelResult f = fresnel_dielectric(cos_B, 1.5f);
    
    // At Brewster angle, Rp -> 0
    assert(f.Rp < 0.01f);
    
    // Rs should be significant
    assert(f.Rs > 0.05f);
    
    std::cout << "  PASSED\n";
}

void test_fresnel_tir() {
    std::cout << "Testing Fresnel TIR...\n";
    
    // Inside glass looking out at grazing angle
    // Critical angle: sin(theta_c) = 1/n = 1/1.5 = 0.667
    float theta_c = std::asin(1.0f / 1.5f);
    float cos_beyond_critical = std::cos(theta_c + 0.1f);
    
    // From glass to air (eta = 1/1.5)
    FresnelResult f = fresnel_dielectric(cos_beyond_critical, 1.0f / 1.5f);
    
    // Total internal reflection
    assert(approx_equal(f.R(), 1.0f, 0.01f));
    assert(approx_equal(f.T(), 0.0f, 0.01f));
    
    std::cout << "  PASSED\n";
}

void test_beam_frame() {
    std::cout << "Testing Beam frame operations...\n";
    
    // Create beam along Z axis
    Beam b = Beam::unpolarized(1.0f, Vec3(0, 0, 1));
    
    // Tangent should be perpendicular to direction
    float dot_result = dot(b.tangent, b.direction);
    assert(approx_equal(dot_result, 0.0f, 0.01f));
    
    // Bitangent should also be perpendicular
    Vec3 bitangent = b.bitangent();
    assert(approx_equal(dot(bitangent, b.direction), 0.0f, 0.01f));
    assert(approx_equal(dot(bitangent, b.tangent), 0.0f, 0.01f));
    
    std::cout << "  PASSED\n";
}

void test_polarizer_chain() {
    std::cout << "Testing crossed polarizers...\n";
    
    // Two crossed polarizers (Malus's law)
    Mueller pol_h = Mueller::linear_polarizer(0.0f);
    Mueller pol_v = Mueller::linear_polarizer(static_cast<float>(M_PI) / 2.0f);
    
    // Crossed polarizers block all light
    Mueller crossed = pol_v * pol_h;
    Stokes unpol = Stokes::unpolarized(1.0f);
    Stokes result = crossed * unpol;
    
    assert(approx_equal(result.I, 0.0f, 0.01f));
    
    // 45 degree polarizer between crossed -> 1/8 intensity
    Mueller pol_45 = Mueller::linear_polarizer(static_cast<float>(M_PI) / 4.0f);
    Mueller chain = pol_v * pol_45 * pol_h;
    Stokes r2 = chain * unpol;
    
    // Expected: 0.5 * 0.5 * 0.5 = 0.125
    assert(approx_equal(r2.I, 0.125f, 0.02f));
    
    std::cout << "  PASSED\n";
}

int main() {
    std::cout << "=== PLT Module Tests ===\n\n";
    
    test_stokes_creation();
    test_mueller_polarizer();
    test_mueller_rotation();
    test_fresnel_normal_incidence();
    test_fresnel_brewster();
    test_fresnel_tir();
    test_beam_frame();
    test_polarizer_chain();
    
    std::cout << "\n=== All tests PASSED ===\n";
    return 0;
}
