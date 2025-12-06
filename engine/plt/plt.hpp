/**
 * @file plt.hpp
 * @brief Polarized Light Tracing module
 *
 * Main include header for the PLT module. Provides:
 * - Stokes vectors (4-component polarization state)
 * - Mueller matrices (4x4 polarization transforms)
 * - Polarized Fresnel equations
 * - PLT beam structure
 * - PLT path integrator
 *
 * Usage:
 *   #include "plt/plt.hpp"
 *   
 *   plt::PLTIntegrator integrator;
 *   plt::Stokes result = integrator.Li(scene, ray, plt_props, rng);
 */

#pragma once

#include "stokes.hpp"
#include "mueller.hpp"
#include "fresnel.hpp"
#include "beam.hpp"
#include "vec3.hpp"
#include "plt_integrator.hpp"
