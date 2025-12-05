/**
 * @file vec3.h
 * @brief 3D vector mathematics for ray tracing
 * 
 * Provides a vec3 structure and common vector operations
 * for use in ray tracing calculations.
 */

#ifndef VEC3_H
#define VEC3_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 3D vector structure
 */
typedef struct vec3 {
    double x;
    double y;
    double z;
} vec3;

/**
 * @brief Color alias for vec3 (r, g, b stored in x, y, z)
 */
typedef vec3 color;

/**
 * @brief Point alias for vec3
 */
typedef vec3 point3;

/* Construction */
vec3 vec3_create(double x, double y, double z);
vec3 vec3_zero(void);

/* Basic operations */
vec3 vec3_add(vec3 a, vec3 b);
vec3 vec3_sub(vec3 a, vec3 b);
vec3 vec3_mul(vec3 a, vec3 b);
vec3 vec3_scale(vec3 v, double t);
vec3 vec3_negate(vec3 v);

/* Vector products */
double vec3_dot(vec3 a, vec3 b);
vec3 vec3_cross(vec3 a, vec3 b);

/* Length operations */
double vec3_length(vec3 v);
double vec3_length_squared(vec3 v);
vec3 vec3_normalize(vec3 v);

/* Reflection and refraction */
vec3 vec3_reflect(vec3 v, vec3 n);
vec3 vec3_refract(vec3 uv, vec3 n, double etai_over_etat);

/* Utility */
vec3 vec3_lerp(vec3 a, vec3 b, double t);
vec3 vec3_clamp(vec3 v, double min_val, double max_val);

#ifdef __cplusplus
}
#endif

#endif /* VEC3_H */
