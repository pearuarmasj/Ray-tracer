/**
 * @file vec3.c
 * @brief Implementation of 3D vector operations
 */

#include "vec3.h"
#include <math.h>

vec3 vec3_create(double x, double y, double z) {
    return (vec3){x, y, z};
}

vec3 vec3_zero(void) {
    return (vec3){0.0, 0.0, 0.0};
}

vec3 vec3_add(vec3 a, vec3 b) {
    return (vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 vec3_sub(vec3 a, vec3 b) {
    return (vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 vec3_mul(vec3 a, vec3 b) {
    return (vec3){a.x * b.x, a.y * b.y, a.z * b.z};
}

vec3 vec3_scale(vec3 v, double t) {
    return (vec3){v.x * t, v.y * t, v.z * t};
}

vec3 vec3_negate(vec3 v) {
    return (vec3){-v.x, -v.y, -v.z};
}

double vec3_dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 vec3_cross(vec3 a, vec3 b) {
    return (vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

double vec3_length_squared(vec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

double vec3_length(vec3 v) {
    return sqrt(vec3_length_squared(v));
}

vec3 vec3_normalize(vec3 v) {
    double len = vec3_length(v);
    if (len > 0.0) {
        return vec3_scale(v, 1.0 / len);
    }
    return v;
}

vec3 vec3_reflect(vec3 v, vec3 n) {
    return vec3_sub(v, vec3_scale(n, 2.0 * vec3_dot(v, n)));
}

vec3 vec3_refract(vec3 uv, vec3 n, double etai_over_etat) {
    double cos_theta = fmin(vec3_dot(vec3_negate(uv), n), 1.0);
    vec3 r_out_perp = vec3_scale(
        vec3_add(uv, vec3_scale(n, cos_theta)),
        etai_over_etat
    );
    double perp_len_sq = vec3_length_squared(r_out_perp);
    double parallel_factor = -sqrt(fabs(1.0 - perp_len_sq));
    vec3 r_out_parallel = vec3_scale(n, parallel_factor);
    return vec3_add(r_out_perp, r_out_parallel);
}

vec3 vec3_lerp(vec3 a, vec3 b, double t) {
    return vec3_add(vec3_scale(a, 1.0 - t), vec3_scale(b, t));
}

vec3 vec3_clamp(vec3 v, double min_val, double max_val) {
    return (vec3){
        fmin(fmax(v.x, min_val), max_val),
        fmin(fmax(v.y, min_val), max_val),
        fmin(fmax(v.z, min_val), max_val)
    };
}
