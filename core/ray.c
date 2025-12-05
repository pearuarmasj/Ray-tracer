/**
 * @file ray.c
 * @brief Implementation of ray operations
 */

#include "ray.h"

ray ray_create(point3 origin, vec3 direction) {
    return (ray){origin, direction};
}

point3 ray_at(ray r, double t) {
    return vec3_add(r.origin, vec3_scale(r.direction, t));
}
