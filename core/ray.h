/**
 * @file ray.h
 * @brief Ray structure and operations for ray tracing
 */

#ifndef RAY_H
#define RAY_H

#include "vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Ray structure with origin and direction
 */
typedef struct ray {
    point3 origin;
    vec3 direction;
} ray;

/**
 * @brief Create a new ray
 * @param origin Ray origin point
 * @param direction Ray direction vector
 * @return New ray
 */
ray ray_create(point3 origin, vec3 direction);

/**
 * @brief Get point along ray at parameter t
 * @param r The ray
 * @param t Parameter value (distance along ray)
 * @return Point at r.origin + t * r.direction
 */
point3 ray_at(ray r, double t);

#ifdef __cplusplus
}
#endif

#endif /* RAY_H */
