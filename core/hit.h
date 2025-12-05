/**
 * @file hit.h
 * @brief Hit record structure for ray-object intersections
 */

#ifndef HIT_H
#define HIT_H

#include "vec3.h"
#include "ray.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Hit record containing intersection information
 */
typedef struct hit_record {
    point3 point;      /**< Point of intersection */
    vec3 normal;       /**< Surface normal at intersection */
    double t;          /**< Ray parameter at intersection */
    double u;          /**< Texture U coordinate */
    double v;          /**< Texture V coordinate */
    bool front_face;   /**< True if ray hit front face */
    int material_id;   /**< Material identifier for the hit surface */
} hit_record;

/**
 * @brief Initialize a hit record
 * @return Default hit record
 */
hit_record hit_record_init(void);

/**
 * @brief Set the face normal based on ray direction
 * 
 * This function sets the normal to always point against the ray direction,
 * and sets front_face to indicate which side of the surface was hit.
 * 
 * @param rec Hit record to modify
 * @param r The ray
 * @param outward_normal The outward-pointing surface normal
 */
void hit_record_set_face_normal(hit_record* rec, ray r, vec3 outward_normal);

#ifdef __cplusplus
}
#endif

#endif /* HIT_H */
