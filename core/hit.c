/**
 * @file hit.c
 * @brief Implementation of hit record operations
 */

#include "hit.h"

hit_record hit_record_init(void) {
    hit_record rec;
    rec.point = vec3_zero();
    rec.normal = vec3_zero();
    rec.tangent = (vec3){1.0, 0.0, 0.0};  // Default tangent along X-axis
    rec.t = 0.0;
    rec.u = 0.0;
    rec.v = 0.0;
    rec.front_face = true;
    rec.material_id = 0;
    return rec;
}

void hit_record_set_face_normal(hit_record* rec, ray r, vec3 outward_normal) {
    rec->front_face = vec3_dot(r.direction, outward_normal) < 0;
    rec->normal = rec->front_face ? outward_normal : vec3_negate(outward_normal);
}
