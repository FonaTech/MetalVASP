#pragma once
#include "metal_backend.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct metal_nonlocal_batch metal_nonlocal_batch;
typedef struct {
    size_t npl, bands, nions, npro, qchannels, ntypes;
} metal_nonlocal_shape;
typedef struct {
    uint64_t wave_generation, phase_generation;
    const metal_complex64 *wave; size_t ldw;
    const double *radial; size_t qld;
    const metal_complex64 *phase; size_t phase_ld;
    const metal_complex64 *cq;
    const int *types, *lmmax;
    const int64_t *bases;
    /* Optional three column-major npl derivative factors, including k shift. */
    const double *gradient;
    metal_complex64 derivative_multiplier;
} metal_nonlocal_input;
typedef struct {
    uint64_t calls, completed_commands, wave_import_bytes, radial_import_bytes;
    uint64_t phase_import_bytes, metadata_import_bytes, gradient_import_bytes;
    uint64_t export_bytes, wave_reuses, phase_reuses, windows, planned_workspace_peak_bytes;
} metal_nonlocal_stats;
/* Complex, single spinor local projection. No MPI collective is hidden here.
   Output layout: [channel][band][projector], ld >= npro. channels=1 or4.
   Generations are caller promises; change them after any corresponding mutation.
   All host output is committed together only after every window succeeds. */
int metal_nonlocal_create(const metal_backend *, metal_nonlocal_shape, metal_nonlocal_batch **);
void metal_nonlocal_destroy(metal_nonlocal_batch *);
int metal_nonlocal_project(metal_nonlocal_batch *, const metal_nonlocal_input *,
                          unsigned channels, metal_complex64 *output, size_t ld);
void metal_nonlocal_get_stats(const metal_nonlocal_batch *, metal_nonlocal_stats *);
#ifdef __cplusplus
}
#endif
