#pragma once
#include "metal_backend.h"
#include "metal_wave_owner_davidson.h"
#ifdef __cplusplus
extern "C" {
#endif
void vasp_metal_davidson_release(void **);
void vasp_metal_davidson_invalidate(void);
typedef struct vasp_metal_davidson_counters {
 uint64_t calls,success,allocations,reuses,releases,live,imports,exports,basis_imports;
 uint64_t static_uploads,static_reuses,cached_entries,retained_cache_bytes,static_validation_bytes,submissions,control_status_host_bytes,overlap_zero_init_host_bytes;
} vasp_metal_davidson_counters;
void vasp_metal_davidson_get_counters(vasp_metal_davidson_counters *);
int vasp_metal_davidson_prepare(void **,uint64_t,int,int,int,int,int,int,int,int,int,int,int,
 metal_complex64 *const *,metal_complex64 *const *,const double *,int,const int *,const int *,const int *,
 const double *,const metal_complex64 *,const metal_complex64 *,const void *,metal_complex64 *,int,double *,int,
 const metal_complex64 *,const metal_complex64 *,int,int,metal_complex64 *const *,metal_complex64 *const *,int,int,int);
/* Appended basis epoch and zero-based dirty span. Epoch gaps refresh all.
   The original prepare signature and counters struct remain unchanged. */
int vasp_metal_davidson_prepare_delta(void **,uint64_t,int,int,int,int,int,int,int,int,int,int,int,
 metal_complex64 *const *,metal_complex64 *const *,const double *,int,const int *,const int *,const int *,
 const double *,const metal_complex64 *,const metal_complex64 *,const void *,metal_complex64 *,int,double *,int,
 const metal_complex64 *,const metal_complex64 *,int,int,metal_complex64 *const *,metal_complex64 *const *,int,int,int,uint64_t,int,int);
void vasp_metal_davidson_get_basis_counters(metal_wave_basis_transfer_stats *);
#ifdef __cplusplus
}
#endif
