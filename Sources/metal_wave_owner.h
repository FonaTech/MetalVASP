#pragma once
#include "metal_backend.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct metal_wave_owner metal_wave_owner;
typedef struct metal_wave_owner_shape {
    size_t npl,bands,nions,npro,lmdim,ntypes,qchannels;
} metal_wave_owner_shape;
typedef struct metal_wave_prepare_input {
    const metal_complex64 *const *wave;
    const double *preconditioner;
    size_t ldpreconditioner,qld,phase_ld;
    const int *types,*lmmax,*bases;
    const double *radial;
    const metal_complex64 *cq,*phase;
    const void *metric;
    int real_metric,use_overlap;
} metal_wave_prepare_input;
typedef struct metal_wave_owner_stats {
    uint64_t generation,paw_epoch,steps,host_import_bytes,host_export_bytes;
    uint64_t static_uploads,static_reuses,invalidations,device_bytes;
} metal_wave_owner_stats;
/* Serial complex reciprocal-space PAW owner, one spinor and one local rank.
   backend must outlive owner. generation is positive and strictly increasing.
   paw_epoch must change before radial/CQ/phase/type/base data are mutated.
   metric and residual/preconditioner are imported fresh on every prepare.
   prepare performs precondition -> physical PAW projection -> Q metric ->
   PAW norm -> normalization in one submission, retaining all intermediates.
   download validates all outputs before modifying any host destination. */
int metal_wave_owner_create(const metal_backend *,metal_wave_owner_shape,metal_wave_owner **);
void metal_wave_owner_destroy(metal_wave_owner *);
void metal_wave_owner_invalidate(metal_wave_owner *);
int metal_wave_owner_matches(const metal_wave_owner *,metal_wave_owner_shape);
int metal_wave_owner_prepare(metal_wave_owner *,uint64_t generation,uint64_t paw_epoch,const metal_wave_prepare_input *);
int metal_wave_owner_download(metal_wave_owner *,uint64_t generation,metal_complex64 *const *wave,
    metal_complex64 *const *projectors,metal_complex64 *metric_projection,size_t ldmetric,double *scales);
void metal_wave_owner_get_stats(const metal_wave_owner *,metal_wave_owner_stats *);
#ifdef __cplusplus
}
#endif
