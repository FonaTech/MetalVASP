#pragma once
#include "metal_wave_owner.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct metal_wave_basis_input {
 size_t bands,ldwave,ldprojectors;
 const metal_complex64 *wave,*projectors;
 uint64_t generation;
} metal_wave_basis_input;
/* A precise mutation since previous_generation. Zero columns with a new
   generation requests a full refresh. Columns use zero-based indexing. */
typedef struct metal_wave_basis_delta {
 uint64_t previous_generation;
 size_t first_column,column_count;
} metal_wave_basis_delta;
typedef struct metal_wave_basis_transfer_stats {
 uint64_t full_uploads,partial_uploads,reuses,uploaded_columns;
} metal_wave_basis_transfer_stats;
typedef struct metal_wave_davidson_stats {
 uint64_t steps,basis_generation,basis_uploads,basis_reuses,basis_import_bytes;
 uint64_t overlap_export_bytes,device_bytes,control_status_host_bytes,overlap_zero_init_host_bytes;
} metal_wave_davidson_stats;
/* Matches the one-rank EDDAV residual ORTH1 and its two beta=1 GEMM updates.
   Requires a successful, unexported preparation with the same generation.
   C = basis.CW^H * CW + basis.CPROJ^H * (Q * CPROJ),
   CW -= basis.CW * C; CPROJ -= basis.CPROJ * C.
   No second MGS pass or re-normalization is added. Original normalized Q*P
   remains in metric_projection, matching original WS before the update.
   A basis generation may be reused only while all its contents are unchanged.
   Failure invalidates output; no caller-owned host destination is modified. */
int metal_wave_owner_davidson_update(metal_wave_owner *,uint64_t generation,const metal_wave_basis_input *);
/* Eight kernels in one command, with only a final control-status check before
   the separately requested atomic host export. No intermediate CPU gate. */
int metal_wave_owner_davidson_prepare(metal_wave_owner *,uint64_t generation,uint64_t paw_epoch,
 const metal_wave_prepare_input *,const metal_wave_basis_input *);
/* Additive API: the original ABI and full-refresh semantics remain valid. */
int metal_wave_owner_davidson_prepare_delta(metal_wave_owner *,uint64_t generation,uint64_t paw_epoch,
 const metal_wave_prepare_input *,const metal_wave_basis_input *,const metal_wave_basis_delta *);
void metal_wave_owner_davidson_get_transfer_stats(const metal_wave_owner *,metal_wave_basis_transfer_stats *);
int metal_wave_owner_davidson_download_overlap(metal_wave_owner *,uint64_t generation,metal_complex64 *,size_t ld);
void metal_wave_owner_davidson_get_stats(const metal_wave_owner *,metal_wave_davidson_stats *);
/* End a logical EDDAV lease while retaining validated physical static buffers. */
void metal_wave_owner_davidson_forget_outputs(metal_wave_owner *);
#ifdef __cplusplus
}
#endif
