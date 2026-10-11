#pragma once
#include "metal_backend.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed Objective-C objects; consumers retain them and keep backend alive. */
int metal_backend_copy_metal_resources(const metal_backend *,void **device,void **queue,void **library);
int metal_backend_complete_external_command(const metal_backend *,void *command);
void metal_backend_record_external_conversion(const metal_backend *,double seconds);
void metal_backend_set_external_success(const metal_backend *,int success);
uint64_t metal_backend_paw_epoch(const metal_backend *);
uint64_t metal_backend_recommended_working_set_bytes(const metal_backend *);
/* Pressure-only eviction uses try-locks and never releases active resources. */
void metal_backend_evict_idle_buffers(const metal_backend *);
void metal_backend_record_cooperative_orth(const metal_backend *);
#ifdef __cplusplus
}
#endif
