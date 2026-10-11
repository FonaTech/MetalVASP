#pragma once
#import <Metal/Metal.h>
#include <cstddef>

// Metal 4 is an optional command submission path. The public surface contains
// no Metal 4 types, so this header remains usable with macOS 13 SDKs.
struct metal4_dispatch;
metal4_dispatch *metal4_dispatch_create(id<MTLDevice> device, id<MTLLibrary> library,
                                       char *error, size_t error_capacity);
void metal4_dispatch_destroy(metal4_dispatch *dispatch);
bool metal4_dispatch_has_tiled(const metal4_dispatch *dispatch);

// Uses the same ds_zgemm / ds_zgemm_tiled functions and constant-buffer layout
// as the legacy encoder. All buffers must belong to the dispatch's device.
// Returns 0 only after successful GPU completion. On failure the caller may
// execute its existing Metal 3 path. This call does not convert host data.
int metal4_dispatch_zgemm(metal4_dispatch *dispatch, bool tiled,
                         id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                         const void *parameters, size_t parameter_bytes,
                         size_t m, size_t n, size_t threads,
                         double *gpu_seconds, double *wait_seconds,
                         char *error, size_t error_capacity);
