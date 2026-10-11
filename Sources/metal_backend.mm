#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>

#include "metal_backend.h"
#include "metal_compile_policy.hpp"
#include "lobpcg_shader.h"
#include "subspace_shader.h"
#include "hamiltonian_shader.h"
#include "orthogonal_shader.h"
#include "metal_full_orth_policy.hpp"
#include "metal_eigh_policy.hpp"
#include "metal_chunk_policy.hpp"
#include "metal_gemm_policy.hpp"
#include "wave_owner_shader.h"
#include "nonlocal_shader.h"
#include <cfloat>
#include "metal_backend_private.h"
#include "metal4_dispatch.h"
#include "metal_library_loader.hpp"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include "metal_memory_allocator.hpp"
#include "metal_orth_dispatch.hpp"

struct metal_backend {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> gemm = nil;
    id<MTLComputePipelineState> gemm_tiled = nil;
    mutable bool tiled_enabled = false;
    mutable uint64_t tiled_calls = 0, tiled_fallbacks = 0;
    bool tiled_auto_selected = false;
    bool gemm_calibrated = false;
    int gemm_check_mode = 0;
    std::string language = "uncompiled";
    unsigned compile_fallbacks = 0;
    bool metal4_command_requested = false;
    mutable metal4_dispatch *metal4_commands = nullptr;
    mutable uint64_t metal4_command_calls = 0, metal4_command_success = 0, metal4_command_fallbacks = 0;
    mutable std::string metal4_command_reason = "not-requested";
    bool precompiled_requested=false, precompiled_verified=false, precompiled_used=false;
    unsigned precompiled_fallbacks=0;
    std::string precompiled_path, precompiled_reason="not-requested";
    id<MTLComputePipelineState> dft = nil;
    id<MTLComputePipelineState> reduce_stage = nil;
    id<MTLComputePipelineState> dot_stage = nil;
    id<MTLComputePipelineState> reduce_complex_stage = nil;
    id<MTLComputePipelineState> vh_apply = nil;
    id<MTLComputePipelineState> vh_apply_real = nil;
    mutable id<MTLBuffer> fft_buffer0 = nil;
    mutable id<MTLBuffer> fft_buffer1 = nil;
    mutable NSMutableDictionary<NSString *,id<MTLBuffer>> *fft_roots = nil;
    mutable std::mutex fft_mutex;
    // Reuse staging storage across the many small BLAS/Hamiltonian calls made
    // by an SCF step.  Allocating three shared buffers per call dominated the
    // GPU path for realistic VASP work sets.
    mutable id<MTLBuffer> gemm_a = nil;
    mutable id<MTLBuffer> gemm_b = nil;
    mutable id<MTLBuffer> gemm_c = nil;
    mutable id<MTLBuffer> vh_sv = nil;
    mutable id<MTLBuffer> vh_cr = nil;
    mutable id<MTLBuffer> vh_out = nil;
    mutable std::mutex staging_mutex;
    mutable id<MTLBuffer> dot_x = nil;
    mutable id<MTLBuffer> dot_y = nil;
    mutable id<MTLBuffer> dot_r0 = nil;
    mutable id<MTLBuffer> dot_r1 = nil;
    mutable std::mutex dot_mutex;
    id<MTLComputePipelineState> fft_stage = nil;
    id<MTLComputePipelineState> fft3d_reorder = nil;
    id<MTLComputePipelineState> fft3d_stage = nil;
    id<MTLComputePipelineState> mixed_reorder = nil;
    id<MTLComputePipelineState> mixed_stage = nil;
    id<MTLComputePipelineState> mixed_reorder32=nil,mixed_stage32=nil;
    mutable uint64_t fft_compact_axes=0,fft_wide_axes=0;
    id<MTLComputePipelineState> jacobi_select = nil;
    id<MTLComputePipelineState> jacobi_matrix = nil;
    id<MTLComputePipelineState> jacobi_vectors = nil;
    id<MTLComputePipelineState> lob_init = nil, lob_apply = nil, lob_update = nil,lob_diagonal=nil;
    id<MTLComputePipelineState> jacobi_local=nil,chol_pivot=nil,chol_column=nil,lower_inverse=nil,subspace_product=nil,hermitianize=nil,cyclic_params=nil,cyclic_right=nil,cyclic_left=nil,cyclic_check=nil;
    mutable bool generalized_all_gpu=false;
    mutable uint64_t last_eigh_iterations=0;
    mutable int last_eigh_solver=0;
    mutable double gpu_seconds=0.0;
    mutable uint64_t command_count=0, wait_count=0;
    mutable double wait_seconds=0.0, conversion_seconds=0.0;
    mutable id<MTLBuffer> sum_x=nil, sum_r0=nil, sum_r1=nil;
    mutable std::mutex sum_mutex;
    id<MTLComputePipelineState> ham_local=nil,ham_onsite=nil,ham_finish=nil,ham_scatter=nil;
    id<MTLComputePipelineState> resident_hamiltonian=nil,resident_project=nil,resident_orth=nil;
    mutable id<MTLBuffer> ham_sv=nil,ham_cw=nil,ham_proj=nil,ham_d=nil,ham_q=nil,ham_eig=nil,ham_ke=nil,ham_index=nil,ham_site=nil,ham_radial=nil,ham_cq=nil,ham_phase=nil,ham_onsite_buffer=nil,ham_out=nil;
    mutable uint64_t ham_epoch=1,ham_cached_epoch=0,ham_static_uploads=0,ham_static_reuses=0;
    mutable const void *ham_static_key=nullptr;
    mutable size_t ham_static_dims[8]{};
    id<MTLComputePipelineState> orth_overlap=nil,full_orth_metric=nil,full_orth_gram=nil,full_orth_update=nil;
    id<MTLComputePipelineState> nonlocal_project=nil;
    id<MTLComputePipelineState> orth_overlap_group=nil;
    mutable std::atomic<uint64_t> cooperative_orth_calls{0};
    mutable id<MTLBuffer> full_orth_buffers[12]{};
    mutable std::mutex full_orth_mutex;
    mutable metal_chunk_stats chunks{};
    mutable id<MTLBuffer> orth_buffers[5]{};
    mutable std::mutex orth_mutex;
    mutable id<MTLBuffer> small_eigh[11]{};
    mutable std::mutex small_eigh_mutex;
    bool use_gpu = false;
    bool metal4_family = false;
    bool ml_api_available = false;
    bool device_tuned = false;
    NSUInteger thread_execution_width = 0;
    NSUInteger dispatch_width = 256;
    std::string variant = "cpu-unavailable";
    mutable bool last_gpu_success = false;
    mutable std::string error;
    ~metal_backend() { metal4_dispatch_destroy(metal4_commands); }
};

/* Device-resident state for one serial Davidson/SCF stream.  The shape is
   immutable; host pointers are meaningful only at explicit upload/download
   calls.  This prevents a stale host address from masquerading as residency. */
struct metal_resident {
    const metal_backend *backend = nullptr;
    size_t nx=0, ny=0, nz=0, total=0, bands=0, npro=0;
    id<MTLBuffer> wave0=nil, wave1=nil, potential=nil, projectors=nil, projection=nil, result_status=nil;
    id<MTLCommandBuffer> command=nil;
    id<MTLCommandBuffer> last_command=nil;
    id<MTLComputeCommandEncoder> encoder=nil;
    id<MTLBuffer> current=nil, next=nil;
    uint64_t generation=0, steps=0, uploads=0, submissions=0, waits=0;
    bool open=false, encoded=false, wave_valid=false, potential_valid=false, projectors_valid=false;
    bool projection_valid=false, last_accounted=true;
    mutable std::mutex mutex;
};

extern "C" void metal_backend_evict_idle_buffers(const metal_backend *b) {
    if (!b) return;
    // Independent try-locks avoid lock-order cycles when allocation happens
    // inside an operator. Command completion is inside each operator lock.
    {
        auto lock=metal_memory::idle_lock(b->staging_mutex);
        if (lock.owns_lock()) {
            b->gemm_a=nil; b->gemm_b=nil; b->gemm_c=nil;
            b->vh_sv=nil; b->vh_cr=nil; b->vh_out=nil;
        }
    }
    {
        auto lock=metal_memory::idle_lock(b->fft_mutex);
        if (lock.owns_lock()) {
            b->fft_buffer0=nil; b->fft_buffer1=nil; [b->fft_roots removeAllObjects];
            b->ham_sv=nil; b->ham_cw=nil; b->ham_proj=nil; b->ham_d=nil; b->ham_q=nil;
            b->ham_eig=nil; b->ham_ke=nil; b->ham_index=nil; b->ham_site=nil;
            b->ham_radial=nil; b->ham_cq=nil; b->ham_phase=nil;
            b->ham_onsite_buffer=nil; b->ham_out=nil;
            b->ham_static_key=nullptr; b->ham_cached_epoch=0;
        }
    }
    {
        auto lock=metal_memory::idle_lock(b->dot_mutex);
        if (lock.owns_lock()) { b->dot_x=nil; b->dot_y=nil; b->dot_r0=nil; b->dot_r1=nil; }
    }
    {
        auto lock=metal_memory::idle_lock(b->sum_mutex);
        if (lock.owns_lock()) { b->sum_x=nil; b->sum_r0=nil; b->sum_r1=nil; }
    }
    {
        auto lock=metal_memory::idle_lock(b->orth_mutex);
        if (lock.owns_lock()) for (auto &buffer : b->orth_buffers) buffer=nil;
    }
    {
        auto lock=metal_memory::idle_lock(b->full_orth_mutex);
        if (lock.owns_lock()) for (auto &buffer : b->full_orth_buffers) buffer=nil;
    }
    {
        auto lock=metal_memory::idle_lock(b->small_eigh_mutex);
        if (lock.owns_lock()) for (auto &buffer : b->small_eigh) buffer=nil;
    }
}

namespace {
void evict_backend(const void *b) {
    metal_backend_evict_idle_buffers(static_cast<const metal_backend *>(b));
}
id<MTLBuffer> backend_buffer(const metal_backend *b, size_t bytes, MTLResourceOptions options) NS_RETURNS_RETAINED {
    return metal_memory::allocate(b->device, bytes, options, evict_backend, b);
}
using cd = std::complex<double>;

using SteadyClock = std::chrono::steady_clock;
struct ConversionTimer {
    const metal_backend *backend;
    SteadyClock::time_point start=SteadyClock::now();
    ~ConversionTimer() { backend->conversion_seconds+=std::chrono::duration<double>(SteadyClock::now()-start).count(); }
};
int complete_command(const metal_backend *backend, id<MTLCommandBuffer> command) {
    if (!command) return METAL_BACKEND_RUNTIME_FAILED;
    ++backend->command_count;
    [command commit];
    auto start=SteadyClock::now(); ++backend->wait_count;
    [command waitUntilCompleted];
    backend->wait_seconds+=std::chrono::duration<double>(SteadyClock::now()-start).count();
    if(command.GPUEndTime>command.GPUStartTime) backend->gpu_seconds+=command.GPUEndTime-command.GPUStartTime;
    return command.status==MTLCommandBufferStatusCompleted ? METAL_BACKEND_OK : METAL_BACKEND_RUNTIME_FAILED;
}
struct DS { float hi; float lo; };
struct DSC { DS real; DS imag; };
struct HostGemmParams {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    vector_float4 alpha;
    vector_float4 beta;
};
struct FFT3DReorderParamsHost { uint32_t nx, ny, nz, axis, n, stride, lines; };
struct FFT3DStageParamsHost { uint32_t nx, ny, nz, axis, n, stride, lines, stage; int32_t direction; };
struct MixedReorderParams32 {uint32_t n,stride,lines,base_dim,axis,count,factors[32],batches;};
struct MixedStageParams32 {uint32_t n,stride,lines,base_dim,axis,prefix,radix,batches;};
struct MixedReorderParams { uint32_t n,axis,count,factors[32]; uint64_t stride,lines,base_dim,batches; };
struct MixedStageParams { uint32_t n,axis,prefix,radix; uint64_t stride,lines,base_dim,batches; };
struct JacobiParamsHost {
    uint32_t n, p, r;
    uint32_t reserved;
    vector_float2 c;
    vector_float4 s;
    vector_float2 pp;
    vector_float2 rr;
};

constexpr char kBackendShader[] = R"METAL(
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;
#ifndef METAL_VASP_DEVICE_TUNED
#define METAL_VASP_DEVICE_TUNED 0
#endif
#ifndef METAL_VASP_METAL4
#define METAL_VASP_METAL4 0
#endif
#if METAL_VASP_DEVICE_TUNED
// The tuned profile is compiled only after a runtime capability probe. The
// unroll is deliberately conservative; numerical semantics remain DS.
#define METAL_VASP_GEMM_UNROLL 4
#else
#define METAL_VASP_GEMM_UNROLL 1
#endif
struct DS { float hi; float lo; };
struct DSC { DS real; DS imag; };
inline DS ds_add(DS a, DS b) {
    float sum = a.hi + b.hi;
    float virtual_b = sum - a.hi;
    float error = ((a.hi - (sum - virtual_b)) + (b.hi - virtual_b)) + a.lo + b.lo;
    float hi = sum + error;
    return {hi, error - (hi - sum)};
}
inline DS ds_mul(DS a, DS b) {
    float product = a.hi * b.hi;
    float error = fma(a.hi, b.hi, -product) + a.hi * b.lo + a.lo * b.hi + a.lo * b.lo;
    float hi = product + error;
    return {hi, error - (hi - product)};
}
inline DSC dsc_add(DSC a, DSC b) {
    return {ds_add(a.real, b.real), ds_add(a.imag, b.imag)};
}
inline DSC dsc_mul(DSC a, DSC b) {
    DS rr = ds_mul(a.real, b.real);
    DS ii = ds_mul(a.imag, b.imag);
    DS ri = ds_mul(a.real, b.imag);
    DS ir = ds_mul(a.imag, b.real);
    return {ds_add(rr, (DS){-ii.hi, -ii.lo}), ds_add(ri, ir)};
}
inline DSC dsc_scale(DSC a, float c) {
    return {{a.real.hi * c, a.real.lo * c}, {a.imag.hi * c, a.imag.lo * c}};
}
inline DSC dsc_scale_ds(DSC a, float2 c) {
    DS sc = {c.x, c.y};
    return {ds_mul(a.real, sc), ds_mul(a.imag, sc)};
}
inline DSC dsc_conj(DSC a) {
    return {a.real, { -a.imag.hi, -a.imag.lo }};
}
struct GemmParams {
    uint m;
    uint n;
    uint k;
    float4 alpha;
    float4 beta;
};
kernel void ds_zgemm(device const float4* a [[buffer(0)]],
                     device const float4* b [[buffer(1)]],
                     device float4* c [[buffer(2)]],
                     constant GemmParams& p [[buffer(3)]],
                     uint2 grid [[thread_position_in_grid]]) {
    const uint row=grid.y,col=grid.x;
    if(row>=p.m || col>=p.n)return;
    const ulong index=ulong(row)*p.n+col;
    DSC sum = {{0.0f, 0.0f}, {0.0f, 0.0f}};
#if METAL_VASP_DEVICE_TUNED
#pragma clang loop unroll_count(4)
#endif
    for (uint inner = 0; inner < p.k; ++inner) {
        float4 av = a[ulong(row) * p.k + inner];
        float4 bv = b[ulong(inner) * p.n + col];
        DSC left = {{av.x, av.y}, {av.z, av.w}};
        DSC right = {{bv.x, bv.y}, {bv.z, bv.w}};
        sum = dsc_add(sum, dsc_mul(left, right));
    }
    float4 c4 = c[index];
    DSC old = {{c4.x, c4.y}, {c4.z, c4.w}};
    DSC alpha = {{p.alpha.x, p.alpha.y}, {p.alpha.z, p.alpha.w}};
    DSC beta = {{p.beta.x, p.beta.y}, {p.beta.z, p.beta.w}};
    DSC value = dsc_add(dsc_mul(alpha, sum), dsc_mul(beta, old));
    c[index] = float4(value.real.hi, value.real.lo, value.imag.hi, value.imag.lo);
}
// A full 16x16 group cooperatively reuses A/B tiles. Every lane participates
// in both barriers, including edge lanes outside the output matrix.
kernel void ds_zgemm_tiled(device const float4* a [[buffer(0)]],
                           device const float4* b [[buffer(1)]],
                           device float4* c [[buffer(2)]],
                           constant GemmParams& p [[buffer(3)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint2 lane [[thread_position_in_threadgroup]]) {
    threadgroup float4 tile_a[16][16];
    threadgroup float4 tile_b[16][16];
    const uint row=group.y*16+lane.y, col=group.x*16+lane.x;
    DSC sum={{0.0f,0.0f},{0.0f,0.0f}};
    for (uint base=0;base<p.k;base+=16) {
        tile_a[lane.y][lane.x]=(row<p.m && base+lane.x<p.k) ? a[ulong(row)*p.k+base+lane.x] : float4(0.0f);
        tile_b[lane.y][lane.x]=(base+lane.y<p.k && col<p.n) ? b[ulong(base+lane.y)*p.n+col] : float4(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner=0;inner<min(16u,p.k-base);++inner) {
            float4 av=tile_a[lane.y][inner], bv=tile_b[inner][lane.x];
            sum=dsc_add(sum,dsc_mul(DSC{{av.x,av.y},{av.z,av.w}},DSC{{bv.x,bv.y},{bv.z,bv.w}}));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row<p.m && col<p.n) {
        ulong index=ulong(row)*p.n+col;
        float4 old=c[index];
        DSC value=dsc_add(dsc_mul(DSC{{p.alpha.x,p.alpha.y},{p.alpha.z,p.alpha.w}},sum),
                          dsc_mul(DSC{{p.beta.x,p.beta.y},{p.beta.z,p.beta.w}},DSC{{old.x,old.y},{old.z,old.w}}));
        c[index]=float4(value.real.hi,value.real.lo,value.imag.hi,value.imag.lo);
    }
}
kernel void ds_reduce_stage(device const float2* x [[buffer(0)]],
                            device float2* out [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            constant uint& stride [[buffer(3)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint group [[threadgroup_position_in_grid]]) {
    threadgroup DS shared[256];
    DS sum = {0.0f, 0.0f};
    const uint begin = group * stride;
    const uint end = min(begin + stride, n);
    for (uint i = begin + tid; i < end; i += 256u) sum = ds_add(sum, (DS){x[i].x, x[i].y});
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128u; width > 0u; width >>= 1u) {
        if (tid < width) shared[tid] = ds_add(shared[tid], shared[tid + width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[group] = float2(shared[0].hi, shared[0].lo);
}
kernel void ds_zdot_stage(device const float4* x [[buffer(0)]],
                          device const float4* y [[buffer(1)]],
                          device float4* out [[buffer(2)]],
                          constant uint& n [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]]) {
    threadgroup DSC partial[256];
    uint index=group*256u+tid;
    DSC sum={{0.0f,0.0f},{0.0f,0.0f}};
    if(index<n) {
        float4 a=x[index],b=y[index];
        sum=dsc_mul((DSC){{a.x,a.y},{-a.z,-a.w}},(DSC){{b.x,b.y},{b.z,b.w}});
    }
    partial[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint width=128u;width>0u;width>>=1u) {
        if(tid<width) partial[tid]=dsc_add(partial[tid],partial[tid+width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0u) out[group]=float4(partial[0].real.hi,partial[0].real.lo,partial[0].imag.hi,partial[0].imag.lo);
}
kernel void ds_reduce_complex_stage(device const float4* x [[buffer(0)]],
                                    device float4* out [[buffer(1)]],
                                    constant uint& n [[buffer(2)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint group [[threadgroup_position_in_grid]]) {
    threadgroup DSC partial[256];
    uint index=group*256u+tid;
    DSC sum={{0.0f,0.0f},{0.0f,0.0f}};
    if(index<n) {float4 a=x[index];sum=(DSC){{a.x,a.y},{a.z,a.w}};}
    partial[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint width=128u;width>0u;width>>=1u) {
        if(tid<width) partial[tid]=dsc_add(partial[tid],partial[tid+width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0u) out[group]=float4(partial[0].real.hi,partial[0].real.lo,partial[0].imag.hi,partial[0].imag.lo);
}

struct VHApplyParams { uint np; uint sv_stride; uint wf_stride; uint nrspinors; uint nbands; uint batch_stride; float2 rinplw; };
kernel void ds_vh_apply(device const float4* sv [[buffer(0)]],
                        device const float4* cr [[buffer(1)]],
                        device float4* cvr [[buffer(2)]],
                        constant VHApplyParams& p [[buffer(3)]],
                        uint index [[thread_position_in_grid]]) {
    const uint total = p.np * p.nrspinors * p.nbands;
    if (index >= total) return;
    const uint band = index / (p.np * p.nrspinors);
    const uint local = index % (p.np * p.nrspinors);
    const uint m = local % p.np;
    const uint spin = local / p.np;
    DSC sum = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    const uint cr_index = band * p.batch_stride + m + spin * p.wf_stride;
    const uint channel_base = 2u * spin;
    for (uint input_spin = 0u; input_spin < p.nrspinors; ++input_spin) {
        const float4 pot4 = sv[m + (channel_base + input_spin) * p.sv_stride];
        const float4 wave4 = cr[band * p.batch_stride + m + input_spin * p.wf_stride];
        const DSC pot = {{pot4.x, pot4.y}, {pot4.z, pot4.w}};
        const DSC wave = {{wave4.x, wave4.y}, {wave4.z, wave4.w}};
        sum = dsc_add(sum, dsc_mul(pot, wave));
    }
    const DS scale = {p.rinplw.x, p.rinplw.y};
    sum.real = ds_mul(sum.real, scale);
    sum.imag = ds_mul(sum.imag, scale);
    cvr[cr_index] = float4(sum.real.hi, sum.real.lo, sum.imag.hi, sum.imag.lo);
}
kernel void ds_vh_apply_real(device const float2* sv [[buffer(0)]],
                             device const float4* cr [[buffer(1)]],
                             device float4* cvr [[buffer(2)]],
                             constant VHApplyParams& p [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
    const uint total = p.np * p.nrspinors * p.nbands;
    if (index >= total) return;
    const uint band = index / (p.np * p.nrspinors);
    const uint local = index % (p.np * p.nrspinors);
    const uint m = local % p.np;
    const uint spin = local / p.np;
    DSC sum = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    const uint cr_index = band * p.batch_stride + m + spin * p.wf_stride;
    const uint channel_base = 2u * spin;
    for (uint input_spin = 0u; input_spin < p.nrspinors; ++input_spin) {
        const float2 pot2 = sv[m + (channel_base + input_spin) * p.sv_stride];
        const float4 wave4 = cr[band * p.batch_stride + m + input_spin * p.wf_stride];
        const DSC pot = {{pot2.x, pot2.y}, {0.0f, 0.0f}};
        const DSC wave = {{wave4.x, wave4.y}, {wave4.z, wave4.w}};
        sum = dsc_add(sum, dsc_mul(pot, wave));
    }
    const DS scale = {p.rinplw.x, p.rinplw.y};
    sum.real = ds_mul(sum.real, scale); sum.imag = ds_mul(sum.imag, scale);
    cvr[cr_index] = float4(sum.real.hi, sum.real.lo, sum.imag.hi, sum.imag.lo);
}
kernel void dft_f32(device const float2* x [[buffer(0)]],
                    device float2* y [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    constant int& direction [[buffer(3)]],
                    uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    float2 sum = float2(0.0f);
    float sign = direction < 0 ? -1.0f : 1.0f;
    for (uint k = 0; k < n; ++k) {
        float angle = sign * 6.28318530717958647692f * float(index * k) / float(n);
        float2 w = float2(cos(angle), sin(angle));
        float2 v = x[k];
        sum += float2(v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x);
    }
    if (direction > 0) sum /= float(n);
    y[index] = sum;
}

kernel void ds_zdft(device const float4* x [[buffer(0)]],
                    device float4* y [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    constant int& direction [[buffer(3)]],
                    uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    DS sr = {0.0f, 0.0f};
    DS si = {0.0f, 0.0f};
    const float sign = direction < 0 ? -1.0f : 1.0f;
    for (uint k = 0; k < n; ++k) {
        const float angle = sign * 6.28318530717958647692f * float(index * k) / float(n);
        const float wr = cos(angle), wi = sin(angle);
        const float4 v = x[k];
        DSC value = {{v.x, v.y}, {v.z, v.w}};
        DSC twiddle = {{wr, 0.0f}, {wi, 0.0f}};
        DSC product = dsc_mul(value, twiddle);
        sr = ds_add(sr, product.real);
        si = ds_add(si, product.imag);
    }
    if (direction > 0) {
        const float inv_n = 1.0f / float(n);
        sr = ds_mul(sr, (DS){inv_n, 0.0f});
        si = ds_mul(si, (DS){inv_n, 0.0f});
    }
    y[index] = float4(sr.hi, sr.lo, si.hi, si.lo);
}

struct FFTStageParams { uint n; uint stage; int direction; };
kernel void ds_fft_stage(device const float4* x [[buffer(0)]],
                         device float4* y [[buffer(1)]],
                         constant FFTStageParams& p [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    const uint m = 1u << (p.stage + 1u);
    const uint half_count = m >> 1u;
    const uint group = index / half_count;
    const uint j = index % half_count;
    const uint base = group * m;
    const uint i0 = base + j;
    const uint i1 = i0 + half_count;
    const float sign = p.direction < 0 ? -1.0f : 1.0f;
    const float angle = sign * 6.28318530717958647692f * float(j) / float(m);
    const float wr = cos(angle), wi = sin(angle);
    const float4 u4 = x[i0], v4 = x[i1];
    DSC u = {{u4.x, u4.y}, {u4.z, u4.w}};
    DSC v = {{v4.x, v4.y}, {v4.z, v4.w}};
    DSC twiddle = {{wr, 0.0f}, {wi, 0.0f}};
    v = dsc_mul(v, twiddle);
    DSC plus = dsc_add(u, v);
    DSC minus = dsc_add(u, (DSC){{-v.real.hi, -v.real.lo}, {-v.imag.hi, -v.imag.lo}});
    y[i0] = float4(plus.real.hi, plus.real.lo, plus.imag.hi, plus.imag.lo);
    y[i1] = float4(minus.real.hi, minus.real.lo, minus.imag.hi, minus.imag.lo);
}

struct FFT3DReorderParams { uint nx; uint ny; uint nz; uint axis; uint n; uint stride; uint lines; };
struct FFT3DStageParams { uint nx; uint ny; uint nz; uint axis; uint n; uint stride; uint lines; uint stage; int direction; };
inline uint fft3d_base(constant FFT3DReorderParams& p, uint line) {
    if (p.axis == 0u) { uint y = line % p.ny; uint z = line / p.ny; return y * p.nx + z * p.nx * p.ny; }
    if (p.axis == 1u) { uint x = line % p.nx; uint z = line / p.nx; return x + z * p.nx * p.ny; }
    uint x = line % p.nx; uint y = line / p.nx; return x + y * p.nx;
}
inline uint fft3d_base_stage(constant FFT3DStageParams& p, uint line) {
    if (p.axis == 0u) { uint y = line % p.ny; uint z = line / p.ny; return y * p.nx + z * p.nx * p.ny; }
    if (p.axis == 1u) { uint x = line % p.nx; uint z = line / p.nx; return x + z * p.nx * p.ny; }
    uint x = line % p.nx; uint y = line / p.nx; return x + y * p.nx;
}
inline uint reverse_bits_3d(uint value, uint n) {
    uint out = 0u;
    for (uint bit = 1u; bit < n; bit <<= 1u) { out = (out << 1u) | ((value & bit) ? 1u : 0u); }
    return out;
}
kernel void ds_fft3d_reorder(device const float4* x [[buffer(0)]],
                             device float4* y [[buffer(1)]],
                             constant FFT3DReorderParams& p [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    if (index >= p.n * p.lines) return;
    uint line = index / p.n; uint q = index % p.n;
    y[fft3d_base(p, line) + reverse_bits_3d(q, p.n) * p.stride] = x[fft3d_base(p, line) + q * p.stride];
}
kernel void ds_fft3d_stage(device const float4* x [[buffer(0)]],
                           device float4* y [[buffer(1)]],
                           constant FFT3DStageParams& p [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    uint hcount = p.n >> 1u;
    if (index >= hcount * p.lines) return;
    uint line = index / hcount; uint j = index % hcount;
    uint m = 1u << (p.stage + 1u); uint group = j / (m >> 1u); uint jj = j % (m >> 1u);
    uint base = fft3d_base_stage(p, line) + group * m * p.stride;
    uint i0 = base + jj * p.stride; uint i1 = i0 + (m >> 1u) * p.stride;
    float sign = p.direction < 0 ? -1.0f : 1.0f;
    float angle = sign * 6.28318530717958647692f * float(jj) / float(m);
    float wr = cos(angle), wi = sin(angle);
    float4 u4 = x[i0], v4 = x[i1];
    DSC u = {{u4.x, u4.y}, {u4.z, u4.w}}; DSC v = {{v4.x, v4.y}, {v4.z, v4.w}};
    v = dsc_mul(v, (DSC){{wr, 0.0f}, {wi, 0.0f}});
    DSC plus = dsc_add(u, v);
    DSC minus = dsc_add(u, (DSC){{-v.real.hi, -v.real.lo}, {-v.imag.hi, -v.imag.lo}});
    y[i0] = float4(plus.real.hi, plus.real.lo, plus.imag.hi, plus.imag.lo);
    y[i1] = float4(minus.real.hi, minus.real.lo, minus.imag.hi, minus.imag.lo);
}

struct MixedReorderParams32 { uint n, stride, lines, base_dim, axis; uint count; uint factors[32]; uint batches; };
struct MixedStageParams32 { uint n, stride, lines, base_dim, axis, prefix, radix, batches; };
inline uint mixed_base32(uint line, uint base_dim, uint axis, uint n) {
    if (axis == 0u) return line * n;
    return (line % base_dim) + (line / base_dim) * base_dim * n;
}
kernel void ds_mixed_reorder32(device const float4* x [[buffer(0)]],
                             device float4* y [[buffer(1)]],
                             constant MixedReorderParams32& p [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    uint total=p.n*p.lines;
    if (index >= total*p.batches) return;
    uint batch=index/total; index%=total;
    uint line = index / p.n, q = index % p.n;
    uint source = 0u, remain = q, weight = p.n;
    for (uint f = 0u; f < p.count; ++f) {
        weight /= p.factors[f];
        source += (remain % p.factors[f]) * weight;
        remain /= p.factors[f];
    }
    uint base = batch*total + mixed_base32(line, p.base_dim, p.axis, p.n);
    y[base + q * p.stride] = x[base + source * p.stride];
}
kernel void ds_mixed_stage32(device const float4* x [[buffer(0)]],
                           device float4* y [[buffer(1)]],
                           constant MixedStageParams32& p [[buffer(2)]],
                           device const float4* roots [[buffer(3)]],
                           uint index [[thread_position_in_grid]]) {
    uint total=p.n*p.lines;
    if (index >= total*p.batches) return;
    uint batch=index/total; index%=total;
    uint line = index / p.n, q = index % p.n;
    uint m = p.prefix * p.radix, group = q / m;
    uint j = q % p.prefix, frequency = q % m;
    uint base = batch*total + mixed_base32(line, p.base_dim, p.axis, p.n);
    DSC sum = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (uint k = 0; k < p.radix; ++k) {
        float4 xx = x[base + (group*m + j + k*p.prefix)*p.stride];
        float4 tw = roots[(k * frequency * (p.n/m)) % p.n];
        sum = dsc_add(sum, dsc_mul((DSC){{xx.x,xx.y},{xx.z,xx.w}},
                                  (DSC){{tw.x,tw.y},{tw.z,tw.w}}));
    }
    y[base + q*p.stride] = float4(sum.real.hi,sum.real.lo,sum.imag.hi,sum.imag.lo);
}
struct MixedReorderParams { uint n,axis,count,factors[32]; ulong stride,lines,base_dim,batches; };
struct MixedStageParams { uint n,axis,prefix,radix; ulong stride,lines,base_dim,batches; };
inline ulong mixed_base(ulong line, ulong base_dim, uint axis, uint n) {
    if (axis == 0u) return line * n;
    return (line % base_dim) + (line / base_dim) * base_dim * n;
}
kernel void ds_mixed_reorder(device const float4* x [[buffer(0)]],
                             device float4* y [[buffer(1)]],
                             constant MixedReorderParams& p [[buffer(2)]],
                             uint2 grid [[thread_position_in_grid]]) {
    ulong index=ulong(grid.x)+ulong(grid.y)*65536ul;
    const ulong total=ulong(p.n)*p.lines;
    if (index >= total*p.batches) return;
    ulong batch=index/total; index%=total;
    ulong line=index/p.n;uint q=uint(index%p.n);
    uint source = 0u, remain = q, weight = p.n;
    for (uint f = 0u; f < p.count; ++f) {
        weight /= p.factors[f];
        source += (remain % p.factors[f]) * weight;
        remain /= p.factors[f];
    }
    ulong base = batch*total + mixed_base(line, p.base_dim, p.axis, p.n);
    y[base + q * p.stride] = x[base + source * p.stride];
}
kernel void ds_mixed_stage(device const float4* x [[buffer(0)]],
                           device float4* y [[buffer(1)]],
                           constant MixedStageParams& p [[buffer(2)]],
                           device const float4* roots [[buffer(3)]],
                           uint2 grid [[thread_position_in_grid]]) {
    ulong index=ulong(grid.x)+ulong(grid.y)*65536ul;
    const ulong total=ulong(p.n)*p.lines;
    if (index >= total*p.batches) return;
    ulong batch=index/total; index%=total;
    ulong line=index/p.n;uint q=uint(index%p.n);
    uint m = p.prefix * p.radix, group = q / m;
    uint j = q % p.prefix, frequency = q % m;
    ulong base = batch*total + mixed_base(line, p.base_dim, p.axis, p.n);
    DSC sum = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (uint k = 0; k < p.radix; ++k) {
        float4 xx = x[base + (group*m + j + k*p.prefix)*p.stride];
        float4 tw = roots[(ulong(k) * frequency * (p.n/m)) % p.n];
        sum = dsc_add(sum, dsc_mul((DSC){{xx.x,xx.y},{xx.z,xx.w}},
                                  (DSC){{tw.x,tw.y},{tw.z,tw.w}}));
    }
    y[base + q*p.stride] = float4(sum.real.hi,sum.real.lo,sum.imag.hi,sum.imag.lo);
}
struct JacobiParams {
    uint n;
    uint p;
    uint r;
    uint reserved;
    float2 c;
    float4 s;
    float2 pp;
    float2 rr;
};
inline DSC load_dsc(device const float4* x, ulong index) {
    const float4 v = x[index];
    return {{v.x, v.y}, {v.z, v.w}};
}
inline float4 store_dsc(DSC x) {
    return float4(x.real.hi, x.real.lo, x.imag.hi, x.imag.lo);
}
inline DS ds_neg(DS x) { return {-x.hi,-x.lo}; }
inline DS ds_div(DS a, DS b) {
    DS q={a.hi/b.hi,0.0f};
    DS rem=ds_add(a,ds_neg(ds_mul(b,q)));
    return ds_add(q,(DS){rem.hi/b.hi,0.0f});
}
inline DS ds_sqrt(DS x) {
    if (x.hi<=0.0f) return {0.0f,0.0f};
    DS q={sqrt(x.hi),0.0f};
    DS rem=ds_add(x,ds_neg(ds_mul(q,q)));
    return ds_add(q,ds_div(rem,ds_add(q,q)));
}
kernel void ds_jacobi_select(device const float4* h [[buffer(0)]],
                             device JacobiParams* out [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    if(index!=0u) return;
    uint pp=0u,rr=0u; float largest=0.0f,scale=1.0f;
    for(uint col=0;col<n;++col) {
        scale=max(scale,abs(h[col+ulong(col)*n].x));
        for(uint row=0;row<col;++row) {
            float4 x=h[row+ulong(col)*n]; float mag=length(float2(x.x+x.y,x.z+x.w));
            if(mag>largest) {largest=mag;pp=row;rr=col;}
        }
    }
    out->n=n; out->p=pp; out->r=rr; out->reserved=largest>1.0e-12f*scale ? 1u:0u;
    if(out->reserved==0u) return;
    DSC hpq=load_dsc(h,pp+ulong(rr)*n);
    DS mag=ds_sqrt(ds_add(ds_mul(hpq.real,hpq.real),ds_mul(hpq.imag,hpq.imag)));
    float4 app4=h[pp+ulong(pp)*n],arr4=h[rr+ulong(rr)*n];
    DS app={app4.x,app4.y},arr={arr4.x,arr4.y},one={1.0f,0.0f};
    DS tau=ds_div(ds_add(arr,ds_neg(app)),ds_add(mag,mag));
    DS abs_tau=tau.hi<0.0f ? ds_neg(tau):tau;
    DS t=ds_div((DS){tau.hi>=0.0f ? 1.0f:-1.0f,0.0f},ds_add(abs_tau,ds_sqrt(ds_add(one,ds_mul(tau,tau)))));
    DS c=ds_div(one,ds_sqrt(ds_add(one,ds_mul(t,t))));
    DS factor=ds_div(ds_mul(c,t),mag);
    DSC ss={ds_mul(hpq.real,factor),ds_mul(hpq.imag,factor)};
    DS diagp=ds_add(app,ds_neg(ds_mul(t,mag))),diagr=ds_add(arr,ds_mul(t,mag));
    out->c=float2(c.hi,c.lo); out->s=store_dsc(ss);
    out->pp=float2(diagp.hi,diagp.lo); out->rr=float2(diagr.hi,diagr.lo);
}
kernel void ds_jacobi_matrix(device const float4* old_h [[buffer(0)]],
                             device float4* new_h [[buffer(1)]],
                             constant JacobiParams& p [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    if (index >= ulong(p.n)*p.n) return;
    if (p.reserved == 0u) { new_h[index] = old_h[index]; return; }
    const uint i = index % p.n;
    const uint j = index / p.n;
    const uint pp = p.p, rr = p.r;
    DSC out = load_dsc(old_h, index);
    const DSC s = {{p.s.x, p.s.y}, {p.s.z, p.s.w}};
    if (i == pp && j == pp) {
        out = {{p.pp.x, p.pp.y}, {0.0f, 0.0f}};
    } else if (i == rr && j == rr) {
        out = {{p.rr.x, p.rr.y}, {0.0f, 0.0f}};
    } else if ((i == pp && j == rr) || (i == rr && j == pp)) {
        out = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    } else if (j == pp && i != pp && i != rr) {
        const DSC hp = load_dsc(old_h, i + ulong(pp)*p.n);
        const DSC hr = load_dsc(old_h, i + ulong(rr)*p.n);
        out = dsc_add(dsc_scale_ds(hp, p.c),
                      dsc_scale(dsc_mul(dsc_conj(s), hr), -1.0f));
    } else if (j == rr && i != pp && i != rr) {
        const DSC hp = load_dsc(old_h, i + ulong(pp)*p.n);
        const DSC hr = load_dsc(old_h, i + ulong(rr)*p.n);
        out = dsc_add(dsc_scale_ds(hr, p.c), dsc_mul(s, hp));
    } else if (i == pp && j != pp && j != rr) {
        const DSC hp = load_dsc(old_h, j + ulong(pp)*p.n);
        const DSC hr = load_dsc(old_h, j + ulong(rr)*p.n);
        out = dsc_conj(dsc_add(dsc_scale_ds(hp, p.c),
                               dsc_scale(dsc_mul(dsc_conj(s), hr), -1.0f)));
    } else if (i == rr && j != pp && j != rr) {
        const DSC hp = load_dsc(old_h, j + ulong(pp)*p.n);
        const DSC hr = load_dsc(old_h, j + ulong(rr)*p.n);
        out = dsc_conj(dsc_add(dsc_scale_ds(hr, p.c), dsc_mul(s, hp)));
    }
    new_h[index] = store_dsc(out);
}
kernel void ds_jacobi_vectors(device const float4* old_q [[buffer(0)]],
                              device float4* new_q [[buffer(1)]],
                              constant JacobiParams& p [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index >= ulong(p.n)*p.n) return;
    if (p.reserved == 0u) { new_q[index] = old_q[index]; return; }
    const uint i = index % p.n;
    const uint j = index / p.n;
    DSC out = load_dsc(old_q, index);
    const DSC s = {{p.s.x, p.s.y}, {p.s.z, p.s.w}};
    if (j == p.p) {
        const DSC qp = load_dsc(old_q, i + ulong(p.p)*p.n);
        const DSC qr = load_dsc(old_q, i + ulong(p.r)*p.n);
        out = dsc_add(dsc_scale_ds(qp, p.c), dsc_scale(dsc_mul(dsc_conj(s), qr), -1.0f));
    } else if (j == p.r) {
        const DSC qp = load_dsc(old_q, i + ulong(p.p)*p.n);
        const DSC qr = load_dsc(old_q, i + ulong(p.r)*p.n);
        out = dsc_add(dsc_scale_ds(qr, p.c), dsc_mul(s, qp));
    }
    new_q[index] = store_dsc(out);
}
)METAL";

/* Kernels used only by the lifetime-managed resident context.  They consume
   the same DS/DSC helpers as the regular backend and never expose an
   intermediate buffer to the host. */
constexpr char kResidentShader[] = R"METAL(
struct ResidentParams { uint total, bands, npro; };
kernel void ds_resident_hamiltonian(device const float4 *wave [[buffer(0)]],
                                    device const float4 *potential [[buffer(1)]],
                                    device float4 *out [[buffer(2)]],
                                    constant ResidentParams &p [[buffer(3)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= p.total*p.bands) return;
    DSC w=load_dsc(wave,i), v=load_dsc(potential,i%p.total);
    out[i]=store_dsc(dsc_mul(w,v));
}
kernel void ds_resident_project(device const float4 *wave [[buffer(0)]],
                                device const float4 *projectors [[buffer(1)]],
                                device float4 *out [[buffer(2)]],
                                constant ResidentParams &p [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p.bands*p.npro) return;
    uint band=i/p.npro, proj=i%p.npro; DSC sum={{0,0},{0,0}};
    for (uint g=0; g<p.total; ++g) {
        DSC a=dsc_conj(load_dsc(projectors,proj*p.total+g));
        sum=dsc_add(sum,dsc_mul(a,load_dsc(wave,band*p.total+g)));
    }
    out[i]=store_dsc(sum);
}
/* Correctness-first modified Gram-Schmidt.  It is intentionally one GPU
   thread: the resident ABI can therefore be validated without a host fence,
   while the production VASP path continues to use the tiled full-orth kernel
   until a parallel reduction variant is proven equivalent. */
kernel void ds_resident_orthonormalize(device float4 *wave [[buffer(0)]],
                                       constant ResidentParams &p [[buffer(1)]],
                                       device uint *result_status [[buffer(2)]],
                                       uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    for (uint band=0; band<p.bands; ++band) {
        // A second MGS pass reduces cancellation for nearly parallel columns.
        for (uint pass=0; pass<2u; ++pass) for (uint prev=0; prev<band; ++prev) {
            DSC dot={{0,0},{0,0}};
            for (uint g=0; g<p.total; ++g)
                dot=dsc_add(dot,dsc_mul(dsc_conj(load_dsc(wave,prev*p.total+g)),
                                        load_dsc(wave,band*p.total+g)));
            for (uint g=0; g<p.total; ++g) {
                DSC z=load_dsc(wave,band*p.total+g);
                DSC q=load_dsc(wave,prev*p.total+g);
                z=dsc_add(z,dsc_scale(dsc_mul(q,dot),-1.0f));
                wave[band*p.total+g]=store_dsc(z);
            }
        }
        DS norm={0,0};
        for (uint g=0; g<p.total; ++g) {
            DSC z=load_dsc(wave,band*p.total+g);
            norm=ds_add(norm,dsc_mul(dsc_conj(z),z).real);
        }
        norm=ds_sqrt(norm);
        if (!isfinite(norm.hi) || norm.hi <= 0.0f) { result_status[0]=1u; return; }
        for (uint g=0; g<p.total; ++g) {
            DSC z=load_dsc(wave,band*p.total+g);
            z.real=ds_div(z.real,norm); z.imag=ds_div(z.imag,norm);
            wave[band*p.total+g]=store_dsc(z);
        }
    }
}
)METAL";

double split_to_double(float hi, float lo) {
    return static_cast<double>(hi) + static_cast<double>(lo);
}

DS split_double(double value) {
    const float hi = static_cast<float>(value);
    return {hi, static_cast<float>(value - static_cast<double>(hi))};
}

bool valid_matrix(size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc) {
    return m > 0 && n > 0 && k > 0 && lda >= m && ldb >= k && ldc >= m;
}

int cpu_zgemm(size_t m, size_t n, size_t k, const metal_complex64 *a, size_t lda,
             const metal_complex64 *b, size_t ldb, metal_complex64 *c, size_t ldc) {
    if (!valid_matrix(m, n, k, lda, ldb, ldc) || !a || !b || !c) return METAL_BACKEND_INVALID_ARGUMENT;
    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < m; ++row) {
            cd sum{0.0, 0.0};
            for (size_t inner = 0; inner < k; ++inner) {
                const cd av{a[row + inner * lda].real, a[row + inner * lda].imag};
                const cd bv{b[inner + col * ldb].real, b[inner + col * ldb].imag};
                sum += av * bv;
            }
            c[row + col * ldc] = {sum.real(), sum.imag()};
        }
    }
    return METAL_BACKEND_OK;
}

int cpu_fft(size_t n, const metal_complex64 *input, metal_complex64 *output, int direction) {
    if (!input || !output || n == 0 || (direction != -1 && direction != 1)) return METAL_BACKEND_INVALID_ARGUMENT;
    const double sign = direction < 0 ? -1.0 : 1.0;
    const double pi2 = 2.0 * std::acos(-1.0);
    for (size_t q = 0; q < n; ++q) {
        cd sum{0.0, 0.0};
        for (size_t p = 0; p < n; ++p) {
            const cd value{input[p].real, input[p].imag};
            const double angle = sign * pi2 * static_cast<double>(p * q) / static_cast<double>(n);
            sum += value * cd{std::cos(angle), std::sin(angle)};
        }
        if (direction > 0) sum /= static_cast<double>(n);
        output[q] = {sum.real(), sum.imag()};
    }
    return METAL_BACKEND_OK;
}

int cpu_fft3d(size_t nx, size_t ny, size_t nz, const metal_complex64 *input,
              metal_complex64 *output, int direction) {
    if (!input || !output || nx == 0 || ny == 0 || nz == 0 || (direction != -1 && direction != 1)) return METAL_BACKEND_INVALID_ARGUMENT;
    const size_t total = nx * ny * nz;
    std::vector<metal_complex64> current(input, input + total), next(total), line_in, line_out;
    auto index = [nx, ny](size_t x, size_t y, size_t z) { return x + nx * (y + ny * z); };
    const size_t dims[3] = {nx, ny, nz};
    for (size_t axis = 0; axis < 3; ++axis) {
        const size_t n = dims[axis]; line_in.resize(n); line_out.resize(n);
        const size_t lines = total / n;
        for (size_t line = 0; line < lines; ++line) {
            for (size_t q = 0; q < n; ++q) {
                size_t x = 0, y = 0, z = 0;
                if (axis == 0) { y = line % ny; z = line / ny; x = q; }
                else if (axis == 1) { x = line % nx; z = line / nx; y = q; }
                else { x = line % nx; y = line / nx; z = q; }
                line_in[q] = current[index(x, y, z)];
            }
            const int rc = cpu_fft(n, line_in.data(), line_out.data(), direction);
            if (rc != METAL_BACKEND_OK) return rc;
            for (size_t q = 0; q < n; ++q) {
                size_t x = 0, y = 0, z = 0;
                if (axis == 0) { y = line % ny; z = line / ny; x = q; }
                else if (axis == 1) { x = line % nx; z = line / nx; y = q; }
                else { x = line % nx; y = line / nx; z = q; }
                next[index(x, y, z)] = line_out[q];
            }
        }
        current.swap(next);
    }
    std::copy(current.begin(), current.end(), output);
    return METAL_BACKEND_OK;
}

int cpu_sum(size_t n, const double *input, double *result) {
    if (!input || !result || n == 0) return METAL_BACKEND_INVALID_ARGUMENT;
    // Pairwise blocks reduce loss of significance while remaining deterministic.
    std::vector<double> partials;
    partials.reserve((n + 255) / 256);
    for (size_t base = 0; base < n; base += 256) {
        const size_t end = std::min(n, base + 256);
        double sum = 0.0;
        for (size_t i = base; i < end; ++i) sum += input[i];
        partials.push_back(sum);
    }
    while (partials.size() > 1) {
        std::vector<double> next;
        next.reserve((partials.size() + 1) / 2);
        for (size_t i = 0; i < partials.size(); i += 2)
            next.push_back(partials[i] + (i + 1 < partials.size() ? partials[i + 1] : 0.0));
        partials.swap(next);
    }
    *result = partials.front();
    return std::isfinite(*result) ? METAL_BACKEND_OK : METAL_BACKEND_RUNTIME_FAILED;
}

int cpu_eigh(size_t n, const metal_complex64 *a, size_t lda, double *w,
             metal_complex64 *v, size_t ldv) {
    if (!a || !w || !v || n == 0 || lda < n || ldv < n) return METAL_BACKEND_INVALID_ARGUMENT;
    std::vector<cd> h(n * n);
    for (size_t col = 0; col < n; ++col)
        for (size_t row = 0; row < n; ++row)
            h[row + col * n] = cd{a[row + col * lda].real, a[row + col * lda].imag};
    std::vector<cd> q(n * n, cd{0.0, 0.0});
    for (size_t i = 0; i < n; ++i) q[i + i * n] = 1.0;
    const size_t max_sweeps = std::max<size_t>(32, 16 * n * n);
    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        size_t p = 0, r = 0;
        double largest = 0.0;
        for (size_t j = 0; j < n; ++j) for (size_t i = 0; i < j; ++i) {
            if (std::abs(h[i + j * n]) > largest) { largest = std::abs(h[i + j * n]); p = i; r = j; }
        }
        if (largest < 1e-13) break;
        const cd hpq = h[p + r * n];
        const double mag = std::abs(hpq);
        if (mag == 0.0) continue;
        const double app = h[p + p * n].real();
        const double arr = h[r + r * n].real();
        const double tau = (arr - app) / (2.0 * mag);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double sreal = c * t;
        const cd phase = hpq / mag;
        const cd s = sreal * phase;
        for (size_t k = 0; k < n; ++k) {
            if (k == p || k == r) continue;
            const cd hkp = h[k + p * n], hkr = h[k + r * n];
            h[k + p * n] = c * hkp - std::conj(s) * hkr;
            h[p + k * n] = std::conj(h[k + p * n]);
            h[k + r * n] = c * hkr + s * hkp;
            h[r + k * n] = std::conj(h[k + r * n]);
        }
        h[p + p * n] = cd{app - t * mag, 0.0};
        h[r + r * n] = cd{arr + t * mag, 0.0};
        h[p + r * n] = h[r + p * n] = cd{0.0, 0.0};
        for (size_t k = 0; k < n; ++k) {
            const cd qkp = q[k + p * n], qkr = q[k + r * n];
            q[k + p * n] = c * qkp - std::conj(s) * qkr;
            q[k + r * n] = c * qkr + s * qkp;
        }
    }
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) { w[i] = h[i + i * n].real(); order[i] = i; }
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return w[x] < w[y]; });
    std::vector<double> ws(n);
    std::vector<cd> vs(n * n);
    for (size_t j = 0; j < n; ++j) {
        ws[j] = w[order[j]];
        for (size_t i = 0; i < n; ++i) vs[i + j * n] = q[i + order[j] * n];
    }
    for (size_t j = 0; j < n; ++j) for (size_t i = 0; i < n; ++i) {
        const cd x = vs[i + j * n];
        v[i + j * ldv] = {x.real(), x.imag()};
    }
    std::copy(ws.begin(), ws.end(), w);
    return METAL_BACKEND_OK;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) return nil;
    return [device newComputePipelineStateWithFunction:function error:&error];
}

bool validate_tiled_gemm(metal_backend *backend);
void calibrate_gemm(metal_backend *backend);

bool compile_gpu(metal_backend *backend) {
    backend->device = MTLCreateSystemDefaultDevice();
    if (!backend->device) {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (devices.count > 0) {
            for (id<MTLDevice> candidate in devices) {
                if (![candidate isLowPower]) { backend->device = candidate; break; }
            }
            if (!backend->device) backend->device = devices.firstObject;
        }
    }
    if (!backend->device) { backend->error = "Metal runtime exposed neither a default device nor any enumerated devices"; return false; }
    backend->metal4_family = false;
    backend->ml_api_available = false;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        backend->metal4_family = [backend->device supportsFamily:MTLGPUFamilyMetal4];
        backend->ml_api_available = backend->metal4_family;
    }
#endif
    backend->device_tuned = metal_policy::advanced(backend->device);
    backend->variant = backend->device_tuned ? "tiled-compute" : "generic-metal";
    backend->dispatch_width = 256;
    backend->queue = [backend->device newCommandQueue];
    if (!backend->queue) { backend->error = "Could not create Metal command queue"; return false; }
    NSError *error = nil;
    NSString *source = [[NSString stringWithUTF8String:kBackendShader] stringByAppendingString:[NSString stringWithUTF8String:kResidentShader]];
    source=[source stringByAppendingString:[NSString stringWithUTF8String:kLobpcgShader]];
    source=[source stringByAppendingString:[NSString stringWithUTF8String:kSubspaceShader]];
    source=[source stringByAppendingString:[NSString stringWithUTF8String:kCyclicShader]];
    source=[source stringByAppendingString:[NSString stringWithUTF8String:hamiltonian_shader]];
    source=[source stringByAppendingString:[NSString stringWithUTF8String:orthogonal_shader]];
    source=[source stringByAppendingString:[NSString stringWithUTF8String:wave_owner_shader]];
    source=[source stringByAppendingString:[NSString stringWithUTF8String:nonlocal_shader]];
    auto compiled=metal_policy::Compilation{};
    const bool metal4_macro=backend->device_tuned && backend->metal4_family;
    if(backend->precompiled_requested) {
        metal_library::Expectations expected;
        expected.device_tuned=backend->device_tuned?1:0;expected.metal4=metal4_macro?1:0;
        const auto allowed=metal_policy::languages(backend->device);
        expected.language_ceiling=metal_policy::language_name(allowed.front());
        const auto os=NSProcessInfo.processInfo.operatingSystemVersion;
        expected.os_version={{int(os.majorVersion),int(os.minorVersion),int(os.patchVersion)}};
        const char *pieces[]={kBackendShader,kResidentShader,kLobpcgShader,kSubspaceShader,kCyclicShader,hamiltonian_shader,orthogonal_shader,wave_owner_shader,nonlocal_shader};
        auto loaded=metal_library::load(backend->device,backend->precompiled_path.c_str(),source,pieces,expected);
        backend->precompiled_verified=loaded.verified;backend->precompiled_reason=loaded.reason;
        if(loaded.library) {
            compiled.library=loaded.library;
            const unsigned code=metal_library::language_code(loaded.language);
            compiled.language=static_cast<MTLLanguageVersion>(((code/100)<<16)+(code%100));
            backend->precompiled_used=true;
        } else ++backend->precompiled_fallbacks;
    }
    if(!compiled.library) compiled=metal_policy::compile(backend->device,source,@{
        @"METAL_VASP_DEVICE_TUNED": @(backend->device_tuned ? 1 : 0),
        @"METAL_VASP_METAL4": @(metal4_macro ? 1 : 0),
    });
    backend->compile_fallbacks=compiled.failures;
    if (!compiled.library && backend->device_tuned) {
        backend->device_tuned=false;
        backend->variant="generic-metal-after-compile-fallback";
        compiled=metal_policy::compile(backend->device,source,@{@"METAL_VASP_DEVICE_TUNED": @0, @"METAL_VASP_METAL4": @0});
        backend->compile_fallbacks+=compiled.failures;
    }
    backend->library=compiled.library;
    backend->language=metal_policy::language_name(compiled.language);
    if (!backend->library) { backend->error=compiled.error; return false; }
    backend->gemm = pipeline(backend->device, backend->library, @"ds_zgemm");
    if (backend->device_tuned) {
        backend->gemm_tiled=pipeline(backend->device,backend->library,@"ds_zgemm_tiled");
        if (backend->gemm_tiled) {
            MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
            descriptor.computeFunction=[backend->library newFunctionWithName:@"ds_zgemm_tiled"];
            // Only this kernel always dispatches complete 16x16 groups.
            descriptor.maxTotalThreadsPerThreadgroup=256;
            const NSUInteger width=backend->gemm_tiled.threadExecutionWidth;
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=width && 256%width==0;
            NSError *tile_error=nil;
            auto hinted=[backend->device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&tile_error];
            if (hinted && (!descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth ||
                           (hinted.threadExecutionWidth && 256%hinted.threadExecutionWidth==0))) backend->gemm_tiled=hinted;
        }
        backend->tiled_enabled=backend->gemm_tiled &&
            backend->gemm_tiled.maxTotalThreadsPerThreadgroup>=256 &&
            backend->gemm_tiled.staticThreadgroupMemoryLength<=backend->device.maxThreadgroupMemoryLength;
        if (!backend->tiled_enabled) ++backend->tiled_fallbacks;
    }
    backend->thread_execution_width=backend->gemm.threadExecutionWidth;
    backend->dispatch_width=std::max<NSUInteger>(1,std::min<NSUInteger>(256,backend->gemm.maxTotalThreadsPerThreadgroup));
    backend->reduce_stage = pipeline(backend->device, backend->library, @"ds_reduce_stage");
    backend->dot_stage = pipeline(backend->device, backend->library, @"ds_zdot_stage");
    backend->reduce_complex_stage = pipeline(backend->device, backend->library, @"ds_reduce_complex_stage");
    backend->vh_apply = pipeline(backend->device, backend->library, @"ds_vh_apply");
    backend->vh_apply_real = pipeline(backend->device, backend->library, @"ds_vh_apply_real");
    backend->dft = pipeline(backend->device, backend->library, @"ds_zdft");
    backend->fft_stage = pipeline(backend->device, backend->library, @"ds_fft_stage");
    backend->fft3d_reorder = pipeline(backend->device, backend->library, @"ds_fft3d_reorder");
    backend->fft3d_stage = pipeline(backend->device, backend->library, @"ds_fft3d_stage");
    backend->mixed_reorder = pipeline(backend->device, backend->library, @"ds_mixed_reorder");
    backend->mixed_stage = pipeline(backend->device, backend->library, @"ds_mixed_stage");
    backend->mixed_reorder32=pipeline(backend->device,backend->library,@"ds_mixed_reorder32");
    backend->mixed_stage32=pipeline(backend->device,backend->library,@"ds_mixed_stage32");
    backend->jacobi_select = pipeline(backend->device, backend->library, @"ds_jacobi_select");
    backend->jacobi_matrix = pipeline(backend->device, backend->library, @"ds_jacobi_matrix");
    backend->jacobi_vectors = pipeline(backend->device, backend->library, @"ds_jacobi_vectors");
    if (!backend->gemm || !backend->dot_stage || !backend->reduce_complex_stage || !backend->reduce_stage || !backend->vh_apply || !backend->vh_apply_real || !backend->dft || !backend->fft_stage || !backend->fft3d_reorder || !backend->fft3d_stage || !backend->mixed_reorder || !backend->mixed_stage || !backend->jacobi_select || !backend->jacobi_matrix || !backend->jacobi_vectors) { backend->error = "Metal backend pipeline creation failed"; return false; }
    backend->lob_init=pipeline(backend->device,backend->library,@"ds_lob_init");
    backend->lob_apply=pipeline(backend->device,backend->library,@"ds_lob_apply");
    backend->lob_update=pipeline(backend->device,backend->library,@"ds_lob_update");
    backend->lob_diagonal=pipeline(backend->device,backend->library,@"ds_lob_diagonal");
    if(!backend->lob_init || !backend->lob_apply || !backend->lob_update || !backend->lob_diagonal) { backend->error="LOBPCG pipeline creation failed"; return false; }
    backend->jacobi_local=pipeline(backend->device,backend->library,@"ds_jacobi_local");
    backend->chol_pivot=pipeline(backend->device,backend->library,@"ds_chol_pivot");
    backend->chol_column=pipeline(backend->device,backend->library,@"ds_chol_column");
    backend->lower_inverse=pipeline(backend->device,backend->library,@"ds_lower_inverse");
    backend->subspace_product=pipeline(backend->device,backend->library,@"ds_subspace_product");
    backend->hermitianize=pipeline(backend->device,backend->library,@"ds_hermitianize");
    if(!backend->jacobi_local || !backend->chol_pivot || !backend->chol_column || !backend->lower_inverse || !backend->subspace_product || !backend->hermitianize) {backend->error="Subspace pipeline creation failed";return false;}
    backend->cyclic_params=pipeline(backend->device,backend->library,@"ds_cyclic_params");
    backend->cyclic_right=pipeline(backend->device,backend->library,@"ds_cyclic_right");
    backend->cyclic_left=pipeline(backend->device,backend->library,@"ds_cyclic_left");
    backend->cyclic_check=pipeline(backend->device,backend->library,@"ds_cyclic_check");
    if(!backend->cyclic_params || !backend->cyclic_right || !backend->cyclic_left || !backend->cyclic_check) {backend->error="Cyclic Jacobi pipeline creation failed";return false;}
    backend->ham_scatter=pipeline(backend->device,backend->library,@"ds_ham_scatter");
    backend->ham_local=pipeline(backend->device,backend->library,@"ds_ham_local");
    backend->ham_onsite=pipeline(backend->device,backend->library,@"ds_ham_onsite");
    backend->ham_finish=pipeline(backend->device,backend->library,@"ds_ham_finish");
    if(!backend->ham_scatter || !backend->ham_local || !backend->ham_onsite || !backend->ham_finish) {backend->error="Hamiltonian pipeline creation failed";return false;}
    backend->orth_overlap=pipeline(backend->device,backend->library,@"ds_orth_overlap");
    backend->orth_overlap_group=pipeline(backend->device,backend->library,@"ds_orth_overlap_group");
    if(!backend->orth_overlap) {backend->error="Orthogonalization pipeline creation failed";return false;}
    backend->full_orth_metric=pipeline(backend->device,backend->library,@"ds_full_orth_metric");
    backend->full_orth_gram=pipeline(backend->device,backend->library,@"ds_full_orth_gram");
    backend->full_orth_update=pipeline(backend->device,backend->library,@"ds_full_orth_update");
    if(!backend->full_orth_metric || !backend->full_orth_gram || !backend->full_orth_update) {backend->error="Full orthogonalization pipeline creation failed";return false;}
    backend->resident_hamiltonian=pipeline(backend->device,backend->library,@"ds_resident_hamiltonian");
    backend->resident_project=pipeline(backend->device,backend->library,@"ds_resident_project");
    backend->resident_orth=pipeline(backend->device,backend->library,@"ds_resident_orthonormalize");
    if(!backend->resident_hamiltonian || !backend->resident_project || !backend->resident_orth) {backend->error="Resident pipeline creation failed";return false;}
    // The immutable projector pipeline belongs to the backend library, rather
    // than being recreated for each force/stress owner. Unsupported projector
    // compilation leaves this optional hook unavailable; other GPU routes stay.
    backend->nonlocal_project=pipeline(backend->device,backend->library,@"ds_nonlocal_project");
    backend->use_gpu = true;
    if (backend->tiled_enabled && !validate_tiled_gemm(backend)) {
        backend->tiled_enabled=false;
        ++backend->tiled_fallbacks;
        backend->variant="generic-metal-after-tiled-numerical-fallback";
    }
    if (backend->tiled_enabled) {
        calibrate_gemm(backend);
        backend->variant=backend->tiled_auto_selected ? "tiled-compute-calibrated" :
            (backend->gemm_calibrated ? "generic-compute-calibrated" : "generic-compute-uncalibrated");
    }
    // Initialization checks must not be mistaken for VASP work or throughput.
    backend->tiled_calls=0;
    backend->command_count=backend->wait_count=0;
    backend->gpu_seconds=backend->wait_seconds=backend->conversion_seconds=0.;
    backend->metal4_command_calls=backend->metal4_command_success=backend->metal4_command_fallbacks=0;
    if(backend->metal4_command_requested) {
        char reason[2048]{};
        backend->metal4_commands=metal4_dispatch_create(backend->device,backend->library,reason,sizeof(reason));
        backend->metal4_command_reason=backend->metal4_commands ? "none" : reason;
    }
    return true;
}

bool trans_valid(char t) {
    return t == 'N' || t == 'n' || t == 'T' || t == 't' || t == 'C' || t == 'c';
}

bool trans_conjugates(char t) { return t == 'C' || t == 'c'; }
bool transposes(char t) { return t != 'N' && t != 'n'; }

cd load_blas(const metal_complex64 *x, size_t ld, size_t row, size_t col, char trans) {
    if (!transposes(trans)) return cd{x[row + col * ld].real, x[row + col * ld].imag};
    cd v{x[col + row * ld].real, x[col + row * ld].imag};
    return trans_conjugates(trans) ? std::conj(v) : v;
}

int gpu_zgemm_window(const metal_backend *backend, char ta, char tb,
                   size_t m, size_t n, size_t k, metal_complex64 alpha,
                   const metal_complex64 *a, size_t lda,
                   const metal_complex64 *b, size_t ldb,
                   metal_complex64 beta, metal_complex64 *c, size_t ldc, bool validate_tile=false) {
    if (!trans_valid(ta) || !trans_valid(tb) || !a || !b || !c || m == 0 || n == 0 || k == 0) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard staging_guard(backend->staging_mutex);
    const size_t a_rows = transposes(ta) ? k : m;
    const size_t b_rows = transposes(tb) ? n : k;
    const size_t required[]={m*k*sizeof(vector_float4),k*n*sizeof(vector_float4),m*n*sizeof(vector_float4)};
    id<MTLBuffer> __strong *cache[]={&backend->gemm_a,&backend->gemm_b,&backend->gemm_c};
    size_t retained=0;for(size_t i=0;i<3;++i) {
        auto bytes=std::max<size_t>(required[i],(*cache[i]).length);
        if(retained>SIZE_MAX-bytes){retained=SIZE_MAX;break;}retained+=bytes;
    }
    const size_t working=metal_memory::budget(backend->device) ? metal_memory::budget(backend->device):backend->device.maxBufferLength;
    if(retained>working)for(size_t i=0;i<3;++i)if((*cache[i]).length>required[i])*cache[i]=nil;
    if (!metal_gemm::host_shape(m,n,k,transposes(ta),transposes(tb),lda,ldb,ldc)) return METAL_BACKEND_INVALID_ARGUMENT;
    if (!backend->gemm_a || backend->gemm_a.length < m*k*sizeof(vector_float4)) {backend->gemm_a=nil;backend->gemm_a=backend_buffer(backend,m*k*sizeof(vector_float4),MTLResourceStorageModeShared);}
    if (!backend->gemm_b || backend->gemm_b.length < k*n*sizeof(vector_float4)) {backend->gemm_b=nil;backend->gemm_b=backend_buffer(backend,k*n*sizeof(vector_float4),MTLResourceStorageModeShared);}
    if (!backend->gemm_c || backend->gemm_c.length < m*n*sizeof(vector_float4)) {backend->gemm_c=nil;backend->gemm_c=backend_buffer(backend,m*n*sizeof(vector_float4),MTLResourceStorageModeShared);}
    id<MTLBuffer> ab=backend->gemm_a,bb=backend->gemm_b,cb=backend->gemm_c;
    if(!ab || !bb || !cb) return METAL_BACKEND_ALLOCATION_FAILED;
    {
    ConversionTimer timer{backend};
    auto *ah=static_cast<vector_float4 *>(ab.contents),*bh=static_cast<vector_float4 *>(bb.contents),*ch=static_cast<vector_float4 *>(cb.contents);
    for (size_t row = 0; row < m; ++row) for (size_t inner = 0; inner < k; ++inner) {
        const cd v = load_blas(a, lda, row, inner, ta);
        const auto x = split_double(v.real()), y = split_double(v.imag());
        if(!std::isfinite(x.hi) || !std::isfinite(x.lo) || !std::isfinite(y.hi) || !std::isfinite(y.lo))return METAL_BACKEND_UNSUPPORTED;
        ah[row * k + inner] = (vector_float4){x.hi, x.lo, y.hi, y.lo};
    }
    for (size_t inner = 0; inner < k; ++inner) for (size_t col = 0; col < n; ++col) {
        const cd v = load_blas(b, ldb, inner, col, tb);
        const auto x = split_double(v.real()), y = split_double(v.imag());
        if(!std::isfinite(x.hi) || !std::isfinite(x.lo) || !std::isfinite(y.hi) || !std::isfinite(y.lo))return METAL_BACKEND_UNSUPPORTED;
        bh[inner * n + col] = (vector_float4){x.hi, x.lo, y.hi, y.lo};
    }
    for (size_t row = 0; row < m; ++row) for (size_t col = 0; col < n; ++col) {
        // BLAS beta=0 must never read C: VASP uses uninitialized work arrays.
        const bool zero_beta = beta.real == 0.0 && beta.imag == 0.0;
        const auto x = split_double(zero_beta ? 0.0 : c[row + col * ldc].real);
        const auto y = split_double(zero_beta ? 0.0 : c[row + col * ldc].imag);
        if(!std::isfinite(x.hi) || !std::isfinite(x.lo) || !std::isfinite(y.hi) || !std::isfinite(y.lo))return METAL_BACKEND_UNSUPPORTED;
        ch[row * n + col] = (vector_float4){x.hi, x.lo, y.hi, y.lo};
    }
    }
    const auto ar = split_double(alpha.real), ai = split_double(alpha.imag);
    const auto br = split_double(beta.real), bi = split_double(beta.imag);
    for(auto z:{ar,ai,br,bi})if(!std::isfinite(z.hi) || !std::isfinite(z.lo))return METAL_BACKEND_UNSUPPORTED;
    HostGemmParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                 (vector_float4){ar.hi, ar.lo, ai.hi, ai.lo},
                 (vector_float4){br.hi, br.lo, bi.hi, bi.lo}};
    const char *kernel_override=std::getenv("VASP_METAL_GEMM_KERNEL");
    const bool force_generic=backend->gemm_check_mode==1 ||
        (backend->gemm_check_mode==0 && kernel_override && std::strcmp(kernel_override,"generic")==0);
    const bool force_tiled=backend->gemm_check_mode==2 ||
        (backend->gemm_check_mode==0 && kernel_override && std::strcmp(kernel_override,"tiled")==0);
    const bool tiled=backend->tiled_enabled && (validate_tile || !force_generic) &&
        (validate_tile || force_tiled || (backend->tiled_auto_selected && m>=256 && n>=256 && k>=256));
    NSUInteger threads = std::min<NSUInteger>(backend->gemm.maxTotalThreadsPerThreadgroup, backend->dispatch_width);
    bool completed=false;
    if(backend->metal4_command_requested && backend->metal4_commands) {
        char reason[2048]{}; double gpu=0.,wait=0.;
        ++backend->metal4_command_calls;
        const int rc=metal4_dispatch_zgemm(backend->metal4_commands,tiled,ab,bb,cb,&p,sizeof(p),m,n,threads,&gpu,&wait,reason,sizeof(reason));
        if(wait>0.) { ++backend->command_count; ++backend->wait_count; backend->wait_seconds+=wait; }
        backend->gpu_seconds+=gpu;
        if(rc==0) { completed=true; ++backend->metal4_command_success; }
        else {
            backend->metal4_command_reason=reason;
            // Only the GPU staging C may have changed. Caller-owned C is still
            // the original BLAS input, so restore it before applying beta again.
            ConversionTimer timer{backend}; auto *ch=static_cast<vector_float4 *>(cb.contents);
            const bool zero_beta=beta.real==0. && beta.imag==0.;
            for(size_t row=0;row<m;++row) for(size_t col=0;col<n;++col) {
                const auto x=split_double(zero_beta ? 0. : c[row+col*ldc].real);
                const auto y=split_double(zero_beta ? 0. : c[row+col*ldc].imag);
                ch[row*n+col]=(vector_float4){x.hi,x.lo,y.hi,y.lo};
            }
        }
    }
    if(!completed) {
        if(backend->metal4_command_requested) ++backend->metal4_command_fallbacks;
        id<MTLCommandBuffer> command=[backend->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if(!command || !encoder) return METAL_BACKEND_RUNTIME_FAILED;
        [encoder setComputePipelineState:tiled ? backend->gemm_tiled : backend->gemm]; [encoder setBuffer:ab offset:0 atIndex:0];
        [encoder setBuffer:bb offset:0 atIndex:1]; [encoder setBuffer:cb offset:0 atIndex:2]; [encoder setBytes:&p length:sizeof(p) atIndex:3];
        if(tiled) [encoder dispatchThreadgroups:MTLSizeMake((n+15)/16,(m+15)/16,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        else [encoder dispatchThreads:MTLSizeMake(n,m,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        [encoder endEncoding];
        if(complete_command(backend,command)!=METAL_BACKEND_OK) {
            if(tiled) {backend->tiled_enabled=false;++backend->tiled_fallbacks;}
            return METAL_BACKEND_RUNTIME_FAILED;
        }
    }
    if (tiled) ++backend->tiled_calls;
    const auto *out = static_cast<const vector_float4 *>(cb.contents);
    for(size_t i=0;i<m*n;++i)if(!std::isfinite(out[i].x) || !std::isfinite(out[i].y) || !std::isfinite(out[i].z) || !std::isfinite(out[i].w))return METAL_BACKEND_RUNTIME_FAILED;
    for (size_t col = 0; col < n; ++col) for (size_t row = 0; row < m; ++row) {
        const auto &x = out[row * n + col]; c[row + col * ldc] = {split_to_double(x.x, x.y), split_to_double(x.z, x.w)};
    }
    return METAL_BACKEND_OK;
}

int gpu_zgemm_blas(const metal_backend *backend,char ta,char tb,size_t m,size_t n,size_t k,
 metal_complex64 alpha,const metal_complex64 *a,size_t lda,const metal_complex64 *b,size_t ldb,
 metal_complex64 beta,metal_complex64 *c,size_t ldc,bool validate_tile=false) {
    if(!backend || !backend->use_gpu)return METAL_BACKEND_UNSUPPORTED;
    if(!trans_valid(ta) || !trans_valid(tb) || !a || !b || !c)return METAL_BACKEND_INVALID_ARGUMENT;
    size_t cap=backend->device.maxBufferLength;
    if(const char *value=std::getenv("VASP_METAL_GEMM_CHUNK_BYTES")) {
        char *end=nullptr;auto parsed=std::strtoull(value,&end,10);
        if(end && *end=='\0' && parsed && parsed<=SIZE_MAX)cap=std::min(cap,size_t(parsed));
    }
    metal_gemm::Plan plan;
    auto result=metal_gemm::plan(m,n,k,transposes(ta),transposes(tb),lda,ldb,ldc,
        cap,metal_memory::budget(backend->device),plan);
    if(result==metal_gemm::Result::invalid_shape)return METAL_BACKEND_INVALID_ARGUMENT;
    if(result!=metal_gemm::Result::ok)return METAL_BACKEND_ALLOCATION_FAILED;
    if(!plan.chunked)return gpu_zgemm_window(backend,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,validate_tile);
    ++backend->chunks.gemm_calls;
    backend->chunks.gemm_peak_workspace_bytes=std::max<uint64_t>(backend->chunks.gemm_peak_workspace_bytes,plan.atomic_host_bytes+plan.bytes[0]+plan.bytes[1]+plan.bytes[2]);
    auto pack=[](cd z) {auto re=split_double(z.real()),im=split_double(z.imag());return (vector_float4){re.hi,re.lo,im.hi,im.lo};};
    auto finite=[](vector_float4 z) {return std::isfinite(z.x) && std::isfinite(z.y) && std::isfinite(z.z) && std::isfinite(z.w);};
    const auto alpha_ds=pack({alpha.real,alpha.imag}),beta_ds=pack({beta.real,beta.imag});
    if(!finite(alpha_ds) || !finite(beta_ds))return METAL_BACKEND_UNSUPPORTED;
    try {
        // Caller C is unchanged until every panel succeeds, including when a
        // late input or command failure forces the original CPU fallback.
        std::vector<metal_complex64> staged(m*n);
        metal_memory::OperatorGuard lock(backend->staging_mutex);
        id<MTLBuffer> __strong *slots[]={&backend->gemm_a,&backend->gemm_b,&backend->gemm_c};
        for(size_t i=0;i<3;++i) {
            if(*slots[i] && (*slots[i]).length>plan.bytes[i])*slots[i]=nil;
            if(!*slots[i] || (*slots[i]).length<plan.bytes[i])
                *slots[i]=backend_buffer(backend,plan.bytes[i],MTLResourceStorageModeShared);
            if(!*slots[i])return METAL_BACKEND_ALLOCATION_FAILED;
        }
        const auto ab=backend->gemm_a,bb=backend->gemm_b,cb=backend->gemm_c;
        auto *ah=static_cast<vector_float4 *>(ab.contents),*bh=static_cast<vector_float4 *>(bb.contents),*ch=static_cast<vector_float4 *>(cb.contents);
        const bool zero_beta=beta.real==0. && beta.imag==0.;
        const char *override=std::getenv("VASP_METAL_GEMM_KERNEL");
        const bool generic=backend->gemm_check_mode==1 || (backend->gemm_check_mode==0 && override && std::strcmp(override,"generic")==0);
        const bool forced=backend->gemm_check_mode==2 || (backend->gemm_check_mode==0 && override && std::strcmp(override,"tiled")==0);
        for(size_t col=0;col<n;col+=plan.cols)for(size_t row=0;row<m;row+=plan.rows) {
            const size_t rows=std::min(plan.rows,m-row),cols=std::min(plan.cols,n-col);
            {ConversionTimer timer{backend};
             for(size_t i=0;i<rows;++i)for(size_t j=0;j<cols;++j) {
                auto z=zero_beta ? pack({0.,0.}):pack({c[row+i+(col+j)*ldc].real,c[row+i+(col+j)*ldc].imag});
                if(!finite(z))return METAL_BACKEND_UNSUPPORTED;ch[i*cols+j]=z;
             }}
            for(size_t inner=0;inner<k;inner+=plan.inner) {
                const size_t width=std::min(plan.inner,k-inner);
                {ConversionTimer timer{backend};
                 for(size_t i=0;i<rows;++i)for(size_t q=0;q<width;++q) {
                    auto z=pack(load_blas(a,lda,row+i,inner+q,ta));if(!finite(z))return METAL_BACKEND_UNSUPPORTED;ah[i*width+q]=z;
                 }
                 for(size_t q=0;q<width;++q)for(size_t j=0;j<cols;++j) {
                    auto z=pack(load_blas(b,ldb,inner+q,col+j,tb));if(!finite(z))return METAL_BACKEND_UNSUPPORTED;bh[q*cols+j]=z;
                 }}
                HostGemmParams parameters{(uint32_t)rows,(uint32_t)cols,(uint32_t)width,alpha_ds,
                    inner==0 ? beta_ds:(vector_float4){1,0,0,0}};
                const bool tiled=backend->tiled_enabled && !generic && (forced || (backend->tiled_auto_selected && rows>=256 && cols>=256 && width>=256));
                // Metal 3 handles bounded panels. It keeps C resident across K
                // without a host restore boundary on Metal 4 command failure.
                if(backend->metal4_command_requested)++backend->metal4_command_fallbacks;
                id<MTLCommandBuffer> command=[backend->queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
                if(!command || !encoder)return METAL_BACKEND_RUNTIME_FAILED;
                [encoder setComputePipelineState:tiled ? backend->gemm_tiled:backend->gemm];
                [encoder setBuffer:ab offset:0 atIndex:0];[encoder setBuffer:bb offset:0 atIndex:1];[encoder setBuffer:cb offset:0 atIndex:2];
                [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
                if(tiled)[encoder dispatchThreadgroups:MTLSizeMake((cols+15)/16,(rows+15)/16,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
                else [encoder dispatchThreads:MTLSizeMake(cols,rows,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(backend->dispatch_width,backend->gemm.maxTotalThreadsPerThreadgroup),1,1)];
                [encoder endEncoding];++backend->chunks.gemm_windows;int rc=complete_command(backend,command);if(rc)return rc;
                if(tiled)++backend->tiled_calls;
            }
            {ConversionTimer timer{backend};
             for(size_t i=0;i<rows;++i)for(size_t j=0;j<cols;++j) {
                auto z=ch[i*cols+j];if(!finite(z))return METAL_BACKEND_RUNTIME_FAILED;
                staged[row+i+(col+j)*m]={split_to_double(z.x,z.y),split_to_double(z.z,z.w)};
             }}
        }
        for(size_t j=0;j<n;++j)std::copy_n(staged.data()+j*m,m,c+j*ldc);
        return METAL_BACKEND_OK;
    } catch(const std::bad_alloc &) {return METAL_BACKEND_ALLOCATION_FAILED;}
}

bool validate_tiled_gemm(metal_backend *backend) {
    // Startup gate covers partial tiles, complex alpha/beta, and cancellation.
    constexpr size_t m=17,n=19,k=23,ldc=20;
    std::vector<metal_complex64> a(m*k),b(k*n),c(ldc*n,{123.,-456.}),reference(c);
    for(size_t i=0;i<a.size();++i) a[i]={std::sin(.13*i),std::cos(.17*i)};
    for(size_t i=0;i<b.size();++i) b[i]={std::cos(.11*i),std::sin(.19*i)};
    const metal_complex64 alpha{.7,-.3}, beta{-.2,.4};
    for(size_t j=0;j<n;++j) for(size_t i=0;i<m;++i) {
        std::complex<long double> sum=0.;
        for(size_t q=0;q<k;++q) sum+=std::complex<long double>(a[i+q*m].real,a[i+q*m].imag)*std::complex<long double>(b[q+j*k].real,b[q+j*k].imag);
        auto x=std::complex<long double>(alpha.real,alpha.imag)*sum+std::complex<long double>(beta.real,beta.imag)*std::complex<long double>(123.,-456.);
        reference[i+j*ldc]={double(x.real()),double(x.imag())};
    }
    if (gpu_zgemm_blas(backend,'N','N',m,n,k,alpha,a.data(),m,b.data(),k,beta,c.data(),ldc,true)) return false;
    for(size_t i=0;i<c.size();++i)
        if (!std::isfinite(c[i].real) || !std::isfinite(c[i].imag) ||
            std::hypot(c[i].real-reference[i].real,c[i].imag-reference[i].imag)>1e-10) return false;
    return true;
}

void calibrate_gemm(metal_backend *backend) {
    const char *setting=std::getenv("VASP_METAL_GEMM_CALIBRATE");
    if(setting && std::strcmp(setting,"0")==0) return;
    constexpr size_t side=256;
    std::vector<metal_complex64> a(side*side),b(side*side),c(side*side);
    for(size_t i=0;i<a.size();++i) {a[i]={std::sin(.01*i)*.1,std::cos(.02*i)*.1};b[i]={std::cos(.03*i)*.1,std::sin(.04*i)*.1};}
    std::vector<double> wall[2],gpu[2];
    // Warm both PSOs, then alternate the timing order to reduce clock bias.
    for(int round=-1;round<3;++round) for(int index=0;index<2;++index) {
        const int mode=(index+(round>0 ? round%2:0))%2;
        backend->gemm_check_mode=mode+1;
        auto start=SteadyClock::now();double seconds=backend->gpu_seconds;
        if(gpu_zgemm_blas(backend,'N','N',side,side,side,{1,0},a.data(),side,b.data(),side,{0,0},c.data(),side)) {
            backend->gemm_check_mode=0;return;
        }
        if(round>=0) {wall[mode].push_back(std::chrono::duration<double>(SteadyClock::now()-start).count());gpu[mode].push_back(backend->gpu_seconds-seconds);}
    }
    backend->gemm_check_mode=0;
    for(int mode=0;mode<2;++mode) {std::sort(wall[mode].begin(),wall[mode].end());std::sort(gpu[mode].begin(),gpu[mode].end());}
    backend->gemm_calibrated=true;
    // Require a material benefit in both operator wall time and device time.
    const bool measured=std::isfinite(wall[0][1]) && std::isfinite(wall[1][1]) &&
        std::isfinite(gpu[0][1]) && std::isfinite(gpu[1][1]) &&
        wall[0][1]>0 && wall[1][1]>0 && gpu[0][1]>0 && gpu[1][1]>0;
    backend->tiled_auto_selected=measured && wall[1][1]<.90*wall[0][1] && gpu[1][1]<.90*gpu[0][1];
}

int gpu_sum(const metal_backend *backend,size_t n,const double *input,double *result) {
    if(!input || !result || !n || n>UINT32_MAX) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard guard(backend->sum_mutex);
    size_t bytes=n*sizeof(vector_float2),partial_bytes=std::max<size_t>(1,(n+255)/256)*sizeof(vector_float2);
    if(!backend->sum_x || backend->sum_x.length<bytes) {backend->sum_x=nil;backend->sum_x=backend_buffer(backend,bytes,MTLResourceStorageModeShared);}
    for(auto *slot:{&backend->sum_r0,&backend->sum_r1})
        if(!*slot || (*slot).length<partial_bytes) {*slot=nil;*slot=backend_buffer(backend,partial_bytes,MTLResourceStorageModeShared);}
    if(!backend->sum_x || !backend->sum_r0 || !backend->sum_r1) return METAL_BACKEND_ALLOCATION_FAILED;
    { ConversionTimer timer{backend};auto *x=static_cast<vector_float2 *>(backend->sum_x.contents);
      for(size_t i=0;i<n;++i) {auto d=split_double(input[i]);x[i]=(vector_float2){d.hi,d.lo};} }
    id<MTLCommandBuffer> command=[backend->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    id<MTLBuffer> current=backend->sum_x,next=backend->sum_r0;
    size_t count=n;
    while(count>1) {
        uint32_t c=count,chunk=256;size_t next_count=(count+255)/256;
        [encoder setComputePipelineState:backend->reduce_stage];[encoder setBuffer:current offset:0 atIndex:0];[encoder setBuffer:next offset:0 atIndex:1];
        [encoder setBytes:&c length:sizeof(c) atIndex:2];[encoder setBytes:&chunk length:sizeof(chunk) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(next_count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];current=next;next=current==backend->sum_r0 ? backend->sum_r1:backend->sum_r0;count=next_count;
    }
    [encoder endEncoding];int rc=complete_command(backend,command);if(rc) return rc;
    auto d=static_cast<const vector_float2 *>(current.contents)[0];double value=split_to_double(d.x,d.y);
    if(!std::isfinite(value)) return METAL_BACKEND_RUNTIME_FAILED;
    *result=value;return METAL_BACKEND_OK;
}

int gpu_vh_apply_batch(const metal_backend *backend, size_t np, size_t mplwv, size_t nrspinors,
                       size_t nbands, const metal_complex64 *sv, const metal_complex64 *cr,
                       metal_complex64 *cvr, double rinplw);
int gpu_vh_apply_real_batch(const metal_backend *backend, size_t np, size_t mplwv, size_t nrspinors,
                            size_t nbands, const double *sv, const metal_complex64 *cr,
                            metal_complex64 *cvr, double rinplw);

int gpu_vh_apply(const metal_backend *backend, size_t np, size_t mplwv, size_t nrspinors,
                 const metal_complex64 *sv, const metal_complex64 *cr, metal_complex64 *cvr,
                 double rinplw) {
    return gpu_vh_apply_batch(backend, np, mplwv, nrspinors, 1, sv, cr, cvr, rinplw);
}

int gpu_vh_apply_batch(const metal_backend *backend, size_t np, size_t mplwv, size_t nrspinors,
                       size_t nbands, const metal_complex64 *sv, const metal_complex64 *cr,
                       metal_complex64 *cvr, double rinplw) {
    if (!backend || !backend->use_gpu || !sv || !cr || !cvr || np == 0 || mplwv < np ||
        nbands == 0 || (nrspinors != 1 && nrspinors != 2) || !std::isfinite(rinplw))
        return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard staging_guard(backend->staging_mutex);
    const size_t sv_count = mplwv * nrspinors * nrspinors;
    const size_t wf_count = mplwv * nrspinors * nbands;
    if(!backend->vh_sv || backend->vh_sv.length<sv_count*sizeof(vector_float4)) {backend->vh_sv=nil;backend->vh_sv=backend_buffer(backend,sv_count*sizeof(vector_float4),MTLResourceStorageModeShared);}
    if(!backend->vh_cr || backend->vh_cr.length<wf_count*sizeof(vector_float4)) {backend->vh_cr=nil;backend->vh_cr=backend_buffer(backend,wf_count*sizeof(vector_float4),MTLResourceStorageModeShared);}
    if(!backend->vh_out || backend->vh_out.length<wf_count*sizeof(vector_float4)) {backend->vh_out=nil;backend->vh_out=backend_buffer(backend,wf_count*sizeof(vector_float4),MTLResourceStorageModeShared);}
    id<MTLBuffer> sb=backend->vh_sv,cb=backend->vh_cr,out=backend->vh_out;
    if(!sb || !cb || !out) return METAL_BACKEND_ALLOCATION_FAILED;
    { ConversionTimer timer{backend};auto *sh=static_cast<vector_float4 *>(sb.contents);auto *ch=static_cast<vector_float4 *>(cb.contents);
    for (size_t i = 0; i < sv_count; ++i) {
        const auto r = split_double(sv[i].real), im = split_double(sv[i].imag);
        sh[i] = (vector_float4){r.hi, r.lo, im.hi, im.lo};
    }
    for (size_t i = 0; i < wf_count; ++i) {
        const auto r = split_double(cr[i].real), im = split_double(cr[i].imag);
        ch[i] = (vector_float4){r.hi, r.lo, im.hi, im.lo};
    }
    }
    struct VHApplyParams { uint32_t np, sv_stride, wf_stride, nrspinors, nbands, batch_stride; vector_float2 rinplw; } p{
        static_cast<uint32_t>(np), static_cast<uint32_t>(mplwv), static_cast<uint32_t>(mplwv), static_cast<uint32_t>(nrspinors), static_cast<uint32_t>(nbands), static_cast<uint32_t>(mplwv*nrspinors),
        (vector_float2){split_double(rinplw).hi, split_double(rinplw).lo}};
    id<MTLCommandBuffer> command = [backend->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:backend->vh_apply];
    [encoder setBuffer:sb offset:0 atIndex:0]; [encoder setBuffer:cb offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2]; [encoder setBytes:&p length:sizeof(p) atIndex:3];
    NSUInteger threads = std::min<NSUInteger>(backend->vh_apply.maxTotalThreadsPerThreadgroup, backend->dispatch_width);
    [encoder dispatchThreads:MTLSizeMake(np * nrspinors * nbands, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding]; if (complete_command(backend,command)!=METAL_BACKEND_OK) return METAL_BACKEND_RUNTIME_FAILED;
    const auto *result = static_cast<const vector_float4 *>(out.contents);
    for (size_t b=0;b<nbands;++b) for(size_t spin=0;spin<nrspinors;++spin) for(size_t j=0;j<np;++j) {
        size_t i=b*mplwv*nrspinors+spin*mplwv+j; cvr[i] = {split_to_double(result[i].x, result[i].y), split_to_double(result[i].z, result[i].w)}; }
    return METAL_BACKEND_OK;
}

int gpu_vh_apply_real(const metal_backend *backend, size_t np, size_t mplwv, size_t nrspinors,
                      const double *sv, const metal_complex64 *cr, metal_complex64 *cvr,
                      double rinplw) {
    return gpu_vh_apply_real_batch(backend, np, mplwv, nrspinors, 1, sv, cr, cvr, rinplw);
}

int gpu_vh_apply_real_batch(const metal_backend *backend, size_t np, size_t mplwv, size_t nrspinors,
                            size_t nbands, const double *sv, const metal_complex64 *cr,
                            metal_complex64 *cvr, double rinplw) {
    if (!backend || !backend->use_gpu || !sv || !cr || !cvr || np == 0 || mplwv < np ||
        nbands == 0 || (nrspinors != 1 && nrspinors != 2) || !std::isfinite(rinplw))
        return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard staging_guard(backend->staging_mutex);
    const size_t sv_count = 2 * mplwv * nrspinors * nrspinors;
    const size_t wf_count = mplwv * nrspinors * nbands;
    if(!backend->vh_sv || backend->vh_sv.length<sv_count*sizeof(vector_float2)) {backend->vh_sv=nil;backend->vh_sv=backend_buffer(backend,sv_count*sizeof(vector_float2),MTLResourceStorageModeShared);}
    if(!backend->vh_cr || backend->vh_cr.length<wf_count*sizeof(vector_float4)) {backend->vh_cr=nil;backend->vh_cr=backend_buffer(backend,wf_count*sizeof(vector_float4),MTLResourceStorageModeShared);}
    if(!backend->vh_out || backend->vh_out.length<wf_count*sizeof(vector_float4)) {backend->vh_out=nil;backend->vh_out=backend_buffer(backend,wf_count*sizeof(vector_float4),MTLResourceStorageModeShared);}
    id<MTLBuffer> sb=backend->vh_sv,cb=backend->vh_cr,out=backend->vh_out;
    if(!sb || !cb || !out) return METAL_BACKEND_ALLOCATION_FAILED;
    { ConversionTimer timer{backend};auto *sh=static_cast<vector_float2 *>(sb.contents);auto *ch=static_cast<vector_float4 *>(cb.contents);
    for (size_t i = 0; i < sv_count; ++i) { auto x = split_double(sv[i]); sh[i] = (vector_float2){x.hi, x.lo}; }
    for (size_t i = 0; i < wf_count; ++i) {
        auto r = split_double(cr[i].real), im = split_double(cr[i].imag);
        ch[i] = (vector_float4){r.hi, r.lo, im.hi, im.lo};
    }
    }
    struct VHApplyParams { uint32_t np, sv_stride, wf_stride, nrspinors, nbands, batch_stride; vector_float2 rinplw; } p{
        static_cast<uint32_t>(np), static_cast<uint32_t>(2*mplwv), static_cast<uint32_t>(mplwv), static_cast<uint32_t>(nrspinors), static_cast<uint32_t>(nbands), static_cast<uint32_t>(mplwv*nrspinors),
        (vector_float2){split_double(rinplw).hi, split_double(rinplw).lo}};
    id<MTLCommandBuffer> command = [backend->queue commandBuffer]; id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:backend->vh_apply_real]; [encoder setBuffer:sb offset:0 atIndex:0]; [encoder setBuffer:cb offset:0 atIndex:1]; [encoder setBuffer:out offset:0 atIndex:2]; [encoder setBytes:&p length:sizeof(p) atIndex:3];
    NSUInteger threads = std::min<NSUInteger>(backend->vh_apply_real.maxTotalThreadsPerThreadgroup, backend->dispatch_width);
    [encoder dispatchThreads:MTLSizeMake(np*nrspinors*nbands,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [encoder endEncoding]; if (complete_command(backend,command)!=METAL_BACKEND_OK) return METAL_BACKEND_RUNTIME_FAILED;
    const auto *result = static_cast<const vector_float4 *>(out.contents);
    for(size_t b=0;b<nbands;++b) for(size_t spin=0;spin<nrspinors;++spin) for(size_t j=0;j<np;++j) {size_t i=b*mplwv*nrspinors+spin*mplwv+j;cvr[i]={split_to_double(result[i].x,result[i].y),split_to_double(result[i].z,result[i].w)};}
    return METAL_BACKEND_OK;
}

int gpu_zdotc(const metal_backend *backend,size_t n,const metal_complex64 *x,ptrdiff_t incx,
                const metal_complex64 *y,ptrdiff_t incy,metal_complex64 *result) {
    if(!backend || !backend->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    if(!x || !y || !result || !n || !incx || !incy || n>UINT32_MAX) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard guard(backend->dot_mutex);
    const size_t input_bytes=n*sizeof(vector_float4);
    const size_t partial_count=(n+255)/256;
    const size_t partial_bytes=std::max<size_t>(1,partial_count)*sizeof(vector_float4);
    if(!backend->dot_x || backend->dot_x.length<input_bytes) {backend->dot_x=nil;backend->dot_x=backend_buffer(backend,input_bytes,MTLResourceStorageModeShared);}
    if(!backend->dot_y || backend->dot_y.length<input_bytes) {backend->dot_y=nil;backend->dot_y=backend_buffer(backend,input_bytes,MTLResourceStorageModeShared);}
    if(!backend->dot_r0 || backend->dot_r0.length<partial_bytes) {backend->dot_r0=nil;backend->dot_r0=backend_buffer(backend,partial_bytes,MTLResourceStorageModeShared);}
    if(!backend->dot_r1 || backend->dot_r1.length<partial_bytes) {backend->dot_r1=nil;backend->dot_r1=backend_buffer(backend,partial_bytes,MTLResourceStorageModeShared);}
    if(!backend->dot_x || !backend->dot_y || !backend->dot_r0 || !backend->dot_r1) return METAL_BACKEND_ALLOCATION_FAILED;
    auto *xh=static_cast<vector_float4 *>(backend->dot_x.contents);
    auto *yh=static_cast<vector_float4 *>(backend->dot_y.contents);
    const ptrdiff_t start_x=incx<0 ? (1-static_cast<ptrdiff_t>(n))*incx:0;
    const ptrdiff_t start_y=incy<0 ? (1-static_cast<ptrdiff_t>(n))*incy:0;
    for(size_t i=0;i<n;++i) {
        const auto a=x[start_x+static_cast<ptrdiff_t>(i)*incx],b=y[start_y+static_cast<ptrdiff_t>(i)*incy];
        const auto ar=split_double(a.real),ai=split_double(a.imag),br=split_double(b.real),bi=split_double(b.imag);
        xh[i]=(vector_float4){ar.hi,ar.lo,ai.hi,ai.lo}; yh[i]=(vector_float4){br.hi,br.lo,bi.hi,bi.lo};
    }
    id<MTLBuffer> xb=backend->dot_x, yb=backend->dot_y;
    size_t count=partial_count;
    id<MTLBuffer> current=backend->dot_r0, next=backend->dot_r1;
    id<MTLCommandBuffer> command=[backend->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];uint32_t size=static_cast<uint32_t>(n);
    [enc setComputePipelineState:backend->dot_stage];[enc setBuffer:xb offset:0 atIndex:0];[enc setBuffer:yb offset:0 atIndex:1];[enc setBuffer:current offset:0 atIndex:2];[enc setBytes:&size length:sizeof(size) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
    while(count>1) {
        size_t next_count=(count+255)/256;
        uint32_t nn=static_cast<uint32_t>(count);enc=[command computeCommandEncoder];
        [enc setComputePipelineState:backend->reduce_complex_stage];[enc setBuffer:current offset:0 atIndex:0];[enc setBuffer:next offset:0 atIndex:1];[enc setBytes:&nn length:sizeof(nn) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(next_count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
        std::swap(current,next);count=next_count;
    }
    if (complete_command(backend,command)!=METAL_BACKEND_OK) return METAL_BACKEND_RUNTIME_FAILED;
    const auto v=static_cast<const vector_float4 *>(current.contents)[0];
    const metal_complex64 value={split_to_double(v.x,v.y),split_to_double(v.z,v.w)};
    if(!std::isfinite(value.real) || !std::isfinite(value.imag)) return METAL_BACKEND_RUNTIME_FAILED;
    *result=value;return METAL_BACKEND_OK;
}

bool fft_shape(size_t nx,size_t ny,size_t nz,size_t nbatch,size_t stride,size_t &total) {
    if(!nx || !ny || !nz || !nbatch || nx>UINT32_MAX || ny>UINT32_MAX || nz>UINT32_MAX ||
       !metal_chunk::multiply(nx,ny,total) || !metal_chunk::multiply(total,nz,total))return false;
    return stride>=total && stride<=SIZE_MAX/nbatch/sizeof(metal_complex64);
}
int fft_plan(size_t nx,size_t ny,size_t nz,size_t nbatch,MixedReorderParams (&rp)[3]) {
    const size_t dims[3]={nx,ny,nz},total=nx*ny*nz;
    for(uint32_t axis=0;axis<3;++axis) {
        auto &p=rp[axis];p={};p.n=dims[axis];
        p.stride=axis==0 ? 1u:(axis==1 ? nx:nx*ny);p.lines=total/p.n;
        p.base_dim=axis==0 ? 1u:p.stride;p.axis=axis;p.batches=nbatch;
        uint32_t rest=p.n;
        for(uint32_t f=2;f<=13 && rest>1;++f)
            while(rest%f==0) {p.factors[p.count++]=f;rest/=f;}
        if(rest>1) return METAL_BACKEND_UNSUPPORTED;
    }
    return METAL_BACKEND_OK;
}
bool fft_storage(const metal_backend *backend,size_t bytes) {
    for(auto *slot:{&backend->fft_buffer0,&backend->fft_buffer1})
        if(!*slot || (*slot).length<bytes) {*slot=nil;*slot=backend_buffer(backend,bytes,MTLResourceStorageModeShared);}
    if(!backend->fft_roots) backend->fft_roots=[NSMutableDictionary new];
    return backend->fft_buffer0 && backend->fft_buffer1;
}
int encode_fft(const metal_backend *backend,id<MTLComputeCommandEncoder> encoder,
               const MixedReorderParams (&rp)[3],int direction,
               id<MTLBuffer> __strong &current,id<MTLBuffer> __strong &next) {
    for(uint32_t axis=0;axis<3;++axis) {
        const auto &r=rp[axis];if(r.n==1) continue;
        NSString *key=[NSString stringWithFormat:@"%u:%d",r.n,direction];
        id<MTLBuffer> roots=backend->fft_roots[key];
        if(!roots) {
            roots=backend_buffer(backend,r.n*sizeof(vector_float4),MTLResourceStorageModeShared);
            if(!roots) return METAL_BACKEND_ALLOCATION_FAILED;
            auto *values=static_cast<vector_float4 *>(roots.contents);
            for(uint32_t i=0;i<r.n;++i) {
                double theta=direction*2.0*std::acos(-1.0)*double(i)/double(r.n);
                auto re=split_double(std::cos(theta)),im=split_double(std::sin(theta));
                values[i]=(vector_float4){re.hi,re.lo,im.hi,im.lo};
            }
            backend->fft_roots[key]=roots;
        }
        size_t count=size_t(r.n)*r.lines*r.batches;
        const char *width=std::getenv("VASP_METAL_FFT_INDEX_WIDTH");
        // Current window's span and worst twiddle product must fit. Wide
        // dispatch remains available for large windows and controlled tests.
        const bool compact=(!width || std::strcmp(width,"wide")) && count<=UINT32_MAX &&
            r.stride<=UINT32_MAX && r.lines<=UINT32_MAX && r.base_dim<=UINT32_MAX &&
            r.batches<=UINT32_MAX && r.n<=UINT32_MAX/13 &&
            backend->mixed_reorder32 && backend->mixed_stage32;
        MixedReorderParams32 r32{};
        if(compact) {
            r32.n=r.n;r32.stride=uint32_t(r.stride);r32.lines=uint32_t(r.lines);
            r32.base_dim=uint32_t(r.base_dim);r32.axis=r.axis;r32.count=r.count;r32.batches=uint32_t(r.batches);
            std::copy(std::begin(r.factors),std::end(r.factors),std::begin(r32.factors));
            ++backend->fft_compact_axes;
        } else ++backend->fft_wide_axes;
        [encoder setComputePipelineState:compact ? backend->mixed_reorder32:backend->mixed_reorder];
        [encoder setBuffer:current offset:0 atIndex:0];[encoder setBuffer:next offset:0 atIndex:1];
        if(compact)[encoder setBytes:&r32 length:sizeof(r32) atIndex:2];
        else [encoder setBytes:&r length:sizeof(r) atIndex:2];
        auto dispatch=[&]() {
            if(compact)[encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            else [encoder dispatchThreads:MTLSizeMake(std::min<size_t>(count,65536),(count+65535)/65536,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        };
        dispatch();[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];std::swap(current,next);
        uint32_t prefix=1;
        for(uint32_t f=0;f<r.count;++f) {
            MixedStageParams p{r.n,r.axis,prefix,r.factors[f],r.stride,r.lines,r.base_dim,r.batches};
            MixedStageParams32 p32{r.n,uint32_t(r.stride),uint32_t(r.lines),uint32_t(r.base_dim),r.axis,prefix,r.factors[f],uint32_t(r.batches)};
            [encoder setComputePipelineState:compact ? backend->mixed_stage32:backend->mixed_stage];
            [encoder setBuffer:current offset:0 atIndex:0];[encoder setBuffer:next offset:0 atIndex:1];
            if(compact)[encoder setBytes:&p32 length:sizeof(p32) atIndex:2];
            else [encoder setBytes:&p length:sizeof(p) atIndex:2];
            [encoder setBuffer:roots offset:0 atIndex:3];dispatch();
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];std::swap(current,next);prefix*=r.factors[f];
        }
    }
    return METAL_BACKEND_OK;
}
int gpu_fft3d_window(const metal_backend *backend,size_t nx,size_t ny,size_t nz,size_t nbatch,size_t stride,
                  const metal_complex64 *input,metal_complex64 *output,int direction,bool normalize=true,size_t output_stride=0) {
    size_t total=0;
    if(!backend || !backend->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    if(!input || !output || (direction!=-1 && direction!=1) || !fft_shape(nx,ny,nz,nbatch,stride,total)) return METAL_BACKEND_INVALID_ARGUMENT;
    if(!output_stride)output_stride=stride;
    if(output_stride<total || output_stride>SIZE_MAX/nbatch/sizeof(metal_complex64))return METAL_BACKEND_INVALID_ARGUMENT;
    MixedReorderParams rp[3]{};int rc=fft_plan(nx,ny,nz,nbatch,rp);if(rc) return rc;
    metal_memory::OperatorGuard guard(backend->fft_mutex);
    if(!fft_storage(backend,total*nbatch*sizeof(vector_float4))) return METAL_BACKEND_ALLOCATION_FAILED;
    {
        ConversionTimer timer{backend};auto *host=static_cast<vector_float4 *>(backend->fft_buffer0.contents);
        for(size_t b=0;b<nbatch;++b) for(size_t i=0;i<total;++i) {
            auto r=split_double(input[b*stride+i].real),im=split_double(input[b*stride+i].imag);
            if(!std::isfinite(r.hi) || !std::isfinite(r.lo) || !std::isfinite(im.hi) || !std::isfinite(im.lo))return METAL_BACKEND_UNSUPPORTED;
            host[b*total+i]=(vector_float4){r.hi,r.lo,im.hi,im.lo};
        }
    }
    id<MTLBuffer> current=backend->fft_buffer0,next=backend->fft_buffer1;
    id<MTLCommandBuffer> command=[backend->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(!encoder) return METAL_BACKEND_RUNTIME_FAILED;
    rc=encode_fft(backend,encoder,rp,direction,current,next);[encoder endEncoding];if(rc) return rc;
    rc=complete_command(backend,command);if(rc) return rc;
    {
        ConversionTimer timer{backend};auto *out=static_cast<const vector_float4 *>(current.contents);
        for(size_t i=0;i<total*nbatch;++i)if(!std::isfinite(out[i].x) || !std::isfinite(out[i].y) || !std::isfinite(out[i].z) || !std::isfinite(out[i].w))return METAL_BACKEND_RUNTIME_FAILED;
        double scale=normalize && direction>0 ? 1.0/double(total):1.0;
        for(size_t b=0;b<nbatch;++b) for(size_t i=0;i<total;++i) {
            auto x=out[b*total+i];output[b*output_stride+i]={scale*split_to_double(x.x,x.y),scale*split_to_double(x.z,x.w)};
        }
    }
    return METAL_BACKEND_OK;
}
int gpu_fft3d_many(const metal_backend *backend,size_t nx,size_t ny,size_t nz,size_t nbatch,size_t stride,
                  const metal_complex64 *input,metal_complex64 *output,int direction,bool normalize=true) {
    if(!backend || !backend->use_gpu)return METAL_BACKEND_UNSUPPORTED;
    if(!input || !output || (direction!=-1 && direction!=1))return METAL_BACKEND_INVALID_ARGUMENT;
    size_t buffer_limit=backend->device.maxBufferLength;
    if(const char *value=std::getenv("VASP_METAL_FFT_CHUNK_BYTES")) {
        char *end=nullptr;auto parsed=std::strtoull(value,&end,10);
        if(end && *end=='\0' && parsed>0 && parsed<=SIZE_MAX)buffer_limit=std::min(buffer_limit,size_t(parsed));
    }
    metal_chunk::FFTPlan plan;
    auto result=metal_chunk::fft(nx,ny,nz,nbatch,stride,buffer_limit,metal_memory::budget(backend->device),plan);
    if(result==metal_chunk::Result::invalid_shape)return METAL_BACKEND_INVALID_ARGUMENT;
    if(result!=metal_chunk::Result::ok)return METAL_BACKEND_ALLOCATION_FAILED;
    // Retained caches participate in the same budget as active buffers. Trim
    // larger old buffers or old-sign/shape roots only under resource pressure.
    {metal_memory::OperatorGuard guard(backend->fft_mutex);
     size_t retained=plan.atomic_host_bytes;
     bool overflow=false;
     for(auto buffer:{backend->fft_buffer0,backend->fft_buffer1}) {
        size_t bytes=std::max<size_t>(buffer.length,plan.buffer_bytes);
        if(retained>SIZE_MAX-bytes){overflow=true;break;}retained+=bytes;
     }
     size_t roots=0;for(NSString *key in backend->fft_roots) {
        id<MTLBuffer> buffer=backend->fft_roots[key];
        if(roots>SIZE_MAX-buffer.length){overflow=true;break;}roots+=buffer.length;
     }
     const size_t dimensions[3]={nx,ny,nz};
     for(size_t i=0;i<3;++i)if(dimensions[i]>1) {
        bool seen=false;for(size_t j=0;j<i;++j)seen|=dimensions[j]==dimensions[i];
        NSString *key=[NSString stringWithFormat:@"%u:%d",(uint32_t)dimensions[i],direction];
        if(!seen && !backend->fft_roots[key]) {
            size_t bytes=dimensions[i]*sizeof(vector_float4);
            if(roots>SIZE_MAX-bytes){overflow=true;break;}roots+=bytes;
        }
     }
     size_t working=metal_memory::budget(backend->device) ? metal_memory::budget(backend->device):backend->device.maxBufferLength;
     if(overflow || retained>working || std::max<size_t>(roots,plan.root_bytes)>working-retained) {
        for(auto *slot:{&backend->fft_buffer0,&backend->fft_buffer1})if((*slot).length>plan.buffer_bytes)*slot=nil;
        [backend->fft_roots removeAllObjects];
     }}
    if(!plan.chunked)return gpu_fft3d_window(backend,nx,ny,nz,nbatch,stride,input,output,direction,normalize);
    ++backend->chunks.fft_calls;
    backend->chunks.fft_peak_workspace_bytes=std::max<uint64_t>(backend->chunks.fft_peak_workspace_bytes,plan.atomic_host_bytes+2*plan.buffer_bytes+plan.root_bytes);
    // Preflight the complete input before any command, then retain output until
    // every bounded dispatch succeeds. A fallback sees the original host data.
    for(size_t b=0;b<nbatch;++b)for(size_t i=0;i<plan.total;++i) {
        const auto v=input[b*stride+i];auto re=split_double(v.real),im=split_double(v.imag);
        if(!std::isfinite(re.hi) || !std::isfinite(re.lo) || !std::isfinite(im.hi) || !std::isfinite(im.lo))return METAL_BACKEND_UNSUPPORTED;
    }
    try {
        std::vector<metal_complex64> staged(plan.total*nbatch);
        // A formerly large cache cannot defeat the declared window budget.
        {metal_memory::OperatorGuard guard(backend->fft_mutex);
         for(auto *slot:{&backend->fft_buffer0,&backend->fft_buffer1})
             if(*slot && (*slot).length>plan.buffer_bytes)*slot=nil;}
        for(size_t first=0;first<nbatch;first+=plan.chunk_batches) {
            size_t count=std::min(plan.chunk_batches,nbatch-first);
            // Read the original padded input directly into reusable Metal
            // storage. Only the atomic result is staged on the host.
            int rc=gpu_fft3d_window(backend,nx,ny,nz,count,stride,input+first*stride,staged.data()+first*plan.total,direction,normalize,plan.total);
            ++backend->chunks.fft_windows;if(rc)return rc;
        }
        for(const auto v:staged)if(!std::isfinite(v.real) || !std::isfinite(v.imag))return METAL_BACKEND_RUNTIME_FAILED;
        for(size_t b=0;b<nbatch;++b)std::copy_n(staged.data()+b*plan.total,plan.total,output+b*stride);
        return METAL_BACKEND_OK;
    } catch(const std::bad_alloc &) {return METAL_BACKEND_ALLOCATION_FAILED;}
}

int gpu_fft3d(const metal_backend *backend,size_t nx,size_t ny,size_t nz,
              const metal_complex64 *input,metal_complex64 *output,int direction) {
    size_t total=0;
    if(!fft_shape(nx,ny,nz,1,SIZE_MAX/sizeof(metal_complex64),total)) return METAL_BACKEND_INVALID_ARGUMENT;
    return gpu_fft3d_many(backend,nx,ny,nz,1,total,input,output,direction);
}
int gpu_fft(const metal_backend *backend,size_t n,const metal_complex64 *input,metal_complex64 *output,int direction) {
    return gpu_fft3d(backend,n,1,1,input,output,direction);
}
struct HamParamsHost { uint32_t total,npl,bands,nions,npro,lmdim,ntypes,qld,qchannels,real_potential; };
struct HamSiteHost { uint32_t ion,type,base,count,l; };
int gpu_hamiltonian(const metal_backend *b,size_t nx,size_t ny,size_t nz,size_t npl,size_t bands,
 size_t nions,size_t npro,size_t lmdim,size_t ntypes,size_t qld,size_t qchannels,size_t phase_ld,
 int real_fields,int from_pw,const void *sv,const metal_complex64 *const *cr,const metal_complex64 *const *cw,
 const metal_complex64 *const *proj,const int *indices,const double *ke,const double *eig,const void *d,const void *q,
 const int *types,const int *lmmax,const int *lmbase,const double *radial,
 const metal_complex64 *cq,const metal_complex64 *phase,metal_complex64 *out,size_t ldout) {
    if(!b || !b->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    size_t total=0;
    if(!fft_shape(nx,ny,nz,bands,nx*ny*nz,total) || !npl || npl>total || !nions || !npro ||
       !lmdim || !ntypes || qchannels<lmdim || qld<npl || phase_ld<npl || ldout<npl ||
       !sv || !cr || !cw || !proj || !indices || !ke || !eig || !d || !q || !types || !lmmax ||
       !lmbase || !radial || !cq || !phase || !out || npro>UINT32_MAX/bands ||
       ntypes>UINT32_MAX/qchannels || nions>UINT32_MAX/lmdim/lmdim || nions>UINT32_MAX/npl)
       return METAL_BACKEND_INVALID_ARGUMENT;
    for(size_t band=0;band<bands;++band) if(!cr[band] || !cw[band] || !proj[band]) return METAL_BACKEND_INVALID_ARGUMENT;
    for(size_t i=0;i<npl;++i) if(indices[i]<1 || size_t(indices[i])>total) return METAL_BACKEND_INVALID_ARGUMENT;
    if(lmbase[0]!=0 || lmbase[nions]!=int(npro)) return METAL_BACKEND_UNSUPPORTED;
    for(size_t i=0;i<nions;++i) {
        if(types[i]<1 || size_t(types[i])>ntypes) return METAL_BACKEND_INVALID_ARGUMENT;
        int count=lmmax[types[i]-1];
        if(count<0 || size_t(count)>lmdim || lmbase[i]<0 || lmbase[i+1]!=lmbase[i]+count) return METAL_BACKEND_INVALID_ARGUMENT;
    }
    MixedReorderParams rp[3]{};int rc=fft_plan(nx,ny,nz,bands,rp);if(rc) return rc;
    metal_memory::OperatorGuard guard(b->fft_mutex);
    auto ensure=[&](id<MTLBuffer> __strong &buffer,size_t count,size_t width) {
        if(!count || count>SIZE_MAX/width || count*width>b->device.maxBufferLength) return false;
        if(!buffer || buffer.length<count*width) {
            b->ham_static_key=nullptr;b->ham_cached_epoch=0;
            buffer=nil;buffer=backend_buffer(b,count*width,MTLResourceStorageModeShared);
        }
        return buffer!=nil;
    };
    const size_t mat=lmdim*lmdim*nions,rad_count=npl*qchannels*ntypes;
    if(!fft_storage(b,total*bands*sizeof(vector_float4)) ||
       !ensure(b->ham_sv,total+1,sizeof(vector_float4)) || !ensure(b->ham_cw,npl*bands,sizeof(vector_float4)) ||
       !ensure(b->ham_proj,npro*bands,sizeof(vector_float4)) || !ensure(b->ham_d,mat,sizeof(vector_float4)) ||
       !ensure(b->ham_q,mat,sizeof(vector_float4)) || !ensure(b->ham_eig,bands,sizeof(vector_float2)) ||
       !ensure(b->ham_ke,npl,sizeof(vector_float2)) || !ensure(b->ham_index,npl,sizeof(uint32_t)) ||
       !ensure(b->ham_site,npro,sizeof(HamSiteHost)) || !ensure(b->ham_radial,rad_count,sizeof(vector_float2)) ||
       !ensure(b->ham_cq,qchannels*ntypes,sizeof(vector_float4)) || !ensure(b->ham_phase,npl*nions,sizeof(vector_float4)) ||
       !ensure(b->ham_onsite_buffer,npro*bands,sizeof(vector_float4)) || !ensure(b->ham_out,npl*bands,sizeof(vector_float4))) return METAL_BACKEND_ALLOCATION_FAILED;
    auto splitz=[](metal_complex64 z) {auto r=split_double(z.real),im=split_double(z.imag);return (vector_float4){r.hi,r.lo,im.hi,im.lo};};
    auto splitr=[](double z) {auto r=split_double(z);return (vector_float2){r.hi,r.lo};};
    const size_t dims[]={npl,nions,npro,lmdim,ntypes,qld,qchannels,phase_ld};
    bool cached=b->ham_static_key==radial && b->ham_cached_epoch==b->ham_epoch &&
        std::equal(std::begin(dims),std::end(dims),b->ham_static_dims);
    {
        ConversionTimer timer{b};
        auto *grid=static_cast<vector_float4 *>(b->fft_buffer0.contents);
        auto *wave=static_cast<vector_float4 *>(b->ham_cw.contents),*project=static_cast<vector_float4 *>(b->ham_proj.contents);
        auto *potential=static_cast<vector_float4 *>(b->ham_sv.contents);
        for(size_t band=0;band<bands;++band) {
            if(!from_pw) for(size_t i=0;i<total;++i) grid[band*total+i]=splitz(cr[band][i]);
            for(size_t i=0;i<npl;++i) wave[band*npl+i]=splitz(cw[band][i]);
            for(size_t i=0;i<npro;++i) project[band*npro+i]=splitz(proj[band][i]);
            static_cast<vector_float2 *>(b->ham_eig.contents)[band]=splitr(eig[band]);
        }
        for(size_t i=0;i<total;++i) potential[i]=splitz(real_fields ? metal_complex64{static_cast<const double *>(sv)[i],0}:static_cast<const metal_complex64 *>(sv)[i]);
        potential[total]=splitz({1.0/double(total),0});
        for(size_t i=0;i<mat;++i) {
            static_cast<vector_float4 *>(b->ham_d.contents)[i]=splitz(real_fields ? metal_complex64{static_cast<const double *>(d)[i],0}:static_cast<const metal_complex64 *>(d)[i]);
            static_cast<vector_float4 *>(b->ham_q.contents)[i]=splitz(real_fields ? metal_complex64{static_cast<const double *>(q)[i],0}:static_cast<const metal_complex64 *>(q)[i]);
        }
        for(size_t i=0;i<npl;++i) {
            static_cast<vector_float2 *>(b->ham_ke.contents)[i]=splitr(ke[i]);
            static_cast<uint32_t *>(b->ham_index.contents)[i]=indices[i]-1;
        }
        if(!cached) {
            auto *site=static_cast<HamSiteHost *>(b->ham_site.contents);
            for(size_t ion=0;ion<nions;++ion) for(int l=0;l<lmmax[types[ion]-1];++l)
                site[lmbase[ion]+l]={(uint32_t)ion,(uint32_t)(types[ion]-1),(uint32_t)lmbase[ion],(uint32_t)lmmax[types[ion]-1],(uint32_t)l};
            auto *rad=static_cast<vector_float2 *>(b->ham_radial.contents);
            for(size_t c=0;c<qchannels*ntypes;++c) for(size_t g=0;g<npl;++g) rad[g+npl*c]=splitr(radial[g+qld*c]);
            for(size_t i=0;i<qchannels*ntypes;++i) static_cast<vector_float4 *>(b->ham_cq.contents)[i]=splitz(cq[i]);
            for(size_t ion=0;ion<nions;++ion) for(size_t g=0;g<npl;++g)
                static_cast<vector_float4 *>(b->ham_phase.contents)[g+npl*ion]=splitz(phase[g+phase_ld*ion]);
            b->ham_static_key=radial;b->ham_cached_epoch=b->ham_epoch;
            std::copy(std::begin(dims),std::end(dims),b->ham_static_dims);++b->ham_static_uploads;
        } else ++b->ham_static_reuses;
    }
    HamParamsHost p{(uint32_t)total,(uint32_t)npl,(uint32_t)bands,(uint32_t)nions,(uint32_t)npro,
        (uint32_t)lmdim,(uint32_t)ntypes,(uint32_t)npl,(uint32_t)qchannels,(uint32_t)real_fields};
    id<MTLCommandBuffer> command=[b->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];if(!encoder) return METAL_BACKEND_RUNTIME_FAILED;
    id<MTLBuffer> current=b->fft_buffer0,next=b->fft_buffer1;
    if(from_pw) {
        // A blit initializes the sparse box, then all arithmetic stays resident.
        [encoder endEncoding];id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
        [blit fillBuffer:current range:NSMakeRange(0,total*bands*sizeof(vector_float4)) value:0];[blit endEncoding];
        encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:b->ham_scatter];[encoder setBuffer:current offset:0 atIndex:0];
        [encoder setBuffer:b->ham_cw offset:0 atIndex:1];[encoder setBuffer:b->ham_index offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(npl*bands,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        rc=encode_fft(b,encoder,rp,1,current,next);if(rc) {[encoder endEncoding];return rc;}
    }
    [encoder setComputePipelineState:b->ham_local];[encoder setBuffer:current offset:0 atIndex:0];
    [encoder setBuffer:b->ham_sv offset:0 atIndex:1];[encoder setBytes:&p length:sizeof(p) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(total*bands,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    rc=encode_fft(b,encoder,rp,-1,current,next);if(rc) {[encoder endEncoding];return rc;}
    [encoder setComputePipelineState:b->ham_onsite];
    id<MTLBuffer> buffers[]={b->ham_d,b->ham_q,b->ham_proj,b->ham_eig,b->ham_site,b->ham_onsite_buffer};
    for(int i=0;i<6;++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder setBytes:&p length:sizeof(p) atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(npro*bands,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:b->ham_finish];
    id<MTLBuffer> finish[]={current,b->ham_index,b->ham_cw,b->ham_ke,b->ham_eig,b->ham_onsite_buffer,b->ham_radial,b->ham_cq,b->ham_phase,b->ham_site,b->ham_out};
    for(int i=0;i<11;++i) [encoder setBuffer:finish[i] offset:0 atIndex:i];
    [encoder setBytes:&p length:sizeof(p) atIndex:11];
    [encoder dispatchThreads:MTLSizeMake(npl*bands,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [encoder endEncoding];rc=complete_command(b,command);if(rc) return rc;
    {ConversionTimer timer{b};auto *value=static_cast<const vector_float4 *>(b->ham_out.contents);
     for(size_t band=0;band<bands;++band) for(size_t i=0;i<npl;++i) {auto z=value[band*npl+i];out[band*ldout+i]={split_to_double(z.x,z.y),split_to_double(z.z,z.w)};}}
    return METAL_BACKEND_OK;
}
int gpu_orth_overlap(const metal_backend *b,size_t rows,size_t cols,size_t npl,size_t npro,
 const metal_complex64 *wa,const metal_complex64 *wb,size_t ldw,const metal_complex64 *pa,const metal_complex64 *pb,size_t ldp,
 metal_complex64 *out,size_t ldo) {
 if(!b || !b->use_gpu) return METAL_BACKEND_UNSUPPORTED;
 if(!rows || !cols || (!npl && !npro) || ldw<npl || ldp<npro || ldo<rows || !out ||
    (npl && (!wa || !wb)) || (npro && (!pa || !pb)) ||
    rows>UINT32_MAX || cols>UINT32_MAX || npl>UINT32_MAX || npro>UINT32_MAX) return METAL_BACKEND_INVALID_ARGUMENT;
 size_t span=0;
 for(auto shape:{std::pair<size_t,size_t>{ldw,rows},{ldw,cols},{ldp,rows},{ldp,cols},{ldo,cols}})
  if(!metal_chunk::multiply(shape.first,shape.second,span) || span>SIZE_MAX/sizeof(metal_complex64))return METAL_BACKEND_INVALID_ARGUMENT;
 size_t bytes[5]{},aggregate=0;
 const size_t dimensions[5][2]={{rows,npl},{cols,npl},{rows,npro},{cols,npro},{rows,cols}};
 for(size_t i=0;i<5;++i) {
  if(!metal_chunk::multiply(dimensions[i][0],dimensions[i][1],span) || !metal_chunk::multiply(std::max<size_t>(1,span),16,bytes[i]))return METAL_BACKEND_INVALID_ARGUMENT;
  if(bytes[i]>b->device.maxBufferLength || aggregate>SIZE_MAX-bytes[i])return METAL_BACKEND_ALLOCATION_FAILED;
  aggregate+=bytes[i];
 }
 const size_t working=metal_memory::budget(b->device) ? metal_memory::budget(b->device):b->device.maxBufferLength;
 if(aggregate>working)return METAL_BACKEND_ALLOCATION_FAILED;
 metal_memory::OperatorGuard guard(b->orth_mutex);
 size_t retained=0;for(size_t i=0;i<5;++i) {
  size_t length=std::max<size_t>(bytes[i],b->orth_buffers[i].length);
  if(retained>SIZE_MAX-length){retained=SIZE_MAX;break;}retained+=length;
 }
 for(size_t i=0;i<5;++i) {
  if(retained>working && b->orth_buffers[i].length>bytes[i])b->orth_buffers[i]=nil;
  if(!b->orth_buffers[i] || b->orth_buffers[i].length<bytes[i]) {b->orth_buffers[i]=nil;b->orth_buffers[i]=backend_buffer(b,bytes[i],MTLResourceStorageModeShared);}
  if(!b->orth_buffers[i])return METAL_BACKEND_ALLOCATION_FAILED;
 }
 auto upload=[](id<MTLBuffer> buffer,const metal_complex64 *x,size_t r,size_t c,size_t ld) {
  auto *dst=static_cast<vector_float4 *>(buffer.contents);
  for(size_t j=0;j<c;++j) for(size_t i=0;i<r;++i) {auto z=x[i+j*ld];auto re=split_double(z.real),im=split_double(z.imag);dst[i+j*r]=(vector_float4){re.hi,re.lo,im.hi,im.lo};}
 };
 {ConversionTimer timer{b};upload(b->orth_buffers[0],wa,npl,rows,ldw);upload(b->orth_buffers[1],wb,npl,cols,ldw);
  upload(b->orth_buffers[2],pa,npro,rows,ldp);upload(b->orth_buffers[3],pb,npro,cols,ldp);upload(b->orth_buffers[4],out,rows,cols,ldo);}
 struct Params {uint32_t rows,cols,npl,npro;} p{(uint32_t)rows,(uint32_t)cols,(uint32_t)npl,(uint32_t)npro};
 id<MTLCommandBuffer> command=[b->queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
 const bool grouped=metal_orth::cooperative(npl,npro,b->orth_overlap_group && b->orth_overlap_group.maxTotalThreadsPerThreadgroup>=256);
 [encoder setComputePipelineState:grouped ? b->orth_overlap_group:b->orth_overlap];for(size_t i=0;i<5;++i) [encoder setBuffer:b->orth_buffers[i] offset:0 atIndex:i];
 [encoder setBytes:&p length:sizeof(p) atIndex:5];
 if(grouped)[encoder dispatchThreadgroups:MTLSizeMake(rows,cols,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 else [encoder dispatchThreads:MTLSizeMake(rows,cols,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 [encoder endEncoding];int rc=complete_command(b,command);if(rc) return rc;
 if(grouped)++b->cooperative_orth_calls;
 {ConversionTimer timer{b};auto *x=static_cast<const vector_float4 *>(b->orth_buffers[4].contents);
  for(size_t j=0;j<cols;++j) for(size_t i=0;i<rows;++i) {auto v=x[i+j*rows];out[i+j*ldo]={split_to_double(v.x,v.y),split_to_double(v.z,v.w)};}}
 return METAL_BACKEND_OK;
}
int gpu_full_orth(const metal_backend *b,size_t npl,size_t npro,size_t bands,size_t nions,size_t lmdim,
 int real_fields,const void *metric,const int *counts,const int *bases,metal_complex64 *wave,size_t ldw,metal_complex64 *proj,size_t ldp) {
 if(!b || !b->use_gpu) return METAL_BACKEND_UNSUPPORTED;
 if(!npl || !bands || !wave || ldw<npl || ldp<npro || (npro && (!proj || !metric || !counts || !bases || !nions || !lmdim)) ||
    npl>UINT32_MAX || npro>UINT32_MAX || bands>UINT32_MAX) return METAL_BACKEND_INVALID_ARGUMENT;
 metal_full_orth::Plan storage;
 const auto planned=metal_full_orth::plan(npl,npro,bands,nions,lmdim,ldw,ldp,
     b->device.maxBufferLength,metal_memory::budget(b->device),storage);
 if(planned==metal_full_orth::Result::invalid_shape) return METAL_BACKEND_INVALID_ARGUMENT;
 if(planned!=metal_full_orth::Result::ok) return METAL_BACKEND_ALLOCATION_FAILED;
 if(npro) {if(bases[0]!=0 || bases[nions]!=int(npro)) return METAL_BACKEND_UNSUPPORTED;
  for(size_t i=0;i<nions;++i) if(counts[i]<0 || size_t(counts[i])>lmdim || bases[i]<0 || bases[i+1]!=bases[i]+counts[i]) return METAL_BACKEND_INVALID_ARGUMENT;}
 static_assert(sizeof(HamSiteHost)==20,"Full orth site allocation must match the Metal ABI");
 metal_memory::OperatorGuard guard(b->full_orth_mutex);
 size_t retained=0;
 for(size_t i=0;i<12;++i) {
  const size_t bytes=std::max<size_t>(storage.bytes[i],b->full_orth_buffers[i].length);
  if(retained>SIZE_MAX-bytes) return METAL_BACKEND_ALLOCATION_FAILED;
  retained+=bytes;
 }
 const size_t working=metal_memory::budget(b->device);
 if(retained>(working ? working:b->device.maxBufferLength))
  for(size_t i=0;i<12;++i)if(b->full_orth_buffers[i].length>storage.bytes[i])b->full_orth_buffers[i]=nil;
 for(size_t i=0;i<12;++i) {
  const size_t bytes=storage.bytes[i];auto &buffer=b->full_orth_buffers[i];
  if(!buffer || buffer.length<bytes) {buffer=nil;buffer=backend_buffer(b,bytes,MTLResourceStorageModeShared);}
  if(!buffer) return METAL_BACKEND_ALLOCATION_FAILED;
 }
 auto splitz=[](metal_complex64 z) {auto r=split_double(z.real),im=split_double(z.imag);return (vector_float4){r.hi,r.lo,im.hi,im.lo};};
 auto &buf=b->full_orth_buffers;
 {ConversionTimer timer{b};
  for(size_t j=0;j<bands;++j){for(size_t g=0;g<npl;++g) static_cast<vector_float4 *>(buf[0].contents)[g+j*npl]=splitz(wave[g+j*ldw]);
   for(size_t p=0;p<npro;++p) static_cast<vector_float4 *>(buf[1].contents)[p+j*npro]=splitz(proj[p+j*ldp]);}
  for(size_t i=0;i<(npro ? nions*lmdim*lmdim:0);++i) static_cast<vector_float4 *>(buf[3].contents)[i]=splitz(real_fields ? metal_complex64{static_cast<const double *>(metric)[i],0}:static_cast<const metal_complex64 *>(metric)[i]);
  for(size_t ion=0;ion<(npro ? nions:0);++ion) for(int l=0;l<counts[ion];++l)
   static_cast<HamSiteHost *>(buf[4].contents)[bases[ion]+l]={(uint32_t)ion,0,(uint32_t)bases[ion],(uint32_t)counts[ion],(uint32_t)l};
  std::memset(buf[5].contents,0,bands*8);std::memset(buf[8].contents,0,4);
 }
 struct Params {uint32_t npl,npro,bands;} p{(uint32_t)npl,(uint32_t)npro,(uint32_t)bands};
 uint32_t dim=bands;
 id<MTLCommandBuffer> command=[b->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
 if(npro) {
  struct MetricParams {uint32_t npro,bands,nions,lmdim;} mp{(uint32_t)npro,(uint32_t)bands,(uint32_t)nions,(uint32_t)lmdim};
  [enc setComputePipelineState:b->full_orth_metric];
  [enc setBuffer:buf[3] offset:0 atIndex:0];[enc setBuffer:buf[4] offset:0 atIndex:1];
  [enc setBuffer:buf[1] offset:0 atIndex:2];[enc setBuffer:buf[2] offset:0 atIndex:3];[enc setBytes:&mp length:sizeof(mp) atIndex:4];
  [enc dispatchThreads:MTLSizeMake(npro,bands,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
 }
 [enc setComputePipelineState:b->full_orth_gram];id<MTLBuffer> inputs[]={buf[0],buf[1],buf[2],buf[6]};
 for(int i=0;i<4;++i) [enc setBuffer:inputs[i] offset:0 atIndex:i];[enc setBytes:&p length:sizeof(p) atIndex:4];
 [enc dispatchThreads:MTLSizeMake(bands,bands,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
 for(uint32_t col=0;col<dim;++col) {
  [enc setComputePipelineState:b->chol_pivot];[enc setBuffer:buf[6] offset:0 atIndex:0];[enc setBuffer:buf[8] offset:0 atIndex:1];
  [enc setBytes:&dim length:sizeof(dim) atIndex:2];[enc setBytes:&col length:sizeof(col) atIndex:3];
  [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [enc setComputePipelineState:b->chol_column];[enc dispatchThreads:MTLSizeMake(bands,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
 }
 [enc setComputePipelineState:b->lower_inverse];[enc setBuffer:buf[6] offset:0 atIndex:0];[enc setBuffer:buf[7] offset:0 atIndex:1];
 [enc setBuffer:buf[8] offset:0 atIndex:2];[enc setBytes:&dim length:sizeof(dim) atIndex:3];
 [enc dispatchThreads:MTLSizeMake(bands,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
 [enc setComputePipelineState:b->full_orth_update];id<MTLBuffer> update[]={buf[0],buf[1],buf[7],buf[9],buf[10],buf[8]};
 for(int i=0;i<6;++i) [enc setBuffer:update[i] offset:0 atIndex:i];[enc setBytes:&p length:sizeof(p) atIndex:6];
 [enc dispatchThreads:MTLSizeMake(std::min<size_t>((npl+npro)*bands,65536),((npl+npro)*bands+65535)/65536,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
 int rc=complete_command(b,command);if(rc) return rc;if(static_cast<const uint32_t *>(buf[8].contents)[0]) return METAL_BACKEND_RUNTIME_FAILED;
 auto *w=static_cast<const vector_float4 *>(buf[9].contents),*pr=static_cast<const vector_float4 *>(buf[10].contents);
 // Reject nonfinite results before writing any caller-owned data.
 for(size_t i=0;i<npl*bands;++i) if(!std::isfinite(w[i].x) || !std::isfinite(w[i].y) || !std::isfinite(w[i].z) || !std::isfinite(w[i].w)) return METAL_BACKEND_RUNTIME_FAILED;
 for(size_t i=0;i<npro*bands;++i) if(!std::isfinite(pr[i].x) || !std::isfinite(pr[i].y) || !std::isfinite(pr[i].z) || !std::isfinite(pr[i].w)) return METAL_BACKEND_RUNTIME_FAILED;
 {ConversionTimer timer{b};for(size_t j=0;j<bands;++j) {
  for(size_t g=0;g<npl;++g) {auto z=w[g+j*npl];wave[g+j*ldw]={split_to_double(z.x,z.y),split_to_double(z.z,z.w)};}
  for(size_t i=0;i<npro;++i) {auto z=pr[i+j*npro];proj[i+j*ldp]={split_to_double(z.x,z.y),split_to_double(z.z,z.w)};}
 }}return METAL_BACKEND_OK;
}
int eigh_resource_check(const metal_backend *b,size_t n,size_t lda,size_t ldb,size_t ldv,bool generalized) {
    if(!b || !b->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    metal_eigh::Plan workspace;
    const auto rc=metal_eigh::plan(n,lda,ldb,ldv,generalized,b->device.maxBufferLength,
                                  metal_memory::budget(b->device),workspace);
    if(rc==metal_eigh::Result::invalid_shape) return METAL_BACKEND_INVALID_ARGUMENT;
    return rc==metal_eigh::Result::ok ? METAL_BACKEND_OK:METAL_BACKEND_ALLOCATION_FAILED;
}

struct LobStateHost { uint32_t n,locked,k,dim,status,iterations; float scale,residual; };
int gpu_lobpcg_buffer(const metal_backend *backend,size_t n,id<MTLBuffer> ab,
                     id<MTLBuffer> __strong &q_out,id<MTLBuffer> __strong &diagonal_out) {
    LobStateHost init{(uint32_t)n,0,0,0,1,0,1.0f,0};
    id<MTLBuffer> sb=backend_buffer(backend,n*6*sizeof(vector_float4),MTLResourceStorageModeShared);
    id<MTLBuffer> asb=backend_buffer(backend,n*6*sizeof(vector_float4),MTLResourceStorageModeShared);
    id<MTLBuffer> tmp=backend_buffer(backend,n*6*sizeof(vector_float4),MTLResourceStorageModeShared);
    id<MTLBuffer> vb=backend_buffer(backend,n*n*sizeof(vector_float4),MTLResourceStorageModeShared);
    id<MTLBuffer> wb=backend_buffer(backend,n*sizeof(vector_float2),MTLResourceStorageModeShared);
    id<MTLBuffer> state=[backend->device newBufferWithBytes:&init length:sizeof(init) options:MTLResourceStorageModeShared];
    if(!ab || !sb || !asb || !tmp || !vb || !wb || !state) return METAL_BACKEND_ALLOCATION_FAILED;
    size_t max_iterations=32*n+64;
    if(const char *limit=std::getenv("METAL_LOBPCG_MAX_ITERATIONS")) {
        char *end=nullptr;unsigned long parsed=std::strtoul(limit,&end,10);
        if(end && *end=='\0' && parsed>=16 && parsed<=8192) max_iterations=parsed;
    }
    for(size_t batch=0;batch<max_iterations;batch+=16) {
        id<MTLCommandBuffer> command=[backend->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
        if(batch==0) {
            [enc setComputePipelineState:backend->lob_init];
            [enc setBuffer:sb offset:0 atIndex:0];[enc setBuffer:vb offset:0 atIndex:1];[enc setBuffer:state offset:0 atIndex:2];[enc setBuffer:ab offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        for(int iteration=0;iteration<16;++iteration) {
            [enc setComputePipelineState:backend->lob_apply];
            [enc setBuffer:ab offset:0 atIndex:0];[enc setBuffer:sb offset:0 atIndex:1];[enc setBuffer:asb offset:0 atIndex:2];[enc setBuffer:state offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(n*6,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:backend->lob_update];
            [enc setBuffer:ab offset:0 atIndex:0];[enc setBuffer:sb offset:0 atIndex:1];[enc setBuffer:asb offset:0 atIndex:2];[enc setBuffer:vb offset:0 atIndex:3];
            [enc setBuffer:wb offset:0 atIndex:4];[enc setBuffer:tmp offset:0 atIndex:5];[enc setBuffer:state offset:0 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [enc endEncoding];if (complete_command(backend,command)!=METAL_BACKEND_OK) return METAL_BACKEND_RUNTIME_FAILED;
        auto st=static_cast<const LobStateHost *>(state.contents);
        backend->last_eigh_iterations=st->iterations;
        if(st->status==2) return METAL_BACKEND_RUNTIME_FAILED;
        if(st->status==0) {
            id<MTLBuffer> diagonal=backend_buffer(backend,n*n*sizeof(vector_float4),MTLResourceStorageModeShared);
            if(!diagonal) return METAL_BACKEND_ALLOCATION_FAILED;
            command=[backend->queue commandBuffer];enc=[command computeCommandEncoder];uint32_t dim=static_cast<uint32_t>(n);
            [enc setComputePipelineState:backend->lob_diagonal];[enc setBuffer:wb offset:0 atIndex:0];[enc setBuffer:diagonal offset:0 atIndex:1];[enc setBytes:&dim length:sizeof(dim) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc endEncoding];if (complete_command(backend,command)!=METAL_BACKEND_OK) return METAL_BACKEND_RUNTIME_FAILED;
            q_out=vb;diagonal_out=diagonal;
            return METAL_BACKEND_OK;
        }
    }
    return METAL_BACKEND_RUNTIME_FAILED;
}

int gpu_jacobi_buffer(const metal_backend *backend,size_t n,id<MTLBuffer> input,
                       id<MTLBuffer> __strong &q_out,id<MTLBuffer> __strong &diagonal_out) {
    const bool local=n<=40 && n*n*sizeof(vector_float4)+2048<=backend->device.maxThreadgroupMemoryLength;
    if(local) {
        id<MTLBuffer> q=backend_buffer(backend,n*n*sizeof(vector_float4),MTLResourceStorageModeShared);
        id<MTLBuffer> w=backend_buffer(backend,n*n*sizeof(vector_float4),MTLResourceStorageModeShared);
        uint32_t init[2]={1,0};id<MTLBuffer> state=[backend->device newBufferWithBytes:init length:sizeof(init) options:MTLResourceStorageModeShared];
        if(!q || !w || !state) return METAL_BACKEND_ALLOCATION_FAILED;
        uint32_t dim=static_cast<uint32_t>(n);id<MTLCommandBuffer> command=[backend->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
        [enc setComputePipelineState:backend->jacobi_local];[enc setBuffer:input offset:0 atIndex:0];[enc setBuffer:q offset:0 atIndex:1];
        [enc setBuffer:w offset:0 atIndex:2];[enc setBuffer:state offset:0 atIndex:3];[enc setBytes:&dim length:sizeof(dim) atIndex:4];
        [enc setThreadgroupMemoryLength:n*n*sizeof(vector_float4) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc endEncoding];
        if(complete_command(backend,command)!=METAL_BACKEND_OK) return METAL_BACKEND_RUNTIME_FAILED;
        if(command.status!=MTLCommandBufferStatusCompleted || static_cast<const uint32_t *>(state.contents)[0]!=0) return METAL_BACKEND_RUNTIME_FAILED;
        q_out=q;diagonal_out=w;return METAL_BACKEND_OK;
    }
    uint32_t dim=static_cast<uint32_t>(n),padded=dim+(dim%2);std::vector<vector_float4> qhost(n*n,(vector_float4){0,0,0,0});
    for(size_t i=0;i<n;++i) qhost[i+i*n]=(vector_float4){1,0,0,0};
    auto buffer=[&](){return backend_buffer(backend,n*n*sizeof(vector_float4),MTLResourceStorageModeShared);};
    id<MTLBuffer> hcur=input,hnext=buffer(),tmp=buffer();
    id<MTLBuffer> qcur=[backend->device newBufferWithBytes:qhost.data() length:n*n*sizeof(vector_float4) options:MTLResourceStorageModeShared],qnext=buffer();
    id<MTLBuffer> params=backend_buffer(backend,(padded/2)*sizeof(JacobiParamsHost),MTLResourceStorageModeShared);
    uint32_t init=1;id<MTLBuffer> state=[backend->device newBufferWithBytes:&init length:sizeof(init) options:MTLResourceStorageModeShared];
    if(!hnext || !tmp || !qcur || !qnext || !params || !state) return METAL_BACKEND_ALLOCATION_FAILED;
    struct Round {uint32_t n,padded,round;};
    for(uint32_t sweep=0;sweep<40;++sweep) {
        id<MTLCommandBuffer> command=[backend->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
        for(uint32_t round=0;round<padded-1;++round) {
            Round p{dim,padded,round};
            [enc setComputePipelineState:backend->cyclic_params];[enc setBuffer:hcur offset:0 atIndex:0];[enc setBuffer:params offset:0 atIndex:1];[enc setBytes:&p length:sizeof(p) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(padded/2,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            auto right=[&](id<MTLBuffer> in,id<MTLBuffer> out) {
                [enc setComputePipelineState:backend->cyclic_right];[enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];[enc setBuffer:params offset:0 atIndex:2];[enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(n,n,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            };
            right(hcur,tmp);right(qcur,qnext);[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:backend->cyclic_left];[enc setBuffer:tmp offset:0 atIndex:0];[enc setBuffer:hnext offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n,n,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            std::swap(hcur,hnext);std::swap(qcur,qnext);
        }
        [enc setComputePipelineState:backend->cyclic_check];[enc setBuffer:hcur offset:0 atIndex:0];[enc setBuffer:state offset:0 atIndex:1];[enc setBytes:&dim length:sizeof(dim) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];if (complete_command(backend,command)!=METAL_BACKEND_OK) return METAL_BACKEND_RUNTIME_FAILED;
        if(static_cast<const uint32_t *>(state.contents)[0]==0) {q_out=qcur;diagonal_out=hcur;return METAL_BACKEND_OK;}
    }
    return METAL_BACKEND_RUNTIME_FAILED;
}

int unpack_eigenpairs(size_t n,id<MTLBuffer> diagonal,id<MTLBuffer> vectors,double *w,metal_complex64 *v,size_t ldv) {
    const auto hf=static_cast<const vector_float4 *>(diagonal.contents),qf=static_cast<const vector_float4 *>(vectors.contents);
    std::vector<size_t> order(n);std::vector<double> values(n);
    for(size_t i=0;i<n;++i) {values[i]=split_to_double(hf[i+i*n].x,hf[i+i*n].y);if(!std::isfinite(values[i])) return METAL_BACKEND_RUNTIME_FAILED;order[i]=i;}
    std::sort(order.begin(),order.end(),[&](size_t i,size_t j){return values[i]<values[j];});
    for(size_t j=0;j<n;++j) {w[j]=values[order[j]];for(size_t i=0;i<n;++i) {auto x=qf[i+order[j]*n];auto value=metal_complex64{split_to_double(x.x,x.y),split_to_double(x.z,x.w)};if(!std::isfinite(value.real) || !std::isfinite(value.imag)) return METAL_BACKEND_RUNTIME_FAILED;v[i+j*ldv]=value;}}
    return METAL_BACKEND_OK;
}

int gpu_eigh_buffer(const metal_backend *backend,size_t n,id<MTLBuffer> input,
                    id<MTLBuffer> __strong &q,id<MTLBuffer> __strong &d) {
    backend->last_eigh_iterations=0;backend->last_eigh_solver=0;
    const char *solver=std::getenv("METAL_SUBSPACE_SOLVER");
    if(solver && std::strcmp(solver,"lobpcg")==0) {
        int rc=gpu_lobpcg_buffer(backend,n,input,q,d);
        backend->last_eigh_solver=rc==METAL_BACKEND_OK ? 1:2;
        if(rc==METAL_BACKEND_OK || std::getenv("METAL_LOBPCG_STRICT")) return rc;
    }
    return gpu_jacobi_buffer(backend,n,input,q,d);
}
int gpu_hermitian_eigh(const metal_backend *backend, size_t n, const metal_complex64 *a,
                       size_t lda, double *w, metal_complex64 *v, size_t ldv) {
    if(!a || !w || !v) return METAL_BACKEND_INVALID_ARGUMENT;
    const int resource=eigh_resource_check(backend,n,lda,0,ldv,false);
    if(resource!=METAL_BACKEND_OK) return resource;
    std::vector<vector_float4> host(n*n);
    for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) {
        auto re=split_double(a[i+j*lda].real),im=split_double(i==j ? 0.0:a[i+j*lda].imag);
        if(!std::isfinite(re.hi) || !std::isfinite(re.lo) || !std::isfinite(im.hi) || !std::isfinite(im.lo)) return METAL_BACKEND_UNSUPPORTED;
        host[i+j*n]=(vector_float4){re.hi,re.lo,im.hi,im.lo};
    }
    id<MTLBuffer> input=[backend->device newBufferWithBytes:host.data() length:host.size()*sizeof(vector_float4) options:MTLResourceStorageModeShared],q=nil,d=nil;
    if(!input) return METAL_BACKEND_ALLOCATION_FAILED;
    int rc=gpu_eigh_buffer(backend,n,input,q,d);
    if(rc!=METAL_BACKEND_OK) return rc;
    std::vector<double> values(n);std::vector<metal_complex64> vectors(n*n);
    rc=unpack_eigenpairs(n,d,q,values.data(),vectors.data(),n);if(rc!=METAL_BACKEND_OK) return rc;
    double scale=1.0;for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) scale=std::max(scale,std::hypot(a[i+j*lda].real,a[i+j*lda].imag));
    for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) {
        cd residual=0.;for(size_t k=0;k<n;++k) residual+=cd{a[i+k*lda].real,a[i+k*lda].imag}*cd{vectors[k+j*n].real,vectors[k+j*n].imag};
        residual-=values[j]*cd{vectors[i+j*n].real,vectors[i+j*n].imag};
        if(!std::isfinite(std::abs(residual)) || std::abs(residual)>2e-10*scale*n) return METAL_BACKEND_RUNTIME_FAILED;
    }
    std::copy(values.begin(),values.end(),w);for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) v[i+j*ldv]=vectors[i+j*n];
    return METAL_BACKEND_OK;
}

bool valid_triangle(char uplo) { return uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l'; }
bool wants_vectors(char jobz) { return jobz == 'V' || jobz == 'v'; }

std::vector<cd> hermitian_triangle(const metal_complex64 *src, size_t n, size_t ld, char uplo) {
    std::vector<cd> out(n * n, cd{0.0, 0.0});
    const bool upper = uplo == 'U' || uplo == 'u';
    for (size_t col = 0; col < n; ++col) for (size_t row = 0; row < n; ++row) {
        const bool stored = upper ? row <= col : row >= col;
        if (stored) {
            const cd x{src[row + col * ld].real, src[row + col * ld].imag};
            out[row + col * n] = x;
            if (row != col) out[col + row * n] = std::conj(x);
        }
    }
    return out;
}

int gpu_generalized_eigh(const metal_backend *backend, int itype, char uplo, char jobz,
                         size_t n, const metal_complex64 *a, size_t lda,
                         const metal_complex64 *b, size_t ldb, double *w,
                         metal_complex64 *v, size_t ldv,metal_complex64 *factor=nullptr,size_t ldf=0) {
    if(backend) backend->generalized_all_gpu=false;
    if (!backend || !backend->use_gpu || itype != 1 || !valid_triangle(uplo) ||
        (jobz != 'N' && jobz != 'n' && !wants_vectors(jobz)) || n == 0 ||
        !a || !b || !w || lda < n || ldb < n || (wants_vectors(jobz) && (!v || ldv < n)) || (factor && (ldf<n || ldf>SIZE_MAX/n/sizeof(metal_complex64)))) return METAL_BACKEND_UNSUPPORTED;
    const int resource=eigh_resource_check(backend,n,lda,ldb,wants_vectors(jobz) ? ldv:n,true);
    if(resource!=METAL_BACKEND_OK) return resource;
    const bool upper=uplo=='U' || uplo=='u';
    const char *solver=std::getenv("METAL_SUBSPACE_SOLVER");
    const bool local=(!solver || std::strcmp(solver,"lobpcg")!=0) && n<=40 &&
        n*n*sizeof(vector_float4)+2048<=backend->device.maxThreadgroupMemoryLength;
    metal_memory::OperatorGuard eig_guard(backend->small_eigh_mutex);
    // Release stale capacities before growing. Status slots need bytes, not
    // whole matrices; global Jacobi produces its own Q and diagonal buffers.
    for(size_t i=0;i<11;++i) {
        auto &slot=backend->small_eigh[i];
        if(!local && (i==7 || i==8)) {slot=nil;continue;}
        const size_t bytes=i==6 ? sizeof(uint32_t):i==10 ? 2*sizeof(uint32_t):n*n*sizeof(vector_float4);
        if(!slot || slot.length!=bytes) {
            slot=nil;slot=backend_buffer(backend,bytes,MTLResourceStorageModeShared);
        }
        if(!slot) return METAL_BACKEND_ALLOCATION_FAILED;
    }
    id<MTLBuffer> ab=backend->small_eigh[0],lb=backend->small_eigh[1],inv=backend->small_eigh[2],tmp=backend->small_eigh[3],raw=backend->small_eigh[4],c=backend->small_eigh[5];
    id<MTLBuffer> state=backend->small_eigh[6],q=backend->small_eigh[7],d=backend->small_eigh[8],vectors=backend->small_eigh[9],eig_state=backend->small_eigh[10];
    uint32_t dim=static_cast<uint32_t>(n);std::memset(state.contents,0,sizeof(uint32_t));std::memset(eig_state.contents,0,2*sizeof(uint32_t));
    {ConversionTimer timer{backend};auto *ah=static_cast<vector_float4 *>(ab.contents),*bh=static_cast<vector_float4 *>(lb.contents);
    for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) {
        bool stored=upper ? i<=j:i>=j;
        auto x=stored ? a[i+j*lda]:a[j+i*lda];auto y=stored ? b[i+j*ldb]:b[j+i*ldb];
        if(!stored) {x.imag=-x.imag;y.imag=-y.imag;}if(i==j) {x.imag=0;y.imag=0;}
        auto xr=split_double(x.real),xi=split_double(x.imag),yr=split_double(y.real),yi=split_double(y.imag);
        if(!std::isfinite(xr.hi) || !std::isfinite(xr.lo) || !std::isfinite(xi.hi) || !std::isfinite(xi.lo) || !std::isfinite(yr.hi) || !std::isfinite(yr.lo) || !std::isfinite(yi.hi) || !std::isfinite(yi.lo)) return METAL_BACKEND_UNSUPPORTED;
        ah[i+j*n]=(vector_float4){xr.hi,xr.lo,xi.hi,xi.lo};bh[i+j*n]=(vector_float4){yr.hi,yr.lo,yi.hi,yi.lo};
    }
    }
    id<MTLCommandBuffer> command=[backend->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
    for(uint32_t col=0;col<dim;++col) {
        [enc setComputePipelineState:backend->chol_pivot];[enc setBuffer:lb offset:0 atIndex:0];[enc setBuffer:state offset:0 atIndex:1];
        [enc setBytes:&dim length:sizeof(dim) atIndex:2];[enc setBytes:&col length:sizeof(col) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:backend->chol_column];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [enc setComputePipelineState:backend->lower_inverse];[enc setBuffer:lb offset:0 atIndex:0];[enc setBuffer:inv offset:0 atIndex:1];
    [enc setBuffer:state offset:0 atIndex:2];[enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    auto product=[&](id<MTLComputeCommandEncoder> encoder,id<MTLBuffer> left,id<MTLBuffer> right,id<MTLBuffer> result,uint32_t mode) {
        [encoder setComputePipelineState:backend->subspace_product];[encoder setBuffer:left offset:0 atIndex:0];[encoder setBuffer:right offset:0 atIndex:1];[encoder setBuffer:result offset:0 atIndex:2];
        [encoder setBuffer:state offset:0 atIndex:3];[encoder setBytes:&dim length:sizeof(dim) atIndex:4];[encoder setBytes:&mode length:sizeof(mode) atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(n,n,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    product(enc,inv,ab,tmp,0);product(enc,tmp,inv,raw,1);
    [enc setComputePipelineState:backend->hermitianize];[enc setBuffer:raw offset:0 atIndex:0];[enc setBuffer:c offset:0 atIndex:1];[enc setBytes:&dim length:sizeof(dim) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n,n,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    if(local) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:backend->jacobi_local];[enc setBuffer:c offset:0 atIndex:0];[enc setBuffer:q offset:0 atIndex:1];
        [enc setBuffer:d offset:0 atIndex:2];[enc setBuffer:eig_state offset:0 atIndex:3];[enc setBytes:&dim length:sizeof(dim) atIndex:4];
        [enc setThreadgroupMemoryLength:n*n*sizeof(vector_float4) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];product(enc,inv,q,vectors,2);[enc endEncoding];
        int rc=complete_command(backend,command);if(rc) return rc;
        if(static_cast<const uint32_t *>(state.contents)[0] || static_cast<const uint32_t *>(eig_state.contents)[0]) return METAL_BACKEND_RUNTIME_FAILED;
        backend->last_eigh_solver=0;
    } else {
        [enc endEncoding];int rc=complete_command(backend,command);if(rc) return rc;
        if(static_cast<const uint32_t *>(state.contents)[0]) return METAL_BACKEND_RUNTIME_FAILED;
        q=nil;d=nil;
        int solve_rc=gpu_eigh_buffer(backend,n,c,q,d);if(solve_rc) return solve_rc;
        command=[backend->queue commandBuffer];enc=[command computeCommandEncoder];product(enc,inv,q,vectors,2);[enc endEncoding];
        rc=complete_command(backend,command);if(rc) return rc;
    }
    std::vector<metal_complex64> vv(n*n);std::vector<double> ww(n);
    int rc=unpack_eigenpairs(n,d,vectors,ww.data(),vv.data(),n);if(rc!=METAL_BACKEND_OK) return rc;
    // Reject unsafe results before caller-owned outputs change. Float64 here
    // is independent verification of the GPU result, not the solution path.
    const auto af=hermitian_triangle(a,n,lda,uplo),bf=hermitian_triangle(b,n,ldb,uplo);
    double scale=1.0;for(auto x:af) scale=std::max(scale,std::abs(x));
    for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) {
        cd ax=0,bx=0;for(size_t k=0;k<n;++k) {cd z{vv[k+j*n].real,vv[k+j*n].imag};ax+=af[i+k*n]*z;bx+=bf[i+k*n]*z;}
        if(!std::isfinite(std::abs(ax-ww[j]*bx)) || std::abs(ax-ww[j]*bx)>2e-10*scale*n) return METAL_BACKEND_RUNTIME_FAILED;
    }
    std::copy(ww.begin(),ww.end(),w);
    if(wants_vectors(jobz)) for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) v[i+j*ldv]=vv[i+j*n];
    if(factor) {
        const auto lf=static_cast<const vector_float4 *>(lb.contents);
        for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) if(upper ? i<=j:i>=j) {
            auto x=upper ? lf[j+i*n]:lf[i+j*n];factor[i+j*ldf]={split_to_double(x.x,x.y),(upper ? -1:1)*split_to_double(x.z,x.w)};
        }
    }
    backend->generalized_all_gpu=true;
    return METAL_BACKEND_OK;
}

} // namespace

extern "C" int metal_backend_eigh_resources(const metal_backend *b,size_t n,size_t lda,size_t ldb,size_t ldv,int generalized) {
    return eigh_resource_check(b,n,lda,ldb,ldv,generalized!=0);
}

extern "C" int metal_backend_create(metal_backend **out, int prefer_gpu) {
    if (!out) return METAL_BACKEND_INVALID_ARGUMENT;
    auto *backend = new (std::nothrow) metal_backend;
    if (!backend) return METAL_BACKEND_ALLOCATION_FAILED;
    const char *command_api=std::getenv("VASP_METAL_COMMAND_API");
    backend->metal4_command_requested=command_api && std::strcmp(command_api,"4")==0;
    if(backend->metal4_command_requested) backend->metal4_command_reason="GPU backend unavailable";
    const char *library_path=std::getenv("VASP_METAL_LIBRARY");
    backend->precompiled_requested=library_path && *library_path;
    if(backend->precompiled_requested) {backend->precompiled_path=library_path;backend->precompiled_reason="GPU backend unavailable";}
    if (prefer_gpu) compile_gpu(backend);
    *out = backend;
    return METAL_BACKEND_OK;
}

extern "C" void metal_backend_destroy(metal_backend *backend) { delete backend; }
extern "C" int metal_backend_gpu_enabled(const metal_backend *backend) { return backend && backend->use_gpu ? 1 : 0; }
extern "C" const char *metal_backend_variant(const metal_backend *backend) { return backend ? backend->variant.c_str() : "cpu-unavailable"; }
extern "C" int metal_backend_metal4_enabled(const metal_backend *backend) { return backend && backend->metal4_family ? 1 : 0; }
extern "C" int metal_backend_metal4_command_requested(const metal_backend *b) {return b && b->metal4_command_requested;}
extern "C" int metal_backend_metal4_command_available(const metal_backend *b) {return b && b->metal4_commands;}
extern "C" uint64_t metal_backend_metal4_command_calls(const metal_backend *b) {return b ? b->metal4_command_calls:0;}
extern "C" uint64_t metal_backend_metal4_command_success(const metal_backend *b) {return b ? b->metal4_command_success:0;}
extern "C" uint64_t metal_backend_metal4_command_fallbacks(const metal_backend *b) {return b ? b->metal4_command_fallbacks:0;}
extern "C" const char *metal_backend_metal4_command_reason(const metal_backend *b) {return b ? b->metal4_command_reason.c_str():"backend-unavailable";}
extern "C" int metal_backend_precompiled_requested(const metal_backend *b) {return b && b->precompiled_requested;}
extern "C" int metal_backend_precompiled_verified(const metal_backend *b) {return b && b->precompiled_verified;}
extern "C" int metal_backend_precompiled_used(const metal_backend *b) {return b && b->precompiled_used;}
extern "C" unsigned metal_backend_precompiled_fallbacks(const metal_backend *b) {return b ? b->precompiled_fallbacks:0;}
extern "C" const char *metal_backend_precompiled_reason(const metal_backend *b) {return b ? b->precompiled_reason.c_str():"backend-unavailable";}
extern "C" const char *metal_backend_precompiled_path(const metal_backend *b) {return b ? b->precompiled_path.c_str():"";}
extern "C" int metal_backend_ml_api_available(const metal_backend *backend) { return backend && backend->ml_api_available ? 1 : 0; }
extern "C" int metal_backend_device_tuned(const metal_backend *backend) { return backend && backend->device_tuned ? 1 : 0; }
extern "C" const char *metal_backend_language_version(const metal_backend *b) {return b ? b->language.c_str():"uncompiled";}
extern "C" unsigned metal_backend_compile_fallbacks(const metal_backend *b) {return b ? b->compile_fallbacks:0;}
extern "C" int metal_backend_tiled_gemm_enabled(const metal_backend *b) {return b && b->tiled_enabled;}
extern "C" uint64_t metal_backend_tiled_gemm_calls(const metal_backend *b) {return b ? b->tiled_calls:0;}
extern "C" uint64_t metal_backend_tiled_gemm_fallbacks(const metal_backend *b) {return b ? b->tiled_fallbacks:0;}
extern "C" int metal_backend_tiled_gemm_autoselected(const metal_backend *b) {return b && b->tiled_auto_selected;}
extern "C" int metal_backend_gemm_calibrated(const metal_backend *b) {return b && b->gemm_calibrated;}
extern "C" int metal_backend_last_gpu_success(const metal_backend *backend) { return backend && backend->last_gpu_success ? 1 : 0; }
extern "C" const char *metal_backend_last_error(const metal_backend *backend) { return backend ? backend->error.c_str() : "invalid backend"; }
extern "C" int metal_backend_last_eigh_solver(const metal_backend *backend) { return backend ? backend->last_eigh_solver:0; }
extern "C" void metal_backend_hamiltonian_invalidate(metal_backend *b) {
 if(b) {metal_memory::OperatorGuard lock(b->fft_mutex);++b->ham_epoch;}
}
extern "C" uint64_t metal_backend_hamiltonian_static_uploads(const metal_backend *b) {return b ? b->ham_static_uploads:0;}
extern "C" uint64_t metal_backend_hamiltonian_static_reuses(const metal_backend *b) {return b ? b->ham_static_reuses:0;}
extern "C" int metal_backend_hamiltonian_gpu(const metal_backend *b,size_t nx,size_t ny,size_t nz,size_t npl,size_t bands,
 size_t nions,size_t npro,size_t lmdim,size_t ntypes,size_t qld,size_t qchannels,size_t phase_ld,
 int real_fields,int from_pw,const void *sv,const metal_complex64 *const *cr,const metal_complex64 *const *cw,
 const metal_complex64 *const *proj,const int *indices,const double *ke,const double *eig,const void *d,const void *q,
 const int *types,const int *lmmax,const int *lmbase,const double *radial,
 const metal_complex64 *cq,const metal_complex64 *phase,metal_complex64 *out,size_t ldout) {
 if(b) b->last_gpu_success=false;
 int rc=gpu_hamiltonian(b,nx,ny,nz,npl,bands,nions,npro,lmdim,ntypes,qld,qchannels,phase_ld,real_fields,from_pw,sv,cr,cw,proj,indices,ke,eig,d,q,types,lmmax,lmbase,radial,cq,phase,out,ldout);
 if(b && rc==METAL_BACKEND_OK) b->last_gpu_success=true;return rc;
}
extern "C" int metal_backend_orth_overlap_gpu(const metal_backend *b,size_t rows,size_t cols,size_t npl,size_t npro,
 const metal_complex64 *wa,const metal_complex64 *wb,size_t ldw,const metal_complex64 *pa,const metal_complex64 *pb,size_t ldp,
 metal_complex64 *out,size_t ldo) {
 if(b) b->last_gpu_success=false;
 int rc=gpu_orth_overlap(b,rows,cols,npl,npro,wa,wb,ldw,pa,pb,ldp,out,ldo);
 if(b && !rc) b->last_gpu_success=true;return rc;
}
extern "C" int metal_backend_full_orth_gpu(const metal_backend *b,size_t npl,size_t npro,size_t bands,size_t nions,size_t lmdim,
 int real_fields,const void *metric,const int *counts,const int *bases,metal_complex64 *wave,size_t ldw,metal_complex64 *proj,size_t ldp) {
 if(b) b->last_gpu_success=false;int rc=gpu_full_orth(b,npl,npro,bands,nions,lmdim,real_fields,metric,counts,bases,wave,ldw,proj,ldp);
 if(b && !rc) b->last_gpu_success=true;return rc;
}
extern "C" uint64_t metal_backend_command_count(const metal_backend *b) {return b ? b->command_count:0;}
extern "C" uint64_t metal_backend_wait_count(const metal_backend *b) {return b ? b->wait_count:0;}
extern "C" double metal_backend_wait_seconds(const metal_backend *b) {return b ? b->wait_seconds:0;}
extern "C" double metal_backend_conversion_seconds(const metal_backend *b) {return b ? b->conversion_seconds:0;}
extern "C" double metal_backend_gpu_seconds(const metal_backend *backend) { return backend ? backend->gpu_seconds:0.0; }
extern "C" uint64_t metal_backend_last_eigh_iterations(const metal_backend *backend) { return backend ? backend->last_eigh_iterations:0; }
extern "C" int metal_backend_try_gpu(const metal_backend *backend) { return backend && backend->use_gpu ? 1 : 0; }

extern "C" int metal_backend_copy_metal_resources(const metal_backend *b,void **device,void **queue,void **library) {
    if(!device || !queue || !library) return METAL_BACKEND_INVALID_ARGUMENT;
    *device=nullptr; *queue=nullptr; *library=nullptr;
    if(!b || !b->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    *device=(__bridge void *)b->device; *queue=(__bridge void *)b->queue; *library=(__bridge void *)b->library;
    return METAL_BACKEND_OK;
}
extern "C" int metal_backend_complete_external_command(const metal_backend *b,void *command) {
    if(!b || !b->use_gpu || !command) return METAL_BACKEND_INVALID_ARGUMENT;
    b->last_gpu_success=false;
    int rc=complete_command(b,(__bridge id<MTLCommandBuffer>)command);
    if(!rc) b->last_gpu_success=true;
    return rc;
}
extern "C" void metal_backend_record_external_conversion(const metal_backend *b,double seconds) {
    if(b && std::isfinite(seconds) && seconds>=0) b->conversion_seconds+=seconds;
}
extern "C" void metal_backend_set_external_success(const metal_backend *b,int success) {if(b) b->last_gpu_success=success!=0;}
extern "C" uint64_t metal_backend_paw_epoch(const metal_backend *b) {
    if(!b) return 0;
    metal_memory::OperatorGuard lock(b->fft_mutex); return b->ham_epoch;
}

namespace {
int resident_require_open(const metal_resident *r) {
    return r && r->backend && r->backend->use_gpu && r->open ? METAL_BACKEND_OK : METAL_BACKEND_INVALID_ARGUMENT;
}
bool resident_size_ok(size_t count, size_t width) {
    return width != 0 && count <= SIZE_MAX / width;
}
int resident_wait_last(metal_resident *r) {
    if (!r || !r->last_command) return METAL_BACKEND_OK;
    if (r->last_command.status != MTLCommandBufferStatusCompleted && r->last_command.status != MTLCommandBufferStatusError) {
        auto start=SteadyClock::now(); ++r->waits; ++r->backend->wait_count;
        [r->last_command waitUntilCompleted];
        r->backend->wait_seconds+=std::chrono::duration<double>(SteadyClock::now()-start).count();
    }
    if (!r->last_accounted) {
        if (r->last_command.GPUEndTime>r->last_command.GPUStartTime)
            r->backend->gpu_seconds+=r->last_command.GPUEndTime-r->last_command.GPUStartTime;
        r->last_accounted=true;
    }
    if (r->last_command.status!=MTLCommandBufferStatusCompleted ||
        *static_cast<const uint32_t *>(r->result_status.contents)!=0u) {
        r->backend->last_gpu_success=false;
        r->wave_valid=false; r->projection_valid=false;
        r->last_command=nil;
        return METAL_BACKEND_RUNTIME_FAILED;
    }
    r->backend->last_gpu_success=true;
    return METAL_BACKEND_OK;
}
int resident_buffer(id<MTLDevice> device, id<MTLBuffer> __strong &buffer, size_t count, size_t width) {
    if (!resident_size_ok(count,width) || count*width > device.maxBufferLength) return METAL_BACKEND_INVALID_ARGUMENT;
    if (!buffer || buffer.length < count*width)
        buffer=metal_memory::allocate(device,std::max<size_t>(1,count*width),MTLResourceStorageModeShared);
    return buffer ? METAL_BACKEND_OK : METAL_BACKEND_ALLOCATION_FAILED;
}
void resident_upload_complex(id<MTLBuffer> buffer, const metal_complex64 *src, size_t rows, size_t cols, size_t ld) {
    auto *dst=static_cast<vector_float4 *>(buffer.contents);
    for(size_t col=0;col<cols;++col) for(size_t row=0;row<rows;++row) {
        auto re=split_double(src[row+col*ld].real), im=split_double(src[row+col*ld].imag);
        dst[row+col*rows]=(vector_float4){re.hi,re.lo,im.hi,im.lo};
    }
}
}

extern "C" int metal_backend_resident_create(const metal_backend *backend,
                                               size_t nx, size_t ny, size_t nz,
                                               size_t bands, size_t npro,
                                               metal_resident **out) {
    if (!out) return METAL_BACKEND_INVALID_ARGUMENT;
    *out=nullptr;
    if (!backend || !nx || !ny || !nz || !bands ||
        nx > UINT32_MAX || ny > UINT32_MAX || nz > UINT32_MAX || bands > UINT32_MAX || npro > UINT32_MAX)
        return METAL_BACKEND_INVALID_ARGUMENT;
    if (!backend->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    if (ny > SIZE_MAX/nx || nz > SIZE_MAX/(nx*ny)) return METAL_BACKEND_INVALID_ARGUMENT;
    const size_t total=nx*ny*nz;
    // All shader indices and products use uint, including projection offsets.
    if (total > UINT32_MAX || bands > UINT32_MAX/total || npro > UINT32_MAX/total ||
        (npro && bands > UINT32_MAX/npro)) return METAL_BACKEND_INVALID_ARGUMENT;
    auto *r=new (std::nothrow) metal_resident;
    if (!r) return METAL_BACKEND_ALLOCATION_FAILED;
    r->backend=backend; r->nx=nx; r->ny=ny; r->nz=nz; r->total=total; r->bands=bands; r->npro=npro;
    const int rc0=resident_buffer(backend->device,r->wave0,total*bands,sizeof(vector_float4));
    const int rc1=resident_buffer(backend->device,r->wave1,total*bands,sizeof(vector_float4));
    const int rc2=resident_buffer(backend->device,r->result_status,1,sizeof(uint32_t));
    if (rc0 || rc1 || rc2) { delete r; return rc0 ? rc0 : (rc1 ? rc1 : rc2); }
    r->current=r->wave0; r->next=r->wave1; *out=r; return METAL_BACKEND_OK;
}

extern "C" void metal_backend_resident_destroy(metal_resident *resident) {
    if (!resident) return;
    {
        metal_memory::OperatorGuard lock(resident->mutex);
        if (resident->encoder) [resident->encoder endEncoding];
        resident->encoder=nil; resident->command=nil;
        resident_wait_last(resident);
    }
    delete resident;
}

extern "C" int metal_backend_resident_begin_step(metal_resident *r) {
    if (!r || !r->backend || !r->backend->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    metal_memory::OperatorGuard lock(r->mutex);
    if (r->open) return METAL_BACKEND_INVALID_ARGUMENT;
    int rc=resident_wait_last(r); if (rc) return rc;
    r->backend->last_gpu_success=false;
    r->command=[r->backend->queue commandBuffer];
    r->encoder=[r->command computeCommandEncoder];
    if (!r->command || !r->encoder) { r->command=nil; r->encoder=nil; return METAL_BACKEND_RUNTIME_FAILED; }
    *static_cast<uint32_t *>(r->result_status.contents)=0u;
    r->open=true; r->encoded=false; return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_abort_step(metal_resident *r) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r)) return METAL_BACKEND_INVALID_ARGUMENT;
    [r->encoder endEncoding]; r->encoder=nil; r->command=nil; r->open=false;
    r->backend->last_gpu_success=false;
    // Ping-pong pointers may describe unsubmitted output. Require a fresh wave.
    r->wave_valid=false; r->projection_valid=false;
    r->current=r->wave0; r->next=r->wave1;
    return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_upload_wavefunction(metal_resident *r,
                                                            const metal_complex64 *wave,
                                                            size_t ld, int force) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r) || !wave || ld<r->total || ld>SIZE_MAX/r->bands/sizeof(*wave)) return METAL_BACKEND_INVALID_ARGUMENT;
    if (!force && r->wave_valid) return METAL_BACKEND_OK;
    if (r->encoded) return METAL_BACKEND_INVALID_ARGUMENT;
    { ConversionTimer timer{r->backend}; resident_upload_complex(r->wave0,wave,r->total,r->bands,ld); }
    r->current=r->wave0; r->next=r->wave1; r->wave_valid=true; r->projection_valid=false; ++r->uploads; ++r->generation;
    return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_upload_potential(metal_resident *r,
                                                         const metal_complex64 *potential,
                                                         int force) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r) || !potential) return METAL_BACKEND_INVALID_ARGUMENT;
    if (!force && r->potential_valid) return METAL_BACKEND_OK;
    if (r->encoded) return METAL_BACKEND_INVALID_ARGUMENT;
    int rc=resident_buffer(r->backend->device,r->potential,r->total,sizeof(vector_float4)); if (rc) return rc;
    { ConversionTimer timer{r->backend}; resident_upload_complex(r->potential,potential,r->total,1,r->total); }
    r->potential_valid=true; ++r->uploads; ++r->generation; return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_upload_projectors(metal_resident *r,
                                                           const metal_complex64 *projectors,
                                                           size_t ld, int force) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r) || !r->npro || !projectors || ld<r->total || ld>SIZE_MAX/r->npro/sizeof(*projectors)) return METAL_BACKEND_INVALID_ARGUMENT;
    if (!force && r->projectors_valid) return METAL_BACKEND_OK;
    if (r->encoded) return METAL_BACKEND_INVALID_ARGUMENT;
    int rc=resident_buffer(r->backend->device,r->projectors,r->total*r->npro,sizeof(vector_float4)); if (rc) return rc;
    rc=resident_buffer(r->backend->device,r->projection,r->bands*r->npro,sizeof(vector_float4)); if (rc) return rc;
    { ConversionTimer timer{r->backend}; resident_upload_complex(r->projectors,projectors,r->total,r->npro,ld); }
    r->projectors_valid=true; r->projection_valid=false; ++r->uploads; ++r->generation; return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_fft(metal_resident *r, int direction) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r) || (direction!=-1 && direction!=1) || !r->wave_valid) return METAL_BACKEND_INVALID_ARGUMENT;
    MixedReorderParams plan[3]{}; int rc=fft_plan(r->nx,r->ny,r->nz,r->bands,plan); if (rc) return rc;
    metal_memory::OperatorGuard fft_lock(r->backend->fft_mutex);
    if (!r->backend->fft_roots) r->backend->fft_roots=[NSMutableDictionary new];
    r->encoded=true; r->projection_valid=false;
    rc=encode_fft(r->backend,r->encoder,plan,direction,r->current,r->next);
    if (rc) r->wave_valid=false;
    return rc;
}

extern "C" int metal_backend_resident_apply_hamiltonian(metal_resident *r) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r) || !r->wave_valid || !r->potential_valid) return METAL_BACKEND_INVALID_ARGUMENT;
    struct ResidentParams { uint32_t total,bands,npro; } p{(uint32_t)r->total,(uint32_t)r->bands,(uint32_t)r->npro};
    [r->encoder setComputePipelineState:r->backend->resident_hamiltonian];
    [r->encoder setBuffer:r->current offset:0 atIndex:0]; [r->encoder setBuffer:r->potential offset:0 atIndex:1];
    [r->encoder setBuffer:r->next offset:0 atIndex:2]; [r->encoder setBytes:&p length:sizeof(p) atIndex:3];
    [r->encoder dispatchThreads:MTLSizeMake(r->total*r->bands,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [r->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]; std::swap(r->current,r->next);
    r->encoded=true; r->projection_valid=false; return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_project(metal_resident *r) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r) || !r->wave_valid || !r->projectors_valid) return METAL_BACKEND_INVALID_ARGUMENT;
    struct ResidentParams { uint32_t total,bands,npro; } p{(uint32_t)r->total,(uint32_t)r->bands,(uint32_t)r->npro};
    [r->encoder setComputePipelineState:r->backend->resident_project];
    [r->encoder setBuffer:r->current offset:0 atIndex:0]; [r->encoder setBuffer:r->projectors offset:0 atIndex:1];
    [r->encoder setBuffer:r->projection offset:0 atIndex:2]; [r->encoder setBytes:&p length:sizeof(p) atIndex:3];
    [r->encoder dispatchThreads:MTLSizeMake(r->bands*r->npro,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [r->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    r->encoded=true; r->projection_valid=true; return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_orthogonalize(metal_resident *r) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r) || !r->wave_valid || r->bands>r->total) return METAL_BACKEND_INVALID_ARGUMENT;
    struct ResidentParams { uint32_t total,bands,npro; } p{(uint32_t)r->total,(uint32_t)r->bands,(uint32_t)r->npro};
    [r->encoder setComputePipelineState:r->backend->resident_orth]; [r->encoder setBuffer:r->current offset:0 atIndex:0];
    [r->encoder setBytes:&p length:sizeof(p) atIndex:1]; [r->encoder setBuffer:r->result_status offset:0 atIndex:2];
    [r->encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [r->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    r->encoded=true; r->projection_valid=false; return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_end_step(metal_resident *r, int wait_for_host) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (resident_require_open(r) || !r->wave_valid) return METAL_BACKEND_INVALID_ARGUMENT;
    [r->encoder endEncoding]; r->encoder=nil; [r->command commit]; ++r->submissions; ++r->backend->command_count;
    r->last_command=r->command; r->last_accounted=false; r->command=nil; r->open=false; ++r->steps;
    if (wait_for_host) return resident_wait_last(r);
    return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_download_wavefunction(metal_resident *r,
                                                              metal_complex64 *wave,
                                                              size_t ld) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (!wave || ld<r->total || ld>SIZE_MAX/r->bands/sizeof(*wave) || r->open || !r->wave_valid) return METAL_BACKEND_INVALID_ARGUMENT;
    int rc=resident_wait_last(r); if (rc) return rc;
    { ConversionTimer timer{r->backend}; const auto *src=static_cast<const vector_float4 *>(r->current.contents);
      for(size_t band=0;band<r->bands;++band) for(size_t g=0;g<r->total;++g) { auto x=src[band*r->total+g]; wave[g+band*ld]={split_to_double(x.x,x.y),split_to_double(x.z,x.w)}; } }
    return METAL_BACKEND_OK;
}

extern "C" int metal_backend_resident_download_projection(metal_resident *r, metal_complex64 *out, size_t ld) {
    if (!r) return METAL_BACKEND_INVALID_ARGUMENT;
    metal_memory::OperatorGuard lock(r->mutex);
    if (!out || !r->npro || ld<r->npro || ld>SIZE_MAX/r->bands/sizeof(*out) || r->open || !r->projection_valid) return METAL_BACKEND_INVALID_ARGUMENT;
    int rc=resident_wait_last(r); if (rc) return rc;
    ConversionTimer timer{r->backend}; const auto *src=static_cast<const vector_float4 *>(r->projection.contents);
    for (size_t band=0;band<r->bands;++band) for (size_t p=0;p<r->npro;++p) {
        auto x=src[band*r->npro+p]; out[p+band*ld]={split_to_double(x.x,x.y),split_to_double(x.z,x.w)};
    }
    return METAL_BACKEND_OK;
}

extern "C" uint64_t metal_backend_resident_generation(const metal_resident *r) { if (!r) return 0; metal_memory::OperatorGuard lock(r->mutex); return r->generation; }
extern "C" uint64_t metal_backend_resident_steps(const metal_resident *r) { if (!r) return 0; metal_memory::OperatorGuard lock(r->mutex); return r->steps; }
extern "C" uint64_t metal_backend_resident_uploads(const metal_resident *r) { if (!r) return 0; metal_memory::OperatorGuard lock(r->mutex); return r->uploads; }
extern "C" uint64_t metal_backend_resident_submissions(const metal_resident *r) { if (!r) return 0; metal_memory::OperatorGuard lock(r->mutex); return r->submissions; }
extern "C" uint64_t metal_backend_resident_waits(const metal_resident *r) { if (!r) return 0; metal_memory::OperatorGuard lock(r->mutex); return r->waits; }

extern "C" int metal_backend_get_chunk_stats(const metal_backend *b,metal_chunk_stats *out) {
    if(!b || !out)return METAL_BACKEND_INVALID_ARGUMENT;
    std::scoped_lock lock(b->staging_mutex,b->fft_mutex);*out=b->chunks;return METAL_BACKEND_OK;
}
extern "C" int metal_backend_get_memory_stats(const metal_backend *b,metal_memory_stats *out) {
    if (!b || !out) return METAL_BACKEND_INVALID_ARGUMENT;
    *out=metal_memory::snapshot(b->device); return METAL_BACKEND_OK;
}
extern "C" void metal_backend_prepare_memory_stats(void) {
    (void)metal_memory::registry();
}
extern "C" uint64_t metal_backend_cooperative_orth_calls(const metal_backend *b) {
    return b ? b->cooperative_orth_calls.load():0;
}
extern "C" void metal_backend_record_cooperative_orth(const metal_backend *b) {
    if(b)++b->cooperative_orth_calls;
}
extern "C" uint64_t metal_backend_fft_compact_axes(const metal_backend *b) {
    if(!b)return 0;metal_memory::OperatorGuard guard(b->fft_mutex);return b->fft_compact_axes;
}
extern "C" uint64_t metal_backend_fft_wide_axes(const metal_backend *b) {
    if(!b)return 0;metal_memory::OperatorGuard guard(b->fft_mutex);return b->fft_wide_axes;
}
extern "C" int metal_backend_zgemm_blas_gpu(const metal_backend *backend,char ta,char tb,
 size_t m,size_t n,size_t k,metal_complex64 alpha,const metal_complex64 *a,size_t lda,
 const metal_complex64 *b,size_t ldb,metal_complex64 beta,metal_complex64 *c,size_t ldc) {
    if(backend)backend->last_gpu_success=false;
    int rc=gpu_zgemm_blas(backend,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    if(backend && rc==METAL_BACKEND_OK)backend->last_gpu_success=true;return rc;
}

extern "C" int metal_backend_zgemm_blas(const metal_backend *backend, char ta, char tb,
                                          size_t m, size_t n, size_t k, metal_complex64 alpha,
                                          const metal_complex64 *a, size_t lda,
                                          const metal_complex64 *b, size_t ldb,
                                          metal_complex64 beta, metal_complex64 *c, size_t ldc) {
    if (!backend) return METAL_BACKEND_INVALID_ARGUMENT;
    backend->last_gpu_success = false;
    if (backend->use_gpu) {
        const int code = gpu_zgemm_blas(backend, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        if (code == METAL_BACKEND_OK) { backend->last_gpu_success = true; return code; }
    }
    if (!trans_valid(ta) || !trans_valid(tb) || !a || !b || !c || m == 0 || n == 0 || k == 0) return METAL_BACKEND_INVALID_ARGUMENT;
    const size_t a_rows = transposes(ta) ? k : m;
    const size_t b_rows = transposes(tb) ? n : k;
    if (!metal_gemm::host_shape(m,n,k,transposes(ta),transposes(tb),lda,ldb,ldc)) return METAL_BACKEND_INVALID_ARGUMENT;
    for (size_t col = 0; col < n; ++col) for (size_t row = 0; row < m; ++row) {
        cd sum{0.0, 0.0};
        for (size_t inner = 0; inner < k; ++inner) sum += load_blas(a, lda, row, inner, ta) * load_blas(b, ldb, inner, col, tb);
        const cd old = beta.real == 0.0 && beta.imag == 0.0 ? cd{0.0, 0.0} : cd{c[row + col * ldc].real, c[row + col * ldc].imag};
        const cd value{alpha.real, alpha.imag};
        const cd scale{beta.real, beta.imag};
        const cd out = value * sum + scale * old;
        c[row + col * ldc] = {out.real(), out.imag()};
    }
    return METAL_BACKEND_OK;
}

extern "C" int metal_backend_zgemm(const metal_backend *backend, size_t m, size_t n, size_t k,
                                    const metal_complex64 *a, size_t lda, const metal_complex64 *b, size_t ldb,
                                    metal_complex64 *c, size_t ldc) {
    return metal_backend_zgemm_blas(backend, 'N', 'N', m, n, k,
                                    metal_complex64{1.0, 0.0}, a, lda, b, ldb,
                                    metal_complex64{0.0, 0.0}, c, ldc);
}

extern "C" int metal_backend_zfft(const metal_backend *backend, size_t n, const metal_complex64 *input,
                                   metal_complex64 *output, int direction) {
    if (backend) backend->last_gpu_success = false;
    if (backend && backend->use_gpu) {
        const int code = gpu_fft(backend, n, input, output, direction);
        if (code == METAL_BACKEND_OK) { backend->last_gpu_success = true; return code; }
    }
    return cpu_fft(n, input, output, direction);
}

extern "C" int metal_backend_zfft3d(const metal_backend *backend, size_t nx, size_t ny, size_t nz,
                                     const metal_complex64 *input, metal_complex64 *output, int direction) {
    if (backend) backend->last_gpu_success = false;
    if (backend && backend->use_gpu) {
        const int code = gpu_fft3d(backend, nx, ny, nz, input, output, direction);
        if (code == METAL_BACKEND_OK) { backend->last_gpu_success = true; return code; }
    }
    return cpu_fft3d(nx, ny, nz, input, output, direction);
}

extern "C" int metal_backend_zfft3d_batch_gpu(const metal_backend *backend,
 size_t nx,size_t ny,size_t nz,size_t nbatch,size_t stride,const metal_complex64 *input,metal_complex64 *output,int direction,int normalize) {
    if(backend) backend->last_gpu_success=false;
    const int rc=gpu_fft3d_many(backend,nx,ny,nz,nbatch,stride,input,output,direction,normalize!=0);
    if(backend && rc==METAL_BACKEND_OK) backend->last_gpu_success=true;
    return rc;
}
extern "C" int metal_backend_zfft3d_batch(const metal_backend *backend,
 size_t nx,size_t ny,size_t nz,size_t nbatch,size_t stride,const metal_complex64 *input,metal_complex64 *output,int direction) {
    size_t total=0;
    if(!backend || !input || !output || (direction!=-1 && direction!=1) || !fft_shape(nx,ny,nz,nbatch,stride,total)) return METAL_BACKEND_INVALID_ARGUMENT;
    int rc=metal_backend_zfft3d_batch_gpu(backend,nx,ny,nz,nbatch,stride,input,output,direction,1);
    if(rc==METAL_BACKEND_OK) return rc;
    for(size_t b=0;b<nbatch;++b) { rc=cpu_fft3d(nx,ny,nz,input+b*stride,output+b*stride,direction);if(rc) return rc; }
    return METAL_BACKEND_OK;
}
extern "C" int metal_backend_dsum(const metal_backend *backend, size_t n, const double *input, double *result) {
    if (!backend) return METAL_BACKEND_INVALID_ARGUMENT;
    backend->last_gpu_success = false;
    if (backend->use_gpu) { const int code = gpu_sum(backend, n, input, result); if (code == METAL_BACKEND_OK) { backend->last_gpu_success = true; return code; } }
    return cpu_sum(n, input, result);
}

extern "C" int metal_backend_vh_apply(const metal_backend *backend, size_t np, size_t mplwv,
                                         size_t nrspinors, const metal_complex64 *sv,
                                         const metal_complex64 *cr, metal_complex64 *cvr,
                                         double rinplw) {
    if (backend) backend->last_gpu_success = false;
    if (!backend || !backend->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    const int code = gpu_vh_apply(backend, np, mplwv, nrspinors, sv, cr, cvr, rinplw);
    if (backend && code == METAL_BACKEND_OK) backend->last_gpu_success = true;
    return code;
}

extern "C" int metal_backend_vh_apply_batch(const metal_backend *backend, size_t np, size_t mplwv,
                                              size_t nrspinors, size_t nbands,
                                              const metal_complex64 *sv, const metal_complex64 *cr,
                                              metal_complex64 *cvr, double rinplw) {
    if (backend) backend->last_gpu_success = false;
    if (!backend || !backend->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    const int code = gpu_vh_apply_batch(backend, np, mplwv, nrspinors, nbands, sv, cr, cvr, rinplw);
    if (code == METAL_BACKEND_OK) backend->last_gpu_success = true;
    return code;
}

extern "C" int metal_backend_vh_apply_real(const metal_backend *backend, size_t np, size_t mplwv,
                                             size_t nrspinors, const double *sv,
                                             const metal_complex64 *cr, metal_complex64 *cvr,
                                             double rinplw) {
    if (backend) backend->last_gpu_success = false;
    if (!backend || !backend->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    const int code = gpu_vh_apply_real(backend,np,mplwv,nrspinors,sv,cr,cvr,rinplw);
    if (backend && code == METAL_BACKEND_OK) backend->last_gpu_success = true;
    return code;
}

extern "C" int metal_backend_vh_apply_real_batch(const metal_backend *backend, size_t np, size_t mplwv,
                                                   size_t nrspinors, size_t nbands, const double *sv,
                                                   const metal_complex64 *cr, metal_complex64 *cvr,
                                                   double rinplw) {
    if (backend) backend->last_gpu_success = false;
    if (!backend || !backend->use_gpu) return METAL_BACKEND_UNSUPPORTED;
    const int code = gpu_vh_apply_real_batch(backend, np, mplwv, nrspinors, nbands, sv, cr, cvr, rinplw);
    if (code == METAL_BACKEND_OK) backend->last_gpu_success = true;
    return code;
}

extern "C" int metal_backend_hermitian_eigh(const metal_backend *backend, size_t n, const metal_complex64 *a,
                                             size_t lda, double *w, metal_complex64 *v, size_t ldv) {
    if (backend) backend->last_gpu_success = false;
    return cpu_eigh(n, a, lda, w, v, ldv);
}

extern "C" int metal_backend_hermitian_eigh_gpu(const metal_backend *backend, size_t n, const metal_complex64 *a,
                                                 size_t lda, double *w, metal_complex64 *v, size_t ldv) {
    if (backend) backend->last_gpu_success = false;
    const int rc = gpu_hermitian_eigh(backend, n, a, lda, w, v, ldv);
    if (backend && rc == METAL_BACKEND_OK) backend->last_gpu_success = true;
    return rc;
}

extern "C" int metal_backend_generalized_eigh_gpu(const metal_backend *backend, int itype, char uplo, char jobz,
                                                    size_t n, const metal_complex64 *a, size_t lda,
                                                    const metal_complex64 *b, size_t ldb, double *w,
                                                    metal_complex64 *v, size_t ldv) {
    if (backend) backend->last_gpu_success = false;
    const int rc = gpu_generalized_eigh(backend, itype, uplo, jobz, n, a, lda, b, ldb, w, v, ldv);
    if (backend && rc == METAL_BACKEND_OK) backend->last_gpu_success = true;
    return rc;
}

extern "C" int metal_backend_generalized_all_gpu(const metal_backend *backend) { return backend && backend->generalized_all_gpu ? 1:0; }
extern "C" int metal_backend_generalized_eigh_factor_gpu(const metal_backend *backend,int itype,char uplo,char jobz,
 size_t n,const metal_complex64 *a,size_t lda,const metal_complex64 *b,size_t ldb,double *w,metal_complex64 *v,size_t ldv,metal_complex64 *factor,size_t ldf) {
    if(backend) backend->last_gpu_success=false;
    const int rc=gpu_generalized_eigh(backend,itype,uplo,jobz,n,a,lda,b,ldb,w,v,ldv,factor,ldf);
    if(backend && rc==METAL_BACKEND_OK) backend->last_gpu_success=true;
    return rc;
}

// Integer character codes avoid scalar CHARACTER ABI differences among Apple
// Silicon Fortran compilers; alpha/beta use explicit pointers in this bridge.
extern "C" int metal_backend_zgemm_blas_f90(const metal_backend *backend, int ta, int tb,
    size_t m, size_t n, size_t k, const metal_complex64 *alpha,
    const metal_complex64 *a, size_t lda, const metal_complex64 *b, size_t ldb,
    const metal_complex64 *beta, metal_complex64 *c, size_t ldc) {
    if (!alpha || !beta) return METAL_BACKEND_INVALID_ARGUMENT;
    return metal_backend_zgemm_blas(backend,static_cast<char>(ta),static_cast<char>(tb),m,n,k,*alpha,a,lda,b,ldb,*beta,c,ldc);
}
extern "C" int metal_backend_generalized_eigh_gpu_f90(const metal_backend *backend,
    int itype, int uplo, int jobz, size_t n, const metal_complex64 *a, size_t lda,
    const metal_complex64 *b, size_t ldb, double *w, metal_complex64 *v, size_t ldv) {
    return metal_backend_generalized_eigh_gpu(backend,itype,static_cast<char>(uplo),static_cast<char>(jobz),n,a,lda,b,ldb,w,v,ldv);
}

extern "C" int metal_backend_zdotc_gpu(const metal_backend *backend,size_t n,
    const metal_complex64 *x,ptrdiff_t incx,const metal_complex64 *y,ptrdiff_t incy,metal_complex64 *result) {
    if(backend) backend->last_gpu_success=false;
    int rc=gpu_zdotc(backend,n,x,incx,y,incy,result);
    if(backend && rc==METAL_BACKEND_OK) backend->last_gpu_success=true;
    return rc;
}
extern "C" int metal_backend_zdotc(const metal_backend *backend,size_t n,
    const metal_complex64 *x,ptrdiff_t incx,const metal_complex64 *y,ptrdiff_t incy,metal_complex64 *result) {
    if(!result || (n && (!x || !y || !incx || !incy))) return METAL_BACKEND_INVALID_ARGUMENT;
    if(backend) backend->last_gpu_success=false;
    if(n && backend && backend->use_gpu) {
        int rc=metal_backend_zdotc_gpu(backend,n,x,incx,y,incy,result);
        if(rc==METAL_BACKEND_OK) return rc;
    }
    const ptrdiff_t sx=incx<0 ? (1-static_cast<ptrdiff_t>(n))*incx:0;
    const ptrdiff_t sy=incy<0 ? (1-static_cast<ptrdiff_t>(n))*incy:0;
    std::complex<double> sum=0.;
    for(size_t i=0;i<n;++i) {
        auto a=x[sx+static_cast<ptrdiff_t>(i)*incx],b=y[sy+static_cast<ptrdiff_t>(i)*incy];
        sum+=std::conj(std::complex<double>(a.real,a.imag))*std::complex<double>(b.real,b.imag);
    }
    *result={sum.real(),sum.imag()};return METAL_BACKEND_OK;
}

extern "C" uint64_t metal_backend_recommended_working_set_bytes(const metal_backend *b) {
 return b && b->device ? metal_memory::budget(b->device):0;
}

#include "metal_nonlocal_impl.h"
