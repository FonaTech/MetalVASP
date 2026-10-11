#include "metal_memory_allocator.hpp"
#include "metal_orth_dispatch.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>
#include "metal_wave_owner_davidson.h"
#include "metal_wave_owner_internal.h"
#include "metal_backend_private.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>
namespace {
void evict_backend(const void *b) {metal_backend_evict_idle_buffers(static_cast<const metal_backend *>(b));}
struct OrthParams {uint32_t rows,cols,npl,npro;};
struct GemmParams {uint32_t m,n,k;vector_float4 alpha,beta;};
struct Extension {
 id<MTLComputePipelineState> orth=nil,gemm=nil;
 id<MTLComputePipelineState> orth_group=nil;
 id<MTLBuffer> wave=nil,projectors=nil,overlap=nil;
 size_t bands=0;uint64_t updated_generation=0,invalidations=0;
 bool basis_valid=false;
 metal_wave_davidson_stats stats{};metal_wave_basis_transfer_stats transfers{};
};
vector_float4 split(metal_complex64 z) {
 float r=float(z.real),i=float(z.imag);return {r,float(z.real-r),i,float(z.imag-i)};
}
metal_complex64 join(vector_float4 z) {return {double(z.x)+z.y,double(z.z)+z.w};}
struct Timer {
 const metal_backend *b;std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
 ~Timer(){metal_backend_record_external_conversion(b,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());}
};
std::shared_ptr<Extension> extension(metal_wave_owner *o) {
 if(o->davidson) return std::static_pointer_cast<Extension>(o->davidson);
 auto e=std::make_shared<Extension>();NSError *error=nil;
 auto orth=[o->library newFunctionWithName:@"ds_orth_overlap"],gemm=[o->library newFunctionWithName:@"ds_zgemm"];
 if(orth) e->orth=[o->device newComputePipelineStateWithFunction:orth error:&error];
 auto group=[o->library newFunctionWithName:@"ds_orth_overlap_group"];
 if(group)e->orth_group=[o->device newComputePipelineStateWithFunction:group error:&error];
 if(gemm) e->gemm=[o->device newComputePipelineStateWithFunction:gemm error:&error];
 if(!e->orth || !e->gemm) return {};
 o->davidson=e;return e;
}
}
static int run_update_locked(metal_wave_owner *o,uint64_t generation,const metal_wave_basis_input *basis,id<MTLCommandBuffer> command,const metal_wave_basis_delta *delta=nullptr) {
 const auto s=o->shape;const bool fused=command!=nil;
 const bool prepared=fused ? generation==o->accepted_generation :
     (o->valid && generation==o->stats.generation && generation!=o->exported_generation);
 o->valid=false;metal_backend_set_external_success(o->backend,0);
 if(!prepared || !basis || !basis->generation || !basis->bands || !basis->wave || !basis->projectors ||
    basis->ldwave<s.npl || basis->ldprojectors<s.npro || basis->bands>UINT32_MAX/s.npl ||
    basis->bands>UINT32_MAX/s.npro || basis->bands>UINT32_MAX/s.bands ||
    basis->ldwave>SIZE_MAX/basis->bands/sizeof(metal_complex64) ||
    basis->ldprojectors>SIZE_MAX/basis->bands/sizeof(metal_complex64)) return METAL_BACKEND_INVALID_ARGUMENT;
 if(delta && (delta->first_column>basis->bands || delta->column_count>basis->bands-delta->first_column))
   return METAL_BACKEND_INVALID_ARGUMENT;
 try {
   auto e=extension(o);if(!e) return METAL_BACKEND_UNSUPPORTED;
   if(e->updated_generation==generation || (e->basis_valid && basis->generation<e->stats.basis_generation)) return METAL_BACKEND_INVALID_ARGUMENT;
   if(delta && e->basis_valid && basis->generation==e->stats.basis_generation && delta->column_count)
     return METAL_BACKEND_INVALID_ARGUMENT;
   const bool resized=e->bands!=basis->bands;
   if(resized) {
     const size_t wc=s.npl*basis->bands,pc=s.npro*basis->bands,oc=basis->bands*s.bands;
     if(wc>SIZE_MAX/16 || pc>SIZE_MAX/16 || oc>SIZE_MAX/16 || wc*16>o->device.maxBufferLength ||
        pc*16>o->device.maxBufferLength || oc*16>o->device.maxBufferLength) return METAL_BACKEND_ALLOCATION_FAILED;
     auto wave=metal_memory::allocate(o->device,wc*16,MTLResourceStorageModeShared,evict_backend,o->backend);
     auto projectors=metal_memory::allocate(o->device,pc*16,MTLResourceStorageModeShared,evict_backend,o->backend);
     auto overlap=metal_memory::allocate(o->device,oc*16,MTLResourceStorageModeShared,evict_backend,o->backend);
     if(!wave || !projectors || !overlap) return METAL_BACKEND_ALLOCATION_FAILED;
     o->stats.device_bytes-=e->stats.device_bytes;
     e->wave=wave;e->projectors=projectors;e->overlap=overlap;e->bands=basis->bands;e->basis_valid=false;
     e->stats.device_bytes=(wc+pc+oc)*16;o->stats.device_bytes+=e->stats.device_bytes;
   }
   const bool refresh=!e->basis_valid || e->stats.basis_generation!=basis->generation || e->invalidations!=o->stats.invalidations;
   if(refresh) {
     const bool partial=delta && e->basis_valid && !resized && e->invalidations==o->stats.invalidations &&
         e->stats.basis_generation!=UINT64_MAX && basis->generation==e->stats.basis_generation+1 &&
         delta->previous_generation==e->stats.basis_generation && delta->column_count>0;
     const size_t first=partial ? delta->first_column:0,count=partial ? delta->column_count:basis->bands;
     Timer timer{o->backend};
     for(size_t b=first;b<first+count;++b) {
       for(size_t g=0;g<s.npl;++g) static_cast<vector_float4 *>(e->wave.contents)[g+b*s.npl]=split(basis->wave[g+b*basis->ldwave]);
       for(size_t j=0;j<s.npro;++j) static_cast<vector_float4 *>(e->projectors.contents)[j+b*s.npro]=split(basis->projectors[j+b*basis->ldprojectors]);
     }
     const uint64_t bytes=(s.npl+s.npro)*count*16;
     e->stats.basis_import_bytes+=bytes;o->stats.host_import_bytes+=bytes;++e->stats.basis_uploads;
     e->stats.basis_generation=basis->generation;e->invalidations=o->stats.invalidations;e->basis_valid=true;
     if(partial) ++e->transfers.partial_uploads;else ++e->transfers.full_uploads;
     e->transfers.uploaded_columns+=count;
   } else {++e->stats.basis_reuses;++e->transfers.reuses;}
   // The original caller zeroes CORTHO before ORTH1's beta=1 accumulation.
   // This is a host initialization, not a device-result export.
   std::memset(e->overlap.contents,0,e->overlap.length);
   e->stats.overlap_zero_init_host_bytes+=e->overlap.length;o->stats.host_import_bytes+=e->overlap.length;
   if(!command) command=[o->queue commandBuffer];auto encoder=[command computeCommandEncoder];
   if(!command || !encoder) return METAL_BACKEND_RUNTIME_FAILED;
   OrthParams p{uint32_t(basis->bands),uint32_t(s.bands),uint32_t(s.npl),uint32_t(o->use_overlap ? s.npro:0)};
   const bool grouped=metal_orth::cooperative(s.npl,o->use_overlap ? s.npro:0,e->orth_group && e->orth_group.maxTotalThreadsPerThreadgroup>=256);
   [encoder setComputePipelineState:grouped ? e->orth_group:e->orth];
   [encoder setBuffer:e->wave offset:0 atIndex:0];[encoder setBuffer:o->wave offset:0 atIndex:1];
   [encoder setBuffer:e->projectors offset:0 atIndex:2];[encoder setBuffer:o->metric_projection offset:0 atIndex:3];
   [encoder setBuffer:e->overlap offset:0 atIndex:4];[encoder setBytes:&p length:sizeof(p) atIndex:5];
   if(grouped)[encoder dispatchThreadgroups:MTLSizeMake(basis->bands,s.bands,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
   else [encoder dispatchThreads:MTLSizeMake(basis->bands,s.bands,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,e->orth.maxTotalThreadsPerThreadgroup),1,1)];
   [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
   // Column-major C and A are row-major C^T and A^T. The existing strict
   // DS kernel computes (A*C)^T=C^T*A^T directly without packing or staging.
   auto update=[&](id<MTLBuffer> a,id<MTLBuffer> out,size_t rows) {
     GemmParams gp{uint32_t(s.bands),uint32_t(rows),uint32_t(basis->bands),{-1,0,0,0},{1,0,0,0}};
     [encoder setComputePipelineState:e->gemm];[encoder setBuffer:e->overlap offset:0 atIndex:0];
     [encoder setBuffer:a offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];
     [encoder setBytes:&gp length:sizeof(gp) atIndex:3];
     [encoder dispatchThreads:MTLSizeMake(rows,s.bands,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,e->gemm.maxTotalThreadsPerThreadgroup),1,1)];
     [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
   };
   update(e->wave,o->wave,s.npl);update(e->projectors,o->projection,s.npro);
   [encoder endEncoding];int rc=metal_backend_complete_external_command(o->backend,(__bridge void *)command);
   if(rc) return rc;
   if(grouped)metal_backend_record_cooperative_orth(o->backend);
   if(fused) {
     rc=metal_wave_owner_check_norm_locked(o);if(rc) return rc;
     o->stats.generation=generation;++o->stats.steps;
   }
   e->updated_generation=generation;++e->stats.steps;o->valid=true;metal_backend_set_external_success(o->backend,1);return METAL_BACKEND_OK;
 } catch(const std::bad_alloc &) {return METAL_BACKEND_ALLOCATION_FAILED;}
}
extern "C" int metal_wave_owner_davidson_update(metal_wave_owner *o,uint64_t generation,const metal_wave_basis_input *basis) {
 if(!o) return METAL_BACKEND_INVALID_ARGUMENT;std::lock_guard<std::mutex> lock(o->mutex);
 return run_update_locked(o,generation,basis,nil);
}
extern "C" int metal_wave_owner_davidson_prepare(metal_wave_owner *o,uint64_t generation,uint64_t epoch,
 const metal_wave_prepare_input *input,const metal_wave_basis_input *basis) {
 if(!o) return METAL_BACKEND_INVALID_ARGUMENT;std::lock_guard<std::mutex> lock(o->mutex);
 id<MTLCommandBuffer> command=nil;
 int rc=metal_wave_owner_prepare_locked(o,generation,epoch,input,&command);if(rc) return rc;
 return run_update_locked(o,generation,basis,command);
}
extern "C" int metal_wave_owner_davidson_prepare_delta(metal_wave_owner *o,uint64_t generation,uint64_t epoch,
 const metal_wave_prepare_input *input,const metal_wave_basis_input *basis,const metal_wave_basis_delta *delta) {
 if(!o || !delta) return METAL_BACKEND_INVALID_ARGUMENT;std::lock_guard<std::mutex> lock(o->mutex);
 id<MTLCommandBuffer> command=nil;
 int rc=metal_wave_owner_prepare_locked(o,generation,epoch,input,&command);if(rc) return rc;
 return run_update_locked(o,generation,basis,command,delta);
}
extern "C" void metal_wave_owner_davidson_get_transfer_stats(const metal_wave_owner *o,metal_wave_basis_transfer_stats *out) {
 if(!out) return;*out={};if(!o) return;std::lock_guard<std::mutex> lock(o->mutex);
 if(o->davidson) *out=std::static_pointer_cast<Extension>(o->davidson)->transfers;
}
extern "C" int metal_wave_owner_davidson_download_overlap(metal_wave_owner *o,uint64_t generation,metal_complex64 *out,size_t ld) {
 if(!o) return METAL_BACKEND_INVALID_ARGUMENT;std::lock_guard<std::mutex> lock(o->mutex);
 auto e=std::static_pointer_cast<Extension>(o->davidson);
 if(!o->valid || !e || e->updated_generation!=generation || !out || ld<e->bands || ld>SIZE_MAX/o->shape.bands/sizeof(*out)) return METAL_BACKEND_INVALID_ARGUMENT;
 try {
   Timer timer{o->backend};std::vector<metal_complex64> values(e->bands*o->shape.bands);
   for(size_t i=0;i<values.size();++i) {
     values[i]=join(static_cast<const vector_float4 *>(e->overlap.contents)[i]);
     if(!std::isfinite(values[i].real) || !std::isfinite(values[i].imag)) {o->valid=false;metal_backend_set_external_success(o->backend,0);return METAL_BACKEND_RUNTIME_FAILED;}
   }
   for(size_t b=0;b<o->shape.bands;++b) std::copy_n(values.data()+b*e->bands,e->bands,out+b*ld);
   e->stats.overlap_export_bytes+=values.size()*sizeof(*out);o->stats.host_export_bytes+=values.size()*sizeof(*out);return METAL_BACKEND_OK;
 } catch(const std::bad_alloc &) {return METAL_BACKEND_ALLOCATION_FAILED;}
}
extern "C" void metal_wave_owner_davidson_get_stats(const metal_wave_owner *o,metal_wave_davidson_stats *out) {
 if(!out) return;*out={};if(!o) return;std::lock_guard<std::mutex> lock(o->mutex);
 if(o->davidson) *out=std::static_pointer_cast<Extension>(o->davidson)->stats;
 out->control_status_host_bytes=o->control_status_host_bytes;
}
extern "C" void metal_wave_owner_davidson_forget_outputs(metal_wave_owner *o) {
 if(!o) return;std::lock_guard<std::mutex> lock(o->mutex);o->valid=false;
}
