#include "metal_memory_allocator.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>
#include "metal_wave_owner.h"
#include "metal_backend_private.h"
#include "metal_wave_owner_internal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <vector>
#include <initializer_list>

namespace {
void evict_backend(const void *b) {metal_backend_evict_idle_buffers(static_cast<const metal_backend *>(b));}
struct Site {uint32_t ion,type,base,count,l;};
struct Params {uint32_t npl,bands,npro,nions,lmdim,qchannels,use_overlap;};
struct Timer {
 const metal_backend *backend;std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
 ~Timer(){metal_backend_record_external_conversion(backend,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());}
};
vector_float2 split(double x) {float hi=static_cast<float>(x);return (vector_float2){hi,static_cast<float>(x-double(hi))};}
vector_float4 split(metal_complex64 z) {auto r=split(z.real),i=split(z.imag);return (vector_float4){r.x,r.y,i.x,i.y};}
metal_complex64 join(vector_float4 z){return {double(z.x)+z.y,double(z.z)+z.w};}
bool finite(metal_complex64 z){return std::isfinite(z.real)&&std::isfinite(z.imag);}
bool shape_valid(metal_wave_owner_shape s) {
 if(!s.npl || !s.bands || !s.nions || !s.npro || !s.lmdim || !s.ntypes || s.qchannels<s.lmdim) return false;
 if(s.npl>UINT32_MAX || s.bands>UINT32_MAX || s.npro>UINT32_MAX || s.lmdim>UINT32_MAX || s.qchannels>UINT32_MAX) return false;
 if(s.npro>(UINT32_MAX-s.npl)/2 || s.bands>UINT32_MAX/(s.npl+2*s.npro)) return false;
 if(s.nions>UINT32_MAX/s.lmdim/s.lmdim || s.ntypes>UINT32_MAX/s.qchannels) return false;
 return s.npl<=UINT32_MAX/(s.ntypes*s.qchannels) && s.nions<=UINT32_MAX/s.npl;
}
bool shape_equal(metal_wave_owner_shape a,metal_wave_owner_shape b) {
 return a.npl==b.npl && a.bands==b.bands && a.nions==b.nions && a.npro==b.npro &&
        a.lmdim==b.lmdim && a.ntypes==b.ntypes && a.qchannels==b.qchannels;
}
}
extern "C" int metal_wave_owner_create(const metal_backend *b,metal_wave_owner_shape s,metal_wave_owner **out) {
 if(!out) return METAL_BACKEND_INVALID_ARGUMENT;*out=nullptr;
 if(!b || !shape_valid(s)) return METAL_BACKEND_INVALID_ARGUMENT;
 void *device=nullptr,*queue=nullptr,*library=nullptr;
 int rc=metal_backend_copy_metal_resources(b,&device,&queue,&library);if(rc) return rc;
 auto *o=new(std::nothrow) metal_wave_owner;if(!o) return METAL_BACKEND_ALLOCATION_FAILED;
 o->backend=b;o->shape=s;o->device=(__bridge id<MTLDevice>)device;o->queue=(__bridge id<MTLCommandQueue>)queue;o->library=(__bridge id<MTLLibrary>)library;
 auto pso=[&](NSString *name){NSError *error=nil;auto f=[o->library newFunctionWithName:name];return f ? [o->device newComputePipelineStateWithFunction:f error:&error]:nil;};
 o->precondition=pso(@"ds_wave_precondition");o->project=pso(@"ds_wave_project");o->metric=pso(@"ds_wave_metric");
 o->norm=pso(@"ds_wave_norm");o->normalize=pso(@"ds_wave_normalize");
 if(!o->precondition || !o->project || !o->metric || !o->norm || !o->normalize ||
    o->project.maxTotalThreadsPerThreadgroup<256 || o->norm.maxTotalThreadsPerThreadgroup<256) {delete o;return METAL_BACKEND_UNSUPPORTED;}
 auto allocate=[&](id<MTLBuffer> __strong &buffer,size_t count,size_t width) {
   if(count>SIZE_MAX/width || count*width>o->device.maxBufferLength) return false;
   buffer=metal_memory::allocate(o->device,count*width,MTLResourceStorageModeShared,evict_backend,o->backend);
   if(buffer) o->stats.device_bytes+=count*width;return buffer!=nil;
 };
 bool ok=allocate(o->wave,s.npl*s.bands,16) && allocate(o->precon,s.npl*s.bands,8) &&
   allocate(o->projection,s.npro*s.bands,16) && allocate(o->metric_projection,s.npro*s.bands,16) &&
   allocate(o->q,s.lmdim*s.lmdim*s.nions,16) && allocate(o->radial,s.npl*s.qchannels*s.ntypes,8) &&
   allocate(o->cq,s.qchannels*s.ntypes,16) && allocate(o->phase,s.npl*s.nions,16) &&
   allocate(o->site,s.npro,sizeof(Site)) && allocate(o->scale,s.bands,8) && allocate(o->status,s.bands,4);
 if(!ok) {delete o;return METAL_BACKEND_ALLOCATION_FAILED;}
 *out=o;return METAL_BACKEND_OK;
}
extern "C" void metal_wave_owner_destroy(metal_wave_owner *o){delete o;}
extern "C" void metal_wave_owner_invalidate(metal_wave_owner *o) {
 if(!o) return;std::lock_guard<std::mutex> guard(o->mutex);o->valid=false;o->static_valid=false;++o->stats.invalidations;
}
extern "C" int metal_wave_owner_matches(const metal_wave_owner *o,metal_wave_owner_shape s) {return o && shape_equal(o->shape,s);}
int metal_wave_owner_prepare_locked(metal_wave_owner *o,uint64_t generation,uint64_t epoch,const metal_wave_prepare_input *in,id<MTLCommandBuffer> __strong *deferred) {
 if(deferred) *deferred=nil;
 o->valid=false;metal_backend_set_external_success(o->backend,0);
 const auto s=o->shape;
 if(!in || !generation || generation<=o->accepted_generation || !epoch || !in->wave || !in->preconditioner ||
    !in->types || !in->lmmax || !in->bases || !in->radial || !in->cq || !in->phase ||
    (in->use_overlap && !in->metric) || (in->real_metric!=0 && in->real_metric!=1) ||
    (in->use_overlap!=0 && in->use_overlap!=1) || in->ldpreconditioner<s.npl || in->qld<s.npl || in->phase_ld<s.npl ||
    in->ldpreconditioner>SIZE_MAX/s.bands/sizeof(double) || in->qld>SIZE_MAX/(s.qchannels*s.ntypes)/sizeof(double) ||
    in->phase_ld>SIZE_MAX/s.nions/sizeof(metal_complex64)) return METAL_BACKEND_INVALID_ARGUMENT;
 for(size_t b=0;b<s.bands;++b) if(!in->wave[b]) return METAL_BACKEND_INVALID_ARGUMENT;
 if(in->bases[0]!=0 || in->bases[s.nions]!=static_cast<int64_t>(s.npro)) return METAL_BACKEND_UNSUPPORTED;
 for(size_t ion=0;ion<s.nions;++ion) {
   int type=in->types[ion];if(type<1 || static_cast<size_t>(type)>s.ntypes) return METAL_BACKEND_INVALID_ARGUMENT;
   int count=in->lmmax[type-1];
   if(count<0 || static_cast<size_t>(count)>s.lmdim || in->bases[ion]<0 ||
      static_cast<int64_t>(in->bases[ion+1])-in->bases[ion]!=count) return METAL_BACKEND_INVALID_ARGUMENT;
 }
 o->accepted_generation=generation;o->use_overlap=in->use_overlap!=0;
 const bool refresh=!o->static_valid || o->stats.paw_epoch!=epoch;
 {
   Timer timer{o->backend};auto *w=static_cast<vector_float4 *>(o->wave.contents);auto *pre=static_cast<vector_float2 *>(o->precon.contents);
   for(size_t b=0;b<s.bands;++b) for(size_t g=0;g<s.npl;++g) {w[g+b*s.npl]=split(in->wave[b][g]);pre[g+b*s.npl]=split(in->preconditioner[g+b*in->ldpreconditioner]);}
   o->stats.host_import_bytes+=s.npl*s.bands*(sizeof(metal_complex64)+sizeof(double));
   if(o->use_overlap) {
     auto *q=static_cast<vector_float4 *>(o->q.contents);const size_t count=s.lmdim*s.lmdim*s.nions;
     for(size_t i=0;i<count;++i) q[i]=split(in->real_metric ? metal_complex64{static_cast<const double *>(in->metric)[i],0}:static_cast<const metal_complex64 *>(in->metric)[i]);
     o->stats.host_import_bytes+=count*sizeof(vector_float4);
   }
   if(refresh) {
     auto *site=static_cast<Site *>(o->site.contents);auto *rad=static_cast<vector_float2 *>(o->radial.contents);
     for(size_t ion=0;ion<s.nions;++ion) for(int l=0;l<in->lmmax[in->types[ion]-1];++l)
       site[in->bases[ion]+l]={(uint32_t)ion,(uint32_t)(in->types[ion]-1),(uint32_t)in->bases[ion],(uint32_t)in->lmmax[in->types[ion]-1],(uint32_t)l};
     for(size_t c=0;c<s.qchannels*s.ntypes;++c) for(size_t g=0;g<s.npl;++g) rad[g+c*s.npl]=split(in->radial[g+c*in->qld]);
     for(size_t c=0;c<s.qchannels*s.ntypes;++c) static_cast<vector_float4 *>(o->cq.contents)[c]=split(in->cq[c]);
     for(size_t ion=0;ion<s.nions;++ion) for(size_t g=0;g<s.npl;++g) static_cast<vector_float4 *>(o->phase.contents)[g+ion*s.npl]=split(in->phase[g+ion*in->phase_ld]);
     o->stats.host_import_bytes+=s.npl*s.qchannels*s.ntypes*sizeof(double)+s.qchannels*s.ntypes*sizeof(metal_complex64)+
       s.npl*s.nions*sizeof(metal_complex64)+s.npro*sizeof(Site);
     o->stats.paw_epoch=epoch;o->static_valid=true;++o->stats.static_uploads;
   } else ++o->stats.static_reuses;
 }
 Params p{(uint32_t)s.npl,(uint32_t)s.bands,(uint32_t)s.npro,(uint32_t)s.nions,(uint32_t)s.lmdim,(uint32_t)s.qchannels,(uint32_t)o->use_overlap};
 id<MTLCommandBuffer> command=[o->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
 if(!command || !enc) return METAL_BACKEND_RUNTIME_FAILED;
 auto dispatch=[&](id<MTLComputePipelineState> ps,size_t count,std::initializer_list<id<MTLBuffer>> buffers,NSUInteger param_index,bool groups) {
   [enc setComputePipelineState:ps];NSUInteger index=0;for(auto buffer:buffers) [enc setBuffer:buffer offset:0 atIndex:index++];
   [enc setBytes:&p length:sizeof(p) atIndex:param_index];
   if(groups) [enc dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
   else [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,ps.maxTotalThreadsPerThreadgroup),1,1)];
   [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
 };
 dispatch(o->precondition,s.npl*s.bands,{o->wave,o->precon},2,false);
 dispatch(o->project,s.npro*s.bands,{o->wave,o->radial,o->cq,o->phase,o->site,o->projection},6,true);
 dispatch(o->metric,s.npro*s.bands,{o->projection,o->q,o->site,o->metric_projection},4,false);
 dispatch(o->norm,s.bands,{o->wave,o->projection,o->metric_projection,o->scale,o->status},5,true);
 dispatch(o->normalize,(s.npl+2*s.npro)*s.bands,{o->wave,o->projection,o->metric_projection,o->scale},4,false);
 [enc endEncoding];
 if(deferred) {*deferred=command;return METAL_BACKEND_OK;}
 int rc=metal_backend_complete_external_command(o->backend,(__bridge void *)command);if(rc) return rc;
 rc=metal_wave_owner_check_norm_locked(o);if(rc) return rc;
 o->stats.generation=generation;++o->stats.steps;o->valid=true;return METAL_BACKEND_OK;
}
int metal_wave_owner_check_norm_locked(metal_wave_owner *o) {
 const auto *status=static_cast<const uint32_t *>(o->status.contents);
 o->control_status_host_bytes+=o->shape.bands*sizeof(uint32_t);
 uint32_t failed=0;for(size_t b=0;b<o->shape.bands;++b) failed|=status[b];
 if(failed) {metal_backend_set_external_success(o->backend,0);return METAL_BACKEND_RUNTIME_FAILED;}
 return METAL_BACKEND_OK;
}
extern "C" int metal_wave_owner_prepare(metal_wave_owner *o,uint64_t generation,uint64_t epoch,const metal_wave_prepare_input *in) {
 if(!o) return METAL_BACKEND_INVALID_ARGUMENT;std::lock_guard<std::mutex> guard(o->mutex);
 return metal_wave_owner_prepare_locked(o,generation,epoch,in,nullptr);
}
extern "C" int metal_wave_owner_download(metal_wave_owner *o,uint64_t generation,metal_complex64 *const *wave,
 metal_complex64 *const *projectors,metal_complex64 *metric,size_t ldmetric,double *scales) {
 if(!o) return METAL_BACKEND_INVALID_ARGUMENT;std::lock_guard<std::mutex> guard(o->mutex);const auto s=o->shape;
 if(!o->valid || generation!=o->stats.generation || !wave || !projectors || !scales ||
    (o->use_overlap && (!metric || ldmetric<s.npro || ldmetric>SIZE_MAX/s.bands/sizeof(*metric)))) return METAL_BACKEND_INVALID_ARGUMENT;
 for(size_t b=0;b<s.bands;++b) if(!wave[b] || !projectors[b]) return METAL_BACKEND_INVALID_ARGUMENT;
 try {
   Timer timer{o->backend};std::vector<metal_complex64> w(s.npl*s.bands),p(s.npro*s.bands),m(o->use_overlap ? s.npro*s.bands:0);
   std::vector<double> scale(s.bands);
   for(size_t b=0;b<s.bands;++b) {
     auto factor=static_cast<const vector_float2 *>(o->scale.contents)[b];scale[b]=double(factor.x)+factor.y;
     if(!std::isfinite(scale[b]) || scale[b]<=0) {o->valid=false;metal_backend_set_external_success(o->backend,0);return METAL_BACKEND_RUNTIME_FAILED;}
   }
   for(size_t i=0;i<w.size();++i) {w[i]=join(static_cast<const vector_float4 *>(o->wave.contents)[i]);if(!finite(w[i])) {o->valid=false;metal_backend_set_external_success(o->backend,0);return METAL_BACKEND_RUNTIME_FAILED;}}
   for(size_t i=0;i<p.size();++i) {p[i]=join(static_cast<const vector_float4 *>(o->projection.contents)[i]);if(!finite(p[i])) {o->valid=false;metal_backend_set_external_success(o->backend,0);return METAL_BACKEND_RUNTIME_FAILED;}}
   for(size_t i=0;i<m.size();++i) {m[i]=join(static_cast<const vector_float4 *>(o->metric_projection.contents)[i]);if(!finite(m[i])) {o->valid=false;metal_backend_set_external_success(o->backend,0);return METAL_BACKEND_RUNTIME_FAILED;}}
   // This is the sole host consumer boundary. No fallible operation follows.
   for(size_t b=0;b<s.bands;++b) {
     std::copy_n(w.data()+b*s.npl,s.npl,wave[b]);std::copy_n(p.data()+b*s.npro,s.npro,projectors[b]);
     if(o->use_overlap) std::copy_n(m.data()+b*s.npro,s.npro,metric+b*ldmetric);
     scales[b]=scale[b];
   }
   o->stats.host_export_bytes+=w.size()*sizeof(metal_complex64)+p.size()*sizeof(metal_complex64)+m.size()*sizeof(metal_complex64)+scale.size()*sizeof(double);
   o->exported_generation=generation;
   return METAL_BACKEND_OK;
 } catch(const std::bad_alloc &) {return METAL_BACKEND_ALLOCATION_FAILED;}
}
extern "C" void metal_wave_owner_get_stats(const metal_wave_owner *o,metal_wave_owner_stats *out) {
 if(!out) return;*out={};if(!o) return;std::lock_guard<std::mutex> guard(o->mutex);*out=o->stats;
}
