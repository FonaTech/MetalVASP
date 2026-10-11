#pragma once
#include "metal_nonlocal.h"
#include "metal_nonlocal_policy.hpp"
#include <unordered_map>
#include <cerrno>
struct metal_nonlocal_batch {
    const metal_backend *backend=nullptr;
    metal_nonlocal_shape shape{};
    id<MTLComputePipelineState> kernel=nil;
    id<MTLBuffer> wave=nil,radial=nil,phase=nil,cq=nil,site=nil,gradient=nil,result=nil;
    uint64_t wave_generation=0,phase_generation=0;
    size_t phase_first=SIZE_MAX,phase_sites=0,phase_g=SIZE_MAX,phase_ng=0;
    bool wave_full=false;
    mutable std::mutex mutex;
    metal_nonlocal_stats stats{};
};
namespace nonlocal_impl {
struct Site {uint64_t ion,type,l;};
struct Params {uint64_t ng,ns,nb,nq,wave_stride,wave_g,wave_band;uint32_t channels,first_g;vector_float4 multiplier;};
static_assert(sizeof(Params)==80 && sizeof(Site)==24,"Host/MSL64-bit layout");
inline vector_float2 split(double x) {float hi=float(x);return {hi,float(x-double(hi))};}
inline vector_float4 split(metal_complex64 z) {auto r=split(z.real),i=split(z.imag);return {r.x,r.y,i.x,i.y};}
inline metal_complex64 join(vector_float4 z) {return {double(z.x)+z.y,double(z.z)+z.w};}
inline bool finite(metal_complex64 z) {return std::isfinite(z.real)&&std::isfinite(z.imag);}
inline bool representable(double x) {return std::isfinite(x) && std::abs(x)<=double(FLT_MAX);}
inline bool representable(metal_complex64 z) {return representable(z.real)&&representable(z.imag);}
inline size_t setting(const char *name,size_t fallback) {
 const char *s=std::getenv(name);if(!s || *s<'0' || *s>'9') return fallback;
 errno=0;char *end=nullptr;auto value=std::strtoull(s,&end,10);
 return !errno && end && !*end && value && value<=SIZE_MAX ? size_t(value):fallback;
}
inline bool ensure(metal_nonlocal_batch *o,id<MTLBuffer> __strong &buffer,size_t bytes) {
 if(buffer && buffer.length>=bytes) return true;
 buffer=nil;
 buffer=metal_memory::allocate(o->backend->device,bytes,MTLResourceStorageModeShared,
   [](const void *b){metal_backend_evict_idle_buffers(static_cast<const metal_backend *>(b));},o->backend);
 return buffer!=nil;
}
}
extern "C" int metal_nonlocal_create(const metal_backend *b,metal_nonlocal_shape s,metal_nonlocal_batch **out) {
 if(!out) return METAL_BACKEND_INVALID_ARGUMENT;*out=nullptr;
 size_t value;
 using namespace metal_nonlocal_policy;
 if(!b || !b->use_gpu || !s.npl || !s.bands || !s.nions || !s.npro || !s.qchannels || !s.ntypes ||
    !multiply(s.npl,s.bands,value) || !multiply(value,16,value) ||
    !multiply(s.npro,s.bands,value) || !multiply(value,4*16,value) ||
    !multiply(s.qchannels,s.ntypes,value) || !multiply(value,s.npl,value) || !multiply(value,8,value))
   return METAL_BACKEND_INVALID_ARGUMENT;
 auto *o=new(std::nothrow) metal_nonlocal_batch;if(!o) return METAL_BACKEND_ALLOCATION_FAILED;
 o->backend=b;o->shape=s;o->kernel=b->nonlocal_project;
 if(!o->kernel || o->kernel.maxTotalThreadsPerThreadgroup<256) {delete o;return METAL_BACKEND_UNSUPPORTED;}
 *out=o;return METAL_BACKEND_OK;
}
extern "C" void metal_nonlocal_destroy(metal_nonlocal_batch *o) {delete o;}
extern "C" void metal_nonlocal_get_stats(const metal_nonlocal_batch *o,metal_nonlocal_stats *out) {
 if(!out)return;*out={};if(o) {std::lock_guard<std::mutex> lock(o->mutex);*out=o->stats;}
}
extern "C" int metal_nonlocal_project(metal_nonlocal_batch *o,const metal_nonlocal_input *in,unsigned channels,metal_complex64 *output,size_t ld) {
 using namespace nonlocal_impl;using namespace metal_nonlocal_policy;
 if(!o)return METAL_BACKEND_INVALID_ARGUMENT;
 std::lock_guard<std::mutex> lock(o->mutex);const auto s=o->shape;const auto *b=o->backend;
 metal_backend_set_external_success(b,0);++o->stats.calls;
 size_t radial_columns,out_columns,full_wave_bytes;
 if(!in || (channels!=1 && channels!=4) || !output || !in->wave || !in->radial || !in->phase || !in->cq ||
    !in->types || !in->lmmax || !in->bases || !in->wave_generation || !in->phase_generation ||
    (channels==4 && (!in->gradient || !representable(in->derivative_multiplier))) ||
    !multiply(s.qchannels,s.ntypes,radial_columns) || !multiply(s.bands,channels,out_columns) ||
    !span(s.npl,s.bands,in->ldw,16) || !span(s.npl,radial_columns,in->qld,8) ||
    !span(s.npl,s.nions,in->phase_ld,16) || !span(s.npro,out_columns,ld,16) ||
    !multiply(s.npl,s.bands,full_wave_bytes) || !multiply(full_wave_bytes,16,full_wave_bytes)) return METAL_BACKEND_INVALID_ARGUMENT;
 if(in->bases[0]!=0 || in->bases[s.nions]<0 || uint64_t(in->bases[s.nions])!=s.npro) return METAL_BACKEND_INVALID_ARGUMENT;
 try {
  std::vector<Site> all_sites(s.npro);
  for(size_t ion=0;ion<s.nions;++ion) {
   int type=in->types[ion];if(type<1 || size_t(type)>s.ntypes)return METAL_BACKEND_INVALID_ARGUMENT;
   int count=in->lmmax[type-1];if(count<0 || size_t(count)>s.qchannels || in->bases[ion]<0 ||
      in->bases[ion+1]<in->bases[ion] || uint64_t(in->bases[ion+1]-in->bases[ion])!=uint64_t(count))return METAL_BACKEND_INVALID_ARGUMENT;
   for(int l=0;l<count;++l)all_sites[size_t(in->bases[ion])+size_t(l)]={ion,uint64_t(type-1),uint64_t(l)};
  }
  std::vector<metal_complex64> staged(s.npro*out_columns);
  const size_t available=metal_memory::budget(b->device)>b->device.currentAllocatedSize ?
     metal_memory::budget(b->device)-b->device.currentAllocatedSize:0;
  const size_t workspace=std::min(setting("VASP_METAL_NONLOCAL_WORKSPACE_BYTES",size_t(2)*1024*1024*1024),available);
  if(workspace<4096)return METAL_BACKEND_ALLOCATION_FAILED;
  const size_t forced_g=setting("VASP_METAL_NONLOCAL_G_WINDOW",s.npl);
  size_t site_window=std::min({s.npro,size_t(16384),setting("VASP_METAL_NONLOCAL_SITE_WINDOW",s.npro)});
  size_t band_window=std::min(s.bands,setting("VASP_METAL_NONLOCAL_BAND_WINDOW",32));
  // Cache the complete wave only when the scoped budget and driver limit permit.
  // Otherwise every G/band window is imported explicitly; no logical size cap.
  bool cache_wave=forced_g>=s.npl && full_wave_bytes<=workspace/2 && full_wave_bytes<=b->device.maxBufferLength;
  if(o->wave_full!=cache_wave) {o->wave=nil;o->wave_generation=0;o->wave_full=cache_wave;}
  if(cache_wave) {
   if(!ensure(o,o->wave,full_wave_bytes))return METAL_BACKEND_ALLOCATION_FAILED;
   if(o->wave_generation!=in->wave_generation) {
    ConversionTimer timer{b};auto *dst=(vector_float4 *)o->wave.contents;
    for(size_t band=0;band<s.bands;++band)for(size_t g=0;g<s.npl;++g) {
     auto z=in->wave[g+in->ldw*band];if(!representable(z)) {o->wave_generation=0;return METAL_BACKEND_INVALID_ARGUMENT;}
     dst[g+s.npl*band]=split(z);
    }
    o->wave_generation=in->wave_generation;o->stats.wave_import_bytes+=full_wave_bytes;
   } else ++o->stats.wave_reuses;
  }
  const size_t tile_budget=workspace-(cache_wave ? full_wave_bytes:0);
  for(size_t first=0;first<s.npro;) {
   size_t ns=std::min(site_window,s.npro-first),nb=std::min(band_window,s.bands);
   // Reduce result/site windows until metadata and at least oneG fit.
   std::vector<uint64_t> ions,types;std::vector<Site> local;
   size_t g_window=0,per_g=0,result_bytes=0,metadata_bytes=0;
   while(true) {
    ions.clear();types.clear();local.clear();std::unordered_map<uint64_t,uint64_t> im,tm;
    for(size_t j=0;j<ns;++j) {
     auto x=all_sites[first+j];
     if(!im.count(x.ion)) {im[x.ion]=ions.size();ions.push_back(x.ion);}
     if(!tm.count(x.type)) {tm[x.type]=types.size();types.push_back(x.type);}
     local.push_back({im[x.ion],tm[x.type],x.l});
    }
    nb=bands_per_dispatch(ns,nb);
    size_t value;
    if(!nb || !multiply(ns,nb,value) || !multiply(value,channels*16,result_bytes) ||
       !multiply(types.size(),s.qchannels,value) || !multiply(value,16,value) ||
       ns>SIZE_MAX/sizeof(Site) || value>SIZE_MAX-ns*sizeof(Site))return METAL_BACKEND_INVALID_ARGUMENT;
    metadata_bytes=ns*sizeof(Site)+value;
    // Types/ions are bounded by the site window, including arbitrarily many types.
    const size_t qbytes=types.size()*s.qchannels*8;
    if(ions.size()>SIZE_MAX/16 || qbytes>SIZE_MAX-ions.size()*16-24-nb*16)return METAL_BACKEND_INVALID_ARGUMENT;
    per_g=qbytes+ions.size()*16+(channels==4 ? 24:0)+(cache_wave ? 0:nb*16);
    if(result_bytes<b->device.maxBufferLength && metadata_bytes+8<tile_budget && result_bytes<tile_budget-metadata_bytes-8 &&
       per_g && per_g<=tile_budget-metadata_bytes-result_bytes-8) {
     g_window=std::min({s.npl,forced_g,(tile_budget-metadata_bytes-result_bytes-8)/per_g,
                       size_t(b->device.maxBufferLength)/std::max({qbytes,ions.size()*16,nb*16,size_t(24)})});
     if(g_window)break;
    }
    if(ns>1)ns=(ns+1)/2;else if(nb>1)nb=(nb+1)/2;else return METAL_BACKEND_ALLOCATION_FAILED;
   }
   const size_t ngmax=g_window;
   if(!ensure(o,o->radial,ngmax*types.size()*s.qchannels*8) || !ensure(o,o->phase,ngmax*ions.size()*16) ||
      !ensure(o,o->cq,types.size()*s.qchannels*16) || !ensure(o,o->site,ns*sizeof(Site)) ||
      !ensure(o,o->gradient,channels==4 ? ngmax*24:size_t(8)) || !ensure(o,o->result,result_bytes) ||
      (!cache_wave && !ensure(o,o->wave,ngmax*nb*16))) return METAL_BACKEND_ALLOCATION_FAILED;
   o->stats.planned_workspace_peak_bytes=std::max<uint64_t>(o->stats.planned_workspace_peak_bytes,
        metadata_bytes+8+result_bytes+ngmax*per_g+(cache_wave ? full_wave_bytes:0));
   {ConversionTimer timer{b};std::memcpy(o->site.contents,local.data(),ns*sizeof(Site));
    auto *dst=(vector_float4 *)o->cq.contents;
    for(size_t t=0;t<types.size();++t)for(size_t l=0;l<s.qchannels;++l) {
     auto z=l<size_t(in->lmmax[types[t]]) ? in->cq[l+s.qchannels*types[t]]:metal_complex64{0,0};if(!representable(z))return METAL_BACKEND_INVALID_ARGUMENT;
     dst[l+s.qchannels*t]=split(z);
    }
    o->stats.metadata_import_bytes+=metadata_bytes;
   }
   for(size_t band_first=0;band_first<s.bands;band_first+=nb) {
    const size_t bands=std::min(nb,s.bands-band_first);
    for(size_t g_first=0;g_first<s.npl;g_first+=ngmax) {
     const size_t ng=std::min(ngmax,s.npl-g_first);
     {ConversionTimer timer{b};
      if(!cache_wave) {
       auto *dst=(vector_float4 *)o->wave.contents;
       for(size_t band=0;band<bands;++band)for(size_t g=0;g<ng;++g) {
        auto z=in->wave[g_first+g+in->ldw*(band_first+band)];if(!representable(z))return METAL_BACKEND_INVALID_ARGUMENT;
        dst[g+ng*band]=split(z);
       }
       o->stats.wave_import_bytes+=ng*bands*16;
      }
      auto *rad=(vector_float2 *)o->radial.contents;
      for(size_t t=0;t<types.size();++t)for(size_t l=0;l<s.qchannels;++l)for(size_t g=0;g<ng;++g) {
       double z=l<size_t(in->lmmax[types[t]]) ? in->radial[g_first+g+in->qld*(l+s.qchannels*types[t])]:0;if(!representable(z))return METAL_BACKEND_INVALID_ARGUMENT;
       rad[g+ng*(l+s.qchannels*t)]=split(z);
      }
      o->stats.radial_import_bytes+=ng*types.size()*s.qchannels*8;
      const bool phase_reuse=o->phase_generation==in->phase_generation && o->phase_first==first &&
          o->phase_sites==ns && o->phase_g==g_first && o->phase_ng==ng;
      if(!phase_reuse) {
       auto *dst=(vector_float4 *)o->phase.contents;
       o->phase_generation=0;
       for(size_t ion=0;ion<ions.size();++ion)for(size_t g=0;g<ng;++g) {
        auto z=in->phase[g_first+g+in->phase_ld*ions[ion]];if(!representable(z))return METAL_BACKEND_INVALID_ARGUMENT;
        dst[g+ng*ion]=split(z);
       }
       o->phase_generation=in->phase_generation;o->phase_first=first;o->phase_sites=ns;o->phase_g=g_first;o->phase_ng=ng;
       o->stats.phase_import_bytes+=ng*ions.size()*16;
      } else ++o->stats.phase_reuses;
      if(channels==4) {
       auto *dst=(vector_float2 *)o->gradient.contents;
       for(size_t c=0;c<3;++c)for(size_t g=0;g<ng;++g) {
        double z=in->gradient[g_first+g+s.npl*c];if(!representable(z))return METAL_BACKEND_INVALID_ARGUMENT;
        dst[g+ng*c]=split(z);
       }
       o->stats.gradient_import_bytes+=ng*3*8;
      }
     }
     Params p{ng,ns,bands,s.qchannels,cache_wave ? s.npl:ng,cache_wave ? g_first:0,
              cache_wave ? band_first:0,channels,g_first==0 ? 1u:0u,split(in->derivative_multiplier)};
     id<MTLCommandBuffer> command=[b->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
     if(!command || !enc)return METAL_BACKEND_RUNTIME_FAILED;
     [enc setComputePipelineState:o->kernel];id<MTLBuffer> buffers[]={o->wave,o->radial,o->phase,o->cq,o->site,o->gradient,o->result};
     for(NSUInteger at=0;at<7;++at)[enc setBuffer:buffers[at] offset:0 atIndex:at];
     [enc setBytes:&p length:sizeof(p) atIndex:7];
     [enc dispatchThreadgroups:MTLSizeMake(ns*bands,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
     [enc endEncoding];int rc=metal_backend_complete_external_command(b,(__bridge void *)command);if(rc)return rc;
     ++o->stats.completed_commands;++o->stats.windows;
     if(setting("VASP_METAL_NONLOCAL_FAIL_WINDOW",SIZE_MAX)==o->stats.windows) {metal_backend_set_external_success(b,0);return METAL_BACKEND_RUNTIME_FAILED;}
    }
    {ConversionTimer timer{b};const auto *src=(const vector_float4 *)o->result.contents;
     for(size_t c=0;c<channels;++c)for(size_t band=0;band<bands;++band)for(size_t j=0;j<ns;++j) {
      auto z=join(src[j+ns*(band+bands*c)]);if(!finite(z)) {metal_backend_set_external_success(b,0);return METAL_BACKEND_RUNTIME_FAILED;}
      staged[first+j+s.npro*(band_first+band+s.bands*c)]=z;
     }
    }
   }
   first+=ns;
  }
  // Sole output commit; no allocation, validation or GPU operation follows.
  for(size_t c=0;c<channels;++c)for(size_t band=0;band<s.bands;++band)
    std::copy_n(staged.data()+s.npro*(band+s.bands*c),s.npro,output+ld*(band+s.bands*c));
  o->stats.export_bytes+=s.npro*out_columns*16;metal_backend_set_external_success(b,1);return METAL_BACKEND_OK;
 } catch(const std::bad_alloc &) {metal_backend_set_external_success(b,0);return METAL_BACKEND_ALLOCATION_FAILED;}
}
