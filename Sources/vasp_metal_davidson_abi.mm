// Independent opt-in EDDAV closure. The established BLAS/FFT ABI is untouched.
#include "metal_wave_owner_davidson.h"
#include "vasp_metal_davidson_abi.h"
#include "metal_stats_path.hpp"
#include "metal_backend_private.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include <limits>
namespace {
struct Lease {
 metal_wave_owner *wave=nullptr;uint64_t generation=0,client_generation=0;
 uint64_t checked_epoch=0,static_generation=0,basis_generation=0,client_basis_epoch=0;
 std::string key;std::vector<unsigned char> static_snapshot;
 ~Lease(){metal_wave_owner_destroy(wave);}
};
struct State {
 std::mutex mutex;metal_backend *backend=nullptr;bool initialized=false;
 std::vector<std::unique_ptr<Lease>> cached;
 uint64_t paw_epoch=1,static_uploads=0,static_reuses=0,static_validation_bytes=0;
 uint64_t calls=0,success=0,allocations=0,reuses=0,releases=0,live=0;
 metal_wave_basis_transfer_stats basis_transfers{};
 uint64_t imports=0,exports=0,basis_imports=0,peak_bytes=0,control_status_host_bytes=0,overlap_zero_init_host_bytes=0;
 ~State(){cached.clear();metal_backend_destroy(backend);}
};
State &state(){static State s;return s;}
bool enabled(const char *name,bool fallback=false) {
 auto value=std::getenv(name);return value ? std::strcmp(value,"0") && std::strcmp(value,"off"):fallback;
}
uint64_t cache_limit() {
 auto value=std::getenv("VASP_METAL_DAVIDSON_CACHE_BYTES");
 if(!value || !*value) {
   const uint64_t mib=1024*1024,working=metal_backend_recommended_working_set_bytes(state().backend);
   return working ? std::min(uint64_t(512)*mib,std::max(uint64_t(256)*mib,working/64)):uint64_t(256)*mib;
 }
 long long bytes=0;return metal_stats::nonnegative(value,bytes) ? uint64_t(bytes):0;
}
std::string layout() {
 const auto mpi=metal_stats::mpi_identity();return std::to_string(mpi.rank)+":"+std::to_string(mpi.size);
}
uint64_t retained_bytes(const Lease *lease) {
 metal_wave_owner_stats stats{};metal_wave_owner_get_stats(lease->wave,&stats);
 return stats.device_bytes+lease->static_snapshot.size();
}
void release_locked(State &s,void **handle) {
 if(!handle || !*handle) return;std::unique_ptr<Lease> lease(static_cast<Lease *>(*handle));*handle=nullptr;
 metal_wave_owner_davidson_forget_outputs(lease->wave);lease->client_generation=0;lease->client_basis_epoch=0;
 ++s.releases;if(s.live) --s.live;
 const auto limit=cache_limit(),bytes=retained_bytes(lease.get());
 uint64_t cached=0;for(const auto &entry:s.cached) cached+=retained_bytes(entry.get());
 while(!s.cached.empty() && cached>limit) {cached-=retained_bytes(s.cached.front().get());s.cached.erase(s.cached.begin());}
 if(bytes>limit) return;
 while(!s.cached.empty() && cached>limit-bytes) {cached-=retained_bytes(s.cached.front().get());s.cached.erase(s.cached.begin());}
 try {s.cached.push_back(std::move(lease));} catch(const std::bad_alloc &) {}
}
bool refresh_static(State &s,Lease *lease,uint64_t epoch,const metal_wave_owner_shape &shape,const metal_wave_prepare_input &input) {
 if(lease->checked_epoch==epoch && !lease->static_snapshot.empty()) return true;
 // Compare exact source bits at an explicit scientific invalidation boundary.
 // This bounded shadow is counted alongside retained device allocations.
 std::vector<unsigned char> snapshot;
 auto append=[&](const void *data,size_t bytes) {
   if(bytes>snapshot.max_size()-snapshot.size()) throw std::bad_alloc();
   auto p=static_cast<const unsigned char *>(data);snapshot.insert(snapshot.end(),p,p+bytes);
 };
 if(!input.types || !input.lmmax || !input.bases || !input.radial || !input.cq || !input.phase) return false;
 for(size_t ion=0;ion<shape.nions;++ion) {
   int type=input.types[ion];if(type<1 || size_t(type)>shape.ntypes) return false;
   append(&type,sizeof(type));append(input.lmmax+type-1,sizeof(int));
 }
 append(input.bases,(shape.nions+1)*sizeof(int));
 for(size_t c=0;c<shape.qchannels*shape.ntypes;++c) append(input.radial+c*input.qld,shape.npl*sizeof(double));
 append(input.cq,shape.qchannels*shape.ntypes*sizeof(metal_complex64));
 for(size_t ion=0;ion<shape.nions;++ion) append(input.phase+ion*input.phase_ld,shape.npl*sizeof(metal_complex64));
 s.static_validation_bytes+=snapshot.size();
 if(snapshot!=lease->static_snapshot) {
   if(lease->static_generation==UINT64_MAX) return false;
   ++lease->static_generation;lease->static_snapshot.swap(snapshot);
 }
 lease->checked_epoch=epoch;return true;
}
void report() {
 auto &s=state();std::lock_guard<std::mutex> lock(s.mutex);
 const char *configured=std::getenv("VASP_METAL_DAVIDSON_STATS");
 std::string path;
 if(configured && *configured) path=configured;
 else if(auto original=std::getenv("VASP_METAL_STATS")) path=std::string(original)+".davidson.json";
 if(path.empty()) return;
 const auto mpi=metal_stats::mpi_identity();path=metal_stats::path(path.c_str(),mpi);
 uint64_t retained=0;for(const auto &entry:s.cached) retained+=retained_bytes(entry.get());
 std::ofstream out(path);if(!out) return;
 out << "{\"davidson_owner_calls\":" << s.calls << ",\"davidson_owner_success\":" << s.success
     << ",\"davidson_owner_allocations\":" << s.allocations << ",\"davidson_owner_cache_reuses\":" << s.reuses
     << ",\"davidson_owner_releases\":" << s.releases << ",\"davidson_owner_live\":" << s.live
     << ",\"davidson_owner_host_import_bytes\":" << s.imports << ",\"davidson_owner_host_export_bytes\":" << s.exports
     << ",\"davidson_owner_basis_import_bytes\":" << s.basis_imports
     << ",\"davidson_owner_basis_full_uploads\":" << s.basis_transfers.full_uploads
     << ",\"davidson_owner_basis_partial_uploads\":" << s.basis_transfers.partial_uploads
     << ",\"davidson_owner_basis_reuses\":" << s.basis_transfers.reuses
     << ",\"davidson_owner_basis_uploaded_columns\":" << s.basis_transfers.uploaded_columns << ",\"davidson_owner_intermediate_host_bytes\":0"
     << ",\"davidson_owner_peak_bytes\":" << s.peak_bytes
     << ",\"davidson_owner_paw_static_uploads\":" << s.static_uploads
     << ",\"davidson_owner_paw_static_reuses\":" << s.static_reuses
     << ",\"davidson_owner_cached_entries\":" << s.cached.size()
     << ",\"davidson_owner_retained_cache_bytes\":" << retained
     << ",\"davidson_owner_static_validation_host_bytes\":" << s.static_validation_bytes
     << ",\"davidson_owner_control_status_host_bytes\":" << s.control_status_host_bytes
     << ",\"davidson_owner_overlap_zero_init_host_bytes\":" << s.overlap_zero_init_host_bytes
     << ",\"davidson_owner_gpu_submissions\":" << metal_backend_command_count(s.backend)
     << ",\"davidson_owner_gpu_seconds\":" << metal_backend_gpu_seconds(s.backend)
     << ",\"rank\":" << mpi.rank << ",\"world_size\":" << mpi.size << ",\"local_only_reductions\":1}\n";
}
}
extern "C" void vasp_metal_davidson_release(void **handle) {
 auto &s=state();std::lock_guard<std::mutex> lock(s.mutex);
 release_locked(s,handle);
}
extern "C" void vasp_metal_davidson_invalidate() {
 auto &s=state();std::lock_guard<std::mutex> lock(s.mutex);
 if(s.paw_epoch!=UINT64_MAX) ++s.paw_epoch;
 else {s.cached.clear();s.paw_epoch=1;}
}
extern "C" void vasp_metal_davidson_get_counters(vasp_metal_davidson_counters *out) {
 if(!out) return;auto &s=state();std::lock_guard<std::mutex> lock(s.mutex);uint64_t retained=0;
 for(const auto &entry:s.cached) retained+=retained_bytes(entry.get());
 *out={s.calls,s.success,s.allocations,s.reuses,s.releases,s.live,s.imports,s.exports,s.basis_imports,
   s.static_uploads,s.static_reuses,s.cached.size(),retained,s.static_validation_bytes,metal_backend_command_count(s.backend),s.control_status_host_bytes,s.overlap_zero_init_host_bytes};
}
static int prepare_impl(void **handle,uint64_t client_generation,int npl,int bands,int nions,int npro,
 int lm,int ntypes,int qchannels,int qld,int phase_ld,int real_metric,int overlap,metal_complex64 *const *input_wave,
 metal_complex64 *const *unused_input_projectors,const double *precon,int ldprecon,const int *types,const int *lmmax,const int *bases,
 const double *radial,const metal_complex64 *cq,const metal_complex64 *phase,const void *metric,
 metal_complex64 *metric_projection,int ldmetric,double *scales,int basisbands,
 const metal_complex64 *basiswave,const metal_complex64 *basisprojectors,int ldbasiswave,int ldbasisprojectors,
 metal_complex64 *const *output_wave,metal_complex64 *const *output_projectors,int kpoint,int spin,int band_layout,
 uint64_t basis_epoch,int dirty_first,int dirty_count) {
 (void)unused_input_projectors;
 auto &s=state();std::lock_guard<std::mutex> lock(s.mutex);++s.calls;
 if(!enabled("VASP_METAL") || !enabled("VASP_METAL_WAVE_OWNER",true) || !enabled("VASP_METAL_DAVIDSON_OWNER") ||
    !handle || !client_generation || npl<=0 || bands<=0 || nions<=0 || npro<=0 || lm<=0 || ntypes<=0 || qchannels<=0 ||
    qld<npl || phase_ld<npl || ldprecon<npl || ldmetric<0 || basisbands<=0 || ldbasiswave<npl || ldbasisprojectors<npro ||
    kpoint<=0 || spin<=0 || band_layout<=0 || dirty_first<0 || dirty_count<0 ||
    dirty_first>basisbands || dirty_count>basisbands-dirty_first) return 0;
 try {
 if(!s.initialized) {
   s.initialized=true;metal_backend_create(&s.backend,1);std::atexit(report);
 }
 if(!s.backend || !metal_backend_gpu_enabled(s.backend)) return 0;
 metal_wave_owner_shape shape{size_t(npl),size_t(bands),size_t(nions),size_t(npro),size_t(lm),size_t(ntypes),size_t(qchannels)};
 const std::string key=std::to_string(kpoint)+":"+std::to_string(spin)+":"+std::to_string(band_layout)+":"+layout();
 auto *lease=static_cast<Lease *>(*handle);
 if(lease && (lease->key!=key || !metal_wave_owner_matches(lease->wave,shape))) {
   release_locked(s,handle);lease=nullptr;
 }
 if(!lease) {
   std::unique_ptr<Lease> fresh;
   for(auto it=s.cached.begin();it!=s.cached.end();++it) if((*it)->key==key && metal_wave_owner_matches((*it)->wave,shape)) {
     fresh=std::move(*it);s.cached.erase(it);++s.reuses;break;
   }
   if(!fresh) {
     fresh.reset(new(std::nothrow) Lease);if(!fresh || metal_wave_owner_create(s.backend,shape,&fresh->wave)) return 0;
     fresh->key=key;
     ++s.allocations;
   }
   lease=fresh.release();*handle=lease;++s.live;
 }
 if(client_generation<=lease->client_generation || lease->generation==UINT64_MAX) {metal_wave_owner_invalidate(lease->wave);return 0;}
 lease->client_generation=client_generation;const auto generation=++lease->generation;
 metal_wave_basis_delta delta{};
 if(lease->basis_generation==UINT64_MAX) {metal_wave_owner_invalidate(lease->wave);return 0;}
 if(!basis_epoch) {++lease->basis_generation;lease->client_basis_epoch=0;}
 else if(!lease->client_basis_epoch) {++lease->basis_generation;lease->client_basis_epoch=basis_epoch;}
 else if(basis_epoch==lease->client_basis_epoch) {
   // The last dirty span may still be supplied; its epoch was already applied.
 } else if(basis_epoch>lease->client_basis_epoch) {
   if(lease->client_basis_epoch!=UINT64_MAX && basis_epoch==lease->client_basis_epoch+1) {
     delta.previous_generation=lease->basis_generation;
     delta.first_column=size_t(dirty_first);delta.column_count=size_t(dirty_count);
   }
   ++lease->basis_generation;lease->client_basis_epoch=basis_epoch;
 } else {metal_wave_owner_invalidate(lease->wave);lease->client_basis_epoch=0;return 0;}
 metal_wave_owner_stats before{},after{};metal_wave_davidson_stats dbefore{},dafter{};
 metal_wave_basis_transfer_stats tbefore{},tafter{};
 metal_wave_owner_davidson_get_transfer_stats(lease->wave,&tbefore);
 metal_wave_owner_get_stats(lease->wave,&before);metal_wave_owner_davidson_get_stats(lease->wave,&dbefore);
 metal_wave_prepare_input input{reinterpret_cast<const metal_complex64 *const *>(input_wave),precon,size_t(ldprecon),size_t(qld),size_t(phase_ld),
   types,lmmax,bases,radial,cq,phase,metric,real_metric,overlap};
 if(input.qld>SIZE_MAX/(shape.qchannels*shape.ntypes)/sizeof(double) ||
    input.phase_ld>SIZE_MAX/shape.nions/sizeof(metal_complex64)) {metal_wave_owner_invalidate(lease->wave);return 0;}
 int rc=METAL_BACKEND_INVALID_ARGUMENT;
 metal_wave_basis_input basis{size_t(basisbands),size_t(ldbasiswave),size_t(ldbasisprojectors),basiswave,basisprojectors,lease->basis_generation};
 try {
   if(refresh_static(s,lease,s.paw_epoch,shape,input)) {
     if(basis_epoch) rc=metal_wave_owner_davidson_prepare_delta(lease->wave,generation,lease->static_generation,&input,&basis,&delta);
     else rc=metal_wave_owner_davidson_prepare(lease->wave,generation,lease->static_generation,&input,&basis);
   }
 } catch(const std::bad_alloc &) {rc=METAL_BACKEND_ALLOCATION_FAILED;}
 if(!rc) rc=metal_wave_owner_download(lease->wave,generation,output_wave,output_projectors,metric_projection,size_t(ldmetric),scales);
 metal_wave_owner_get_stats(lease->wave,&after);metal_wave_owner_davidson_get_stats(lease->wave,&dafter);
 metal_wave_owner_davidson_get_transfer_stats(lease->wave,&tafter);
 s.basis_transfers.full_uploads+=tafter.full_uploads-tbefore.full_uploads;
 s.basis_transfers.partial_uploads+=tafter.partial_uploads-tbefore.partial_uploads;
 s.basis_transfers.reuses+=tafter.reuses-tbefore.reuses;
 s.basis_transfers.uploaded_columns+=tafter.uploaded_columns-tbefore.uploaded_columns;
 s.imports+=after.host_import_bytes-before.host_import_bytes;s.exports+=after.host_export_bytes-before.host_export_bytes;
 s.basis_imports+=dafter.basis_import_bytes-dbefore.basis_import_bytes;s.peak_bytes=std::max(s.peak_bytes,after.device_bytes);
 s.static_uploads+=after.static_uploads-before.static_uploads;s.static_reuses+=after.static_reuses-before.static_reuses;
 s.control_status_host_bytes+=dafter.control_status_host_bytes-dbefore.control_status_host_bytes;
 s.overlap_zero_init_host_bytes+=dafter.overlap_zero_init_host_bytes-dbefore.overlap_zero_init_host_bytes;
 if(rc) {metal_wave_owner_invalidate(lease->wave);lease->checked_epoch=0;return 0;}
 ++s.success;return 1;
 } catch(const std::bad_alloc &) {
   if(handle && *handle) metal_wave_owner_invalidate(static_cast<Lease *>(*handle)->wave);
   return 0;
 }
}

extern "C" int vasp_metal_davidson_prepare(void **handle,uint64_t client_generation,int npl,int bands,int nions,int npro,
 int lm,int ntypes,int qchannels,int qld,int phase_ld,int real_metric,int overlap,metal_complex64 *const *input_wave,
 metal_complex64 *const *unused_input_projectors,const double *precon,int ldprecon,const int *types,const int *lmmax,const int *bases,
 const double *radial,const metal_complex64 *cq,const metal_complex64 *phase,const void *metric,
 metal_complex64 *metric_projection,int ldmetric,double *scales,int basisbands,
 const metal_complex64 *basiswave,const metal_complex64 *basisprojectors,int ldbasiswave,int ldbasisprojectors,
 metal_complex64 *const *output_wave,metal_complex64 *const *output_projectors,int kpoint,int spin,int band_layout) {
 return prepare_impl(handle,client_generation,npl,bands,nions,npro,lm,ntypes,qchannels,qld,phase_ld,real_metric,overlap,
 input_wave,unused_input_projectors,precon,ldprecon,types,lmmax,bases,radial,cq,phase,metric,metric_projection,ldmetric,scales,
 basisbands,basiswave,basisprojectors,ldbasiswave,ldbasisprojectors,output_wave,output_projectors,kpoint,spin,band_layout,0,0,0);
}

extern "C" int vasp_metal_davidson_prepare_delta(void **handle,uint64_t client_generation,int npl,int bands,int nions,int npro,
 int lm,int ntypes,int qchannels,int qld,int phase_ld,int real_metric,int overlap,metal_complex64 *const *input_wave,
 metal_complex64 *const *unused_input_projectors,const double *precon,int ldprecon,const int *types,const int *lmmax,const int *bases,
 const double *radial,const metal_complex64 *cq,const metal_complex64 *phase,const void *metric,
 metal_complex64 *metric_projection,int ldmetric,double *scales,int basisbands,
 const metal_complex64 *basiswave,const metal_complex64 *basisprojectors,int ldbasiswave,int ldbasisprojectors,
 metal_complex64 *const *output_wave,metal_complex64 *const *output_projectors,int kpoint,int spin,int band_layout,
 uint64_t basis_epoch,int dirty_first,int dirty_count) {
 if(!basis_epoch) return 0;
 return prepare_impl(handle,client_generation,npl,bands,nions,npro,lm,ntypes,qchannels,qld,phase_ld,real_metric,overlap,
 input_wave,unused_input_projectors,precon,ldprecon,types,lmmax,bases,radial,cq,phase,metric,metric_projection,ldmetric,scales,
 basisbands,basiswave,basisprojectors,ldbasiswave,ldbasisprojectors,output_wave,output_projectors,kpoint,spin,band_layout,basis_epoch,dirty_first,dirty_count);
}

extern "C" void vasp_metal_davidson_get_basis_counters(metal_wave_basis_transfer_stats *out) {
 if(!out) return;auto &s=state();std::lock_guard<std::mutex> lock(s.mutex);*out=s.basis_transfers;
}
