#include <time.h>
#include <sstream>
#include <climits>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_backend.h"
#include "metal_nonlocal.h"
#include "metal_eigh_policy.hpp"
#include "metal_chunk_policy.hpp"
#include "metal_gemm_policy.hpp"
#include "metal_wave_owner.h"
#include "metal_backend_private.h"
#include "metal_stats_path.hpp"
#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>

namespace {
static_assert(sizeof(blasint)==sizeof(int), "MetalVASP BLAS interception requires LP64 integers");
struct fcomplex { double real; double imag; };
struct ApiProfile {
    unsigned long long calls=0;
    double process_cpu_seconds=0,gpu_command_seconds=0,wait_seconds=0,conversion_seconds=0;
};
double process_cpu_time() {
    timespec time{};
    if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&time))return 0;
    return double(time.tv_sec)+double(time.tv_nsec)*1e-9;
}
struct ApiTimer {
    double &total;
    const metal_backend *backend;
    ApiProfile &profile;
    double cpu_start,gpu_start,wait_start,conversion_start;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    ApiTimer(double &value,const metal_backend *b,ApiProfile &p) : total(value),backend(b),profile(p),
        cpu_start(process_cpu_time()),gpu_start(metal_backend_gpu_seconds(b)),
        wait_start(metal_backend_wait_seconds(b)),conversion_start(metal_backend_conversion_seconds(b)) {}
    ~ApiTimer() {
        total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ++profile.calls;
        profile.process_cpu_seconds+=std::max(0.,process_cpu_time()-cpu_start);
        profile.gpu_command_seconds+=std::max(0.,metal_backend_gpu_seconds(backend)-gpu_start);
        profile.wait_seconds+=std::max(0.,metal_backend_wait_seconds(backend)-wait_start);
        profile.conversion_seconds+=std::max(0.,metal_backend_conversion_seconds(backend)-conversion_start);
    }
};
struct state {
  metal_nonlocal_stats nonlocal_stats{};
  uint64_t nonlocal_success=0;
  double nonlocal_seconds=0;
  ApiProfile nonlocal_profile;
  unsigned long long hamiltonian_calls=0,hamiltonian_success=0,hamiltonian_from_pw_success=0;
  unsigned long long orth_calls=0,orth_success=0;
  unsigned long long full_orth_calls=0,full_orth_success=0;
  double full_orth_seconds=0;
  double orth_seconds=0;
  double hamiltonian_seconds=0;
  double gemm_seconds=0,fft_seconds=0,reduce_seconds=0,dot_seconds=0,eigh_seconds=0;
  ApiProfile gemm_profile,fft_profile,reduce_profile,dot_profile,eigh_profile,orth_profile,full_orth_profile,hamiltonian_profile;
  metal_backend *backend = nullptr;
  bool enabled = false;
  unsigned long long calls = 0;
  unsigned long long gpu_eligible = 0;
  unsigned long long gpu_success = 0;
  unsigned long long fft3d_calls = 0;
  unsigned long long fft3d_success = 0;
  unsigned long long fft3d_batch_calls = 0;
  unsigned long long fft3d_batch_success = 0;
  metal_resident *fft_resident = nullptr;
  size_t fft_resident_shape[4]{};
  size_t fft_resident_bytes = 0;
  unsigned long long resident_fft_creates=0,resident_fft_reuses=0,resident_fft_success=0;
  uint64_t wave_owner_calls=0,wave_owner_success=0,wave_owner_creates=0,wave_owner_releases=0,wave_owner_live=0;
  uint64_t wave_owner_host_import_bytes=0,wave_owner_host_export_bytes=0,wave_owner_static_uploads=0,wave_owner_static_reuses=0;
  uint64_t wave_owner_peak_bytes=0;
  unsigned long long dsum_calls = 0;
  unsigned long long dsum_gpu_eligible = 0;
  unsigned long long dsum_gpu_success = 0;
  unsigned long long zdotc_gpu_eligible=0;
  unsigned long long zdotc_calls = 0;
  unsigned long long zdotc_gpu_success = 0;
  unsigned long long vh_apply_calls = 0;
  unsigned long long vh_apply_gpu_success = 0;
  unsigned long long generalized_all_gpu_success=0,gpu_cholesky_success=0,gpu_transform_success=0,gpu_backtransform_success=0;
  unsigned long long zhegv_calls = 0;
  unsigned long long zhegv_gpu_eligible = 0;
  unsigned long long zhegv_gpu_success = 0;
  unsigned long long lobpcg_gpu_success=0,lobpcg_jacobi_rescue=0;
  unsigned long long lobpcg_iterations=0;
  unsigned long long zheevx_calls=0,zheevx_gpu_eligible=0,zheevx_gpu_success=0;
  unsigned long long adaptive_cpu_gemm=0, adaptive_cpu_fft=0, adaptive_cpu_eigh=0;
  unsigned long long zheev_calls = 0;
  unsigned long long zheev_gpu_eligible = 0;
  unsigned long long zheev_gpu_success = 0;
  std::once_flag once;
  std::recursive_mutex api_mutex;
  ~state() { metal_backend_resident_destroy(fft_resident); metal_backend_destroy(backend); }
};
state &S() {
  // Portable initialization has no calibration allocations. Explicitly create
  // telemetry first so its static destructor follows report() and ~state().
  metal_backend_prepare_memory_stats();
  static state s; return s;
}
bool selected(const char *name) {
  const char *env = std::getenv(name);
  return !env || (std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0);
}
long long env_limit(const char *name, long long fallback) {
  const char *value = std::getenv(name);
  if (!value || !*value) return fallback;
  char *end = nullptr;
  long long parsed = std::strtoll(value, &end, 10);
  return end && *end == '\0' && parsed >= 0 ? parsed : fallback;
}
void init();
void report() {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);
  if (!s.enabled && s.calls == 0) return;
  const char *path = std::getenv("VASP_METAL_STATS");
  const auto mpi=metal_stats::mpi_identity();
  const std::string p=metal_stats::path(path,mpi);
  std::ofstream out(p);
  if (!out) return;
  auto profile_json=[](const ApiProfile &p,double wall) {
    std::ostringstream out;
    out << "{\"calls\":" << p.calls << ",\"api_wall_seconds\":" << wall
        << ",\"process_cpu_seconds\":" << p.process_cpu_seconds
        << ",\"gpu_command_seconds\":" << p.gpu_command_seconds
        << ",\"host_wait_seconds\":" << p.wait_seconds
        << ",\"conversion_instrumented_seconds\":" << p.conversion_seconds << "}";
    return out.str();
  };
  metal_chunk_stats chunks{};metal_backend_get_chunk_stats(s.backend,&chunks);
  metal_memory_stats memory{};metal_backend_get_memory_stats(s.backend,&memory);
  out << "{\"enabled\":" << (s.enabled ? 1 : 0)
      << ",\"mpi_rank\":" << mpi.rank
      << ",\"mpi_world_size\":" << mpi.size
      << ",\"calls\":" << s.calls
      << ",\"gpu_eligible\":" << s.gpu_eligible
      << ",\"gpu_success\":" << s.gpu_success
      << ",\"cooperative_orth_completed_commands\":" << metal_backend_cooperative_orth_calls(s.backend)
      << ",\"fft_compact_axis_batches\":" << metal_backend_fft_compact_axes(s.backend)
      << ",\"fft_wide_axis_batches\":" << metal_backend_fft_wide_axes(s.backend)
      << ",\"metal_memory_scope\":\"device resource accounting sampled at buffer allocations; excludes CPU staging, other processes and unobserved driver peaks\""
      << ",\"metal_current_device_allocated_bytes\":" << memory.current_device_allocated_bytes
      << ",\"metal_observed_device_peak_bytes\":" << memory.observed_device_peak_bytes
      << ",\"metal_device_budget_bytes\":" << memory.device_budget_bytes
      << ",\"metal_allocation_requests\":" << memory.allocation_requests
      << ",\"metal_budget_rejections\":" << memory.budget_rejections
      << ",\"metal_driver_allocation_failures\":" << memory.driver_allocation_failures
      << ",\"metal_pressure_eviction_attempts\":" << memory.pressure_eviction_attempts
      << ",\"operator_profile_scope\":\"API wall and process CPU activity; GPU/wait/conversion overlap; CPU work after rejected Fortran hooks is outside API scope\""
      << ",\"operator_profiles\":{\"gemm\":" << profile_json(s.gemm_profile,s.gemm_seconds)
      << ",\"fft\":" << profile_json(s.fft_profile,s.fft_seconds)
      << ",\"reduce\":" << profile_json(s.reduce_profile,s.reduce_seconds)
      << ",\"dot\":" << profile_json(s.dot_profile,s.dot_seconds)
      << ",\"eigh\":" << profile_json(s.eigh_profile,s.eigh_seconds)
      << ",\"orth_overlap\":" << profile_json(s.orth_profile,s.orth_seconds)
      << ",\"full_orth\":" << profile_json(s.full_orth_profile,s.full_orth_seconds)
      << ",\"hamiltonian\":" << profile_json(s.hamiltonian_profile,s.hamiltonian_seconds)
      << ",\"nonlocal_projection\":" << profile_json(s.nonlocal_profile,s.nonlocal_seconds) << "}"
      << ",\"chunk_workspace_scope\":\"planned active staging bound\""
      << ",\"fft_chunk_calls\":" << chunks.fft_calls
      << ",\"fft_chunk_windows\":" << chunks.fft_windows
      << ",\"fft_chunk_peak_workspace_bytes\":" << chunks.fft_peak_workspace_bytes
      << ",\"gemm_chunk_calls\":" << chunks.gemm_calls
      << ",\"gemm_chunk_windows\":" << chunks.gemm_windows
      << ",\"gemm_chunk_peak_workspace_bytes\":" << chunks.gemm_peak_workspace_bytes
      << ",\"fft3d_calls\":" << s.fft3d_calls
      << ",\"fft3d_success\":" << s.fft3d_success
      << ",\"fft3d_batch_calls\":" << s.fft3d_batch_calls
      << ",\"fft3d_batch_success\":" << s.fft3d_batch_success
      << ",\"resident_fft_creates\":" << s.resident_fft_creates
      << ",\"resident_fft_reuses\":" << s.resident_fft_reuses
      << ",\"resident_fft_success\":" << s.resident_fft_success
      << ",\"resident_fft_cache_bytes\":" << s.fft_resident_bytes
      << ",\"wave_owner_calls\":" << s.wave_owner_calls
      << ",\"wave_owner_success\":" << s.wave_owner_success
      << ",\"wave_owner_creates\":" << s.wave_owner_creates
      << ",\"wave_owner_releases\":" << s.wave_owner_releases
      << ",\"wave_owner_live\":" << s.wave_owner_live
      << ",\"wave_owner_host_import_bytes\":" << s.wave_owner_host_import_bytes
      << ",\"wave_owner_host_export_bytes\":" << s.wave_owner_host_export_bytes
      << ",\"wave_owner_intermediate_host_bytes\":0"
      << ",\"wave_owner_static_uploads\":" << s.wave_owner_static_uploads
      << ",\"wave_owner_static_reuses\":" << s.wave_owner_static_reuses
      << ",\"wave_owner_peak_bytes\":" << s.wave_owner_peak_bytes
      << ",\"dsum_calls\":" << s.dsum_calls
      << ",\"dsum_gpu_eligible\":" << s.dsum_gpu_eligible
      << ",\"dsum_gpu_success\":" << s.dsum_gpu_success
      << ",\"zdotc_gpu_eligible\":" << s.zdotc_gpu_eligible
      << ",\"zdotc_calls\":" << s.zdotc_calls
      << ",\"zdotc_gpu_success\":" << s.zdotc_gpu_success
      << ",\"vh_apply_calls\":" << s.vh_apply_calls
      << ",\"vh_apply_gpu_success\":" << s.vh_apply_gpu_success
      << ",\"generalized_all_gpu_success\":" << s.generalized_all_gpu_success
      << ",\"gpu_cholesky_success\":" << s.gpu_cholesky_success
      << ",\"gpu_transform_success\":" << s.gpu_transform_success
      << ",\"gpu_backtransform_success\":" << s.gpu_backtransform_success
      << ",\"zhegv_calls\":" << s.zhegv_calls
      << ",\"zhegv_gpu_eligible\":" << s.zhegv_gpu_eligible
      << ",\"zhegv_gpu_success\":" << s.zhegv_gpu_success
      << ",\"zheev_calls\":" << s.zheev_calls
      << ",\"zheev_gpu_eligible\":" << s.zheev_gpu_eligible
      << ",\"zheev_gpu_success\":" << s.zheev_gpu_success
      << ",\"lobpcg_gpu_success\":" << s.lobpcg_gpu_success
      << ",\"lobpcg_jacobi_rescue\":" << s.lobpcg_jacobi_rescue
      << ",\"lobpcg_iterations\":" << s.lobpcg_iterations
      << ",\"zheevx_gpu_eligible\":" << s.zheevx_gpu_eligible
      << ",\"zheevx_calls\":" << s.zheevx_calls
      << ",\"zheevx_gpu_success\":" << s.zheevx_gpu_success
      << ",\"adaptive_cpu_gemm\":" << s.adaptive_cpu_gemm
      << ",\"adaptive_cpu_fft\":" << s.adaptive_cpu_fft
      << ",\"adaptive_cpu_eigh\":" << s.adaptive_cpu_eigh
      << ",\"gemm_api_seconds\":" << s.gemm_seconds
      << ",\"fft_api_seconds\":" << s.fft_seconds
      << ",\"reduce_api_seconds\":" << s.reduce_seconds
      << ",\"dot_api_seconds\":" << s.dot_seconds
      << ",\"eigh_api_seconds\":" << s.eigh_seconds
      << ",\"full_orth_calls\":" << s.full_orth_calls
      << ",\"full_orth_success\":" << s.full_orth_success
      << ",\"full_orth_api_seconds\":" << s.full_orth_seconds
      << ",\"orth_overlap_calls\":" << s.orth_calls
      << ",\"orth_overlap_success\":" << s.orth_success
      << ",\"orth_overlap_api_seconds\":" << s.orth_seconds
      << ",\"hamiltonian_calls\":" << s.hamiltonian_calls
      << ",\"hamiltonian_success\":" << s.hamiltonian_success
      << ",\"hamiltonian_from_pw_success\":" << s.hamiltonian_from_pw_success
      << ",\"hamiltonian_api_seconds\":" << s.hamiltonian_seconds
      << ",\"nonlocal_calls\":" << s.nonlocal_stats.calls
      << ",\"nonlocal_completed_commands\":" << s.nonlocal_stats.completed_commands
      << ",\"nonlocal_wave_import_bytes\":" << s.nonlocal_stats.wave_import_bytes
      << ",\"nonlocal_radial_import_bytes\":" << s.nonlocal_stats.radial_import_bytes
      << ",\"nonlocal_phase_import_bytes\":" << s.nonlocal_stats.phase_import_bytes
      << ",\"nonlocal_metadata_import_bytes\":" << s.nonlocal_stats.metadata_import_bytes
      << ",\"nonlocal_gradient_import_bytes\":" << s.nonlocal_stats.gradient_import_bytes
      << ",\"nonlocal_export_bytes\":" << s.nonlocal_stats.export_bytes
      << ",\"nonlocal_wave_reuses\":" << s.nonlocal_stats.wave_reuses
      << ",\"nonlocal_phase_reuses\":" << s.nonlocal_stats.phase_reuses
      << ",\"nonlocal_windows\":" << s.nonlocal_stats.windows
      << ",\"nonlocal_planned_workspace_peak_bytes\":" << s.nonlocal_stats.planned_workspace_peak_bytes
      << ",\"nonlocal_success\":" << s.nonlocal_success
      << ",\"paw_static_uploads\":" << metal_backend_hamiltonian_static_uploads(s.backend)
      << ",\"paw_static_reuses\":" << metal_backend_hamiltonian_static_reuses(s.backend)
      << ",\"gpu_command_seconds\":" << metal_backend_gpu_seconds(s.backend)
      << ",\"gpu_submissions\":" << metal_backend_command_count(s.backend)
      << ",\"gpu_waits\":" << metal_backend_wait_count(s.backend)
      << ",\"host_wait_seconds\":" << metal_backend_wait_seconds(s.backend)
      << ",\"host_conversion_instrumented_seconds\":" << metal_backend_conversion_seconds(s.backend)
      << ",\"gpu_device\":" << (s.backend && metal_backend_gpu_enabled(s.backend) ? 1 : 0)
      << ",\"gpu_variant\":\"" << metal_backend_variant(s.backend) << "\""
      << ",\"metal4_family\":" << metal_backend_metal4_enabled(s.backend)
      << ",\"metal4_command_requested\":" << metal_backend_metal4_command_requested(s.backend)
      << ",\"metal4_command_available\":" << metal_backend_metal4_command_available(s.backend)
      << ",\"metal4_command_calls\":" << metal_backend_metal4_command_calls(s.backend)
      << ",\"metal4_command_success\":" << metal_backend_metal4_command_success(s.backend)
      << ",\"metal4_command_fallbacks\":" << metal_backend_metal4_command_fallbacks(s.backend)
      << ",\"metal4_command_reason\":" << metal_stats::json_string(metal_backend_metal4_command_reason(s.backend))
      << ",\"precompiled_library_requested\":" << metal_backend_precompiled_requested(s.backend)
      << ",\"precompiled_library_verified\":" << metal_backend_precompiled_verified(s.backend)
      << ",\"precompiled_library_used\":" << metal_backend_precompiled_used(s.backend)
      << ",\"precompiled_library_fallbacks\":" << metal_backend_precompiled_fallbacks(s.backend)
      << ",\"precompiled_library_reason\":" << metal_stats::json_string(metal_backend_precompiled_reason(s.backend))
      << ",\"precompiled_library_path\":" << metal_stats::json_string(metal_backend_precompiled_path(s.backend))
      << ",\"metal4_ml_api_available\":" << metal_backend_ml_api_available(s.backend)
      << ",\"device_tuned_profile\":" << metal_backend_device_tuned(s.backend)
      << ",\"shader_language\":\"" << metal_backend_language_version(s.backend) << "\""
      << ",\"shader_math_mode\":\"safe-precise\""
      << ",\"shader_compile_fallbacks\":" << metal_backend_compile_fallbacks(s.backend)
      << ",\"tiled_gemm_enabled\":" << metal_backend_tiled_gemm_enabled(s.backend)
      << ",\"tiled_gemm_calls\":" << metal_backend_tiled_gemm_calls(s.backend)
      << ",\"tiled_gemm_fallbacks\":" << metal_backend_tiled_gemm_fallbacks(s.backend)
      << ",\"tiled_gemm_autoselected\":" << metal_backend_tiled_gemm_autoselected(s.backend)
      << ",\"gemm_calibrated\":" << metal_backend_gemm_calibrated(s.backend)
      << "}\n";
}

void cpu_sum(const double *input, size_t n, double *result) {
  if (!input || !result || n == 0) return;
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
}

extern "C" int vasp_metal_hamiltonian_available() {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);std::call_once(s.once,init);
 return s.enabled && selected("VASP_METAL_HAMILTONIAN") && s.backend && metal_backend_gpu_enabled(s.backend);
}
extern "C" void vasp_metal_hamiltonian_invalidate() {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);
 std::call_once(s.once,init);metal_backend_hamiltonian_invalidate(s.backend);
}
extern "C" int vasp_metal_hamiltonian(int nx,int ny,int nz,int npl,int bands,int nions,int npro,
 int lmdim,int ntypes,int qld,int qchannels,int phase_ld,int real_fields,int from_pw,const void *sv,
 const metal_complex64 *const *cr,const metal_complex64 *const *cw,const metal_complex64 *const *proj,
 const int *indices,const double *ke,const double *eig,const void *d,const void *q,
 const int *types,const int *lmmax,const int *lmbase,const double *radial,
 const metal_complex64 *cq,const metal_complex64 *phase,metal_complex64 *out,int ldout) {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);std::call_once(s.once,init);
 ++s.hamiltonian_calls;ApiTimer timer{s.hamiltonian_seconds,s.backend,s.hamiltonian_profile};
 if(!s.enabled || !selected("VASP_METAL_HAMILTONIAN") || !s.backend || !metal_backend_gpu_enabled(s.backend) ||
    nx<=0 || ny<=0 || nz<=0 || npl<=0 || bands<=0 || nions<=0 || npro<=0 || lmdim<=0 || ntypes<=0 ||
    qld<=0 || qchannels<=0 || phase_ld<=0 || ldout<=0) return 0;
 int rc=metal_backend_hamiltonian_gpu(s.backend,nx,ny,nz,npl,bands,nions,npro,lmdim,ntypes,qld,qchannels,phase_ld,
    real_fields,from_pw,sv,cr,cw,proj,indices,ke,eig,d,q,types,lmmax,lmbase,radial,cq,phase,out,ldout);
 if(rc!=METAL_BACKEND_OK) return 0;
 ++s.hamiltonian_success;if(from_pw) ++s.hamiltonian_from_pw_success;return 1;
}

extern "C" int vasp_metal_orth_overlap(int rows,int cols,int npl,int npro,const metal_complex64 *wa,const metal_complex64 *wb,
 int ldw,const metal_complex64 *pa,const metal_complex64 *pb,int ldp,metal_complex64 *out,int ldo) {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);std::call_once(s.once,init);++s.orth_calls;ApiTimer timer{s.orth_seconds,s.backend,s.orth_profile};
 if(!s.enabled || !selected("VASP_METAL_ORTH") || !s.backend || !metal_backend_gpu_enabled(s.backend) || rows<=0 || cols<=0 || npl<0 || npro<0 || ldw<0 || ldp<0 || ldo<=0) return 0;
 int rc=metal_backend_orth_overlap_gpu(s.backend,rows,cols,npl,npro,wa,wb,ldw,pa,pb,ldp,out,ldo);
 if(rc) return 0;++s.orth_success;return 1;
}
extern "C" int vasp_metal_full_orth(int npl,int npro,int bands,int nions,int lm,int real_fields,const void *metric,
 const int *counts,const int *bases,metal_complex64 *wave,int ldw,metal_complex64 *proj,int ldp) {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);std::call_once(s.once,init);++s.full_orth_calls;ApiTimer timer{s.full_orth_seconds,s.backend,s.full_orth_profile};
 if(!s.enabled || !selected("VASP_METAL_FULL_ORTH") || !s.backend || !metal_backend_gpu_enabled(s.backend) || npl<=0 || npro<0 || bands<=0 || nions<0 || lm<0 || ldw<=0 || ldp<0) return 0;
 int rc=metal_backend_full_orth_gpu(s.backend,npl,npro,bands,nions,lm,real_fields,metric,counts,bases,wave,ldw,proj,ldp);
 if(rc) return 0;++s.full_orth_success;return 1;
}
// One shape cache bounds process memory. Host arrays are uploaded on every ABI
// call; cache reuse refers to allocations, not a claim that SCF data stay resident.
int resident_fft(state &s,size_t nx,size_t ny,size_t nz,size_t nbatch,size_t stride,
                 metal_complex64 *data,int direction) {
  const size_t limit=static_cast<size_t>(env_limit("VASP_METAL_RESIDENT_FFT_MAX_BYTES",256LL*1024*1024));
  if(!selected("VASP_METAL_RESIDENT_FFT") || s.fft_resident_bytes>limit) {
    metal_backend_resident_destroy(s.fft_resident); s.fft_resident=nullptr; s.fft_resident_bytes=0;
  }
  if(!selected("VASP_METAL_RESIDENT_FFT")) return METAL_BACKEND_UNSUPPORTED;
  if(nx>UINT32_MAX || ny>UINT32_MAX/nx || nz>UINT32_MAX/(nx*ny)) return METAL_BACKEND_INVALID_ARGUMENT;
  const size_t total=nx*ny*nz;
  if(nbatch>UINT32_MAX/total || stride<total || stride>SIZE_MAX/nbatch/sizeof(*data)) return METAL_BACKEND_INVALID_ARGUMENT;
  const size_t bytes=total*nbatch*2*sizeof(metal_complex64)+sizeof(uint32_t);
  if(bytes>limit) return METAL_BACKEND_UNSUPPORTED;
  const size_t shape[4]={nx,ny,nz,nbatch};
  if(s.fft_resident && std::equal(shape,shape+4,s.fft_resident_shape)) ++s.resident_fft_reuses;
  else {
    metal_backend_resident_destroy(s.fft_resident); s.fft_resident=nullptr; s.fft_resident_bytes=0;
    int rc=metal_backend_resident_create(s.backend,nx,ny,nz,nbatch,0,&s.fft_resident);
    if(rc) return rc;
    std::copy(shape,shape+4,s.fft_resident_shape); s.fft_resident_bytes=bytes; ++s.resident_fft_creates;
  }
  int rc=metal_backend_resident_begin_step(s.fft_resident);
  if(!rc) rc=metal_backend_resident_upload_wavefunction(s.fft_resident,data,stride,1);
  if(!rc) rc=metal_backend_resident_fft(s.fft_resident,direction);
  if(!rc) rc=metal_backend_resident_end_step(s.fft_resident,0);
  if(!rc) rc=metal_backend_resident_download_wavefunction(s.fft_resident,data,stride);
  if(rc) {
    metal_backend_resident_abort_step(s.fft_resident);
    metal_backend_resident_destroy(s.fft_resident); s.fft_resident=nullptr; s.fft_resident_bytes=0;
    return rc;
  }
  ++s.resident_fft_success; return METAL_BACKEND_OK;
}
extern "C" void vasp_metal_resident_fft_cache_stats(uint64_t *creates,uint64_t *reuses,uint64_t *bytes) {
  auto &s=S(); std::lock_guard<std::recursive_mutex> guard(s.api_mutex);
  if(creates) *creates=s.resident_fft_creates;
  if(reuses) *reuses=s.resident_fft_reuses;
  if(bytes) *bytes=s.fft_resident_bytes;
}

extern "C" void vasp_metal_wave_owner_release(metal_wave_owner **owner) {
  auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);
  if(owner && *owner) {metal_wave_owner_destroy(*owner);*owner=nullptr;++s.wave_owner_releases;if(s.wave_owner_live) --s.wave_owner_live;}
}
extern "C" int vasp_metal_wave_prepare(metal_wave_owner **owner,uint64_t generation,int npl,int bands,int nions,int npro,
 int lm,int ntypes,int qchannels,int qld,int phase_ld,int real_metric,int overlap,metal_complex64 *const *wave,
 metal_complex64 *const *proj,const double *precon,int ldprecon,const int *types,const int *lmmax,const int *bases,
 const double *radial,const metal_complex64 *cq,const metal_complex64 *phase,const void *metric,
 metal_complex64 *metric_projection,int ldmetric,double *scales) {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);std::call_once(s.once,init);++s.wave_owner_calls;
 if(!s.enabled || !std::getenv("VASP_METAL_WAVE_OWNER") || !selected("VASP_METAL_WAVE_OWNER") || !s.backend || !metal_backend_gpu_enabled(s.backend) || !owner ||
    npl<=0 || bands<=0 || nions<=0 || npro<=0 || lm<=0 || ntypes<=0 || qchannels<=0 || qld<=0 || phase_ld<=0 || ldprecon<=0 || ldmetric<0) return 0;
 metal_wave_owner_shape shape{(size_t)npl,(size_t)bands,(size_t)nions,(size_t)npro,(size_t)lm,(size_t)ntypes,(size_t)qchannels};
 if(*owner && !metal_wave_owner_matches(*owner,shape)) vasp_metal_wave_owner_release(owner);
 if(!*owner) {
   if(metal_wave_owner_create(s.backend,shape,owner)) return 0;
   ++s.wave_owner_creates;++s.wave_owner_live;
 }
 metal_wave_owner_stats before{},after{};metal_wave_owner_get_stats(*owner,&before);
 metal_wave_prepare_input input{reinterpret_cast<const metal_complex64 *const *>(wave),precon,(size_t)ldprecon,(size_t)qld,(size_t)phase_ld,
   types,lmmax,bases,radial,cq,phase,metric,real_metric,overlap};
 int rc=metal_wave_owner_prepare(*owner,generation,metal_backend_paw_epoch(s.backend),&input);
 if(!rc) rc=metal_wave_owner_download(*owner,generation,wave,proj,metric_projection,(size_t)ldmetric,scales);
 metal_wave_owner_get_stats(*owner,&after);
 s.wave_owner_host_import_bytes+=after.host_import_bytes-before.host_import_bytes;
 s.wave_owner_host_export_bytes+=after.host_export_bytes-before.host_export_bytes;
 s.wave_owner_static_uploads+=after.static_uploads-before.static_uploads;s.wave_owner_static_reuses+=after.static_reuses-before.static_reuses;
 s.wave_owner_peak_bytes=std::max(s.wave_owner_peak_bytes,after.device_bytes);
 if(rc) {metal_wave_owner_invalidate(*owner);return 0;}
 ++s.wave_owner_success;return 1;
}

extern "C" int vasp_metal_fft3d(int nx, int ny, int nz,
                                  fcomplex *data, int isign) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);
  std::call_once(s.once, init);ApiTimer timer{s.fft_seconds,s.backend,s.fft_profile};
  ++s.fft3d_calls;
  if (!s.enabled || !selected("VASP_METAL_FFT") || !s.backend || !metal_backend_gpu_enabled(s.backend) || !data ||
      nx <= 0 || ny <= 0 || nz <= 0 || (isign != -1 && isign != 1)) return 0;
  size_t volume=0;
  if(!metal_chunk::multiply(size_t(nx),size_t(ny),volume) || !metal_chunk::multiply(volume,size_t(nz),volume) || volume>SIZE_MAX/sizeof(*data))return 0;
  const long long fft_work = static_cast<long long>(volume);
  if (fft_work < env_limit("VASP_METAL_FFT_MIN_POINTS", 32768)) {
    ++s.adaptive_cpu_fft;
    return 0;
  }
  int rc=resident_fft(s,nx,ny,nz,1,static_cast<size_t>(fft_work),reinterpret_cast<metal_complex64 *>(data),isign);
  if(rc!=METAL_BACKEND_OK) rc=metal_backend_zfft3d_batch_gpu(s.backend,nx,ny,nz,1,
      static_cast<size_t>(nx)*ny*nz,reinterpret_cast<const metal_complex64 *>(data),
      reinterpret_cast<metal_complex64 *>(data),isign,0);
  if(rc!=METAL_BACKEND_OK) return 0;
  ++s.fft3d_success;return 1;
}
extern "C" int vasp_metal_fft3d_batch(int nx,int ny,int nz,int nbatch,int ldc,fcomplex *data,int isign) {
  auto &s=S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);
  std::call_once(s.once,init);ApiTimer timer{s.fft_seconds,s.backend,s.fft_profile};++s.fft3d_batch_calls;
  if(!s.enabled || !selected("VASP_METAL_FFT") || !s.backend || !metal_backend_gpu_enabled(s.backend) ||
      !data || nx<=0 || ny<=0 || nz<=0 || nbatch<=0 || ldc<=0 || (isign!=-1 && isign!=1)) return 0;
  size_t volume=0;
  if(!metal_chunk::multiply(size_t(nx),size_t(ny),volume) || !metal_chunk::multiply(volume,size_t(nz),volume) || volume>SIZE_MAX/sizeof(*data))return 0;
  const size_t total=volume;
  if(static_cast<size_t>(ldc)<total || static_cast<size_t>(ldc)>SIZE_MAX/static_cast<size_t>(nbatch)/sizeof(*data)) return 0;
  const long long fft_work = static_cast<long long>(total)*nbatch;
  if (fft_work < env_limit("VASP_METAL_FFT_MIN_POINTS", 32768)) {
    ++s.adaptive_cpu_fft;
    return 0;
  }
  int rc=resident_fft(s,nx,ny,nz,nbatch,ldc,reinterpret_cast<metal_complex64 *>(data),isign);
  if(rc!=METAL_BACKEND_OK) rc=metal_backend_zfft3d_batch_gpu(s.backend,nx,ny,nz,nbatch,ldc,
      reinterpret_cast<const metal_complex64 *>(data),reinterpret_cast<metal_complex64 *>(data),isign,0);
  if(rc!=METAL_BACKEND_OK) return 0;
  ++s.fft3d_batch_success;return 1;
}

extern "C" int vasp_metal_dsum(const double *input, int n, double *result) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);
  std::call_once(s.once, init);ApiTimer timer{s.reduce_seconds,s.backend,s.reduce_profile};
  ++s.dsum_calls;
  if (!input || !result || n <= 0) return 0;
  if (s.enabled && selected("VASP_METAL_REDUCE") && s.backend && metal_backend_gpu_enabled(s.backend)) {
    ++s.dsum_gpu_eligible;
    if (metal_backend_dsum(s.backend, static_cast<size_t>(n), input, result) == METAL_BACKEND_OK &&
        metal_backend_last_gpu_success(s.backend)) {
      ++s.dsum_gpu_success;
      return 1;
    }
  }
  cpu_sum(input, static_cast<size_t>(n), result);
  return 1;
}
extern "C" int vasp_metal_zdotc(const fcomplex *x,const fcomplex *y,int n,fcomplex *result) {
  auto &s=S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);std::call_once(s.once,init);ApiTimer timer{s.dot_seconds,s.backend,s.dot_profile};++s.zdotc_calls;
  if(!s.enabled || !selected("VASP_METAL_DOT") || !s.backend || !metal_backend_gpu_enabled(s.backend) || n<=0 || !x || !y || !result) return 0;
  ++s.zdotc_gpu_eligible;
  int rc=metal_backend_zdotc_gpu(s.backend,static_cast<size_t>(n),reinterpret_cast<const metal_complex64 *>(x),1,
       reinterpret_cast<const metal_complex64 *>(y),1,reinterpret_cast<metal_complex64 *>(result));
  if(rc!=METAL_BACKEND_OK || !metal_backend_last_gpu_success(s.backend)) return 0;
  ++s.zdotc_gpu_success;return 1;
}

extern "C" int vasp_metal_vh_apply(int np, int mplwv, int nrspinors,
                                     const fcomplex *sv, const fcomplex *cr,
                                     fcomplex *cvr, double rinplw) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex); std::call_once(s.once, init); ++s.vh_apply_calls;
  if (!s.enabled || !selected("VASP_METAL_VHAMIL") || !s.backend ||
      !metal_backend_gpu_enabled(s.backend) || np <= 0 || mplwv < np ||
      (nrspinors != 1 && nrspinors != 2) || !sv || !cr || !cvr) return 0;
  const int rc = metal_backend_vh_apply(s.backend, static_cast<size_t>(np),
      static_cast<size_t>(mplwv), static_cast<size_t>(nrspinors),
      reinterpret_cast<const metal_complex64 *>(sv), reinterpret_cast<const metal_complex64 *>(cr),
      reinterpret_cast<metal_complex64 *>(cvr), rinplw);
  if (rc != METAL_BACKEND_OK || !metal_backend_last_gpu_success(s.backend)) return 0;
  ++s.vh_apply_gpu_success; return 1;
}

extern "C" int vasp_metal_vh_apply_batch(int np, int mplwv, int nrspinors, int nbands,
                                           const fcomplex *sv, const fcomplex *cr,
                                           fcomplex *cvr, double rinplw) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex); std::call_once(s.once, init); ++s.vh_apply_calls;
  if (!s.enabled || !selected("VASP_METAL_VHAMIL") || !s.backend || !metal_backend_gpu_enabled(s.backend) ||
      np <= 0 || mplwv < np || nbands <= 0 || (nrspinors != 1 && nrspinors != 2) || !sv || !cr || !cvr) return 0;
  const int rc = metal_backend_vh_apply_batch(s.backend, (size_t)np, (size_t)mplwv,
      (size_t)nrspinors, (size_t)nbands, reinterpret_cast<const metal_complex64 *>(sv),
      reinterpret_cast<const metal_complex64 *>(cr), reinterpret_cast<metal_complex64 *>(cvr), rinplw);
  if (rc != METAL_BACKEND_OK || !metal_backend_last_gpu_success(s.backend)) return 0;
  ++s.vh_apply_gpu_success; return 1;
}

extern "C" int vasp_metal_vh_apply_real(int np, int mplwv, int nrspinors,
                                          const double *sv, const fcomplex *cr,
                                          fcomplex *cvr, double rinplw) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex); std::call_once(s.once, init); ++s.vh_apply_calls;
  if (!s.enabled || !selected("VASP_METAL_VHAMIL") || !s.backend || !metal_backend_gpu_enabled(s.backend) ||
      np <= 0 || mplwv < np || (nrspinors != 1 && nrspinors != 2) || !sv || !cr || !cvr) return 0;
  const int rc = metal_backend_vh_apply_real(s.backend, static_cast<size_t>(np), static_cast<size_t>(mplwv),
      static_cast<size_t>(nrspinors), sv, reinterpret_cast<const metal_complex64 *>(cr),
      reinterpret_cast<metal_complex64 *>(cvr), rinplw);
  if (rc != METAL_BACKEND_OK || !metal_backend_last_gpu_success(s.backend)) return 0;
  ++s.vh_apply_gpu_success; return 1;
}

extern "C" int vasp_metal_vh_apply_real_batch(int np, int mplwv, int nrspinors, int nbands,
                                                const double *sv, const fcomplex *cr,
                                                fcomplex *cvr, double rinplw) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex); std::call_once(s.once, init); ++s.vh_apply_calls;
  if (!s.enabled || !selected("VASP_METAL_VHAMIL") || !s.backend || !metal_backend_gpu_enabled(s.backend) ||
      np <= 0 || mplwv < np || nbands <= 0 || (nrspinors != 1 && nrspinors != 2) || !sv || !cr || !cvr) return 0;
  const int rc = metal_backend_vh_apply_real_batch(s.backend, (size_t)np, (size_t)mplwv,
      (size_t)nrspinors, (size_t)nbands, sv, reinterpret_cast<const metal_complex64 *>(cr),
      reinterpret_cast<metal_complex64 *>(cvr), rinplw);
  if (rc != METAL_BACKEND_OK || !metal_backend_last_gpu_success(s.backend)) return 0;
  ++s.vh_apply_gpu_success; return 1;
}

void init() {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);
  const char *env = std::getenv("VASP_METAL");
  s.enabled = env && std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0;
  if (s.enabled) {
    metal_backend_create(&s.backend, 1);
    std::atexit(report);
    if (std::getenv("VASP_METAL_REQUIRE_GPU") && selected("VASP_METAL_REQUIRE_GPU") &&
        (!s.backend || !metal_backend_gpu_enabled(s.backend))) {
      std::fprintf(stderr, "Metal GPU required but unavailable: %s\n", metal_backend_last_error(s.backend));
      std::exit(86);
    }
  }
}
CBLAS_TRANSPOSE trans(char t) {
  if (t == 'C' || t == 'c') return CblasConjTrans;
  if (t == 'T' || t == 't') return CblasTrans;
  return CblasNoTrans;
}
bool unit_one(const fcomplex *z) { return z && z->real == 1.0 && z->imag == 0.0; }
bool unit_zero(const fcomplex *z) { return z && z->real == 0.0 && z->imag == 0.0; }
}

extern "C" void zgemm_(const char *ta, const char *tb, const int *m, const int *n, const int *k,
                         const fcomplex *alpha, const fcomplex *a, const int *lda,
                         const fcomplex *b, const int *ldb, const fcomplex *beta,
                         fcomplex *c, const int *ldc) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);
  std::call_once(s.once, init);ApiTimer timer{s.gemm_seconds,s.backend,s.gemm_profile};
  ++s.calls;
  if (!m || !n || !k || !lda || !ldb || !ldc || *m <= 0 || *n <= 0) return;
  const bool valid_trans = ta && tb &&
      (*ta == 'N' || *ta == 'n' || *ta == 'T' || *ta == 't' || *ta == 'C' || *ta == 'c') &&
      (*tb == 'N' || *tb == 'n' || *tb == 'T' || *tb == 't' || *tb == 'C' || *tb == 'c');
  const size_t a_rows = valid_trans && (*ta == 'N' || *ta == 'n') ? (size_t)*m : (size_t)*k;
  const size_t b_rows = valid_trans && (*tb == 'N' || *tb == 'n') ? (size_t)*k : (size_t)*n;
  const bool eligible = *k > 0 && alpha && beta && s.enabled && selected("VASP_METAL_GEMM") && s.backend && metal_backend_gpu_enabled(s.backend) && valid_trans &&
                        a && b && c && *lda >= (int)a_rows && *ldb >= (int)b_rows && *ldc >= *m &&
                        metal_gemm::host_shape(*m,*n,*k,(*ta!='N' && *ta!='n'),(*tb!='N' && *tb!='n'),*lda,*ldb,*ldc);
  const unsigned long long limit=static_cast<unsigned long long>(LLONG_MAX);
  const auto mm=static_cast<unsigned long long>(*m),nn=static_cast<unsigned long long>(*n),kk=*k>0 ? static_cast<unsigned long long>(*k):0;
  const long long gemm_work=kk && (mm>limit/nn || mm*nn>limit/kk) ? LLONG_MAX:static_cast<long long>(mm*nn*kk);
  const bool adaptive_gpu = gemm_work >= env_limit("VASP_METAL_GEMM_MIN_FLOPS", 262144);
  if (eligible && adaptive_gpu) {
    ++s.gpu_eligible;
    const int rc = metal_backend_zgemm_blas_gpu(s.backend, *ta, *tb, (size_t)*m, (size_t)*n, (size_t)*k,
                                       *reinterpret_cast<const metal_complex64 *>(alpha),
                                       reinterpret_cast<const metal_complex64 *>(a), (size_t)*lda,
                                       reinterpret_cast<const metal_complex64 *>(b), (size_t)*ldb,
                                       *reinterpret_cast<const metal_complex64 *>(beta),
                                       reinterpret_cast<metal_complex64 *>(c), (size_t)*ldc);
    if (rc == METAL_BACKEND_OK && metal_backend_last_gpu_success(s.backend)) {
      ++s.gpu_success;
      return;
    }
  } else if (eligible) {
    ++s.adaptive_cpu_gemm;
  }
  cblas_zgemm(CblasColMajor, trans(*ta), trans(*tb), *m, *n, *k,
              alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

using zhegv_fn = void (*)(const int *, const char *, const char *, const int *, fcomplex *, const int *,
                          fcomplex *, const int *, double *, fcomplex *, const int *, double *, int *);
using zheev_fn = void (*)(const char *, const char *, const int *, fcomplex *, const int *, double *,
                          fcomplex *, const int *, double *, int *);
using zpotrf_fn = void (*)(const char *, const int *, fcomplex *, const int *, int *);

extern "C" void zhegv_(const int *itype, const char *jobz, const char *uplo, const int *n,
                        fcomplex *a, const int *lda, fcomplex *b, const int *ldb, double *w,
                        fcomplex *work, const int *lwork, double *rwork, int *info) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);
  std::call_once(s.once, init);ApiTimer timer{s.eigh_seconds,s.backend,s.eigh_profile};
  ++s.zhegv_calls;
  const bool valid_gpu_call = s.enabled && selected("VASP_METAL_EIGH") && s.backend && metal_backend_gpu_enabled(s.backend) &&
      itype && *itype == 1 && jobz && uplo && n && lda && ldb && *n > 0 &&
      (*jobz == 'V' || *jobz == 'v' || *jobz == 'N' || *jobz == 'n') &&
      (*uplo == 'U' || *uplo == 'u' || *uplo == 'L' || *uplo == 'l') &&
      *lda >= *n && *ldb >= *n && a && b && w && work && rwork && info && lwork && *lwork >= std::max<long long>(1,2LL*(*n)-1) &&
      metal_backend_eigh_resources(s.backend,*n,*lda,*ldb,*n,1)==METAL_BACKEND_OK;
  const long long eigh_work = (n && *n > 0) ? metal_eigh::saturated_cubic(*n) : 0;
  const bool adaptive_gpu = eigh_work >= env_limit("VASP_METAL_EIGH_MIN_CUBIC", 262144);
  if (valid_gpu_call && adaptive_gpu) {
    ++s.zhegv_gpu_eligible;
    std::vector<metal_complex64> vectors(static_cast<size_t>(*n) * static_cast<size_t>(*n));
    std::vector<metal_complex64> factor(static_cast<size_t>(*n)*static_cast<size_t>(*n));
    const int rc = metal_backend_generalized_eigh_factor_gpu(
        s.backend, *itype, *uplo, *jobz, static_cast<size_t>(*n),
        reinterpret_cast<const metal_complex64 *>(a), static_cast<size_t>(*lda),
        reinterpret_cast<const metal_complex64 *>(b), static_cast<size_t>(*ldb),
        w, vectors.data(), static_cast<size_t>(*n),factor.data(),static_cast<size_t>(*n));
    if (rc == METAL_BACKEND_OK && metal_backend_last_gpu_success(s.backend)) {
      if (*jobz == 'V' || *jobz == 'v') {
        for (int col = 0; col < *n; ++col) for (int row = 0; row < *n; ++row)
          a[row + static_cast<size_t>(col) * (*lda)] = {vectors[row + static_cast<size_t>(col) * (*n)].real, vectors[row + static_cast<size_t>(col) * (*n)].imag};
      }
      const bool upper=*uplo=='U' || *uplo=='u';
      for(int j=0;j<*n;++j) for(int i=0;i<*n;++i) if(upper ? i<=j:i>=j)
          b[i+static_cast<size_t>(j)*(*ldb)]={factor[i+static_cast<size_t>(j)*(*n)].real,factor[i+static_cast<size_t>(j)*(*n)].imag};
      if(metal_backend_generalized_all_gpu(s.backend)) {++s.generalized_all_gpu_success;++s.gpu_cholesky_success;++s.gpu_transform_success;++s.gpu_backtransform_success;}
      work[0] = {static_cast<double>(std::max<long long>(1,2LL*(*n)-1)),0.0};
      if (info) *info = 0;
      ++s.zhegv_gpu_success;
      if(metal_backend_last_eigh_solver(s.backend)==1) ++s.lobpcg_gpu_success;
      if(metal_backend_last_eigh_solver(s.backend)==2) ++s.lobpcg_jacobi_rescue;
      s.lobpcg_iterations+=metal_backend_last_eigh_iterations(s.backend);
      return;
    }
  } else if (valid_gpu_call) {
    ++s.adaptive_cpu_eigh;
  }
  static zhegv_fn next = nullptr;
  if (!next) next = reinterpret_cast<zhegv_fn>(dlsym(RTLD_NEXT, "zhegv_"));
  if (next) {
    next(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, info);
    return;
  }
  if (info) *info = -999;
}

extern "C" void zheev_(const char *jobz, const char *uplo, const int *n, fcomplex *a, const int *lda,
                        double *w, fcomplex *work, const int *lwork, double *rwork, int *info) {
  auto &s = S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);
  std::call_once(s.once, init);ApiTimer timer{s.eigh_seconds,s.backend,s.eigh_profile};
  ++s.zheev_calls;
  const bool eligible = s.enabled && selected("VASP_METAL_EIGH") && s.backend && metal_backend_gpu_enabled(s.backend) &&
      jobz && uplo && n && lda && lwork && a && w && work && rwork && info &&
      *n>0 && *lda>=*n && *lwork>=std::max<long long>(1,2LL*(*n)-1) &&
      (*jobz=='V' || *jobz=='v' || *jobz=='N' || *jobz=='n') &&
      (*uplo=='U' || *uplo=='u' || *uplo=='L' || *uplo=='l') &&
      metal_backend_eigh_resources(s.backend,*n,*lda,0,*n,0)==METAL_BACKEND_OK;
  const long long eigh_work = (n && *n > 0) ? metal_eigh::saturated_cubic(*n) : 0;
  const bool adaptive_gpu = eigh_work >= env_limit("VASP_METAL_EIGH_MIN_CUBIC", 262144);
  if (eligible && adaptive_gpu) {
    ++s.zheev_gpu_eligible;
    std::vector<metal_complex64> full(static_cast<size_t>(*n)*static_cast<size_t>(*n)),v(full.size());
    const bool upper=*uplo=='U' || *uplo=='u';
    for(int j=0;j<*n;++j) for(int i=0;i<*n;++i) {
      const bool stored=upper ? i<=j : i>=j;
      const fcomplex x=stored ? a[i+static_cast<size_t>(j)*(*lda)] : a[j+static_cast<size_t>(i)*(*lda)];
      full[i+static_cast<size_t>(j)*(*n)]={x.real,i==j ? 0.0 : (stored ? x.imag : -x.imag)};
    }
    const int rc=metal_backend_hermitian_eigh_gpu(s.backend,static_cast<size_t>(*n),full.data(),static_cast<size_t>(*n),w,v.data(),static_cast<size_t>(*n));
    if(rc==METAL_BACKEND_OK && metal_backend_last_gpu_success(s.backend)) {
      if(*jobz=='V' || *jobz=='v') for(int j=0;j<*n;++j) for(int i=0;i<*n;++i) a[i+static_cast<size_t>(j)*(*lda)]={v[i+static_cast<size_t>(j)*(*n)].real,v[i+static_cast<size_t>(j)*(*n)].imag};
      work[0]={static_cast<double>(std::max<long long>(1,2LL*(*n)-1)),0.0}; *info=0;
      ++s.zheev_gpu_success;
      if(metal_backend_last_eigh_solver(s.backend)==1) ++s.lobpcg_gpu_success;
      if(metal_backend_last_eigh_solver(s.backend)==2) ++s.lobpcg_jacobi_rescue;
      s.lobpcg_iterations+=metal_backend_last_eigh_iterations(s.backend); return;
    }
  } else if (eligible) {
    ++s.adaptive_cpu_eigh;
  }
  static zheev_fn next = nullptr;
  if (!next) next = reinterpret_cast<zheev_fn>(dlsym(RTLD_NEXT, "zheev_"));
  if (next) {
    next(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
    return;
  }
  if (info) *info = -999;
}

using zheevx_fn=void (*)(const char*,const char*,const char*,const int*,fcomplex*,const int*,const double*,const double*,const int*,const int*,const double*,int*,double*,fcomplex*,const int*,fcomplex*,const int*,double*,int*,int*,int*);
extern "C" void zheevx_(const char *jobz,const char *range,const char *uplo,const int *n,fcomplex *a,const int *lda,
 const double *vl,const double *vu,const int *il,const int *iu,const double *abstol,int *m,double *w,fcomplex *z,const int *ldz,
 fcomplex *work,const int *lwork,double *rwork,int *iwork,int *ifail,int *info) {
    auto &s=S(); std::lock_guard<std::recursive_mutex> api_guard(s.api_mutex);std::call_once(s.once,init);ApiTimer timer{s.eigh_seconds,s.backend,s.eigh_profile};++s.zheevx_calls;
    const bool vectors=jobz && (*jobz=='V' || *jobz=='v');
    const bool all=range && (*range=='A' || *range=='a'),indices=range && (*range=='I' || *range=='i'),bounds=range && (*range=='V' || *range=='v');
    if(s.enabled && selected("VASP_METAL_EIGH") && s.backend && metal_backend_gpu_enabled(s.backend) &&
      jobz && (vectors || *jobz=='N' || *jobz=='n') && (all || indices || bounds) && uplo && (*uplo=='U' || *uplo=='u' || *uplo=='L' || *uplo=='l') &&
      n && *n>0 && lda && *lda>=*n && ldz && *ldz>=1 && (!vectors || *ldz>=*n) &&
      a && w && m && z && work && lwork && *lwork>=2LL*(*n) && rwork && iwork && ifail && info &&
      (!indices || (il && iu && *il>=1 && *iu>=*il && *iu<=*n)) && (!bounds || (vl && vu && *vu>*vl)) &&
      metal_backend_eigh_resources(s.backend,*n,*lda,0,*n,0)==METAL_BACKEND_OK) {
        ++s.zheevx_gpu_eligible;
        std::vector<metal_complex64> full((size_t)*n*(size_t)*n),vv(full.size());std::vector<double> ww(*n);
        bool upper=*uplo=='U' || *uplo=='u';
        for(int j=0;j<*n;++j) for(int i=0;i<*n;++i) {
            bool stored=upper ? i<=j:i>=j;auto x=stored ? a[i+static_cast<size_t>(j)*(*lda)]:a[j+static_cast<size_t>(i)*(*lda)];
            full[i+static_cast<size_t>(j)*(*n)]={x.real,i==j ? 0.0:(stored ? x.imag:-x.imag)};
        }
        int rc=metal_backend_hermitian_eigh_gpu(s.backend,*n,full.data(),*n,ww.data(),vv.data(),*n);
        if(rc==METAL_BACKEND_OK && metal_backend_last_gpu_success(s.backend)) {
            *m=0;
            for(int j=0;j<*n;++j) if(all || (indices && j>=*il-1 && j<*iu) || (bounds && ww[j]>*vl && ww[j]<=*vu)) {
                w[*m]=ww[j];if(vectors) for(int i=0;i<*n;++i) z[i+static_cast<size_t>(*m)*(*ldz)]={vv[i+static_cast<size_t>(j)*(*n)].real,vv[i+static_cast<size_t>(j)*(*n)].imag};
                ++*m;
            }
            if(vectors) std::fill(ifail,ifail+*m,0);
            work[0]={static_cast<double>(2LL*(*n)),0};*info=0;++s.zheevx_gpu_success;
            if(metal_backend_last_eigh_solver(s.backend)==1) ++s.lobpcg_gpu_success;
            if(metal_backend_last_eigh_solver(s.backend)==2) ++s.lobpcg_jacobi_rescue;
            s.lobpcg_iterations+=metal_backend_last_eigh_iterations(s.backend);
            return;
        }
    }
    static auto next=reinterpret_cast<zheevx_fn>(dlsym(RTLD_NEXT,"zheevx_"));
    if(next) next(jobz,range,uplo,n,a,lda,vl,vu,il,iu,abstol,m,w,z,ldz,work,lwork,rwork,iwork,ifail,info);
    else if(info) *info=-999;
}

// Opt-in local complex projector; Fortran owns layout/spinor/MPI eligibility.
extern "C" void *vasp_metal_nonlocal_create(size_t npl,size_t bands,size_t nions,size_t npro,size_t qchannels,size_t ntypes) {
 @autoreleasepool {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);std::call_once(s.once,init);
 const char *setting=std::getenv("VASP_METAL_NONLOCAL");
 if(!s.enabled || !setting || std::strcmp(setting,"1"))return nullptr;
 metal_nonlocal_batch *o=nullptr;
 if(metal_nonlocal_create(s.backend,{npl,bands,nions,npro,qchannels,ntypes},&o))return nullptr;
 return o;
 }
}
extern "C" void vasp_metal_nonlocal_destroy(void *owner) {
 @autoreleasepool {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);
 metal_nonlocal_destroy(static_cast<metal_nonlocal_batch *>(owner));
 }
}
extern "C" int vasp_metal_nonlocal_project(void *owner,uint64_t wave_generation,uint64_t phase_generation,
 const metal_complex64 *wave,size_t ldw,const double *radial,size_t qld,const metal_complex64 *phase,size_t phase_ld,
 const metal_complex64 *cq,const int *types,const int *lmmax,const int64_t *bases,const double *gradient,
 const metal_complex64 *multiplier,int channels,metal_complex64 *output,size_t ld) {
 @autoreleasepool {
 auto &s=S();std::lock_guard<std::recursive_mutex> guard(s.api_mutex);std::call_once(s.once,init);
 if(!s.enabled || !owner || !multiplier)return 0;
 ApiTimer timer{s.nonlocal_seconds,s.backend,s.nonlocal_profile};
 auto *o=static_cast<metal_nonlocal_batch *>(owner);metal_nonlocal_stats a{},b{};
 metal_nonlocal_get_stats(o,&a);
 metal_nonlocal_input in{wave_generation,phase_generation,wave,ldw,radial,qld,phase,phase_ld,cq,types,lmmax,bases,gradient,*multiplier};
 int rc=metal_nonlocal_project(o,&in,static_cast<unsigned>(channels),output,ld);
 metal_nonlocal_get_stats(o,&b);
 s.nonlocal_stats.calls+=b.calls-a.calls;
 s.nonlocal_stats.completed_commands+=b.completed_commands-a.completed_commands;
 s.nonlocal_stats.wave_import_bytes+=b.wave_import_bytes-a.wave_import_bytes;
 s.nonlocal_stats.radial_import_bytes+=b.radial_import_bytes-a.radial_import_bytes;
 s.nonlocal_stats.phase_import_bytes+=b.phase_import_bytes-a.phase_import_bytes;
 s.nonlocal_stats.metadata_import_bytes+=b.metadata_import_bytes-a.metadata_import_bytes;
 s.nonlocal_stats.gradient_import_bytes+=b.gradient_import_bytes-a.gradient_import_bytes;
 s.nonlocal_stats.export_bytes+=b.export_bytes-a.export_bytes;
 s.nonlocal_stats.wave_reuses+=b.wave_reuses-a.wave_reuses;
 s.nonlocal_stats.phase_reuses+=b.phase_reuses-a.phase_reuses;
 s.nonlocal_stats.windows+=b.windows-a.windows;
 s.nonlocal_stats.planned_workspace_peak_bytes=std::max(s.nonlocal_stats.planned_workspace_peak_bytes,b.planned_workspace_peak_bytes);
 if(!rc)++s.nonlocal_success;return rc==METAL_BACKEND_OK ? 1:0;
 }
}
