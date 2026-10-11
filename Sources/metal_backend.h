#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct metal_complex64 {
    double real;
    double imag;
} metal_complex64;

typedef struct metal_backend metal_backend;
/* A resident context owns device buffers across steps. Shape is immutable;
   the backend must outlive the context. Destroy requires exclusive ownership. */
typedef struct metal_resident metal_resident;

enum {
    METAL_BACKEND_OK = 0,
    METAL_BACKEND_INVALID_ARGUMENT = 1,
    METAL_BACKEND_ALLOCATION_FAILED = 2,
    METAL_BACKEND_RUNTIME_FAILED = 3,
    METAL_BACKEND_UNSUPPORTED = 4
};

int metal_backend_create(metal_backend **out, int prefer_gpu);
void metal_backend_destroy(metal_backend *backend);
int metal_backend_gpu_enabled(const metal_backend *backend);
/* Runtime-selected shader profile. This is a capability profile, not a
   product-name claim: generic, metal4-device-tuned, or cpu-unavailable. */
const char *metal_backend_variant(const metal_backend *backend);
int metal_backend_metal4_enabled(const metal_backend *backend);
/* Opt-in command API diagnostics, distinct from the device's Metal 4 family.
   calls counts helper dispatch attempts, success requires GPU completion, and
   fallbacks counts production GEMMs routed back to the legacy encoder. */
int metal_backend_metal4_command_requested(const metal_backend *backend);
int metal_backend_metal4_command_available(const metal_backend *backend);
uint64_t metal_backend_metal4_command_calls(const metal_backend *backend);
uint64_t metal_backend_metal4_command_success(const metal_backend *backend);
uint64_t metal_backend_metal4_command_fallbacks(const metal_backend *backend);
const char *metal_backend_metal4_command_reason(const metal_backend *backend);
/* Verified opt-in offline library; rejected artifacts fall back to runtime
   compilation. verified covers metadata/hash gates, used requires disk load
   and all kernel pipelines to succeed. */
int metal_backend_precompiled_requested(const metal_backend *backend);
int metal_backend_precompiled_verified(const metal_backend *backend);
int metal_backend_precompiled_used(const metal_backend *backend);
unsigned metal_backend_precompiled_fallbacks(const metal_backend *backend);
const char *metal_backend_precompiled_reason(const metal_backend *backend);
const char *metal_backend_precompiled_path(const metal_backend *backend);
/* Metal 4 machine-learning command API availability only. This is not proof
   that an ANE can execute exact VASP arithmetic. */
int metal_backend_ml_api_available(const metal_backend *backend);
int metal_backend_device_tuned(const metal_backend *backend);
const char *metal_backend_language_version(const metal_backend *backend);
unsigned metal_backend_compile_fallbacks(const metal_backend *backend);
int metal_backend_tiled_gemm_enabled(const metal_backend *backend);
uint64_t metal_backend_tiled_gemm_calls(const metal_backend *backend);
uint64_t metal_backend_tiled_gemm_fallbacks(const metal_backend *backend);
int metal_backend_tiled_gemm_autoselected(const metal_backend *backend);
int metal_backend_gemm_calibrated(const metal_backend *backend);
/* True only when the most recent operation completed on Metal; false for CPU fallback. */
int metal_backend_last_gpu_success(const metal_backend *backend);
const char *metal_backend_last_error(const metal_backend *backend);
/* 0=Jacobi, 1=converged LOBPCG, 2=Jacobi rescue after failed LOBPCG. */
int metal_backend_last_eigh_solver(const metal_backend *backend);
uint64_t metal_backend_command_count(const metal_backend *backend);
uint64_t metal_backend_wait_count(const metal_backend *backend);
double metal_backend_wait_seconds(const metal_backend *backend);
double metal_backend_conversion_seconds(const metal_backend *backend);
double metal_backend_gpu_seconds(const metal_backend *backend);
uint64_t metal_backend_last_eigh_iterations(const metal_backend *backend);
int metal_backend_try_gpu(const metal_backend *backend);

/* Resident FFT, local potential, dense projection and Euclidean orthogonalization.
   These primitives do not implement the full VASP PAW Hamiltonian/overlap.
   The API is safe to use
   only for one in-flight step per context.  begin_step() opens one command
   buffer; the encode calls append work to that buffer and end_step() submits
   and waits only when host visibility is requested.  A failed encode leaves
   caller-owned host arrays untouched. Abort a failed step before reuse.
   Uploads must precede encoding. force=0 retains that buffer's existing data;
   callers must set force=1 after changing host values. npro=0 is FFT-only.
   begin_step preserves the last device result. Both FFT signs are unnormalized;
   forward=-1, inverse=+1, so an FFT round trip multiplies by nx*ny*nz.
   Downloads require a closed step and wait for pending work before writing. */
int metal_backend_resident_create(const metal_backend *backend,
                                  size_t nx, size_t ny, size_t nz,
                                  size_t bands, size_t npro,
                                  metal_resident **out);
void metal_backend_resident_destroy(metal_resident *resident);
int metal_backend_resident_begin_step(metal_resident *resident);
int metal_backend_resident_abort_step(metal_resident *resident);
int metal_backend_resident_upload_wavefunction(metal_resident *resident,
                                               const metal_complex64 *wave,
                                               size_t ld, int force);
int metal_backend_resident_upload_potential(metal_resident *resident,
                                            const metal_complex64 *potential,
                                            int force);
int metal_backend_resident_upload_projectors(metal_resident *resident,
                                              const metal_complex64 *projectors,
                                              size_t ld, int force);
int metal_backend_resident_fft(metal_resident *resident, int direction);
int metal_backend_resident_apply_hamiltonian(metal_resident *resident);
int metal_backend_resident_project(metal_resident *resident);
int metal_backend_resident_orthogonalize(metal_resident *resident);
int metal_backend_resident_end_step(metal_resident *resident, int wait_for_host);
int metal_backend_resident_download_wavefunction(metal_resident *resident,
                                                 metal_complex64 *wave,
                                                 size_t ld);
/* Column-major npro x bands coefficients, <projector|wave>, leading dim ld. */
int metal_backend_resident_download_projection(metal_resident *resident,
                                               metal_complex64 *projection,
                                               size_t ld);
uint64_t metal_backend_resident_generation(const metal_resident *resident);
uint64_t metal_backend_resident_steps(const metal_resident *resident);
uint64_t metal_backend_resident_uploads(const metal_resident *resident);
uint64_t metal_backend_resident_submissions(const metal_resident *resident);
uint64_t metal_backend_resident_waits(const metal_resident *resident);

/* Column-major complex GEMM: C = A B. The arrays use real/imag pairs. */
int metal_backend_zgemm(const metal_backend *backend,
                        size_t m, size_t n, size_t k,
                        const metal_complex64 *a, size_t lda,
                        const metal_complex64 *b, size_t ldb,
                        metal_complex64 *c, size_t ldc);

/* BLAS-compatible complex GEMM. transa/transb accept N, T, or C. */
int metal_backend_zgemm_blas(const metal_backend *backend,
                             char transa, char transb,
                             size_t m, size_t n, size_t k,
                             metal_complex64 alpha,
                             const metal_complex64 *a, size_t lda,
                             const metal_complex64 *b, size_t ldb,
                             metal_complex64 beta,
                             metal_complex64 *c, size_t ldc);

/* GPU-only variant. Failure leaves C unchanged, enabling an atomic fallback. */
int metal_backend_zgemm_blas_gpu(const metal_backend *,char,char,size_t,size_t,size_t,
    metal_complex64,const metal_complex64 *,size_t,const metal_complex64 *,size_t,
    metal_complex64,metal_complex64 *,size_t);

/* Chunk counters cover these ABI staging paths, not all VASP communication.
   Workspace fields are planned active staging bounds, not sampled RSS. */
typedef struct {
    uint64_t fft_calls,fft_windows,gemm_calls,gemm_windows;
    uint64_t fft_peak_workspace_bytes,gemm_peak_workspace_bytes;
} metal_chunk_stats;
int metal_backend_get_chunk_stats(const metal_backend *,metal_chunk_stats *);
/* Additive ABI. Allocation-time samples of the device's resource accounting,
   including other caches/owners on that device. Not RSS, CPU staging, other
   processes, or a continuous unified-memory measurement. */
typedef struct {
    uint64_t current_device_allocated_bytes, observed_device_peak_bytes;
    uint64_t device_budget_bytes, allocation_requests, budget_rejections;
    uint64_t driver_allocation_failures, pressure_eviction_attempts;
} metal_memory_stats;
int metal_backend_get_memory_stats(const metal_backend *,metal_memory_stats *);
/* Construct host telemetry before static owners/exit reporters that consume it.
   This allocates no Metal resources and does not enable the GPU. */
void metal_backend_prepare_memory_stats(void);
/* Completed ordinary or fused-owner commands that used grouped overlap. */
uint64_t metal_backend_cooperative_orth_calls(const metal_backend *);
/* Axis batches encoded with checked32-bit or64-bit FFT addressing. */
uint64_t metal_backend_fft_compact_axes(const metal_backend *);
uint64_t metal_backend_fft_wide_axes(const metal_backend *);

/* In-place/out-of-place complex FFT. direction is -1 (forward) or +1 (inverse).
   The inverse includes 1/n normalization. */
int metal_backend_zfft(const metal_backend *backend,
                       size_t n, const metal_complex64 *input,
                       metal_complex64 *output, int direction);

/* Contiguous VASP/FFTW layout: x + nx*(y + ny*z). */
int metal_backend_zfft3d(const metal_backend *backend,
                         size_t nx, size_t ny, size_t nz,
                         const metal_complex64 *input,
                         metal_complex64 *output, int direction);

/* Batched VASP/FFTW layout. Each transform starts batch_stride elements after
   the previous one; input and output may alias only within each transform. */
int metal_backend_zfft3d_batch(const metal_backend *backend,
                               size_t nx, size_t ny, size_t nz,
                               size_t nbatch, size_t batch_stride,
                               const metal_complex64 *input,
                               metal_complex64 *output, int direction);

/* GPU-only; failed calls leave output untouched. normalize=0 gives VASP/FFTW convention. */
int metal_backend_zfft3d_batch_gpu(const metal_backend *,size_t,size_t,size_t,size_t,size_t,
    const metal_complex64 *,metal_complex64 *,int,int);

/* One-submit collinear reciprocal-PAW Hamiltonian. All intermediate fields
   remain on GPU until the completed result is downloaded. Inputs are fresh
   per call; static PAW radial/phase data obey explicit invalidate epochs. */
void metal_backend_hamiltonian_invalidate(metal_backend *);
uint64_t metal_backend_hamiltonian_static_uploads(const metal_backend *);
uint64_t metal_backend_hamiltonian_static_reuses(const metal_backend *);
int metal_backend_hamiltonian_gpu(const metal_backend *,size_t nx,size_t ny,size_t nz,
 size_t npl,size_t bands,size_t nions,size_t npro,size_t lmdim,size_t ntypes,
 size_t qld,size_t qchannels,size_t phase_ld,int real_fields,int from_pw,const void *sv,
 const metal_complex64 *const *cr,const metal_complex64 *const *cw,const metal_complex64 *const *proj,
 const int *indices,const double *kinetic,const double *eig,const void *d,const void *q,
 const int *types,const int *lmmax,const int *lmbase,const double *radial,
 const metal_complex64 *cq,const metal_complex64 *phase,metal_complex64 *out,size_t ldout);

/* Plane-wave and PAW metric overlap in one command, output += A^H B + P^H Q. */
int metal_backend_orth_overlap_gpu(const metal_backend *,size_t,size_t,size_t,size_t,
 const metal_complex64 *,const metal_complex64 *,size_t,const metal_complex64 *,const metal_complex64 *,size_t,metal_complex64 *,size_t);
/* Preflight without host-array reads; no fixed matrix-dimension threshold. */
int metal_backend_eigh_resources(const metal_backend *, size_t n, size_t lda,
                                 size_t ldb, size_t ldv, int generalized);
/* Complete collinear PAW-metric Cholesky orthogonalization within resources. */
int metal_backend_full_orth_gpu(const metal_backend *,size_t,size_t,size_t,size_t,size_t,
 int,const void *,const int *,const int *,metal_complex64 *,size_t,metal_complex64 *,size_t);
int metal_backend_dsum(const metal_backend *backend,
                       size_t n, const double *input, double *result);

/* Conjugated complex dot product. Strides follow BLAS, including negative
   increments; x/y point to the start of the allocated vector storage. */
int metal_backend_zdotc(const metal_backend *backend, size_t n,
                         const metal_complex64 *x, ptrdiff_t incx,
                         const metal_complex64 *y, ptrdiff_t incy,
                         metal_complex64 *result);
int metal_backend_zdotc_gpu(const metal_backend *backend, size_t n,
                             const metal_complex64 *x, ptrdiff_t incx,
                             const metal_complex64 *y, ptrdiff_t incy,
                             metal_complex64 *result);

/* Apply VASP's local real-space potential to one or two complex spinors. */
int metal_backend_vh_apply(const metal_backend *backend, size_t np, size_t mplwv,
                           size_t nrspinors, const metal_complex64 *sv,
                           const metal_complex64 *cr, metal_complex64 *cvr,
                           double rinplw);
/* Batched local-potential application.  cr/cvr are column-major batches with
   batch stride mplwv*nrspinors; one Metal command covers all bands. */
int metal_backend_vh_apply_batch(const metal_backend *backend, size_t np, size_t mplwv,
                                 size_t nrspinors, size_t nbands,
                                 const metal_complex64 *sv,
                                 const metal_complex64 *cr, metal_complex64 *cvr,
                                 double rinplw);
int metal_backend_vh_apply_real(const metal_backend *backend, size_t np, size_t mplwv,
                                size_t nrspinors, const double *sv,
                                const metal_complex64 *cr, metal_complex64 *cvr,
                                double rinplw);
int metal_backend_vh_apply_real_batch(const metal_backend *backend, size_t np, size_t mplwv,
                                      size_t nrspinors, size_t nbands,
                                      const double *sv,
                                      const metal_complex64 *cr, metal_complex64 *cvr,
                                      double rinplw);

/* Apply VASP's local real-space potential to one or two complex spinors.
   sv is column-major with leading dimension mplwv and nrspinors^2 channels;
   cr/cvr contain nrspinors wavefunction channels with the same leading dimension.
   The operation is CVR = SV * CR / GRID%NPLWV on the first np real-space points. */
int metal_backend_vh_apply(const metal_backend *backend, size_t np, size_t mplwv,
                           size_t nrspinors, const metal_complex64 *sv,
                           const metal_complex64 *cr, metal_complex64 *cvr,
                           double rinplw);

/* Small Hermitian eigensolver reference path. Eigenvectors are column-major. */
int metal_backend_hermitian_eigh(const metal_backend *backend,
                                 size_t n, const metal_complex64 *a,
                                 size_t lda, double *eigenvalues,
                                 metal_complex64 *eigenvectors, size_t ldv);

/* GPU-only standard Hermitian eigensolver. Returns METAL_BACKEND_UNSUPPORTED
   when no Metal device is active; it never silently falls back to CPU. */
int metal_backend_hermitian_eigh_gpu(const metal_backend *backend,
                                     size_t n, const metal_complex64 *a,
                                     size_t lda, double *eigenvalues,
                                     metal_complex64 *eigenvectors, size_t ldv);

/* Optional factor output in the requested triangle, preserving ZHEGV's B.
   All Cholesky/transform/back-transform arithmetic executes on Metal. */
int metal_backend_generalized_all_gpu(const metal_backend *backend);
int metal_backend_generalized_eigh_factor_gpu(const metal_backend *backend,int itype,char uplo,char jobz,
    size_t n,const metal_complex64 *a,size_t lda,const metal_complex64 *b,size_t ldb,
    double *w,metal_complex64 *v,size_t ldv,metal_complex64 *factor,size_t ldf);

/* GPU-only generalized Hermitian-definite eigensolver for itype=1.
   The transformation uses B=L L^H and solves A x = lambda B x. */
int metal_backend_generalized_eigh_gpu(const metal_backend *backend,
                                       int itype, char uplo, char jobz,
                                       size_t n, const metal_complex64 *a,
                                       size_t lda, const metal_complex64 *b,
                                       size_t ldb, double *eigenvalues,
                                       metal_complex64 *eigenvectors,
                                       size_t ldv);

#ifdef __cplusplus
}
#endif
