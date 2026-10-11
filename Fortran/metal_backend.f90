module metal_backend
  use iso_c_binding
  implicit none
  private
  public :: metal_backend_handle, metal_backend_create_f, metal_backend_destroy_f
  public :: metal_backend_zgemm_f, metal_backend_zfft_f, metal_backend_dsum_f
  public :: metal_backend_zdotc_f
  public :: metal_backend_zgemm_blas_f
  public :: metal_backend_zfft3d_f
  public :: metal_complex64
  public :: metal_backend_gpu_enabled_f, metal_backend_hermitian_eigh_gpu_f
  public :: metal_backend_generalized_eigh_gpu_f
  public :: metal_resident_handle, metal_backend_resident_create_f, metal_backend_resident_destroy_f
  public :: metal_backend_resident_begin_step_f, metal_backend_resident_upload_wave_f
  public :: metal_backend_resident_upload_potential_f, metal_backend_resident_upload_projectors_f
  public :: metal_backend_resident_fft_f, metal_backend_resident_apply_hamiltonian_f
  public :: metal_backend_resident_project_f, metal_backend_resident_orthogonalize_f
  public :: metal_backend_resident_end_step_f, metal_backend_resident_download_wave_f
  public :: metal_backend_resident_abort_step_f, metal_backend_resident_download_projection_f

  type, bind(C) :: metal_backend_handle
     type(c_ptr) :: raw = c_null_ptr
  end type
  type, bind(C) :: metal_resident_handle
     type(c_ptr) :: raw = c_null_ptr
  end type
  type, bind(C) :: metal_complex64
     real(c_double) :: real
     real(c_double) :: imag
  end type

  interface
     function metal_backend_create(raw, prefer_gpu) bind(C, name="metal_backend_create") result(code)
       import :: c_ptr, c_int
       type(c_ptr) :: raw
       integer(c_int), value :: prefer_gpu
       integer(c_int) :: code
     end function
     subroutine metal_backend_destroy(raw) bind(C, name="metal_backend_destroy")
       import :: c_ptr
       type(c_ptr), value :: raw
     end subroutine
     function metal_backend_zgemm(raw, m, n, k, a, lda, b, ldb, c, ldc) bind(C, name="metal_backend_zgemm") result(code)
       import :: c_ptr, c_size_t, c_int, metal_complex64
       type(c_ptr), value :: raw
       integer(c_size_t), value :: m, n, k, lda, ldb, ldc
       type(metal_complex64) :: a(*), b(*), c(*)
       integer(c_int) :: code
     end function
     function metal_backend_zgemm_blas(raw, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
       bind(C, name="metal_backend_zgemm_blas_f90") result(code)
       import :: c_ptr, c_size_t, c_int, c_char, metal_complex64
       type(c_ptr), value :: raw
       integer(c_int), value :: transa, transb
       integer(c_size_t), value :: m, n, k, lda, ldb, ldc
       type(metal_complex64) :: alpha, beta
       type(metal_complex64) :: a(*), b(*), c(*)
       integer(c_int) :: code
     end function
     function metal_backend_zfft(raw, n, input, output, direction) bind(C, name="metal_backend_zfft") result(code)
       import :: c_ptr, c_size_t, c_int, metal_complex64
       type(c_ptr), value :: raw
       integer(c_size_t), value :: n
       type(metal_complex64) :: input(*), output(*)
       integer(c_int), value :: direction
       integer(c_int) :: code
     end function
     function metal_backend_zfft3d(raw, nx, ny, nz, input, output, direction) bind(C, name="metal_backend_zfft3d") result(code)
       import :: c_ptr, c_size_t, c_int, metal_complex64
       type(c_ptr), value :: raw
       integer(c_size_t), value :: nx, ny, nz
       type(metal_complex64) :: input(*), output(*)
       integer(c_int), value :: direction
       integer(c_int) :: code
     end function
     function metal_backend_dsum(raw, n, input, result) bind(C, name="metal_backend_dsum") result(code)
       import :: c_ptr, c_size_t, c_int, c_double
       type(c_ptr), value :: raw
       integer(c_size_t), value :: n
       real(c_double) :: input(*), result
       integer(c_int) :: code
     end function
     function metal_backend_zdotc(raw,n,x,incx,y,incy,result) bind(C,name="metal_backend_zdotc") result(code)
       import :: c_ptr,c_size_t,c_int,c_intptr_t,metal_complex64
       type(c_ptr), value :: raw
       integer(c_size_t), value :: n
       integer(c_intptr_t), value :: incx,incy
       type(metal_complex64) :: x(*),y(*),result
       integer(c_int) :: code
     end function
     function metal_backend_gpu_enabled(raw) bind(C, name="metal_backend_gpu_enabled") result(enabled)
       import :: c_ptr, c_int
       type(c_ptr), value :: raw
       integer(c_int) :: enabled
     end function
     function metal_backend_resident_create(raw,nx,ny,nz,bands,npro,out) bind(C,name="metal_backend_resident_create") result(code)
       import :: c_ptr,c_size_t,c_int
       type(c_ptr), value :: raw
       integer(c_size_t), value :: nx,ny,nz,bands,npro
       type(c_ptr) :: out
       integer(c_int) :: code
     end function
     subroutine metal_backend_resident_destroy(raw) bind(C,name="metal_backend_resident_destroy")
       import :: c_ptr
       type(c_ptr), value :: raw
     end subroutine
     function metal_backend_resident_begin_step(raw) bind(C,name="metal_backend_resident_begin_step") result(code)
       import :: c_ptr,c_int
       type(c_ptr), value :: raw
       integer(c_int) :: code
     end function
     function metal_backend_resident_abort_step(raw) bind(C,name="metal_backend_resident_abort_step") result(code)
       import :: c_ptr,c_int
       type(c_ptr), value :: raw
       integer(c_int) :: code
     end function
     function metal_backend_resident_upload_wavefunction(raw,wave,ld,force) bind(C,name="metal_backend_resident_upload_wavefunction") result(code)
       import :: c_ptr,c_size_t,c_int,metal_complex64
       type(c_ptr), value :: raw
       type(metal_complex64) :: wave(*)
       integer(c_size_t), value :: ld
       integer(c_int), value :: force
       integer(c_int) :: code
     end function
     function metal_backend_resident_upload_potential(raw,potential,force) bind(C,name="metal_backend_resident_upload_potential") result(code)
       import :: c_ptr,c_int,metal_complex64
       type(c_ptr), value :: raw
       type(metal_complex64) :: potential(*)
       integer(c_int), value :: force
       integer(c_int) :: code
     end function
     function metal_backend_resident_upload_projectors(raw,projectors,ld,force) bind(C,name="metal_backend_resident_upload_projectors") result(code)
       import :: c_ptr,c_size_t,c_int,metal_complex64
       type(c_ptr), value :: raw
       type(metal_complex64) :: projectors(*)
       integer(c_size_t), value :: ld
       integer(c_int), value :: force
       integer(c_int) :: code
     end function
     function metal_backend_resident_fft(raw,direction) bind(C,name="metal_backend_resident_fft") result(code)
       import :: c_ptr,c_int
       type(c_ptr), value :: raw
       integer(c_int), value :: direction
       integer(c_int) :: code
     end function
     function metal_backend_resident_apply_hamiltonian(raw) bind(C,name="metal_backend_resident_apply_hamiltonian") result(code)
       import :: c_ptr,c_int
       type(c_ptr), value :: raw
       integer(c_int) :: code
     end function
     function metal_backend_resident_project(raw) bind(C,name="metal_backend_resident_project") result(code)
       import :: c_ptr,c_int
       type(c_ptr), value :: raw
       integer(c_int) :: code
     end function
     function metal_backend_resident_orthogonalize(raw) bind(C,name="metal_backend_resident_orthogonalize") result(code)
       import :: c_ptr,c_int
       type(c_ptr), value :: raw
       integer(c_int) :: code
     end function
     function metal_backend_resident_end_step(raw,wait_for_host) bind(C,name="metal_backend_resident_end_step") result(code)
       import :: c_ptr,c_int
       type(c_ptr), value :: raw
       integer(c_int), value :: wait_for_host
       integer(c_int) :: code
     end function
     function metal_backend_resident_download_wavefunction(raw,wave,ld) bind(C,name="metal_backend_resident_download_wavefunction") result(code)
       import :: c_ptr,c_size_t,c_int,metal_complex64
       type(c_ptr), value :: raw
       type(metal_complex64) :: wave(*)
       integer(c_size_t), value :: ld
       integer(c_int) :: code
     end function
     function metal_backend_resident_download_projection(raw,projection,ld) bind(C,name="metal_backend_resident_download_projection") result(code)
       import :: c_ptr,c_size_t,c_int,metal_complex64
       type(c_ptr), value :: raw
       type(metal_complex64) :: projection(*)
       integer(c_size_t), value :: ld
       integer(c_int) :: code
     end function
     function metal_backend_hermitian_eigh_gpu(raw,n,a,lda,w,v,ldv) bind(C,name="metal_backend_hermitian_eigh_gpu") result(code)
       import :: c_ptr, c_size_t, c_int, c_double, metal_complex64
       type(c_ptr), value :: raw
       integer(c_size_t), value :: n,lda,ldv
       type(metal_complex64) :: a(*),v(*)
       real(c_double) :: w(*)
       integer(c_int) :: code
     end function
     function metal_backend_generalized_eigh_gpu(raw,itype,uplo,jobz,n,a,lda,b,ldb,w,v,ldv) &
       bind(C,name="metal_backend_generalized_eigh_gpu_f90") result(code)
       import :: c_ptr, c_size_t, c_int, c_char, c_double, metal_complex64
       type(c_ptr), value :: raw
       integer(c_int), value :: itype
       integer(c_int), value :: uplo,jobz
       integer(c_size_t), value :: n,lda,ldb,ldv
       type(metal_complex64) :: a(*),b(*),v(*)
       real(c_double) :: w(*)
       integer(c_int) :: code
     end function
  end interface

contains
  subroutine metal_backend_create_f(handle, prefer_gpu, status)
    type(metal_backend_handle), intent(out) :: handle
    logical, intent(in) :: prefer_gpu
    integer(c_int), intent(out) :: status
    integer(c_int) :: flag
    flag = merge(1_c_int, 0_c_int, prefer_gpu)
    status = metal_backend_create(handle%raw, flag)
  end subroutine

  subroutine metal_backend_destroy_f(handle)
    type(metal_backend_handle), intent(inout) :: handle
    if (c_associated(handle%raw)) call metal_backend_destroy(handle%raw)
    handle%raw = c_null_ptr
  end subroutine

  integer(c_int) function metal_backend_zgemm_f(handle, m, n, k, a, lda, b, ldb, c, ldc) result(status)
    type(metal_backend_handle), intent(in) :: handle
    integer(c_size_t), intent(in) :: m, n, k, lda, ldb, ldc
    type(metal_complex64), intent(in) :: a(*), b(*)
    type(metal_complex64), intent(inout) :: c(*)
    status = metal_backend_zgemm(handle%raw, m, n, k, a, lda, b, ldb, c, ldc)
  end function

  integer(c_int) function metal_backend_zgemm_blas_f(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
    result(status)
    type(metal_backend_handle), intent(in) :: handle
    character(kind=c_char), intent(in) :: transa, transb
    integer(c_size_t), intent(in) :: m, n, k, lda, ldb, ldc
    type(metal_complex64), intent(in) :: alpha, beta
    type(metal_complex64), intent(in) :: a(*), b(*)
    type(metal_complex64), intent(inout) :: c(*)
    status = metal_backend_zgemm_blas(handle%raw, int(iachar(transa),c_int), int(iachar(transb),c_int), &
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  end function

  integer(c_int) function metal_backend_zfft_f(handle, n, input, output, direction) result(status)
    type(metal_backend_handle), intent(in) :: handle
    integer(c_size_t), intent(in) :: n
    type(metal_complex64), intent(in) :: input(*)
    type(metal_complex64), intent(out) :: output(*)
    integer(c_int), intent(in) :: direction
    status = metal_backend_zfft(handle%raw, n, input, output, direction)
  end function

  integer(c_int) function metal_backend_zfft3d_f(handle, nx, ny, nz, input, output, direction) result(status)
    type(metal_backend_handle), intent(in) :: handle
    integer(c_size_t), intent(in) :: nx, ny, nz
    type(metal_complex64), intent(in) :: input(*)
    type(metal_complex64), intent(out) :: output(*)
    integer(c_int), intent(in) :: direction
    status = metal_backend_zfft3d(handle%raw, nx, ny, nz, input, output, direction)
  end function

  integer(c_int) function metal_backend_dsum_f(handle, n, input, result) result(status)
    type(metal_backend_handle), intent(in) :: handle
    integer(c_size_t), intent(in) :: n
    real(c_double), intent(in) :: input(*)
    real(c_double), intent(out) :: result
    status = metal_backend_dsum(handle%raw, n, input, result)
  end function
  integer(c_int) function metal_backend_zdotc_f(handle,n,x,incx,y,incy,result) result(status)
    type(metal_backend_handle), intent(in) :: handle
    integer(c_size_t), intent(in) :: n
    integer(c_intptr_t), intent(in) :: incx,incy
    type(metal_complex64), intent(in) :: x(*),y(*)
    type(metal_complex64), intent(out) :: result
    status=metal_backend_zdotc(handle%raw,n,x,incx,y,incy,result)
  end function

  logical function metal_backend_gpu_enabled_f(handle) result(enabled)
    type(metal_backend_handle), intent(in) :: handle
    enabled=metal_backend_gpu_enabled(handle%raw)==1_c_int
  end function

  integer(c_int) function metal_backend_hermitian_eigh_gpu_f(handle,n,a,lda,w,v,ldv) result(status)
    type(metal_backend_handle), intent(in) :: handle
    integer(c_size_t), intent(in) :: n,lda,ldv
    type(metal_complex64), intent(in) :: a(*)
    type(metal_complex64), intent(out) :: v(*)
    real(c_double), intent(out) :: w(*)
    status=metal_backend_hermitian_eigh_gpu(handle%raw,n,a,lda,w,v,ldv)
  end function

  integer(c_int) function metal_backend_generalized_eigh_gpu_f(handle,itype,uplo,jobz,n,a,lda,b,ldb,w,v,ldv) result(status)
    type(metal_backend_handle), intent(in) :: handle
    integer(c_int), intent(in) :: itype
    character(kind=c_char), intent(in) :: uplo,jobz
    integer(c_size_t), intent(in) :: n,lda,ldb,ldv
    type(metal_complex64), intent(in) :: a(*),b(*)
    type(metal_complex64), intent(out) :: v(*)
    real(c_double), intent(out) :: w(*)
    status=metal_backend_generalized_eigh_gpu(handle%raw,itype,int(iachar(uplo),c_int),int(iachar(jobz),c_int), &
      n,a,lda,b,ldb,w,v,ldv)
  end function

  subroutine metal_backend_resident_create_f(handle,nx,ny,nz,bands,npro,resident,status)
    type(metal_backend_handle), intent(in) :: handle
    integer(c_size_t), intent(in) :: nx,ny,nz,bands,npro
    type(metal_resident_handle), intent(out) :: resident
    integer(c_int), intent(out) :: status
    status=metal_backend_resident_create(handle%raw,nx,ny,nz,bands,npro,resident%raw)
  end subroutine
  subroutine metal_backend_resident_destroy_f(resident)
    type(metal_resident_handle), intent(inout) :: resident
    if(c_associated(resident%raw)) call metal_backend_resident_destroy(resident%raw)
    resident%raw=c_null_ptr
  end subroutine
  integer(c_int) function metal_backend_resident_begin_step_f(resident) result(status)
    type(metal_resident_handle), intent(in) :: resident
    status=metal_backend_resident_begin_step(resident%raw)
  end function
  integer(c_int) function metal_backend_resident_abort_step_f(resident) result(status)
    type(metal_resident_handle), intent(in) :: resident
    status=metal_backend_resident_abort_step(resident%raw)
  end function
  integer(c_int) function metal_backend_resident_upload_wave_f(resident,wave,ld,force) result(status)
    type(metal_resident_handle), intent(in) :: resident
    type(metal_complex64), intent(in) :: wave(*)
    integer(c_size_t), intent(in) :: ld
    logical, intent(in) :: force
    status=metal_backend_resident_upload_wavefunction(resident%raw,wave,ld,merge(1_c_int,0_c_int,force))
  end function
  integer(c_int) function metal_backend_resident_upload_potential_f(resident,potential,force) result(status)
    type(metal_resident_handle), intent(in) :: resident
    type(metal_complex64), intent(in) :: potential(*)
    logical, intent(in) :: force
    status=metal_backend_resident_upload_potential(resident%raw,potential,merge(1_c_int,0_c_int,force))
  end function
  integer(c_int) function metal_backend_resident_upload_projectors_f(resident,projectors,ld,force) result(status)
    type(metal_resident_handle), intent(in) :: resident
    type(metal_complex64), intent(in) :: projectors(*)
    integer(c_size_t), intent(in) :: ld
    logical, intent(in) :: force
    status=metal_backend_resident_upload_projectors(resident%raw,projectors,ld,merge(1_c_int,0_c_int,force))
  end function
  integer(c_int) function metal_backend_resident_fft_f(resident,direction) result(status)
    type(metal_resident_handle), intent(in) :: resident
    integer(c_int), intent(in) :: direction
    status=metal_backend_resident_fft(resident%raw,direction)
  end function
  integer(c_int) function metal_backend_resident_apply_hamiltonian_f(resident) result(status)
    type(metal_resident_handle), intent(in) :: resident
    status=metal_backend_resident_apply_hamiltonian(resident%raw)
  end function
  integer(c_int) function metal_backend_resident_project_f(resident) result(status)
    type(metal_resident_handle), intent(in) :: resident
    status=metal_backend_resident_project(resident%raw)
  end function
  integer(c_int) function metal_backend_resident_orthogonalize_f(resident) result(status)
    type(metal_resident_handle), intent(in) :: resident
    status=metal_backend_resident_orthogonalize(resident%raw)
  end function
  integer(c_int) function metal_backend_resident_end_step_f(resident,wait_for_host) result(status)
    type(metal_resident_handle), intent(in) :: resident
    logical, intent(in) :: wait_for_host
    status=metal_backend_resident_end_step(resident%raw,merge(1_c_int,0_c_int,wait_for_host))
  end function
  integer(c_int) function metal_backend_resident_download_wave_f(resident,wave,ld) result(status)
    type(metal_resident_handle), intent(in) :: resident
    type(metal_complex64), intent(inout) :: wave(*)
    integer(c_size_t), intent(in) :: ld
    status=metal_backend_resident_download_wavefunction(resident%raw,wave,ld)
  end function
  integer(c_int) function metal_backend_resident_download_projection_f(resident,projection,ld) result(status)
    type(metal_resident_handle), intent(in) :: resident
    type(metal_complex64), intent(inout) :: projection(*)
    integer(c_size_t), intent(in) :: ld
    status=metal_backend_resident_download_projection(resident%raw,projection,ld)
  end function
end module
