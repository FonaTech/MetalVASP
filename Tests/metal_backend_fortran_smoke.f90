program metal_backend_fortran_smoke
  use iso_c_binding
  use metal_backend
  implicit none
  type(metal_backend_handle) :: handle
  type(metal_complex64) :: a(4), b(4), c(4)
  integer(c_int) :: status
  real(c_double) :: w(2)
  type(metal_complex64) :: eigvec(4),alpha,beta,dot
  type(metal_resident_handle) :: resident
  type(metal_complex64) :: resident_wave(2),resident_potential(2),resident_projectors(2),resident_projection(1)
  integer(c_size_t) :: two
  character(len=16) :: require_gpu
  integer :: env_status

  two = 2_c_size_t
  call metal_backend_create_f(handle, .true., status)
  if (status /= 0_c_int) error stop 10
  call get_environment_variable("METAL_REQUIRE_GPU",require_gpu,status=env_status)
  if(env_status==0 .and. .not.metal_backend_gpu_enabled_f(handle)) error stop 99

  call metal_backend_resident_create_f(handle,two,1_c_size_t,1_c_size_t,1_c_size_t,1_c_size_t,resident,status)
  if(metal_backend_gpu_enabled_f(handle)) then
    if(status/=0_c_int) error stop 40
    resident_wave(1)=metal_complex64(1.0_c_double,2.0_c_double)
    resident_wave(2)=metal_complex64(3.0_c_double,-1.0_c_double)
    resident_potential=metal_complex64(2.0_c_double,0.0_c_double)
    resident_projectors(1)=metal_complex64(1.0_c_double,0.0_c_double)
    resident_projectors(2)=metal_complex64(0.0_c_double,1.0_c_double)
    status=metal_backend_resident_begin_step_f(resident)
    if(status/=0_c_int) error stop 41
    status=metal_backend_resident_upload_wave_f(resident,resident_wave,two,.false.)
    if(status/=0_c_int) error stop 42
    status=metal_backend_resident_upload_potential_f(resident,resident_potential,.false.)
    if(status/=0_c_int) error stop 43
    status=metal_backend_resident_upload_projectors_f(resident,resident_projectors,two,.false.)
    if(status/=0_c_int) error stop 44
    status=metal_backend_resident_apply_hamiltonian_f(resident)
    if(status/=0_c_int) error stop 45
    status=metal_backend_resident_project_f(resident)
    if(status/=0_c_int) error stop 46
    status=metal_backend_resident_end_step_f(resident,.false.)
    if(status/=0_c_int) error stop 47
    status=metal_backend_resident_download_projection_f(resident,resident_projection,1_c_size_t)
    if(status/=0_c_int) error stop 48
    if(abs(resident_projection(1)%real)>1.0e-12_c_double) error stop 49
    if(abs(resident_projection(1)%imag+2.0_c_double)>1.0e-12_c_double) error stop 50
    status=metal_backend_resident_download_wave_f(resident,resident_wave,two)
    if(status/=0_c_int) error stop 51
    if(abs(resident_wave(1)%real-2.0_c_double)>1.0e-12_c_double) error stop 52
    status=metal_backend_resident_begin_step_f(resident)
    if(status/=0_c_int) error stop 53
    status=metal_backend_resident_abort_step_f(resident)
    if(status/=0_c_int) error stop 54
  else
    if(status/=4_c_int .or. c_associated(resident%raw)) error stop 55
  end if
  call metal_backend_resident_destroy_f(resident)
  if(c_associated(resident%raw)) error stop 56

  a(1) = metal_complex64(1.0_c_double, 2.0_c_double)
  a(2) = metal_complex64(3.0_c_double, -1.0_c_double)
  a(3) = metal_complex64(-2.0_c_double, 0.5_c_double)
  a(4) = metal_complex64(4.0_c_double, 1.0_c_double)
  b(1) = metal_complex64(0.5_c_double, -1.0_c_double)
  b(2) = metal_complex64(2.0_c_double, 3.0_c_double)
  b(3) = metal_complex64(-1.0_c_double, 2.0_c_double)
  b(4) = metal_complex64(1.5_c_double, -0.25_c_double)
  c = metal_complex64(0.0_c_double, 0.0_c_double)

  status = metal_backend_zgemm_f(handle, two, two, two, a, two, b, two, c, two)
  if (status /= 0_c_int) error stop 11
  if (abs(c(1)%real + 3.0_c_double) > 1.0e-12_c_double) error stop 12
  if (abs(c(1)%imag + 5.0_c_double) > 1.0e-12_c_double) error stop 13

  status=metal_backend_zdotc_f(handle,two,a,1_c_intptr_t,b,1_c_intptr_t,dot)
  if (status/=0_c_int) error stop 30
  if (abs(dot%real-1.5_c_double)>1.0e-12_c_double) error stop 31
  if (abs(dot%imag-9.0_c_double)>1.0e-12_c_double) error stop 32

  alpha=metal_complex64(1.0_c_double,0.0_c_double)
  beta=metal_complex64(0.0_c_double,0.0_c_double)
  status=metal_backend_zgemm_blas_f(handle,'N','N',two,two,two,alpha,a,two,b,two,beta,c,two)
  if (status/=0_c_int) error stop 14
  if (abs(c(1)%real+3.0_c_double)>1.0e-12_c_double) error stop 15
  a(1)=metal_complex64(2.0_c_double,0.0_c_double)
  a(2)=metal_complex64(1.0_c_double,-1.0_c_double)
  a(3)=metal_complex64(1.0_c_double,1.0_c_double)
  a(4)=metal_complex64(3.0_c_double,0.0_c_double)
  status=metal_backend_hermitian_eigh_gpu_f(handle,two,a,two,w,eigvec,two)
  if (metal_backend_gpu_enabled_f(handle)) then
     if (status/=0_c_int) error stop 20
     if (abs(w(1)-1.0_c_double)>1.0e-10_c_double) error stop 21
     if (abs(w(2)-4.0_c_double)>1.0e-10_c_double) error stop 22
     b=metal_complex64(0.0_c_double,0.0_c_double)
     b(1)=metal_complex64(1.0_c_double,0.0_c_double)
     b(4)=metal_complex64(1.0_c_double,0.0_c_double)
     status=metal_backend_generalized_eigh_gpu_f(handle,1_c_int,'U','V',two,a,two,b,two,w,eigvec,two)
     print *, "Fortran generalized status=",status," w=",w
     if (status/=0_c_int) error stop 23
     if (abs(w(1)-1.0_c_double)>1.0e-10_c_double) error stop 24
  else
     if (status/=4_c_int) error stop 25
  end if
  call metal_backend_destroy_f(handle)
end program
