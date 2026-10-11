#include "metal_memory_allocator.hpp"
#include "metal_backend_private.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
int main() {
    metal_backend *backend=nullptr;assert(!metal_backend_create(&backend,1));
    if(!metal_backend_gpu_enabled(backend))return std::getenv("METAL_REQUIRE_GPU") ? 1:77;
    void *raw=nullptr,*queue=nullptr,*library=nullptr;
    assert(!metal_backend_copy_metal_resources(backend,&raw,&queue,&library));
    id<MTLDevice> device=(__bridge id<MTLDevice>)raw;
    metal_backend_evict_idle_buffers(backend);
    id<MTLBuffer> held=metal_memory::allocate(device,2*1024*1024,MTLResourceStorageModeShared);
    assert(held);metal_memory_stats before{};assert(!metal_backend_get_memory_stats(backend,&before));
    const auto constrained=std::to_string(before.current_device_allocated_bytes+128*1024);
    setenv("VASP_METAL_MEMORY_BUDGET_BYTES",constrained.c_str(),1);
    // External/live resources must be included in the common device budget.
    assert(!metal_memory::allocate(device,256*1024,MTLResourceStorageModeShared));
    assert(held.length==2*1024*1024);
    const size_t n=64;
    std::vector<metal_complex64> a(n*n,{0,0}),b(n*n,{1,0}),c(n*n,{7,0});
    for(size_t i=0;i<n;++i)a[i+n*i]={1,0};auto original=c;
    assert(metal_backend_zgemm_blas_gpu(backend,'N','N',n,n,n,{1,0},a.data(),n,b.data(),n,{1,0},c.data(),n)!=0);
    assert(!metal_backend_last_gpu_success(backend));
    for(size_t i=0;i<c.size();++i)assert(c[i].real==original[i].real && c[i].imag==original[i].imag);
    metal_memory_stats rejected{};assert(!metal_backend_get_memory_stats(backend,&rejected));
    assert(rejected.budget_rejections>=before.budget_rejections+2);
    assert(rejected.pressure_eviction_attempts>before.pressure_eviction_attempts);
    // Recovery preserves the beta=1 contract; failure committed no partial C.
    unsetenv("VASP_METAL_MEMORY_BUDGET_BYTES");
    assert(!metal_backend_zgemm_blas_gpu(backend,'N','N',n,n,n,{1,0},a.data(),n,b.data(),n,{1,0},c.data(),n));
    for(auto value:c)assert(std::abs(value.real-8)<1e-12 && std::abs(value.imag)<1e-12);
    metal_backend_evict_idle_buffers(backend);
    metal_memory_stats final{};assert(!metal_backend_get_memory_stats(backend,&final));
    assert(final.observed_device_peak_bytes>=before.current_device_allocated_bytes);
    std::printf("Device-accounted pressure rejection, active-buffer retention, atomic GEMM and recovery passed; peak=%llu\n",
                (unsigned long long)final.observed_device_peak_bytes);
    held=nil;metal_backend_destroy(backend);
}
