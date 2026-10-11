#include "metal_backend.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main() {
    metal_backend *b=nullptr;assert(!metal_backend_create(&b,1));
    if(!metal_backend_gpu_enabled(b))return std::getenv("METAL_REQUIRE_GPU") ? 1:77;
    const size_t bands=5,stride=64,total=60;const metal_complex64 pad{827,-932};
    std::vector<metal_complex64> input(bands*stride,pad),compact(input.size(),pad),wide(input.size(),pad),returned(input.size(),pad);
    for(size_t band=0;band<bands;++band)for(size_t k=0;k<total;++k)input[k+stride*band]={std::sin(.13*k+band),std::cos(.19*k-band)};
    setenv("VASP_METAL_FFT_CHUNK_BYTES","1920",1);
    setenv("VASP_METAL_FFT_INDEX_WIDTH","auto",1);
    assert(!metal_backend_zfft3d_batch_gpu(b,5,4,3,bands,stride,input.data(),compact.data(),-1,0));
    assert(metal_backend_fft_compact_axes(b)>=9 && !metal_backend_fft_wide_axes(b));
    setenv("VASP_METAL_FFT_INDEX_WIDTH","wide",1);
    assert(!metal_backend_zfft3d_batch_gpu(b,5,4,3,bands,stride,input.data(),wide.data(),-1,0));
    assert(metal_backend_fft_wide_axes(b)>=9);
    for(size_t i=0;i<compact.size();++i)assert(std::hypot(compact[i].real-wide[i].real,compact[i].imag-wide[i].imag)<1e-10);
    assert(!metal_backend_zfft3d_batch_gpu(b,5,4,3,bands,stride,compact.data(),returned.data(),1,1));
    for(size_t i=0;i<input.size();++i)assert(std::hypot(returned[i].real-input[i].real,returned[i].imag-input[i].imag)<1e-10);
    unsetenv("VASP_METAL_FFT_INDEX_WIDTH");unsetenv("VASP_METAL_FFT_CHUNK_BYTES");
    metal_backend_destroy(b);std::puts("Actual checked32/wide64 chunked FFT parity, padding and cross-width roundtrip passed");
}
