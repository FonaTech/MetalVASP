#include "metal_backend.h"
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main() {
    metal_backend *b=nullptr;assert(metal_backend_create(&b,1)==0);
    if(!metal_backend_gpu_enabled(b)){metal_backend_destroy(b);return std::getenv("METAL_REQUIRE_GPU") ? 1:77;}
    const size_t n=60,bands=5,stride=64;const metal_complex64 pad{827,-932};
    std::vector<metal_complex64> input(bands*stride,pad),output(input.size(),pad),returned(input.size(),pad);
    for(size_t band=0;band<bands;++band)for(size_t i=0;i<n;++i)input[band*stride+i]={std::sin(.13*i+band),std::cos(.21*i-band)};
    setenv("VASP_METAL_FFT_CHUNK_BYTES","1920",1);
    assert(metal_backend_zfft3d_batch_gpu(b,5,4,3,bands,stride,input.data(),output.data(),-1,0)==0);
    assert(metal_backend_last_gpu_success(b));double worst=0;
    for(size_t band=0;band<bands;++band)for(size_t kz=0;kz<3;++kz)for(size_t ky=0;ky<4;++ky)for(size_t kx=0;kx<5;++kx) {
        std::complex<double> reference=0;
        for(size_t z=0;z<3;++z)for(size_t y=0;y<4;++y)for(size_t x=0;x<5;++x) {
            auto v=input[band*stride+x+5*(y+4*z)];double angle=-2*std::acos(-1.)*(double(kx*x)/5+double(ky*y)/4+double(kz*z)/3);
            reference+=std::complex<double>(v.real,v.imag)*std::polar(1.,angle);
        }
        auto v=output[band*stride+kx+5*(ky+4*kz)];worst=std::max(worst,std::abs(std::complex<double>(v.real,v.imag)-reference));
    }
    assert(worst<1e-10);
    assert(metal_backend_zfft3d_batch_gpu(b,5,4,3,bands,stride,output.data(),returned.data(),1,1)==0);
    for(size_t band=0;band<bands;++band)for(size_t i=0;i<stride;++i) {
        const auto v=returned[band*stride+i];
        if(i<n) assert(std::hypot(v.real-input[band*stride+i].real,v.imag-input[band*stride+i].imag)<1e-11);
        else assert(v.real==pad.real && v.imag==pad.imag);
    }
    const auto saved=returned;input[4*stride+20].real=NAN;
    assert(metal_backend_zfft3d_batch_gpu(b,5,4,3,bands,stride,input.data(),returned.data(),-1,0)!=0);
    for(size_t i=0;i<saved.size();++i) assert(returned[i].real==saved[i].real && returned[i].imag==saved[i].imag);
    metal_chunk_stats stats{};assert(metal_backend_get_chunk_stats(b,&stats)==0);
    assert(stats.fft_calls==3 && stats.fft_windows==6 && stats.fft_peak_workspace_bytes==8832);
    unsetenv("VASP_METAL_FFT_CHUNK_BYTES");
    std::printf("bounded5-batch3D FFT, independent FP64 DFT, padding and atomic failure passed; error=%.17g\n",worst);
    metal_backend_destroy(b);
}
