#include "metal_backend.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main() {
    metal_backend *b=nullptr;assert(!metal_backend_create(&b,1));
    if(!metal_backend_gpu_enabled(b))return 1;
    std::puts("nx,ny,nz,batches,width,repeat,warmup,api_wall_seconds,gpu_command_seconds,host_wait_seconds,conversion_seconds");
    for(auto shape:std::vector<std::vector<size_t>>{{24,24,24,8},{64,64,64,8},{96,80,64,4}}) {
        auto nx=shape[0],ny=shape[1],nz=shape[2],batches=shape[3],total=nx*ny*nz;
        std::vector<metal_complex64> input(total*batches),output(input.size());
        for(size_t i=0;i<input.size();++i)input[i]={std::sin(.13*i),std::cos(.19*i)};
        for(int repeat=0;repeat<8;++repeat)for(int order=0;order<2;++order) {
            const char *width=(order+repeat)%2 ? "wide":"auto";
            setenv("VASP_METAL_FFT_INDEX_WIDTH",width,1);
            auto gpu=metal_backend_gpu_seconds(b),wait=metal_backend_wait_seconds(b),conversion=metal_backend_conversion_seconds(b);
            auto begin=std::chrono::steady_clock::now();
            assert(!metal_backend_zfft3d_batch_gpu(b,nx,ny,nz,batches,total,input.data(),output.data(),-1,0));
            assert(metal_backend_last_gpu_success(b));
            double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            for(auto value:output)assert(std::isfinite(value.real) && std::isfinite(value.imag));
            std::printf("%zu,%zu,%zu,%zu,%s,%d,%d,%.12g,%.12g,%.12g,%.12g\n",nx,ny,nz,batches,width,repeat,repeat==0,
                wall,metal_backend_gpu_seconds(b)-gpu,metal_backend_wait_seconds(b)-wait,metal_backend_conversion_seconds(b)-conversion);
        }
    }
    assert(metal_backend_fft_compact_axes(b) && metal_backend_fft_wide_axes(b));
    unsetenv("VASP_METAL_FFT_INDEX_WIDTH");metal_backend_destroy(b);
}
