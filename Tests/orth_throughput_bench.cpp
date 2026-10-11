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
    std::puts("rows,cols,npl,npro,kernel,repeat,warmup,api_wall_seconds,gpu_command_seconds,host_wait_seconds,conversion_seconds");
    for(auto shape:std::vector<std::vector<size_t>>{{32,8,2048,128},{128,8,8192,256},{128,8,32768,256},{256,16,32768,512}}) {
        auto rows=shape[0],cols=shape[1],npl=shape[2],npro=shape[3];
        std::vector<metal_complex64> wa(rows*npl),wb(cols*npl),pa(rows*npro),pb(cols*npro),out(rows*cols);
        for(size_t i=0;i<wa.size();++i)wa[i]={std::sin(.13*i),std::cos(.19*i)};
        for(size_t i=0;i<wb.size();++i)wb[i]={std::cos(.17*i),std::sin(.11*i)};
        for(size_t i=0;i<pa.size();++i)pa[i]={.1*std::sin(i),.1*std::cos(i)};
        for(size_t i=0;i<pb.size();++i)pb[i]={.2*std::cos(i),.2*std::sin(i)};
        for(int repeat=0;repeat<8;++repeat)for(int order=0;order<2;++order) {
            const char *mode=(order+repeat)%2 ? "cooperative":"serial";
            setenv("VASP_METAL_ORTH_KERNEL",mode,1);
            for(auto &value:out)value={0,0};
            auto gpu=metal_backend_gpu_seconds(b),wait=metal_backend_wait_seconds(b),conversion=metal_backend_conversion_seconds(b);
            auto begin=std::chrono::steady_clock::now();
            assert(!metal_backend_orth_overlap_gpu(b,rows,cols,npl,npro,wa.data(),wb.data(),npl,pa.data(),pb.data(),npro,out.data(),rows));
            assert(metal_backend_last_gpu_success(b));
            double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            for(auto value:out)assert(std::isfinite(value.real) && std::isfinite(value.imag));
            std::printf("%zu,%zu,%zu,%zu,%s,%d,%d,%.12g,%.12g,%.12g,%.12g\n",rows,cols,npl,npro,mode,repeat,repeat==0,
                wall,metal_backend_gpu_seconds(b)-gpu,metal_backend_wait_seconds(b)-wait,metal_backend_conversion_seconds(b)-conversion);
        }
    }
    assert(metal_backend_cooperative_orth_calls(b)==32);
    unsetenv("VASP_METAL_ORTH_KERNEL");metal_backend_destroy(b);
}
