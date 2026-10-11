#include <algorithm>
#include "metal_backend.h"
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main() {
    metal_backend *b=nullptr;assert(!metal_backend_create(&b,1));
    if(!metal_backend_gpu_enabled(b))return std::getenv("METAL_REQUIRE_GPU") ? 1:77;
    const size_t rows=11,cols=7,npl=4097,npro=73,ldw=npl+3,ldp=npro+2,ldo=rows+4;
    const metal_complex64 pad{827,-932};
    std::vector<metal_complex64> wa(ldw*rows,pad),wb(ldw*cols,pad),pa(ldp*rows,pad),pb(ldp*cols,pad),initial(ldo*cols,pad);
    for(size_t j=0;j<rows;++j) {
        for(size_t k=0;k<npl;++k)wa[k+j*ldw]={std::sin(.13*k+.37*j),std::cos(.21*k-.19*j)};
        for(size_t k=0;k<npro;++k)pa[k+j*ldp]={.1*std::sin(k+j),.1*std::cos(double(k)-j)};
    }
    for(size_t j=0;j<cols;++j) {
        for(size_t k=0;k<npl;++k)wb[k+j*ldw]={std::cos(.23*k+.17*j),std::sin(.11*k-.31*j)};
        for(size_t k=0;k<npro;++k)pb[k+j*ldp]={.2*std::cos(k+j),.2*std::sin(double(k)-j)};
        for(size_t i=0;i<rows;++i)initial[i+j*ldo]={.1*i,.2*j};
    }
    double worst=0;
    for(const char *mode:{"serial","cooperative","auto"}) {
        setenv("VASP_METAL_ORTH_KERNEL",mode,1);auto result=initial;
        assert(!metal_backend_orth_overlap_gpu(b,rows,cols,npl,npro,wa.data(),wb.data(),ldw,pa.data(),pb.data(),ldp,result.data(),ldo));
        assert(metal_backend_last_gpu_success(b));
        for(size_t j=0;j<cols;++j)for(size_t i=0;i<ldo;++i) {
            auto value=result[i+j*ldo];
            if(i>=rows){assert(value.real==pad.real && value.imag==pad.imag);continue;}
            std::complex<long double> reference(initial[i+j*ldo].real,initial[i+j*ldo].imag);
            auto dot=[&](const auto &x,const auto &y,size_t count,size_t ld) {
                for(size_t k=0;k<count;++k) {
                    auto a=x[k+i*ld],z=y[k+j*ld];
                    reference+=std::conj(std::complex<long double>(a.real,a.imag))*std::complex<long double>(z.real,z.imag);
                }
            };
            dot(wa,wb,npl,ldw);dot(pa,pb,npro,ldp);
            const double error=double(std::abs(std::complex<long double>(value.real,value.imag)-reference));
            worst=std::max(worst,error);assert(error<1e-10);
        }
    }
    unsetenv("VASP_METAL_ORTH_KERNEL");
    assert(metal_backend_cooperative_orth_calls(b)==2);
    std::printf("Cooperative/serial/auto PW+PAW FP64 overlap and additive/padding controls passed; error=%.17g\n",worst);
    metal_backend_destroy(b);
}
