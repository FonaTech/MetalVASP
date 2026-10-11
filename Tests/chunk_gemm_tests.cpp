#include "metal_backend.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>
using C=std::complex<long double>;
static C load(const std::vector<metal_complex64> &a,size_t ld,size_t i,size_t j,char trans) {
    auto x=trans=='N' ? a[i+j*ld]:a[j+i*ld];C z(x.real,x.imag);return trans=='C' ? std::conj(z):z;
}
int main() {
    metal_backend *backend=nullptr;assert(metal_backend_create(&backend,1)==0);
    if(!metal_backend_gpu_enabled(backend)){metal_backend_destroy(backend);return std::getenv("METAL_REQUIRE_GPU") ? 1:77;}
    setenv("VASP_METAL_GEMM_CHUNK_BYTES","1024",1);
    const size_t m=19,n=17,k=31;double worst=0;
    for(auto kernel:{"generic","tiled"})for(char ta:{'N','T','C'})for(char tb:{'N','T','C'})for(bool zero:{true,false}) {
        setenv("VASP_METAL_GEMM_KERNEL",kernel,1);
        const size_t lda=(ta=='N' ? m:k)+3,ldb=(tb=='N' ? k:n)+2,ldc=m+5;
        const metal_complex64 pad{827,-932},alpha{.7,-.3},beta=zero ? metal_complex64{0,0}:metal_complex64{-.2,.4};
        std::vector<metal_complex64> a(lda*(ta=='N' ? k:m),pad),b(ldb*(tb=='N' ? n:k),pad),c(ldc*n,pad);
        for(size_t j=0;j<(ta=='N' ? k:m);++j)for(size_t i=0;i<(ta=='N' ? m:k);++i)a[i+j*lda]={std::sin(.13*i+.37*j),std::cos(.21*i-.19*j)};
        for(size_t j=0;j<(tb=='N' ? n:k);++j)for(size_t i=0;i<(tb=='N' ? k:n);++i)b[i+j*ldb]={std::cos(.23*i+.17*j),std::sin(.11*i-.31*j)};
        for(size_t j=0;j<n;++j)for(size_t i=0;i<m;++i)c[i+j*ldc]=zero ? metal_complex64{NAN,NAN}:metal_complex64{std::sin(i+j),std::cos(double(i)-double(j))};
        auto original=c;std::vector<C> reference(m*n);
        for(size_t j=0;j<n;++j)for(size_t i=0;i<m;++i) {
            C sum=0;for(size_t q=0;q<k;++q)sum+=load(a,lda,i,q,ta)*load(b,ldb,q,j,tb);
            reference[i+j*m]=C(alpha.real,alpha.imag)*sum+(zero ? C(0):C(beta.real,beta.imag)*load(original,ldc,i,j,'N'));
        }
        assert(metal_backend_zgemm_blas_gpu(backend,ta,tb,m,n,k,alpha,a.data(),lda,b.data(),ldb,beta,c.data(),ldc)==0);
        assert(metal_backend_last_gpu_success(backend));
        for(size_t j=0;j<n;++j)for(size_t i=0;i<ldc;++i) {
            auto v=c[i+j*ldc];if(i>=m){assert(v.real==pad.real && v.imag==pad.imag);continue;}
            double error=double(std::abs(C(v.real,v.imag)-reference[i+j*m]));worst=std::max(worst,error);assert(error<2e-10);
        }
        // A late bad input must preserve all caller results even after earlier
        // output panels have completed. The GPU-only API performs no CPU work.
        original=c;a[(ta=='N' ? m-1:k-1)+(ta=='N' ? k-1:m-1)*lda].real=NAN;
        assert(metal_backend_zgemm_blas_gpu(backend,ta,tb,m,n,k,alpha,a.data(),lda,b.data(),ldb,beta,c.data(),ldc)!=0);
        for(size_t i=0;i<c.size();++i)assert(c[i].real==original[i].real && c[i].imag==original[i].imag);
    }
    metal_chunk_stats stats{};assert(metal_backend_get_chunk_stats(backend,&stats)==0);
    assert(stats.gemm_calls==72 && stats.gemm_windows>stats.gemm_calls && stats.gemm_peak_workspace_bytes<=19*17*16+3*1024);
    unsetenv("VASP_METAL_GEMM_CHUNK_BYTES");unsetenv("VASP_METAL_GEMM_KERNEL");
    std::printf("36 bounded GEMM controls: N/T/C, alpha/beta, beta-zero NaN, padding, late atomic failure; FP64 error=%.17g windows=%llu\n",worst,(unsigned long long)stats.gemm_windows);
    metal_backend_destroy(backend);
}
