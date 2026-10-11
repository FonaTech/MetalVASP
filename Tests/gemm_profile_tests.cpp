#include "metal_backend.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using Z=std::complex<long double>;
Z load(const std::vector<metal_complex64> &a,size_t ld,size_t i,size_t j,char t) {
    auto v=t=='N' ? a[i+j*ld] : a[j+i*ld];
    Z z(v.real,v.imag);return t=='C' ? std::conj(z):z;
}
int main(int argc,char **) {
    metal_backend *backend=nullptr;
    assert(metal_backend_create(&backend,1)==0);
    if(!metal_backend_gpu_enabled(backend)) {metal_backend_destroy(backend);return std::getenv("METAL_REQUIRE_GPU") ? 1:77;}
    if(argc>1) {
        for(size_t side:{128u,256u,512u}) {
            std::vector<metal_complex64> a(side*side),b(side*side),c(side*side);
            for(size_t i=0;i<a.size();++i) {a[i]={std::sin(.01*i)*.1,std::cos(.02*i)*.1};b[i]={std::cos(.03*i)*.1,std::sin(.04*i)*.1};}
            for(const char *kernel:{"generic","tiled"}) {
                setenv("VASP_METAL_GEMM_KERNEL",kernel,1);
                auto run=[&] {assert(metal_backend_zgemm_blas(backend,'N','N',side,side,side,{1,0},a.data(),side,b.data(),side,{0,0},c.data(),side)==0);assert(metal_backend_last_gpu_success(backend));};
                run();std::vector<double> wall,gpu;
                for(int r=0;r<3;++r) {double g=metal_backend_gpu_seconds(backend);auto start=std::chrono::steady_clock::now();run();wall.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());gpu.push_back(metal_backend_gpu_seconds(backend)-g);}
                std::sort(wall.begin(),wall.end());std::sort(gpu.begin(),gpu.end());
                std::printf("{\"side\":%zu,\"kernel\":\"%s\",\"tiled_available\":%d,\"median_wall_seconds\":%.9f,\"median_gpu_seconds\":%.9f,\"repeats\":3}\n",side,kernel,metal_backend_tiled_gemm_enabled(backend),wall[1],gpu[1]);
            }
        }
    } else {
        for(const char *kernel:{"generic","tiled"}) for(char ta:{'N','T','C'}) for(char tb:{'N','T','C'}) {
            setenv("VASP_METAL_GEMM_KERNEL",kernel,1);
            for(size_t m:{1u,17u,33u}) {
                const size_t n=19,k=23,lda=(ta=='N' ? m:k)+3,ldb=(tb=='N' ? k:n)+5,ldc=m+7;
                std::vector<metal_complex64> a(lda*(ta=='N' ? k:m)),b(ldb*(tb=='N' ? n:k));
                for(size_t i=0;i<a.size();++i) a[i]={std::sin(.17*i),std::cos(.13*i)};
                for(size_t i=0;i<b.size();++i) b[i]={std::cos(.19*i),std::sin(.23*i)};
                for(bool zero_beta:{false,true}) {
                    std::vector<metal_complex64> c(ldc*n,{123,-456});
                    const metal_complex64 alpha{.7,-.3},beta=zero_beta ? metal_complex64{0,0}:metal_complex64{-.2,.4};
                    if(zero_beta) for(size_t j=0;j<n;++j) for(size_t i=0;i<m;++i) c[i+j*ldc]={std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()};
                    assert(metal_backend_zgemm_blas(backend,ta,tb,m,n,k,alpha,a.data(),lda,b.data(),ldb,beta,c.data(),ldc)==0);
                    assert(metal_backend_last_gpu_success(backend));
                    for(size_t j=0;j<n;++j) for(size_t i=0;i<ldc;++i) {
                        if(i>=m) {assert(c[i+j*ldc].real==123 && c[i+j*ldc].imag==-456);continue;}
                        Z sum=0.;for(size_t q=0;q<k;++q) sum+=load(a,lda,i,q,ta)*load(b,ldb,q,j,tb);
                        const Z expected=Z(alpha.real,alpha.imag)*sum+Z(beta.real,beta.imag)*Z(123,-456);
                        assert(std::isfinite(c[i+j*ldc].real) && std::isfinite(c[i+j*ldc].imag));
                        assert(std::abs(Z(c[i+j*ldc].real,c[i+j*ldc].imag)-expected)<1e-10);
                    }
                }
            }
        }
        if(metal_backend_tiled_gemm_enabled(backend)) assert(metal_backend_tiled_gemm_calls(backend)>0);
        std::printf("profile=%s language=%s tiled=%d tiled_calls=%llu\n",metal_backend_variant(backend),metal_backend_language_version(backend),metal_backend_tiled_gemm_enabled(backend),(unsigned long long)metal_backend_tiled_gemm_calls(backend));
    }
    if(metal_backend_metal4_command_requested(backend) && metal_backend_metal4_command_available(backend)) {
        const auto calls=metal_backend_metal4_command_calls(backend);
        assert(calls>0);
        const char *fault=std::getenv("VASP_METAL_TEST_METAL4_FAILURE");
        if(fault && std::strcmp(fault,"after-completion")==0) {
            assert(metal_backend_metal4_command_success(backend)==0);
            assert(metal_backend_metal4_command_fallbacks(backend)==calls);
        } else {
            assert(metal_backend_metal4_command_success(backend)==calls);
            assert(metal_backend_metal4_command_fallbacks(backend)==0);
        }
    }
    std::printf("metal4_requested=%d available=%d calls=%llu success=%llu fallbacks=%llu reason=%s\n",
        metal_backend_metal4_command_requested(backend),metal_backend_metal4_command_available(backend),
        (unsigned long long)metal_backend_metal4_command_calls(backend),(unsigned long long)metal_backend_metal4_command_success(backend),
        (unsigned long long)metal_backend_metal4_command_fallbacks(backend),metal_backend_metal4_command_reason(backend));
    metal_backend_destroy(backend);
}
