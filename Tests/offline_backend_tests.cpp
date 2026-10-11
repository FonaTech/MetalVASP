#include "metal_backend.h"
#include "metal_stats_path.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc,char **argv) {
    if(argc!=3 && argc!=4) {std::fprintf(stderr,"usage: offline-backend-tests /absolute/backend.metallib used|fallback [expected-reason]\n");return 2;}
    const bool want_used=std::strcmp(argv[2],"used")==0;
    if(!want_used && std::strcmp(argv[2],"fallback")!=0) return 2;
    setenv("VASP_METAL_LIBRARY",argv[1],1);
    metal_backend *backend=nullptr;
    if(metal_backend_create(&backend,1)!=0 || !backend) return 1;
    if(!metal_backend_gpu_enabled(backend)) {std::fprintf(stderr,"GPU backend unavailable: %s\n",metal_backend_last_error(backend));metal_backend_destroy(backend);return 77;}
    bool pass=metal_backend_precompiled_requested(backend)==1 &&
        metal_backend_precompiled_used(backend)==int(want_used) &&
        metal_backend_precompiled_fallbacks(backend)==unsigned(want_used?0:1);
    if(want_used) pass=pass && metal_backend_precompiled_verified(backend)==1;
    if(argc==4) pass=pass && std::strcmp(metal_backend_precompiled_reason(backend),argv[3])==0;
    using cd=std::complex<double>;
    const size_t m=33,n=47,k=29,lda=m+3,ldb=k+5,ldc=m+7;
    const cd alpha(.7,-.3),beta(-.2,.4);
    std::vector<metal_complex64> a(lda*k),b(ldb*n),c(ldc*n);std::vector<cd> reference(m*n);
    for(size_t j=0;j<k;++j) for(size_t i=0;i<m;++i) a[i+j*lda]={std::sin(.13*(i+j*m)),std::cos(.17*(i+j*m))};
    for(size_t j=0;j<n;++j) for(size_t i=0;i<k;++i) b[i+j*ldb]={std::cos(.11*(i+j*k)),std::sin(.19*(i+j*k))};
    for(size_t j=0;j<n;++j) for(size_t i=0;i<m;++i) {
        const cd initial(std::sin(.07*(i+j*m)),std::cos(.23*(i+j*m)));c[i+j*ldc]={initial.real(),initial.imag()};
        cd sum{};for(size_t q=0;q<k;++q) sum+=cd(a[i+q*lda].real,a[i+q*lda].imag)*cd(b[q+j*ldb].real,b[q+j*ldb].imag);
        reference[i+j*m]=alpha*sum+beta*initial;
    }
    const auto rc=metal_backend_zgemm_blas(backend,'N','N',m,n,k,{alpha.real(),alpha.imag()},a.data(),lda,b.data(),ldb,{beta.real(),beta.imag()},c.data(),ldc);
    double maximum=0.;bool finite=true;
    for(size_t j=0;j<n;++j) for(size_t i=0;i<m;++i) {
        cd actual(c[i+j*ldc].real,c[i+j*ldc].imag);finite=finite&&std::isfinite(actual.real())&&std::isfinite(actual.imag());
        maximum=std::max(maximum,std::abs(actual-reference[i+j*m]));
    }
    pass=pass && rc==0 && finite && maximum<1e-8 && metal_backend_last_gpu_success(backend) &&
        metal_backend_command_count(backend)>0 && metal_backend_gpu_seconds(backend)>0.;
    const char *api=std::getenv("VASP_METAL_COMMAND_API");
    if(api && std::strcmp(api,"4")==0) pass=pass && metal_backend_metal4_command_available(backend) &&
        metal_backend_metal4_command_success(backend)>0 && metal_backend_metal4_command_fallbacks(backend)==0;
    std::printf("{\"requested\":%d,\"verified\":%d,\"used\":%d,\"fallbacks\":%u,\"reason\":%s,\"language\":%s,\"max_abs_error\":%.9g,\"gpu_commands\":%llu,\"metal4_success\":%llu,\"passed\":%s}\n",
        metal_backend_precompiled_requested(backend),metal_backend_precompiled_verified(backend),metal_backend_precompiled_used(backend),metal_backend_precompiled_fallbacks(backend),
        metal_stats::json_string(metal_backend_precompiled_reason(backend)).c_str(),metal_stats::json_string(metal_backend_language_version(backend)).c_str(),maximum,
        static_cast<unsigned long long>(metal_backend_command_count(backend)),static_cast<unsigned long long>(metal_backend_metal4_command_success(backend)),pass?"true":"false");
    metal_backend_destroy(backend);return pass?0:1;
}
