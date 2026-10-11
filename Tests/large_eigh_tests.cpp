#include "metal_backend.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

extern "C" void zheev_(const char *,const char *,const int *,metal_complex64 *,const int *,
                       double *,metal_complex64 *,const int *,double *,int *);
extern "C" void zhegv_(const int *,const char *,const char *,const int *,metal_complex64 *,const int *,
                       metal_complex64 *,const int *,double *,metal_complex64 *,const int *,double *,int *);
using C=std::complex<double>;
static C z(metal_complex64 x) {return {x.real,x.imag};}
static metal_complex64 mz(C x) {return {x.real(),x.imag()};}
static void solve(metal_backend *backend,int dim,bool generalized,char triangle) {
    const size_t n=dim,ld=n+3;std::mt19937 gen(7919+dim);std::uniform_real_distribution<double> random(-1,1);
    const auto sentinel=mz(C(812,-936));
    std::vector<metal_complex64> a(ld*n,sentinel),b(ld*n,sentinel);
    for(size_t j=0;j<n;++j) for(size_t i=0;i<=j;++i) {
        const C av=i==j ? C(1.+.025*i):C(.025*random(gen),.025*random(gen))/std::sqrt(double(n));
        const C bv=i==j ? C(1.+.0005*i):C(.01*random(gen),.01*random(gen))/std::sqrt(double(n));
        a[i+j*ld]=mz(av);a[j+i*ld]=mz(std::conj(av));b[i+j*ld]=mz(bv);b[j+i*ld]=mz(std::conj(bv));
    }
    const auto original_a=a,original_b=b;
    std::vector<double> reference(n),values(n,-999);std::vector<metal_complex64> work(4*n);
    std::vector<double> rwork(3*n);int info=0,lwork=int(work.size()),leading=int(ld),itype=1;char job='V';
    const auto cpu_start=std::chrono::steady_clock::now();
    if(generalized) zhegv_(&itype,&job,&triangle,&dim,a.data(),&leading,b.data(),&leading,
                           reference.data(),work.data(),&lwork,rwork.data(),&info);
    else zheev_(&job,&triangle,&dim,a.data(),&leading,reference.data(),work.data(),&lwork,rwork.data(),&info);
    const double cpu_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-cpu_start).count();
    assert(info==0);
    std::vector<metal_complex64> vectors(ld*n,sentinel),factor(ld*n,sentinel);
    // Poison the unstored triangle. The generalized API must use uplo only.
    a=original_a;b=original_b;
    if(generalized) for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i)
        if(triangle=='U' ? i>j:i<j) a[i+j*ld]=b[i+j*ld]=mz(C(NAN,NAN));
    const auto start=std::chrono::steady_clock::now();
    int rc=generalized ? metal_backend_generalized_eigh_factor_gpu(backend,itype,triangle,job,n,
        a.data(),ld,b.data(),ld,values.data(),vectors.data(),ld,factor.data(),ld):
        metal_backend_hermitian_eigh_gpu(backend,n,a.data(),ld,values.data(),vectors.data(),ld);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if(rc!=0) {std::fprintf(stderr,"EIGH n=%zu generalized=%d rc=%d\n",n,generalized,rc);std::abort();}
    assert(metal_backend_last_gpu_success(backend));
    double eigen_error=0,residual=0,orth_error=0,factor_error=0;
    double scale=std::max(1.,std::abs(reference.back()));
    std::vector<C> bvectors(n*n);
    for(size_t j=0;j<n;++j) for(size_t i=0;i<n;++i) {
        C value=z(vectors[i+j*ld]);
        if(generalized) {value=0;for(size_t k=0;k<n;++k) value+=z(original_b[i+k*ld])*z(vectors[k+j*ld]);}
        bvectors[i+j*n]=value;
    }
    for(size_t j=0;j<n;++j) {
        eigen_error=std::max(eigen_error,std::abs(values[j]-reference[j])/scale);
        for(size_t i=0;i<n;++i) {
            C ax=0;for(size_t k=0;k<n;++k) ax+=z(original_a[i+k*ld])*z(vectors[k+j*ld]);
            residual=std::max(residual,std::abs(ax-values[j]*bvectors[i+j*n])/scale);
            C overlap=0;
            for(size_t k=0;k<n;++k) {
                overlap+=std::conj(z(vectors[k+i*ld]))*bvectors[k+j*n];
            }
            orth_error=std::max(orth_error,std::abs(overlap-(i==j ? C(1):C(0))));
            if(generalized) {
                C bb=0;
                for(size_t k=0;k<=std::min(i,j);++k) {
                    const C li=triangle=='L' ? z(factor[i+k*ld]):std::conj(z(factor[k+i*ld]));
                    const C lj=triangle=='L' ? z(factor[j+k*ld]):std::conj(z(factor[k+j*ld]));
                    bb+=li*std::conj(lj);
                }
                factor_error=std::max(factor_error,std::abs(bb-z(original_b[i+j*ld])));
            }
        }
        for(size_t i=n;i<ld;++i) assert(z(vectors[i+j*ld])==z(sentinel) && z(factor[i+j*ld])==z(sentinel));
    }
    std::printf("n=%zu generalized=%d uplo=%c eigen_relative=%.16g residual_relative=%.16g orth=%.16g factor=%.16g gpu_api_seconds=%.9g cpu_lapack_seconds=%.9g\n",
        n,generalized,triangle,eigen_error,residual,orth_error,factor_error,seconds,cpu_seconds);
    assert(eigen_error<3e-10 && residual<3e-10 && orth_error<3e-10 && factor_error<3e-10);
    if(generalized) {
        auto saved_vectors=vectors,saved_factor=factor;auto saved_values=values;
        b=original_b;b[0]=mz(C(-1));
        assert(metal_backend_generalized_eigh_factor_gpu(backend,1,triangle,'V',n,a.data(),ld,b.data(),ld,
            values.data(),vectors.data(),ld,factor.data(),ld)!=0);
        assert(values==saved_values);
        for(size_t i=0;i<vectors.size();++i) assert(z(vectors[i])==z(saved_vectors[i]) && z(factor[i])==z(saved_factor[i]));
    }
}
int main() {
    metal_backend *b=nullptr;assert(metal_backend_create(&b,1)==0);
    if(!metal_backend_gpu_enabled(b)) {metal_backend_destroy(b);return std::getenv("METAL_REQUIRE_GPU") ? 1:77;}
    solve(b,129,false,'U');solve(b,160,true,'L');solve(b,257,false,'L');solve(b,320,true,'U');
    assert(metal_backend_eigh_resources(b,SIZE_MAX,SIZE_MAX,0,SIZE_MAX,0)!=0);
    metal_backend_destroy(b);
}
