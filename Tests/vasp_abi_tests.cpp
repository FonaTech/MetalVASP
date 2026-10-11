#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <limits>
#include <cstdio>
#include <vector>
#include "metal_backend.h"
struct Z { double real,imag; };
extern "C" void zheevx_(const char*,const char*,const char*,const int*,Z*,const int*,const double*,const double*,const int*,const int*,const double*,int*,double*,Z*,const int*,Z*,const int*,double*,int*,int*,int*);
extern "C" void zhegv_(const int*,const char*,const char*,const int*,Z*,const int*,Z*,const int*,double*,Z*,const int*,double*,int*);
extern "C" int vasp_metal_fft3d_batch(int,int,int,int,int,Z*,int);
extern "C" void vasp_metal_resident_fft_cache_stats(uint64_t*,uint64_t*,uint64_t*);
int main() {
    setenv("VASP_METAL","1",1);
    // This test validates the exported ABI on a deliberately tiny transform;
    // bypass the production arithmetic-intensity router so a CPU fallback is
    // not mistaken for an ABI failure.
    setenv("VASP_METAL_FFT_MIN_POINTS","0",1);
    setenv("VASP_METAL_RESIDENT_FFT","1",1);
    setenv("VASP_METAL_RESIDENT_FFT_MAX_BYTES","268435456",1);
    setenv("VASP_METAL_GEMM_MIN_FLOPS","0",1);
    setenv("VASP_METAL_EIGH_MIN_CUBIC","0",1);
    if(std::getenv("METAL_REQUIRE_GPU")) setenv("VASP_METAL_REQUIRE_GPU","1",1);
    metal_backend *probe=nullptr;
    assert(metal_backend_create(&probe,1)==METAL_BACKEND_OK);
    if(!metal_backend_gpu_enabled(probe)) {
        std::printf("SKIP: no Metal device visible: %s\n",metal_backend_last_error(probe));
        metal_backend_destroy(probe);
        return std::getenv("METAL_REQUIRE_GPU") ? 86 : 77;
    }
    metal_backend_destroy(probe);
    const int n=3,ld=5,lwork=128,il=2,iu=3;int m=-1,info=-999;
    const double vl=0.5,vu=2.5,abstol=0;char job='V',uplo='U';
    std::vector<Z> a(ld*n),z(ld*n),work(lwork);double w[n],rwork[7*n];int iw[5*n],fail[n];
    auto reset=[&](){
        for(auto &x:a) x={std::numeric_limits<double>::quiet_NaN(),0};
        for(int j=0;j<n;++j) for(int i=0;i<=j;++i) a[i+j*ld]={i==j ? double(j+1):0.0,0.0};
    };
    for(char range: {'A','I','V'}) {
        reset();
        zheevx_(&job,&range,&uplo,&n,a.data(),&ld,&vl,&vu,&il,&iu,&abstol,&m,w,z.data(),&ld,work.data(),&lwork,rwork,iw,fail,&info);
        assert(info==0);assert(m==(range=='A' ? 3:2));
        for(int j=0;j<m;++j) {
            double expected=range=='I' ? j+2:j+1;assert(std::abs(w[j]-expected)<1e-10);assert(fail[j]==0);
            for(int i=0;i<n;++i) {
                auto x=z[i+j*ld];double residual=std::hypot((i+1-w[j])*x.real,(i+1-w[j])*x.imag);
                assert(residual<1e-10);
            }
        }
    }
    reset();int query=-1;char range='A';
    zheevx_(&job,&range,&uplo,&n,a.data(),&ld,&vl,&vu,&il,&iu,&abstol,&m,w,z.data(),&ld,work.data(),&query,rwork,iw,fail,&info);
    assert(info==0 && work[0].real>=2*n);
    // Generalized ABI must return the Cholesky factor in B, not its input.
    reset();std::vector<Z> b(ld*n);
    for(int j=0;j<n;++j) b[j+j*ld]={4.0,0.0};int itype=1;
    zhegv_(&itype,&job,&uplo,&n,a.data(),&ld,b.data(),&ld,w,work.data(),&lwork,rwork,&info);
    assert(info==0);
    for(int j=0;j<n;++j) {assert(std::abs(w[j]-(j+1)*0.25)<1e-10);assert(std::abs(b[j+j*ld].real-2.0)<1e-10);}
    // Nontrivial complex metric. Verify the returned factor against input B
    // for both triangles, and the eigenpairs against the original A/B.
    for(char triangle: {'U','L'}) {
        std::vector<std::complex<double>> lower(n*n),original_b(n*n),original_a(n*n);
        for(int i=0;i<n;++i) lower[i+i*n]=1.25+0.3*i;
        lower[1]={0.2,-0.1};lower[2]={-0.15,0.08};lower[2+n]={0.11,0.19};
        for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
            for(int k=0;k<n;++k) original_b[i+j*n]+=lower[i+k*n]*std::conj(lower[j+k*n]);
            original_a[i+j*n]=i==j ? std::complex<double>(i+1,0):std::complex<double>(0.1,0.07*(j-i));
            a[i+j*ld]={std::numeric_limits<double>::quiet_NaN(),0};b[i+j*ld]=a[i+j*ld];
            if(triangle=='U' ? i<=j:i>=j) {
                a[i+j*ld]={original_a[i+j*n].real(),original_a[i+j*n].imag()};
                b[i+j*ld]={original_b[i+j*n].real(),original_b[i+j*n].imag()};
            }
        }
        zhegv_(&itype,&job,&triangle,&n,a.data(),&ld,b.data(),&ld,w,work.data(),&lwork,rwork,&info);
        assert(info==0);
        for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
            std::complex<double> reconstructed=0,ax=0,bx=0;
            for(int k=0;k<n;++k) {
                auto left=triangle=='U' ? (k<=i ? std::conj(std::complex<double>(b[k+i*ld].real,b[k+i*ld].imag)):0.0):(i>=k ? std::complex<double>(b[i+k*ld].real,b[i+k*ld].imag):0.0);
                auto right=triangle=='U' ? (k<=j ? std::complex<double>(b[k+j*ld].real,b[k+j*ld].imag):0.0):(j>=k ? std::conj(std::complex<double>(b[j+k*ld].real,b[j+k*ld].imag)):0.0);
                reconstructed+=left*right;
                auto x=std::complex<double>(a[k+j*ld].real,a[k+j*ld].imag);ax+=original_a[i+k*n]*x;bx+=original_b[i+k*n]*x;
            }
            assert(std::abs(reconstructed-original_b[i+j*n])<1e-11);
            assert(std::abs(ax-w[j]*bx)<1e-11);
        }
    }
    // Non-positive metric must use LAPACK's positive INFO semantics; no fake
    // GPU success is possible when the Cholesky pivot fails.
    reset();b.assign(ld*n,Z{0,0});for(int j=0;j<n;++j) b[j+j*ld]={j==1 ? -1.0:1.0,0};
    zhegv_(&itype,&job,&uplo,&n,a.data(),&ld,b.data(),&ld,w,work.data(),&lwork,rwork,&info);
    assert(info==n+2);
    // Exercise the exported Fortran-facing batch FFT ABI with an LDC larger
    // than the transform size. VASP/FFTW leaves the inverse unnormalized.
    const int nx=4, ny=2, nz=2, total=nx*ny*nz, nbatch=3, ldc=total+5;
    std::vector<Z> batch(static_cast<size_t>(nbatch)*ldc), original(batch);
    for (int batch_id=0; batch_id<nbatch; ++batch_id) {
        for (int i=0; i<total; ++i) {
            batch[batch_id*ldc+i]={std::sin(0.11*(i+batch_id)), std::cos(0.17*(i+2*batch_id))};
            original[batch_id*ldc+i]=batch[batch_id*ldc+i];
        }
        for (int i=total; i<ldc; ++i) batch[batch_id*ldc+i]={321.0,-654.0};
    }
    uint64_t cache_creates=0,cache_reuses=0,cache_bytes=0;
    vasp_metal_resident_fft_cache_stats(&cache_creates,&cache_reuses,&cache_bytes);
    const auto initial_creates=cache_creates,initial_reuses=cache_reuses;
    assert(vasp_metal_fft3d_batch(nx,ny,nz,nbatch,ldc,batch.data(),-1)==1);
    assert(vasp_metal_fft3d_batch(nx,ny,nz,nbatch,ldc,batch.data(),+1)==1);
    double batch_error=0.0;
    for (int batch_id=0; batch_id<nbatch; ++batch_id) {
        for (int i=0; i<total; ++i) {
            batch_error=std::max(batch_error,std::abs(batch[batch_id*ldc+i].real-total*original[batch_id*ldc+i].real));
            batch_error=std::max(batch_error,std::abs(batch[batch_id*ldc+i].imag-total*original[batch_id*ldc+i].imag));
        }
        for (int i=total; i<ldc; ++i) {
            assert(batch[batch_id*ldc+i].real==321.0);
            assert(batch[batch_id*ldc+i].imag==-654.0);
        }
    }
    std::printf("batch FFT ABI max unnormalized round-trip error=%.3e\n",batch_error);
    assert(batch_error<3e-8);
    vasp_metal_resident_fft_cache_stats(&cache_creates,&cache_reuses,&cache_bytes);
    assert(cache_creates==initial_creates+1 && cache_reuses==initial_reuses+1);
    assert(cache_bytes==static_cast<uint64_t>(total*nbatch*32+4));
    // Reducing the memory budget evicts the resident allocation and keeps the
    // existing GPU FFT path as a compatible fallback, with fresh host input.
    setenv("VASP_METAL_RESIDENT_FFT_MAX_BYTES","0",1);
    batch=original;
    assert(vasp_metal_fft3d_batch(nx,ny,nz,nbatch,ldc,batch.data(),-1)==1);
    assert(vasp_metal_fft3d_batch(nx,ny,nz,nbatch,ldc,batch.data(),1)==1);
    for(int b=0;b<nbatch;++b) for(int i=0;i<total;++i) {
        assert(std::abs(batch[b*ldc+i].real-total*original[b*ldc+i].real)<3e-8);
        assert(std::abs(batch[b*ldc+i].imag-total*original[b*ldc+i].imag)<3e-8);
    }
    vasp_metal_resident_fft_cache_stats(&cache_creates,&cache_reuses,&cache_bytes);
    assert(cache_bytes==0 && cache_creates==initial_creates+1 && cache_reuses==initial_reuses+1);
    std::printf("resident FFT ABI cache creates=%llu reuses=%llu budget fallback passed\n",
        static_cast<unsigned long long>(cache_creates),static_cast<unsigned long long>(cache_reuses));
    std::puts("LAPACK ABI range, query, padded triangle and Cholesky postconditions passed");
}
