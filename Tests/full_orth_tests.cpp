#include "metal_backend.h"
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using C = std::complex<double>;
static C z(metal_complex64 x) { return {x.real, x.imag}; }
static metal_complex64 mz(C x) { return {x.real(), x.imag()}; }

// Reference lower Cholesky and inverse for the small test metric.  This is
// intentionally independent of the Metal implementation so the test catches
// triangle/order mistakes in the GPU path.
static bool chol_inverse(const std::vector<C>& a, size_t n, std::vector<C>& inv) {
    std::vector<C> l(n*n, C(0));
    for (size_t i=0; i<n; ++i) {
        for (size_t j=0; j<=i; ++j) {
            C s=a[i+n*j];
            for (size_t k=0; k<j; ++k) s-=l[i+n*k]*std::conj(l[j+n*k]);
            if (i==j) {
                if (std::abs(s.imag())>1e-10 || s.real()<=0) return false;
                l[i+n*j]=std::sqrt(s.real());
            } else l[i+n*j]=s/l[j+n*j];
        }
    }
    inv.assign(n*n,C(0));
    for (size_t col=0; col<n; ++col) {
        for (size_t i=0; i<n; ++i) {
            C s=(i==col?C(1):C(0));
            for (size_t k=0; k<i; ++k) s-=l[i+n*k]*inv[k+n*col];
            inv[i+n*col]=s/l[i+n*i];
        }
    }
    return true;
}

int main() {
    setenv("VASP_METAL", "1", 1);
    metal_backend *b=nullptr; assert(metal_backend_create(&b,1)==0);
    if (!metal_backend_gpu_enabled(b)) {
        metal_backend_destroy(b);
        return std::getenv("METAL_REQUIRE_GPU") ? 1 : 77;
    }

    const size_t npl=23, bands=4, nions=1, lmdim=2, npro=2;
    int counts[] = {2}; int bases[] = {0,2};
    std::vector<C> metric={C(2.4,0),C(0.25,-0.12),C(0.25,0.12),C(1.8,0)};
    std::vector<metal_complex64> wave(npl*bands), proj(npro*bands);
    for (size_t j=0;j<bands;++j) {
        for (size_t i=0;i<npl;++i) wave[i+j*npl]=mz(C(0.07*std::sin(0.31*(i+1)*(j+1)), -0.03*std::cos(0.19*(i+2)*(j+1))));
        proj[0+j*npro]=mz(C(0.10*(j+1),-0.04*j));
        proj[1+j*npro]=mz(C(-0.06*j,0.08*(j+1)));
    }
    auto wave0=wave, proj0=proj;
    std::vector<C> gram(bands*bands,C(0));
    for (size_t i=0;i<bands;++i) for (size_t j=0;j<bands;++j) {
        C s=0;
        for(size_t g=0;g<npl;++g) s+=std::conj(z(wave0[g+i*npl]))*z(wave0[g+j*npl]);
        C pi[2]={z(proj0[0+i*npro]),z(proj0[1+i*npro])};
        C pj[2]={z(proj0[0+j*npro]),z(proj0[1+j*npro])};
        for(size_t p=0;p<2;++p) for(size_t q=0;q<2;++q) s+=std::conj(pi[p])*metric[p+2*q]*pj[q];
        gram[i+bands*j]=s;
    }
    std::vector<C> linv;
    if (!chol_inverse(gram,bands,linv)) {
        for (size_t i=0;i<bands;++i) std::fprintf(stderr,"gram[%zu,%zu]=%.17g%+.17gi\n",i,i,gram[i+bands*i].real(),gram[i+bands*i].imag());
        return 2;
    }
    std::vector<C> refw(npl*bands), refp(npro*bands);
    for(size_t j=0;j<bands;++j) for(size_t i=0;i<npl;++i) {
        C s=0; for(size_t k=0;k<bands;++k) s+=z(wave0[i+k*npl])*std::conj(linv[j+bands*k]);
        refw[i+j*npl]=s;
    }
    for(size_t j=0;j<bands;++j) for(size_t i=0;i<npro;++i) {
        C s=0; for(size_t k=0;k<bands;++k) s+=z(proj0[i+k*npro])*std::conj(linv[j+bands*k]);
        refp[i+j*npro]=s;
    }
    std::vector<metal_complex64> metric_abi(metric.size()); for(size_t i=0;i<metric.size();++i) metric_abi[i]=mz(metric[i]);
    assert(metal_backend_full_orth_gpu(b,npl,npro,bands,nions,lmdim,0,metric_abi.data(),counts,bases,
        wave.data(),npl,proj.data(),npro)==METAL_BACKEND_OK);
    double maxerr=0;
    for(size_t i=0;i<wave.size();++i) maxerr=std::max(maxerr,std::abs(z(wave[i])-refw[i]));
    for(size_t i=0;i<proj.size();++i) maxerr=std::max(maxerr,std::abs(z(proj[i])-refp[i]));
    std::printf("full orthogonalization reference max error %.3e\n",maxerr);
    assert(maxerr<3e-5);
    double maxgram=0;
    for(size_t i=0;i<bands;++i) for(size_t j=0;j<bands;++j) {
        C s=0; for(size_t g=0;g<npl;++g) s+=std::conj(z(wave[g+i*npl]))*z(wave[g+j*npl]);
        C pi[2]={z(proj[0+i*npro]),z(proj[1+i*npro])}; C pj[2]={z(proj[0+j*npro]),z(proj[1+j*npro])};
        for(size_t p=0;p<2;++p) for(size_t q=0;q<2;++q) s+=std::conj(pi[p])*metric[p+2*q]*pj[q];
        maxgram=std::max(maxgram,std::abs(s-(i==j?C(1):C(0))));
    }
    assert(maxgram<3e-5);

    // A failed Cholesky must leave caller arrays untouched.
    auto savedw=wave, savedp=proj; metric[0]=C(-1,0); metric_abi[0]=mz(metric[0]);
    assert(metal_backend_full_orth_gpu(b,npl,npro,bands,nions,lmdim,0,metric_abi.data(),counts,bases,
        wave.data(),npl,proj.data(),npro)!=METAL_BACKEND_OK);
    for (size_t i=0;i<wave.size();++i) assert(wave[i].real==savedw[i].real && wave[i].imag==savedw[i].imag);
    for (size_t i=0;i<proj.size();++i) assert(proj[i].real==savedp[i].real && proj[i].imag==savedp[i].imag);
    metal_backend_destroy(b);
}
