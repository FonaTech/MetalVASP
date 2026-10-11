#include "metal_backend.h"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <chrono>
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


static void check_case(metal_backend *backend,size_t bands,bool paw,bool real_metric) {
    const size_t npl=2*bands+3,npro=paw ? 8:0,nions=paw ? 4:0,lm=paw ? 4:0;
    const size_t ldw=npl+3,ldp=npro+2;
    const C padding(17,19);
    int counts[]={3,2,0,3},bases[]={0,3,5,5,8};
    std::vector<metal_complex64> wave(ldw*bands,mz(padding)),proj(ldp*bands,mz(padding));
    for(size_t j=0;j<bands;++j) {
        for(size_t i=0;i<npl;++i) {
            C v(0.0007*std::sin(.017*(i+1)*(j+1)),.0005*std::cos(.031*(i+2)*(j+1)));
            if(i==j) v+=1+.0001*j;
            wave[i+j*ldw]=mz(v);
        }
        for(size_t i=0;i<npro;++i) proj[i+j*ldp]=mz(C(.002*std::sin(.07*(i+1)*(j+2)),.003*std::cos(.09*(i+2)*(j+1))));
    }
    std::vector<C> metric(nions*lm*lm);
    std::vector<double> real(metric.size());
    std::vector<metal_complex64> complex_metric(metric.size());
    for(size_t ion=0;ion<nions;++ion) for(size_t col=0;col<lm;++col) for(size_t row=0;row<lm;++row) {
        size_t at=row+lm*col+ion*lm*lm;
        metric[at]=row==col ? C(1.3+.1*ion+.05*row):C(.02,real_metric ? 0:(row<col ? .01:-.01));
        real[at]=metric[at].real();complex_metric[at]=mz(metric[at]);
    }
    auto wave0=wave,proj0=proj;
    auto gram_of=[&](const auto &w,const auto &p) {
        std::vector<C> g(bands*bands);
        for(size_t col=0;col<bands;++col) for(size_t row=0;row<bands;++row) {
            C sum=0;
            for(size_t i=0;i<npl;++i) sum+=std::conj(z(w[i+row*ldw]))*z(w[i+col*ldw]);
            for(size_t ion=0;ion<nions;++ion) for(int i=0;i<counts[ion];++i) for(int j=0;j<counts[ion];++j)
                sum+=std::conj(z(p[bases[ion]+i+row*ldp]))*metric[i+lm*j+ion*lm*lm]*z(p[bases[ion]+j+col*ldp]);
            g[row+col*bands]=sum;
        }
        return g;
    };
    auto original_gram=gram_of(wave0,proj0);
    std::vector<C> inv;assert(chol_inverse(original_gram,bands,inv));
    std::vector<C> refw(npl*bands),refp(npro*bands);
    for(size_t col=0;col<bands;++col) {
        for(size_t i=0;i<npl;++i) for(size_t k=0;k<bands;++k) refw[i+col*npl]+=z(wave0[i+k*ldw])*std::conj(inv[col+k*bands]);
        for(size_t i=0;i<npro;++i) for(size_t k=0;k<bands;++k) refp[i+col*npro]+=z(proj0[i+k*ldp])*std::conj(inv[col+k*bands]);
    }
    auto before=metal_backend_command_count(backend);
    auto start=std::chrono::steady_clock::now();
    assert(metal_backend_full_orth_gpu(backend,npl,npro,bands,nions,lm,real_metric,
        paw ? (real_metric ? static_cast<const void *>(real.data()):static_cast<const void *>(complex_metric.data())):nullptr,
        paw ? counts:nullptr,paw ? bases:nullptr,wave.data(),ldw,paw ? proj.data():nullptr,ldp)==METAL_BACKEND_OK);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    assert(metal_backend_last_gpu_success(backend) && metal_backend_command_count(backend)==before+1);
    double error=0;
    for(size_t col=0;col<bands;++col) {
        for(size_t i=0;i<npl;++i) error=std::max(error,std::abs(z(wave[i+col*ldw])-refw[i+col*npl]));
        for(size_t i=0;i<npro;++i) error=std::max(error,std::abs(z(proj[i+col*ldp])-refp[i+col*npro]));
        for(size_t i=npl;i<ldw;++i) assert(z(wave[i+col*ldw])==padding);
        for(size_t i=npro;i<ldp;++i) assert(z(proj[i+col*ldp])==padding);
    }
    auto identity=gram_of(wave,proj);double defect=0;
    for(size_t col=0;col<bands;++col) for(size_t row=0;row<bands;++row)
        defect=std::max(defect,std::abs(identity[row+col*bands]-(row==col ? C(1):C(0))));
    std::printf("large full orth bands=%zu PAW=%d real_metric=%d FP64_error=%.12g metric_identity_error=%.12g GPU_API_seconds=%.6f\n",
                bands,paw,real_metric,error,defect,seconds);
    assert(error<3e-10 && defect<3e-10);
    // An exactly zero column produces a singular Gram matrix. Failure must
    // preserve the complete caller buffers, including the padding.
    for(size_t i=0;i<npl;++i) wave[i+ldw]={0,0};
    for(size_t i=0;i<npro;++i) proj[i+ldp]={0,0};
    wave0=wave;proj0=proj;
    assert(metal_backend_full_orth_gpu(backend,npl,npro,bands,nions,lm,real_metric,
        paw ? (real_metric ? static_cast<const void *>(real.data()):static_cast<const void *>(complex_metric.data())):nullptr,
        paw ? counts:nullptr,paw ? bases:nullptr,wave.data(),ldw,paw ? proj.data():nullptr,ldp)!=METAL_BACKEND_OK);
    assert(std::memcmp(wave.data(),wave0.data(),wave.size()*sizeof(wave[0]))==0);
    assert(std::memcmp(proj.data(),proj0.data(),proj.size()*sizeof(proj[0]))==0);
}
int main() {
    metal_backend *backend=nullptr;assert(!metal_backend_create(&backend,1));
    if(!metal_backend_gpu_enabled(backend)) {metal_backend_destroy(backend);return std::getenv("METAL_REQUIRE_GPU") ? 1:77;}
    check_case(backend,129,true,false);
    check_case(backend,160,true,true);
    check_case(backend,320,true,false);
    check_case(backend,512,false,false);
    // Reject impossible uint-offset shapes before dereferencing metadata.
    metal_complex64 dummy{1,2},saved=dummy;double q=1;int one=1,bases[]={0,1};
    assert(metal_backend_full_orth_gpu(backend,1,1,65536,1,1,1,&q,&one,bases,&dummy,1,&dummy,1)!=METAL_BACKEND_OK);
    assert(dummy.real==saved.real && dummy.imag==saved.imag);
    metal_backend_destroy(backend);
    std::puts("large full orth independent FP64, PAW identity, padding, singular rollback and shape rejection passed");
}
