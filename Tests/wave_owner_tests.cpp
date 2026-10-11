#include "metal_wave_owner.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
using C=std::complex<double>;
static C z(metal_complex64 v){return {v.real,v.imag};}
static metal_complex64 pair(C v){return {v.real(),v.imag()};}
int main() {
 metal_backend *backend=nullptr;assert(!metal_backend_create(&backend,1));
 metal_wave_owner *invalid=reinterpret_cast<metal_wave_owner *>(uintptr_t(1));
 assert(metal_wave_owner_create(backend,{0,1,1,1,1,1,1},&invalid)==METAL_BACKEND_INVALID_ARGUMENT && !invalid);
 assert(metal_wave_owner_create(backend,{SIZE_MAX,1,1,1,1,1,1},&invalid)==METAL_BACKEND_INVALID_ARGUMENT && !invalid);
 if(!metal_backend_gpu_enabled(backend)) {metal_backend_destroy(backend);return std::getenv("METAL_REQUIRE_GPU") ? 1:77;}
 constexpr size_t npl=257,bands=3,nions=2,npro=3,lm=2,ntypes=2,qchannels=2,qld=npl+3,pld=npl+5,ldpre=npl+7,ldmetric=npro+2;
 metal_wave_owner_shape shape{npl,bands,nions,npro,lm,ntypes,qchannels};metal_wave_owner *owner=nullptr;
 assert(!metal_wave_owner_create(backend,shape,&owner));assert(metal_wave_owner_matches(owner,shape));
 int types[]={1,2},lmmax[]={2,1},bases[]={0,2,3};
 std::vector<metal_complex64> original[bands],wave[bands],project[bands];
 const metal_complex64 *input_wave[bands];metal_complex64 *output_wave[bands],*output_project[bands];
 std::vector<double> pre(ldpre*bands),radial(qld*qchannels*ntypes),qreal(lm*lm*nions),scales(bands,-999);
 std::vector<metal_complex64> cq(qchannels*ntypes),phase(pld*nions),qcomplex(qreal.size()),metric(ldmetric*bands,{123,-456});
 for(size_t b=0;b<bands;++b) {
   original[b].resize(npl);wave[b].resize(npl+2);project[b].resize(npro+2,{123,-456});
   for(size_t g=0;g<npl;++g) original[b][g]={.1*std::sin(.017*(g+3*b)),.2*std::cos(.013*(g+7*b))};
   input_wave[b]=original[b].data();output_wave[b]=wave[b].data();output_project[b]=project[b].data();
 }
 for(size_t i=0;i<pre.size();++i) pre[i]=.5+.003*std::sin(.2*i);
 for(size_t i=0;i<radial.size();++i) radial[i]=.03*std::cos(.11*i);
 for(size_t i=0;i<cq.size();++i) cq[i]=pair(std::polar(1.,.13*i));
 for(size_t i=0;i<phase.size();++i) phase[i]=pair(std::polar(1.,.007*i));
 for(size_t ion=0;ion<nions;++ion) {
   qreal[ion*lm*lm]=.2;qreal[1+lm+ion*lm*lm]=.3;
   qreal[1+ion*lm*lm]=qreal[lm+ion*lm*lm]=.01;
   for(size_t i=0;i<lm*lm;++i) qcomplex[ion*lm*lm+i]={qreal[ion*lm*lm+i],0};
   qcomplex[1+ion*lm*lm].imag=.012;qcomplex[lm+ion*lm*lm].imag=-.012;
 }
 metal_wave_prepare_input input{input_wave,pre.data(),ldpre,qld,pld,types,lmmax,bases,radial.data(),cq.data(),phase.data(),qreal.data(),1,1};
 auto verify=[&](bool real,bool overlap) {
   double error=0;
   for(size_t b=0;b<bands;++b) {
     std::vector<C> w(npl),p(npro),m(npro);
     for(size_t g=0;g<npl;++g) w[g]=z(original[b][g])*pre[g+b*ldpre];
     for(size_t ion=0;ion<nions;++ion) for(int l=0;l<lmmax[types[ion]-1];++l) {
       C sum=0;for(size_t g=0;g<npl;++g) sum+=z(phase[g+ion*pld])*w[g]*radial[g+qld*(l+qchannels*(types[ion]-1))];
       p[bases[ion]+l]=sum*z(cq[l+qchannels*(types[ion]-1)]);
     }
     if(overlap) for(size_t ion=0;ion<nions;++ion) for(int l=0;l<lmmax[types[ion]-1];++l)
       for(int k=0;k<lmmax[types[ion]-1];++k) {
         size_t at=k+l*lm+ion*lm*lm;C q=real ? C(qreal[at],0):z(qcomplex[at]);m[bases[ion]+l]+=q*p[bases[ion]+k];
       }
     double norm=0;for(auto value:w) norm+=std::norm(value);
     if(overlap) for(size_t j=0;j<npro;++j) norm+=std::real(std::conj(p[j])*m[j]);
     const double scale=1./std::sqrt(norm);assert(std::abs(scales[b]-scale)<2e-12);
     for(size_t g=0;g<npl;++g) error=std::max(error,std::abs(z(wave[b][g])-w[g]*scale));
     for(size_t j=0;j<npro;++j) {
       error=std::max(error,std::abs(z(project[b][j])-p[j]*scale));
       if(overlap) error=std::max(error,std::abs(z(metric[j+b*ldmetric])-m[j]*scale));
     }
     for(size_t j=npro;j<ldmetric;++j) assert(metric[j+b*ldmetric].real==123 && metric[j+b*ldmetric].imag==-456);
     for(size_t j=npro;j<project[b].size();++j) assert(project[b][j].real==123 && project[b][j].imag==-456);
   }
   std::printf("physical PAW owner real_metric=%d overlap=%d FP64 reference error=%.3e\n",real,overlap,error);assert(error<1e-10);
 };
 uint64_t generation=1,epoch=1,commands=metal_backend_command_count(backend);
 assert(!metal_wave_owner_prepare(owner,generation,epoch,&input));assert(metal_backend_command_count(backend)==commands+1);
 metal_wave_owner_stats stats{};metal_wave_owner_get_stats(owner,&stats);
 assert(stats.host_export_bytes==0 && stats.steps==1 && stats.static_uploads==1);
 assert(!metal_wave_owner_download(owner,generation,output_wave,output_project,metric.data(),ldmetric,scales.data()));verify(true,true);
 const uint64_t dynamic_bytes=npl*bands*24+lm*lm*nions*16;
 const uint64_t static_bytes=npl*qchannels*ntypes*8+qchannels*ntypes*16+npl*nions*16+npro*20;
 metal_wave_owner_get_stats(owner,&stats);assert(stats.host_import_bytes==dynamic_bytes+static_bytes);
 assert(stats.host_export_bytes==npl*bands*16+2*npro*bands*16+bands*8);
 // Same address, new residual and metric: only explicit PAW static epoch caches.
 original[1][7].real+=.031;pre[3]+=.2;qcomplex[0].real+=.1;
 input.real_metric=0;input.metric=qcomplex.data();++generation;
 assert(!metal_wave_owner_prepare(owner,generation,epoch,&input));
 assert(!metal_wave_owner_download(owner,generation,output_wave,output_project,metric.data(),ldmetric,scales.data()));verify(false,true);
 metal_wave_owner_get_stats(owner,&stats);assert(stats.static_reuses==1 && stats.static_uploads==1);
 ++epoch;radial[0]+=.014;phase[7].real-=.01;++generation;
 assert(!metal_wave_owner_prepare(owner,generation,epoch,&input));
 assert(!metal_wave_owner_download(owner,generation,output_wave,output_project,metric.data(),ldmetric,scales.data()));verify(false,true);
 metal_wave_owner_get_stats(owner,&stats);assert(stats.static_uploads==2);
 input.use_overlap=0;input.metric=nullptr;++generation;
 assert(!metal_wave_owner_prepare(owner,generation,epoch,&input));
 assert(!metal_wave_owner_download(owner,generation,output_wave,output_project,nullptr,0,scales.data()));verify(false,false);
 // Replay, invalid metadata, zero norm, and non-finite data all preserve every
 // caller-owned destination; stale device generations are never exported.
 auto saved_wave=wave[0],saved_proj=project[0],saved_metric=metric;
 auto saved_scales=scales;
 assert(metal_wave_owner_prepare(owner,generation,epoch,&input)==METAL_BACKEND_INVALID_ARGUMENT);
 assert(metal_wave_owner_download(owner,generation,output_wave,output_project,metric.data(),ldmetric,scales.data())==METAL_BACKEND_INVALID_ARGUMENT);
 bases[1]=1;++generation;assert(metal_wave_owner_prepare(owner,generation,epoch,&input)==METAL_BACKEND_INVALID_ARGUMENT);bases[1]=2;
 for(auto &w:original) std::fill(w.begin(),w.end(),metal_complex64{0,0});++generation;
 assert(metal_wave_owner_prepare(owner,generation,epoch,&input)==METAL_BACKEND_RUNTIME_FAILED);
 original[0][0].real=std::numeric_limits<double>::quiet_NaN();++generation;
 assert(metal_wave_owner_prepare(owner,generation,epoch,&input)==METAL_BACKEND_RUNTIME_FAILED);
 assert(metal_wave_owner_download(owner,generation,output_wave,output_project,metric.data(),ldmetric,scales.data())==METAL_BACKEND_INVALID_ARGUMENT);
 for(size_t i=0;i<saved_wave.size();++i) assert(wave[0][i].real==saved_wave[i].real && wave[0][i].imag==saved_wave[i].imag);
 for(size_t i=0;i<saved_proj.size();++i) assert(project[0][i].real==saved_proj[i].real && project[0][i].imag==saved_proj[i].imag);
 for(size_t i=0;i<metric.size();++i) assert(metric[i].real==saved_metric[i].real && metric[i].imag==saved_metric[i].imag);
 assert(scales==saved_scales);
 metal_wave_owner_invalidate(owner);metal_wave_owner_get_stats(owner,&stats);assert(stats.invalidations==1);
 for(size_t b=0;b<bands;++b) for(size_t g=0;g<npl;++g) original[b][g]={.1*std::sin(.017*(g+3*b)),.2*std::cos(.013*(g+7*b))};
 ++generation;assert(!metal_wave_owner_prepare(owner,generation,epoch,&input));
 auto *saved_destination=output_project[2];output_project[2]=nullptr;
 assert(metal_wave_owner_download(owner,generation,output_wave,output_project,metric.data(),ldmetric,scales.data())==METAL_BACKEND_INVALID_ARGUMENT);
 for(size_t i=0;i<saved_wave.size();++i) assert(wave[0][i].real==saved_wave[i].real && wave[0][i].imag==saved_wave[i].imag);
 output_project[2]=saved_destination;
 assert(!metal_wave_owner_download(owner,generation,output_wave,output_project,nullptr,0,scales.data()));verify(false,false);
 metal_wave_owner_destroy(owner);metal_backend_destroy(backend);
 std::puts("physical wave owner: generations, transfers, PAW metric, lifecycle and atomic fallback passed");
}
