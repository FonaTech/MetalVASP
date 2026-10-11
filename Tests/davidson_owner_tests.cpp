#include "metal_wave_owner_davidson.h"
#include "vasp_metal_davidson_abi.h"
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
 if(!metal_backend_gpu_enabled(backend)) {metal_backend_destroy(backend);return std::getenv("METAL_REQUIRE_GPU") ? 1:77;}
 constexpr size_t ng=257,nb=3,na=5,ni=2,np=3,lm=2,nt=2,qc=2;
 constexpr size_t ldpre=ng+2,qld=ng+3,phld=ng+4,ldbw=ng+5,ldbp=np+3,ldm=np+2,ldc=na+2;
 int types[]={1,2},counts[]={2,1},bases[]={0,2,3};
 std::vector<metal_complex64> original[nb],output[nb],projector[nb];
 const metal_complex64 *input_wave[nb];metal_complex64 *out_wave[nb],*out_projector[nb],*dummy_projector[nb];
 for(size_t b=0;b<nb;++b) {
   original[b].resize(ng);output[b].resize(ng+1,{901,-902});projector[b].resize(np+1,{901,-902});
   for(size_t g=0;g<ng;++g) original[b][g]={.15*std::sin(.023*(g+b*7)),.1*std::cos(.019*(g+b*13))};
   input_wave[b]=original[b].data();out_wave[b]=output[b].data();out_projector[b]=projector[b].data();dummy_projector[b]=projector[b].data();
 }
 std::vector<double> pre(ldpre*nb),radial(qld*qc*nt),qreal(lm*lm*ni),scales(nb,-333);
 std::vector<metal_complex64> phase(phld*ni),cq(qc*nt),qcomplex(lm*lm*ni),basiswave(ldbw*na),basisprojector(ldbp*na);
 std::vector<metal_complex64> metric(ldm*nb,{901,-902}),overlap(ldc*nb,{901,-902});
 for(size_t i=0;i<pre.size();++i) pre[i]=.55+.05*std::sin(.11*i);
 for(size_t i=0;i<radial.size();++i) radial[i]=.03*std::cos(.09*i);
 for(size_t i=0;i<phase.size();++i) phase[i]=pair(std::polar(1.,.007*i));
 for(size_t i=0;i<cq.size();++i) cq[i]=pair(std::polar(.9,.17*i));
 for(size_t ion=0;ion<ni;++ion) {
   qreal[lm*lm*ion]=.25;qreal[1+lm+lm*lm*ion]=.18;
   qreal[1+lm*lm*ion]=qreal[lm+lm*lm*ion]=.012;
   for(size_t j=0;j<lm*lm;++j) qcomplex[j+lm*lm*ion]={qreal[j+lm*lm*ion],0};
   qcomplex[1+lm*lm*ion].imag=.023;qcomplex[lm+lm*lm*ion].imag=-.023;
 }
 for(size_t b=0;b<na;++b) {
   for(size_t g=0;g<ng;++g) basiswave[g+ldbw*b]={.012*std::cos(.037*(g+b*11)),.017*std::sin(.031*(g+b*3))};
   for(size_t j=0;j<np;++j) basisprojector[j+ldbp*b]={.02*std::sin(.17*(j+b*7)),.03*std::cos(.13*(j+b*5))};
 }
 metal_wave_owner_shape shape{ng,nb,ni,np,lm,nt,qc};metal_wave_owner *owner=nullptr;
 assert(!metal_wave_owner_create(backend,shape,&owner));
 metal_wave_prepare_input input{input_wave,pre.data(),ldpre,qld,phld,types,counts,bases,radial.data(),cq.data(),phase.data(),qreal.data(),1,1};
 metal_wave_basis_input basis{na,ldbw,ldbp,basiswave.data(),basisprojector.data(),1};
 auto verify=[&](bool real,bool use_metric,bool compare_overlap) {
   double err=0;
   for(size_t b=0;b<nb;++b) {
     std::vector<C> w(ng),p(np),m(np),c(na);
     for(size_t g=0;g<ng;++g) w[g]=z(original[b][g])*pre[g+ldpre*b];
     for(size_t ion=0;ion<ni;++ion) for(int l=0;l<counts[types[ion]-1];++l) {
       C value=0;for(size_t g=0;g<ng;++g) value+=radial[g+qld*(l+qc*(types[ion]-1))]*z(phase[g+phld*ion])*w[g];
       p[bases[ion]+l]=value*z(cq[l+qc*(types[ion]-1)]);
     }
     if(use_metric) for(size_t ion=0;ion<ni;++ion) for(int l=0;l<counts[types[ion]-1];++l)
       for(int k=0;k<counts[types[ion]-1];++k) {size_t at=k+l*lm+ion*lm*lm;m[bases[ion]+l]+=(real ? C(qreal[at],0):z(qcomplex[at]))*p[bases[ion]+k];}
     double norm=0;for(auto value:w) norm+=std::norm(value);
     if(use_metric) for(size_t j=0;j<np;++j) norm+=std::real(std::conj(p[j])*m[j]);
     const double scale=1/std::sqrt(norm);
     for(auto &value:w) value*=scale;for(auto &value:p) value*=scale;for(auto &value:m) value*=scale;
     for(size_t a=0;a<na;++a) {
       for(size_t g=0;g<ng;++g) c[a]+=std::conj(z(basiswave[g+ldbw*a]))*w[g];
       if(use_metric) for(size_t j=0;j<np;++j) c[a]+=std::conj(z(basisprojector[j+ldbp*a]))*m[j];
       if(compare_overlap) err=std::max(err,std::abs(z(overlap[a+ldc*b])-c[a]));
     }
     for(size_t a=0;a<na;++a) {
       for(size_t g=0;g<ng;++g) w[g]-=z(basiswave[g+ldbw*a])*c[a];
       for(size_t j=0;j<np;++j) p[j]-=z(basisprojector[j+ldbp*a])*c[a];
     }
     for(size_t g=0;g<ng;++g) err=std::max(err,std::abs(z(output[b][g])-w[g]));
     for(size_t j=0;j<np;++j) {err=std::max(err,std::abs(z(projector[b][j])-p[j]));if(use_metric) err=std::max(err,std::abs(z(metric[j+ldm*b])-m[j]));}
     assert(std::abs(scales[b]-scale)<2e-12);
     assert(output[b][ng].real==901 && projector[b][np].real==901);
     for(size_t j=np;j<ldm;++j) assert(metric[j+ldm*b].real==901);
     if(compare_overlap) for(size_t a=na;a<ldc;++a) assert(overlap[a+ldc*b].real==901);
   }
   std::printf("Davidson PAW residual+overlap+two updates real=%d metric=%d error=%.3e\n",real,use_metric,err);assert(err<1e-10);
 };
 uint64_t generation=1,commands=metal_backend_command_count(backend);
 assert(!metal_wave_owner_prepare(owner,generation,1,&input));
 assert(!metal_wave_owner_davidson_update(owner,generation,&basis));
 assert(metal_backend_command_count(backend)==commands+2);
 metal_wave_owner_stats stats{};metal_wave_owner_get_stats(owner,&stats);assert(stats.host_export_bytes==0);
 const uint64_t dynamic=ng*nb*24+lm*lm*ni*16;
 const uint64_t physical_static=ng*qc*nt*8+qc*nt*16+ng*ni*16+np*20;
 const uint64_t basis_bytes=(ng+np)*na*16;
 assert(stats.host_import_bytes==dynamic+physical_static+basis_bytes+na*nb*16);
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,metric.data(),ldm,scales.data()));
 assert(!metal_wave_owner_davidson_download_overlap(owner,generation,overlap.data(),ldc));verify(true,true,true);
 // The combined API submits all eight kernels once and checks control status
 // only after that command completes; scientific intermediates stay resident.
 ++generation;commands=metal_backend_command_count(backend);
 assert(!metal_wave_owner_davidson_prepare(owner,generation,1,&input,&basis));
 assert(metal_backend_command_count(backend)==commands+1);
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,metric.data(),ldm,scales.data()));verify(true,true,false);
 // New residual + CQIJ at identical addresses; the explicitly unchanged WA
 // basis may be reused. Two submissions still have zero intermediate exports.
 original[1][9].real+=.07;pre[5]+=.1;input.real_metric=0;input.metric=qcomplex.data();++generation;
 assert(!metal_wave_owner_prepare(owner,generation,1,&input));assert(!metal_wave_owner_davidson_update(owner,generation,&basis));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,metric.data(),ldm,scales.data()));verify(false,true,false);
 metal_wave_davidson_stats ds{};metal_wave_owner_davidson_get_stats(owner,&ds);assert(ds.steps==3 && ds.basis_uploads==1 && ds.basis_reuses==2 && ds.basis_import_bytes==basis_bytes);
 assert(ds.control_status_host_bytes==3*nb*sizeof(uint32_t));
 // A CPU eig/GEMM basis mutation requires an explicit generation change.
 basiswave[7].imag-=.04;basisprojector[2].real+=.05;++basis.generation;++generation;
 input.use_overlap=0;input.metric=nullptr;
 assert(!metal_wave_owner_prepare(owner,generation,1,&input));assert(!metal_wave_owner_davidson_update(owner,generation,&basis));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,nullptr,0,scales.data()));verify(false,false,false);
 const auto saved_output=output[0],saved_projector=projector[0],saved_metric=metric;const auto saved_scales=scales;
 // A previously exported preparation cannot be upgraded to a hidden closure.
 assert(metal_wave_owner_davidson_update(owner,generation,&basis)==METAL_BACKEND_INVALID_ARGUMENT);
 ++generation;assert(!metal_wave_owner_prepare(owner,generation,1,&input));basis.ldwave=ng-1;
 assert(metal_wave_owner_davidson_update(owner,generation,&basis)==METAL_BACKEND_INVALID_ARGUMENT);basis.ldwave=ldbw;
 assert(metal_wave_owner_download(owner,generation,out_wave,out_projector,nullptr,0,scales.data())==METAL_BACKEND_INVALID_ARGUMENT);
 ++generation;assert(!metal_wave_owner_prepare(owner,generation,1,&input));assert(!metal_wave_owner_davidson_update(owner,generation,&basis));
 out_projector[nb-1]=nullptr;assert(metal_wave_owner_download(owner,generation,out_wave,out_projector,nullptr,0,scales.data())==METAL_BACKEND_INVALID_ARGUMENT);
 out_projector[nb-1]=projector[nb-1].data();assert(output[0][0].real==saved_output[0].real && projector[0][0].real==saved_projector[0].real && scales==saved_scales);
 // Non-finite basis propagates to final staging and fails before any write.
 basiswave[0].real=std::numeric_limits<double>::quiet_NaN();++basis.generation;++generation;
 assert(!metal_wave_owner_prepare(owner,generation,1,&input));assert(!metal_wave_owner_davidson_update(owner,generation,&basis));
 assert(metal_wave_owner_download(owner,generation,out_wave,out_projector,nullptr,0,scales.data())==METAL_BACKEND_RUNTIME_FAILED);
 assert(output[0][0].real==saved_output[0].real && projector[0][0].real==saved_projector[0].real && scales==saved_scales);
 basiswave[0].real=.02;++basis.generation;metal_wave_owner_invalidate(owner);
 ++generation;assert(!metal_wave_owner_prepare(owner,generation,2,&input));assert(!metal_wave_owner_davidson_update(owner,generation,&basis));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,nullptr,0,scales.data()));verify(false,false,false);
 // In the single-command API a failed norm is checked after all encoders;
 // no status or scientific result is consumed between the eight kernels.
 std::vector<metal_complex64> backup[nb];for(size_t b=0;b<nb;++b) {backup[b]=original[b];std::fill(original[b].begin(),original[b].end(),metal_complex64{0,0});}
 ++generation;commands=metal_backend_command_count(backend);
 assert(metal_wave_owner_davidson_prepare(owner,generation,2,&input,&basis)==METAL_BACKEND_RUNTIME_FAILED);
 assert(metal_backend_command_count(backend)==commands+1);
 assert(metal_wave_owner_download(owner,generation,out_wave,out_projector,nullptr,0,scales.data())==METAL_BACKEND_INVALID_ARGUMENT);
 for(size_t b=0;b<nb;++b) std::copy(backup[b].begin(),backup[b].end(),original[b].begin());
 // An encoding-time validation error discards the uncommitted command.
 ++generation;commands=metal_backend_command_count(backend);basis.ldprojectors=np-1;
 assert(metal_wave_owner_davidson_prepare(owner,generation,2,&input,&basis)==METAL_BACKEND_INVALID_ARGUMENT);
 assert(metal_backend_command_count(backend)==commands);basis.ldprojectors=ldbp;
 ++generation;assert(!metal_wave_owner_davidson_prepare(owner,generation,2,&input,&basis));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,nullptr,0,scales.data()));verify(false,false,false);
 // Delta uploads are checked against the same independent FP64 reference.
 input.use_overlap=1;input.metric=qcomplex.data();input.real_metric=0;
 metal_wave_basis_transfer_stats transfer_before{},transfer_after{};
 metal_wave_owner_davidson_get_transfer_stats(owner,&transfer_before);
 metal_wave_owner_davidson_get_stats(owner,&ds);const auto bytes_before=ds.basis_import_bytes;
 metal_wave_basis_delta delta{0,0,0};++basis.generation;++generation;
 assert(!metal_wave_owner_davidson_prepare_delta(owner,generation,2,&input,&basis,&delta));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,metric.data(),ldm,scales.data()));verify(false,true,false);
 ++generation;assert(!metal_wave_owner_davidson_prepare_delta(owner,generation,2,&input,&basis,&delta));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,metric.data(),ldm,scales.data()));verify(false,true,false);
 basiswave[7+2*ldbw].imag+=.033;basisprojector[2+2*ldbp].real-=.021;
 delta={basis.generation,2,1};++basis.generation;++generation;
 assert(!metal_wave_owner_davidson_prepare_delta(owner,generation,2,&input,&basis,&delta));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,metric.data(),ldm,scales.data()));verify(false,true,false);
 // A missed epoch cannot be represented by the last span: refresh all columns.
 basiswave[8+ldbw].real+=.02;basiswave[9+3*ldbw].imag-=.04;
 delta={basis.generation,3,1};basis.generation+=2;++generation;
 assert(!metal_wave_owner_davidson_prepare_delta(owner,generation,2,&input,&basis,&delta));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,metric.data(),ldm,scales.data()));verify(false,true,false);
 // Explicit invalidation defeats a superficially valid partial provenance.
 metal_wave_owner_invalidate(owner);delta={basis.generation,0,1};++basis.generation;++generation;
 assert(!metal_wave_owner_davidson_prepare_delta(owner,generation,2,&input,&basis,&delta));
 assert(!metal_wave_owner_download(owner,generation,out_wave,out_projector,metric.data(),ldm,scales.data()));verify(false,true,false);
 metal_wave_owner_davidson_get_transfer_stats(owner,&transfer_after);metal_wave_owner_davidson_get_stats(owner,&ds);
 assert(transfer_after.full_uploads-transfer_before.full_uploads==3);
 assert(transfer_after.partial_uploads-transfer_before.partial_uploads==1);
 assert(transfer_after.reuses-transfer_before.reuses==1);
 assert(transfer_after.uploaded_columns-transfer_before.uploaded_columns==3*na+1);
 assert(ds.basis_import_bytes-bytes_before==(3*na+1)*(ng+np)*16);
 auto delta_saved=output[0];delta={basis.generation,na,1};++generation;
 commands=metal_backend_command_count(backend);
 assert(metal_wave_owner_davidson_prepare_delta(owner,generation,2,&input,&basis,&delta)==METAL_BACKEND_INVALID_ARGUMENT);
 assert(metal_backend_command_count(backend)==commands && output[0][0].real==delta_saved[0].real);
 // Describing changes without advancing the version is a caller error.
 delta={basis.generation,0,1};++generation;
 assert(metal_wave_owner_davidson_prepare_delta(owner,generation,2,&input,&basis,&delta)==METAL_BACKEND_INVALID_ARGUMENT);
 metal_wave_owner_destroy(owner);metal_backend_destroy(backend);
 input.use_overlap=0;input.metric=nullptr;
 // The independent ABI is opt-in, retains leases across EDDAV calls, and
 // still imports current wave/metric/basis after CPU changes between SCFs.
 setenv("VASP_METAL","1",1);setenv("VASP_METAL_WAVE_OWNER","1",1);
 setenv("VASP_METAL_DAVIDSON_CACHE_BYTES","268435456",1);unsetenv("VASP_METAL_DAVIDSON_OWNER");void *handle=nullptr;
 auto abi=[&](uint64_t gen,int nk=1) {
   return vasp_metal_davidson_prepare(&handle,gen,ng,nb,ni,np,lm,nt,qc,qld,phld,input.real_metric,input.use_overlap,
     const_cast<metal_complex64 *const *>(input_wave),dummy_projector,pre.data(),ldpre,types,counts,bases,radial.data(),cq.data(),phase.data(),input.metric,
     metric.data(),ldm,scales.data(),na,basiswave.data(),basisprojector.data(),ldbw,ldbp,out_wave,out_projector,nk,1,1);
 };
 assert(abi(1)==0 && !handle);setenv("VASP_METAL_DAVIDSON_OWNER","1",1);
 assert(abi(1)==1 && handle);verify(false,false,false);vasp_metal_davidson_release(&handle);assert(!handle);
 vasp_metal_davidson_counters counters{};vasp_metal_davidson_get_counters(&counters);
 assert(counters.allocations==1 && counters.live==0 && counters.cached_entries==1 && counters.static_uploads==1 && counters.submissions==1);
 assert(counters.control_status_host_bytes==nb*sizeof(uint32_t));
 original[2][8].imag+=.08;assert(abi(1)==1);verify(false,false,false);
 // Epoch changed with bit-identical data must reuse verified static data.
 vasp_metal_davidson_invalidate();assert(abi(2)==1);verify(false,false,false);
 vasp_metal_davidson_get_counters(&counters);assert(counters.reuses==1 && counters.static_uploads==1 && counters.static_reuses==2);
 // Same addresses with changed static PAW require an explicit epoch boundary.
 radial[0]+=.031;phase[6].imag-=.014;vasp_metal_davidson_invalidate();assert(abi(3)==1);verify(false,false,false);
 vasp_metal_davidson_get_counters(&counters);assert(counters.static_uploads==2);
 auto abi_saved=output[0];out_projector[2]=nullptr;assert(abi(4)==0);out_projector[2]=projector[2].data();assert(output[0][0].real==abi_saved[0].real);
 assert(abi(5)==1);verify(false,false,false);
 assert(abi(6,2)==1);verify(false,false,false);vasp_metal_davidson_release(&handle);
 assert(abi(1,1)==1);verify(false,false,false);vasp_metal_davidson_release(&handle);
 vasp_metal_davidson_get_counters(&counters);assert(counters.live==0 && counters.cached_entries==2 && counters.allocations==2 && counters.reuses>=2);
 assert(counters.retained_cache_bytes<=uint64_t(256)*1024*1024);
 setenv("VASP_METAL_DAVIDSON_CACHE_BYTES","0",1);assert(abi(1,1)==1);vasp_metal_davidson_release(&handle);
 vasp_metal_davidson_get_counters(&counters);assert(counters.cached_entries==0 && counters.retained_cache_bytes==0 && counters.live==0);
 assert(counters.submissions==9 && counters.control_status_host_bytes==9*nb*sizeof(uint32_t));
 assert(counters.overlap_zero_init_host_bytes==9*na*nb*16);
 input.use_overlap=1;input.metric=qcomplex.data();
 // The additive ABI keeps unchanged WA on device while its residual varies.
 setenv("VASP_METAL_DAVIDSON_CACHE_BYTES","268435456",1);
 auto abi_delta=[&](uint64_t gen,uint64_t epoch,int first=0,int count=0,int nk=1) {
   return vasp_metal_davidson_prepare_delta(&handle,gen,ng,nb,ni,np,lm,nt,qc,qld,phld,input.real_metric,input.use_overlap,
     const_cast<metal_complex64 *const *>(input_wave),dummy_projector,pre.data(),ldpre,types,counts,bases,radial.data(),cq.data(),phase.data(),input.metric,
     metric.data(),ldm,scales.data(),na,basiswave.data(),basisprojector.data(),ldbw,ldbp,out_wave,out_projector,nk,1,1,epoch,first,count);
 };
 vasp_metal_davidson_get_basis_counters(&transfer_before);
 vasp_metal_davidson_get_counters(&counters);const auto abi_bytes_before=counters.basis_imports;
 assert(abi_delta(1,1)==1);verify(false,true,false);
 original[1][10].real+=.03;assert(abi_delta(2,1)==1);verify(false,true,false);
 basiswave[11+2*ldbw].real+=.06;basisprojector[1+2*ldbp].imag-=.02;
 assert(abi_delta(3,2,2,1)==1);verify(false,true,false);
 // Repeated calls can retain the last dirty span after its epoch is accepted.
 assert(abi_delta(4,2,2,1)==1);verify(false,true,false);
 basiswave[12+ldbw].imag+=.05;basiswave[13+3*ldbw].real-=.04;
 assert(abi_delta(5,4,3,1)==1);verify(false,true,false);
 vasp_metal_davidson_release(&handle);
 assert(abi_delta(1,1)==1);verify(false,true,false);vasp_metal_davidson_release(&handle);
 vasp_metal_davidson_get_basis_counters(&transfer_after);vasp_metal_davidson_get_counters(&counters);
 assert(transfer_after.full_uploads-transfer_before.full_uploads==3);
 assert(transfer_after.partial_uploads-transfer_before.partial_uploads==1);
 assert(transfer_after.reuses-transfer_before.reuses==2);
 assert(counters.basis_imports-abi_bytes_before==(3*na+1)*(ng+np)*16);
 // Rank/k/shape leases, bad spans and exporter failures never reuse stale WA.
 assert(abi_delta(1,1,0,0,2)==1);verify(false,true,false);
 auto abi_delta_saved=output[0];assert(abi_delta(2,2,na,1,2)==0);
 assert(output[0][0].real==abi_delta_saved[0].real);
 out_projector[2]=nullptr;assert(abi_delta(3,1,0,0,2)==0);out_projector[2]=projector[2].data();
 assert(abi_delta(4,1,0,0,2)==1);verify(false,true,false);
 assert(abi_delta(5,0,0,0,2)==0);vasp_metal_davidson_release(&handle);
 std::puts("Davidson owner: independent FP64 references, no intermediate exports, generations, atomic failure, cross-EDDAV lease reuse passed");
}
