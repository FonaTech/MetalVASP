#include "metal_nonlocal.h"
#include "metal_nonlocal_policy.hpp"
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
using C=std::complex<double>;
C z(metal_complex64 a) {return {a.real,a.imag};}
metal_complex64 pair(C a) {return {a.real(),a.imag()};}
int main() {
 size_t product;
 assert(metal_nonlocal_policy::multiply(size_t(UINT32_MAX)+1,3,product) && product>UINT32_MAX);
 assert(!metal_nonlocal_policy::multiply(SIZE_MAX,2,product));
 assert(metal_nonlocal_policy::bands_per_dispatch(4096,size_t(UINT32_MAX))==UINT32_MAX/4096);
 metal_backend *backend=nullptr;assert(!metal_backend_create(&backend,1));
 if(!metal_backend_gpu_enabled(backend)) {metal_backend_destroy(backend);return getenv("METAL_REQUIRE_GPU") ? 1:77;}
 constexpr size_t npl=1025,bands=7,nions=4,npro=5,qch=3,ntypes=3;
 constexpr size_t ldw=npl+3,qld=npl+5,pld=npl+7,ld=npro+2;
 int types[]={1,2,3,1},lmmax[]={2,1,0};int64_t bases[]={0,2,3,3,5};
 std::vector<metal_complex64> wave(ldw*bands),phase(pld*nions),cq(qch*ntypes),out(ld*bands*4,{999,-777});
 std::vector<double> radial(qld*qch*ntypes),gradient(npl*3);
 for(size_t i=0;i<wave.size();++i)wave[i]={.07*sin(.17*i),.11*cos(.13*i)};
 for(size_t i=0;i<phase.size();++i)phase[i]=pair(std::polar(1.,.031*i));
 for(size_t i=0;i<cq.size();++i)cq[i]=pair(std::polar(1.,.21*i));
 for(size_t i=0;i<radial.size();++i)radial[i]=.03*cos(.071*i);
 for(size_t i=0;i<gradient.size();++i)gradient[i]=double(int(i%29)-14)+.25;
 metal_nonlocal_input in{1,1,wave.data(),ldw,radial.data(),qld,phase.data(),pld,cq.data(),types,lmmax,bases,gradient.data(),{0,6.283185307179586}};
 auto reference=[&](unsigned channels) {
  std::vector<C> r(npro*bands*channels);
  for(size_t c=0;c<channels;++c)for(size_t band=0;band<bands;++band)for(size_t ion=0;ion<nions;++ion)
   for(int l=0;l<lmmax[types[ion]-1];++l) {
    C sum=0;
    for(size_t g=0;g<npl;++g) {
     C v=z(phase[g+pld*ion])*z(wave[g+ldw*band])*radial[g+qld*(l+qch*(types[ion]-1))];
     if(c)v*=gradient[g+npl*(c-1)];sum+=v;
    }
    r[bases[ion]+l+npro*(band+bands*c)]=sum*z(cq[l+qch*(types[ion]-1)])*(c ? z(in.derivative_multiplier):C(1,0));
   }
  return r;
 };
 auto verify=[&](unsigned channels) {
  auto ref=reference(channels);double error=0;
  for(size_t c=0;c<channels;++c)for(size_t band=0;band<bands;++band) {
   for(size_t j=0;j<npro;++j)error=std::max(error,std::abs(z(out[j+ld*(band+bands*c)])-ref[j+npro*(band+bands*c)]));
   for(size_t j=npro;j<ld;++j)assert(out[j+ld*(band+bands*c)].real==999 && out[j+ld*(band+bands*c)].imag==-777);
  }
  printf("nonlocal channels=%u FP64 max_error=%.17g\n",channels,error);assert(error<2e-10);
 };
 metal_nonlocal_batch *ctx=nullptr;
 metal_nonlocal_shape s{npl,bands,nions,npro,qch,ntypes};assert(!metal_nonlocal_create(backend,s,&ctx));
 assert(!metal_nonlocal_project(ctx,&in,4,out.data(),ld));verify(4);
 metal_nonlocal_stats first{},second{};metal_nonlocal_get_stats(ctx,&first);
 radial[0]+=.004;assert(!metal_nonlocal_project(ctx,&in,1,out.data(),ld));verify(1);
 metal_nonlocal_get_stats(ctx,&second);assert(second.wave_import_bytes==first.wave_import_bytes && second.wave_reuses==1);
 assert(second.phase_import_bytes==first.phase_import_bytes && second.phase_reuses>first.phase_reuses);
 wave[2].real+=.034;++in.wave_generation;phase[11].imag-=.02;++in.phase_generation;
 assert(!metal_nonlocal_project(ctx,&in,4,out.data(),ld));verify(4);
 auto saved=out;types[0]=0;assert(metal_nonlocal_project(ctx,&in,4,out.data(),ld)==METAL_BACKEND_INVALID_ARGUMENT);
 assert(!memcmp(out.data(),saved.data(),out.size()*sizeof(out[0])));types[0]=1;
 gradient[npl-1]=std::numeric_limits<double>::quiet_NaN();assert(metal_nonlocal_project(ctx,&in,4,out.data(),ld)!=0);
 assert(!memcmp(out.data(),saved.data(),out.size()*sizeof(out[0])));gradient[npl-1]=.25;
 metal_nonlocal_destroy(ctx);
 setenv("VASP_METAL_NONLOCAL_WORKSPACE_BYTES","4096",1);
 setenv("VASP_METAL_NONLOCAL_G_WINDOW","17",1);setenv("VASP_METAL_NONLOCAL_SITE_WINDOW","2",1);setenv("VASP_METAL_NONLOCAL_BAND_WINDOW","2",1);
 assert(!metal_nonlocal_create(backend,s,&ctx));assert(!metal_nonlocal_project(ctx,&in,4,out.data(),ld));verify(4);
 metal_nonlocal_get_stats(ctx,&second);assert(second.windows>100 && second.planned_workspace_peak_bytes<=4096);
 saved=out;metal_nonlocal_destroy(ctx);assert(!metal_nonlocal_create(backend,s,&ctx));
 setenv("VASP_METAL_NONLOCAL_FAIL_WINDOW","2",1);
 assert(metal_nonlocal_project(ctx,&in,4,out.data(),ld)==METAL_BACKEND_RUNTIME_FAILED);
 assert(!memcmp(out.data(),saved.data(),out.size()*sizeof(out[0])));unsetenv("VASP_METAL_NONLOCAL_FAIL_WINDOW");
 setenv("VASP_METAL_MEMORY_BUDGET_BYTES","1",1);
 assert(metal_nonlocal_project(ctx,&in,4,out.data(),ld)==METAL_BACKEND_ALLOCATION_FAILED);
 assert(!memcmp(out.data(),saved.data(),out.size()*sizeof(out[0])));
 unsetenv("VASP_METAL_MEMORY_BUDGET_BYTES");unsetenv("VASP_METAL_NONLOCAL_WORKSPACE_BYTES");unsetenv("VASP_METAL_NONLOCAL_G_WINDOW");
 unsetenv("VASP_METAL_NONLOCAL_SITE_WINDOW");unsetenv("VASP_METAL_NONLOCAL_BAND_WINDOW");
 metal_nonlocal_destroy(ctx);metal_backend_destroy(backend);
 puts("nonlocal FP64/generations/wide-policy/windows/pressure/late-failure/padding passed");
}
