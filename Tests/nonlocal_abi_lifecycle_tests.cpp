#include "metal_nonlocal.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void *vasp_metal_nonlocal_create(size_t,size_t,size_t,size_t,size_t,size_t);
extern "C" void vasp_metal_nonlocal_destroy(void *);
extern "C" int vasp_metal_nonlocal_project(void *,uint64_t,uint64_t,const metal_complex64 *,size_t,
 const double *,size_t,const metal_complex64 *,size_t,const metal_complex64 *,const int *,const int *,
 const int64_t *,const double *,const metal_complex64 *,int,metal_complex64 *,size_t);
extern "C" int vasp_metal_fft3d_batch(int,int,int,int,int,metal_complex64 *,int);
extern "C" void vasp_metal_resident_fft_cache_stats(uint64_t *,uint64_t *,uint64_t *);
using C=std::complex<double>;
static C z(metal_complex64 a) {return {a.real,a.imag};}

int main() {
 setenv("VASP_METAL","1",1);setenv("VASP_METAL_NONLOCAL","1",1);
 setenv("VASP_METAL_REQUIRE_GPU","1",1);setenv("VASP_METAL_FFT_MIN_POINTS","0",1);
 setenv("VASP_METAL_RESIDENT_FFT","1",1);
 constexpr size_t ng=257,nb=3,ni=2,np=3,nq=2,nt=2,ld=np+2;
 int types[]{1,2},lm[]{2,1};int64_t bases[]{0,2,3};
 std::vector<metal_complex64> wave(ng*nb),phase(ng*ni),cq(nq*nt),out(ld*nb*4,{999,-777});
 std::vector<double> radial(ng*nq*nt),grad(ng*3);
 for(size_t i=0;i<wave.size();++i)wave[i]={.03*std::sin(.07*i),.08*std::cos(.09*i)};
 for(size_t i=0;i<phase.size();++i)phase[i]={std::cos(.02*i),std::sin(.02*i)};
 for(size_t i=0;i<cq.size();++i)cq[i]={std::cos(.3*i),std::sin(.3*i)};
 for(size_t i=0;i<radial.size();++i)radial[i]=.06*std::cos(.013*i);
 for(size_t i=0;i<grad.size();++i)grad[i]=double(int(i%17)-8)+.25;
 metal_complex64 multiplier{0,6.283185307179586};
 std::vector<metal_complex64> fft(16);
 for(size_t i=0;i<fft.size();++i)fft[i]={.01*double(i),.02*double(i)};
 const auto fft_reference=fft;
 uint64_t creates=0,reuses=0,bytes=0;
 vasp_metal_resident_fft_cache_stats(&creates,&reuses,&bytes);
 const uint64_t initial_creates=creates,initial_reuses=reuses;
 double max_error=0;
 for(unsigned round=0;round<16;++round) {
  // Cache created before an owner must remain attached to the same backend.
  fft=fft_reference;assert(vasp_metal_fft3d_batch(4,2,2,1,16,fft.data(),-1)==1);
  void *owner=vasp_metal_nonlocal_create(ng,nb,ni,np,nq,nt);assert(owner);
  for(unsigned reuse=0;reuse<2;++reuse) {
   assert(vasp_metal_nonlocal_project(owner,1,1,wave.data(),ng,radial.data(),ng,phase.data(),ng,
      cq.data(),types,lm,bases,grad.data(),&multiplier,4,out.data(),ld)==1);
   for(size_t channel=0;channel<4;++channel)for(size_t band=0;band<nb;++band)
    for(size_t ion=0;ion<ni;++ion)for(int l=0;l<lm[types[ion]-1];++l) {
     C sum=0;for(size_t g=0;g<ng;++g) {
      C value=z(wave[g+ng*band])*z(phase[g+ng*ion])*radial[g+ng*(l+nq*(types[ion]-1))];
      if(channel)value*=grad[g+ng*(channel-1)];sum+=value;
     }
     sum*=z(cq[l+nq*(types[ion]-1)])*(channel ? z(multiplier):C(1,0));
     max_error=std::max(max_error,std::abs(z(out[bases[ion]+l+ld*(band+nb*channel)])-sum));
    }
   for(size_t col=0;col<nb*4;++col)for(size_t j=np;j<ld;++j)
    assert(out[j+ld*col].real==999 && out[j+ld*col].imag==-777);
  }
  auto saved=out;setenv("VASP_METAL_NONLOCAL_FAIL_WINDOW","3",1);
  assert(vasp_metal_nonlocal_project(owner,1,1,wave.data(),ng,radial.data(),ng,phase.data(),ng,
     cq.data(),types,lm,bases,grad.data(),&multiplier,4,out.data(),ld)==0);
  assert(!std::memcmp(out.data(),saved.data(),out.size()*sizeof(out[0])));
  unsetenv("VASP_METAL_NONLOCAL_FAIL_WINDOW");vasp_metal_nonlocal_destroy(owner);
  assert(vasp_metal_fft3d_batch(4,2,2,1,16,fft.data(),1)==1);
  for(size_t i=0;i<fft.size();++i)assert(std::abs(z(fft[i])-16.*z(fft_reference[i]))<1e-10);
 }
 vasp_metal_resident_fft_cache_stats(&creates,&reuses,&bytes);
 assert(creates==initial_creates+1 && reuses==initial_reuses+31 && bytes>0);
 assert(max_error<2e-10);
 std::printf("mixed FFT/nonlocal owners, padding, failure atomicity and FP64 max_error=%.17g passed\n",max_error);
 // The supervising gate also reads actual ABI JSON at normal process exit:
 // projection GPU/wait/conversion increments and cumulative command totals
 // must remain visible after the final FFT. A replaced backend fails that gate.
}
