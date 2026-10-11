#include "metal_backend.h"
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
#include <cstdlib>
using C=std::complex<double>;
static C z(metal_complex64 x){return {x.real,x.imag};}
int main(){
 metal_backend *b=nullptr;assert(!metal_backend_create(&b,1));
 if(!metal_backend_gpu_enabled(b)){if(std::getenv("METAL_REQUIRE_GPU")) return 1;return 77;}
 const size_t nx=3,ny=5,nz=3,total=nx*ny*nz,npl=7,bands=3,nions=2,npro=3,lm=2,nt=2,qld=9,qchannels=2,pld=10,ldout=11;
 int types[]={1,2},maxl[]={2,1},base[]={0,2,3},ind[]={1,3,9,13,20,29,40};
 std::vector<metal_complex64> cr[bands],cw[bands],proj[bands];
 const metal_complex64 *crptr[bands],*cwptr[bands],*prptr[bands];
 std::vector<double> potential(total),ke(npl),eig(bands),d(lm*lm*nions),q(d.size()),rad(qld*qchannels*nt);
 std::vector<metal_complex64> cq(qchannels*nt),phase(pld*nions),out(ldout*bands,{123,-321});
 for(size_t i=0;i<total;++i) potential[i]=0.3*std::cos(0.17*i);
 for(size_t i=0;i<npl;++i) ke[i]=0.13*i;
 for(size_t i=0;i<d.size();++i){d[i]=0.11*std::sin(i+0.3);q[i]=0.07*std::cos(i+0.6);}
 for(size_t i=0;i<rad.size();++i) rad[i]=0.2*std::cos(0.2*i);
 for(size_t i=0;i<cq.size();++i) cq[i]={std::cos(0.3*i),std::sin(0.3*i)};
 for(size_t i=0;i<phase.size();++i) phase[i]={std::cos(0.13*i),std::sin(0.13*i)};
 for(size_t j=0;j<bands;++j){
  eig[j]=-0.2+0.1*j;cr[j].resize(total);cw[j].resize(npl);proj[j].resize(npro);
  for(size_t i=0;i<total;++i) cr[j][i]={std::sin(0.11*(i+j)),std::cos(0.07*(i+2*j))};
  for(size_t i=0;i<npl;++i) cw[j][i]={0.1*std::sin(i+j),0.2*std::cos(i-j)};
  for(size_t i=0;i<npro;++i) proj[j][i]={0.11*(i+j),-0.07*(i+2*j)};
  crptr[j]=cr[j].data();cwptr[j]=cw[j].data();prptr[j]=proj[j].data();
 }
 auto call=[&](){return metal_backend_hamiltonian_gpu(b,nx,ny,nz,npl,bands,nions,npro,lm,nt,qld,qchannels,pld,1,0,potential.data(),crptr,cwptr,prptr,ind,ke.data(),eig.data(),d.data(),q.data(),types,maxl,base,rad.data(),cq.data(),phase.data(),out.data(),ldout);};
 auto verify=[&](){double error=0;
  for(size_t band=0;band<bands;++band){
   C onsite[npro]{};
   for(size_t ion=0;ion<nions;++ion) for(int l=0;l<maxl[types[ion]-1];++l)
    for(int k=0;k<maxl[types[ion]-1];++k)
     onsite[base[ion]+l]+=(d[k+l*lm+ion*lm*lm]-eig[band]*q[k+l*lm+ion*lm*lm])*z(proj[band][base[ion]+k]);
   for(size_t g=0;g<npl;++g){
    size_t at=ind[g]-1,gx=at%nx,gy=(at/nx)%ny,gz=at/(nx*ny);C reference=0;
    for(size_t r=0;r<total;++r){size_t x=r%nx,y=(r/nx)%ny,zz=r/(nx*ny);
     double a=-2*std::acos(-1.0)*(double(gx*x)/nx+double(gy*y)/ny+double(gz*zz)/nz);
     reference+=z(cr[band][r])*potential[r]/double(total)*std::polar(1.0,a);
    }
    reference+=z(cw[band][g])*(ke[g]-eig[band]);
    for(size_t ion=0;ion<nions;++ion) for(int l=0;l<maxl[types[ion]-1];++l)
     reference+=onsite[base[ion]+l]*std::conj(z(cq[l+qchannels*(types[ion]-1)]))*rad[g+qld*(l+qchannels*(types[ion]-1))]*std::conj(z(phase[g+pld*ion]));
    error=std::max(error,std::abs(reference-z(out[band*ldout+g])));
   }
   for(size_t g=npl;g<ldout;++g) assert(out[band*ldout+g].real==123 && out[band*ldout+g].imag==-321);
  }
  std::printf("fused Hamiltonian direct FP64 reference max error %.3e\n",error);assert(error<1e-11);
 };
 uint64_t submits=metal_backend_command_count(b);assert(!call());assert(metal_backend_command_count(b)==submits+1);verify();
 assert(metal_backend_hamiltonian_static_uploads(b)==1);
 // Changed dynamic data must be consumed on every call, regardless of pointer reuse.
 cr[1][3].real+=0.12;proj[2][1].imag-=0.17;potential[7]+=0.2;d[1]+=0.04;
 assert(!call());verify();assert(metal_backend_hamiltonian_static_reuses(b)==1);
 // Explicit epoch invalidation is required before mutating static PAW data.
 metal_backend_hamiltonian_invalidate(b);rad[2]+=0.31;phase[0].real-=0.14;
 assert(!call());verify();assert(metal_backend_hamiltonian_static_uploads(b)==2);
 // Resident scatter -> inverse FFT -> potential -> forward FFT -> PAW/kinetic.
 // Build an independent real-space input from CW for the reference check.
 for(size_t band=0;band<bands;++band) for(size_t r=0;r<total;++r) {
  size_t x=r%nx,y=(r/nx)%ny,zz=r/(nx*ny);C value=0;
  for(size_t g=0;g<npl;++g){size_t at=ind[g]-1,gx=at%nx,gy=(at/nx)%ny,gz=at/(nx*ny);
   value+=z(cw[band][g])*std::polar(1.0,2*std::acos(-1.0)*(double(gx*x)/nx+double(gy*y)/ny+double(gz*zz)/nz));}
  cr[band][r]={value.real(),value.imag()};
 }
 auto before=metal_backend_command_count(b);
 assert(metal_backend_hamiltonian_gpu(b,nx,ny,nz,npl,bands,nions,npro,lm,nt,qld,qchannels,pld,1,1,potential.data(),crptr,cwptr,prptr,ind,ke.data(),eig.data(),d.data(),q.data(),types,maxl,base,rad.data(),cq.data(),phase.data(),out.data(),ldout)==0);
 assert(metal_backend_command_count(b)==before+1);verify();
 // Invalid layout cannot modify the caller's output before CPU fallback.
 auto saved=out;ind[2]=0;assert(call()==METAL_BACKEND_INVALID_ARGUMENT);
 for(size_t i=0;i<out.size();++i) assert(out[i].real==saved[i].real && out[i].imag==saved[i].imag);
 metal_backend_destroy(b);
}
