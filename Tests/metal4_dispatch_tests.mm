#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>
#include "metal4_dispatch.h"
#include "metal_compile_policy.hpp"
#include "double_single.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

// Independent reference exercise of the same DS arithmetic and GEMM constant
// layout used by production. It checks a real MTL4 compiler + compute encoder;
// production backend tests additionally check the full library and BLAS ABI.
static const char *shader=R"METAL(
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;
struct DS { float hi,lo; };
struct DSC { DS real,imag; };
inline DS ds_add(DS a,DS b) {
 float sum=a.hi+b.hi, vb=sum-a.hi;
 float error=((a.hi-(sum-vb))+(b.hi-vb))+a.lo+b.lo;
 float hi=sum+error; return {hi,error-(hi-sum)};
}
inline DS ds_mul(DS a,DS b) {
 float product=a.hi*b.hi;
 float error=fma(a.hi,b.hi,-product)+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo;
 float hi=product+error; return {hi,error-(hi-product)};
}
inline DSC add(DSC a,DSC b) { return {ds_add(a.real,b.real),ds_add(a.imag,b.imag)}; }
inline DSC mul(DSC a,DSC b) {
 DS rr=ds_mul(a.real,b.real), ii=ds_mul(a.imag,b.imag);
 return {ds_add(rr,DS{-ii.hi,-ii.lo}),ds_add(ds_mul(a.real,b.imag),ds_mul(a.imag,b.real))};
}
inline DSC unpack(float4 a) { return {{a.x,a.y},{a.z,a.w}}; }
struct GemmParams { uint m,n,k; float4 alpha,beta; };
kernel void ds_zgemm(device const float4 *a [[buffer(0)]],
                     device const float4 *b [[buffer(1)]],
                     device float4 *c [[buffer(2)]],
                     constant GemmParams &p [[buffer(3)]],
                     uint2 grid [[thread_position_in_grid]]) {
 uint row=grid.y,col=grid.x;if(row>=p.m || col>=p.n)return;
 ulong index=ulong(row)*p.n+col;
 DSC sum={{0,0},{0,0}};
 for(uint inner=0;inner<p.k;++inner) sum=add(sum,mul(unpack(a[ulong(row)*p.k+inner]),unpack(b[ulong(inner)*p.n+col])));
 DSC value=add(mul(unpack(p.alpha),sum),mul(unpack(p.beta),unpack(c[index])));
 c[index]=float4(value.real.hi,value.real.lo,value.imag.hi,value.imag.lo);
}
)METAL";

using cd=std::complex<double>;
struct Parameters { uint32_t m,n,k; vector_float4 alpha,beta; };
vector_float4 pack(cd value) {
 auto r=DoubleSingle::split(value.real()), i=DoubleSingle::split(value.imag());
 return {r.hi,r.lo,i.hi,i.lo};
}
cd unpack(vector_float4 value) { return {double(value.x)+value.y,double(value.z)+value.w}; }
double median(std::vector<double> values) {
 std::sort(values.begin(),values.end()); return values[values.size()/2];
}

int main() {
 @autoreleasepool {
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  if(!device) { std::puts("Metal device unavailable"); return 77; }
  auto compiled=metal_policy::compile(device,[NSString stringWithUTF8String:shader]);
  if(!compiled.library) { std::fprintf(stderr,"library: %s\n",compiled.error.c_str()); return 1; }
  char error[2048]{};
  metal4_dispatch *modern=metal4_dispatch_create(device,compiled.library,error,sizeof(error));
  if(!modern) { std::fprintf(stderr,"Metal 4 unavailable: %s\n",error); return 77; }
  NSError *problem=nil;
  id<MTLComputePipelineState> legacy=[device newComputePipelineStateWithFunction:[compiled.library newFunctionWithName:@"ds_zgemm"] error:&problem];
  id<MTLCommandQueue> queue=[device newCommandQueue];
  if(!legacy || !queue) { metal4_dispatch_destroy(modern); return 1; }
  std::printf("{\"device\":\"%s\",\"language\":\"%s\",\"cases\":[\n",device.name.UTF8String,metal_policy::language_name(compiled.language).c_str());
  bool all_ok=true; const size_t shapes[][3]={{33,47,29},{256,256,256},{512,64,352}};
  for(size_t shape=0;shape<3;++shape) {
   const size_t m=shapes[shape][0],n=shapes[shape][1],k=shapes[shape][2];
   std::mt19937_64 rng(20261009+shape); std::uniform_real_distribution<double> draw(-1.,1.);
   std::vector<cd> a(m*k),b(k*n),c(m*n),reference(m*n);
   for(auto &value:a) value={draw(rng),draw(rng)};
   for(auto &value:b) value={draw(rng),draw(rng)};
   for(auto &value:c) value={draw(rng),draw(rng)};
   // Nontrivial alpha/beta and deliberately nearly cancelling input pairs.
   for(size_t row=0;row<m;++row) for(size_t inner=1;inner<k;inner+=2)
    a[row*k+inner]=-a[row*k+inner-1]+cd(1e-8,-2e-8);
   const cd alpha(.7,-.3),beta(-.2,.15);
   for(size_t row=0;row<m;++row) for(size_t col=0;col<n;++col) {
    cd sum{}; for(size_t inner=0;inner<k;++inner) sum+=a[row*k+inner]*b[inner*n+col];
    reference[row*n+col]=alpha*sum+beta*c[row*n+col];
   }
   id<MTLBuffer> ab=[device newBufferWithLength:a.size()*sizeof(vector_float4) options:MTLResourceStorageModeShared];
   id<MTLBuffer> bb=[device newBufferWithLength:b.size()*sizeof(vector_float4) options:MTLResourceStorageModeShared];
   id<MTLBuffer> cb=[device newBufferWithLength:c.size()*sizeof(vector_float4) options:MTLResourceStorageModeShared];
   if(!ab||!bb||!cb) { metal4_dispatch_destroy(modern); return 1; }
   for(size_t i=0;i<a.size();++i) static_cast<vector_float4 *>(ab.contents)[i]=pack(a[i]);
   for(size_t i=0;i<b.size();++i) static_cast<vector_float4 *>(bb.contents)[i]=pack(b[i]);
   Parameters p{uint32_t(m),uint32_t(n),uint32_t(k),pack(alpha),pack(beta)};
   std::vector<double> wall[2],gpu[2]; double worst[2]={},between=0.; std::vector<cd> outputs[2];
   for(size_t repeat=0;repeat<11;++repeat) for(size_t order=0;order<2;++order) {
    const size_t api=(repeat+order)%2;
    for(size_t i=0;i<c.size();++i) static_cast<vector_float4 *>(cb.contents)[i]=pack(c[i]);
    double seconds=0.,wait=0.; auto start=std::chrono::steady_clock::now();
    if(api==1) {
     if(metal4_dispatch_zgemm(modern,false,ab,bb,cb,&p,sizeof(p),m,n,256,&seconds,&wait,error,sizeof(error))) {
      std::fprintf(stderr,"Metal 4 dispatch: %s\n",error); metal4_dispatch_destroy(modern); return 1;
     }
    } else {
     id<MTLCommandBuffer> command=[queue commandBuffer]; id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
     [encoder setComputePipelineState:legacy]; [encoder setBuffer:ab offset:0 atIndex:0];
     [encoder setBuffer:bb offset:0 atIndex:1]; [encoder setBuffer:cb offset:0 atIndex:2]; [encoder setBytes:&p length:sizeof(p) atIndex:3];
     [encoder dispatchThreads:MTLSizeMake(n,m,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
     [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
     if(command.status!=MTLCommandBufferStatusCompleted) { metal4_dispatch_destroy(modern); return 1; }
     seconds=command.GPUEndTime-command.GPUStartTime;
    }
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if(repeat>=2) { wall[api].push_back(elapsed); gpu[api].push_back(seconds); }
    outputs[api].resize(c.size());
    for(size_t i=0;i<c.size();++i) {
     auto value=unpack(static_cast<vector_float4 *>(cb.contents)[i]); outputs[api][i]=value;
     worst[api]=std::max(worst[api],std::abs(value-reference[i]));
     if(!std::isfinite(value.real()) || !std::isfinite(value.imag())) all_ok=false;
    }
   }
   for(size_t i=0;i<c.size();++i) between=std::max(between,std::abs(outputs[0][i]-outputs[1][i]));
   bool ok=worst[0]<1e-8 && worst[1]<1e-8 && between<1e-10 && median(gpu[1])>0.; all_ok=all_ok&&ok;
   std::printf("%s{\"shape\":[%zu,%zu,%zu],\"timed_repeats\":9,\"metal3_wall_s\":%.9g,\"metal4_wall_s\":%.9g,\"metal3_gpu_s\":%.9g,\"metal4_gpu_s\":%.9g,\"metal3_max_abs_error\":%.9g,\"metal4_max_abs_error\":%.9g,\"between_max_abs_error\":%.9g,\"passed\":%s}\n",shape?",":"",m,n,k,median(wall[0]),median(wall[1]),median(gpu[0]),median(gpu[1]),worst[0],worst[1],between,ok?"true":"false");
  }
  std::printf("],\"passed\":%s}\n",all_ok?"true":"false"); metal4_dispatch_destroy(modern);
  return all_ok ? 0 : 1;
 }
}
